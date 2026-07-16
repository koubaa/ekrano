// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy-based renderer for offscreen and texture rendering.
//!
//!
//! We use Goldy's bindless descriptor indexing (global arrays of up to 16K
//! descriptors per type) rather than actual buffer device addresses (BDA).
//! Push constants carry bindless indices per dispatch via Slang `uniform`
//! entry-point parameters.

pub const MAX_BINDLESS_SLOTS: usize = 16;

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use goldy::types::{BufferFlags, TextureFormat};
use goldy::{
    Buffer, BufferKind, ComputePipeline, Context, Device, Grant, RetainedPool, Texture, TexturePool, TimelineValue,
};

/// Ekrano uses a single-frame fire-and-forget model.
///
/// Stable pipeline parcels live in [`RetainedPool`] deeds reused across frames only while
/// depth stays at 1 (see [`StablePipelineBuffers`](crate::graph_gpu_resources::StablePipelineBuffers)).
pub(crate) const FRAME_PIPELINE_DEPTH: usize = 1;

use crate::{Error, RenderParams, Result, resource_proxy::BindType};
use ekrano_encoding::{BumpAllocators, Layout, RenderConfig, Resolver};

pub(crate) const MAX_BUMP_RETRIES: usize = 2;

/// Timeline-guarded slots for cross-frame render-target reuse.
/// Two slots suffice at depth=1: one aging entry plus one empty install slot.
pub(crate) const RESOURCE_CACHE_SLOTS: usize = 2;

pub(crate) fn find_empty_cache_slot<T>(slots: &[Option<T>]) -> Option<usize> {
    slots.iter().position(Option::is_none)
}

/// Per-frame render statistics returned by [`GoldyRenderer::render_to_texture`].
///
/// Non-zero `bump_retries` means the GPU bump allocator overflowed at least once
/// and the frame was re-rendered with larger buffers.  Callers that want to surface
/// this to the user (e.g. to detect scenes that are too complex for the default
/// buffer estimates) can print a warning when `bump_retries > 0`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameStats {
    /// Number of times the bump allocator overflowed and the frame was retried.
    /// Zero on a clean frame.
    pub bump_retries: u32,
}

/// Snapshot of frame-scheduling state, useful for tests and diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct AllocatorStats {
    /// Number of frames waiting in the [`goldy::FrameOrchestrator`] ring (always ≤ 1).
    pub cleanup_ring_depth: usize,
}

/// Snapshot of the resource pool's state, useful for tests and diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct ResourcePoolStats {
    /// Total number of pooled buffers across all keys.
    pub total_pooled_buffers: usize,
    /// Number of distinct `(size, access, name, flags)` keys in the pool.
    pub distinct_keys: usize,
    /// Committed bytes held in the retained pool (scheme-path only; 0 on Classic).
    ///
    /// Increases when `retain_pool.acquire_*` adds a new buffer, stays flat when
    /// existing allocations are reused. Useful for asserting that the
    /// `cached_scheme_indirect` composite buffer is reused rather than reallocated
    /// frame-to-frame.
    pub retained_pool_buffer_bytes: u64,
}

/// Upper bound applied to observed bump counters before they're fed into
/// `RenderConfig::with_bump_estimates`. Legitimate scenes need far less than
/// this (the tiger hits ~13K segments, paris-30k a few hundred thousand),
/// but a stale/corrupt read of `ekrano.bump_buf` — which we've observed
/// intermittently — can make a counter look like a billion, at which point
/// `grow()` rounds it up to `next_power_of_two` and the pool allocation goes
/// to multiple GB, exhausts Metal's heap, and puts the app in an infinite
/// retry loop. 16M entries for any single counter covers even paris-30k
/// plus an order of magnitude of headroom; anything larger is treated as
/// garbage and clamped.
const BUMP_SANITY_CAP: u32 = 16 * 1024 * 1024;

/// Clamp any counter that looks implausibly large back to the default.
/// `with_bump_estimates` ignores fields that are below the current buffer
/// size, so substituting 0 for an absurd value is equivalent to "stay at
/// default" without risking an allocation blow-up.
pub(crate) fn sanitize_bump(bump: &BumpAllocators) -> BumpAllocators {
    let clamp = |v: u32| if v > BUMP_SANITY_CAP { 0 } else { v };
    BumpAllocators {
        failed: bump.failed,
        binning: clamp(bump.binning),
        ptcl: clamp(bump.ptcl),
        tile: clamp(bump.tile),
        seg_counts: clamp(bump.seg_counts),
        segments: clamp(bump.segments),
        blend: clamp(bump.blend),
        lines: clamp(bump.lines),
    }
}

/// Rolling frame counter, used purely to rate-limit per-frame logging so a
/// long run doesn't flood the terminal and truncate the scrollback. Shared
/// across all `GoldyRenderer` instances in the process; that's fine because
/// in practice there's only one.
pub(crate) static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Log total GPU memory usage at most once every 5 seconds.
///
/// Prefers DXGI `CurrentUsage` (true process GPU residency) when available, and
/// always includes Goldy's live tracked allocator bytes (allocations − frees).
pub(crate) fn maybe_log_gpu_memory(device: &Device, backend: &'static str) {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static LAST_CLASSIC: Mutex<Option<Instant>> = Mutex::new(None);
    static LAST_SCHEME: Mutex<Option<Instant>> = Mutex::new(None);
    let last_slot = if backend == "classic" {
        &LAST_CLASSIC
    } else {
        &LAST_SCHEME
    };
    {
        let mut last = last_slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = *last
            && t.elapsed() < Duration::from_secs(5)
        {
            return;
        }
        *last = Some(Instant::now());
    }

    fn fmt_mib(bytes: u64) -> String {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    }

    let tracked = device.tracked_vram_bytes();
    match device.video_memory_info() {
        Some(info) => {
            log::info!(
                "[GPU-MEM] backend={backend} dxgi_local={} / budget={} dxgi_non_local={} tracked={}",
                fmt_mib(info.local_current_bytes),
                fmt_mib(info.local_budget_bytes),
                fmt_mib(info.non_local_current_bytes),
                fmt_mib(tracked),
            );
        }
        None => {
            log::info!(
                "[GPU-MEM] backend={backend} tracked={} (no DXGI video-memory query on this backend)",
                fmt_mib(tracked),
            );
        }
    }
}

// -----------------------------------------------------------------------
// Deferred per-frame work
// -----------------------------------------------------------------------
//
// The `FrameOrchestrator` ring now carries `()` — all resource retirement
// (render targets, pipeline buffers, owned buffers, bump readback) bypasses
// the ring via timeline-guarded cache slots + `Device::defer_release`
// + `PersistentState::pending_bump_readback`.  The ring is a pure scheduling
// primitive: depth enforcement + timeline tracking only.

/// Token pushed into a [`DeferredPayload`] when owned pipeline buffers are retired.
///
/// [`DeferredPayload`]: goldy::DeferredPayload
struct DeferredOwnedBuffersToken {
    pending: Arc<Mutex<Vec<(Buffer, &'static str)>>>,
    buffers: Vec<(Buffer, &'static str)>,
}

impl Drop for DeferredOwnedBuffersToken {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.append(&mut self.buffers);
        }
    }
}

/// Token pushed into a [`DeferredPayload`] when intermediate textures are retired.
///
/// When the `VramAllocator` ring drops this token (after `gpu_progress >= epoch`),
/// it enqueues the textures into `pending_texture_returns`. The next
/// `run_frame` call drains that queue and returns them to `TexturePool`.
///
/// [`DeferredPayload`]: goldy::DeferredPayload
struct DeferredTextureToken {
    pending: Arc<Mutex<Vec<Texture>>>,
    textures: Vec<Texture>,
}

impl Drop for DeferredTextureToken {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.pending.lock() {
            pending.append(&mut self.textures);
        }
    }
}

/// Which caches received new entries during
/// [`crate::graph_renderer::GraphRecorder::schedule_pipeline_cleanup`].
#[derive(Debug, Default)]
pub(crate) struct CacheScheduleOutcome {
    pub(crate) cached_render_targets_slot: Option<usize>,
    /// Set when the scheme path stored its single persistent `out_image` into
    /// [`PersistentState::cached_scheme_rt`]; the post-submit step then stamps the
    /// frame timeline as a cache/reclamation marker (not a `begin_frame` GPU wait).
    pub(crate) scheme_rt_stored: bool,
}

/// Outcome of [`crate::graph_renderer::GraphRecorder::finish`]: orchestrator submit result plus resources
/// that bypass the ring and are deferred via [`Device::defer_release`].
pub(crate) struct FrameFinishOutcome {
    pub(crate) timeline: TimelineValue,
    pub(crate) surface_frame: Option<goldy::Frame>,
    pub(crate) bump_readback: Option<Buffer>,
    pub(crate) deferred_textures: Vec<Texture>,
    pub(crate) recyclable_owned: Vec<(Buffer, &'static str)>,
    /// Scheme-path submission returned by [`goldy::Scheme::submit`].
    /// `None` on the Classic (`TaskGraph`) path.
    ///
    /// Held here so that `SchemeRenderer::run_frame_from_prepared` can build a
    /// [`PresentToken`] after `finish()` returns.
    pub(crate) scheme_submission: Option<goldy::Submission>,
}

/// Scheme-path present token — analogue of [`goldy::Frame`] on the Classic path.
///
/// Produced by [`GoldyRenderer::submit_to_swapchain`]. Hand to `TID_PRESENT` for async
/// scanout, or call [`Self::present`] synchronously (e.g. [`crate::scheme_renderer::SchemeRenderer::render_to_swapchain`]).
pub struct PresentToken {
    pub(crate) grant: goldy::PresentGrant,
    pub(crate) submission: goldy::Submission,
}

impl PresentToken {
    /// Perform scanout via [`goldy::PresentGrant::consume`].
    pub fn present(self) -> Result<()> {
        self.grant
            .consume(&self.submission)
            .map_err(|e| Error::Shader(e.to_string()))
    }
}

/// CPU-resolved scene data ready for GPU submission.
///
/// Produced by [`GoldyRenderer::prepare`] (pure CPU — safe to call while the
/// previous frame is still on the GPU) and consumed by
/// [`GoldyRenderer::submit_to_surface`] or [`GoldyRenderer::submit_prepared`].
/// Owns all data extracted from the scene encoding so it can be stored across
/// event loop iterations.
pub struct PreparedFrame {
    pub(crate) packed: Vec<u8>,
    pub(crate) layout: Layout,
    pub(crate) ramps_data: Vec<u32>,
    pub(crate) ramps_width: u32,
    pub(crate) ramps_height: u32,
    pub(crate) images_width: u32,
    pub(crate) images_height: u32,
    pub(crate) image_entries: Vec<(peniko::ImageData, u32, u32)>,
    pub(crate) config: RenderConfig,
    pub(crate) params: RenderParams,
    pub(crate) resolver: Resolver,
    /// Owned copy of `Encoding::coverage_mask` — used by `PipelineResources::prepare`.
    pub(crate) coverage_mask: Option<ekrano_encoding::CoverageMask>,
    /// Owned copy of `Encoding::layer_filter_effects` — used by `record_fine` and
    /// `record_filter_effects`.
    pub(crate) layer_filter_effects: Vec<ekrano_encoding::LayerFilterEffect>,
}

impl PreparedFrame {
    /// Render width this frame was prepared for.
    pub fn width(&self) -> u32 {
        self.params.width
    }

    /// Render height this frame was prepared for.
    pub fn height(&self) -> u32 {
        self.params.height
    }
}

/// Defer textures and recyclable owned buffers until `tv` retires on the GPU.
///
/// Uses a single [`Context::defer_release`] per frame (one mutex push) instead of
/// multiple deferred cleanup calls.
pub(crate) fn defer_frame_gpu_resources(
    ctx: &Context,
    persistent: &PersistentState,
    tv: TimelineValue,
    textures: Vec<Texture>,
    recyclable_owned: Vec<(Buffer, &'static str)>,
) {
    let mut payload = goldy::DeferredPayload::new();
    if !textures.is_empty() {
        payload.push(DeferredTextureToken {
            pending: Arc::clone(&persistent.pending_texture_returns),
            textures,
        });
    }
    if !recyclable_owned.is_empty() {
        payload.push(DeferredOwnedBuffersToken {
            pending: Arc::clone(&persistent.pending_owned_returns),
            buffers: recyclable_owned,
        });
    }
    if !payload.is_empty() {
        ctx.defer_release(tv, payload);
    }
}

pub(crate) struct GoldyShader {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bindings: Vec<BindType>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct BufferKey {
    size: u64,
    access: BufferKind,
    name: &'static str,
    buffer_flags: BufferFlags,
}

fn pool_key_for_return(buf: &Buffer, name: &'static str) -> BufferKey {
    BufferKey {
        size: buf.byte_size(),
        access: BufferKind::Scattered,
        name,
        buffer_flags: buf.flags(),
    }
}

type PendingOwnedReturns = Arc<Mutex<Vec<(Buffer, &'static str)>>>;

#[derive(Default)]
pub(crate) struct ResourcePool {
    bufs: HashMap<BufferKey, Vec<Buffer>>,
    /// Reference to the pending returns queue. When a buffer allocation fails
    /// (heap exhausted), the pool flushes deferred deletions, drains this queue
    /// back into itself, and retries before blocking for GPU progress.
    pending_owned_returns: Option<PendingOwnedReturns>,
}

impl ResourcePool {
    /// Wire the pool to the renderer's `pending_owned_returns` queue so that
    /// pool-misses can self-replenish before triggering a GPU wait.
    pub(crate) fn set_pending_returns(&mut self, pending: PendingOwnedReturns) {
        self.pending_owned_returns = Some(pending);
    }

    /// Drain `pending_owned_returns` into this pool.
    fn drain_pending(&mut self) {
        let Some(ref pending) = self.pending_owned_returns else {
            return;
        };
        let Ok(mut guard) = pending.try_lock() else {
            return;
        };
        let drained: Vec<_> = guard.drain(..).collect();
        drop(guard);
        for (buf, name) in drained {
            self.return_buf(buf, name);
        }
    }

    pub(crate) fn get_buf_with_stride(
        &mut self,
        retained_pool: &mut RetainedPool,
        ctx: &Context,
        size: u64,
        name: &'static str,
        access: BufferKind,
        stride: Option<u32>,
        buffer_flags: BufferFlags,
    ) -> Result<Buffer> {
        let key = BufferKey {
            size,
            access,
            name,
            buffer_flags,
        };
        // Fast path: pool hit.
        if let Some(buf) = self.bufs.entry(key.clone()).or_default().pop() {
            return Ok(buf);
        }

        // Pool miss: try a fresh allocation via the retained pool door.
        match retained_pool.acquire_buffer(size, access, stride, buffer_flags, None) {
            Ok(buf) => Ok(buf),
            Err(_) => {
                // Attempt 2 (non-blocking): flush deferred deletions so that any GPU
                // work that completed since the last flush moves buffers from the vram
                // allocator ring into pending_owned_returns, then drain pending back into
                // this pool and retry. This handles the common case where one or more
                // frames finished between the last run_frame flush and this alloc.
                ctx.flush_deferred_deletions();
                self.drain_pending();
                if let Some(buf) = self.bufs.entry(key.clone()).or_default().pop() {
                    return Ok(buf);
                }

                // Attempt 3 (blocking): the `Oversubscribed` signal was queued by the
                // allocation failure above; wait for the oldest outstanding GPU work to
                // retire, which frees one frame's worth of heap. A single wait suffices —
                // the previous loop logic iterated through deferred epochs one-by-one, but
                // peek_oldest_in_flight() gives us the exact fence value we need to unblock.
                {
                    let _tz = goldy::tracy_zone!("ekrano.resource_pool.wait_reclaim");
                    if let Some(oldest_in_flight) = ctx.peek_oldest_in_flight() {
                        log::debug!(
                            "ResourcePool heap pressure for {name} — waiting for GPU epoch \
                             {oldest_in_flight} to reclaim archive",
                        );
                        let _ = ctx.wait_until_timeout(oldest_in_flight, 2000);
                        ctx.flush_deferred_deletions();
                        self.drain_pending();
                        if let Some(buf) = self.bufs.entry(key.clone()).or_default().pop() {
                            return Ok(buf);
                        }
                    }
                }

                // Final attempt: one more fresh allocation (heap may have space now that
                // the DeletionQueue was processed inside flush_deferred_deletions).
                retained_pool
                    .acquire_buffer(size, access, stride, buffer_flags, None)
                    .map_err(|e| Error::Shader(e.to_string()))
            }
        }
    }

    /// Return a buffer to the pool for reuse by a future frame.
    pub(crate) fn return_buf(&mut self, buf: Buffer, name: &'static str) {
        let key = pool_key_for_return(&buf, name);
        self.bufs.entry(key).or_default().push(buf);
    }

    /// Drop excess pooled [`Buffer`]s per `(size, access, name)` key to bound memory.
    pub(crate) fn cap_pool_depth(&mut self, max_per_key: usize) {
        let mut empties = Vec::<BufferKey>::new();
        for (key, stack) in self.bufs.iter_mut() {
            while stack.len() > max_per_key {
                drop(stack.pop());
            }
            if stack.is_empty() {
                empties.push(key.clone());
            }
        }
        for k in empties {
            self.bufs.remove(&k);
        }
    }

    /// Total number of pooled buffers across all keys.
    pub(crate) fn total_pooled_buffers(&self) -> usize {
        self.bufs.values().map(|v| v.len()).sum()
    }

    /// Number of distinct buffer keys in the pool.
    pub(crate) fn distinct_keys(&self) -> usize {
        self.bufs.len()
    }
}

/// Override [`RenderParams::robust`] from the environment for benchmarking.
///
/// `EKRANO_ROBUST=0|false|no|off` disables bump readback (same as `robust: false`).
/// `EKRANO_ROBUST=1|true|yes|on` forces it on. Unset → use the caller's `RenderParams`.
pub(crate) fn env_robust_override() -> Option<bool> {
    std::env::var("EKRANO_ROBUST")
        .ok()
        .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"))
}

// -----------------------------------------------------------------------
// PersistentState — GPU resources that survive across frames
// -----------------------------------------------------------------------

/// GPU resources that live for the lifetime of the renderer and are reused
/// across frames. Pool growth, texture reuse, and bump estimates all live here.
pub(crate) struct PersistentState {
    /// Owned buffer cache: recycles pool-exempt buffers (bump, indirect, etc.)
    pub(crate) pool: ResourcePool,
    /// Retained pool for the seven stable pipeline buffers (`resource-pool.md` §4).
    /// Valid only at [`FRAME_PIPELINE_DEPTH`] = 1; see [`StablePipelineBuffers`](crate::graph_gpu_resources::StablePipelineBuffers).
    pub(crate) retained_pool: RetainedPool,
    /// Texture pool for intermediate render targets (gradient, filter layers, etc.)
    pub(crate) tex_pool: TexturePool,
    /// Bump allocator counters from the most recently drained frame.
    /// `None` until the first GPU readback completes.
    last_drained_bump: Option<BumpAllocators>,
    /// Bump readback buffer whose GPU timeline is not yet complete.
    ///
    /// At depth=1 at most one readback is outstanding between frames.
    pending_bump_readback: Option<(TimelineValue, Buffer)>,
    /// Persistent linear-filter + clamp-to-edge sampler for hardware-filtered texture reads
    /// (gradient ramps, image atlas bilinear). Lazily created on first render.
    pub(crate) linear_clamp_sampler: Option<goldy::Sampler>,
    /// Persistent nearest-filter + clamp-to-edge sampler for `IMAGE_QUALITY_LOW` reads.
    pub(crate) nearest_clamp_sampler: Option<goldy::Sampler>,
    /// Cached render targets (`out_image` + `filter_layers`) from previous frames.
    ///
    /// Timeline-guarded slots for cross-frame render-target reuse. Written by
    /// `schedule_pipeline_cleanup`; readable when `context.gpu_progress() >= cached_rt_timelines[i]`.
    pub(crate) cached_render_targets: [Option<(Texture, [Texture; 4])>; RESOURCE_CACHE_SLOTS],
    /// Timeline of the frame that last wrote each render-target slot. `0` when empty.
    pub(crate) cached_rt_timelines: [TimelineValue; RESOURCE_CACHE_SLOTS],
    /// Scheme-path render-target reuse: a single persistent `out_image` + filter
    /// layers, intentionally decoupled from [`RESOURCE_CACHE_SLOTS`].
    pub(crate) cached_scheme_rt: Option<(Texture, [Texture; 4], TimelineValue)>,
    /// Cached pipeline buffers from the previous frame. At depth=1 only one
    /// entry exists at a time: take-then-install within a single `run_frame`.
    pub(crate) cached_pipeline: Option<crate::graph_gpu_resources::CachedPipeline>,
    /// Maps RT cache slot index → swapchain image index of the last frame that
    /// used that slot, so [`Self::mark_rt_slot_returned`] can mark it reusable.
    pub(crate) rt_slot_swapchain_image: [Option<u32>; RESOURCE_CACHE_SLOTS],
    /// Capacity hints for `GraphRecorder` scratch allocations. Updated after each
    /// `finish()` call so that the next frame pre-allocates the right amount and
    /// avoids re-allocations on the hot path.
    pub(crate) deferred_owned_cap_hint: usize,
    pub(crate) deferred_textures_cap_hint: usize,
    /// Static MSAA8 mask LUT buffer (uploaded once, reused without re-upload).
    ///
    /// Keeping this persistent avoids a staging-belt `CopyBufferRegion` in the
    /// retained fine command list, which would reference a stale staging chunk on
    /// resubmission.  `None` until first needed; never freed after that.
    pub(crate) stable_mask_lut_msaa8: Option<Buffer>,
    /// Static MSAA16 mask LUT buffer (same rationale as `stable_mask_lut_msaa8`).
    pub(crate) stable_mask_lut_msaa16: Option<Buffer>,
    /// Cached GPU workgroup counts buffer. Avoids a `WriteBuffer` upload when the
    /// workgroup counts are identical to the previous frame (scene topology unchanged).
    /// The buffer lives here permanently — not returned to the pool between frames.
    pub(crate) cached_wg_counts: Option<(ekrano_encoding::WorkgroupCountsGpu, Buffer)>,
    /// Cached GPU `ConfigUniform` buffer. Stable across frames once bump estimates
    /// converge; eliminates `WriteBuffer` from the dispatch graph at steady state.
    pub(crate) cached_config_uniform: Option<(ekrano_encoding::ConfigUniform, Buffer)>,
    /// Per-slot cached `FilterUniform` buffers, indexed by filter dispatch order.
    /// Stable for scenes with fixed filter effects (e.g. a static drop shadow).
    pub(crate) cached_filter_uniforms: Vec<Option<(ekrano_encoding::FilterUniform, Buffer)>>,
    /// Composite per-stage indirect `DispatchShape` buffer, retained across frames.
    /// Cache key is the `WorkgroupCountsGpu` that seeded the allocation.
    /// Scheme-path only; never touches `ResourcePool`.
    pub(crate) cached_scheme_indirect: Option<(ekrano_encoding::WorkgroupCountsGpu, Buffer)>,
    /// Stable scene buffer for the retained worker scheme (bucket capacity, buffer).
    pub(crate) cached_scene: Option<(u64, Buffer)>,
    /// Logical upload buffer for scene staging (bucket capacity, declaration).
    pub(crate) cached_scene_upload: Option<(u64, goldy::UploadBuffer)>,
    /// Logical upload buffer for config uniform staging.
    pub(crate) cached_config_upload: Option<goldy::UploadBuffer>,
    /// Stable bump buffer keyed by byte size; read back via [`Self::cached_bump_grant`].
    pub(crate) cached_bump: Option<(u64, Buffer)>,
    /// Recorded once on the worker when `robust` is enabled.
    pub(crate) cached_bump_grant: Option<goldy::ReadGrant<goldy::GrantBuffer>>,
    /// Stable gradient atlas (width, height, texture).
    pub(crate) cached_gradient: Option<(u32, u32, Texture)>,
    /// Stable image atlas (width, height, texture).
    pub(crate) cached_image_atlas: Option<(u32, u32, Texture)>,
    /// Stable mask atlas (width, height, texture).
    pub(crate) cached_mask_atlas: Option<(u32, u32, Texture)>,
    /// Present grant recorded once on the worker when swapchain presentation is enabled.
    pub(crate) cached_present_grant: Option<goldy::PresentGrant>,
    /// `out_image` handle the worker was recorded against (RT cache rotation invalidates retention).
    pub(crate) cached_worker_out_image: Option<goldy::types::ResourceHandle>,
    /// Output texture handle the worker fine pass was recorded against.
    pub(crate) cached_worker_output_texture: Option<goldy::backend::TextureHandle>,
    /// Full worker-bound handles from the last record (debug invariant checks only).
    #[cfg(debug_assertions)]
    pub(crate) cached_worker_resources: Option<crate::worker_retention::WorkerResourceHandles>,
    /// Last worker topology the retained scheme was recorded against.
    pub(crate) cached_worker_topology: Option<crate::worker_retention::WorkerTopology>,
    /// Filter effects from the last worker record (topology comparison).
    pub(crate) cached_worker_filter_effects: Vec<ekrano_encoding::LayerFilterEffect>,
    /// Worker submission from the prior frame, consumed via [`Self::cached_bump_grant`] at drain.
    pub(crate) pending_bump_submission: Option<goldy::Submission>,
    /// Upload key the upload scheme was recorded against (scene bucket + all atlas dims).
    pub(crate) cached_upload_key: Option<crate::worker_retention::UploadKey>,
    /// Logical upload buffer for gradient atlas staging (width, height, capacity, declaration).
    pub(crate) cached_gradient_upload: Option<(u32, u32, u64, goldy::UploadBuffer)>,
    /// Logical upload buffer for mask atlas staging (width, height, capacity, declaration).
    pub(crate) cached_mask_upload: Option<(u32, u32, u64, goldy::UploadBuffer)>,
    /// Logical upload buffers for image atlas region uploads.
    pub(crate) cached_image_region_uploads: Vec<((u32, u32, u32, u32), goldy::UploadBuffer)>,
    /// Textures waiting to be returned to [`Self::tex_pool`] after GPU retirement.
    /// Populated by [`DeferredTextureToken`] drops from [`Context::defer_release`].
    pub(crate) pending_texture_returns: Arc<Mutex<Vec<Texture>>>,
    /// Owned buffers waiting to be returned to [`Self::pool`] after GPU retirement.
    /// Populated by [`DeferredOwnedBuffersToken`] drops from [`Context::defer_release`].
    pub(crate) pending_owned_returns: Arc<Mutex<Vec<(Buffer, &'static str)>>>,
    /// Persistent host buffer parcel for [`crate::scheme_renderer::SchemeRenderer::render_to_buffer`].
    pub(crate) readback_host_buf: Option<(Buffer, u64)>,
}

impl PersistentState {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            pool: ResourcePool::default(),
            retained_pool: RetainedPool::new(Arc::new(device.clone())),
            tex_pool: TexturePool::default(),
            last_drained_bump: None,
            pending_bump_readback: None,
            linear_clamp_sampler: None,
            nearest_clamp_sampler: None,
            cached_render_targets: std::array::from_fn(|_| None),
            cached_rt_timelines: [0; RESOURCE_CACHE_SLOTS],
            cached_scheme_rt: None,
            cached_pipeline: None,
            rt_slot_swapchain_image: [None; RESOURCE_CACHE_SLOTS],
            deferred_owned_cap_hint: 0,
            deferred_textures_cap_hint: 0,
            stable_mask_lut_msaa8: None,
            stable_mask_lut_msaa16: None,
            cached_wg_counts: None,
            cached_config_uniform: None,
            cached_filter_uniforms: Vec::new(),
            cached_scheme_indirect: None,
            cached_scene: None,
            cached_scene_upload: None,
            cached_config_upload: None,
            cached_bump: None,
            cached_bump_grant: None,
            cached_gradient: None,
            cached_image_atlas: None,
            cached_mask_atlas: None,
            cached_present_grant: None,
            cached_worker_out_image: None,
            cached_worker_output_texture: None,
            #[cfg(debug_assertions)]
            cached_worker_resources: None,
            cached_worker_topology: None,
            cached_worker_filter_effects: Vec::new(),
            pending_bump_submission: None,
            cached_upload_key: None,
            cached_gradient_upload: None,
            cached_mask_upload: None,
            cached_image_region_uploads: Vec::new(),
            pending_texture_returns: Arc::new(Mutex::new(Vec::new())),
            pending_owned_returns: Arc::new(Mutex::new(Vec::new())),
            readback_host_buf: None,
        }
    }

    pub(crate) fn acquire_readback_host_buf(&mut self, ctx: &Context, staging_bytes: u64) -> Result<Buffer, Error> {
        let needs_new = self
            .readback_host_buf
            .as_ref()
            .map(|(_, size)| *size != staging_bytes)
            .unwrap_or(true);
        if needs_new {
            if let Some((old, _)) = self.readback_host_buf.take() {
                self.pool.return_buf(old, "ekrano.readback_host_buf");
            }
            let buf = self
                .pool
                .get_buf_with_stride(
                    &mut self.retained_pool,
                    ctx,
                    staging_bytes,
                    "ekrano.readback_host_buf",
                    BufferKind::Scattered,
                    None,
                    BufferFlags::CPU_READABLE,
                )
                .map_err(|e| Error::Shader(e.to_string()))?;
            Ok(buf)
        } else if let Some((buf, _)) = self.readback_host_buf.take() {
            Ok(buf)
        } else {
            self.pool
                .get_buf_with_stride(
                    &mut self.retained_pool,
                    ctx,
                    staging_bytes,
                    "ekrano.readback_host_buf",
                    BufferKind::Scattered,
                    None,
                    BufferFlags::CPU_READABLE,
                )
                .map_err(|e| Error::Shader(e.to_string()))
        }
    }

    pub(crate) fn store_readback_host_buf(&mut self, buf: Buffer, staging_bytes: u64) {
        self.readback_host_buf = Some((buf, staging_bytes));
    }

    pub(crate) fn take_scheme_render_targets(
        &mut self,
        ctx: &Context,
        width: u32,
        height: u32,
        out_format: TextureFormat,
    ) -> Option<(Texture, [Texture; 4])> {
        let (out, layers, tv) = self.cached_scheme_rt.take()?;
        if out.width() == width && out.height() == height && out.format() == out_format {
            // `tv` is only a cache/reclamation stamp. Ordering for reuse is either
            // submit-side (DX12/Vulkan nonblocking) or the orchestrator ring (Metal).
            let _ = tv;
            return Some((out, layers));
        }
        log::warn!(
            "[RT-CACHE] scheme MISS (resize) timeline={tv} {}x{} vs {width}x{height} fmt={out_format:?}",
            out.width(),
            out.height(),
        );
        self.reclaim_scheme_render_targets(ctx, out, layers, tv);
        None
    }

    /// Release scheme render targets to the texture pool, deferring until `tv` retires.
    fn reclaim_scheme_render_targets(&mut self, ctx: &Context, out: Texture, layers: [Texture; 4], tv: TimelineValue) {
        if tv != 0 && ctx.gpu_progress() < tv {
            let mut textures = Vec::with_capacity(5);
            textures.push(out);
            textures.extend(layers);
            let mut payload = goldy::DeferredPayload::new();
            payload.push(DeferredTextureToken {
                pending: Arc::clone(&self.pending_texture_returns),
                textures,
            });
            ctx.defer_release(tv, payload);
            return;
        }
        self.tex_pool.release(out);
        for l in layers {
            self.tex_pool.release(l);
        }
    }

    pub(crate) fn store_scheme_render_targets(&mut self, out: Texture, layers: [Texture; 4], timeline: TimelineValue) {
        self.cached_scheme_rt = Some((out, layers, timeline));
    }
}

#[cfg(test)]
impl PersistentState {
    /// Minimal stub for unit tests that only inspect plain fields (no GPU resources).
    pub(crate) fn new_test_only() -> Self {
        let mock_device = goldy::test_support::mock_device();
        Self::new(&mock_device)
    }
}

impl PersistentState {
    /// Drop all cached render targets when any occupied slot no longer matches the
    /// requested dimensions. Waits for the oldest slot timeline so heap-backed textures
    /// are retired before new allocations during resize.
    pub(crate) fn purge_render_target_cache_if_mismatch(
        &mut self,
        ctx: &Context,
        width: u32,
        height: u32,
        out_format: TextureFormat,
    ) -> bool {
        let mismatch = self.cached_render_targets.iter().any(|slot| {
            slot.as_ref()
                .is_some_and(|(out, _)| out.width() != width || out.height() != height || out.format() != out_format)
        });
        if !mismatch {
            return false;
        }

        let progress = ctx.gpu_progress();
        if ctx.peek_oldest_in_flight().is_none() {
            // GPU idle — skip the timeline scan entirely.
        } else {
            let oldest = (0..RESOURCE_CACHE_SLOTS)
                .filter(|&i| self.cached_render_targets[i].is_some())
                .map(|i| self.cached_rt_timelines[i])
                .min();
            if let Some(oldest) = oldest
                && progress < oldest
            {
                let _ = ctx.wait_until(oldest);
            }
        }

        for i in 0..RESOURCE_CACHE_SLOTS {
            if let Some((out, layers)) = self.cached_render_targets[i].take() {
                // Drop directly (not via tex_pool.release) so that the Metal heap
                // allocation is reclaimed immediately. tex_pool.release would keep
                // the mismatched-size texture in the pool, keeping the heap full and
                // causing subsequent allocations at the new dimensions to fail.
                drop(out);
                for l in layers {
                    drop(l);
                }
                self.cached_rt_timelines[i] = 0;
            }
        }
        true
    }

    pub(crate) fn take_cached_render_targets(
        &mut self,
        progress: TimelineValue,
        width: u32,
        height: u32,
        out_format: TextureFormat,
    ) -> Option<(Texture, [Texture; 4])> {
        for i in 0..RESOURCE_CACHE_SLOTS {
            if progress < self.cached_rt_timelines[i] {
                continue;
            }
            let Some((out, layers)) = self.cached_render_targets[i].take() else {
                continue;
            };
            if out.width() == width && out.height() == height && out.format() == out_format {
                return Some((out, layers));
            }
            log::warn!(
                "[RT-CACHE] MISS (resize) slot={i}: progress={progress} timeline={} {}x{} vs {}x{} fmt={out_format:?}",
                self.cached_rt_timelines[i],
                out.width(),
                out.height(),
                width,
                height,
            );
            self.tex_pool.release(out);
            for l in layers {
                self.tex_pool.release(l);
            }
        }
        None
    }
}

impl PersistentState {
    pub(crate) fn last_drained_bump(&self) -> Option<&BumpAllocators> {
        self.last_drained_bump.as_ref()
    }

    pub(crate) fn take_last_drained_bump(&mut self) -> Option<BumpAllocators> {
        self.last_drained_bump.take()
    }

    pub(crate) fn queue_bump_readback(&mut self, timeline: TimelineValue, buf: Buffer) {
        self.pending_bump_readback = Some((timeline, buf));
    }

    pub(crate) fn queue_bump_submission(&mut self, submission: goldy::Submission) {
        self.pending_bump_submission = Some(submission);
    }

    pub(crate) fn drain_ready_bump_readbacks(&mut self, device: &Device, ctx: &Context) -> Result<()> {
        if let Some(submission) = self.pending_bump_submission.take() {
            if let Some(grant) = self.cached_bump_grant.as_ref() {
                let tv = submission.timeline_value();
                if tv > ctx.gpu_progress() {
                    self.pending_bump_submission = Some(submission);
                    return Ok(());
                }
                let _tz = goldy::tracy_zone!("ekrano.drain_ready_bump_readbacks.grant");
                let loan = grant.consume(&submission).map_err(|e| Error::Shader(e.to_string()))?;
                read_bump_bytes(self, &loan);
                return Ok(());
            }
            self.pending_bump_submission = Some(submission);
        }

        let Some((timeline, buf)) = self.pending_bump_readback.take() else {
            return Ok(());
        };
        if timeline > ctx.gpu_progress() {
            self.pending_bump_readback = Some((timeline, buf));
            return Ok(());
        }
        let _tz = goldy::tracy_zone!("ekrano.drain_ready_bump_readbacks");
        read_bump_buffer(device, self, buf)
    }

    /// Claim cached pipeline buffers for this frame.
    ///
    /// Pipeline buffers are fully GPU-overwritten each frame. At depth=1, [`goldy::FrameOrchestrator`]
    /// retirement in `begin_frame` provides cross-frame ordering — no `gpu_progress` gate here.
    pub(crate) fn take_cached_pipeline(&mut self) -> Option<crate::graph_gpu_resources::CachedPipeline> {
        if let Some(c) = self.cached_pipeline.take() {
            log::debug!("[PIPE-CACHE] HIT");
            return Some(c);
        }
        None
    }

    /// Return textures and owned buffers whose GPU retirement completed since the last frame.
    pub(crate) fn drain_pending_returns(&mut self) {
        if let Ok(mut pending) = self.pending_texture_returns.lock() {
            for tex in pending.drain(..) {
                self.tex_pool.release(tex);
            }
        }
        if let Ok(mut pending) = self.pending_owned_returns.lock() {
            for (buf, name) in pending.drain(..) {
                self.pool.return_buf(buf, name);
            }
        }
    }

    /// Mark an RT cache slot immediately reusable when its swapchain image returns.
    pub(crate) fn mark_rt_slot_returned(&mut self, ctx: &Context, image_index: u32) {
        let progress = ctx.gpu_progress();
        for (i, entry) in self.rt_slot_swapchain_image.iter_mut().enumerate() {
            if *entry == Some(image_index) {
                debug_assert!(
                    progress >= self.cached_rt_timelines[i],
                    "SwapchainReturned for image {image_index} before RT slot {i} timeline completed",
                );
                self.cached_rt_timelines[i] = 0;
                *entry = None;
            }
        }
    }
}

fn read_bump_bytes(persistent: &mut PersistentState, bytes: &[u8]) {
    let _parse = goldy::tracy_zone!("ekrano.bump_readback.parse");
    persistent.last_drained_bump = Some(bytemuck::pod_read_unaligned(bytes));
}

fn read_bump_buffer(device: &Device, persistent: &mut PersistentState, buf: Buffer) -> Result<()> {
    let _bump = goldy::tracy_zone!("ekrano.bump_readback");
    let size = buf.byte_size() as usize;
    let mut output = {
        let _alloc = goldy::tracy_zone!("ekrano.bump_readback.alloc");
        vec![0_u8; size]
    };
    {
        let _read = goldy::tracy_zone!("ekrano.bump_readback.read_to_cpu");
        buf.read_to_cpu(device, &mut output)
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
    {
        read_bump_bytes(persistent, &output);
    }
    {
        let _return = goldy::tracy_zone!("ekrano.bump_readback.return_pool");
        persistent.pool.return_buf(buf, "ekrano.bump_buf");
    }
    Ok(())
}

// -----------------------------------------------------------------------
// GoldyBackend — runtime backend selector
// -----------------------------------------------------------------------

/// Selects which frame-loop backend [`GoldyRenderer`] uses at runtime.
///
/// Both variants share the same public API.  The `Classic` path is the
/// existing `TaskGraph`-based loop; `Scheme` will use the retained-scheme
/// loop once it is implemented (Phase 2 of the retained-scheme migration).
///
/// The flag is **runtime** (not a Cargo feature) so both paths can be kept
/// alive simultaneously during the transition, enabling side-by-side
/// correctness checks and FPS comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GoldyBackend {
    /// `TaskGraph`-based frame loop (current production path).
    #[default]
    Classic,
    /// Retained-`Scheme`-based frame loop (Phase 2 work-in-progress).
    Scheme,
}

// -----------------------------------------------------------------------
// GoldyRenderer — thin dispatch wrapper
// -----------------------------------------------------------------------

/// Goldy-based 2D renderer.
///
/// Dispatches to either [`crate::graph_renderer::GraphRenderer`] (Classic) or
/// [`crate::scheme_renderer::SchemeRenderer`] (Scheme) depending on the backend
/// selected at construction time.
///
/// Callers that need backend-specific surface APIs should use the concrete renderer
/// types directly. This type is provided for source compatibility with code that
/// does not care about surface rendering or uses [`Self::render_to_buffer`] /
/// [`Self::render_to_texture`].
pub enum GoldyRenderer {
    /// Classic [`TaskGraph`](goldy::TaskGraph)-based renderer.
    Classic(Box<crate::graph_renderer::GraphRenderer>),
    /// Retained-[`Scheme`](goldy::Scheme)-based renderer.
    Scheme(Box<crate::scheme_renderer::SchemeRenderer>),
}

impl GoldyRenderer {
    /// Create a renderer using the backend selected by `EKRANO_BACKEND` (default: Classic).
    pub fn new(device: &Device) -> Result<Self> {
        let backend = match std::env::var("EKRANO_BACKEND").as_deref() {
            Ok("scheme") => GoldyBackend::Scheme,
            _ => GoldyBackend::Classic,
        };
        Self::new_with_backend(device, backend)
    }

    /// Create a renderer with an explicit backend selector.
    pub fn new_with_backend(device: &Device, backend: GoldyBackend) -> Result<Self> {
        match backend {
            GoldyBackend::Classic => Ok(Self::Classic(Box::new(crate::graph_renderer::GraphRenderer::new(
                device,
            )?))),
            GoldyBackend::Scheme => Ok(Self::Scheme(Box::new(crate::scheme_renderer::SchemeRenderer::new(
                device,
            )?))),
        }
    }

    /// Returns the active backend.
    pub fn backend(&self) -> GoldyBackend {
        match self {
            Self::Classic(_) => GoldyBackend::Classic,
            Self::Scheme(_) => GoldyBackend::Scheme,
        }
    }

    /// Returns a clone of the renderer's submission context.
    pub fn submission_context(&self) -> Context {
        match self {
            Self::Classic(r) => r.submission_context(),
            Self::Scheme(r) => r.submission_context(),
        }
    }

    /// GPU device handle shared by this renderer.
    pub fn device(&self) -> &Device {
        match self {
            Self::Classic(r) => r.device(),
            Self::Scheme(r) => r.device(),
        }
    }

    /// Drain signals and reclaim GPU resources tied to completed frames.
    pub fn poll_and_reclaim(&mut self) {
        match self {
            Self::Classic(r) => r.poll_and_reclaim(),
            Self::Scheme(r) => r.poll_and_reclaim(),
        }
    }

    /// Renders a scene to a texture.
    pub fn render_to_texture(
        &mut self,
        scene: &crate::Scene,
        texture: &Texture,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        match self {
            Self::Classic(r) => r.render_to_texture(scene, texture, params),
            Self::Scheme(r) => r.render_to_texture(scene, texture, params),
        }
    }

    /// Phase 1: resolve scene encoding to CPU buffers.
    pub fn prepare(&mut self, scene: &crate::Scene, params: &RenderParams) -> Result<PreparedFrame> {
        match self {
            Self::Classic(r) => r.prepare(scene, params),
            Self::Scheme(r) => r.prepare(scene, params),
        }
    }

    /// Render a scene directly to a swapchain [`Surface`](goldy::Surface) (Classic path).
    ///
    /// Panics if called on the Scheme backend — use [`Self::render_to_swapchain`] instead.
    pub fn render_to_surface(
        &mut self,
        scene: &crate::Scene,
        surface: &goldy::Surface,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        match self {
            Self::Classic(r) => r.render_to_surface(scene, surface, params),
            Self::Scheme(_) => panic!(
                "GoldyRenderer::render_to_surface called on Scheme backend — \
                 use render_to_swapchain(&mut SchemeRenderer, pool, params) instead"
            ),
        }
    }

    /// Render a scene to a [`SwapchainPool`](goldy::SwapchainPool) (Scheme path).
    ///
    /// Panics if called on the Classic backend — use [`Self::render_to_surface`] instead.
    pub fn render_to_swapchain(
        &mut self,
        scene: &crate::Scene,
        pool: &goldy::SwapchainPool,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        match self {
            Self::Classic(_) => panic!(
                "GoldyRenderer::render_to_swapchain called on Classic backend — \
                 use render_to_surface(&mut GraphRenderer, surface, params) instead"
            ),
            Self::Scheme(r) => r.render_to_swapchain(scene, pool, params),
        }
    }

    /// Phase 2: record GPU work, present, and return frame stats (Classic path).
    ///
    /// Panics if called on the Scheme backend.
    pub fn submit_to_surface(&mut self, prepared: PreparedFrame, surface: &goldy::Surface) -> Result<FrameStats> {
        match self {
            Self::Classic(r) => r.submit_to_surface(prepared, surface),
            Self::Scheme(_) => panic!(
                "GoldyRenderer::submit_to_surface called on Scheme backend — \
                 use submit_to_swapchain(prepared, pool) instead"
            ),
        }
    }

    /// Phase 2: record GPU work and return frame stats plus a present token (Scheme path).
    ///
    /// Does not call [`PresentToken::present`]; the caller must present synchronously or
    /// hand the token to `TID_PRESENT` for async scanout.
    ///
    /// Panics if called on the Classic backend.
    pub fn submit_to_swapchain(
        &mut self,
        prepared: PreparedFrame,
        pool: &goldy::SwapchainPool,
    ) -> Result<(FrameStats, PresentToken)> {
        self.submit_to_swapchain_with(prepared, pool, || Ok(()))
    }

    /// Like [`Self::submit_to_swapchain`], but runs `pre_acquire` after the upload
    /// scheme is submitted and immediately before the worker acquires its drawable
    /// (Scheme backend only).
    pub fn submit_to_swapchain_with<F>(
        &mut self,
        prepared: PreparedFrame,
        pool: &goldy::SwapchainPool,
        pre_acquire: F,
    ) -> Result<(FrameStats, PresentToken)>
    where
        F: FnOnce() -> Result<()>,
    {
        match self {
            Self::Classic(_) => panic!(
                "GoldyRenderer::submit_to_swapchain called on Classic backend — \
                 use submit_to_surface(prepared, surface) instead"
            ),
            Self::Scheme(r) => r.submit_to_swapchain_with(prepared, pool, pre_acquire),
        }
    }

    /// Submit without presenting (Classic path only).
    pub fn submit_prepared(
        &mut self,
        prepared: PreparedFrame,
        surface: &goldy::Surface,
    ) -> Result<(FrameStats, goldy::Frame)> {
        match self {
            Self::Classic(r) => r.submit_prepared(prepared, surface),
            Self::Scheme(_) => panic!(
                "GoldyRenderer::submit_prepared called on Scheme backend — \
                 use submit_to_swapchain(prepared, pool) instead"
            ),
        }
    }

    /// Acknowledge a swapchain frame after [`goldy::Frame::present`] (Classic path).
    pub fn note_frame_presented(&mut self, tv: TimelineValue) {
        match self {
            Self::Classic(r) => r.note_frame_presented(tv),
            Self::Scheme(_) => {}
        }
    }

    /// Render a scene and return the pixel data as RGBA bytes (synchronous).
    pub fn render_to_buffer(&mut self, scene: &crate::Scene, params: &RenderParams) -> Result<Vec<u8>> {
        match self {
            Self::Classic(r) => r.render_to_buffer(scene, params),
            Self::Scheme(r) => r.render_to_buffer(scene, params),
        }
    }

    /// Query frame-scheduling state for diagnostics or test assertions.
    pub fn allocator_stats(&self) -> AllocatorStats {
        match self {
            Self::Classic(r) => r.allocator_stats(),
            Self::Scheme(r) => r.allocator_stats(),
        }
    }

    /// Query the resource pool's current state for diagnostics or test assertions.
    pub fn resource_pool_stats(&self) -> ResourcePoolStats {
        match self {
            Self::Classic(r) => r.resource_pool_stats(),
            Self::Scheme(r) => r.resource_pool_stats(),
        }
    }

    /// `true` if the submission context still holds unreclaimed deferred payloads.
    pub fn has_deferred_payloads(&self) -> bool {
        match self {
            Self::Classic(r) => r.has_deferred_payloads(),
            Self::Scheme(r) => r.has_deferred_payloads(),
        }
    }

    /// Pull-side reclamation: drain the submission context's deferred-deletion ring.
    pub fn flush_deferred_deletions(&self) {
        match self {
            Self::Classic(r) => r.flush_deferred_deletions(),
            Self::Scheme(r) => r.flush_deferred_deletions(),
        }
    }

    /// Query the render context's placement heap state.
    pub fn placement_heap_stats(&self) -> Option<goldy::placement_heap::PlacementHeapStats> {
        match self {
            Self::Classic(r) => r.placement_heap_stats(),
            Self::Scheme(r) => r.placement_heap_stats(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::graph_gpu_resources::PipelineResources;
    use crate::graph_renderer::GraphRecorder;
    use crate::{RenderParams, Scene};
    use ekrano_encoding::{RenderConfig, Resolver};
    use goldy::{
        Adapter, BackendType, Device, DeviceDescriptor, FrameOrchestrator, Instance, RequestAdapterOptions, TaskGraph,
    };
    use std::ops::Deref;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    static WARP_LIB_TEST_SERIAL: OnceLock<Mutex<()>> = OnceLock::new();

    fn is_dx12_warp_adapter(instance: &Instance, adapter: &Adapter) -> bool {
        // goldy::WARP_ADAPTER_ID is u32::MAX; ekrano does not gate on goldy's `dx12` feature.
        instance.backend_type() == BackendType::Dx12 && adapter.id() == u32::MAX
    }

    fn warp_lib_test_serial() -> &'static Mutex<()> {
        WARP_LIB_TEST_SERIAL.get_or_init(|| Mutex::new(()))
    }

    /// Fresh GPU device for ordinary `--lib` tests, with WARP serialization when needed.
    ///
    /// One device per test body via the public [`Instance`] / [`RequestAdapterOptions`] /
    /// [`DeviceDescriptor`] APIs. On DX12 WARP, holds a process-wide lock for the guard's
    /// lifetime so parallel `cargo test --lib` trials do not interleave WARP work.
    pub(crate) struct GpuTestDevice {
        device: Device,
        _warp_guard: Option<MutexGuard<'static, ()>>,
    }

    impl Deref for GpuTestDevice {
        type Target = Device;

        fn deref(&self) -> &Device {
            &self.device
        }
    }

    fn try_create_gpu_test_device() -> Option<GpuTestDevice> {
        let instance = Instance::new().ok()?;
        let adapter = instance.request_adapter(&RequestAdapterOptions::default()).ok()?;

        // Lock before `request_device` when the selected adapter is WARP — adapter
        // selection is cheap/non-racy; device open is what must be serialized.
        let _warp_guard = if is_dx12_warp_adapter(&instance, &adapter) {
            Some(warp_lib_test_serial().lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            None
        };

        let device = adapter.request_device(&DeviceDescriptor::default()).ok()?;
        drop(instance);

        Some(GpuTestDevice { device, _warp_guard })
    }

    /// Shared test helper: fresh GPU device plus a fresh [`PersistentState`].
    ///
    /// Returns `None` when no GPU adapter is available. Hold the returned
    /// [`GpuTestDevice`] for the full test body so WARP serialization stays active.
    pub(crate) fn make_device_and_persistent() -> Option<(GpuTestDevice, PersistentState)> {
        let gpu = try_create_gpu_test_device()?;
        let mut persistent = PersistentState::new(&gpu);
        let pending = persistent.pending_owned_returns.clone();
        persistent.pool.set_pending_returns(pending);
        Some((gpu, persistent))
    }

    /// Regression test: `PipelineResources::prepare` must create `out_image` with
    /// the format supplied by the caller, not hardcode `Rgba8Unorm`.
    ///
    /// Before the fix, `acquire_texture_rgba` was called unconditionally, so
    /// `out_image` was always `Rgba8Unorm`. When the surface rendering path then
    /// executed `CopyResource(out_image → frame.texture())` where `frame.texture()`
    /// has the `Bgra8Unorm` swapchain format, the byte-level copy swapped R and B
    /// channels — turning orange content blue (the velato tiger regression).
    ///
    /// The fix passes `out_image_format` through to `prepare`, which uses it when
    /// acquiring the texture. This test verifies that invariant for both the
    /// surface (BGRA) and headless (RGBA) paths.
    #[test]
    fn prepare_out_image_format_matches_requested() {
        let Some((gpu, mut persistent)) = make_device_and_persistent() else {
            return;
        };

        let scene = Scene::new();
        let encoding = scene.encoding();

        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        for &expected_format in &[TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm] {
            let mut resolver = Resolver::new();
            let mut packed = Vec::new();
            let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
            let config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);
            let mut graph = TaskGraph::new();

            let ctx = gpu.create_context().expect("context");
            let mut frame_pipeline = FrameOrchestrator::new(&ctx, FRAME_PIPELINE_DEPTH);
            let frame_handle = frame_pipeline
                .begin_frame(|_, _| Ok::<(), Error>(()))
                .expect("begin_frame");
            let pipeline = {
                let mut recorder = GraphRecorder::new(
                    &gpu,
                    &ctx,
                    &mut graph,
                    &mut frame_pipeline,
                    frame_handle,
                    &mut persistent,
                    &[],
                    None,
                    None,
                );
                PipelineResources::prepare(
                    &mut recorder,
                    encoding.coverage_mask.as_ref(),
                    packed,
                    ramps,
                    images,
                    &params,
                    &config,
                    expected_format,
                )
                .unwrap_or_else(|e| panic!("PipelineResources::prepare({expected_format:?}) failed: {e}"))
            };

            assert_eq!(
                pipeline.out_image.format(),
                expected_format,
                "out_image must use the requested format {expected_format:?}; \
                 using Rgba8Unorm unconditionally would cause CopyResource to swap \
                 R and B when copying to a Bgra8Unorm swapchain texture"
            );
        }
    }
}
