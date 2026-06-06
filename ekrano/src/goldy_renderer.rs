// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy-based renderer for offscreen and texture rendering.
//!
//! Use this when building with `--no-default-features --features goldy`.
//!
//! ## Phase 3c Option 1A: Bindless descriptor model
//!
//! We use Goldy's bindless descriptor indexing (global arrays of up to 16K
//! descriptors per type) rather than actual buffer device addresses (BDA).
//! Push constants carry bindless indices per dispatch via Slang `uniform`
//! entry-point parameters. This is simpler than wgpu's per-pipeline bind group
//! layouts and satisfies the "simplify the binding model" goal. BDA would only
//! be needed for GPU-side pointer chasing (e.g. buffer pools); we defer that
//! unless required.

pub const MAX_BINDLESS_SLOTS: usize = 16;

use std::collections::{HashMap, VecDeque};
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use goldy::task_graph::{NodeAccess, NodeBuilder};
use goldy::types::{BufferFlags, TextureFlags, TextureFormat, TextureKind};
use goldy::{
    Buffer, BufferKind, BufferPool, BufferView, ComputePipeline, Context, Device, FrameHandle,
    FrameOrchestrator, ShaderModule, Signal, TaskGraph, Texture, TexturePool, TimelineValue,
};

/// Ekrano uses a single-frame fire-and-forget model (vello #71 baseline).
const FRAME_PIPELINE_DEPTH: usize = 1;

use mem::size_of;

use crate::{
    Error, RenderParams, Result, Scene,
    gpu_resources::{
        BufferLifetime, GpuBinding, GpuBuf, bind_type_to_node_access,
        collect_bindless_indices_into, record_upload_bytes_owned,
    },
    render::Render,
    resource_proxy::{BindType, ShaderId},
    shaders::{self, FullShaders},
};
use ekrano_encoding::{BumpAllocators, Images, Layout, Ramps, RenderConfig, Resolver};

const MAX_BUMP_RETRIES: usize = 2;

/// Timeline-guarded slots for cross-frame render-target and pipeline-buffer reuse.
/// Three slots match Metal/Vulkan `MAX_FRAMES_IN_FLIGHT=3`: the oldest entry can
/// retire once acquire waits on the slot being reused (~3 frames ago).
const RESOURCE_CACHE_SLOTS: usize = 3;

fn find_empty_cache_slot<T>(slots: &[Option<T>]) -> Option<usize> {
    slots.iter().position(Option::is_none)
}

/// Return the oldest GPU-ready cache slot index, or `None` if no slot qualifies.
#[inline]
#[allow(dead_code, reason = "alternative cache eviction strategies")]
fn oldest_ready_cache_slot(
    timelines: [TimelineValue; RESOURCE_CACHE_SLOTS],
    occupied: [bool; RESOURCE_CACHE_SLOTS],
    progress: TimelineValue,
) -> Option<usize> {
    let mut best: Option<(TimelineValue, usize)> = None;
    for i in 0..RESOURCE_CACHE_SLOTS {
        if occupied[i] && progress >= timelines[i] {
            match best {
                None => best = Some((timelines[i], i)),
                Some((best_tv, _)) if timelines[i] < best_tv => {
                    best = Some((timelines[i], i));
                }
                _ => {}
            }
        }
    }
    best.map(|(_, i)| i)
}

/// Return ready slot indices in ascending timeline order (padded with `None`).
#[inline]
#[allow(dead_code, reason = "alternative cache eviction strategies")]
fn ready_cache_slots_oldest_first(
    timelines: [TimelineValue; RESOURCE_CACHE_SLOTS],
    occupied: [bool; RESOURCE_CACHE_SLOTS],
    progress: TimelineValue,
) -> [Option<usize>; RESOURCE_CACHE_SLOTS] {
    let mut ready: Vec<(TimelineValue, usize)> = (0..RESOURCE_CACHE_SLOTS)
        .filter(|&i| occupied[i] && progress >= timelines[i])
        .map(|i| (timelines[i], i))
        .collect();
    ready.sort_by_key(|&(tv, _)| tv);
    let mut out = [None; RESOURCE_CACHE_SLOTS];
    for (j, &(_, idx)) in ready.iter().enumerate().take(RESOURCE_CACHE_SLOTS) {
        out[j] = Some(idx);
    }
    out
}

/// Extra space added to the storage pool beyond the exact allocation required.
///
/// Provides headroom for sub-allocation rounding and small over-allocations
/// from the bump allocator growth path without triggering a full pool realloc.
const POOL_SIZE_SLACK: u64 = 256 * 1024;

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
    /// Reserved for API compatibility (ekrano no longer uses a transient allocator).
    pub capacity: u64,
    /// Reserved for API compatibility (ekrano no longer uses a transient allocator).
    pub used: u64,
    /// Number of frames waiting in the [`FrameOrchestrator`] ring (always ≤ 1).
    pub cleanup_ring_depth: usize,
}

/// Snapshot of the resource pool's state, useful for tests and diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct ResourcePoolStats {
    /// Total number of pooled buffers across all keys.
    pub total_pooled_buffers: usize,
    /// Number of distinct `(size, access, name, flags)` keys in the pool.
    pub distinct_keys: usize,
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
fn sanitize_bump(bump: &BumpAllocators) -> BumpAllocators {
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
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

// -----------------------------------------------------------------------
// Deferred per-frame work
// -----------------------------------------------------------------------
//
// The `FrameOrchestrator` ring now carries `()` — all resource retirement
// (render targets, pipeline buffers, pool views, owned buffers, bump readback)
// bypasses the ring via timeline-guarded cache slots + `Device::defer_release`
// + `PersistentState::pending_bump_readbacks`.  The ring is a pure scheduling
// primitive: depth enforcement + timeline tracking only.

/// Token pushed into a [`DeferredPayload`] when pool views are retired.
///
/// Batching all views into one payload avoids per-view `defer_release` overhead on
/// the hot path (dozens of mutex pushes per frame at 2000+ FPS).
///
/// [`DeferredPayload`]: goldy::DeferredPayload
struct DeferredPoolViewsToken(
    #[allow(dead_code, reason = "views dropped when token is reclaimed")] Vec<BufferView>,
);

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

/// Which persistent cache slots received new entries during
/// [`FrameRecorder::schedule_pipeline_cleanup`].
#[derive(Debug, Default)]
pub(crate) struct CacheScheduleOutcome {
    cached_render_targets_slot: Option<usize>,
    cached_pipeline_slot: Option<usize>,
}

/// Outcome of [`FrameRecorder::finish`]: orchestrator submit result plus resources
/// that bypass the ring and are deferred via [`Device::defer_release`].
struct FrameFinishOutcome {
    timeline: Option<TimelineValue>,
    surface_frame: Option<goldy::Frame>,
    bump_readback: Option<Buffer>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
    recyclable_owned: Vec<(Buffer, &'static str)>,
}

/// CPU-resolved scene data ready for GPU submission.
///
/// Produced by [`GoldyRenderer::prepare`] (pure CPU — safe to call while the
/// previous frame is still on the GPU) and consumed by
/// [`GoldyRenderer::submit_to_surface`] or [`GoldyRenderer::submit_prepared`].
/// Owns all data extracted from the scene encoding so it can be stored across
/// event loop iterations.
pub struct PreparedFrame {
    packed: Vec<u8>,
    layout: Layout,
    ramps_data: Vec<u32>,
    ramps_width: u32,
    ramps_height: u32,
    images_width: u32,
    images_height: u32,
    image_entries: Vec<(peniko::ImageData, u32, u32)>,
    config: RenderConfig,
    pool_size: u64,
    params: RenderParams,
    scene_fingerprint: u64,
    resolver: Resolver,
    /// Owned copy of `Encoding::coverage_mask` — used by `PipelineResources::prepare`.
    coverage_mask: Option<ekrano_encoding::CoverageMask>,
    /// Owned copy of `Encoding::layer_filter_effects` — used by `record_fine` and
    /// `record_filter_effects`.
    layer_filter_effects: Vec<ekrano_encoding::LayerFilterEffect>,
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

/// Defer pool views, textures, and recyclable owned buffers until `tv` retires on the GPU.
///
/// Uses a single [`Context::defer_release`] per frame (one mutex push) instead of
/// multiple deferred cleanup calls.
fn defer_frame_gpu_resources(
    ctx: &Context,
    persistent: &PersistentState,
    tv: TimelineValue,
    pool_views: Vec<BufferView>,
    textures: Vec<Texture>,
    recyclable_owned: Vec<(Buffer, &'static str)>,
) {
    let mut payload = goldy::DeferredPayload::new();
    if !pool_views.is_empty() {
        payload.push(DeferredPoolViewsToken(pool_views));
    }
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

struct GoldyShader {
    pipeline: ComputePipeline,
    bindings: Vec<BindType>,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct BufferKey {
    size: u64,
    access: BufferKind,
    name: &'static str,
    buffer_flags: BufferFlags,
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
        device: &Device,
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

        // Pool miss: try a fresh allocation.
        match device.alloc_buffer(size, access, stride, buffer_flags) {
            Ok(b) => Ok(b),
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
                device
                    .alloc_buffer(size, access, stride, buffer_flags)
                    .map_err(|e| Error::Shader(e.to_string()))
            }
        }
    }

    /// Return a buffer to the pool for reuse by a future frame.
    pub(crate) fn return_buf(&mut self, buf: Buffer, name: &'static str) {
        let key = BufferKey {
            size: buf.size(),
            access: buf.access(),
            name,
            buffer_flags: buf.flags(),
        };
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
fn env_robust_override() -> Option<bool> {
    std::env::var("EKRANO_ROBUST").ok().map(|v| {
        !matches!(
            v.to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

// -----------------------------------------------------------------------
// PersistentState — GPU resources that survive across frames
// -----------------------------------------------------------------------

/// GPU resources that live for the lifetime of the renderer and are reused
/// across frames. Pool growth, texture reuse, and bump estimates all live here.
pub(crate) struct PersistentState {
    /// Owned buffer cache: recycles pool-exempt buffers (bump, indirect, etc.)
    pub(crate) pool: ResourcePool,
    /// Texture pool for intermediate render targets (gradient, filter layers, etc.)
    pub(crate) tex_pool: TexturePool,
    /// Bump allocator counters from the most recently drained frame.
    /// `None` until the first GPU readback completes.
    last_drained_bump: Option<BumpAllocators>,
    /// Bump readback buffers whose GPU timelines are not yet complete.
    ///
    /// Used for pipelined strategies so bump readback does not force
    /// [`FrameOrchestrator::begin_frame`] to wait before admitting a frame.
    pending_bump_readbacks: VecDeque<(TimelineValue, Buffer)>,
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
    /// Cached `OwnedShared` + pool-exempt pipeline buffers from previous frames.
    ///
    /// Same timeline-guard pattern as `cached_render_targets`.
    pub(crate) cached_pipeline:
        [Option<crate::gpu_resources::CachedPipeline>; RESOURCE_CACHE_SLOTS],
    /// Timeline of the frame that last wrote each pipeline cache slot.
    pub(crate) cached_pipeline_timelines: [TimelineValue; RESOURCE_CACHE_SLOTS],
    /// Maps RT cache slot index → swapchain image index of the last frame that
    /// used that slot, so [`Self::mark_rt_slot_returned`] can mark it reusable.
    pub(crate) rt_slot_swapchain_image: [Option<u32>; RESOURCE_CACHE_SLOTS],
    /// Capacity hints for `FrameRecorder` scratch allocations. Updated after each
    /// `finish()` call so that the next frame pre-allocates the right amount and
    /// avoids re-allocations on the hot path.
    pub(crate) deferred_owned_cap_hint: usize,
    pub(crate) deferred_pool_cap_hint: usize,
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
    /// Textures waiting to be returned to [`Self::tex_pool`] after GPU retirement.
    /// Populated by [`DeferredTextureToken`] drops from [`Context::defer_release`].
    pending_texture_returns: Arc<Mutex<Vec<Texture>>>,
    /// Owned buffers waiting to be returned to [`Self::pool`] after GPU retirement.
    /// Populated by [`DeferredOwnedBuffersToken`] drops from [`Context::defer_release`].
    pending_owned_returns: Arc<Mutex<Vec<(Buffer, &'static str)>>>,
}

/// Timeline-guarded render target cache.
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
            slot.as_ref().is_some_and(|(out, _)| {
                out.width() != width || out.height() != height || out.format() != out_format
            })
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
        ctx: &Context,
        progress: TimelineValue,
        width: u32,
        height: u32,
        out_format: TextureFormat,
    ) -> Option<(Texture, [Texture; 4])> {
        let occupied = std::array::from_fn(|i| self.cached_render_targets[i].is_some());
        let mut progress = progress;
        let order = ready_cache_slots_oldest_first(self.cached_rt_timelines, occupied, progress);
        if order[0].is_none() {
            let all_full = occupied.iter().all(|&o| o);
            if all_full {
                let oldest = (0..RESOURCE_CACHE_SLOTS)
                    .filter(|&i| occupied[i])
                    .map(|i| self.cached_rt_timelines[i])
                    .min()
                    .unwrap_or(0);
                if progress < oldest {
                    // Fast-path guard: if nothing is in-flight the GPU is idle and all
                    // cache slots must already be ready — skip the blocking wait.
                    if ctx.peek_oldest_in_flight().is_none() {
                        progress = ctx.gpu_progress();
                    } else {
                        log::debug!(
                            "[RT-CACHE] backpressure: progress={progress} < oldest={oldest}, waiting"
                        );
                        let wait_ok = ctx.wait_until(oldest).is_ok();
                        progress = ctx.gpu_progress();
                        if wait_ok {
                            let order2 = ready_cache_slots_oldest_first(
                                self.cached_rt_timelines,
                                occupied,
                                progress,
                            );
                            if order2[0].is_some() {
                                return self.take_cached_render_targets(
                                    ctx, progress, width, height, out_format,
                                );
                            }
                        }
                    }
                }
            }
            log::warn!(
                "[RT-CACHE] MISS (gpu): progress={progress} timelines={:?} occupied={:?}",
                self.cached_rt_timelines,
                occupied,
            );
            return None;
        }
        for i in order.into_iter().flatten() {
            let Some((out, layers)) = self.cached_render_targets[i].take() else {
                continue;
            };
            if out.width() == width && out.height() == height && out.format() == out_format {
                log::debug!(
                    "[RT-CACHE] HIT slot={i}: progress={progress} timeline={} delta={}",
                    self.cached_rt_timelines[i],
                    progress as i64 - self.cached_rt_timelines[i] as i64,
                );
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
        log::warn!(
            "[RT-CACHE] MISS (no match): progress={progress} timelines={:?} occupied={:?}",
            self.cached_rt_timelines,
            occupied,
        );
        None
    }
}

impl PersistentState {
    fn last_drained_bump(&self) -> Option<&BumpAllocators> {
        self.last_drained_bump.as_ref()
    }

    fn take_last_drained_bump(&mut self) -> Option<BumpAllocators> {
        self.last_drained_bump.take()
    }

    fn queue_bump_readback(&mut self, timeline: TimelineValue, buf: Buffer) {
        self.pending_bump_readbacks.push_back((timeline, buf));
    }

    fn drain_ready_bump_readbacks(&mut self, device: &Device, ctx: &Context) -> Result<()> {
        if self.pending_bump_readbacks.is_empty() {
            return Ok(());
        }
        let _tz = goldy::tracy_zone!("ekrano.drain_ready_bump_readbacks");
        let progress = ctx.gpu_progress();
        while self
            .pending_bump_readbacks
            .front()
            .is_some_and(|(timeline, _)| *timeline <= progress)
        {
            let (_, buf) = self
                .pending_bump_readbacks
                .pop_front()
                .expect("front existed");
            read_bump_buffer(device, self, buf)?;
        }
        Ok(())
    }

    /// Claim a pipeline buffer cache slot.
    ///
    /// Pipeline buffers are fully overwritten each frame, so reuse is safe
    /// without waiting for the GPU to finish the prior frame's reads.
    pub(crate) fn take_cached_pipeline(
        &mut self,
        progress: TimelineValue,
    ) -> Option<crate::gpu_resources::CachedPipeline> {
        let occupied = std::array::from_fn(|i| self.cached_pipeline[i].is_some());
        let Some(i) = oldest_ready_cache_slot(self.cached_pipeline_timelines, occupied, progress)
        else {
            log::debug!(
                "[PIPE-CACHE] MISS (gpu): progress={progress} timelines={:?} occupied={:?}",
                self.cached_pipeline_timelines,
                occupied,
            );
            return None;
        };
        if let Some(c) = self.cached_pipeline[i].take() {
            log::debug!(
                "[PIPE-CACHE] HIT slot={i}: timeline={}",
                self.cached_pipeline_timelines[i],
            );
            Some(c)
        } else {
            log::debug!("[PIPE-CACHE] MISS (empty)");
            None
        }
    }

    /// Return textures and owned buffers whose GPU retirement completed since the last frame.
    fn drain_pending_returns(&mut self) {
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

fn read_bump_buffer(device: &Device, persistent: &mut PersistentState, buf: Buffer) -> Result<()> {
    let _bump = goldy::tracy_zone!("ekrano.bump_readback");
    let size = buf.size() as usize;
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
        let _parse = goldy::tracy_zone!("ekrano.bump_readback.parse");
        persistent.last_drained_bump = Some(bytemuck::pod_read_unaligned(&output));
    }
    {
        let _return = goldy::tracy_zone!("ekrano.bump_readback.return_pool");
        persistent.pool.return_buf(buf, "ekrano.bump_buf");
    }
    Ok(())
}

// -----------------------------------------------------------------------
// GoldyRenderer — the merged struct
// -----------------------------------------------------------------------

/// Goldy-based 2D renderer.
///
/// Renders scenes to textures using the Goldy GPU backend with Slang shaders.
pub struct GoldyRenderer {
    device: Device,
    /// Submission context for the caller's (unbudgeted) device VRAM ring.
    unbudgeted_context: Context,
    context: Context,
    shaders: FullShaders,
    resolver: Resolver,
    engine_shaders: Vec<GoldyShader>,
    /// Cross-frame GPU resources: pools, texture cache, bump readback.
    persistent: PersistentState,
    /// Pipelined frame scheduling: depth enforcement and timeline tracking.
    frame_pipeline: FrameOrchestrator<()>,
    /// Persistent bump estimates: running max across frames. Used to pre-size
    /// buffers even when no overflow occurs, avoiding the cold-start ramp-up.
    persistent_bump: Option<BumpAllocators>,
    /// Frame counter for rate-limiting housekeeping operations.
    cleanup_frame_counter: u64,
    /// Fingerprint of the last rendered scene (hash of packed bytes + params).
    /// Used to detect scene changes for command-list caching.
    last_scene_fingerprint: Option<u64>,
    /// Length of the packed scene bytes from the previous frame for quick-reject.
    last_packed_len: Option<usize>,
    /// Long-lived task graph cleared (not replaced) each frame so the schedule cache
    /// survives across frames. `FrameRecorder` borrows this mutably per frame.
    graph: TaskGraph,
}

// -----------------------------------------------------------------------
// FrameRecorder — direct-execution recorder that builds TaskGraph nodes
// -----------------------------------------------------------------------

pub(crate) struct FrameRecorder<'a> {
    pub(crate) device: &'a Device,
    pub(crate) context: &'a Context,
    graph: &'a mut TaskGraph,
    frame_pipeline: &'a mut FrameOrchestrator<()>,
    frame_handle: FrameHandle,
    pub(crate) persistent: &'a mut PersistentState,
    shaders: &'a [GoldyShader],
    surface: Option<&'a goldy::Surface>,
    preacquired_frame: Option<goldy::Frame>,
    last_timeline: Option<TimelineValue>,
    /// Set to `true` by `finish` or `abort`; the `Drop` impl aborts the open frame
    /// if the recorder is dropped without being properly completed (e.g. on a `?` return).
    finished: bool,
    /// The bump readback buffer for the current frame (`robust=true` only).
    /// Queued into `PersistentState::pending_bump_readbacks` after GPU submit.
    bump_buf_for_readback: Option<Buffer>,
    deferred_owned_buffers: Vec<(Buffer, &'static str)>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
    /// Reusable scratch buffer for bindless index collection. Pre-allocated to
    /// `MAX_BINDLESS_SLOTS` capacity so each dispatch swaps it out rather than
    /// calling `malloc` on a fresh `Vec`.
    indices_scratch: Vec<u32>,
    /// Per-frame filter dispatch slot counter, incremented by each `filter_dispatch` call.
    /// Used to index into `PersistentState::cached_filter_uniforms` for cache lookup.
    /// Reset to 0 at the start of each frame.
    pub(crate) filter_dispatch_slot: usize,
}

impl<'a> FrameRecorder<'a> {
    fn new(
        device: &'a Device,
        context: &'a Context,
        graph: &'a mut TaskGraph,
        frame_pipeline: &'a mut FrameOrchestrator<()>,
        frame_handle: FrameHandle,
        persistent: &'a mut PersistentState,
        shaders: &'a [GoldyShader],
        surface: Option<&'a goldy::Surface>,
        preacquired_frame: Option<goldy::Frame>,
    ) -> Self {
        // Clear the long-lived graph so the schedule cache is preserved but
        // the node list is empty and ready for this frame's recording.
        graph.clear();
        // Read capacity hints before persistent is moved into Self.
        let owned_cap = persistent.deferred_owned_cap_hint;
        let pool_cap = persistent.deferred_pool_cap_hint;
        let tex_cap = persistent.deferred_textures_cap_hint;

        Self {
            device,
            context,
            graph,
            frame_pipeline,
            frame_handle,
            persistent,
            shaders,
            surface,
            preacquired_frame,
            last_timeline: None,
            bump_buf_for_readback: None,
            deferred_owned_buffers: Vec::with_capacity(owned_cap),
            deferred_pool_views: Vec::with_capacity(pool_cap),
            deferred_textures: Vec::with_capacity(tex_cap),
            finished: false,
            indices_scratch: Vec::with_capacity(MAX_BINDLESS_SLOTS),
            filter_dispatch_slot: 0,
        }
    }

    pub(crate) fn graph_and_persistent(&mut self) -> (&mut TaskGraph, &mut PersistentState) {
        (&mut *self.graph, self.persistent)
    }

    // NOTE: There is no explicit mid-frame flush. The backend is free to split
    // a single TaskGraph into multiple command buffers as an implementation
    // detail (e.g. coarse vs fine partitions). Callers should not assume or
    // rely on intra-frame submission boundaries.

    pub(crate) fn alloc_pipeline_buffer_named(
        &mut self,
        size: u64,
        stride: u32,
        name: &'static str,
        flags: BufferFlags,
        lifetime: BufferLifetime,
    ) -> Result<GpuBuf, Error> {
        crate::gpu_resources::alloc_pipeline_buffer(
            self.device,
            self.context,
            self.graph,
            self.persistent,
            size,
            stride,
            name,
            flags,
            lifetime,
        )
    }

    pub(crate) fn defer_gpu_buf(&mut self, buf: GpuBuf, name: &'static str) {
        match buf {
            GpuBuf::Owned(b) => self.deferred_owned_buffers.push((b, name)),
        }
    }

    pub(crate) fn defer_texture(&mut self, tex: Texture) {
        self.deferred_textures.push(tex);
    }

    fn defer_cached_pipeline_owned_buffers(&mut self, c: crate::gpu_resources::CachedPipeline) {
        self.deferred_owned_buffers
            .push((c.info_bin_data, "ekrano.info_bin_data_buf"));
        self.deferred_owned_buffers
            .push((c.tile, "ekrano.tile_buf"));
        self.deferred_owned_buffers
            .push((c.segments, "ekrano.segments_buf"));
        self.deferred_owned_buffers
            .push((c.ptcl, "ekrano.ptcl_buf"));
        self.deferred_owned_buffers
            .push((c.blend_spill, "ekrano.blend_spill"));
        macro_rules! defer_coarse {
            ($field:expr, $name:expr) => {
                if let Some(b) = $field {
                    self.deferred_owned_buffers.push((b, $name));
                }
            };
        }
        defer_coarse!(c.reduced, "ekrano.reduced_buf");
        defer_coarse!(c.reduced2, "ekrano.reduced2_buf");
        defer_coarse!(c.reduced_scan, "ekrano.reduced_scan_buf");
        defer_coarse!(c.tagmonoid, "ekrano.tagmonoid_buf");
        defer_coarse!(c.path_bbox, "ekrano.path_bbox_buf");
        defer_coarse!(c.lines, "ekrano.lines_buf");
        defer_coarse!(c.draw_reduced, "ekrano.draw_reduced_buf");
        defer_coarse!(c.draw_monoid, "ekrano.draw_monoid_buf");
        defer_coarse!(c.clip_inp, "ekrano.clip_inp_buf");
        defer_coarse!(c.clip_el, "ekrano.clip_el_buf");
        defer_coarse!(c.clip_bic, "ekrano.clip_bic_buf");
        defer_coarse!(c.clip_bbox, "ekrano.clip_bbox_buf");
        defer_coarse!(c.draw_bbox, "ekrano.draw_bbox_buf");
        defer_coarse!(c.bin_header, "ekrano.bin_header_buf");
        defer_coarse!(c.path, "ekrano.path_buf");
        defer_coarse!(c.seg_counts, "ekrano.seg_counts_buf");
    }

    pub(crate) fn schedule_pipeline_cleanup(
        &mut self,
        pipeline: crate::gpu_resources::PipelineResources,
        bump_readback: bool,
    ) -> CacheScheduleOutcome {
        let mut outcome = CacheScheduleOutcome::default();
        let crate::gpu_resources::PipelineResources {
            gradient,
            image_atlas,
            mask_atlas,
            scene,
            config,
            indirect,
            info_bin_data,
            tile,
            segments,
            ptcl,
            reduced,
            reduced2,
            reduced_scan,
            tagmonoid,
            path_bbox,
            bump,
            lines,
            draw_reduced,
            draw_monoid,
            clip_inp,
            clip_el,
            clip_bic,
            clip_bbox,
            draw_bbox,
            bin_header,
            path,
            seg_counts,
            blend_spill,
            out_image,
            filter_layers,
            buffer_sizes,
            config_uniform_value,
        } = pipeline;

        self.defer_texture(gradient);
        self.defer_texture(image_atlas);
        self.defer_texture(mask_atlas);
        self.defer_gpu_buf(scene, "ekrano.scene");
        // Stash the config buffer back into the persistent cache so the next frame can
        // reuse it without recording a WriteBuffer node when the value is unchanged.
        match config {
            GpuBuf::Owned(buf) => {
                self.persistent.cached_config_uniform = Some((config_uniform_value, buf));
            }
            other => self.defer_gpu_buf(other, "ekrano.config"),
        }
        if let Some(b) = indirect {
            self.defer_gpu_buf(b, "ekrano.indirect_dispatch");
        }
        // Stash the OwnedShared and pool-exempt buffers for cross-frame reuse.
        // They will be stored in PersistentState::cached_pipeline after GPU retirement
        // and reused by the next PipelineResources::prepare when buffer_sizes match.
        let cacheable_info_bin_data = match info_bin_data {
            GpuBuf::Owned(b) => Some(b),
            other => {
                self.defer_gpu_buf(other, "ekrano.info_bin_data_buf");
                None
            }
        };
        let cacheable_tile = match tile {
            GpuBuf::Owned(b) => Some(b),
            other => {
                self.defer_gpu_buf(other, "ekrano.tile_buf");
                None
            }
        };
        let cacheable_segments = match segments {
            GpuBuf::Owned(b) => Some(b),
            other => {
                self.defer_gpu_buf(other, "ekrano.segments_buf");
                None
            }
        };
        let cacheable_ptcl = match ptcl {
            GpuBuf::Owned(b) => Some(b),
            other => {
                self.defer_gpu_buf(other, "ekrano.ptcl_buf");
                None
            }
        };
        macro_rules! stash_coarse_buf {
            ($buf:expr, $name:expr) => {
                match $buf {
                    GpuBuf::Owned(b) => Some(b),
                    other => {
                        self.defer_gpu_buf(other, $name);
                        None
                    }
                }
            };
        }
        let cacheable_reduced = stash_coarse_buf!(reduced, "ekrano.reduced");
        let cacheable_reduced2 = stash_coarse_buf!(reduced2, "ekrano.reduced2");
        let cacheable_reduced_scan = stash_coarse_buf!(reduced_scan, "ekrano.reduced_scan");
        let cacheable_tagmonoid = stash_coarse_buf!(tagmonoid, "ekrano.tagmonoid");
        let cacheable_path_bbox = stash_coarse_buf!(path_bbox, "ekrano.path_bbox");
        if bump_readback {
            if let GpuBuf::Owned(b) = bump {
                self.bump_buf_for_readback = Some(b);
            } else {
                self.defer_gpu_buf(bump, "ekrano.bump_buf");
            }
        } else {
            self.defer_gpu_buf(bump, "ekrano.bump_buf");
        }
        let cacheable_lines = stash_coarse_buf!(lines, "ekrano.lines");
        let cacheable_draw_reduced = stash_coarse_buf!(draw_reduced, "ekrano.draw_reduced");
        let cacheable_draw_monoid = stash_coarse_buf!(draw_monoid, "ekrano.draw_monoid");
        let cacheable_clip_inp = stash_coarse_buf!(clip_inp, "ekrano.clip_inp");
        let cacheable_clip_el = stash_coarse_buf!(clip_el, "ekrano.clip_el");
        let cacheable_clip_bic = stash_coarse_buf!(clip_bic, "ekrano.clip_bic");
        let cacheable_clip_bbox = stash_coarse_buf!(clip_bbox, "ekrano.clip_bbox");
        let cacheable_draw_bbox = stash_coarse_buf!(draw_bbox, "ekrano.draw_bbox");
        let cacheable_bin_header = stash_coarse_buf!(bin_header, "ekrano.bin_header");
        let cacheable_path = stash_coarse_buf!(path, "ekrano.path");
        let cacheable_seg_counts = stash_coarse_buf!(seg_counts, "ekrano.seg_counts");
        let cacheable_blend_spill = match blend_spill {
            GpuBuf::Owned(b) => Some(b),
            other => {
                self.defer_gpu_buf(other, "ekrano.blend_spill");
                None
            }
        };
        // Build CachedPipeline only when all six OwnedShared buffers were Owned.
        // If any resolved to a different variant (e.g. Pooled on WARP/CPU device),
        // fall through to the normal deferred-pool path for those.
        match (
            cacheable_info_bin_data,
            cacheable_tile,
            cacheable_segments,
            cacheable_ptcl,
            cacheable_blend_spill,
        ) {
            (Some(info_bin_data), Some(tile), Some(segments), Some(ptcl), Some(blend_spill)) => {
                let pipeline_cache = crate::gpu_resources::CachedPipeline {
                    info_bin_data,
                    tile,
                    segments,
                    ptcl,
                    blend_spill,
                    reduced: cacheable_reduced,
                    reduced2: cacheable_reduced2,
                    reduced_scan: cacheable_reduced_scan,
                    tagmonoid: cacheable_tagmonoid,
                    path_bbox: cacheable_path_bbox,
                    lines: cacheable_lines,
                    draw_reduced: cacheable_draw_reduced,
                    draw_monoid: cacheable_draw_monoid,
                    clip_inp: cacheable_clip_inp,
                    clip_el: cacheable_clip_el,
                    clip_bic: cacheable_clip_bic,
                    clip_bbox: cacheable_clip_bbox,
                    draw_bbox: cacheable_draw_bbox,
                    bin_header: cacheable_bin_header,
                    path: cacheable_path,
                    seg_counts: cacheable_seg_counts,
                    buffer_sizes,
                };
                // Install into the first empty slot; defer if both are still aging.
                if let Some(i) = find_empty_cache_slot(&self.persistent.cached_pipeline) {
                    self.persistent.cached_pipeline[i] = Some(pipeline_cache);
                    outcome.cached_pipeline_slot = Some(i);
                    log::debug!("[PIPE-CACHE] schedule: cached slot={i}");
                } else {
                    self.defer_cached_pipeline_owned_buffers(pipeline_cache);
                    log::debug!("[PIPE-CACHE] schedule: all slots full — deferred current frame");
                }
            }
            (info_bin_data, tile, segments, ptcl, blend_spill) => {
                // Partial — couldn't cache all fields; defer any Owned ones to pool.
                if let Some(b) = info_bin_data {
                    self.deferred_owned_buffers
                        .push((b, "ekrano.info_bin_data_buf"));
                }
                if let Some(b) = tile {
                    self.deferred_owned_buffers.push((b, "ekrano.tile_buf"));
                }
                if let Some(b) = segments {
                    self.deferred_owned_buffers.push((b, "ekrano.segments_buf"));
                }
                if let Some(b) = ptcl {
                    self.deferred_owned_buffers.push((b, "ekrano.ptcl_buf"));
                }
                if let Some(b) = blend_spill {
                    self.deferred_owned_buffers.push((b, "ekrano.blend_spill"));
                }
                // Defer CoarseOnly owned buffers if present (shouldn't normally happen in this branch).
                macro_rules! defer_coarse {
                    ($field:expr, $name:expr) => {
                        if let Some(b) = $field {
                            self.deferred_owned_buffers.push((b, $name));
                        }
                    };
                }
                defer_coarse!(cacheable_reduced, "ekrano.reduced_buf");
                defer_coarse!(cacheable_reduced2, "ekrano.reduced2_buf");
                defer_coarse!(cacheable_reduced_scan, "ekrano.reduced_scan_buf");
                defer_coarse!(cacheable_tagmonoid, "ekrano.tagmonoid_buf");
                defer_coarse!(cacheable_path_bbox, "ekrano.path_bbox_buf");
                defer_coarse!(cacheable_lines, "ekrano.lines_buf");
                defer_coarse!(cacheable_draw_reduced, "ekrano.draw_reduced_buf");
                defer_coarse!(cacheable_draw_monoid, "ekrano.draw_monoid_buf");
                defer_coarse!(cacheable_clip_inp, "ekrano.clip_inp_buf");
                defer_coarse!(cacheable_clip_el, "ekrano.clip_el_buf");
                defer_coarse!(cacheable_clip_bic, "ekrano.clip_bic_buf");
                defer_coarse!(cacheable_clip_bbox, "ekrano.clip_bbox_buf");
                defer_coarse!(cacheable_draw_bbox, "ekrano.draw_bbox_buf");
                defer_coarse!(cacheable_bin_header, "ekrano.bin_header_buf");
                defer_coarse!(cacheable_path, "ekrano.path_buf");
                defer_coarse!(cacheable_seg_counts, "ekrano.seg_counts_buf");
            }
        }
        if let Some(i) = find_empty_cache_slot(&self.persistent.cached_render_targets) {
            self.persistent.cached_render_targets[i] = Some((out_image, filter_layers));
            outcome.cached_render_targets_slot = Some(i);
            log::debug!("[RT-CACHE] schedule: cached slot={i}");
        } else {
            self.defer_texture(out_image);
            for l in filter_layers {
                self.defer_texture(l);
            }
            log::debug!("[RT-CACHE] schedule: all slots full — deferred current frame RTs");
        }
        outcome
    }

    #[cfg_attr(
        not(feature = "debug_layers"),
        allow(dead_code, reason = "debug_layers only uses FrameRecorder::upload")
    )]
    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> Buffer {
        match record_upload_bytes_owned(
            self.device,
            self.context,
            self.graph,
            self.persistent,
            name,
            1,
            data.into(),
        )
        .expect("upload failed")
        {
            GpuBuf::Owned(b) => b,
            _ => panic!("upload must produce owned buffer"),
        }
    }

    pub fn upload_strided(
        &mut self,
        name: &'static str,
        element_stride: u32,
        data: impl Into<Vec<u8>>,
    ) -> Buffer {
        match record_upload_bytes_owned(
            self.device,
            self.context,
            self.graph,
            self.persistent,
            name,
            element_stride,
            data.into(),
        )
        .expect("upload_strided failed")
        {
            GpuBuf::Owned(b) => b,
            _ => panic!("upload_strided must produce owned buffer"),
        }
    }

    pub fn upload_typed<T: bytemuck::Pod>(&mut self, name: &'static str, data: &T) -> Buffer {
        // Small struct: borrow slice directly to avoid an intermediate Vec allocation.
        use crate::gpu_resources::record_upload_bytes;
        match record_upload_bytes(
            self.device,
            self.context,
            self.graph,
            self.persistent,
            name,
            size_of::<T>() as u32,
            bytemuck::bytes_of(data),
        )
        .expect("upload_typed failed")
        {
            GpuBuf::Owned(b) => b,
            _ => panic!("upload_typed must produce owned buffer"),
        }
    }

    pub fn dispatch(
        &mut self,
        shader: ShaderId,
        wg_size: (u32, u32, u32),
        bindings: &[GpuBinding<'_>],
    ) {
        self.dispatch_inner(shader, wg_size, bindings, &[]);
    }

    pub fn dispatch_with_push_tail(
        &mut self,
        shader: ShaderId,
        wg_size: (u32, u32, u32),
        bindings: &[GpuBinding<'_>],
        push_tail: &[u32],
    ) {
        self.dispatch_inner(shader, wg_size, bindings, push_tail);
    }

    fn dispatch_inner(
        &mut self,
        shader_id: ShaderId,
        (x, y, z): (u32, u32, u32),
        bindings: &[GpuBinding<'_>],
        push_tail: &[u32],
    ) {
        if x == 0 || y == 0 || z == 0 {
            log::warn!(
                "Skipping Dispatch for shader {} with zero grid dimension ({x}, {y}, {z}); \
                 this may indicate a bug in the caller",
                shader_id.0
            );
            return;
        }
        let bind_types = &self.shaders[shader_id.0].bindings;
        // Fill the scratch vec in-place (reusing its allocation), then swap it out
        // so the graph node takes ownership of the collected indices. Replace with a
        // fresh fixed-capacity scratch for the next dispatch in this frame.
        collect_bindless_indices_into(
            &mut self.indices_scratch,
            bindings,
            bind_types,
            MAX_BINDLESS_SLOTS,
        )
        .expect("collect_bindless_indices_into failed in dispatch");
        let indices = mem::replace(
            &mut self.indices_scratch,
            Vec::with_capacity(MAX_BINDLESS_SLOTS),
        );

        let mut node = self
            .graph
            .node("dispatch", &self.shaders[shader_id.0].pipeline);
        node = bind_graph_direct(node, bindings, bind_types);
        if !indices.is_empty() || !push_tail.is_empty() {
            node = node.bind_resources_raw_with_user(indices, push_tail);
        }
        node.dispatch(x, y, z);
    }

    pub fn dispatch_indirect(
        &mut self,
        shader: ShaderId,
        indirect_buf: &Buffer,
        offset: u64,
        bindings: &[GpuBinding<'_>],
    ) {
        self.dispatch_indirect_inner(shader, indirect_buf, offset, bindings);
    }

    fn dispatch_indirect_inner(
        &mut self,
        shader_id: ShaderId,
        indirect_buf: &Buffer,
        offset: u64,
        bindings: &[GpuBinding<'_>],
    ) {
        let bind_types = &self.shaders[shader_id.0].bindings;
        collect_bindless_indices_into(
            &mut self.indices_scratch,
            bindings,
            bind_types,
            MAX_BINDLESS_SLOTS,
        )
        .expect("collect_bindless_indices_into failed in dispatch_indirect");
        let indices = mem::replace(
            &mut self.indices_scratch,
            Vec::with_capacity(MAX_BINDLESS_SLOTS),
        );

        let mut node = self
            .graph
            .node("dispatch_indirect", &self.shaders[shader_id.0].pipeline);
        node = bind_graph_direct(node, bindings, bind_types);
        node = node.bind_buffer(indirect_buf, NodeAccess::Read);
        if !indices.is_empty() {
            node = node.bind_resources_raw(indices);
        }
        node.dispatch_indirect(indirect_buf, offset);
    }

    /// Stub for debug-layer draw commands (not yet implemented in Goldy).
    #[cfg(feature = "debug_layers")]
    pub fn draw(&mut self, params: crate::resource_proxy::DrawParams) {
        if let Some(vb) = params.vertex_buffer {
            self.defer_owned_buffer(vb, "ekrano.debug.vertex_buffer");
        }
        for b in params.resources {
            self.defer_owned_buffer(b, "ekrano.debug.resource");
        }
        if let Some(tex) = params.target {
            self.defer_texture(tex);
        }
    }

    #[cfg_attr(not(feature = "debug_layers"), allow(dead_code))]
    pub(crate) fn defer_owned_buffer(&mut self, buf: Buffer, name: &'static str) {
        self.deferred_owned_buffers.push((buf, name));
    }

    /// Finish dispatch: flush the final graph and register a frame slot with
    /// the orchestrator.
    ///
    /// Returns the submit timeline and an optional surface frame awaiting present.
    ///
    /// Surface paths call [`goldy::Frame::submit_frame`] before returning so the
    /// timeline is valid for cache stamping before [`goldy::Frame::present`].
    fn finish(mut self) -> Result<FrameFinishOutcome> {
        self.finished = true;

        self.persistent.deferred_owned_cap_hint = self.deferred_owned_buffers.capacity();
        self.persistent.deferred_pool_cap_hint = self.deferred_pool_views.capacity();
        self.persistent.deferred_textures_cap_hint = self.deferred_textures.capacity();

        let deferred_pool_views = mem::take(&mut self.deferred_pool_views);
        let deferred_textures = mem::take(&mut self.deferred_textures);
        let bump_readback = self.bump_buf_for_readback.take();
        let recyclable_owned = mem::take(&mut self.deferred_owned_buffers);

        let frame_handle = self.frame_handle;
        if let Some(surface) = self.surface {
            let mut frame = if let Some(frame) = self.preacquired_frame.take() {
                self.frame_pipeline
                    .end_frame_for_acquired_surface(
                        frame_handle,
                        self.graph,
                        surface,
                        frame,
                        (),
                    )
                    .map_err(|e| Error::Shader(e.to_string()))?
            } else {
                self.frame_pipeline
                    .end_frame_for_surface(frame_handle, self.graph, surface, ())
                    .map_err(|e| Error::Shader(e.to_string()))?
            };
            let submit_tv = frame
                .submit_frame()
                .map_err(|e| Error::Shader(e.to_string()))?;
            Ok(FrameFinishOutcome {
                timeline: Some(submit_tv),
                surface_frame: Some(frame),
                bump_readback,
                deferred_pool_views,
                deferred_textures,
                recyclable_owned,
            })
        } else {
            let tv = self
                .frame_pipeline
                .end_frame_standalone(frame_handle, self.graph, self.last_timeline, ())
                .map_err(|e| Error::Shader(e.to_string()))?;
            Ok(FrameFinishOutcome {
                timeline: Some(tv),
                surface_frame: None,
                bump_readback,
                deferred_pool_views,
                deferred_textures,
                recyclable_owned,
            })
        }
    }
}

impl Drop for FrameRecorder<'_> {
    fn drop(&mut self) {
        if !self.finished {
            // The recorder is being dropped without finish() or abort() — this happens when
            // an early `?` propagates an error out of run_frame before finish() is reached.
            // Abort the open frame so the orchestrator is not permanently stuck.
            self.frame_pipeline.abort_frame(self.frame_handle);
        }
    }
}

fn bind_graph_direct<'a>(
    mut node: NodeBuilder<'a>,
    bindings: &[GpuBinding<'a>],
    bind_types: &[BindType],
) -> NodeBuilder<'a> {
    for (i, binding) in bindings.iter().enumerate() {
        let access = bind_types
            .get(i)
            .copied()
            .map(bind_type_to_node_access)
            .unwrap_or_else(|| {
                log::warn!(
                    "bind_types list is shorter than bindings (index {i}); \
                     defaulting to ReadWrite access — check that shader bindings \
                     and BindType list are in sync",
                );
                NodeAccess::ReadWrite
            });
        node = match binding {
            GpuBinding::Buf(b) => node.bind_buffer(b, access),
            GpuBinding::Tex(t) => node.bind_texture(t, access),
            // Samplers and persistent (pre-initialized) buffers are stateless —
            // their slot index flows through push-constants but they need no
            // resource-barrier tracking in the task graph.  Persistent buffers
            // are guaranteed to be GPU-readable from prior frames; no barriers
            // are required for reads within this frame.
            GpuBinding::Sampler(_) | GpuBinding::PersistentBuf(_) => node,
        };
    }
    node
}

// -----------------------------------------------------------------------
// GoldyRenderer
// -----------------------------------------------------------------------

impl GoldyRenderer {
    /// Create a new renderer for the given device.
    ///
    /// Installs a [`TrackingVramAllocator`](goldy::vram_allocator::TrackingVramAllocator) with a 512 MiB budget on the device
    /// to prevent runaway GPU memory growth under heavy pipelining.
    pub fn new(device: &Device) -> Result<Self> {
        use goldy::vram_allocator::{DefaultVramAllocator, TrackingVramAllocator};

        let _tz = goldy::tracy_zone!("ekrano.GoldyRenderer::new");

        let budget = 512 * 1024 * 1024; // 512 MiB
        let tracking =
            TrackingVramAllocator::with_budget(Arc::new(DefaultVramAllocator::new()), budget);
        let tracked_device = device.with_vram_allocator(Arc::new(tracking));

        let unbudgeted_context = device
            .create_context()
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let context = tracked_device
            .create_context()
            .map_err(|e| Error::Gpu(e.to_string()))?;
        let frame_pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.GoldyRenderer::new.frame_orchestrator");
            FrameOrchestrator::new(&unbudgeted_context, FRAME_PIPELINE_DEPTH)
        };
        let mut renderer = Self {
            device: tracked_device.clone(),
            unbudgeted_context,
            context,
            shaders: FullShaders::empty(),
            resolver: Resolver::new(),
            engine_shaders: Vec::new(),
            persistent: PersistentState {
                pool: ResourcePool::default(),
                tex_pool: TexturePool::default(),
                last_drained_bump: None,
                pending_bump_readbacks: VecDeque::new(),
                linear_clamp_sampler: None,
                nearest_clamp_sampler: None,
                cached_render_targets: std::array::from_fn(|_| None),
                cached_rt_timelines: [0; RESOURCE_CACHE_SLOTS],
                cached_pipeline: std::array::from_fn(|_| None),
                cached_pipeline_timelines: [0; RESOURCE_CACHE_SLOTS],
                rt_slot_swapchain_image: [None; RESOURCE_CACHE_SLOTS],
                deferred_owned_cap_hint: 0,
                deferred_pool_cap_hint: 0,
                deferred_textures_cap_hint: 0,
                stable_mask_lut_msaa8: None,
                stable_mask_lut_msaa16: None,
                cached_wg_counts: None,
                cached_config_uniform: None,
                cached_filter_uniforms: Vec::new(),
                pending_texture_returns: Arc::new(Mutex::new(Vec::new())),
                pending_owned_returns: Arc::new(Mutex::new(Vec::new())),
            },
            frame_pipeline,
            persistent_bump: None,
            cleanup_frame_counter: 0,
            last_scene_fingerprint: None,
            last_packed_len: None,
            graph: TaskGraph::new(),
        };
        let shaders = {
            let _tz = goldy::tracy_zone!("ekrano.GoldyRenderer::new.compile_shaders");
            shaders::goldy_full_shaders(&tracked_device, &mut renderer)?
        };
        renderer.shaders = shaders;
        // Wire the pool's self-replenishment from the renderer's pending_owned_returns.
        let pending_returns = renderer.persistent.pending_owned_returns.clone();
        renderer
            .persistent
            .pool
            .set_pending_returns(pending_returns);
        {
            let _tz = goldy::tracy_zone!("ekrano.GoldyRenderer::new.release_compiler");
            tracked_device.release_idle_shader_compiler();
        }
        Ok(renderer)
    }

    // =======================================================================
    // Internal helpers — pool sizing & bump persistence
    // =======================================================================

    /// Update persistent bump estimates with a running component-wise max.
    fn update_persistent_bump(&mut self, bump: &BumpAllocators) {
        let p = self.persistent_bump.get_or_insert(BumpAllocators {
            failed: 0,
            binning: 0,
            ptcl: 0,
            tile: 0,
            seg_counts: 0,
            segments: 0,
            blend: 0,
            lines: 0,
        });
        p.binning = p.binning.max(bump.binning);
        p.ptcl = p.ptcl.max(bump.ptcl);
        p.tile = p.tile.max(bump.tile);
        p.seg_counts = p.seg_counts.max(bump.seg_counts);
        p.segments = p.segments.max(bump.segments);
        p.blend = p.blend.max(bump.blend);
        p.lines = p.lines.max(bump.lines);
    }

    // =======================================================================
    // Public API
    // =======================================================================

    /// Returns a clone of the renderer's submission [`goldy::Context`].
    ///
    /// Pass this context to [`goldy::Surface::new_with_config`] so that the
    /// surface submits GPU work through the **same** timeline semaphore as the
    /// renderer. This keeps `gpu_progress()` and the poller's `BoundaryCrossed`
    /// signals on one consistent clock, enabling correct RT-cache retirement and
    /// resource reclamation without a device-global fallback.
    pub fn submission_context(&self) -> Context {
        self.unbudgeted_context.clone()
    }

    /// Acknowledge a swapchain frame after [`goldy::Frame::present`].
    ///
    /// Fills in the timeline on the most recent `FrameCleanup` entry (the one
    /// pushed by `FrameRecorder::finish` for the surface path where the
    /// timeline isn't known until after present) and informs the transient
    /// allocator so it can retire this frame's regions with the correct epoch.
    pub fn note_frame_presented(&mut self, _device: &Device, tv: TimelineValue) {
        self.frame_pipeline.note_presented(tv);
    }

    /// Drain goldy signals and reclaim GPU resources tied to completed frames.
    ///
    /// Runs automatically at the start of [`Self::submit_prepared`], but can be
    /// called explicitly for fine-grained control between submit and present.
    pub fn poll_and_reclaim(&mut self, _device: &Device) {
        for signal in self.unbudgeted_context.poll_signals_and_service() {
            match signal {
                Signal::BoundaryCrossed { epoch } => {
                    self.persistent.drain_pending_returns();
                    self.frame_pipeline.note_presented(epoch);
                }
                Signal::Oversubscribed { .. } => {
                    if let Some(oldest) = self.unbudgeted_context.peek_oldest_in_flight()
                        && self.unbudgeted_context.wait_until(oldest).is_err()
                    {
                        break;
                    }
                    self.unbudgeted_context.flush_deferred_deletions();
                    self.context.flush_deferred_deletions();
                    self.persistent.drain_pending_returns();
                }
                Signal::SwapchainReturned { image_index } => {
                    self.persistent
                        .mark_rt_slot_returned(&self.unbudgeted_context, image_index);
                }
                Signal::SwapchainAcquired { .. } => {}
            }
        }
    }

    /// **Pipelined:** drains the *previous* frame's GPU work at the start,
    /// then submits the current frame and returns without waiting.  The bump
    /// allocator check uses the previous frame's data — if it overflowed,
    /// buffers are grown for the current frame.  This gives one frame of
    /// CPU/GPU overlap and eliminates the per-frame synchronization stall
    /// that previously bottlenecked throughput.
    ///
    /// Returns [`FrameStats`] on success. Check [`FrameStats::bump_retries`] to detect
    /// scenes that required buffer reallocation (e.g. to print a warning to stdout).
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        scene: &Scene,
        texture: &Texture,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        self.poll_and_reclaim(device);
        self.run_frame(device, scene, params, Some(texture), None)
    }

    /// Render a scene directly to a swapchain [`Surface`](goldy::Surface).
    ///
    /// Internally records the full graph (coarse + fine) with the swapchain as
    /// a late-bound output, then hands the graph to [`goldy::Surface::submit_graph`]
    /// which auto-partitions it, submits early work, acquires the swapchain
    /// image, and presents.  The caller does **not** need to call `acquire`,
    /// `present`, or `note_frame_presented` — everything is handled here.
    ///
    /// For lower latency, call [`Self::prepare`] while the previous frame is
    /// presenting, then [`Self::submit_to_surface`] once the scene is ready.
    pub fn render_to_surface(
        &mut self,
        device: &Device,
        scene: &Scene,
        surface: &goldy::Surface,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        let _tz = goldy::tracy_zone!("ekrano.render_to_surface");
        let prepared = self.prepare(scene, params)?;
        self.submit_to_surface(device, prepared, surface)
    }

    /// Phase 1: resolve scene encoding to CPU buffers (no GPU / backend access).
    ///
    /// Safe to call while the previous frame is still on the GPU. Overlap this
    /// with OS event-loop overhead between frames, then pass the result to
    /// [`Self::submit_to_surface`] or [`Self::submit_prepared`].
    ///
    /// Only one [`PreparedFrame`] may exist at a time: it holds the renderer's
    /// [`Resolver`] until consumed by submit.
    pub fn prepare(&mut self, scene: &Scene, params: &RenderParams) -> Result<PreparedFrame> {
        let _tz = goldy::tracy_zone!("ekrano.prepare");
        let encoding = scene.encoding();
        let mut params = RenderParams {
            base_color: params.base_color,
            width: params.width,
            height: params.height,
            antialiasing_method: params.antialiasing_method,
            robust: params.robust,
        };
        if let Some(robust) = env_robust_override() {
            params.robust = robust;
        }

        let mut resolver = mem::take(&mut self.resolver);
        let mut packed = vec![];
        let (layout, ramps, images) = {
            let _rz = goldy::tracy_zone!("ekrano.resolve");
            resolver.resolve(encoding, &mut packed)
        };

        let scene_fingerprint: u64 = 0;
        self.last_packed_len = Some(packed.len());
        let base_config =
            RenderConfig::new(&layout, params.width, params.height, &params.base_color);
        let config = if let Some(ref persistent) = self.persistent_bump {
            base_config.with_bump_estimates(persistent)
        } else {
            base_config
        };
        let pool_size = {
            let _tz = goldy::tracy_zone!("ekrano.pool_size");
            BufferPool::padded_size(&config.buffer_sizes.pool_allocs())
                .saturating_add(POOL_SIZE_SLACK)
        };

        Ok(PreparedFrame {
            packed,
            layout,
            ramps_data: ramps.data.to_vec(),
            ramps_width: ramps.width,
            ramps_height: ramps.height,
            images_width: images.width,
            images_height: images.height,
            image_entries: images.images.to_vec(),
            config,
            pool_size,
            params,
            scene_fingerprint,
            resolver,
            coverage_mask: encoding.coverage_mask.clone(),
            layer_filter_effects: encoding.layer_filter_effects.clone(),
        })
    }

    /// Phase 2: record GPU work, present, and return frame stats.
    ///
    /// Must be called after [`Self::prepare`]. Convenience wrapper around
    /// [`Self::submit_prepared`] that also presents the returned frame.
    pub fn submit_to_surface(
        &mut self,
        device: &Device,
        prepared: PreparedFrame,
        surface: &goldy::Surface,
    ) -> Result<FrameStats> {
        let _tz = goldy::tracy_zone!("ekrano.submit_to_surface");
        self.poll_and_reclaim(device);
        let (stats, surface_frame) =
            self.run_frame_from_prepared(device, prepared, None, Some(surface))?;
        if let Some((frame, tv)) = surface_frame {
            frame.present().map_err(|e| Error::Shader(e.to_string()))?;
            self.note_frame_presented(device, tv);
        }
        Ok(stats)
    }

    /// Submit prepared CPU work to the GPU without presenting.
    ///
    /// Returns frame stats and the goldy [`goldy::Frame`], which the caller must
    /// [`present`](goldy::Frame::present) when ready. Frame retirement is driven
    /// by [`Self::poll_and_reclaim`] (`BoundaryCrossed` signals) and the post-submit
    /// `flush_deferred_deletions` in `Self::run_frame`. Internally drains signals
    /// and reclaims resources before acquire + encode + submit.
    pub fn submit_prepared(
        &mut self,
        device: &Device,
        prepared: PreparedFrame,
        surface: &goldy::Surface,
    ) -> Result<(FrameStats, goldy::Frame)> {
        let _tz = goldy::tracy_zone!("ekrano.submit_prepared");
        self.poll_and_reclaim(device);
        let (stats, surface_frame) =
            self.run_frame_from_prepared(device, prepared, None, Some(surface))?;
        let (frame, _tv) = surface_frame.ok_or_else(|| Error::Shader("no surface frame".into()))?;
        Ok((stats, frame))
    }

    /// Query frame-scheduling state for diagnostics or test assertions.
    pub fn allocator_stats(&self) -> Option<AllocatorStats> {
        Some(AllocatorStats {
            capacity: 0,
            used: 0,
            cleanup_ring_depth: self.frame_pipeline.pending_frames(),
        })
    }

    /// Query the resource pool's current state for diagnostics or test assertions.
    pub fn resource_pool_stats(&self) -> ResourcePoolStats {
        ResourcePoolStats {
            total_pooled_buffers: self.persistent.pool.total_pooled_buffers(),
            distinct_keys: self.persistent.pool.distinct_keys(),
        }
    }

    /// `true` if either submission context still holds unreclaimed deferred payloads.
    pub fn has_deferred_payloads(&self) -> bool {
        self.unbudgeted_context.has_deferred_payloads() || self.context.has_deferred_payloads()
    }

    /// Pull-side reclamation on both submission contexts (unbudgeted caller device + tracked device).
    pub fn flush_deferred_deletions(&self) {
        self.unbudgeted_context.flush_deferred_deletions();
        self.context.flush_deferred_deletions();
    }

    /// Query the render context's placement heap state for diagnostics / tests.
    ///
    /// Delegates to [`Context::placement_heap_stats`](goldy::Context::placement_heap_stats).
    /// Returns `None` if no transient-buffer graphs have been submitted yet.
    pub fn placement_heap_stats(&self) -> Option<goldy::placement_heap::PlacementHeapStats> {
        self.context.placement_heap_stats()
    }

    /// Render a scene and return the pixel data as RGBA bytes.
    ///
    /// Unlike [`render_to_texture`](Self::render_to_texture), this path is
    /// **synchronous**: it waits for GPU completion and retries on bump
    /// overflow to guarantee correct output for screenshots / headless
    /// rendering.
    pub fn render_to_buffer(
        &mut self,
        device: &Device,
        scene: &Scene,
        params: &RenderParams,
    ) -> Result<Vec<u8>> {
        let width = params.width;
        let height = params.height;
        let texture = device
            .alloc_texture(
                width,
                height,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .map_err(|e| Error::Gpu(e.to_string()))?;

        for _attempt in 0..=MAX_BUMP_RETRIES {
            self.render_to_texture(device, scene, &texture, params)?;
            self.frame_pipeline
                .drain_all(|_, _| Ok::<(), Error>(()))
                .map_err(|e| Error::Shader(e.to_string()))?;
            self.persistent
                .drain_ready_bump_readbacks(device, &self.unbudgeted_context)?;
            self.unbudgeted_context.flush_deferred_deletions();
            self.context.flush_deferred_deletions();

            match self.persistent.last_drained_bump() {
                Some(bump) if bump.failed != 0 => {
                    log::info!(
                        "Bump overflow in render_to_buffer (0x{:x}), retrying",
                        bump.failed,
                    );
                }
                _ => break,
            }
        }

        if let Some(bump) = self.persistent.last_drained_bump()
            && bump.failed != 0
        {
            log::warn!(
                "render_to_buffer: bump overflow (0x{:x}) persisted after {} retries; \
                 output may be incomplete",
                bump.failed,
                MAX_BUMP_RETRIES
            );
        }

        let mut output = vec![0_u8; texture.byte_size()];
        texture
            .read_to_cpu(&mut output)
            .map_err(|e| Error::Readback(e.to_string()))?;
        Ok(output)
    }

    // =======================================================================
    // Frame execution (private)
    // =======================================================================

    /// Shared implementation for [`render_to_texture`](Self::render_to_texture).
    ///
    /// Creates a [`FrameRecorder`], runs the full coarse+fine pipeline into it,
    /// then flushes the resulting [`TaskGraph`].
    fn run_frame(
        &mut self,
        device: &Device,
        scene: &Scene,
        params: &RenderParams,
        output_texture: Option<&Texture>,
        surface: Option<&goldy::Surface>,
    ) -> Result<FrameStats> {
        let prepared = self.prepare(scene, params)?;
        let (stats, surface_frame) =
            self.run_frame_from_prepared(device, prepared, output_texture, surface)?;
        if let Some((frame, tv)) = surface_frame {
            frame.present().map_err(|e| Error::Shader(e.to_string()))?;
            self.note_frame_presented(device, tv);
        }
        Ok(stats)
    }

    /// GPU submission path shared by [`Self::submit_to_surface`], [`Self::submit_prepared`],
    /// and [`Self::run_frame`].
    fn run_frame_from_prepared(
        &mut self,
        device: &Device,
        prepared: PreparedFrame,
        output_texture: Option<&Texture>,
        surface: Option<&goldy::Surface>,
    ) -> Result<(FrameStats, Option<(goldy::Frame, TimelineValue)>)> {
        let _tz = goldy::tracy_zone!("ekrano.run_frame");
        use std::time::Instant;
        let frame_start = Instant::now();

        let packed = prepared.packed;
        let layout = prepared.layout;
        let ramps_data = prepared.ramps_data;
        let ramps_width = prepared.ramps_width;
        let ramps_height = prepared.ramps_height;
        let images_width = prepared.images_width;
        let images_height = prepared.images_height;
        let image_entries = prepared.image_entries;
        let mut config = prepared.config;
        let _pool_size = prepared.pool_size;
        let params = prepared.params;
        let scene_fingerprint = prepared.scene_fingerprint;
        let resolver = prepared.resolver;
        let coverage_mask = prepared.coverage_mask;
        let layer_filter_effects = prepared.layer_filter_effects;
        let ramps = Ramps {
            data: &ramps_data,
            width: ramps_width,
            height: ramps_height,
        };
        let images = Images {
            width: images_width,
            height: images_height,
            images: &image_entries,
        };
        let mut stats = FrameStats::default();
        let t_resolve = frame_start.elapsed();

        // --- Reclaim completed frames & open recording bracket ---
        // Drain pool returns from prior-frame token drops (cheap; no GPU sync).
        // Drain any bump readbacks whose GPU timeline has already passed.
        // BoundaryCrossed signals are serviced by poll_and_reclaim at frame entry.
        self.persistent.drain_pending_returns();
        self.persistent
            .drain_ready_bump_readbacks(device, &self.unbudgeted_context)?;

        let out_image_format = surface
            .map(|s| s.format())
            .unwrap_or(TextureFormat::Rgba8Unorm);
        self.persistent.purge_render_target_cache_if_mismatch(
            &self.unbudgeted_context,
            params.width,
            params.height,
            out_image_format,
        );

        let t_drain_start = Instant::now();

        let _tz_begin = goldy::tracy_zone!("ekrano.begin_frame");
        let frame_handle = self
            .frame_pipeline
            .begin_frame(|_, _| Ok::<(), Error>(()))
            .map_err(|e| Error::Shader(e.to_string()))?;
        // After begin_frame the GPU has completed the previous frame (depth=1 wait),
        // so any queued bump readback is now guaranteed ready.
        self.persistent
            .drain_ready_bump_readbacks(device, &self.unbudgeted_context)?;
        // Rate-limit housekeeping to avoid per-frame cost in steady state.
        // With OwnedShared buffers and cached render targets, the ResourcePool
        // stays small and overflow heaps are rare; scanning every 64 frames is enough.
        self.cleanup_frame_counter = self.cleanup_frame_counter.wrapping_add(1);
        if self.cleanup_frame_counter.is_multiple_of(64) {
            self.persistent.pool.cap_pool_depth(12);
            self.device.compact_overflow_heaps();
        }
        let t_drain = t_drain_start.elapsed();

        let prev_bump = self.persistent.take_last_drained_bump();

        // Only trust bump readback when robust mode produced valid data.
        // A bump with all-zero counters (no failed flag, no usage) is likely stale
        // or from a frame where the buffer was never written.
        let prev_bump = prev_bump.filter(|b| {
            let any_nonzero = b.lines > 0
                || b.seg_counts > 0
                || b.segments > 0
                || b.tile > 0
                || b.binning > 0
                || b.ptcl > 0
                || b.blend > 0;
            if !any_nonzero && b.failed == 0 {
                log::debug!("Ignoring all-zero bump readback (likely stale)");
                return false;
            }
            true
        });

        if let Some(ref bump) = prev_bump {
            let frame_num = FRAME_COUNTER.load(Ordering::Relaxed);
            log::debug!(
                "[BUMP] frame={} lines={}, seg_counts={}, segments={}, tile={}, failed=0x{:x}",
                frame_num,
                bump.lines,
                bump.seg_counts,
                bump.segments,
                bump.tile,
                bump.failed,
            );
            // Update persistent bump estimates (running max across frames).
            self.update_persistent_bump(&sanitize_bump(bump));

            // On overflow, recompute config/pool_size with the actual overflow counters.
            // Rare in steady state; persistent_bump normally already covers the needed sizes.
            if bump.failed != 0 {
                stats.bump_retries += 1;
                log::info!(
                    "Previous frame bump overflow (0x{:x}), growing buffers",
                    bump.failed,
                );
                config =
                    RenderConfig::new(&layout, params.width, params.height, &params.base_color)
                        .with_bump_estimates(&sanitize_bump(bump));
            }
        }

        let preacquired_frame = if let Some(surface) = surface {
            let _tz = goldy::tracy_zone!("ekrano.surface.acquire_early");
            match surface.begin() {
                Ok(frame) => Some(frame),
                Err(e) => {
                    self.frame_pipeline.abort_frame(frame_handle);
                    return Err(Error::Shader(e.to_string()));
                }
            }
        } else {
            None
        };
        let acquired_image_index = preacquired_frame.as_ref().map(goldy::Frame::image_index);

        let t1 = Instant::now();
        // Lazily create persistent samplers on the first frame.
        if self.persistent.linear_clamp_sampler.is_none() {
            self.persistent.linear_clamp_sampler =
                Some(goldy::Sampler::linear(device).map_err(|e| Error::Gpu(e.to_string()))?);
        }
        if self.persistent.nearest_clamp_sampler.is_none() {
            self.persistent.nearest_clamp_sampler =
                Some(goldy::Sampler::nearest(device).map_err(|e| Error::Gpu(e.to_string()))?);
        }
        self.unbudgeted_context.flush_deferred_deletions();
        let t_pool = t1.elapsed();

        let t2 = Instant::now();
        let mut recorder = FrameRecorder::new(
            device,
            &self.unbudgeted_context,
            &mut self.graph,
            &mut self.frame_pipeline,
            frame_handle,
            &mut self.persistent,
            &self.engine_shaders,
            surface,
            preacquired_frame,
        );

        let ctx = recorder.context;
        let (graph, persistent) = recorder.graph_and_persistent();
        let swapchain_handle = if surface.is_some() {
            Some(graph.declare_swapchain_output())
        } else {
            None
        };
        let mut pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.prepare");
            let pipeline_result = crate::gpu_resources::PipelineResources::prepare(
                device,
                ctx,
                graph,
                persistent,
                coverage_mask.as_ref(),
                packed,
                ramps,
                images,
                &params,
                &config,
                out_image_format,
            );
            self.resolver = resolver;
            match pipeline_result {
                Ok(p) => p,
                Err(e) => {
                    drop(recorder);
                    return Err(e);
                }
            }
        };

        let mut render = Render::new();
        {
            let _tz = goldy::tracy_zone!("ekrano.coarse");
            render.run_coarse(
                &mut pipeline,
                &self.shaders,
                &params,
                params.robust,
                &config,
                &mut recorder,
            );
        }
        let t_coarse = t2.elapsed();

        let t3 = Instant::now();
        {
            let _tz = goldy::tracy_zone!("ekrano.fine");

            render.record_fine(
                &layer_filter_effects,
                &self.shaders,
                &pipeline,
                output_texture,
                &mut recorder,
            );
            #[cfg(feature = "debug_layers")]
            let _ = render.take_captured_buffers();
            crate::render::record_filter_effects(
                &layer_filter_effects,
                &self.shaders,
                &mut recorder,
                &pipeline,
                output_texture,
            );
            if let Some(handle) = swapchain_handle {
                if let Some(surface) = surface {
                    debug_assert_eq!(
                        pipeline.out_image.width(),
                        surface.width(),
                        "out_image width must match swapchain surface",
                    );
                    debug_assert_eq!(
                        pipeline.out_image.height(),
                        surface.height(),
                        "out_image height must match swapchain surface",
                    );
                    debug_assert_eq!(
                        pipeline.out_image.format(),
                        surface.format(),
                        "out_image format must match swapchain surface",
                    );
                }
                recorder
                    .graph
                    .copy_texture_to_swapchain(&pipeline.out_image, handle);
            }
        }
        let t_fine_record = t3.elapsed();

        let t4 = Instant::now();
        let cache_outcome = recorder.schedule_pipeline_cleanup(pipeline, params.robust);
        let FrameFinishOutcome {
            timeline: frame_tv,
            surface_frame,
            bump_readback,
            deferred_pool_views,
            deferred_textures,
            recyclable_owned,
        } = {
            let _tz = goldy::tracy_zone!("ekrano.finish");
            recorder.finish()?
        };

        if let Some(tv) = frame_tv {
            if let Some(i) = cache_outcome.cached_render_targets_slot {
                log::debug!(
                    "[RT-CACHE] stamp slot={i} timeline={tv} (prev={})",
                    self.persistent.cached_rt_timelines[i],
                );
                self.persistent.cached_rt_timelines[i] = tv;
                if let Some(idx) = acquired_image_index {
                    self.persistent.rt_slot_swapchain_image[i] = Some(idx);
                }
            }
            if let Some(i) = cache_outcome.cached_pipeline_slot {
                self.persistent.cached_pipeline_timelines[i] = tv;
            }
            if let Some(buf) = bump_readback {
                self.persistent.queue_bump_readback(tv, buf);
            }
            defer_frame_gpu_resources(
                &self.unbudgeted_context,
                &self.persistent,
                tv,
                deferred_pool_views,
                deferred_textures,
                recyclable_owned,
            );
        }

        {
            let _tz = goldy::tracy_zone!("ekrano.run_frame.post_submit");
            // Keep reclamation keyed to live GPU progress so Vulkan/DX12 are not gated
            // solely by the coarse background fence-poller signal cadence.
            //
            // ekrano runs a dual-device split: `device` (the caller's unbudgeted device)
            // owns the per-frame VRAM ring that `defer_frame_gpu_resources` just pushed
            // this frame's views/textures into above, while `self.device` is the budgeted
            // clone used for persistent/storage allocations. Both share the same backend
            // (and handle), so each call also drains the shared backend deletion queue;
            // the second drain is an idempotent no-op. We must flush the caller's device
            // here — it holds this frame's deferrals — otherwise they stay pinned until the
            // poller-driven `poll_and_reclaim` signal path catches up (the U5 regression).
            self.unbudgeted_context.flush_deferred_deletions();
            self.context.flush_deferred_deletions();
            let t_submit = t4.elapsed();

            let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
            let label = if surface.is_some() { "surface" } else { "" };

            let ring_depth = self.frame_pipeline.pending_frames();

            let (transient_views, transient_textures) = self.context.transient_cache_counts();
            let rt_slots = self
                .persistent
                .cached_render_targets
                .iter()
                .filter(|s| s.is_some())
                .count();
            let pipe_slots = self
                .persistent
                .cached_pipeline
                .iter()
                .filter(|s| s.is_some())
                .count();

            log::debug!(
                "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms ring={} rt_slots={rt_slots} pipe_slots={pipe_slots} tv={} tt={} {label}",
                frame_num,
                t_drain.as_secs_f64() * 1000.0,
                t_resolve.as_secs_f64() * 1000.0,
                t_pool.as_secs_f64() * 1000.0,
                t_coarse.as_secs_f64() * 1000.0,
                t_fine_record.as_secs_f64() * 1000.0,
                t_submit.as_secs_f64() * 1000.0,
                frame_start.elapsed().as_secs_f64() * 1000.0,
                ring_depth,
                transient_views,
                transient_textures,
            );

            // Update fingerprint after a successful frame.
            self.last_scene_fingerprint = Some(scene_fingerprint);
        }

        let surface_frame = match (surface_frame, frame_tv) {
            (Some(frame), Some(tv)) => Some((frame, tv)),
            _ => None,
        };
        Ok((stats, surface_frame))
    }

    // =======================================================================
    // Engine methods
    // =======================================================================

    /// Add a compute shader from Slang source.
    pub(crate) fn add_compute_shader(
        &mut self,
        device: &Device,
        label: &'static str,
        slang_source: &str,
        bindings: &[BindType],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<ShaderId> {
        self.add_compute_shader_with_options(
            device,
            label,
            slang_source,
            bindings,
            search_paths,
            defines,
            goldy::OptimizationLevel::Default,
        )
    }

    /// Add a compute shader with explicit optimization level.
    pub(crate) fn add_compute_shader_with_options(
        &mut self,
        device: &Device,
        _label: &'static str,
        slang_source: &str,
        bindings: &[BindType],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: goldy::OptimizationLevel,
    ) -> Result<ShaderId> {
        let shader_module = {
            let _tz = goldy::tracy_zone!("ekrano.add_shader.slang", _label);
            ShaderModule::from_slang_with_options(
                device,
                slang_source,
                search_paths,
                defines,
                optimization_level,
                &[],
            )
            .map_err(|e| Error::Shader(format!("{:#}", e)))?
        };
        let pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.add_shader.pipeline", _label);
            ComputePipeline::new(device, &shader_module)
                .map_err(|e| Error::Shader(format!("{:#}", e)))?
        };

        let id = ShaderId(self.engine_shaders.len());
        self.engine_shaders.push(GoldyShader {
            pipeline,
            bindings: bindings.to_vec(),
        });
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_resources::PipelineResources;
    use ekrano_encoding::{RenderConfig, Resolver};
    use goldy::Instance;

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
        let Ok(instance) = Instance::new() else {
            return;
        };
        let Ok(device) = instance
            .request_adapter(&goldy::RequestAdapterOptions::default())
            .and_then(|a| a.request_device(&goldy::DeviceDescriptor::default()))
        else {
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

        let mut persistent = PersistentState {
            pool: ResourcePool::default(),
            tex_pool: TexturePool::default(),
            last_drained_bump: None,
            pending_bump_readbacks: VecDeque::new(),
            linear_clamp_sampler: None,
            nearest_clamp_sampler: None,
            cached_render_targets: std::array::from_fn(|_| None),
            cached_rt_timelines: [0; RESOURCE_CACHE_SLOTS],
            cached_pipeline: std::array::from_fn(|_| None),
            cached_pipeline_timelines: [0; RESOURCE_CACHE_SLOTS],
            rt_slot_swapchain_image: [None; RESOURCE_CACHE_SLOTS],
            deferred_owned_cap_hint: 0,
            deferred_pool_cap_hint: 0,
            deferred_textures_cap_hint: 0,
            stable_mask_lut_msaa8: None,
            stable_mask_lut_msaa16: None,
            cached_wg_counts: None,
            cached_config_uniform: None,
            cached_filter_uniforms: Vec::new(),
            pending_texture_returns: Arc::new(Mutex::new(Vec::new())),
            pending_owned_returns: Arc::new(Mutex::new(Vec::new())),
        };
        {
            let pending = persistent.pending_owned_returns.clone();
            persistent.pool.set_pending_returns(pending);
        }

        for &expected_format in &[TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm] {
            let mut resolver = Resolver::new();
            let mut packed = Vec::new();
            let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
            let config =
                RenderConfig::new(&layout, params.width, params.height, &params.base_color);
            let mut graph = TaskGraph::new();

            let ctx = device.create_context().expect("context");
            let pipeline = PipelineResources::prepare(
                &device,
                &ctx,
                &mut graph,
                &mut persistent,
                encoding.coverage_mask.as_ref(),
                packed,
                ramps,
                images,
                &params,
                &config,
                expected_format,
            )
            .unwrap_or_else(|e| {
                panic!("PipelineResources::prepare({expected_format:?}) failed: {e}")
            });

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
