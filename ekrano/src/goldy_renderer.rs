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

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use goldy::types::{BufferFlags, TextureFormat};
use goldy::{
    BackendType, Buffer, BufferKind, ComputePipeline, Context, Device, Grant, RetainedPool, Texture,
    TimelineValue,
};

/// Ekrano uses a single-frame fire-and-forget model.
///
/// Stable pipeline parcels live in [`RetainedPool`] deeds reused across frames only while
/// depth stays at 1 (see [`StablePipelineBuffers`](crate::scheme_gpu_resources::StablePipelineBuffers)).
pub(crate) const FRAME_PIPELINE_DEPTH: usize = 1;

use crate::{Error, RenderParams, Result, resource_proxy::BindType};
use ekrano_encoding::{BumpAllocators, Layout, RenderConfig, Resolver};

pub(crate) const MAX_BUMP_RETRIES: usize = 2;

/// Per-frame render statistics returned by [`crate::GoldyRenderer::render_to_texture`].
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

/// Snapshot of retained-pool buffer accounting, useful for tests and diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct ResourcePoolStats {
    /// Committed bytes held in the retained pool.
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
pub(crate) fn maybe_log_gpu_memory(device: &Device) {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static LAST_LOG: Mutex<Option<Instant>> = Mutex::new(None);
    {
        let mut last = LAST_LOG.lock().unwrap_or_else(|e| e.into_inner());
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
                "[GPU-MEM] dxgi_local={} / budget={} dxgi_non_local={} tracked={}",
                fmt_mib(info.local_current_bytes),
                fmt_mib(info.local_budget_bytes),
                fmt_mib(info.non_local_current_bytes),
                fmt_mib(tracked),
            );
        }
        None => {
            log::info!(
                "[GPU-MEM] tracked={} (no DXGI video-memory query on this backend)",
                fmt_mib(tracked),
            );
        }
    }
}

// -----------------------------------------------------------------------
// Deferred per-frame work
// -----------------------------------------------------------------------
//
// `FrameOrchestrator` is scheduling-only (depth + timeline ring).

/// Which caches received new entries during pipeline cleanup.
#[derive(Debug, Default)]
pub(crate) struct CacheScheduleOutcome {
    /// Set when the scheme path stored its single persistent `out_image` into
    /// [`PersistentState::cached_scheme_rt`]; the post-submit step then stamps the
    /// frame timeline as a cache/reclamation marker (not a `begin_frame` GPU wait).
    pub(crate) scheme_rt_stored: bool,
}

/// Outcome of [`crate::scheme_renderer::SchemeRecorder::finish`]: orchestrator submit
/// result plus filter-scratch textures returned to the transient pool after submit.
pub(crate) struct FrameFinishOutcome {
    pub(crate) timeline: TimelineValue,
    pub(crate) deferred_textures: Vec<Texture>,
    /// Submission returned by [`goldy::Scheme::submit`].
    ///
    /// Held here so that `SchemeRenderer::run_frame_from_prepared` can build a
    /// [`PresentToken`] after `finish()` returns.
    pub(crate) scheme_submission: Option<goldy::Submission>,
}

/// Present token for swapchain scanout.
///
/// Produced by [`crate::GoldyRenderer::submit_to_swapchain`]. Hand to `TID_PRESENT` for async
/// scanout, or call [`Self::present`] synchronously (e.g. [`crate::scheme_renderer::SchemeRenderer::render_to_swapchain`]).
pub struct PresentToken {
    pub(crate) claim: goldy::Claim,
}

impl PresentToken {
    /// Perform scanout via [`goldy::Claim::consume`].
    pub fn present(self) -> Result<()> {
        self.claim.consume().map_err(|e| Error::Shader(e.to_string()))
    }
}

/// CPU-resolved scene data ready for GPU submission.
///
/// Produced by [`crate::GoldyRenderer::prepare`] (pure CPU — safe to call while the
/// previous frame is still on the GPU) and consumed by
/// [`crate::GoldyRenderer::submit_to_swapchain`].
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

/// Return filter-scratch textures to the context transient pool (epoch-gated).
///
/// On Metal, mismatched/resize-prone scratches are dropped instead of parked
/// (overflow-heap pin avoidance).
pub(crate) fn defer_frame_gpu_resources(ctx: &Context, persistent: &PersistentState, textures: Vec<Texture>) {
    if persistent.metal_heap_sensitive {
        drop(textures);
    } else {
        for tex in textures {
            ctx.return_transient_texture(tex);
        }
    }
}

pub(crate) struct GoldyShader {
    pub(crate) pipeline: ComputePipeline,
    pub(crate) bindings: Vec<BindType>,
    /// Registration name (e.g. `"fine_area"`) used for scheme node labels and Metal PSO labels.
    pub(crate) label: &'static str,
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
    /// Retained pool for the seven stable pipeline buffers (`resource-pool.md` §4).
    /// Valid only at [`FRAME_PIPELINE_DEPTH`] = 1; see [`StablePipelineBuffers`](crate::scheme_gpu_resources::StablePipelineBuffers).
    pub(crate) retained_pool: RetainedPool,
    /// Bump allocator counters from the most recently drained frame.
    /// `None` until the first GPU readback completes.
    last_drained_bump: Option<BumpAllocators>,
    /// Persistent linear-filter + clamp-to-edge sampler for hardware-filtered texture reads
    /// (gradient ramps, image atlas bilinear). Lazily created on first render.
    pub(crate) linear_clamp_sampler: Option<goldy::Sampler>,
    /// Persistent nearest-filter + clamp-to-edge sampler for `IMAGE_QUALITY_LOW` reads.
    pub(crate) nearest_clamp_sampler: Option<goldy::Sampler>,
    /// Scheme-path render-target reuse: optional persistent `out_image` + filter layers
    /// (retained deeds; take/store across frames).
    pub(crate) cached_scheme_rt: Option<(Option<Texture>, [Texture; 4], TimelineValue)>,
    /// Cached pipeline buffers from the previous frame. At depth=1 only one
    /// entry exists at a time: take-then-install within a single `run_frame`.
    pub(crate) cached_pipeline: Option<crate::scheme_gpu_resources::CachedPipeline>,
    /// Capacity hint for recorder deferred-texture scratch. Updated after each
    /// `finish()` call so that the next frame pre-allocates the right amount.
    pub(crate) deferred_textures_cap_hint: usize,
    /// Static MSAA8 mask LUT buffer (retained deed, init once).
    ///
    /// Keeping this persistent avoids a staging-belt `CopyBufferRegion` in the
    /// retained fine command list, which would reference a stale staging chunk on
    /// resubmission.  `None` until first needed; never freed after that.
    pub(crate) stable_mask_lut_msaa8: Option<Buffer>,
    /// Static MSAA16 mask LUT buffer (same rationale as `stable_mask_lut_msaa8`).
    pub(crate) stable_mask_lut_msaa16: Option<Buffer>,
    /// Cached GPU `ConfigUniform` buffer. Stable across frames once bump estimates
    /// converge; eliminates `WriteBuffer` from the dispatch graph at steady state.
    pub(crate) cached_config_uniform: Option<(ekrano_encoding::ConfigUniform, Buffer)>,
    /// Per-slot cached `FilterUniform` buffers, indexed by filter dispatch order.
    /// Retained deeds; stable for scenes with fixed filter effects (e.g. a static drop shadow).
    pub(crate) cached_filter_uniforms: Vec<Option<(ekrano_encoding::FilterUniform, Buffer)>>,
    /// Composite per-stage indirect `DispatchShape` buffer, retained across frames.
    /// Cache key is the `WorkgroupCountsGpu` that seeded the allocation.
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
    /// Present transaction recorded once on the worker when surface presentation is enabled.
    /// Claim resolves the present easement at the copy/present-partition timeline
    /// (when `out_image` finishes being read), not the later display-present timeline.
    pub(crate) cached_present_tx: Option<goldy::Transaction>,
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
    /// Persistent host buffer parcel for [`crate::scheme_renderer::SchemeRenderer::render_to_buffer`].
    pub(crate) readback_host_buf: Option<(Buffer, u64)>,
    /// Metal overflow texture heaps stay pinned if mismatched-size RTs are pooled across
    /// resize. When set, reclaim/purge drop and clear aggressively instead of deferred pooling.
    pub(crate) metal_heap_sensitive: bool,
}

impl PersistentState {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            retained_pool: RetainedPool::new(Arc::new(device.clone())),
            last_drained_bump: None,
            linear_clamp_sampler: None,
            nearest_clamp_sampler: None,
            cached_scheme_rt: None,
            cached_pipeline: None,
            deferred_textures_cap_hint: 0,
            stable_mask_lut_msaa8: None,
            stable_mask_lut_msaa16: None,
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
            cached_present_tx: None,
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
            readback_host_buf: None,
            metal_heap_sensitive: device.backend_type() == BackendType::Metal,
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
                self.retained_pool.release_buffer(ctx, old);
            }
            self.retained_pool
                .acquire_buffer(
                    staging_bytes,
                    BufferKind::Scattered,
                    None,
                    BufferFlags::CPU_READABLE,
                    None,
                )
                .map_err(|e| Error::Shader(e.to_string()))
        } else if let Some((buf, _)) = self.readback_host_buf.take() {
            Ok(buf)
        } else {
            self.retained_pool
                .acquire_buffer(
                    staging_bytes,
                    BufferKind::Scattered,
                    None,
                    BufferFlags::CPU_READABLE,
                    None,
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
        direct_present: bool,
    ) -> Option<(Option<Texture>, [Texture; 4])> {
        let (out, layers, tv) = self.cached_scheme_rt.take()?;
        let out_matches = match (&out, direct_present) {
            (None, true) => true,
            (Some(tex), false) => tex.width() == width && tex.height() == height && tex.format() == out_format,
            _ => false,
        };
        if out_matches {
            let _ = tv;
            return Some((out, layers));
        }
        log::warn!(
            "[RT-CACHE] scheme MISS (resize/mode) timeline={tv} direct_present={direct_present} \
             cached_out={} vs {width}x{height} fmt={out_format:?}",
            out.as_ref()
                .map(|t| format!("{}x{}", t.width(), t.height()))
                .unwrap_or_else(|| "none".into()),
        );
        if let Some(out_tex) = out {
            self.reclaim_scheme_render_targets(ctx, Some(out_tex), layers, tv);
        } else {
            self.reclaim_scheme_render_targets(ctx, None, layers, tv);
        }
        None
    }

    /// Retire scheme render targets that no longer match the requested size.
    ///
    /// On Metal, waits for `tv` if still in flight, then **drops** the textures —
    /// releasing them into the transient pool would keep mismatched-size heap
    /// allocations alive across resize churn and exhaust overflow texture heaps.
    ///
    /// On DX12/Vulkan, releases deeds into the retained → transient path (epoch-gated).
    fn reclaim_scheme_render_targets(
        &mut self,
        ctx: &Context,
        out: Option<Texture>,
        layers: [Texture; 4],
        tv: TimelineValue,
    ) {
        if self.metal_heap_sensitive {
            if tv != 0
                && ctx.gpu_progress() < tv
                && let Err(e) = ctx.wait_until(tv)
            {
                log::warn!("[RT-CACHE] wait_until({tv}) failed before Metal scheme RT drop: {e}");
            }
            if let Some(out) = out {
                drop(out);
            }
            for l in layers {
                drop(l);
            }
            return;
        }

        if let Some(out) = out {
            self.retained_pool.release_texture(ctx, out);
        }
        for l in layers {
            self.retained_pool.release_texture(ctx, l);
        }
    }

    pub(crate) fn store_scheme_render_targets(
        &mut self,
        out: Option<Texture>,
        layers: [Texture; 4],
        timeline: TimelineValue,
    ) {
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
    ///
    /// On Metal, also clears [`Self::cached_scheme_rt`] and the context transient
    /// texture bins so obsolete sizes cannot pin overflow heaps across continuous
    /// resize. DX12/Vulkan keep deferred pool reclamation.
    pub(crate) fn purge_render_target_cache_if_mismatch(
        &mut self,
        ctx: &Context,
        width: u32,
        height: u32,
        out_format: TextureFormat,
    ) -> bool {
        let scheme_mismatch = self.cached_scheme_rt.as_ref().is_some_and(|(out, layers, _)| {
            layers[0].width() != width
                || layers[0].height() != height
                || out
                    .as_ref()
                    .is_some_and(|tex| tex.format() != out_format || tex.width() != width || tex.height() != height)
        });
        if !scheme_mismatch {
            return false;
        }

        let progress = ctx.gpu_progress();
        if ctx.peek_oldest_in_flight().is_some()
            && self.metal_heap_sensitive
            && let Some((_, _, tv)) = &self.cached_scheme_rt
            && progress < *tv
            && let Err(e) = ctx.wait_until(*tv)
        {
            log::warn!("[RT-CACHE] wait_until({tv}) failed during resize purge: {e}");
        }

        if let Some((out, layers, tv)) = self.cached_scheme_rt.take() {
            if self.metal_heap_sensitive {
                drop(out);
                for l in layers {
                    drop(l);
                }
            } else {
                self.reclaim_scheme_render_targets(ctx, out, layers, tv);
            }
        }

        if self.metal_heap_sensitive {
            ctx.clear_transient_textures();
            ctx.flush_deferred_deletions();
        }

        true
    }
}

impl PersistentState {
    pub(crate) fn last_drained_bump(&self) -> Option<&BumpAllocators> {
        self.last_drained_bump.as_ref()
    }

    pub(crate) fn take_last_drained_bump(&mut self) -> Option<BumpAllocators> {
        self.last_drained_bump.take()
    }

    pub(crate) fn queue_bump_submission(&mut self, submission: goldy::Submission) {
        self.pending_bump_submission = Some(submission);
    }

    pub(crate) fn drain_ready_bump_readbacks(&mut self, ctx: &Context) -> Result<()> {
        let Some(submission) = self.pending_bump_submission.take() else {
            return Ok(());
        };
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
        Ok(())
    }

    /// Claim cached pipeline buffers for this frame.
    ///
    /// Pipeline buffers are fully GPU-overwritten each frame. At depth=1, [`goldy::FrameOrchestrator`]
    /// retirement in `begin_frame` provides cross-frame ordering — no `gpu_progress` gate here.
    pub(crate) fn take_cached_pipeline(&mut self) -> Option<crate::scheme_gpu_resources::CachedPipeline> {
        if let Some(c) = self.cached_pipeline.take() {
            log::debug!("[PIPE-CACHE] HIT");
            return Some(c);
        }
        None
    }
}

fn read_bump_bytes(persistent: &mut PersistentState, bytes: &[u8]) {
    let _parse = goldy::tracy_zone!("ekrano.bump_readback.parse");
    persistent.last_drained_bump = Some(bytemuck::pod_read_unaligned(bytes));
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use goldy::{Adapter, BackendType, Device, DeviceDescriptor, Instance, RequestAdapterOptions};
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
        let persistent = PersistentState::new(&gpu);
        Some((gpu, persistent))
    }

    #[test]
    fn scheme_rt_cache_reuses_none_out_image_in_direct_present_mode() {
        let device = goldy::test_support::mock_device();
        let mut p = PersistentState::new(&device);
        let layers = [
            p.retained_pool
                .acquire_texture(
                    8,
                    8,
                    TextureFormat::Rgba8Unorm,
                    goldy::types::TextureKind::DirectInterpolated,
                    goldy::types::TextureFlags::empty(),
                    None,
                )
                .expect("layer0"),
            p.retained_pool
                .acquire_texture(
                    8,
                    8,
                    TextureFormat::Rgba8Unorm,
                    goldy::types::TextureKind::DirectInterpolated,
                    goldy::types::TextureFlags::empty(),
                    None,
                )
                .expect("layer1"),
            p.retained_pool
                .acquire_texture(
                    8,
                    8,
                    TextureFormat::Rgba8Unorm,
                    goldy::types::TextureKind::DirectInterpolated,
                    goldy::types::TextureFlags::empty(),
                    None,
                )
                .expect("layer2"),
            p.retained_pool
                .acquire_texture(
                    8,
                    8,
                    TextureFormat::Rgba8Unorm,
                    goldy::types::TextureKind::DirectInterpolated,
                    goldy::types::TextureFlags::empty(),
                    None,
                )
                .expect("layer3"),
        ];
        p.store_scheme_render_targets(None, layers, 1);
        let ctx = device.create_context().expect("ctx");
        let hit = p.take_scheme_render_targets(&ctx, 8, 8, TextureFormat::Rgba8Unorm, true);
        assert!(hit.is_some());
        let (out, _) = hit.unwrap();
        assert!(out.is_none(), "direct-present cache must not retain out_image");
    }
}
