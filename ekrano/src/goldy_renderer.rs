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

use std::collections::HashMap;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};

use goldy::task_graph::{NodeAccess, NodeBuilder};
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags, TextureFormat};
use goldy::{
    Buffer, BufferPool, BufferView, ComputePipeline, DataAccess, Device, DeviceType, FrameHandle,
    FrameOrchestrator, ShaderModule, TaskGraph, Texture, TexturePool, TimelineValue,
    TransientAllocator, TransientAllocatorConfig, TransientAllocatorStrategy,
};

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
use ekrano_encoding::{BumpAllocators, Resolver};

const MAX_BUMP_RETRIES: usize = 2;

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

/// Snapshot of the transient allocator's state, useful for tests and diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct AllocatorStats {
    /// Total capacity of the backing buffer in bytes.
    pub capacity: u64,
    /// Bytes currently live (allocated minus freed-and-retired).
    pub used: u64,
    /// Number of frames waiting in the cleanup ring.
    pub cleanup_ring_depth: usize,
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
// FrameCleanup — deferred per-frame work processed after GPU completion
// -----------------------------------------------------------------------

struct FrameCleanup {
    /// The bump readback buffer (`ekrano.bump_buf`) for `robust=true` frames.
    /// Read to CPU on cleanup to update bump estimates; `None` for non-robust frames.
    bump_buf: Option<Buffer>,
    /// Owned buffers awaiting return to `ResourcePool` once the GPU retires
    /// the associated timeline. Each entry carries the pool name for reinsertion.
    recyclable_owned: Vec<(Buffer, &'static str)>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
    /// Render targets (`out_image` + `filter_layers`) to cache after GPU retirement,
    /// avoiding `TexturePool` round-trips every frame. Set to `None` when dimensions
    /// changed (e.g. window resize) — those textures go to `deferred_textures` instead.
    cacheable_render_targets: Option<Box<(Texture, [Texture; 4])>>,
    /// `OwnedShared` + pool-exempt pipeline buffers to cache after GPU retirement.
    /// Stored in `PersistentState::cached_pipeline` for reuse next frame.
    cacheable_pipeline: Option<crate::gpu_resources::CachedPipeline>,
}

struct GoldyShader {
    pipeline: ComputePipeline,
    bindings: Vec<BindType>,
}

/// WARP has a bug where SRV descriptors on structured buffers return incorrect
/// data. This manifests both as `FirstElement` being ignored on pool views and
/// as broader SRV corruption under heavy clip workloads. Disable pooling on
/// software adapters and force all buffer bindings to UAV descriptors.
fn use_pool(device: &Device) -> bool {
    device.device_type() != DeviceType::Cpu
}

fn force_uav(device: &Device) -> bool {
    device.device_type() == DeviceType::Cpu
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct BufferKey {
    size: u64,
    access: DataAccess,
    name: &'static str,
    buffer_flags: BufferFlags,
}

#[derive(Default)]
pub(crate) struct ResourcePool {
    bufs: HashMap<BufferKey, Vec<Buffer>>,
}

impl ResourcePool {
    pub(crate) fn get_buf_with_stride(
        &mut self,
        device: &Device,
        size: u64,
        name: &'static str,
        access: DataAccess,
        stride: Option<u32>,
        buffer_flags: BufferFlags,
    ) -> Result<Buffer> {
        let key = BufferKey {
            size,
            access,
            name,
            buffer_flags,
        };
        let pool = self.bufs.entry(key).or_default();
        if let Some(buf) = pool.pop() {
            return Ok(buf);
        }
        device
            .alloc_buffer(size, access, stride, buffer_flags)
            .map_err(|e| Error::Shader(e.to_string()))
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
}

/// Maximum number of unprocessed `FrameCleanup` entries before we force a
/// synchronous wait to prevent unbounded growth.
///
/// This directly controls the amount of per-frame GPU resources (textures,
/// owned buffers, pool views) that can be alive simultaneously. Each frame
/// allocates full-resolution textures (`out_image` + `filter_layers`), so a
/// depth of N means N * ~5 render-target-sized textures in flight. On a
/// Retina display, each texture can be >13 MB, so high values cause OOM
/// when vsync is off and frames outrun the GPU. Kept in sync with the
/// transient allocator's `max_regions`.
///
/// With the Tiger Lottie at Retina resolution, each frame holds ~173 MB of
/// compute buffers (in an allocator region) plus ~8 full-resolution textures
/// (~130 MB) plus ~11 owned buffers. Depth N means (N+1) × that footprint
/// alive simultaneously (ring + current frame). On Apple Silicon with unified
/// memory, depth >= 2 can exceed the GPU memory budget when vsync is off and
/// frames are submitted faster than the GPU retires them. Depth 1 forces a
/// Read the frame strategy from the environment.
///
/// Defaults to `LowLatency` (depth=1). Override via `EKRANO_FRAME_STRATEGY`:
///   - `low_latency` (default)
///   - `balanced`
///   - `max_throughput` or `max_throughput:<N>`
///
/// Legacy `EKRANO_CLEANUP_DEPTH=<N>` is still accepted as a fallback.
fn frame_strategy() -> goldy::FrameStrategy {
    if let Ok(val) = std::env::var("EKRANO_FRAME_STRATEGY") {
        return match val.to_ascii_lowercase().as_str() {
            "low_latency" | "lowlatency" => goldy::FrameStrategy::LowLatency,
            "balanced" => goldy::FrameStrategy::Balanced,
            "max_throughput" | "maxthroughput" => goldy::FrameStrategy::MaxThroughput {
                max_frames_in_flight: None,
            },
            other
                if other.starts_with("max_throughput:") || other.starts_with("maxthroughput:") =>
            {
                let n = other.rsplit(':').next().and_then(|s| s.parse::<u32>().ok());
                goldy::FrameStrategy::MaxThroughput {
                    max_frames_in_flight: n,
                }
            }
            _ => goldy::FrameStrategy::LowLatency,
        };
    }
    if let Ok(depth) = std::env::var("EKRANO_CLEANUP_DEPTH")
        && let Ok(d) = depth.parse::<usize>()
    {
        return match d {
            0 | 1 => goldy::FrameStrategy::LowLatency,
            2 => goldy::FrameStrategy::Balanced,
            n => goldy::FrameStrategy::MaxThroughput {
                max_frames_in_flight: Some(n as u32),
            },
        };
    }
    goldy::FrameStrategy::Balanced
}

// -----------------------------------------------------------------------
// PersistentState — GPU resources that survive across frames
// -----------------------------------------------------------------------

/// GPU resources that live for the lifetime of the renderer and are reused
/// across frames. Pool growth, texture reuse, and bump estimates all live here.
pub(crate) struct PersistentState {
    /// Owned buffer cache: recycles pool-exempt buffers (bump, indirect, etc.)
    pub(crate) pool: ResourcePool,
    /// Pluggable transient allocator for sub-frame pool allocations. Selected by
    /// `EKRANO_TRANSIENT_ALLOCATOR` env var; defaults to [`TransientAllocatorStrategy::Heap`].
    /// Lazily created on first [`Self::prepare_storage_pool`].
    pub(crate) storage_allocator: Option<Box<dyn TransientAllocator>>,
    /// Texture pool for intermediate render targets (gradient, filter layers, etc.)
    pub(crate) tex_pool: TexturePool,
    /// Bump allocator counters from the most recently drained frame.
    /// `None` until the first GPU readback completes.
    last_drained_bump: Option<BumpAllocators>,
    /// Persistent linear-filter + clamp-to-edge sampler for hardware-filtered texture reads
    /// (gradient ramps, image atlas bilinear). Lazily created on first render.
    pub(crate) linear_clamp_sampler: Option<goldy::Sampler>,
    /// Persistent nearest-filter + clamp-to-edge sampler for `IMAGE_QUALITY_LOW` reads.
    pub(crate) nearest_clamp_sampler: Option<goldy::Sampler>,
    /// Cached render targets (`out_image` + `filter_layers`) from the previous frame.
    /// Populated by `process_cleanup_phase_a` after GPU retirement so they can be
    /// reused directly by the next `PipelineResources::prepare` without a `TexturePool`
    /// round-trip. `None` until the first frame completes or after a resize.
    pub(crate) cached_render_targets: Option<Box<(Texture, [Texture; 4])>>,
    /// Cached `OwnedShared` + pool-exempt pipeline buffers from the previous frame.
    /// Populated by `process_cleanup_phase_a` so that `PipelineResources::prepare` can
    /// rebind them directly when buffer sizes are stable, saving `ResourcePool` lookups.
    pub(crate) cached_pipeline: Option<crate::gpu_resources::CachedPipeline>,
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
    /// Frame scheduling strategy. Used by `alloc_pipeline_buffer` to decide whether
    /// `CoarseOnly` buffers should be graph transients (graph coloring for VRAM reuse)
    /// or persistent owned handles (stable bindless indices for CB retention).
    pub(crate) strategy: goldy::FrameStrategy,
}

/// Helper on `PersistentState` to take cached render targets when dimensions match.
/// Returns `None` (and releases stale textures to `tex_pool`) if sizes differ.
impl PersistentState {
    pub(crate) fn take_cached_render_targets(
        &mut self,
        width: u32,
        height: u32,
        out_format: TextureFormat,
    ) -> Option<(Texture, [Texture; 4])> {
        let matches = self.cached_render_targets.as_ref().is_some_and(|b| {
            b.0.width() == width && b.0.height() == height && b.0.format() == out_format
        });
        if matches {
            self.cached_render_targets.take().map(|b| *b)
        } else {
            // Dimensions changed (resize): release stale textures back to the pool.
            if let Some(b) = self.cached_render_targets.take() {
                let (out, layers) = *b;
                self.tex_pool.release(out);
                for l in layers {
                    self.tex_pool.release(l);
                }
            }
            None
        }
    }
}

impl PersistentState {
    /// Prepare the per-frame transient allocator. Lazy-initialises the strategy from
    /// `GOLDY_TRANSIENT_ALLOCATOR` on the first call, then forwards `begin_frame` so the
    /// allocator can reclaim retired regions / wait on the previous epoch / grow as needed.
    fn prepare_storage_pool(&mut self, device: &Device, pool_size: u64) -> Result<()> {
        let _tz = goldy::tracy_zone!("ekrano.prepare_storage_pool");
        if !use_pool(device) {
            return Ok(());
        }

        if self.storage_allocator.is_none() {
            device.reset_buffer_heaps();

            let config = TransientAllocatorConfig {
                initial_size: pool_size,
                min_region_size: pool_size,
                max_regions: frame_strategy().depth(),
                alignment: 256,
                flags: BufferFlags::GPU_ONLY,
            };

            let strategy = std::env::var("EKRANO_TRANSIENT_ALLOCATOR")
                .ok()
                .as_deref()
                .and_then(TransientAllocatorStrategy::parse)
                .unwrap_or_default();
            log::info!(
                "[ekrano] transient allocator strategy = {:?} (set EKRANO_TRANSIENT_ALLOCATOR=bump|epoch to override)",
                strategy
            );
            self.storage_allocator = Some(
                strategy
                    .create(device, config)
                    .map_err(|e| Error::Shader(e.to_string()))?,
            );
        }

        let allocator = self
            .storage_allocator
            .as_mut()
            .expect("storage allocator was just initialised");
        allocator
            .begin_frame(device, pool_size)
            .map_err(|e| Error::Shader(e.to_string()))?;
        Ok(())
    }

    pub(crate) fn storage_allocator_mut(&mut self) -> Option<&mut Box<dyn TransientAllocator>> {
        self.storage_allocator.as_mut()
    }

    fn last_drained_bump(&self) -> Option<&BumpAllocators> {
        self.last_drained_bump.as_ref()
    }

    fn take_last_drained_bump(&mut self) -> Option<BumpAllocators> {
        self.last_drained_bump.take()
    }
}

/// Work that is safe to defer until after the coarse GPU flush.
/// Keeping this off the `begin_frame` critical path lets the GPU start coarse
/// execution while the CPU finishes pool returns and deferred deletions.
struct DeferredCleanup {
    recyclable_owned: Vec<(Buffer, &'static str)>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
    timeline: TimelineValue,
    /// `bump_buf` returned here (not to pool) only when it was used for readback;
    /// in that case it is returned to the `ResourcePool` after readback is done.
    bump_buf_for_pool: Option<Buffer>,
}

/// Phase A: latency-sensitive cleanup that must complete before `PipelineResources::prepare`.
///
/// - Reads bump buffer from GPU to CPU (so `persistent.last_drained_bump` is ready for the
///   next frame's config build).
/// - Stashes cached render targets and pipeline buffers into `PersistentState` so that
///   `PipelineResources::prepare` can reuse them without pool round-trips.
///
/// Returns `DeferredCleanup` containing the work that can safely overlap with GPU coarse execution.
fn process_cleanup_phase_a(
    device: &Device,
    persistent: &mut PersistentState,
    timeline: TimelineValue,
    entry: FrameCleanup,
) -> Result<DeferredCleanup> {
    let _tz = goldy::tracy_zone!("ekrano.cleanup_phase_a");
    let FrameCleanup {
        bump_buf,
        recyclable_owned,
        deferred_pool_views,
        deferred_textures,
        cacheable_render_targets,
        cacheable_pipeline,
    } = entry;

    // Bump readback: must complete before the next frame reads `last_drained_bump`.
    let bump_buf_for_pool = if let Some(ref buf) = bump_buf {
        let size = buf.size() as usize;
        let mut output = vec![0_u8; size];
        buf.read_to_cpu(device, &mut output)
            .map_err(|e| Error::Shader(e.to_string()))?;
        persistent.last_drained_bump = Some(bytemuck::pod_read_unaligned(&output));
        bump_buf
    } else {
        None
    };

    // Stash render target cache: must be set before `take_cached_render_targets` in prepare.
    persistent.cached_render_targets = cacheable_render_targets;

    // Stash pipeline buffer cache: must be set before `persistent.cached_pipeline.take()` in prepare.
    if let Some(new_cache) = cacheable_pipeline
        && let Some(old) = persistent.cached_pipeline.replace(new_cache)
    {
        persistent
            .pool
            .return_buf(old.info_bin_data, "ekrano.info_bin_data_buf");
        persistent.pool.return_buf(old.tile, "ekrano.tile_buf");
        persistent
            .pool
            .return_buf(old.segments, "ekrano.segments_buf");
        persistent.pool.return_buf(old.ptcl, "ekrano.ptcl_buf");
        persistent
            .pool
            .return_buf(old.blend_spill, "ekrano.blend_spill");
        persistent
            .pool
            .return_buf(old.fallback_indirect, "ekrano.indirect_count");
        // Return CoarseOnly buffers if present (depth=1 path).
        macro_rules! return_coarse {
            ($field:expr, $name:expr) => {
                if let Some(b) = $field {
                    persistent.pool.return_buf(b, $name);
                }
            };
        }
        return_coarse!(old.reduced, "ekrano.reduced_buf");
        return_coarse!(old.reduced2, "ekrano.reduced2_buf");
        return_coarse!(old.reduced_scan, "ekrano.reduced_scan_buf");
        return_coarse!(old.tagmonoid, "ekrano.tagmonoid_buf");
        return_coarse!(old.path_bbox, "ekrano.path_bbox_buf");
        return_coarse!(old.lines, "ekrano.lines_buf");
        return_coarse!(old.draw_reduced, "ekrano.draw_reduced_buf");
        return_coarse!(old.draw_monoid, "ekrano.draw_monoid_buf");
        return_coarse!(old.clip_inp, "ekrano.clip_inp_buf");
        return_coarse!(old.clip_el, "ekrano.clip_el_buf");
        return_coarse!(old.clip_bic, "ekrano.clip_bic_buf");
        return_coarse!(old.clip_bbox, "ekrano.clip_bbox_buf");
        return_coarse!(old.draw_bbox, "ekrano.draw_bbox_buf");
        return_coarse!(old.bin_header, "ekrano.bin_header_buf");
        return_coarse!(old.path, "ekrano.path_buf");
        return_coarse!(old.seg_counts, "ekrano.seg_counts_buf");
    }
    // If cacheable_pipeline is None (size-mismatch frame), leave persistent.cached_pipeline
    // as-is; prepare() will detect the mismatch and flush it to pool on next access.

    Ok(DeferredCleanup {
        recyclable_owned,
        deferred_pool_views,
        deferred_textures,
        timeline,
        bump_buf_for_pool,
    })
}

/// Phase B: deferrable cleanup that overlaps with GPU coarse execution.
///
/// Call this after `flush_mid_frame()` so the GPU is already executing the coarse
/// graph while the CPU recycles pool views, textures, and owned buffers.
fn process_cleanup_phase_b(
    device: &Device,
    persistent: &mut PersistentState,
    deferred: DeferredCleanup,
) -> Result<()> {
    let _tz = goldy::tracy_zone!("ekrano.cleanup_phase_b");
    let DeferredCleanup {
        recyclable_owned,
        deferred_pool_views,
        deferred_textures,
        timeline,
        bump_buf_for_pool,
    } = deferred;

    if let Some(allocator) = persistent.storage_allocator.as_mut() {
        for view in &deferred_pool_views {
            allocator.free(view.offset(), view.size(), Some(timeline));
        }
    }
    drop(deferred_pool_views);

    for tex in deferred_textures {
        persistent.tex_pool.release(tex);
    }

    for (buf, name) in recyclable_owned {
        persistent.pool.return_buf(buf, name);
    }

    if let Some(buf) = bump_buf_for_pool {
        persistent.pool.return_buf(buf, "ekrano.bump_buf");
    }

    device.flush_deferred_deletions();
    // compact_overflow_heaps is called periodically from run_frame, not every frame.

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
    shaders: FullShaders,
    resolver: Resolver,
    engine_shaders: Vec<GoldyShader>,
    /// Cross-frame GPU resources: pools, texture cache, bump readback.
    persistent: PersistentState,
    /// Pipelined frame cleanup (see `goldy::FrameOrchestrator`).
    frame_pipeline: FrameOrchestrator<FrameCleanup>,
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
    graph: &'a mut TaskGraph,
    frame_pipeline: &'a mut FrameOrchestrator<FrameCleanup>,
    frame_handle: FrameHandle,
    pub(crate) persistent: &'a mut PersistentState,
    shaders: &'a [GoldyShader],
    force_uav: bool,
    surface: Option<&'a goldy::Surface>,
    last_timeline: Option<TimelineValue>,
    /// Set to `true` by `finish` or `abort`; the `Drop` impl aborts the open frame
    /// if the recorder is dropped without being properly completed (e.g. on a `?` return).
    finished: bool,
    /// The bump readback buffer, separated from the general deferred-buffer list
    /// so it can be read back to CPU in `process_cleanup_phase_a` without index arithmetic.
    bump_buf_for_readback: Option<Buffer>,
    deferred_owned_buffers: Vec<(Buffer, &'static str)>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
    /// Render targets to cache after GPU retirement (populated by `schedule_pipeline_cleanup`,
    /// moved into `FrameCleanup` by `finish()` so they bypass `deferred_textures`).
    cacheable_render_targets: Option<Box<(Texture, [Texture; 4])>>,
    /// `OwnedShared` + pool-exempt pipeline buffers to cache after GPU retirement
    /// (populated by `schedule_pipeline_cleanup`, moved into `FrameCleanup` by `finish()`).
    cacheable_pipeline: Option<crate::gpu_resources::CachedPipeline>,
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
        graph: &'a mut TaskGraph,
        frame_pipeline: &'a mut FrameOrchestrator<FrameCleanup>,
        frame_handle: FrameHandle,
        persistent: &'a mut PersistentState,
        shaders: &'a [GoldyShader],
        surface: Option<&'a goldy::Surface>,
    ) -> Self {
        let fuav = force_uav(device);
        // Clear the long-lived graph so the schedule cache is preserved but
        // the node list is empty and ready for this frame's recording.
        graph.clear();
        // Read capacity hints before persistent is moved into Self.
        let owned_cap = persistent.deferred_owned_cap_hint;
        let pool_cap = persistent.deferred_pool_cap_hint;
        let tex_cap = persistent.deferred_textures_cap_hint;

        Self {
            device,
            graph,
            frame_pipeline,
            frame_handle,
            persistent,
            shaders,
            force_uav: fuav,
            surface,
            last_timeline: None,
            bump_buf_for_readback: None,
            deferred_owned_buffers: Vec::with_capacity(owned_cap),
            deferred_pool_views: Vec::with_capacity(pool_cap),
            deferred_textures: Vec::with_capacity(tex_cap),
            cacheable_render_targets: None,
            cacheable_pipeline: None,
            finished: false,
            indices_scratch: Vec::with_capacity(MAX_BINDLESS_SLOTS),
            filter_dispatch_slot: 0,
        }
    }

    pub(crate) fn graph_and_persistent(&mut self) -> (&mut TaskGraph, &mut PersistentState) {
        (&mut *self.graph, self.persistent)
    }

    /// Submit the current graph as a command buffer and start a fresh one.
    ///
    /// This lets the GPU begin executing early work (e.g. coarse rasterization)
    /// while the CPU continues recording later work (fine rasterization, filters)
    /// into a new command buffer.
    ///
    /// Coarse vs fine **GPU overlap**: coarse PTCL generation still completes before fine is
    /// recorded here — true concurrent coarse+fine compute within one frame would require
    /// bin/tile granular readiness (issue #46). See `doc/ISSUE_46_DEFERRED.md`.
    ///
    /// The coarse graph may contain transient buffers (coarse-only pipeline intermediates).
    /// These are safely submitted here because `alloc_pipeline_buffer` now allocates
    /// cross-phase shared buffers (`ptcl`, `segments`, `info_bin_data`, etc.) as real pooled
    /// handles, so no fine-phase spec leaks into the coarse graph. Goldy's
    /// `FrameOrchestrator::flush` uses `submit_pipelined`, which gives each flush its own
    /// placement-heap region — coarse and fine transients never alias the same bytes.
    #[allow(
        dead_code,
        reason = "Reserved for issue #46 mid-frame flush; not wired yet"
    )]
    pub(crate) fn flush_mid_frame(&mut self) -> Result<()> {
        self.frame_pipeline
            .flush(self.frame_handle, self.graph, &mut self.last_timeline)
            .map_err(|e| Error::Shader(e.to_string()))
    }

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
            GpuBuf::Pooled(v) => self.deferred_pool_views.push(v),
            GpuBuf::Transient(_) => {
                // Transient buffers live in the graph-scoped heap; no per-view cleanup needed.
            }
        }
    }

    pub(crate) fn defer_texture(&mut self, tex: Texture) {
        self.deferred_textures.push(tex);
    }

    pub(crate) fn schedule_pipeline_cleanup(
        &mut self,
        pipeline: crate::gpu_resources::PipelineResources,
        bump_readback: bool,
    ) {
        let crate::gpu_resources::PipelineResources {
            gradient,
            image_atlas,
            mask_atlas,
            scene,
            config,
            wg_counts,
            indirect,
            fallback_indirect,
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
        if let Some(b) = wg_counts {
            self.defer_gpu_buf(b, "ekrano.wg_counts");
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
        let cacheable_fallback_indirect = match fallback_indirect {
            GpuBuf::Owned(b) => Some(b),
            other => {
                self.defer_gpu_buf(other, "ekrano.indirect_count");
                None
            }
        };
        // CoarseOnly buffers: at depth=1 they are GpuBuf::Owned and can be stashed into
        // CachedPipeline for the next frame. At depth>1 they are GpuBuf::Transient and
        // are deferred normally (transient drop is a no-op).
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
        // CoarseOnly buffers are included when present (depth=1 only).
        self.cacheable_pipeline = match (
            cacheable_info_bin_data,
            cacheable_tile,
            cacheable_segments,
            cacheable_ptcl,
            cacheable_blend_spill,
            cacheable_fallback_indirect,
        ) {
            (
                Some(info_bin_data),
                Some(tile),
                Some(segments),
                Some(ptcl),
                Some(blend_spill),
                Some(fallback_indirect),
            ) => Some(crate::gpu_resources::CachedPipeline {
                info_bin_data,
                tile,
                segments,
                ptcl,
                blend_spill,
                fallback_indirect,
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
            }),
            (info_bin_data, tile, segments, ptcl, blend_spill, fallback_indirect) => {
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
                if let Some(b) = fallback_indirect {
                    self.deferred_owned_buffers
                        .push((b, "ekrano.indirect_count"));
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
                None
            }
        };
        // Cache render targets via the FrameCleanup ring rather than deferring
        // to tex_pool. After GPU retirement, process_cleanup_phase_a stores them in
        // persistent.cached_render_targets for the next frame to reuse without a
        // TexturePool round-trip.
        self.cacheable_render_targets = Some(Box::new((out_image, filter_layers)));
    }

    #[cfg_attr(
        not(feature = "debug_layers"),
        allow(dead_code, reason = "debug_layers only uses FrameRecorder::upload")
    )]
    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> Buffer {
        match record_upload_bytes_owned(
            self.device,
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
            self.force_uav,
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
            self.force_uav,
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
    /// Returns `(timeline, frame)`:
    /// - Standalone path: `(Some(tv), None)` — TV used for allocator epoch.
    /// - Surface path: `(None, Some(frame))` — caller presents, then calls
    ///   `note_presented` with the TV from `frame.present()`.
    fn finish(mut self) -> Result<(Option<TimelineValue>, Option<goldy::Frame>)> {
        self.finished = true;

        self.persistent.deferred_owned_cap_hint = self.deferred_owned_buffers.capacity();
        self.persistent.deferred_pool_cap_hint = self.deferred_pool_views.capacity();
        self.persistent.deferred_textures_cap_hint = self.deferred_textures.capacity();

        let cleanup = FrameCleanup {
            bump_buf: self.bump_buf_for_readback.take(),
            recyclable_owned: mem::take(&mut self.deferred_owned_buffers),
            deferred_pool_views: mem::take(&mut self.deferred_pool_views),
            deferred_textures: mem::take(&mut self.deferred_textures),
            cacheable_render_targets: self.cacheable_render_targets.take(),
            cacheable_pipeline: self.cacheable_pipeline.take(),
        };

        if let Some(surface) = self.surface {
            let frame = self
                .frame_pipeline
                .end_frame_for_surface(self.frame_handle, self.graph, surface, cleanup)
                .map_err(|e| Error::Shader(e.to_string()))?;
            Ok((None, Some(frame)))
        } else {
            let tv = self
                .frame_pipeline
                .end_frame_standalone(self.frame_handle, self.graph, self.last_timeline, cleanup)
                .map_err(|e| Error::Shader(e.to_string()))?;
            Ok((Some(tv), None))
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
            GpuBinding::View(v) => node.bind_buffer_view(v, access),
            GpuBinding::Tex(t) => node.bind_texture(t, access),
            GpuBinding::Transient(id) => node.bind_transient_buffer(*id, access),
            GpuBinding::SwapchainOutput(h) => node.bind_swapchain_output(*h, access),
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

        let budget = 512 * 1024 * 1024; // 512 MiB
        let tracking = TrackingVramAllocator::with_budget(
            std::sync::Arc::new(DefaultVramAllocator::new()),
            budget,
        );
        let tracked_device = device.with_vram_allocator(std::sync::Arc::new(tracking));

        // FrameOrchestrator uses the original (unbudgeted) device for transient-buffer
        // submission.  The placement heap for transient allocations must not be
        // constrained by the 512 MiB TrackingVramAllocator budget, because the heap
        // size depends on the scene's transient footprint and can legitimately exceed
        // that figure for large scenes. The tracked device is still used for storage-
        // pool allocations (via `prepare_storage_pool`).
        let mut renderer = Self {
            device: tracked_device.clone(),
            shaders: FullShaders::empty(),
            resolver: Resolver::new(),
            engine_shaders: Vec::new(),
            persistent: PersistentState {
                pool: ResourcePool::default(),
                storage_allocator: None,
                tex_pool: TexturePool::default(),
                last_drained_bump: None,
                linear_clamp_sampler: None,
                nearest_clamp_sampler: None,
                cached_render_targets: None,
                cached_pipeline: None,
                deferred_owned_cap_hint: 0,
                deferred_pool_cap_hint: 0,
                deferred_textures_cap_hint: 0,
                stable_mask_lut_msaa8: None,
                stable_mask_lut_msaa16: None,
                cached_wg_counts: None,
                cached_config_uniform: None,
                cached_filter_uniforms: Vec::new(),
                strategy: frame_strategy(),
            },
            frame_pipeline: FrameOrchestrator::with_strategy(device, frame_strategy()),
            persistent_bump: None,
            cleanup_frame_counter: 0,
            last_scene_fingerprint: None,
            last_packed_len: None,
            graph: TaskGraph::new(),
        };
        let shaders = shaders::goldy_full_shaders(&tracked_device, &mut renderer)?;
        renderer.shaders = shaders;
        tracked_device.release_idle_shader_compiler();
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

    /// Acknowledge a swapchain frame after [`goldy::Frame::present`].
    ///
    /// Fills in the timeline on the most recent `FrameCleanup` entry (the one
    /// pushed by `FrameRecorder::finish` for the surface path where the
    /// timeline isn't known until after present) and informs the transient
    /// allocator so it can retire this frame's regions with the correct epoch.
    pub fn note_frame_presented(&mut self, device: &Device, tv: TimelineValue) {
        self.frame_pipeline.note_presented(tv);
        if let Some(allocator) = self.persistent.storage_allocator_mut() {
            allocator.end_frame(device, tv);
        }
    }

    ///
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
        self.run_frame(device, scene, params, Some(texture), None)
    }

    /// Render a scene directly to a swapchain [`Surface`](goldy::Surface).
    ///
    /// Internally records the full graph (coarse + fine) with the swapchain as
    /// a late-bound output, then hands the graph to [`Surface::submit_graph`]
    /// which auto-partitions it, submits early work, acquires the swapchain
    /// image, and presents.  The caller does **not** need to call `acquire`,
    /// `present`, or `note_frame_presented` — everything is handled here.
    pub fn render_to_surface(
        &mut self,
        device: &Device,
        scene: &Scene,
        surface: &goldy::Surface,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        self.run_frame(device, scene, params, None, Some(surface))
    }

    /// Query the transient allocator's current state for diagnostics or test assertions.
    ///
    /// Returns `None` if the allocator hasn't been initialised yet (no frames rendered).
    pub fn allocator_stats(&self) -> Option<AllocatorStats> {
        self.persistent
            .storage_allocator
            .as_ref()
            .map(|a| AllocatorStats {
                capacity: a.capacity(),
                used: a.used_this_frame(),
                cleanup_ring_depth: self.frame_pipeline.pending_frames(),
            })
    }

    /// Query the device-owned placement heap's state for diagnostics / tests.
    ///
    /// Delegates to [`Device::placement_heap_stats`](goldy::Device::placement_heap_stats).
    /// Returns `None` if no transient-buffer graphs have been submitted yet.
    pub fn placement_heap_stats(&self) -> Option<goldy::placement_heap::PlacementHeapStats> {
        self.device.placement_heap_stats()
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
                SpatialAccess::Direct,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .map_err(|e| Error::Gpu(e.to_string()))?;

        for _attempt in 0..=MAX_BUMP_RETRIES {
            self.render_to_texture(device, scene, &texture, params)?;
            self.frame_pipeline
                .drain_all(|dev, rf| {
                    let deferred =
                        process_cleanup_phase_a(dev, &mut self.persistent, rf.timeline, rf.data)?;
                    process_cleanup_phase_b(dev, &mut self.persistent, deferred)
                })
                .map_err(|e| Error::Shader(e.to_string()))?;

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

    /// Shared implementation for [`render_to_texture`](Self::render_to_texture)
    /// and [`render_to_surface`](Self::render_to_surface).
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
        let _tz = goldy::tracy_zone!("ekrano.run_frame");
        use std::time::Instant;
        let frame_start = Instant::now();

        let encoding = scene.encoding();
        let mut stats = FrameStats::default();

        // Hoist resolve before begin_frame: pure CPU work, no GPU dependency.
        // This overlaps GPU N-1's tail execution while we await begin_frame.
        // Buffer sizing uses persistent_bump (running max across frames), which
        // is equivalent to prev_bump in steady state. On overflow frames we
        // recompute below after drain.
        //
        // Resolver::resolve returns Ramps<'a>/Images<'a> that borrow from the
        // resolver with lifetime 'a.  To allow &mut self calls (begin_frame,
        // update_persistent_bump) while those values are live, we temporarily
        // take the resolver out of self so the returned borrows are against a
        // local variable rather than self.  The resolver is restored after
        // PipelineResources::prepare consumes ramps/images.
        let t_resolve_start = Instant::now();
        let mut resolver = mem::take(&mut self.resolver);
        let mut packed = vec![];
        let (layout, ramps, images) = {
            let _rz = goldy::tracy_zone!("ekrano.resolve");
            resolver.resolve(encoding, &mut packed)
        };

        // Scene fingerprint disabled for baseline measurement.  The retention path
        // (Phase 1 of Command Buffer Reuse) relies on this to detect static scenes;
        // for animated scenes (Tiger benchmark) it is pure overhead because the
        // fingerprint never matches.  Re-enable by reverting this block to the
        // length-based + full-bytes fingerprint computation.
        let scene_fingerprint: u64 = 0;
        let _scene_unchanged: bool = false;
        self.last_packed_len = Some(packed.len());
        let base_config = ekrano_encoding::RenderConfig::new(
            &layout,
            params.width,
            params.height,
            &params.base_color,
        );
        let mut config = if let Some(ref persistent) = self.persistent_bump {
            base_config.with_bump_estimates(persistent)
        } else {
            base_config
        };
        let mut pool_size = {
            let _tz = goldy::tracy_zone!("ekrano.pool_size");
            BufferPool::padded_size(&config.buffer_sizes.pool_allocs())
                .saturating_add(POOL_SIZE_SLACK)
        };
        let t_resolve = t_resolve_start.elapsed();

        // --- Reclaim completed frames & open recording bracket ---
        // Phase A cleanup (latency-sensitive: bump readback + cache stash) runs inside
        // begin_frame, blocking frame start as briefly as possible.
        // Phase B (pool returns + deferred deletions) is deferred to after flush_mid_frame
        // so it overlaps GPU coarse execution.
        let t_drain_start = Instant::now();
        let mut phase_b: Option<DeferredCleanup> = None;
        let frame_handle = {
            let _tz = goldy::tracy_zone!("ekrano.begin_frame");
            self.frame_pipeline
                .begin_frame(|dev, rf| {
                    let deferred =
                        process_cleanup_phase_a(dev, &mut self.persistent, rf.timeline, rf.data)?;
                    phase_b = Some(deferred);
                    Ok::<(), Error>(())
                })
                .map_err(|e| Error::Shader(e.to_string()))?
        };
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
                config = ekrano_encoding::RenderConfig::new(
                    &layout,
                    params.width,
                    params.height,
                    &params.base_color,
                )
                .with_bump_estimates(&sanitize_bump(bump));
                pool_size = BufferPool::padded_size(&config.buffer_sizes.pool_allocs())
                    .saturating_add(POOL_SIZE_SLACK);
            }
        }

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
        {
            let _tz = goldy::tracy_zone!("ekrano.prepare_pool");
            if let Err(e) = self.persistent.prepare_storage_pool(device, pool_size) {
                if let Some(deferred) = phase_b.take() {
                    let _ = process_cleanup_phase_b(device, &mut self.persistent, deferred);
                }
                self.frame_pipeline.abort_frame(frame_handle);
                return Err(e);
            }
        }
        let t_pool = t1.elapsed();

        let t2 = Instant::now();
        let mut recorder = FrameRecorder::new(
            device,
            &mut self.graph,
            &mut self.frame_pipeline,
            frame_handle,
            &mut self.persistent,
            &self.engine_shaders,
            surface,
        );

        let (graph, persistent) = recorder.graph_and_persistent();
        let swapchain_handle = if surface.is_some() {
            Some(graph.declare_swapchain_output())
        } else {
            None
        };
        let mut pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.prepare");
            let out_image_format = surface
                .map(|s| s.format())
                .unwrap_or(TextureFormat::Rgba8Unorm);
            let pipeline_result = crate::gpu_resources::PipelineResources::prepare(
                device,
                graph,
                persistent,
                encoding,
                packed,
                ramps,
                images,
                params,
                &config,
                out_image_format,
            );
            self.resolver = resolver;
            match pipeline_result {
                Ok(p) => p,
                Err(e) => {
                    // Drop recorder first (aborts the frame) so self.persistent is free,
                    // then process deferred cleanup before returning.
                    drop(recorder);
                    if let Some(deferred) = phase_b.take() {
                        let _ = process_cleanup_phase_b(device, &mut self.persistent, deferred);
                    }
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
                params,
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
                encoding,
                &self.shaders,
                &pipeline,
                output_texture,
                swapchain_handle,
                &mut recorder,
            );
            #[cfg(feature = "debug_layers")]
            let _ = render.take_captured_buffers();
            crate::render::record_filter_effects(
                encoding,
                &self.shaders,
                &mut recorder,
                &pipeline,
                output_texture,
            );
        }
        let t_fine_record = t3.elapsed();

        let t4 = Instant::now();
        recorder.schedule_pipeline_cleanup(pipeline, params.robust);
        let (opt_tv, opt_frame) = {
            let _tz = goldy::tracy_zone!("ekrano.finish");
            recorder.finish()?
            // recorder is consumed here; self.persistent borrow is released.
        };

        // Phase B cleanup: pool returns + deferred deletions. Runs after GPU submit
        // so Phase B overlaps GPU coarse+fine execution rather than blocking begin_frame.
        if let Some(deferred) = phase_b.take() {
            process_cleanup_phase_b(device, &mut self.persistent, deferred)?;
        }

        if use_pool(device)
            && let Some(allocator) = self.persistent.storage_allocator_mut()
        {
            let used = allocator.used_this_frame();
            allocator.hint_unused_above(used);
        }

        // Surface path: present and notify allocator.
        if let Some(frame) = opt_frame {
            let tv = frame.present().map_err(|e| Error::Shader(e.to_string()))?;
            self.frame_pipeline.note_presented(tv);
            if let Some(allocator) = self.persistent.storage_allocator_mut() {
                allocator.end_frame(device, tv);
            }
        }
        // Standalone path: notify the allocator about the frame's epoch.
        if let Some(tv) = opt_tv
            && let Some(allocator) = self.persistent.storage_allocator_mut()
        {
            allocator.end_frame(device, tv);
        }
        let t_submit = t4.elapsed();

        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let label = if surface.is_some() { "surface" } else { "" };

        let (alloc_cap_mb, alloc_used_mb, ring_depth) =
            if let Some(a) = self.persistent.storage_allocator.as_ref() {
                (
                    a.capacity() as f64 / (1024.0 * 1024.0),
                    a.used_this_frame() as f64 / (1024.0 * 1024.0),
                    self.frame_pipeline.pending_frames(),
                )
            } else {
                (0.0, 0.0, self.frame_pipeline.pending_frames())
            };

        log::debug!(
            "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms alloc={:.1}/{:.1}MB ring={} {label}",
            frame_num,
            t_drain.as_secs_f64() * 1000.0,
            t_resolve.as_secs_f64() * 1000.0,
            t_pool.as_secs_f64() * 1000.0,
            t_coarse.as_secs_f64() * 1000.0,
            t_fine_record.as_secs_f64() * 1000.0,
            t_submit.as_secs_f64() * 1000.0,
            frame_start.elapsed().as_secs_f64() * 1000.0,
            alloc_used_mb,
            alloc_cap_mb,
            ring_depth,
        );

        // Update fingerprint after a successful frame.
        self.last_scene_fingerprint = Some(scene_fingerprint);

        Ok(stats)
    }

    // =======================================================================
    // Engine methods
    // =======================================================================

    /// Add a compute shader from Slang source.
    pub(crate) fn add_compute_shader(
        &mut self,
        device: &Device,
        _label: &'static str,
        slang_source: &str,
        bindings: &[BindType],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<ShaderId> {
        self.add_compute_shader_with_options(
            device,
            _label,
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
        let shader_module = ShaderModule::from_slang_with_options(
            device,
            slang_source,
            search_paths,
            defines,
            optimization_level,
            &[],
        )
        .map_err(|e| Error::Shader(format!("{:#}", e)))?;
        let pipeline = ComputePipeline::new(device, &shader_module)
            .map_err(|e| Error::Shader(format!("{:#}", e)))?;

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
            .create_device(DeviceType::DiscreteGpu)
            .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
            .or_else(|_| instance.create_device(DeviceType::Other))
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
            storage_allocator: None,
            tex_pool: TexturePool::default(),
            last_drained_bump: None,
            linear_clamp_sampler: None,
            nearest_clamp_sampler: None,
            cached_render_targets: None,
            cached_pipeline: None,
            deferred_owned_cap_hint: 0,
            deferred_pool_cap_hint: 0,
            deferred_textures_cap_hint: 0,
            stable_mask_lut_msaa8: None,
            stable_mask_lut_msaa16: None,
            cached_wg_counts: None,
            cached_config_uniform: None,
            cached_filter_uniforms: Vec::new(),
            strategy: goldy::FrameStrategy::LowLatency,
        };

        for &expected_format in &[TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm] {
            let mut resolver = Resolver::new();
            let mut packed = Vec::new();
            let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
            let config =
                RenderConfig::new(&layout, params.width, params.height, &params.base_color);
            let mut graph = TaskGraph::new();

            let pipeline = PipelineResources::prepare(
                &device,
                &mut graph,
                &mut persistent,
                encoding,
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
