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
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use goldy::task_graph::{NodeAccess, NodeBuilder};
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags, TextureFormat};
use goldy::{
    Buffer, BufferPool, BufferView, ComputePipeline, DataAccess, Device, DeviceType, Frame,
    ShaderModule, TaskGraph, Texture, TexturePool, TimelineValue, TransientAllocator,
    TransientAllocatorConfig, TransientAllocatorStrategy,
};

use mem::size_of;

use crate::{
    Error, RenderParams, Result, Scene,
    gpu_resources::{
        GpuBinding, GpuBuf, bind_type_to_node_access, collect_bindless_indices_direct,
        record_upload_bytes,
    },
    render::Render,
    resource_proxy::{BindType, ShaderId},
    shaders::{self, FullShaders},
};
use ekrano_encoding::{BumpAllocators, Resolver};

static DUMP_DIR: LazyLock<Option<String>> = LazyLock::new(|| std::env::var("EKRANO_DUMP_DIR").ok());

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
// Helper types (formerly in goldy_engine.rs)
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// FrameCleanup — deferred per-frame work processed after GPU completion
// -----------------------------------------------------------------------

struct FrameCleanup {
    timeline: Option<TimelineValue>,
    /// The bump readback buffer (`ekrano.bump_buf`) for `robust=true` frames.
    /// Read to CPU on cleanup to update bump estimates; `None` for non-robust frames.
    bump_buf: Option<Buffer>,
    /// Owned buffers awaiting return to `ResourcePool` once the GPU retires
    /// the associated timeline. Each entry carries the pool name for reinsertion.
    recyclable_owned: Vec<(Buffer, &'static str)>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
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
/// synchronous wait each frame, eliminating pipelining but keeping the total
/// under ~500 MB.
const MAX_CLEANUP_DEPTH: usize = 3;

// -----------------------------------------------------------------------
// PersistentState — GPU resources that survive across frames
// -----------------------------------------------------------------------

/// GPU resources that live for the lifetime of the renderer and are reused
/// across frames. Pool growth, texture reuse, and bump estimates all live here.
pub(crate) struct PersistentState {
    /// Owned buffer cache: recycles pool-exempt buffers (bump, indirect, etc.)
    pub(crate) pool: ResourcePool,
    /// Pluggable transient allocator for sub-frame pool allocations. Selected by
    /// `GOLDY_TRANSIENT_ALLOCATOR` env var; defaults to [`TransientAllocatorStrategy::BumpReset`]
    /// for backwards compatibility. Lazily created on first [`Self::prepare_storage_pool`].
    pub(crate) storage_allocator: Option<Box<dyn TransientAllocator>>,
    /// Texture pool for intermediate render targets (gradient, filter layers, etc.)
    pub(crate) tex_pool: TexturePool,
    /// Bump allocator counters from the most recently drained frame.
    /// `None` until the first GPU readback completes.
    last_drained_bump: Option<BumpAllocators>,
    /// Persistent linear-filter + clamp-to-edge sampler for hardware-filtered texture reads
    /// (gradient ramps, image atlas bilinear). Lazily created on first render.
    pub(crate) linear_clamp_sampler: Option<goldy::Sampler>,
    /// Persistent nearest-filter + clamp-to-edge sampler for IMAGE_QUALITY_LOW reads.
    pub(crate) nearest_clamp_sampler: Option<goldy::Sampler>,
}

impl PersistentState {
    /// Prepare the per-frame transient allocator. Lazy-initialises the strategy from
    /// `GOLDY_TRANSIENT_ALLOCATOR` on the first call, then forwards `begin_frame` so the
    /// allocator can reclaim retired regions / wait on the previous epoch / grow as needed.
    fn prepare_storage_pool(
        &mut self,
        device: &Device,
        _frame: &FrameState,
        pool_size: u64,
        expected_max: u64,
    ) -> Result<()> {
        if !use_pool(device) {
            return Ok(());
        }

        if self.storage_allocator.is_none() {
            // Reserve virtual address range up front so growth within `expected_max` is
            // a no-op page-bind rather than a costly realloc. Equivalent to the old hand-
            // rolled path in this function pre-refactor.
            device.reset_buffer_heaps();
            device.ensure_buffer_heap_capacity(expected_max.max(pool_size));

            let config = TransientAllocatorConfig {
                initial_size: pool_size,
                expected_max,
                min_region_size: pool_size,
                max_regions: MAX_CLEANUP_DEPTH,
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

// -----------------------------------------------------------------------
// FrameState — per-frame bookkeeping (bind map, cleanup ring, downloads)
// -----------------------------------------------------------------------

/// Per-frame bookkeeping: deferred cleanup ring (no bind map).
struct FrameState {
    cleanup_ring: VecDeque<FrameCleanup>,
}

// -----------------------------------------------------------------------
// GoldyRenderer — the merged struct
// -----------------------------------------------------------------------

/// Goldy-based 2D renderer.
///
/// Renders scenes to textures using the Goldy GPU backend with Slang shaders.
pub struct GoldyRenderer {
    #[allow(dead_code, reason = "stored for future use; clone is cheap")]
    device: Device,
    shaders: FullShaders,
    resolver: Resolver,
    engine_shaders: Vec<GoldyShader>,
    /// Cross-frame GPU resources: pools, texture cache, bump readback.
    persistent: PersistentState,
    /// Per-frame bookkeeping: bind map, cleanup ring, download results.
    frame: FrameState,
    /// Persistent bump estimates: running max across frames. Used to pre-size
    /// buffers even when no overflow occurs, avoiding the cold-start ramp-up.
    persistent_bump: Option<BumpAllocators>,
}

// -----------------------------------------------------------------------
// impl FrameState — bookkeeping methods (pool access via PersistentState)
// -----------------------------------------------------------------------

impl FrameState {
    /// Non-blocking drain: process any completed cleanup entries from the
    /// front of the ring. If the ring has grown beyond `MAX_CLEANUP_DEPTH`,
    /// force a synchronous wait on the oldest entry to prevent unbounded growth.
    fn try_drain_completed_frames(
        &mut self,
        device: &Device,
        persistent: &mut PersistentState,
    ) -> Result<()> {
        let progress = device.gpu_progress();

        while let Some(front) = self.cleanup_ring.front() {
            let done = match front.timeline {
                Some(tv) => progress >= tv,
                None => false,
            };
            let must_wait = !done && self.cleanup_ring.len() >= MAX_CLEANUP_DEPTH;
            if done || must_wait {
                let entry = self.cleanup_ring.pop_front().unwrap();
                self.process_cleanup(device, entry, must_wait, persistent)?;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn wait_until_gpu_idle(
        &mut self,
        device: &Device,
        persistent: &mut PersistentState,
    ) -> Result<()> {
        while let Some(entry) = self.cleanup_ring.pop_front() {
            self.process_cleanup(device, entry, true, persistent)?;
        }
        Ok(())
    }

    fn process_cleanup(
        &mut self,
        device: &Device,
        entry: FrameCleanup,
        force_wait: bool,
        persistent: &mut PersistentState,
    ) -> Result<()> {
        if let Some(tv) = entry.timeline {
            let already_done = device.gpu_progress() >= tv;
            if !already_done {
                if force_wait {
                    device
                        .wait_until(tv)
                        .map_err(|e| Error::Shader(e.to_string()))?;
                } else {
                    return Ok(());
                }
            }
        }

        // Read bump counters from GPU buffer (robust frames only).
        if let Some(buf) = &entry.bump_buf {
            let size = buf.size() as usize;
            let mut output = vec![0_u8; size];
            buf.read_to_cpu(device, &mut output)
                .map_err(|e| Error::Shader(e.to_string()))?;
            persistent.last_drained_bump = Some(bytemuck::pod_read_unaligned(&output));
        }

        // Return byte ranges for pool views that were not freed mid-pipeline.
        // The timeline is retired at this point, so the epoch is immediately drainable.
        if let Some(allocator) = persistent.storage_allocator.as_mut() {
            let retired_epoch = entry.timeline.unwrap_or(0);
            for view in &entry.deferred_pool_views {
                allocator.free(view.offset(), view.size(), Some(retired_epoch));
            }
        }
        drop(entry.deferred_pool_views);

        for tex in entry.deferred_textures {
            persistent.tex_pool.release(tex);
        }

        // Return owned buffers to the pool for reuse by future frames.
        for (buf, name) in entry.recyclable_owned {
            persistent.pool.return_buf(buf, name);
        }

        // Return the bump readback buffer to the pool (if present).
        if let Some(buf) = entry.bump_buf {
            persistent.pool.return_buf(buf, "ekrano.bump_buf");
        }

        device.flush_deferred_deletions();
        device.compact_overflow_heaps();

        Ok(())
    }

    fn release_pool(&mut self, device: &Device, persistent: &mut PersistentState) -> Result<()> {
        self.wait_until_gpu_idle(device, persistent)?;
        persistent.storage_allocator = None;
        Ok(())
    }
}

// -----------------------------------------------------------------------
// FrameRecorder — direct-execution recorder that builds TaskGraph nodes
// -----------------------------------------------------------------------

pub(crate) struct FrameRecorder<'a> {
    pub(crate) device: &'a Device,
    graph: TaskGraph,
    frame: &'a mut FrameState,
    pub(crate) persistent: &'a mut PersistentState,
    shaders: &'a [GoldyShader],
    force_uav: bool,
    surface_frame: Option<&'a Frame>,
    last_timeline: Option<TimelineValue>,
    /// The bump readback buffer, separated from the general deferred-buffer list
    /// so it can be read back to CPU in `process_cleanup` without index arithmetic.
    bump_buf_for_readback: Option<Buffer>,
    deferred_owned_buffers: Vec<(Buffer, &'static str)>,
    deferred_pool_views: Vec<BufferView>,
    deferred_textures: Vec<Texture>,
    dispatch_count: usize,
}

impl<'a> FrameRecorder<'a> {
    fn new(
        device: &'a Device,
        frame: &'a mut FrameState,
        persistent: &'a mut PersistentState,
        shaders: &'a [GoldyShader],
        surface_frame: Option<&'a Frame>,
    ) -> Self {
        let fuav = force_uav(device);
        let graph = TaskGraph::new();

        Self {
            device,
            graph,
            frame,
            persistent,
            shaders,
            force_uav: fuav,
            surface_frame,
            last_timeline: None,
            bump_buf_for_readback: None,
            deferred_owned_buffers: Vec::new(),
            deferred_pool_views: Vec::new(),
            deferred_textures: Vec::new(),
            dispatch_count: 0,
        }
    }

    pub(crate) fn graph_and_persistent(&mut self) -> (&mut TaskGraph, &mut PersistentState) {
        (&mut self.graph, self.persistent)
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
    pub(crate) fn flush_mid_frame(&mut self) -> Result<()> {
        // When the graph owns transient resources the flush is skipped: all
        // pipeline resources (coarse + fine) are currently allocated into a
        // single graph before the first flush, so submitting early would leave
        // fine-phase specs with no matching node. Splitting coarse vs fine allocation
        // (or staging transient graphs differently) would unlock the same pipelining
        // transient graphs already skip — still orthogonal to GPU-side coarse/fine overlap.
        if self.graph.has_transient_resources() {
            return Ok(());
        }
        flush_graph(
            &mut self.graph,
            self.device,
            &mut self.last_timeline,
            self.surface_frame,
            self.persistent,
        )
    }

    pub(crate) fn alloc_pipeline_buffer_named(
        &mut self,
        size: u64,
        stride: u32,
        name: &'static str,
        flags: BufferFlags,
    ) -> Result<GpuBuf, Error> {
        crate::gpu_resources::alloc_pipeline_buffer(
            self.device,
            &mut self.graph,
            self.persistent,
            size,
            stride,
            name,
            flags,
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
        } = pipeline;

        self.defer_texture(gradient);
        self.defer_texture(image_atlas);
        self.defer_texture(mask_atlas);
        self.defer_gpu_buf(scene, "ekrano.scene");
        self.defer_gpu_buf(config, "ekrano.config");
        if let Some(b) = wg_counts {
            self.defer_gpu_buf(b, "ekrano.wg_counts");
        }
        if let Some(b) = indirect {
            self.defer_gpu_buf(b, "ekrano.indirect_dispatch");
        }
        self.defer_gpu_buf(fallback_indirect, "ekrano.indirect_count");
        self.defer_gpu_buf(info_bin_data, "ekrano.info_bin_data_buf");
        self.defer_gpu_buf(tile, "ekrano.tile");
        self.defer_gpu_buf(segments, "ekrano.segments_buf");
        self.defer_gpu_buf(ptcl, "ekrano.ptcl_buf");
        self.defer_gpu_buf(reduced, "ekrano.reduced");
        self.defer_gpu_buf(reduced2, "ekrano.reduced2");
        self.defer_gpu_buf(reduced_scan, "ekrano.reduced_scan");
        self.defer_gpu_buf(tagmonoid, "ekrano.tagmonoid");
        self.defer_gpu_buf(path_bbox, "ekrano.path_bbox");
        if bump_readback {
            if let GpuBuf::Owned(b) = bump {
                self.bump_buf_for_readback = Some(b);
            } else {
                self.defer_gpu_buf(bump, "ekrano.bump_buf");
            }
        } else {
            self.defer_gpu_buf(bump, "ekrano.bump_buf");
        }
        self.defer_gpu_buf(lines, "ekrano.lines");
        self.defer_gpu_buf(draw_reduced, "ekrano.draw_reduced");
        self.defer_gpu_buf(draw_monoid, "ekrano.draw_monoid");
        self.defer_gpu_buf(clip_inp, "ekrano.clip_inp");
        self.defer_gpu_buf(clip_el, "ekrano.clip_el");
        self.defer_gpu_buf(clip_bic, "ekrano.clip_bic");
        self.defer_gpu_buf(clip_bbox, "ekrano.clip_bbox");
        self.defer_gpu_buf(draw_bbox, "ekrano.draw_bbox");
        self.defer_gpu_buf(bin_header, "ekrano.bin_header");
        self.defer_gpu_buf(path, "ekrano.path");
        self.defer_gpu_buf(seg_counts, "ekrano.seg_counts");
        self.defer_gpu_buf(blend_spill, "ekrano.blend_spill");
        self.defer_texture(out_image);
        for t in filter_layers {
            self.defer_texture(t);
        }
    }

    #[cfg_attr(
        not(feature = "debug_layers"),
        allow(dead_code, reason = "debug_layers only uses FrameRecorder::upload")
    )]
    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> Buffer {
        let data = data.into();
        match record_upload_bytes(
            self.device,
            &mut self.graph,
            self.persistent,
            name,
            1,
            &data,
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
        let data = data.into();
        match record_upload_bytes(
            self.device,
            &mut self.graph,
            self.persistent,
            name,
            element_stride,
            &data,
        )
        .expect("upload_strided failed")
        {
            GpuBuf::Owned(b) => b,
            _ => panic!("upload_strided must produce owned buffer"),
        }
    }

    pub fn upload_typed<T: bytemuck::Pod>(&mut self, name: &'static str, data: &T) -> Buffer {
        let bytes = bytemuck::bytes_of(data).to_vec();
        self.upload_strided(name, size_of::<T>() as u32, bytes)
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
        let indices = collect_bindless_indices_direct(
            bindings,
            bind_types,
            self.force_uav,
            MAX_BINDLESS_SLOTS,
        )
        .expect("collect_bindless_indices_direct failed in dispatch");

        if let Some(ref dir) = *DUMP_DIR {
            let mut debug_indices = indices.clone();
            debug_indices.extend_from_slice(push_tail);
            dump_dispatch_gpu(
                self.device,
                self.dispatch_count,
                shader_id,
                (x, y, z),
                bindings,
                &debug_indices,
                dir,
            );
        }

        let mut node = self
            .graph
            .node("dispatch", &self.shaders[shader_id.0].pipeline);
        node = bind_graph_direct(node, bindings, bind_types);
        if !indices.is_empty() || !push_tail.is_empty() {
            node = node.bind_resources_raw_with_user(indices, push_tail);
        }
        node.dispatch(x, y, z);
        self.dispatch_count += 1;
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
        let indices = collect_bindless_indices_direct(
            bindings,
            bind_types,
            self.force_uav,
            MAX_BINDLESS_SLOTS,
        )
        .expect("collect_bindless_indices_direct failed in dispatch_indirect");

        if let Some(ref dir) = *DUMP_DIR {
            let indirect_dims =
                GoldyRenderer::read_indirect_dims(self.device, indirect_buf, offset);
            dump_dispatch_gpu(
                self.device,
                self.dispatch_count,
                shader_id,
                indirect_dims,
                bindings,
                &indices,
                dir,
            );
        }

        let mut node = self
            .graph
            .node("dispatch_indirect", &self.shaders[shader_id.0].pipeline);
        node = bind_graph_direct(node, bindings, bind_types);
        node = node.bind_buffer(indirect_buf, NodeAccess::Read);
        if !indices.is_empty() {
            node = node.bind_resources_raw(indices);
        }
        node.dispatch_indirect(indirect_buf, offset);
        self.dispatch_count += 1;
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

    pub(crate) fn defer_owned_buffer(&mut self, buf: Buffer, name: &'static str) {
        self.deferred_owned_buffers.push((buf, name));
    }

    /// Finish dispatch: flush the final graph and push a `FrameCleanup`
    /// entry onto the deque for deferred processing after GPU completion.
    fn finish(mut self) -> Result<()> {
        flush_graph(
            &mut self.graph,
            self.device,
            &mut self.last_timeline,
            self.surface_frame,
            self.persistent,
        )?;

        let timeline = match self.surface_frame {
            None => self.last_timeline.take(),
            Some(_) => None,
        };

        self.frame.cleanup_ring.push_back(FrameCleanup {
            timeline,
            bump_buf: self.bump_buf_for_readback,
            recyclable_owned: self.deferred_owned_buffers,
            deferred_pool_views: self.deferred_pool_views,
            deferred_textures: self.deferred_textures,
        });

        Ok(())
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
            // Samplers are stateless — their slot index flows through push-constants
            // but they need no resource-barrier tracking in the task graph.
            GpuBinding::Sampler(_) => node,
        };
    }
    node
}

#[allow(
    clippy::print_stdout,
    reason = "dump_dispatch prints manifest paths to stdout for debugging when dump is enabled"
)]
fn dump_dispatch_gpu(
    device: &Device,
    dispatch_idx: usize,
    shader_id: ShaderId,
    dims: (u32, u32, u32),
    bindings: &[GpuBinding<'_>],
    indices: &[u32],
    dump_dir: &str,
) {
    use std::io::Write;
    let dir = format!("{dump_dir}/dispatch_{dispatch_idx}");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("[dump] failed to create dump directory {dir}: {e}");
        return;
    }

    let manifest_path = format!("{dir}/manifest.txt");
    let mut manifest = match std::fs::File::create(&manifest_path) {
        Ok(f) => f,
        Err(e) => {
            log::error!("[dump] failed to create manifest {manifest_path}: {e}");
            return;
        }
    };

    macro_rules! wln {
        ($($arg:tt)*) => {
            if let Err(e) = writeln!(manifest, $($arg)*) {
                log::error!("[dump] manifest write failed: {e}");
                return;
            }
        };
    }

    wln!("shader_id: {}", shader_id.0);
    wln!("dispatch: ({}, {}, {})", dims.0, dims.1, dims.2);
    wln!("num_bindings: {}", bindings.len());
    wln!("resource_slots: {:?}", indices);

    for (i, binding) in bindings.iter().enumerate() {
        match binding {
            GpuBinding::Buf(buf) => {
                let size = buf.size() as usize;
                wln!(
                    "binding[{i}]: buf size={size} bindless={}",
                    buf.bindless_index().unwrap_or(u32::MAX)
                );
                let mut data = vec![0_u8; size];
                let ok = buf.read_to_cpu(device, &mut data).is_ok();
                if ok {
                    std::fs::write(format!("{dir}/buf_{i}.bin"), &data).ok();
                } else {
                    wln!("  (read failed)");
                }
            }
            GpuBinding::View(view) => {
                let size = view.size() as usize;
                wln!(
                    "binding[{i}]: buf_view size={size} bindless={}",
                    view.bindless_index().unwrap_or(u32::MAX)
                );
                let mut data = vec![0_u8; size];
                let ok = view.read_to_cpu(device, &mut data).is_ok();
                if ok {
                    std::fs::write(format!("{dir}/buf_{i}.bin"), &data).ok();
                } else {
                    wln!("  (read failed)");
                }
            }
            GpuBinding::Tex(_) => {
                wln!("binding[{i}]: texture (dump not implemented)");
            }
            GpuBinding::Transient(id) => {
                wln!("binding[{i}]: transient(id={})", id.0);
            }
            GpuBinding::Sampler(_) => {
                wln!("binding[{i}]: sampler");
            }
        }
    }
    println!(
        "[dump] dispatch_{dispatch_idx}: shader={} dims={:?} bindings={}",
        shader_id.0,
        dims,
        bindings.len()
    );
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
        let device = device.with_vram_allocator(std::sync::Arc::new(tracking));

        let mut renderer = Self {
            device: device.clone(),
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
            },
            frame: FrameState {
                cleanup_ring: VecDeque::new(),
            },
            persistent_bump: None,
        };
        let shaders = shaders::goldy_full_shaders(&device, &mut renderer)?;
        renderer.shaders = shaders;
        device.release_idle_shader_compiler();
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

    // Compute the pool shrink max: if the current pool is significantly oversized
    // relative to recent demand, returns `Some(target)` to cap the pool and trigger
    // a reallocation. Returns `None` if no shrink needed.
    // =======================================================================
    // Public API (unchanged signatures)
    // =======================================================================

    /// Acknowledge a swapchain frame after [`goldy::Frame::present`].
    ///
    /// Fills in the timeline on the most recent `FrameCleanup` entry (the one
    /// pushed by `FrameRecorder::finish` for the surface path where the
    /// timeline isn't known until after present) and informs the transient
    /// allocator so it can retire this frame's regions with the correct epoch.
    pub fn note_frame_presented(&mut self, device: &Device, tv: TimelineValue) {
        // Stamp the most recent cleanup entry with the presentation timeline so
        // process_cleanup knows when its buffers can be recycled.
        if let Some(back) = self.frame.cleanup_ring.back_mut()
            && back.timeline.is_none()
        {
            back.timeline = Some(tv);
        }

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

    /// Like [`Self::render_to_texture`], but records compute into a swapchain [`Frame`] via
    /// [`Frame::submit_compute`] instead of standalone [`Device::submit`].
    ///
    /// After this returns, call [`goldy::Frame::present`] and then [`Self::note_frame_presented`]
    /// with the returned [`TimelineValue`] before the next `begin` / render call.
    pub fn render_to_frame(
        &mut self,
        device: &Device,
        scene: &Scene,
        frame: &Frame,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        self.run_frame(device, scene, params, Some(frame.texture()), Some(frame))
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
                cleanup_ring_depth: self.frame.cleanup_ring.len(),
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
            while let Some(entry) = self.frame.cleanup_ring.pop_front() {
                self.frame
                    .process_cleanup(device, entry, true, &mut self.persistent)?;
            }

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

        self.frame.release_pool(device, &mut self.persistent)?;

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
    /// and [`render_to_frame`](Self::render_to_frame).
    ///
    /// Creates a [`FrameRecorder`], runs the full coarse+fine pipeline into it,
    /// then flushes the resulting [`TaskGraph`].
    fn run_frame(
        &mut self,
        device: &Device,
        scene: &Scene,
        params: &RenderParams,
        output_texture: Option<&Texture>,
        surface_frame: Option<&Frame>,
    ) -> Result<FrameStats> {
        use std::time::Instant;
        let frame_start = Instant::now();

        let encoding = scene.encoding();
        let mut stats = FrameStats::default();

        // --- Non-blocking drain of completed frames ---
        let t_drain_start = Instant::now();
        self.frame
            .try_drain_completed_frames(device, &mut self.persistent)?;
        self.persistent.pool.cap_pool_depth(12);
        let t_drain = t_drain_start.elapsed();

        let prev_bump = self.persistent.take_last_drained_bump();

        // Fix 5: Only trust bump readback when robust mode produced valid data.
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
            // Fix 3: Update persistent bump estimates (running max across frames).
            self.update_persistent_bump(&sanitize_bump(bump));
        }

        let t0 = Instant::now();
        // Resolve once: pack the scene and obtain ramp/image references.
        // The packed data and resolve results are threaded directly into
        // run_coarse so the scene is never packed a second time.
        let mut packed = vec![];
        let (layout, ramps, images) = self.resolver.resolve(encoding, &mut packed);
        let config = {
            let base = ekrano_encoding::RenderConfig::new(
                &layout,
                params.width,
                params.height,
                &params.base_color,
            );
            if let Some(ref bump) = prev_bump
                && bump.failed != 0
            {
                stats.bump_retries += 1;
                log::info!(
                    "Previous frame bump overflow (0x{:x}), growing buffers",
                    bump.failed,
                );
                base.with_bump_estimates(&sanitize_bump(bump))
            } else if let Some(ref persistent) = self.persistent_bump {
                // Use accumulated knowledge to pre-size even without overflow.
                base.with_bump_estimates(persistent)
            } else {
                base
            }
        };
        let t_resolve = t0.elapsed();

        let base = BufferPool::padded_size(&config.buffer_sizes.pool_allocs());
        let pool_size = base.saturating_add(POOL_SIZE_SLACK);
        let expected_max = {
            let mut cfg_est = ekrano_encoding::RenderConfig::new(
                &layout,
                params.width,
                params.height,
                &params.base_color,
            );
            if let Some(ref b) = self.persistent_bump {
                cfg_est = cfg_est.with_bump_estimates(&sanitize_bump(b));
            }
            let est = BufferPool::padded_size(&cfg_est.buffer_sizes.pool_allocs());
            est.saturating_add(POOL_SIZE_SLACK).max(pool_size)
        };

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
        self.persistent
            .prepare_storage_pool(device, &self.frame, pool_size, expected_max)?;
        let t_pool = t1.elapsed();

        let t2 = Instant::now();
        let mut recorder = FrameRecorder::new(
            device,
            &mut self.frame,
            &mut self.persistent,
            &self.engine_shaders,
            surface_frame,
        );

        let (graph, persistent) = recorder.graph_and_persistent();
        let mut pipeline = crate::gpu_resources::PipelineResources::prepare(
            device, graph, persistent, encoding, packed, ramps, images, params, &config,
        )?;

        let mut render = Render::new();
        render.run_coarse(
            encoding,
            &mut pipeline,
            &self.shaders,
            params,
            params.robust,
            &config,
            &mut recorder,
        );
        let t_coarse = t2.elapsed();

        // Submit coarse wave pipeline before recording fine so the GPU can start coarse work
        // while this thread fills fine dispatch — CPU/GPU overlap only (issue #46 discusses
        // deeper GPU coarse+fine concurrency).
        recorder.flush_mid_frame()?;

        let t3 = Instant::now();
        render.record_fine(
            encoding,
            &self.shaders,
            &pipeline,
            output_texture,
            &mut recorder,
        );
        crate::render::record_filter_effects(
            encoding,
            &self.shaders,
            &mut recorder,
            &pipeline,
            output_texture,
        );
        let t_fine_record = t3.elapsed();

        #[cfg(feature = "debug_layers")]
        if render.take_captured_buffers().is_some() {
            log::debug!(
                "debug_layers: coarse buffer capture is not yet wired for direct GPU resources"
            );
        }

        let t4 = Instant::now();
        recorder.schedule_pipeline_cleanup(pipeline, params.robust);
        recorder.finish()?;
        if use_pool(device)
            && let Some(allocator) = self.persistent.storage_allocator_mut()
        {
            let used = allocator.used_this_frame();
            allocator.hint_unused_above(used);
        }
        // Notify the allocator about the frame's epoch so it can track in-flight allocations.
        // For non-surface paths the timeline is on the just-pushed cleanup entry; for surface
        // paths it's filled in later by `note_frame_presented` (see that method's hook).
        if let Some(tv) = self.frame.cleanup_ring.back().and_then(|e| e.timeline)
            && let Some(allocator) = self.persistent.storage_allocator_mut()
        {
            allocator.end_frame(device, tv);
        }
        let t_submit = t4.elapsed();

        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let label = if surface_frame.is_some() {
            "surface"
        } else {
            ""
        };

        let (alloc_cap_mb, alloc_used_mb, ring_depth) =
            if let Some(a) = self.persistent.storage_allocator.as_ref() {
                (
                    a.capacity() as f64 / (1024.0 * 1024.0),
                    a.used_this_frame() as f64 / (1024.0 * 1024.0),
                    self.frame.cleanup_ring.len(),
                )
            } else {
                (0.0, 0.0, self.frame.cleanup_ring.len())
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

        Ok(stats)
    }

    // =======================================================================
    // Engine methods (formerly on GoldyEngine)
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

    fn read_indirect_dims(device: &Device, buf: &Buffer, offset: u64) -> (u32, u32, u32) {
        let off = offset as usize;
        let decode = |src: &[u8]| -> Option<(u32, u32, u32)> {
            if off + 12 > src.len() {
                return None;
            }
            let x = u32::from_le_bytes(src[off..off + 4].try_into().ok()?);
            let y = u32::from_le_bytes(src[off + 4..off + 8].try_into().ok()?);
            let z = u32::from_le_bytes(src[off + 8..off + 12].try_into().ok()?);
            Some((x, y, z))
        };

        let mut raw = [0_u8; 12];
        if buf.read_to_cpu(device, &mut raw).is_ok() {
            return decode(&raw).unwrap_or((0, 0, 0));
        }

        let mut full = vec![0_u8; buf.size() as usize];
        if buf.read_to_cpu(device, &mut full).is_ok() {
            return decode(&full).unwrap_or((0, 0, 0));
        }

        (0, 0, 0)
    }
}

// -----------------------------------------------------------------------
// Free functions and helper impls (formerly in goldy_engine.rs)
// -----------------------------------------------------------------------

fn flush_graph(
    graph: &mut TaskGraph,
    device: &Device,
    last_timeline: &mut Option<TimelineValue>,
    surface_frame: Option<&Frame>,
    _persistent: &mut PersistentState,
) -> Result<()> {
    if graph.is_empty() {
        return Ok(());
    }

    if let Some(frame) = surface_frame {
        frame
            .submit_compute(graph)
            .map_err(|e| Error::Shader(e.to_string()))?;
    } else {
        let tv = device
            .submit(graph)
            .map_err(|e| Error::Shader(e.to_string()))?;
        *last_timeline = Some(tv);
    }

    *graph = TaskGraph::new();
    Ok(())
}
