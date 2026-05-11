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
    Buffer, BufferPool, BufferPoolRing, BufferView, ComputePipeline, DataAccess, Device,
    DeviceType, Frame, ShaderModule, TaskGraph, Texture, TexturePool, TimelineValue,
};

use mem::size_of;

use crate::{
    Error, RenderParams, Result, Scene,
    low_level::{BufferProxy, ImageProxy, ResourceId, ResourceProxy, ShaderId},
    render::{self, Render},
    resource_proxy::BindType,
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

/// Either an owned buffer (exempt from pooling) or a view from the storage pool.
///
/// Exempt buffers (bump, indirect) need `read_to_cpu`, `dispatch_indirect`, or clear;
/// pooled buffers only need `bindless_index` for compute shader binding.
enum GpuBuffer {
    Owned(Buffer),
    Pooled(BufferView),
}

impl GpuBuffer {
    fn bindless_index(&self) -> Option<u32> {
        match self {
            Self::Owned(b) => b.bindless_index(),
            Self::Pooled(v) => v.bindless_index(),
        }
    }

    fn bindless_srv_index(&self) -> Option<u32> {
        match self {
            Self::Owned(b) => b.bindless_srv_index(),
            Self::Pooled(v) => v.bindless_srv_index(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Owned(b) => b.size(),
            Self::Pooled(v) => v.size(),
        }
    }

    /// For `dispatch_indirect`; only Owned buffers are used as indirect sources.
    fn as_owned(&self) -> Option<&Buffer> {
        match self {
            Self::Owned(b) => Some(b),
            Self::Pooled(_) => None,
        }
    }
}

struct GoldyShader {
    pipeline: ComputePipeline,
    bindings: Vec<BindType>,
}

#[derive(Default)]
struct BindMap {
    buf_map: HashMap<ResourceId, (GpuBuffer, &'static str)>,
    image_map: HashMap<ResourceId, (Texture, &'static str)>,
}

fn is_pool_exempt(name: &str) -> bool {
    matches!(
        name,
        "ekrano.bump_buf"
            | "ekrano.indirect_dispatch"
            | "ekrano.tile_buf"
            | "ekrano.lines_buf"
            | "ekrano.seg_counts_buf"
            | "ekrano.segments_buf"
            | "ekrano.path_buf"
    )
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

#[derive(Hash, PartialEq, Eq)]
struct BufferKey {
    size: u64,
    access: DataAccess,
    name: &'static str,
    buffer_flags: BufferFlags,
}

#[derive(Default)]
struct ResourcePool {
    bufs: HashMap<BufferKey, Vec<Buffer>>,
}

// -----------------------------------------------------------------------
// FrameCleanup — deferred per-frame work processed after GPU completion
// -----------------------------------------------------------------------

struct FrameCleanup {
    timeline: Option<TimelineValue>,
    pending_downloads: Vec<BufferProxy>,
    deferred_free_buffers: Vec<ResourceId>,
    deferred_free_images: Vec<ResourceId>,
    output_image_id: Option<ResourceId>,
}

/// Maximum number of unprocessed `FrameCleanup` entries before we force a
/// synchronous wait to prevent unbounded growth.
const MAX_CLEANUP_DEPTH: usize = 2;

// -----------------------------------------------------------------------
// PersistentState — GPU resources that survive across frames
// -----------------------------------------------------------------------

/// GPU resources that live for the lifetime of the renderer and are reused
/// across frames. Pool growth, texture reuse, and bump estimates all live here.
struct PersistentState {
    /// Owned buffer cache: recycles pool-exempt buffers (bump, indirect, etc.)
    pool: ResourcePool,
    /// Main storage pool: a ring-buffered large GPU allocation that pooled
    /// compute buffers sub-allocate from each frame.
    storage_ring: BufferPoolRing,
    /// Texture pool for intermediate render targets (gradient, filter layers, etc.)
    tex_pool: TexturePool,
    /// Bump allocator counters from the most recently drained frame.
    /// `None` until the first GPU readback completes.
    last_drained_bump: Option<BumpAllocators>,
}

impl PersistentState {
    fn prepare_storage_pool(
        &mut self,
        device: &Device,
        pool_size: u64,
        max_size: Option<u64>,
    ) -> Result<()> {
        if !use_pool(device) {
            return Ok(());
        }
        self.storage_ring
            .prepare_bounded(device, pool_size, max_size)
            .map_err(|e| Error::Shader(e.to_string()))
    }

    fn storage_pool_mut(&mut self) -> Option<&mut BufferPool> {
        self.storage_ring.current_mut()
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

/// Per-frame bookkeeping: the resource bind map, deferred cleanup ring,
/// and in-flight download results.
///
/// The bind map is partly persistent (owned buffers survive until explicitly
/// freed) and partly rebuilt each frame (pooled views are evicted at frame
/// start via [`FrameState::evict_pooled_entries`]).
struct FrameState {
    bind_map: BindMap,
    downloads: HashMap<ResourceId, Vec<u8>>,
    cleanup_ring: VecDeque<FrameCleanup>,
}

// -----------------------------------------------------------------------
// GoldyRenderer — the merged struct
// -----------------------------------------------------------------------

/// Number of frames over which pool-size history is tracked for shrink hysteresis.
const POOL_SHRINK_WINDOW: usize = 60;

/// Only shrink if the current pool is at least this factor larger than the rolling max.
const POOL_SHRINK_THRESHOLD: f64 = 2.0;

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
    /// Rolling history of requested pool sizes for shrink hysteresis.
    pool_size_history: VecDeque<u64>,
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

        let bump_name = bumps_buf_static_name();
        for buf_proxy in &entry.pending_downloads {
            if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id)
                && let GpuBuffer::Owned(buf) = gpu_buf
            {
                let size = buf.size() as usize;
                let mut output = vec![0_u8; size];
                buf.read_to_cpu(device, &mut output)
                    .map_err(|e| Error::Shader(e.to_string()))?;
                self.downloads.insert(buf_proxy.id, output);
                if buf_proxy.name == bump_name {
                    if let Some(data) = self.downloads.get(&buf_proxy.id) {
                        persistent.last_drained_bump =
                            Some(bytemuck::pod_read_unaligned(data));
                    }
                    self.downloads.remove(&buf_proxy.id);
                }
            }
        }

        for id in &entry.deferred_free_buffers {
            self.bind_map.remove_buf(*id);
        }
        for id in &entry.deferred_free_images {
            if let Some((tex, _)) = self.bind_map.take_image(*id) {
                persistent.tex_pool.release(tex);
            }
        }
        if let Some(id) = entry.output_image_id {
            self.bind_map.remove_image(id);
        }

        device.flush_deferred_deletions();

        Ok(())
    }

    /// Remove all `Pooled` entries from `bind_map` so they get rematerialized
    /// into the current frame's storage pool.
    fn evict_pooled_entries(&mut self) {
        self.bind_map
            .buf_map
            .retain(|_, (buf, _)| !matches!(buf, GpuBuffer::Pooled(_)));
    }

    fn ensure_resources_materialized(
        &mut self,
        device: &Device,
        graph: &mut TaskGraph,
        bindings: &[ResourceProxy],
        bind_types: &[BindType],
        persistent: &mut PersistentState,
    ) -> Result<()> {
        for (i, res) in bindings.iter().enumerate() {
            match res {
                ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                    if self.bind_map.get_buf(proxy.id).is_none() {
                        let stride = proxy.element_stride;
                        let gpu_buf = if !is_pool_exempt(proxy.name)
                            && let Some(pool) = persistent.storage_pool_mut()
                        {
                            let view = pool
                                .alloc_bytes(proxy.size, stride)
                                .map_err(|e| Error::Shader(e.to_string()))?;
                            GpuBuffer::Pooled(view)
                        } else {
                            let buf = persistent.pool.get_buf_with_stride(
                                device,
                                proxy.size,
                                proxy.name,
                                DataAccess::Scattered,
                                stride,
                                proxy.buffer_flags,
                            )?;
                            graph.clear_buffer(&buf, 0, proxy.size);
                            GpuBuffer::Owned(buf)
                        };
                        self.bind_map.insert_buf(proxy.id, gpu_buf, proxy.name);
                    }
                }
                ResourceProxy::Image(proxy) => {
                    if self.bind_map.get_image(proxy.id).is_none() {
                        let format = image_format_to_goldy(proxy.format);
                        let access = match bind_types.get(i) {
                            Some(BindType::Image(_)) => SpatialAccess::Direct,
                            _ => SpatialAccess::Interpolated,
                        };
                        let tex = persistent
                            .tex_pool
                            .acquire(
                                device,
                                proxy.width,
                                proxy.height,
                                format,
                                access,
                                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                            )
                            .map_err(|e| Error::Shader(e.to_string()))?;
                        self.bind_map
                            .insert_image(proxy.id, tex, "placeholder_image");
                    }
                }
            }
        }
        Ok(())
    }

    fn record_upload_buffer(
        &mut self,
        device: &Device,
        graph: &mut TaskGraph,
        buf_proxy: &BufferProxy,
        bytes: &[u8],
        persistent: &mut PersistentState,
    ) -> Result<()> {
        if let Some((GpuBuffer::Owned(existing), _)) = self.bind_map.get_buf(buf_proxy.id)
            && existing.size() >= bytes.len() as u64
            && existing.access() == DataAccess::Scattered
            && existing.flags() == buf_proxy.buffer_flags
        {
            graph.write_buffer(existing, 0, bytes.to_vec());
        } else {
            let buf = persistent.pool.get_buf_with_stride(
                device,
                buf_proxy.size,
                buf_proxy.name,
                DataAccess::Scattered,
                buf_proxy.element_stride,
                buf_proxy.buffer_flags,
            )?;
            graph.write_buffer(&buf, 0, bytes.to_vec());
            self.bind_map
                .insert_buf(buf_proxy.id, GpuBuffer::Owned(buf), buf_proxy.name);
        }
        Ok(())
    }

    fn record_upload_image(
        &mut self,
        device: &Device,
        graph: &mut TaskGraph,
        image_proxy: &ImageProxy,
        bytes: &[u8],
        persistent: &mut PersistentState,
    ) -> Result<()> {
        let format = image_format_to_goldy(image_proxy.format);
        let texture = persistent
            .tex_pool
            .acquire(
                device,
                image_proxy.width,
                image_proxy.height,
                format,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST,
            )
            .map_err(|e| Error::Shader(e.to_string()))?;
        graph
            .write_texture(&texture, bytes.to_vec())
            .map_err(|e| Error::Shader(e.to_string()))?;
        self.bind_map
            .insert_image(image_proxy.id, texture, "uploaded_image");
        Ok(())
    }

    fn record_write_image(
        &mut self,
        device: &Device,
        graph: &mut TaskGraph,
        image_proxy: &ImageProxy,
        x: u32,
        y: u32,
        image_data: &peniko::ImageData,
        persistent: &mut PersistentState,
    ) -> Result<()> {
        if self.bind_map.get_image(image_proxy.id).is_none() {
            let format = image_format_to_goldy(image_proxy.format);
            let tex = persistent
                .tex_pool
                .acquire(
                    device,
                    image_proxy.width,
                    image_proxy.height,
                    format,
                    SpatialAccess::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )
                .map_err(|e| Error::Shader(e.to_string()))?;
            self.bind_map
                .insert_image(image_proxy.id, tex, "write_image_target");
        }
        if let Some((tex, _)) = self.bind_map.get_image(image_proxy.id) {
            if image_data.data.is_empty() && image_data.width != 0 && image_data.height != 0 {
                return Err(Error::InvalidImage {
                    id: image_data.data.id(),
                    reason: "image has non-zero dimensions but no pixel data; \
                             it may have been registered to a different renderer \
                             or unregistered before this render was submitted",
                });
            }
            let bytes = image_data.data.data();
            graph
                .write_texture_region(
                    tex,
                    x,
                    y,
                    image_data.width,
                    image_data.height,
                    bytes.to_vec(),
                )
                .map_err(|e| Error::Shader(e.to_string()))?;
        }
        Ok(())
    }

    fn record_clear(
        &mut self,
        device: &Device,
        graph: &mut TaskGraph,
        buf_proxy: &BufferProxy,
        off: u64,
        sz: Option<u64>,
        persistent: &mut PersistentState,
    ) -> Result<()> {
        if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
            let clear_size = sz.unwrap_or_else(|| gpu_buf.size() - off);
            match gpu_buf {
                GpuBuffer::Owned(buf) => graph.clear_buffer(buf, off, clear_size),
                GpuBuffer::Pooled(view) => graph.clear_buffer_view(view, off, clear_size),
            }
        } else {
            let buf = persistent.pool.get_buf_with_stride(
                device,
                buf_proxy.size,
                buf_proxy.name,
                DataAccess::Scattered,
                buf_proxy.element_stride,
                buf_proxy.buffer_flags,
            )?;
            let clear_size = sz.unwrap_or_else(|| buf.size() - off);
            graph.clear_buffer(&buf, off, clear_size);
            self.bind_map
                .insert_buf(buf_proxy.id, GpuBuffer::Owned(buf), buf_proxy.name);
        }
        Ok(())
    }

    fn bind_graph_resources<'a>(
        &self,
        mut node: NodeBuilder<'a>,
        bindings: &[ResourceProxy],
        bind_types: &[BindType],
    ) -> NodeBuilder<'a> {
        for (i, res) in bindings.iter().enumerate() {
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

            match res {
                ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                    if let Some((gpu_buf, _)) = self.bind_map.get_buf(proxy.id) {
                        match gpu_buf {
                            GpuBuffer::Owned(buf) => {
                                node = node.bind_buffer(buf, access);
                            }
                            GpuBuffer::Pooled(view) => {
                                node = node.bind_buffer_view(view, access);
                            }
                        }
                    }
                }
                ResourceProxy::Image(proxy) => {
                    if let Some((tex, _)) = self.bind_map.get_image(proxy.id) {
                        node = node.bind_texture(tex, access);
                    }
                }
            }
        }
        node
    }

    fn release_pool(&mut self, device: &Device, persistent: &mut PersistentState) -> Result<()> {
        self.wait_until_gpu_idle(device, persistent)?;
        self.bind_map.buf_map.clear();
        self.downloads.clear();
        persistent.storage_ring.clear();
        Ok(())
    }

    #[allow(
        clippy::print_stdout,
        reason = "dump_dispatch prints manifest paths to stdout for debugging when dump is enabled"
    )]
    fn dump_dispatch(
        &self,
        device: &Device,
        dispatch_idx: usize,
        shader_id: ShaderId,
        dims: (u32, u32, u32),
        bindings: &[ResourceProxy],
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

        for (i, res) in bindings.iter().enumerate() {
            match res {
                ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                    if let Some((gpu_buf, name)) = self.bind_map.get_buf(proxy.id) {
                        let size = gpu_buf.size() as usize;
                        wln!(
                            "binding[{i}]: buf name={name} size={size} bindless={}",
                            gpu_buf.bindless_index().unwrap_or(u32::MAX)
                        );

                        let mut data = vec![0_u8; size];
                        let ok = match gpu_buf {
                            GpuBuffer::Owned(buf) => buf.read_to_cpu(device, &mut data).is_ok(),
                            GpuBuffer::Pooled(view) => view.read_to_cpu(device, &mut data).is_ok(),
                        };
                        if ok {
                            std::fs::write(format!("{dir}/buf_{i}.bin"), &data).ok();
                        } else {
                            wln!("  (read failed)");
                        }
                    }
                }
                ResourceProxy::Image(proxy) => {
                    wln!(
                        "binding[{i}]: image {}x{} id={}",
                        proxy.width,
                        proxy.height,
                        proxy.id.0
                    );
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
}

// -----------------------------------------------------------------------
// FrameRecorder — direct-execution recorder that builds TaskGraph nodes
// -----------------------------------------------------------------------

pub(crate) struct FrameRecorder<'a> {
    device: &'a Device,
    graph: TaskGraph,
    frame: &'a mut FrameState,
    persistent: &'a mut PersistentState,
    shaders: &'a [GoldyShader],
    force_uav: bool,
    surface_frame: Option<&'a Frame>,
    last_timeline: Option<TimelineValue>,
    pending_downloads: Vec<BufferProxy>,
    deferred_free_buffers: Vec<ResourceId>,
    deferred_free_images: Vec<ImageProxy>,
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
        let mut graph = TaskGraph::new();

        if persistent.storage_ring.take_clear_flag()
            && let Some(pool) = persistent.storage_ring.current()
        {
            graph.clear_buffer(pool.backing_buffer(), 0, pool.capacity());
        }

        Self {
            device,
            graph,
            frame,
            persistent,
            shaders,
            force_uav: fuav,
            surface_frame,
            last_timeline: None,
            pending_downloads: Vec::new(),
            deferred_free_buffers: Vec::new(),
            deferred_free_images: Vec::new(),
            dispatch_count: 0,
        }
    }

    #[cfg_attr(not(feature = "debug_layers"), allow(dead_code))]
    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> BufferProxy {
        let data = data.into();
        let buf_proxy = BufferProxy::new(data.len() as u64, name);
        self.frame
            .record_upload_buffer(self.device, &mut self.graph, &buf_proxy, &data, self.persistent)
            .expect("upload failed");
        buf_proxy
    }

    /// Like [`Self::upload`] but sets an explicit structured buffer element stride.
    ///
    /// Use this for raw byte uploads (e.g. scene data, LUT tables) where the
    /// GPU expects a specific `StructureByteStride`. The data length must be
    /// divisible by `element_stride`.
    pub fn upload_strided(
        &mut self,
        name: &'static str,
        element_stride: u32,
        data: impl Into<Vec<u8>>,
    ) -> BufferProxy {
        let data = data.into();
        let buf_proxy = BufferProxy::with_stride(data.len() as u64, name, element_stride);
        self.frame
            .record_upload_buffer(self.device, &mut self.graph, &buf_proxy, &data, self.persistent)
            .expect("upload_strided failed");
        buf_proxy
    }

    pub fn upload_typed<T: bytemuck::Pod>(&mut self, name: &'static str, data: &T) -> BufferProxy {
        let bytes = bytemuck::bytes_of(data).to_vec();
        let buf_proxy = BufferProxy::with_stride(bytes.len() as u64, name, size_of::<T>() as u32);
        self.frame
            .record_upload_buffer(
                self.device,
                &mut self.graph,
                &buf_proxy,
                &bytes,
                self.persistent,
            )
            .expect("upload_typed failed");
        buf_proxy
    }

    pub fn upload_image(
        &mut self,
        width: u32,
        height: u32,
        format: crate::resource_proxy::ImageFormat,
        data: impl Into<Vec<u8>>,
    ) -> ImageProxy {
        let data = data.into();
        let image_proxy = ImageProxy::new(width, height, format);
        self.frame
            .record_upload_image(
                self.device,
                &mut self.graph,
                &image_proxy,
                &data,
                self.persistent,
            )
            .expect("upload_image failed");
        image_proxy
    }

    pub fn write_image(&mut self, proxy: ImageProxy, x: u32, y: u32, image: peniko::ImageData) {
        self.frame
            .record_write_image(
                self.device,
                &mut self.graph,
                &proxy,
                x,
                y,
                &image,
                self.persistent,
            )
            .expect("write_image failed");
    }

    pub fn dispatch<R>(&mut self, shader: ShaderId, wg_size: (u32, u32, u32), resources: R)
    where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r: Vec<_> = resources.into_iter().map(|x| x.into()).collect();
        self.dispatch_inner(shader, wg_size, &r, &[]);
    }

    pub fn dispatch_with_push_tail<R>(
        &mut self,
        shader: ShaderId,
        wg_size: (u32, u32, u32),
        resources: R,
        push_tail: &[u32],
    ) where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r: Vec<_> = resources.into_iter().map(|x| x.into()).collect();
        self.dispatch_inner(shader, wg_size, &r, push_tail);
    }

    fn dispatch_inner(
        &mut self,
        shader_id: ShaderId,
        (x, y, z): (u32, u32, u32),
        bindings: &[ResourceProxy],
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
        let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
        self.frame
            .ensure_resources_materialized(
                self.device,
                &mut self.graph,
                bindings,
                &bind_types,
                self.persistent,
            )
            .expect("ensure_resources_materialized failed in dispatch");
        let indices =
            collect_bindless_indices(bindings, &bind_types, &self.frame.bind_map, self.force_uav)
                .expect("collect_bindless_indices failed in dispatch");

        if let Some(ref dir) = *DUMP_DIR {
            let mut debug_indices = indices.clone();
            debug_indices.extend_from_slice(push_tail);
            self.frame.dump_dispatch(
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
        node = self.frame.bind_graph_resources(node, bindings, &bind_types);
        if !indices.is_empty() || !push_tail.is_empty() {
            node = node.bind_resources_raw_with_user(&indices, push_tail);
        }
        node.dispatch(x, y, z);
        self.dispatch_count += 1;
    }

    pub fn dispatch_indirect<R>(
        &mut self,
        shader: ShaderId,
        buf: BufferProxy,
        offset: u64,
        resources: R,
    ) where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r: Vec<_> = resources.into_iter().map(|x| x.into()).collect();
        self.dispatch_indirect_inner(shader, &buf, offset, &r);
    }

    fn dispatch_indirect_inner(
        &mut self,
        shader_id: ShaderId,
        buf_proxy: &BufferProxy,
        offset: u64,
        bindings: &[ResourceProxy],
    ) {
        self.frame
            .ensure_resources_materialized(
                self.device,
                &mut self.graph,
                &[ResourceProxy::Buffer(*buf_proxy)],
                &[BindType::Buffer],
                self.persistent,
            )
            .expect("ensure_resources_materialized failed in dispatch_indirect (indirect buf)");
        let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
        self.frame
            .ensure_resources_materialized(
                self.device,
                &mut self.graph,
                bindings,
                &bind_types,
                self.persistent,
            )
            .expect("ensure_resources_materialized failed in dispatch_indirect");
        let indices =
            collect_bindless_indices(bindings, &bind_types, &self.frame.bind_map, self.force_uav)
                .expect("collect_bindless_indices failed in dispatch_indirect");

        let Some((gpu_buf, _)) = self.frame.bind_map.get_buf(buf_proxy.id) else {
            log::error!(
                "DispatchIndirect for shader {} skipped: buffer proxy (id={}) is \
                 either unregistered or pooled (must be an owned buffer)",
                shader_id.0,
                buf_proxy.id.0
            );
            return;
        };
        let Some(indirect_buf) = gpu_buf.as_owned() else {
            log::error!(
                "DispatchIndirect for shader {} skipped: buffer proxy (id={}) is pooled \
                 (must be an owned buffer)",
                shader_id.0,
                buf_proxy.id.0
            );
            return;
        };

        if let Some(ref dir) = *DUMP_DIR {
            let indirect_dims =
                GoldyRenderer::read_indirect_dims(self.device, indirect_buf, offset);
            self.frame.dump_dispatch(
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
        node = self.frame.bind_graph_resources(node, bindings, &bind_types);
        node = node.bind_buffer(indirect_buf, NodeAccess::Read);
        if !indices.is_empty() {
            node = node.bind_resources_raw(&indices);
        }
        node.dispatch_indirect(indirect_buf, offset);
        self.dispatch_count += 1;
    }

    pub fn clear_all(&mut self, buf: BufferProxy) {
        self.frame
            .record_clear(self.device, &mut self.graph, &buf, 0, None, self.persistent)
            .expect("clear_all failed");
    }

    pub fn download(&mut self, buf: BufferProxy) {
        self.pending_downloads.push(buf);
    }

    pub fn free_buffer(&mut self, buf: BufferProxy) {
        self.deferred_free_buffers.push(buf.id);
    }

    pub fn free_image(&mut self, image: ImageProxy) {
        self.deferred_free_images.push(image);
    }

    pub fn free_resource(&mut self, resource: ResourceProxy) {
        match resource {
            ResourceProxy::Buffer(buf) => self.free_buffer(buf),
            ResourceProxy::BufferRange {
                proxy,
                offset: _,
                size: _,
            } => self.free_buffer(proxy),
            ResourceProxy::Image(image) => self.free_image(image),
        }
    }

    /// Stub for debug-layer draw commands (not yet implemented in Goldy).
    #[cfg(feature = "debug_layers")]
    pub fn draw(&mut self, _params: crate::resource_proxy::DrawParams) {}

    /// Register the output image proxy → GPU texture mapping before dispatch
    /// begins. Used by the public render entry points.
    fn set_output_image(&mut self, proxy: &ImageProxy, texture: &Texture) {
        self.frame
            .bind_map
            .insert_image(proxy.id, texture.borrow(), "output");
    }

    /// Finish dispatch: flush the final graph and push a `FrameCleanup`
    /// entry onto the deque for deferred processing after GPU completion.
    fn finish(mut self, output_image_id: Option<ResourceId>) -> Result<()> {
        flush_graph(
            &mut self.graph,
            self.device,
            &mut self.last_timeline,
            self.surface_frame,
        )?;

        let timeline = match self.surface_frame {
            None => self.last_timeline.take(),
            Some(_) => None, // surface path: filled by note_frame_presented
        };

        self.frame.cleanup_ring.push_back(FrameCleanup {
            timeline,
            pending_downloads: self.pending_downloads,
            deferred_free_buffers: self.deferred_free_buffers,
            deferred_free_images: self.deferred_free_images.iter().map(|ip| ip.id).collect(),
            output_image_id,
        });

        Ok(())
    }
}

// -----------------------------------------------------------------------
// GoldyRenderer
// -----------------------------------------------------------------------

impl GoldyRenderer {
    /// Create a new renderer for the given device.
    pub fn new(device: &Device) -> Result<Self> {
        let mut renderer = Self {
            device: device.clone(),
            shaders: FullShaders::empty(),
            resolver: Resolver::new(),
            engine_shaders: Vec::new(),
            persistent: PersistentState {
                pool: ResourcePool::default(),
                storage_ring: BufferPoolRing::new(),
                tex_pool: TexturePool::default(),
                last_drained_bump: None,
            },
            frame: FrameState {
                bind_map: BindMap::default(),
                downloads: HashMap::new(),
                cleanup_ring: VecDeque::new(),
            },
            pool_size_history: VecDeque::with_capacity(POOL_SHRINK_WINDOW),
            persistent_bump: None,
        };
        let shaders = shaders::goldy_full_shaders(device, &mut renderer)?;
        renderer.shaders = shaders;
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

    /// Compute the pool shrink max: if the current pool is significantly
    /// oversized relative to recent demand, returns `Some(target)` to cap
    /// the pool and trigger a reallocation. Returns `None` if no shrink needed.
    // =======================================================================
    // Public API (unchanged signatures)
    // =======================================================================

    /// Acknowledge a swapchain frame after [`goldy::Frame::present`].
    ///
    /// Fills in the timeline on the most recent `FrameCleanup` entry (the one
    /// pushed by `FrameRecorder::finish` for the surface path where the
    /// timeline isn't known until after present).
    pub fn note_frame_presented(&mut self, tv: TimelineValue) {
        if let Some(back) = self.frame.cleanup_ring.back_mut()
            && back.timeline.is_none()
        {
            back.timeline = Some(tv);
        }
    }

    /// Render a scene to the given texture.
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
        let texture = Texture::new(
            device,
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
                self.frame.process_cleanup(device, entry, true, &mut self.persistent)?;
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
        let t_drain = t_drain_start.elapsed();

        // Flip storage pool index and evict stale pooled bind_map entries
        self.persistent.storage_ring.advance();
        self.frame.evict_pooled_entries();

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

        // Fix 2: Track pool size history and allow shrinking with hysteresis.
        self.pool_size_history.push_back(pool_size);
        if self.pool_size_history.len() > POOL_SHRINK_WINDOW {
            self.pool_size_history.pop_front();
        }
        // Inline pool_shrink_max to avoid a &self borrow that conflicts with the
        // self.resolver lifetime held by `ramps` and `images`.
        let shrink_max = {
            let current_capacity = self
                .persistent
                .storage_ring
                .current_capacity()
                .unwrap_or(0);
            if current_capacity == 0 || self.pool_size_history.len() < POOL_SHRINK_WINDOW {
                None
            } else {
                let rolling_max = self.pool_size_history.iter().copied().max().unwrap_or(0);
                let threshold = (rolling_max as f64 * POOL_SHRINK_THRESHOLD) as u64;
                if current_capacity > threshold {
                    let target = rolling_max.saturating_mul(3) / 2;
                    log::info!(
                        "Pool shrink: current={}MB, rolling_max={}MB, target={}MB",
                        current_capacity / 1024 / 1024,
                        rolling_max / 1024 / 1024,
                        target / 1024 / 1024,
                    );
                    Some(target)
                } else {
                    None
                }
            }
        };

        let t1 = Instant::now();
        self.persistent
            .prepare_storage_pool(device, pool_size, shrink_max)?;
        let t_pool = t1.elapsed();

        let t2 = Instant::now();
        let mut recorder = FrameRecorder::new(
            device,
            &mut self.frame,
            &mut self.persistent,
            &self.engine_shaders,
            surface_frame,
        );

        let mut render = Render::new();
        render.run_coarse(
            encoding,
            packed,
            ramps,
            images,
            &self.shaders,
            params,
            params.robust,
            &config,
            &mut recorder,
        );
        let out_image = render.out_image();
        let filter_layers = render.filter_layer_textures();
        let t_coarse = t2.elapsed();

        if let Some(tex) = output_texture {
            recorder.set_output_image(&out_image, tex);
        }

        let t3 = Instant::now();
        render.record_fine(encoding, &self.shaders, &mut recorder);
        render::record_filter_effects(
            encoding,
            &self.shaders,
            &mut recorder,
            params.width,
            params.height,
            &filter_layers,
            out_image,
        );
        let t_fine_record = t3.elapsed();

        #[cfg(feature = "debug_layers")]
        if let Some(captured) = render.take_captured_buffers() {
            captured.release_buffers(&mut recorder);
        }

        let t4 = Instant::now();
        recorder.finish(Some(out_image.id))?;
        let t_submit = t4.elapsed();

        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        let label = if surface_frame.is_some() {
            "surface"
        } else {
            ""
        };
        log::debug!(
            "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms {label}",
            frame_num,
            t_drain.as_secs_f64() * 1000.0,
            t_resolve.as_secs_f64() * 1000.0,
            t_pool.as_secs_f64() * 1000.0,
            t_coarse.as_secs_f64() * 1000.0,
            t_fine_record.as_secs_f64() * 1000.0,
            t_submit.as_secs_f64() * 1000.0,
            frame_start.elapsed().as_secs_f64() * 1000.0,
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

fn bumps_buf_static_name() -> &'static str {
    "ekrano.bump_buf"
}

fn flush_graph(
    graph: &mut TaskGraph,
    device: &Device,
    last_timeline: &mut Option<TimelineValue>,
    surface_frame: Option<&Frame>,
) -> Result<()> {
    if graph.is_empty() {
        return Ok(());
    }
    match surface_frame {
        None => {
            let tv = device
                .submit(graph)
                .map_err(|e| Error::Shader(e.to_string()))?;
            *last_timeline = Some(tv);
        }
        Some(frame) => {
            frame
                .submit_compute(graph)
                .map_err(|e| Error::Shader(e.to_string()))?;
        }
    }
    *graph = TaskGraph::new();
    Ok(())
}

fn image_format_to_goldy(format: crate::resource_proxy::ImageFormat) -> TextureFormat {
    match format {
        crate::resource_proxy::ImageFormat::Rgba8 => TextureFormat::Rgba8Unorm,
        crate::resource_proxy::ImageFormat::Bgra8 => TextureFormat::Bgra8Unorm,
    }
}

fn bind_type_to_node_access(bt: BindType) -> NodeAccess {
    match bt {
        BindType::Buffer | BindType::Image(_) => NodeAccess::ReadWrite,
        BindType::BufReadOnly | BindType::Uniform | BindType::ImageRead(_) => NodeAccess::Read,
    }
}

impl BindMap {
    fn insert_buf(&mut self, id: ResourceId, gpu_buf: GpuBuffer, name: &'static str) {
        self.buf_map.insert(id, (gpu_buf, name));
    }

    fn get_buf(&self, id: ResourceId) -> Option<&(GpuBuffer, &'static str)> {
        self.buf_map.get(&id)
    }

    fn remove_buf(&mut self, id: ResourceId) {
        self.buf_map.remove(&id);
    }

    fn insert_image(&mut self, id: ResourceId, tex: Texture, name: &'static str) {
        self.image_map.insert(id, (tex, name));
    }

    fn get_image(&self, id: ResourceId) -> Option<&(Texture, &'static str)> {
        self.image_map.get(&id)
    }

    fn remove_image(&mut self, id: ResourceId) {
        self.image_map.remove(&id);
    }

    fn take_image(&mut self, id: ResourceId) -> Option<(Texture, &'static str)> {
        self.image_map.remove(&id)
    }
}

impl ResourcePool {
    fn get_buf_with_stride(
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
        Buffer::new_with_stride_and_flags(device, size, access, stride, buffer_flags)
            .map_err(|e| Error::Shader(e.to_string()))
    }
}

fn collect_bindless_indices(
    resources: &[ResourceProxy],
    bind_types: &[BindType],
    bind_map: &BindMap,
    all_uav: bool,
) -> Result<Vec<u32>, Error> {
    let mut indices = Vec::with_capacity(resources.len());
    for (i, res) in resources.iter().enumerate() {
        let is_read_only = !all_uav && matches!(bind_types.get(i), Some(BindType::BufReadOnly));
        let idx = match res {
            ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                let (buf, _) = bind_map
                    .get_buf(proxy.id)
                    .ok_or_else(|| Error::Shader("buffer not found".into()))?;
                if is_read_only {
                    buf.bindless_srv_index()
                        .ok_or_else(|| Error::Shader("buffer has no SRV index".into()))?
                } else {
                    buf.bindless_index()
                        .ok_or_else(|| Error::Shader("buffer has no bindless index".into()))?
                }
            }
            ResourceProxy::Image(proxy) => {
                let entry = bind_map.get_image(proxy.id);
                match entry {
                    Some((tex, name)) => tex.bindless_index().ok_or_else(|| {
                        Error::Shader(format!(
                            "image '{}' (id={}) exists but has no bindless index",
                            name, proxy.id.0
                        ))
                    })?,
                    None => {
                        return Err(Error::Shader(format!(
                            "image not found in bind map (id={}, {}x{})",
                            proxy.id.0, proxy.width, proxy.height
                        )));
                    }
                }
            }
        };
        indices.push(idx);
    }
    if indices.len() > MAX_BINDLESS_SLOTS {
        return Err(Error::Shader(format!(
            "shader requires {} bindless slots, exceeds limit of {}",
            indices.len(),
            MAX_BINDLESS_SLOTS
        )));
    }
    Ok(indices)
}
