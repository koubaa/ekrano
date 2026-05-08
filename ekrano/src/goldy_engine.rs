// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy-based engine for executing Ekrano recordings.
//!
//! This engine replaces the wgpu backend when the `goldy` feature is enabled.
//! It uses Slang shaders compiled for Goldy's bindless model.
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
use std::sync::LazyLock;

use goldy::task_graph::{NodeAccess, NodeBuilder};
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags, TextureFormat};
use goldy::{
    Buffer, BufferPool, BufferView, ComputePipeline, DataAccess, Device, DeviceType, Frame,
    ShaderModule, TaskGraph, Texture, TexturePool, TimelineValue,
};

static DUMP_DIR: LazyLock<Option<String>> = LazyLock::new(|| std::env::var("EKRANO_DUMP_DIR").ok());

use mem::size_of;

use crate::{
    Error, Result,
    low_level::{BufferProxy, Command, ImageProxy, Recording, ResourceId, ResourceProxy, ShaderId},
    recording::BindType,
};

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

/// Goldy-based recording executor.
#[derive(Default)]
pub struct GoldyEngine {
    shaders: Vec<GoldyShader>,
    pool: ResourcePool,
    bind_map: BindMap,
    /// Buffers that were downloaded (`Command::Download`); keyed by `ResourceId`.
    downloads: HashMap<ResourceId, Vec<u8>>,
    /// Single large buffer pool for storage buffers (Phase 3c). None until `prepare_storage_pool`.
    storage_pool: Option<BufferPool>,
    /// When set, the next non-empty graph flush prepends a full clear of the pool backing buffer.
    pool_clear_in_next_submit: bool,
    /// GPU timeline from the previous `run_recording` / `after_surface_present` (not yet drained).
    prev_frame_timeline: Option<TimelineValue>,
    /// Set after [`GoldyEngine::run_recording_to_frame`] until [`GoldyEngine::after_surface_present`].
    surface_frame_in_flight: bool,
    prev_pending_downloads: Vec<BufferProxy>,
    prev_deferred_free_buffers: Vec<ResourceId>,
    prev_deferred_free_images: Vec<ResourceId>,
    prev_output_image_id: Option<ResourceId>,
    /// Bump allocator values read after the **previous** submit's GPU work completed.
    last_drained_bump: Option<ekrano_encoding::BumpAllocators>,
    /// Reuses transient textures across frames (important on DX12).
    tex_pool: TexturePool,
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
        "vello.bump_buf"
            | "vello.indirect_dispatch"
            | "vello.tile_buf"
            | "vello.lines_buf"
            | "vello.seg_counts_buf"
            | "vello.segments_buf"
            | "vello.path_buf"
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

impl GoldyEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Waits on the GPU work from the last [`GoldyEngine::run_recording`] submission and fills
    /// [`GoldyEngine::get_download`] / [`GoldyEngine::take_last_drained_bump`].
    ///
    /// Call after `run_recording` when you need readback from that submission immediately
    /// (e.g. same-frame bump overflow checks). Omit between application frames so the next
    /// `run_recording` drains completed work at the start (**pipelined** steady state).
    pub fn finish_frame_for_readback(&mut self, device: &Device) -> Result<()> {
        self.assert_no_surface_frame_in_flight()?;
        self.drain_completed_submit(device)?;
        Ok(())
    }

    /// Wait for any in-flight queued submission (`prev_frame_timeline`).
    ///
    /// Use before `clear_transients`, screenshot readback paths, etc.
    pub fn wait_until_gpu_idle(&mut self, device: &Device) -> Result<()> {
        self.assert_no_surface_frame_in_flight()?;
        self.drain_completed_submit(device)
    }

    fn drain_completed_submit(&mut self, device: &Device) -> Result<()> {
        let Some(tv) = self.prev_frame_timeline.take() else {
            return Ok(());
        };
        // Wait for GPU work to complete. `wait_until` may return `Err` when a
        // command buffer terminates with a GPU fault
        // (kIOGPUCommandBufferCallbackErrorPageFault or similar).
        //
        // We intentionally do NOT early-return on that error: a faulted command
        // buffer has *terminated* — its argument-buffer descriptors are no longer
        // live — so it is safe to recycle every bindless slot and remove every
        // bind-map entry.  Bailing out early instead causes `prev_deferred_free_buffers`
        // to accumulate across faulted frames; the next successful `run_recording`
        // then replaces the list without freeing it, permanently leaking those
        // entries and exhausting the 64-slot storage-buffer window within 2–3 frames
        // (observed panic: "storage-buffer bindless slots exhausted (64 max)").
        let wait_result = device
            .wait_until(tv)
            .map_err(|e| Error::Shader(e.to_string()));

        // Downloads are only valid when GPU work completed successfully.
        if wait_result.is_ok() {
            let pending = mem::take(&mut self.prev_pending_downloads);
            let bump_name = Self::bumps_buf_static_name();

            for buf_proxy in pending {
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
                            self.last_drained_bump = Some(bytemuck::pod_read_unaligned(data));
                        }
                        self.downloads.remove(&buf_proxy.id);
                    }
                }
            }
        } else {
            // Discard pending downloads — their GPU data is corrupt after a fault.
            self.prev_pending_downloads.clear();
        }

        // Always free bind-map resources regardless of success/error so that
        // bindless slots are recycled and GPU memory is not leaked.
        for id in self.prev_deferred_free_buffers.drain(..) {
            self.bind_map.remove_buf(id);
        }
        for id in self.prev_deferred_free_images.drain(..) {
            if let Some((tex, _)) = self.bind_map.take_image(id) {
                self.tex_pool.release(tex);
            }
        }
        if let Some(id) = self.prev_output_image_id.take() {
            // Output texture is a borrow (`owned=false`); remove drops the borrow only.
            self.bind_map.remove_image(id);
        }

        wait_result
    }

    fn assert_no_surface_frame_in_flight(&self) -> Result<()> {
        if self.surface_frame_in_flight {
            return Err(Error::Shader(
                "GoldyEngine: swapchain frame still open — call frame.present() then GoldyEngine::after_surface_present(tv) before more GPU work."
                    .into(),
            ));
        }
        Ok(())
    }

    /// After [`GoldyEngine::run_recording_to_frame`], call [`goldy::Frame::present`] then pass the
    /// returned [`TimelineValue`] here before the next recording or acquiring another frame.
    pub fn after_surface_present(&mut self, tv: TimelineValue) -> Result<()> {
        if !self.surface_frame_in_flight {
            return Err(Error::Shader(
                "GoldyEngine::after_surface_present: no surface frame in flight (forgot run_recording_to_frame, or called twice)".into(),
            ));
        }
        self.surface_frame_in_flight = false;
        self.prev_frame_timeline = Some(tv);
        Ok(())
    }

    fn bumps_buf_static_name() -> &'static str {
        "vello.bump_buf"
    }

    /// Ensure all resources in bindings are materialized (cf. wgpu lazy materialization).
    /// Buffers that are only written by a shader are never Upload/Clear'd; images like
    /// `gradient_image` or `image_atlas` may be 1x1 placeholders never uploaded.
    /// For images: Image (read-write) needs Direct/UAV; `ImageRead` needs Interpolated/SRV.
    ///
    /// Fresh storage buffers are zeroed via a `ClearBuffer` graph node (batched with dispatches).
    fn ensure_resources_materialized(
        &mut self,
        device: &Device,
        graph: &mut TaskGraph,
        bindings: &[ResourceProxy],
        bind_types: &[BindType],
    ) -> Result<()> {
        for (i, res) in bindings.iter().enumerate() {
            match res {
                ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                    if self.bind_map.get_buf(proxy.id).is_none() {
                        let stride = proxy
                            .element_stride
                            .or_else(|| element_stride_for_buffer(proxy.name));
                        let gpu_buf = if !is_pool_exempt(proxy.name)
                            && let Some(pool) = self.storage_pool.as_mut()
                        {
                            let view = pool
                                .alloc_bytes(proxy.size, stride)
                                .map_err(|e| Error::Shader(e.to_string()))?;
                            GpuBuffer::Pooled(view)
                        } else {
                            let buf = self.pool.get_buf_with_stride(
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
                        // Image = read-write (RWTexture2D) needs Direct. ImageRead = read-only (Texture2D) needs Interpolated.
                        let access = match bind_types.get(i) {
                            Some(BindType::Image(_)) => SpatialAccess::Direct,
                            _ => SpatialAccess::Interpolated,
                        };
                        let tex = self
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

    /// Add a compute shader from Slang source.
    pub fn add_compute_shader(
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
    ///
    /// Use `OptimizationLevel::None` for shaders that hit driver bugs on
    /// software renderers (e.g. lavapipe SSA corruption across barriers).
    pub fn add_compute_shader_with_options(
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

        let id = ShaderId(self.shaders.len());
        self.shaders.push(GoldyShader {
            pipeline,
            bindings: bindings.to_vec(),
        });
        Ok(id)
    }

    /// Execute a recording.
    ///
    /// `output` maps the recording's output image proxy to the actual texture to render into.
    /// The caller gets both from `render::render_full()` which returns `(Recording, target)`.
    ///
    /// Dispatches, buffer clears, and buffer writes are accumulated into a
    /// [`TaskGraph`] which analyzes resource dependencies and inserts per-resource
    /// barriers, replacing the previous per-dispatch command buffer submission pattern.
    ///
    /// Graph submission is deferred until a flush-triggering operation (e.g.
    /// `WriteImage`) or the end of the recording. Completed GPU work may be drained
    /// at the **start** of the next [`GoldyEngine::run_recording`] (pipelining) unless the caller
    /// uses [`GoldyEngine::finish_frame_for_readback`].
    pub fn run_recording(
        &mut self,
        device: &Device,
        recording: &Recording,
        output: Option<(&ImageProxy, &Texture)>,
        label: &'static str,
    ) -> Result<()> {
        self.run_recording_impl(device, recording, output, label, None)
    }

    /// Record to a swapchain [`Frame`] using [`Frame::submit_compute`] (not standalone [`Device::submit`]).
    ///
    /// After this returns, call [`goldy::Frame::present`] on the frame, then [`GoldyEngine::after_surface_present`]
    /// with the returned timeline value before acquiring another frame or calling [`GoldyEngine::run_recording`].
    pub fn run_recording_to_frame(
        &mut self,
        device: &Device,
        recording: &Recording,
        output_image: &ImageProxy,
        frame: &Frame,
        label: &'static str,
    ) -> Result<()> {
        let output = Some((output_image, frame.texture()));
        self.run_recording_impl(device, recording, output, label, Some(frame))
    }

    fn run_recording_impl(
        &mut self,
        device: &Device,
        recording: &Recording,
        output: Option<(&ImageProxy, &Texture)>,
        label: &'static str,
        surface_frame: Option<&Frame>,
    ) -> Result<()> {
        self.assert_no_surface_frame_in_flight()?;
        self.drain_completed_submit(device)?;

        let mut graph = TaskGraph::new();

        // Pool clear: zero-fill the backing buffer as a first-class graph node so the
        // analyzer inserts the correct barrier between it and downstream pool-view dispatches.
        if self.pool_clear_in_next_submit {
            self.pool_clear_in_next_submit = false;
            if let Some(pool) = &self.storage_pool {
                graph.clear_buffer(pool.backing_buffer(), 0, pool.capacity());
            }
        }

        let mut last_timeline: Option<TimelineValue> = None;
        let mut pending_downloads: Vec<BufferProxy> = Vec::new();
        let mut deferred_free_buffers: Vec<ResourceId> = Vec::new();
        let mut deferred_free_images: Vec<ResourceId> = Vec::new();
        let mut dispatch_count: usize = 0;

        let output_image_id = output.map(|(proxy, tex)| {
            self.bind_map.insert_image(proxy.id, tex.borrow(), "output");
            proxy.id
        });

        for command in &recording.commands {
            match command {
                Command::Upload(buf_proxy, bytes) => {
                    // Reuse an existing buffer if it is large enough and has the same type.
                    // The write_buffer node carries NodeAccess::Write, so the analyzer inserts
                    // the necessary barrier between any prior reader (WAR) and this write.
                    if let Some((GpuBuffer::Owned(existing), _)) =
                        self.bind_map.get_buf(buf_proxy.id)
                        && existing.size() >= bytes.len() as u64
                        && existing.access() == DataAccess::Scattered
                        && existing.flags() == buf_proxy.buffer_flags
                    {
                        graph.write_buffer(existing, 0, bytes.to_vec());
                    } else {
                        let stride = buf_proxy
                            .element_stride
                            .or_else(|| element_stride_for_buffer(buf_proxy.name));
                        let buf = self.pool.get_buf_with_stride(
                            device,
                            buf_proxy.size,
                            buf_proxy.name,
                            DataAccess::Scattered,
                            stride,
                            buf_proxy.buffer_flags,
                        )?;
                        graph.write_buffer(&buf, 0, bytes.to_vec());
                        self.bind_map.insert_buf(
                            buf_proxy.id,
                            GpuBuffer::Owned(buf),
                            buf_proxy.name,
                        );
                    }
                }
                Command::UploadUniform(buf_proxy, bytes) => {
                    let needs_flush_before_uniform = matches!(
                        self.bind_map.get_buf(buf_proxy.id),
                        Some((GpuBuffer::Owned(b), _))
                            if b.size() >= bytes.len() as u64
                                && b.access() == DataAccess::Broadcast,
                    );
                    if needs_flush_before_uniform {
                        self.flush_graph(&mut graph, device, &mut last_timeline, surface_frame)?;
                    }
                    if let Some((GpuBuffer::Owned(existing), _)) =
                        self.bind_map.get_buf(buf_proxy.id)
                        && existing.size() >= bytes.len() as u64
                        && existing.access() == DataAccess::Broadcast
                    {
                        existing
                            .write(0, bytes)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                    } else {
                        let buf = self.pool.get_buf(
                            device,
                            buf_proxy.size,
                            buf_proxy.name,
                            DataAccess::Broadcast,
                        )?;
                        buf.write(0, bytes)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                        self.bind_map.insert_buf(
                            buf_proxy.id,
                            GpuBuffer::Owned(buf),
                            buf_proxy.name,
                        );
                    }
                }
                Command::UploadImage(image_proxy, bytes) => {
                    let format = image_format_to_goldy(image_proxy.format);
                    let texture = self
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
                }
                Command::WriteImage(image_proxy, [x, y], image_data) => {
                    if self.bind_map.get_image(image_proxy.id).is_none() {
                        let format = image_format_to_goldy(image_proxy.format);
                        let tex = self
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
                        if image_data.data.is_empty()
                            && image_data.width != 0
                            && image_data.height != 0
                        {
                            panic!(
                                "Tried to draw an invalid empty image (id: {}). \
                                Maybe it was registered to a different renderer, or \
                                unregistered before this render was submitted.",
                                image_data.data.id()
                            );
                        }
                        let bytes = image_data.data.data();
                        graph
                            .write_texture_region(
                                tex,
                                *x,
                                *y,
                                image_data.width,
                                image_data.height,
                                bytes.to_vec(),
                            )
                            .map_err(|e| Error::Shader(e.to_string()))?;
                    }
                }
                Command::Download(buf_proxy) => {
                    pending_downloads.push(*buf_proxy);
                }
                Command::Clear(buf_proxy, off, sz) => {
                    let off = *off;
                    let sz = sz.as_ref().copied();
                    // No flush needed — clear is a proper graph node; the analyzer inserts
                    // barriers relative to any prior or subsequent accesses on the same buffer.
                    if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
                        let clear_size = sz.unwrap_or_else(|| gpu_buf.size() - off);
                        match gpu_buf {
                            GpuBuffer::Owned(buf) => {
                                graph.clear_buffer(buf, off, clear_size);
                            }
                            GpuBuffer::Pooled(view) => {
                                graph.clear_buffer_view(view, off, clear_size);
                            }
                        }
                    } else {
                        let stride = buf_proxy
                            .element_stride
                            .or_else(|| element_stride_for_buffer(buf_proxy.name));
                        let buf = self.pool.get_buf_with_stride(
                            device,
                            buf_proxy.size,
                            buf_proxy.name,
                            DataAccess::Scattered,
                            stride,
                            buf_proxy.buffer_flags,
                        )?;
                        let clear_size = sz.unwrap_or_else(|| buf.size() - off);
                        graph.clear_buffer(&buf, off, clear_size);
                        self.bind_map.insert_buf(
                            buf_proxy.id,
                            GpuBuffer::Owned(buf),
                            buf_proxy.name,
                        );
                    }
                }
                Command::FreeBuffer(buf_proxy) => {
                    deferred_free_buffers.push(buf_proxy.id);
                }
                Command::FreeImage(image_proxy) => {
                    deferred_free_images.push(image_proxy.id);
                }
                Command::Dispatch(shader_id, (x, y, z), bindings, push_tail) => {
                    if *x == 0 || *y == 0 || *z == 0 {
                        continue;
                    }
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, &mut graph, bindings, &bind_types)?;
                    let indices = collect_bindless_indices(
                        bindings,
                        &bind_types,
                        &self.bind_map,
                        force_uav(device),
                    )?;

                    if let Some(ref dir) = *DUMP_DIR {
                        let mut debug_indices = indices.clone();
                        debug_indices.extend_from_slice(push_tail);
                        self.dump_dispatch(
                            device,
                            dispatch_count,
                            *shader_id,
                            (*x, *y, *z),
                            bindings,
                            &debug_indices,
                            dir,
                        );
                    }

                    let mut node = graph.node("dispatch", &self.shaders[shader_id.0].pipeline);
                    node = self.bind_graph_resources(node, bindings, &bind_types);
                    if !indices.is_empty() || !push_tail.is_empty() {
                        node = node.bind_resources_raw_with_user(&indices, push_tail);
                    }
                    node.dispatch(*x, *y, *z);
                    dispatch_count += 1;
                }
                Command::DispatchIndirect(shader_id, buf_proxy, offset, bindings) => {
                    self.ensure_resources_materialized(
                        device,
                        &mut graph,
                        &[ResourceProxy::Buffer(*buf_proxy)],
                        &[BindType::Buffer],
                    )?;
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, &mut graph, bindings, &bind_types)?;
                    let indices = collect_bindless_indices(
                        bindings,
                        &bind_types,
                        &self.bind_map,
                        force_uav(device),
                    )?;
                    if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id)
                        && let Some(indirect_buf) = gpu_buf.as_owned()
                    {
                        if let Some(ref dir) = *DUMP_DIR {
                            let mut indirect_dims = [0_u32; 3];
                            let mut raw = [0_u8; 12];
                            if indirect_buf.read_to_cpu(device, &mut raw).is_ok() {
                                let off = *offset as usize;
                                if off + 12 <= raw.len() {
                                    for i in 0..3 {
                                        indirect_dims[i] = u32::from_le_bytes([
                                            raw[off + i * 4],
                                            raw[off + i * 4 + 1],
                                            raw[off + i * 4 + 2],
                                            raw[off + i * 4 + 3],
                                        ]);
                                    }
                                }
                            } else {
                                let full_size = indirect_buf.size() as usize;
                                let mut full = vec![0_u8; full_size];
                                if indirect_buf.read_to_cpu(device, &mut full).is_ok() {
                                    let off = *offset as usize;
                                    if off + 12 <= full.len() {
                                        for i in 0..3 {
                                            indirect_dims[i] = u32::from_le_bytes([
                                                full[off + i * 4],
                                                full[off + i * 4 + 1],
                                                full[off + i * 4 + 2],
                                                full[off + i * 4 + 3],
                                            ]);
                                        }
                                    }
                                }
                            }
                            self.dump_dispatch(
                                device,
                                dispatch_count,
                                *shader_id,
                                (indirect_dims[0], indirect_dims[1], indirect_dims[2]),
                                bindings,
                                &indices,
                                dir,
                            );
                        }

                        let mut node =
                            graph.node("dispatch_indirect", &self.shaders[shader_id.0].pipeline);
                        node = self.bind_graph_resources(node, bindings, &bind_types);
                        node = node.bind_buffer(indirect_buf, NodeAccess::Read);
                        if !indices.is_empty() {
                            node = node.bind_resources_raw(&indices);
                        }
                        node.dispatch_indirect(indirect_buf, *offset);
                        dispatch_count += 1;
                    }
                }
                #[cfg(feature = "debug_layers")]
                Command::Draw(_) => {}
            }
        }

        self.flush_graph(&mut graph, device, &mut last_timeline, surface_frame)?;

        match surface_frame {
            None => {
                self.prev_frame_timeline = last_timeline.take();
                self.surface_frame_in_flight = false;
            }
            Some(_) => {
                self.surface_frame_in_flight = true;
            }
        }

        self.prev_pending_downloads = pending_downloads;
        self.prev_deferred_free_buffers = deferred_free_buffers;
        self.prev_deferred_free_images = deferred_free_images;
        self.prev_output_image_id = output_image_id;

        log::trace!(
            "run_recording end ({label}): queued GPU submission; frees deferred to next drain"
        );

        Ok(())
    }

    fn flush_graph(
        &mut self,
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

    /// Bind a dispatch's resources to a graph node for dependency tracking.
    ///
    /// For each `ResourceProxy` in `bindings`, looks up the corresponding
    /// `GpuBuffer` or `Texture` in the bind map and registers it on the
    /// node with the appropriate [`NodeAccess`] derived from `bind_types`.
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
                .unwrap_or(NodeAccess::ReadWrite);

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

    /// Get downloaded buffer data (after GPU completion drained at the next
    /// [`GoldyEngine::run_recording`] or via [`GoldyEngine::finish_frame_for_readback`]).
    #[allow(
        dead_code,
        reason = "public readback API; not used by current goldy integration path"
    )]
    pub fn get_download(&self, buf: BufferProxy) -> Option<&[u8]> {
        self.downloads.get(&buf.id).map(|v| v.as_slice())
    }

    /// Reads [`ekrano_encoding::BumpAllocators`] from [`GoldyEngine::get_download`]; call after [`GoldyEngine::finish_frame_for_readback`]
    /// (or rely on draining at the start of the next [`GoldyEngine::run_recording`]).
    #[allow(
        dead_code,
        reason = "public readback API; not used by current goldy integration path"
    )]
    pub fn read_bump_allocators(
        &self,
        proxy: &BufferProxy,
    ) -> Result<ekrano_encoding::BumpAllocators> {
        let data = self
            .get_download(*proxy)
            .ok_or_else(|| Error::Shader("bump buffer download not available".into()))?;
        Ok(bytemuck::pod_read_unaligned(data))
    }

    /// Peek at bump allocator values drained after GPU completion, without consuming.
    pub fn last_drained_bump(&self) -> Option<&ekrano_encoding::BumpAllocators> {
        self.last_drained_bump.as_ref()
    }

    /// Consume bump allocator values drained after GPU completion (`vello.bump_buf` download).
    pub fn take_last_drained_bump(&mut self) -> Option<ekrano_encoding::BumpAllocators> {
        self.last_drained_bump.take()
    }

    /// Prepare the storage pool for a recording. Creates or resets the pool with the given size.
    /// Call before `run_recording` when the pipeline will use pooled buffers.
    pub fn prepare_storage_pool(&mut self, device: &Device, pool_size: u64) -> Result<()> {
        if !use_pool(device) {
            return Ok(());
        }
        let need_new = match &self.storage_pool {
            Some(pool) => pool.capacity() < pool_size,
            None => true,
        };
        if need_new {
            // Drop the old pool first so its backing buffer refcount hits
            // zero before we touch the heap allocator. Then ask the backend
            // to right-size the primary heap and release overflow heaps:
            // without this, each pool growth event (retry cascade or
            // first-NON-EMPTY-frame) leaves behind a `size * 2` overflow
            // heap that nothing reclaims, stacking up to >1 GB extra GPU
            // memory for a complex Lottie scene. `reset_buffer_heaps`
            // blocks internally until in-flight GPU work finishes so it is
            // safe to call here even though we may be mid-frame.
            self.storage_pool = None;
            device.reset_buffer_heaps();
            let pool =
                BufferPool::new(device, pool_size).map_err(|e| Error::Shader(e.to_string()))?;
            self.storage_pool = Some(pool);
        } else {
            let pool = self.storage_pool.as_mut().unwrap();
            pool.reset();
        }
        self.pool_clear_in_next_submit = true;
        Ok(())
    }

    /// Clear all transient resources (buffers, images, downloads) between retry attempts.
    /// Drops the storage pool so the next `prepare_storage_pool` allocates fresh.
    #[allow(
        dead_code,
        reason = "full reset between retries; integration uses release_pool / wait paths"
    )]
    pub fn clear_transients(&mut self, device: &Device) -> Result<()> {
        self.wait_until_gpu_idle(device)?;
        self.pool_clear_in_next_submit = false;
        self.prev_output_image_id = None;
        self.last_drained_bump = None;
        self.bind_map.buf_map.clear();
        self.bind_map.image_map.clear();
        self.downloads.clear();
        self.storage_pool = None;
        Ok(())
    }

    /// Release the storage pool and transient buffers to free GPU memory.
    /// Keeps images intact (caller may still need textures for readback).
    pub fn release_pool(&mut self, device: &Device) -> Result<()> {
        self.wait_until_gpu_idle(device)?;
        self.bind_map.buf_map.clear();
        self.downloads.clear();
        self.storage_pool = None;
        Ok(())
    }

    /// Dump all buffer bindings for a dispatch to `$EKRANO_DUMP_DIR/dispatch_N/`.
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
        std::fs::create_dir_all(&dir).ok();

        let mut manifest = std::fs::File::create(format!("{dir}/manifest.txt")).unwrap();
        writeln!(manifest, "shader_id: {}", shader_id.0).unwrap();
        writeln!(manifest, "dispatch: ({}, {}, {})", dims.0, dims.1, dims.2).unwrap();
        writeln!(manifest, "num_bindings: {}", bindings.len()).unwrap();
        writeln!(manifest, "resource_slots: {:?}", indices).unwrap();

        for (i, res) in bindings.iter().enumerate() {
            match res {
                ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                    if let Some((gpu_buf, name)) = self.bind_map.get_buf(proxy.id) {
                        let size = gpu_buf.size() as usize;
                        writeln!(
                            manifest,
                            "binding[{i}]: buf name={name} size={size} bindless={}",
                            gpu_buf.bindless_index().unwrap_or(u32::MAX)
                        )
                        .unwrap();

                        let mut data = vec![0_u8; size];
                        let ok = match gpu_buf {
                            GpuBuffer::Owned(buf) => buf.read_to_cpu(device, &mut data).is_ok(),
                            GpuBuffer::Pooled(view) => view.read_to_cpu(device, &mut data).is_ok(),
                        };
                        if ok {
                            std::fs::write(format!("{dir}/buf_{i}.bin"), &data).ok();
                        } else {
                            writeln!(manifest, "  (read failed)").unwrap();
                        }
                    }
                }
                ResourceProxy::Image(proxy) => {
                    writeln!(
                        manifest,
                        "binding[{i}]: image {}x{} id={}",
                        proxy.width, proxy.height, proxy.id.0
                    )
                    .unwrap();
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

fn image_format_to_goldy(_format: crate::recording::ImageFormat) -> TextureFormat {
    TextureFormat::Rgba8Unorm
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
    fn get_buf(
        &mut self,
        device: &Device,
        size: u64,
        name: &'static str,
        access: DataAccess,
    ) -> Result<Buffer> {
        self.get_buf_with_stride(device, size, name, access, None, BufferFlags::empty())
    }

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
        let stride = stride.or_else(|| element_stride_for_buffer(name));
        Buffer::new_with_stride_and_flags(device, size, access, stride, buffer_flags)
            .map_err(|e| Error::Shader(e.to_string()))
    }
}

/// Fallback stride lookup for buffers that don't carry an explicit `element_stride`
/// in their [`BufferProxy`]. Prefer setting `BufferProxy::element_stride` at the
/// creation site instead of adding new entries here.
fn element_stride_for_buffer(name: &str) -> Option<u32> {
    match name {
        "vello.path_buf" => Some(32),
        "vello.tile_buf" => Some(8),
        "vello.segments_buf" => Some(24),
        "vello.seg_counts_buf" => Some(8),
        "vello.lines_buf" => Some(24),
        "vello.path_bbox_buf" => Some(24),
        "vello.bump_buf" => Some(32),
        "vello.ptcl_buf" => Some(4),
        "vello.info_bin_data_buf" => Some(4),
        "vello.tagmonoid_buf" => Some(20),
        "vello.draw_monoid_buf" => Some(16),
        "vello.draw_reduced_buf" => Some(16),
        "vello.reduced_buf" => Some(20),
        "vello.reduced2_buf" => Some(20),
        "vello.reduced_scan_buf" => Some(20),
        "vello.draw_bbox_buf" => Some(16),
        "vello.bin_header_buf" => Some(8),
        "vello.clip_inp_buf" => Some(8),
        "vello.clip_el_buf" => Some(32),
        "vello.clip_bic_buf" => Some(8),
        "vello.clip_bbox_buf" => Some(16),
        "vello.indirect_dispatch" => Some(16),
        "vello.indirect_count" => Some(16),
        // Must match `ConfigUniform` / Slang `Config` (includes `mask_active`).
        "vello.config" => Some(size_of::<ekrano_encoding::ConfigUniform>() as u32),
        "vello.wg_counts" => Some(320),
        "vello.scene" | "vello.blend_spill" | "vello.mask_lut" => Some(4),
        _ => {
            debug_assert!(
                false,
                "unknown buffer stride for '{name}' — add entry to element_stride_for_buffer \
                 or set BufferProxy::element_stride at the creation site"
            );
            log::warn!(
                "unknown buffer stride for '{name}', defaulting to 4 — add entry to \
                 element_stride_for_buffer or set BufferProxy::element_stride"
            );
            Some(4)
        }
    }
}

/// Build the push-constant index list for a dispatch.
///
/// Resources must be bound and have valid bindless indices. The number of
/// indices must not exceed `MAX_BINDLESS_SLOTS` (Goldy's push constant limit).
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
