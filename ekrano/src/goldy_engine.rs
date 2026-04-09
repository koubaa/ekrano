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
//! Push constants carry up to 16 bindless indices per dispatch (`GoldyDynamicSlots`).
//! This is simpler than wgpu's per-pipeline bind group layouts and satisfies
//! the "simplify the binding model" goal. BDA would only be needed for GPU-side
//! pointer chasing (e.g. buffer pools); we defer that unless required.
pub const MAX_BINDLESS_SLOTS: usize = 16;

use std::collections::HashMap;
use std::sync::LazyLock;

use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};
use goldy::{
    Buffer, BufferPool, BufferView, ComputeEncoder, ComputePipeline, DataAccess, Device, GpuFuture,
    ShaderModule, Texture,
};

static DUMP_DIR: LazyLock<Option<String>> = LazyLock::new(|| std::env::var("EKRANO_DUMP_DIR").ok());

use crate::{
    Error, Result,
    low_level::{BufferProxy, Command, ImageProxy, Recording, ResourceId, ResourceProxy, ShaderId},
    recording::BindType,
};

/// Return type for `process_recording_commands`: encoder plus pending downloads and deferred frees.
type ProcessRecordingResult = (
    ComputeEncoder,
    Vec<BufferProxy>,
    Vec<ResourceId>,
    Vec<ResourceId>,
);

/// Deferred work after a non-blocking submit: downloads and resource frees.
/// Call [`GoldyE ngine::complete_recording`] after waiting on the future.
#[must_use]
pub struct RecordingCompletion {
    pub(crate) pending_downloads: Vec<BufferProxy>,
    pub(crate) deferred_free_buffers: Vec<ResourceId>,
    pub(crate) deferred_free_images: Vec<ResourceId>,
}

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

    #[allow(dead_code, reason = "reserved for future SRV binding path")]
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

#[derive(Hash, PartialEq, Eq)]
struct BufferKey {
    size: u64,
    access: DataAccess,
    name: &'static str,
}

#[derive(Default)]
struct ResourcePool {
    bufs: HashMap<BufferKey, Vec<Buffer>>,
}

impl GoldyEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure all resources in bindings are materialized (cf. wgpu lazy materialization).
    /// Buffers that are only written by a shader are never Upload/Clear'd; images like
    /// `gradient_image` or `image_atlas` may be 1x1 placeholders never uploaded.
    /// For images: Image (read-write) needs Direct/UAV; `ImageRead` needs Interpolated/SRV.
    fn ensure_resources_materialized(
        &mut self,
        device: &Device,
        bindings: &[ResourceProxy],
        bind_types: &[BindType],
    ) -> Result<()> {
        for (i, res) in bindings.iter().enumerate() {
            match res {
                ResourceProxy::Buffer(proxy) | ResourceProxy::BufferRange { proxy, .. } => {
                    if self.bind_map.get_buf(proxy.id).is_none() {
                        let gpu_buf = if !is_pool_exempt(proxy.name)
                            && let Some(pool) = self.storage_pool.as_mut()
                        {
                            let stride = element_stride_for_buffer(proxy.name);
                            let view = pool
                                .alloc_bytes(proxy.size, stride)
                                .map_err(|e| Error::Shader(e.to_string()))?;
                            GpuBuffer::Pooled(view)
                        } else {
                            let buf = self.pool.get_buf(
                                device,
                                proxy.size,
                                proxy.name,
                                DataAccess::Scattered,
                            )?;
                            buf.clear(device, 0, proxy.size)
                                .map_err(|e| Error::Shader(e.to_string()))?;
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
                        let tex = Texture::new(
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
    #[allow(
        clippy::modulo_one,
        clippy::manual_is_multiple_of,
        reason = "DISPATCHES_PER_SUBMIT is intentionally 1 (flush every dispatch for TDR safety)"
    )]
    pub fn run_recording(
        &mut self,
        device: &Device,
        recording: &Recording,
        output: Option<(&ImageProxy, &Texture)>,
        _label: &'static str,
    ) -> Result<()> {
        let mut encoder = ComputeEncoder::new();
        let mut pending_downloads: Vec<BufferProxy> = Vec::new();
        let mut deferred_free_buffers: Vec<ResourceId> = Vec::new();
        let mut deferred_free_images: Vec<ResourceId> = Vec::new();
        let mut dispatch_count: usize = 0;

        let output_proxy_id = output.map(|(p, _)| p.id);

        if let Some((proxy, tex)) = output {
            self.bind_map.insert_image(proxy.id, tex.clone(), "output");
        }

        for command in &recording.commands {
            match command {
                Command::Upload(buf_proxy, bytes) => {
                    let buf = self.pool.get_buf(
                        device,
                        buf_proxy.size,
                        buf_proxy.name,
                        DataAccess::Scattered,
                    )?;
                    buf.write(0, bytes)
                        .map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map
                        .insert_buf(buf_proxy.id, GpuBuffer::Owned(buf), buf_proxy.name);
                }
                Command::UploadUniform(buf_proxy, bytes) => {
                    let buf = self.pool.get_buf(
                        device,
                        buf_proxy.size,
                        buf_proxy.name,
                        DataAccess::Broadcast,
                    )?;
                    buf.write(0, bytes)
                        .map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map
                        .insert_buf(buf_proxy.id, GpuBuffer::Owned(buf), buf_proxy.name);
                }
                Command::UploadImage(image_proxy, bytes) => {
                    let format = image_format_to_goldy(image_proxy.format);
                    let texture = Texture::with_data(
                        device,
                        bytes,
                        image_proxy.width,
                        image_proxy.height,
                        format,
                        SpatialAccess::Interpolated,
                        TextureFlags::COPY_DST,
                    )
                    .map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map
                        .insert_image(image_proxy.id, texture, "uploaded_image");
                }
                Command::WriteImage(image_proxy, [x, y], image_data) => {
                    if self.bind_map.get_image(image_proxy.id).is_none() {
                        let format = image_format_to_goldy(image_proxy.format);
                        // WriteImage textures are read by shaders as Texture2D (Interpolated), not RWTexture2D
                        let tex = Texture::new(
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
                        tex.write_region(*x, *y, image_data.width, image_data.height, bytes)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                    }
                }
                Command::Download(buf_proxy) => {
                    pending_downloads.push(*buf_proxy);
                }
                Command::Clear(buf_proxy, offset, size) => {
                    if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
                        let clear_size = size.unwrap_or(gpu_buf.size() - offset);
                        match gpu_buf {
                            GpuBuffer::Owned(buf) => {
                                buf.clear(device, *offset, clear_size)
                                    .map_err(|e| Error::Shader(e.to_string()))?;
                            }
                            GpuBuffer::Pooled(view) => {
                                if let Some(ref pool) = self.storage_pool {
                                    pool.backing_buffer()
                                        .clear(device, view.offset() + offset, clear_size)
                                        .map_err(|e| Error::Shader(e.to_string()))?;
                                }
                            }
                        }
                    } else {
                        // Lazy allocation: buffer not yet materialized (cf. wgpu pending_clears).
                        let buf = self.pool.get_buf(
                            device,
                            buf_proxy.size,
                            buf_proxy.name,
                            DataAccess::Scattered,
                        )?;
                        let clear_size = size.unwrap_or(buf.size() - offset);
                        buf.clear(device, *offset, clear_size)
                            .map_err(|e| Error::Shader(e.to_string()))?;
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
                Command::Dispatch(shader_id, (x, y, z), bindings) => {
                    if *x == 0 || *y == 0 || *z == 0 {
                        continue;
                    }
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, bindings, &bind_types)?;
                    let indices = collect_bindless_indices(bindings, &bind_types, &self.bind_map)?;

                    if let Some(ref dir) = *DUMP_DIR {
                        self.dump_dispatch(
                            device,
                            dispatch_count,
                            *shader_id,
                            (*x, *y, *z),
                            bindings,
                            &indices,
                            dir,
                        );
                    }

                    // Split execution: fine reads ptcl written by coarse. Run coarse+path_tiling first, sync, then fine.
                    let is_fine = output_proxy_id.is_some_and(|oid| {
                        bindings.len() == 8
                            && matches!(bindings.get(5), Some(ResourceProxy::Image(ip)) if ip.id == oid)
                    });
                    if is_fine {
                        encoder
                            .dispatch(device)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                        encoder = ComputeEncoder::new();
                    }

                    let mut pass = encoder.begin_compute_pass();
                    pass.set_pipeline(&self.shaders[shader_id.0].pipeline);
                    if !indices.is_empty() {
                        pass.set_push_constants_raw(&indices);
                    }
                    pass.dispatch(*x, *y, *z);
                    dispatch_count += 1;

                    if dispatch_count % Self::DISPATCHES_PER_SUBMIT == 0 {
                        encoder
                            .dispatch(device)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                        encoder = ComputeEncoder::new();
                    }
                }
                Command::DispatchIndirect(shader_id, buf_proxy, offset, bindings) => {
                    self.ensure_resources_materialized(
                        device,
                        &[ResourceProxy::Buffer(*buf_proxy)],
                        &[BindType::Buffer],
                    )?;
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, bindings, &bind_types)?;
                    let indices = collect_bindless_indices(bindings, &bind_types, &self.bind_map)?;
                    if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id)
                        && let Some(buf) = gpu_buf.as_owned()
                    {
                        if let Some(ref dir) = *DUMP_DIR {
                            let mut indirect_dims = [0_u32; 3];
                            let mut raw = [0_u8; 12];
                            if buf.read_to_cpu(device, &mut raw).is_ok() {
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
                                let full_size = buf.size() as usize;
                                let mut full = vec![0_u8; full_size];
                                if buf.read_to_cpu(device, &mut full).is_ok() {
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

                        let mut pass = encoder.begin_compute_pass();
                        pass.set_pipeline(&self.shaders[shader_id.0].pipeline);
                        if !indices.is_empty() {
                            pass.set_push_constants_raw(&indices);
                        }
                        pass.dispatch_indirect(buf, *offset);
                        dispatch_count += 1;

                        if dispatch_count % Self::DISPATCHES_PER_SUBMIT == 0 {
                            encoder
                                .dispatch(device)
                                .map_err(|e| Error::Shader(e.to_string()))?;
                            encoder = ComputeEncoder::new();
                        }
                    }
                }
                #[cfg(feature = "debug_layers")]
                Command::Draw(_) => {}
            }
        }

        encoder
            .dispatch(device)
            .map_err(|e| Error::Shader(e.to_string()))?;

        // Downloads must happen before frees, since a recording may download
        // and then free the same buffer.
        for buf_proxy in pending_downloads {
            if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id)
                && let GpuBuffer::Owned(buf) = gpu_buf
            {
                let size = buf.size() as usize;
                let mut output = vec![0_u8; size];
                buf.read_to_cpu(device, &mut output)
                    .map_err(|e| Error::Shader(e.to_string()))?;
                self.downloads.insert(buf_proxy.id, output);
            }
        }

        for id in deferred_free_buffers {
            self.bind_map.remove_buf(id);
        }
        for id in deferred_free_images {
            self.bind_map.remove_image(id);
        }
        Ok(())
    }

    /// Submit a recording without blocking.
    ///
    /// Returns a [`GpuFuture`] and [`RecordingCompletion`]. The caller must:
    /// 1. Wait on the future (e.g. `future.wait_timeout(2000)`)
    /// 2. Call [`complete_recording`](GoldyEngine::complete_recording) with the completion
    ///
    /// Only supports recordings with `output: None` (coarse pass). For fine passes, use
    /// [`run_recording`](GoldyEngine::run_recording).
    pub fn submit_recording(
        &mut self,
        device: &Device,
        recording: &Recording,
        output: Option<(&ImageProxy, &Texture)>,
        _label: &'static str,
    ) -> Result<(GpuFuture, RecordingCompletion)> {
        if output.is_some() {
            return Err(Error::Shader(
                "submit_recording only supports output=None (coarse pass); use run_recording for fine passes".into(),
            ));
        }
        let (encoder, pending_downloads, deferred_free_buffers, deferred_free_images) =
            self.process_recording_commands(device, recording, output)?;
        let future = encoder
            .submit(device)
            .map_err(|e| Error::Shader(e.to_string()))?;
        Ok((
            future,
            RecordingCompletion {
                pending_downloads,
                deferred_free_buffers,
                deferred_free_images,
            },
        ))
    }

    /// Complete a recording after waiting on its future. Performs downloads and deferred frees.
    pub fn complete_recording(
        &mut self,
        device: &Device,
        completion: RecordingCompletion,
    ) -> Result<()> {
        for buf_proxy in completion.pending_downloads {
            if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id)
                && let GpuBuffer::Owned(buf) = gpu_buf
            {
                let size = buf.size() as usize;
                let mut output = vec![0_u8; size];
                buf.read_to_cpu(device, &mut output)
                    .map_err(|e| Error::Shader(e.to_string()))?;
                self.downloads.insert(buf_proxy.id, output);
            }
        }
        for id in completion.deferred_free_buffers {
            self.bind_map.remove_buf(id);
        }
        for id in completion.deferred_free_images {
            self.bind_map.remove_image(id);
        }
        Ok(())
    }

    /// Max dispatches per GPU submission to avoid TDR on large workloads.
    /// Each submission resets the GPU timeout watchdog.
    const DISPATCHES_PER_SUBMIT: usize = 1;

    /// Process recording commands and build the encoder. Shared by `run_recording` and `submit_recording`.
    #[allow(
        clippy::modulo_one,
        clippy::manual_is_multiple_of,
        reason = "DISPATCHES_PER_SUBMIT is intentionally 1 (flush every dispatch for TDR safety)"
    )]
    fn process_recording_commands(
        &mut self,
        device: &Device,
        recording: &Recording,
        output: Option<(&ImageProxy, &Texture)>,
    ) -> Result<ProcessRecordingResult> {
        let mut encoder = ComputeEncoder::new();
        let mut pending_downloads: Vec<BufferProxy> = Vec::new();
        let mut deferred_free_buffers: Vec<ResourceId> = Vec::new();
        let mut deferred_free_images: Vec<ResourceId> = Vec::new();
        let mut dispatch_count: usize = 0;

        let output_proxy_id = output.map(|(p, _)| p.id);

        if let Some((proxy, tex)) = output {
            self.bind_map.insert_image(proxy.id, tex.clone(), "output");
        }

        for command in &recording.commands {
            match command {
                Command::Upload(buf_proxy, bytes) => {
                    let buf = self.pool.get_buf(
                        device,
                        buf_proxy.size,
                        buf_proxy.name,
                        DataAccess::Scattered,
                    )?;
                    buf.write(0, bytes)
                        .map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map
                        .insert_buf(buf_proxy.id, GpuBuffer::Owned(buf), buf_proxy.name);
                }
                Command::UploadUniform(buf_proxy, bytes) => {
                    let buf = self.pool.get_buf(
                        device,
                        buf_proxy.size,
                        buf_proxy.name,
                        DataAccess::Broadcast,
                    )?;
                    buf.write(0, bytes)
                        .map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map
                        .insert_buf(buf_proxy.id, GpuBuffer::Owned(buf), buf_proxy.name);
                }
                Command::UploadImage(image_proxy, bytes) => {
                    let format = image_format_to_goldy(image_proxy.format);
                    let texture = Texture::with_data(
                        device,
                        bytes,
                        image_proxy.width,
                        image_proxy.height,
                        format,
                        SpatialAccess::Interpolated,
                        TextureFlags::COPY_DST,
                    )
                    .map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map
                        .insert_image(image_proxy.id, texture, "uploaded_image");
                }
                Command::WriteImage(image_proxy, [x, y], image_data) => {
                    if self.bind_map.get_image(image_proxy.id).is_none() {
                        let format = image_format_to_goldy(image_proxy.format);
                        let tex = Texture::new(
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
                        tex.write_region(*x, *y, image_data.width, image_data.height, bytes)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                    }
                }
                Command::Download(buf_proxy) => {
                    pending_downloads.push(*buf_proxy);
                }
                Command::Clear(buf_proxy, offset, size) => {
                    if let Some((gpu_buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
                        let clear_size = size.unwrap_or(gpu_buf.size() - offset);
                        match gpu_buf {
                            GpuBuffer::Owned(buf) => {
                                buf.clear(device, *offset, clear_size)
                                    .map_err(|e| Error::Shader(e.to_string()))?;
                            }
                            GpuBuffer::Pooled(view) => {
                                if let Some(ref pool) = self.storage_pool {
                                    pool.backing_buffer()
                                        .clear(device, view.offset() + offset, clear_size)
                                        .map_err(|e| Error::Shader(e.to_string()))?;
                                }
                            }
                        }
                    } else {
                        let buf = self.pool.get_buf(
                            device,
                            buf_proxy.size,
                            buf_proxy.name,
                            DataAccess::Scattered,
                        )?;
                        let clear_size = size.unwrap_or(buf.size() - offset);
                        buf.clear(device, *offset, clear_size)
                            .map_err(|e| Error::Shader(e.to_string()))?;
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
                Command::Dispatch(shader_id, (x, y, z), bindings) => {
                    if *x == 0 || *y == 0 || *z == 0 {
                        continue;
                    }
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, bindings, &bind_types)?;
                    let indices = collect_bindless_indices(bindings, &bind_types, &self.bind_map)?;

                    if let Some(ref dir) = *DUMP_DIR {
                        self.dump_dispatch(
                            device,
                            dispatch_count,
                            *shader_id,
                            (*x, *y, *z),
                            bindings,
                            &indices,
                            dir,
                        );
                    }

                    let is_fine = output_proxy_id.is_some_and(|oid| {
                        bindings.len() == 8
                            && matches!(bindings.get(5), Some(ResourceProxy::Image(ip)) if ip.id == oid)
                    });
                    if is_fine {
                        return Err(Error::Shader(
                            "submit_recording does not support coarse+fine split; use run_recording".into(),
                        ));
                    }
                    let mut pass = encoder.begin_compute_pass();
                    pass.set_pipeline(&self.shaders[shader_id.0].pipeline);
                    if !indices.is_empty() {
                        pass.set_push_constants_raw(&indices);
                    }
                    pass.dispatch(*x, *y, *z);
                    dispatch_count += 1;

                    // Flush every dispatch to avoid TDR on large workloads
                    if dispatch_count % Self::DISPATCHES_PER_SUBMIT == 0 {
                        encoder
                            .dispatch(device)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                        encoder = ComputeEncoder::new();
                    }
                }
                Command::DispatchIndirect(shader_id, buf_proxy, offset, bindings) => {
                    self.ensure_resources_materialized(
                        device,
                        &[ResourceProxy::Buffer(*buf_proxy)],
                        &[BindType::Buffer],
                    )?;
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, bindings, &bind_types)?;
                    let indices = collect_bindless_indices(bindings, &bind_types, &self.bind_map)?;
                    if let Some((gpu_buf, _buf_name)) = self.bind_map.get_buf(buf_proxy.id)
                        && let Some(buf) = gpu_buf.as_owned()
                    {
                        if let Some(ref dir) = *DUMP_DIR {
                            let mut indirect_dims = [0_u32; 3];
                            let full_size = buf.size() as usize;
                            let mut full = vec![0_u8; full_size];
                            if buf.read_to_cpu(device, &mut full).is_ok() {
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

                        let mut pass = encoder.begin_compute_pass();
                        pass.set_pipeline(&self.shaders[shader_id.0].pipeline);
                        if !indices.is_empty() {
                            pass.set_push_constants_raw(&indices);
                        }
                        pass.dispatch_indirect(buf, *offset);
                        dispatch_count += 1;

                        if dispatch_count % Self::DISPATCHES_PER_SUBMIT == 0 {
                            encoder
                                .dispatch(device)
                                .map_err(|e| Error::Shader(e.to_string()))?;
                            encoder = ComputeEncoder::new();
                        }
                    }
                }
                #[cfg(feature = "debug_layers")]
                Command::Draw(_) => {}
            }
        }

        Ok((
            encoder,
            pending_downloads,
            deferred_free_buffers,
            deferred_free_images,
        ))
    }

    /// Get downloaded buffer data, if the recording contained a Download command for it.
    pub fn get_download(&self, buf: BufferProxy) -> Option<&[u8]> {
        self.downloads.get(&buf.id).map(|v| v.as_slice())
    }

    /// Free a downloaded buffer from the engine's storage.
    pub fn free_download(&mut self, buf: BufferProxy) {
        self.downloads.remove(&buf.id);
    }

    /// Prepare the storage pool for a recording. Creates or resets the pool with the given size.
    /// Call before `run_recording` when the pipeline will use pooled buffers.
    pub fn prepare_storage_pool(&mut self, device: &Device, pool_size: u64) -> Result<()> {
        let need_new = match &self.storage_pool {
            Some(pool) => pool.capacity() < pool_size,
            None => true,
        };
        if need_new {
            let pool =
                BufferPool::new(device, pool_size).map_err(|e| Error::Shader(e.to_string()))?;
            pool.backing_buffer()
                .clear(device, 0, pool_size)
                .map_err(|e| Error::Shader(e.to_string()))?;
            self.storage_pool = Some(pool);
        } else {
            let pool = self.storage_pool.as_mut().unwrap();
            pool.reset();
            pool.backing_buffer()
                .clear(device, 0, pool.capacity())
                .map_err(|e| Error::Shader(e.to_string()))?;
        }
        Ok(())
    }

    /// Clear all transient resources (buffers, images, downloads) between retry attempts.
    /// Drops the storage pool so the next `prepare_storage_pool` allocates fresh.
    pub fn clear_transients(&mut self) {
        self.bind_map.buf_map.clear();
        self.bind_map.image_map.clear();
        self.downloads.clear();
        self.storage_pool = None;
    }

    /// Release the storage pool and transient buffers to free GPU memory.
    /// Keeps images intact (caller may still need textures for readback).
    pub fn release_pool(&mut self) {
        self.bind_map.buf_map.clear();
        self.downloads.clear();
        self.storage_pool = None;
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
        writeln!(manifest, "push_constants: {:?}", indices).unwrap();

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
                            GpuBuffer::Pooled(view) => {
                                if let Some(ref pool) = self.storage_pool {
                                    let full_size = pool.backing_buffer().size() as usize;
                                    let mut full = vec![0_u8; full_size];
                                    if pool.backing_buffer().read_to_cpu(device, &mut full).is_ok()
                                    {
                                        let off = view.offset() as usize;
                                        data.copy_from_slice(&full[off..off + size]);
                                        true
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            }
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
}

impl ResourcePool {
    fn get_buf(
        &mut self,
        device: &Device,
        size: u64,
        name: &'static str,
        access: DataAccess,
    ) -> Result<Buffer> {
        let key = BufferKey { size, access, name };
        let pool = self.bufs.entry(key).or_default();
        if let Some(buf) = pool.pop() {
            return Ok(buf);
        }
        let stride = element_stride_for_buffer(name);
        Buffer::new_with_stride(device, size, access, stride)
            .map_err(|e| Error::Shader(e.to_string()))
    }
}

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
        "vello.draw_bbox_buf" => Some(16),
        "vello.bin_header_buf" => Some(8),
        "vello.clip_inp_buf" => Some(8),
        "vello.clip_el_buf" => Some(32),
        "vello.clip_bic_buf" => Some(8),
        "vello.clip_bbox_buf" => Some(16),
        _ => Some(4),
    }
}

/// Build the push-constant index list for a dispatch.
///
/// Resources must be bound and have valid bindless indices. The number of
/// indices must not exceed `MAX_BINDLESS_SLOTS` (Goldy's push constant limit).
fn collect_bindless_indices(
    resources: &[ResourceProxy],
    _bind_types: &[BindType],
    bind_map: &BindMap,
) -> Result<Vec<u32>, Error> {
    let mut indices = Vec::with_capacity(resources.len());
    for res in resources.iter() {
        let idx = match res {
            ResourceProxy::Buffer(proxy) => bind_map
                .get_buf(proxy.id)
                .and_then(|(buf, _)| buf.bindless_index())
                .ok_or_else(|| Error::Shader("buffer not found or has no bindless index".into()))?,
            ResourceProxy::BufferRange { proxy, .. } => bind_map
                .get_buf(proxy.id)
                .and_then(|(buf, _)| buf.bindless_index())
                .ok_or_else(|| Error::Shader("buffer not found or has no bindless index".into()))?,
            ResourceProxy::Image(proxy) => bind_map
                .get_image(proxy.id)
                .and_then(|(tex, _)| tex.bindless_index())
                .ok_or_else(|| Error::Shader("image not found or has no bindless index".into()))?,
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
