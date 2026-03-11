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

use goldy::{
    Buffer, ComputeEncoder, ComputePipeline, DataAccess, Device, ShaderModule, Texture,
};
use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};

use crate::{
    Error, Result,
    low_level::{BufferProxy, Command, ImageProxy, Recording, ResourceId, ResourceProxy, ShaderId},
    recording::BindType,
};

/// Goldy-based recording executor.
#[derive(Default)]
pub struct GoldyEngine {
    shaders: Vec<GoldyShader>,
    pool: ResourcePool,
    bind_map: BindMap,
    /// Buffers that were downloaded (Command::Download); keyed by ResourceId.
    downloads: HashMap<ResourceId, Vec<u8>>,
}

struct GoldyShader {
    pipeline: ComputePipeline,
    bindings: Vec<BindType>,
}

#[derive(Default)]
struct BindMap {
    buf_map: HashMap<ResourceId, (Buffer, &'static str)>,
    image_map: HashMap<ResourceId, (Texture, &'static str)>,
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
    /// gradient_image or image_atlas may be 1x1 placeholders never uploaded.
    /// For images: Image (read-write) needs Direct/UAV; ImageRead needs Interpolated/SRV.
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
                        let buf = self.pool.get_buf(
                            device,
                            proxy.size,
                            proxy.name,
                            DataAccess::Scattered,
                        )?;
                        buf.clear(device, 0, proxy.size)
                            .map_err(|e| Error::Shader(e.to_string()))?;
                        self.bind_map.insert_buf(proxy.id, buf, proxy.name);
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
        let shader_module =
            ShaderModule::from_slang_with_paths_and_defines(device, slang_source, search_paths, defines)
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
                    buf.write(0, bytes).map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map.insert_buf(buf_proxy.id, buf, buf_proxy.name);
                }
                Command::UploadUniform(buf_proxy, bytes) => {
                    let buf = self.pool.get_buf(
                        device,
                        buf_proxy.size,
                        buf_proxy.name,
                        DataAccess::Broadcast,
                    )?;
                    buf.write(0, bytes).map_err(|e| Error::Shader(e.to_string()))?;
                    self.bind_map.insert_buf(buf_proxy.id, buf, buf_proxy.name);
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
                        if image_data.data.is_empty() && image_data.width != 0 && image_data.height != 0 {
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
                    if let Some((buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
                        let clear_size = size.unwrap_or(buf.size() - offset);
                        buf.clear(device, *offset, clear_size)
                            .map_err(|e| Error::Shader(e.to_string()))?;
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
                        self.bind_map.insert_buf(buf_proxy.id, buf, buf_proxy.name);
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
                    let indices =
                        collect_bindless_indices(bindings, &bind_types, &self.bind_map)?;

                    // Split execution: fine reads ptcl written by coarse. Run coarse+path_tiling first, sync, then fine.
                    let is_fine = output_proxy_id.map_or(false, |oid| {
                        bindings.len() == 8
                            && matches!(bindings.get(5), Some(ResourceProxy::Image(ip)) if ip.id == oid)
                    });
                    if is_fine {
                        encoder.dispatch(device).map_err(|e| Error::Shader(e.to_string()))?;
                        encoder = ComputeEncoder::new();
                    }

                    let mut pass = encoder.begin_compute_pass();
                    pass.set_pipeline(&self.shaders[shader_id.0].pipeline);
                    if !indices.is_empty() {
                        pass.set_push_constants_raw(&indices);
                    }
                    pass.dispatch(*x, *y, *z);
                }
                Command::DispatchIndirect(shader_id, buf_proxy, offset, bindings) => {
                    self.ensure_resources_materialized(
                        device,
                        &[ResourceProxy::Buffer(*buf_proxy)],
                        &[BindType::Buffer],
                    )?;
                    let bind_types: Vec<_> = self.shaders[shader_id.0].bindings.clone();
                    self.ensure_resources_materialized(device, bindings, &bind_types)?;
                    let indices =
                        collect_bindless_indices(bindings, &bind_types, &self.bind_map)?;
                    if let Some((indirect_buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
                        let mut pass = encoder.begin_compute_pass();
                        pass.set_pipeline(&self.shaders[shader_id.0].pipeline);
                        if !indices.is_empty() {
                            pass.set_push_constants_raw(&indices);
                        }
                        pass.dispatch_indirect(indirect_buf, *offset);
                    }
                }
                #[cfg(feature = "debug_layers")]
                Command::Draw(_) => {}
            }
        }

        encoder.dispatch(device).map_err(|e| Error::Shader(e.to_string()))?;

        // Diagnostic readback (enabled by EKRANO_DIAG=1)
        if std::env::var("EKRANO_DIAG").is_ok() {
            for (_id, (buf, name)) in &self.bind_map.buf_map {
                if *name == "vello.tile_buf" {
                    let sz = buf.size() as usize;
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut nonzero_bd = 0u32;
                        let mut nonzero_seg = 0u32;
                        let mut bd_samples = Vec::new();
                        for i in (0..u32s.len()).step_by(2) {
                            let backdrop = u32s[i] as i32;
                            let seg = u32s.get(i + 1).copied().unwrap_or(0);
                            if backdrop != 0 {
                                nonzero_bd += 1;
                                if bd_samples.len() < 10 {
                                    bd_samples.push(format!("[{}]={{bd:{},seg:{}}}", i / 2, backdrop, seg));
                                }
                            }
                            if seg != 0 { nonzero_seg += 1; }
                        }
                        log::warn!("DIAG tile_buf: {} total tiles, {} with nonzero backdrop, {} with nonzero seg. BD samples: {}",
                            u32s.len() / 2, nonzero_bd, nonzero_seg, bd_samples.join(", "));
                    }
                }
                if *name == "vello.segments_buf" {
                    // Segment = {float2 point0, float2 point1, float y_edge, u32 pad} = 24 bytes
                    let seg_stride = 6; // 6 u32s per segment
                    let num_segs = 20;
                    let sz = buf.size().min((num_segs * seg_stride * 4) as u64) as usize;
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut samples = Vec::new();
                        for i in 0..(u32s.len() / seg_stride).min(num_segs) {
                            let off = i * seg_stride;
                            let p0x = f32::from_bits(u32s[off]);
                            let p0y = f32::from_bits(u32s[off+1]);
                            let p1x = f32::from_bits(u32s[off+2]);
                            let p1y = f32::from_bits(u32s[off+3]);
                            let ye = f32::from_bits(u32s[off+4]);
                            if p0x != 0.0 || p0y != 0.0 || p1x != 0.0 || p1y != 0.0 {
                                samples.push(format!("[{}]=({:.2},{:.2})->({:.2},{:.2}) ye={:.2}",
                                    i, p0x, p0y, p1x, p1y, ye));
                            }
                        }
                        let nonzero_segs = (0..u32s.len() / seg_stride)
                            .filter(|&i| {
                                let off = i * seg_stride;
                                u32s[off..off+5].iter().any(|&v| v != 0)
                            })
                            .count();
                        log::warn!("DIAG segments_buf (stride=24): {} total segs, {} nonzero in first {}. Samples: {}",
                            buf.size() / 24, nonzero_segs, u32s.len() / seg_stride, samples.join(", "));
                    }
                }
                if *name == "vello.bump_buf" {
                    let mut data = vec![0u8; 32];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        log::warn!("DIAG bump: failed={} binning={} ptcl={} tile={} seg_counts={} segments={} blend={} lines={}",
                            u32s[0], u32s[1], u32s[2], u32s[3], u32s[4], u32s[5], u32s[6], u32s[7]);
                    }
                }
                if *name == "vello.path_buf" {
                    // Path (Rust) = {bbox: [u32;4], tiles: u32, _pad: [u32;3]} = 32 bytes
                    let path_stride = 8; // 8 u32s per Path
                    let sz = buf.size().min(320) as usize; // first 10 paths
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut samples = Vec::new();
                        for p in 0..(u32s.len() / path_stride).min(10) {
                            let off = p * path_stride;
                            samples.push(format!("[{}]={{bbox:[{},{},{},{}],tiles:{},pad:[{},{},{}]}}",
                                p, u32s[off], u32s[off+1], u32s[off+2], u32s[off+3], u32s[off+4],
                                u32s[off+5], u32s[off+6], u32s[off+7]));
                        }
                        log::warn!("DIAG path_buf (stride=32): {}", samples.join(", "));
                    }
                }
                if *name == "vello.path_bbox_buf" {
                    // PathBbox = {x0, y0, x1, y1, draw_flags, trans_ix} = 24 bytes per entry
                    let sz = buf.size().min(240) as usize; // first 10 bboxes
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut samples = Vec::new();
                        for p in 0..(u32s.len() / 6).min(5) {
                            let off = p * 6;
                            let x0 = u32s[off] as i32;
                            let y0 = u32s[off+1] as i32;
                            let x1 = u32s[off+2] as i32;
                            let y1 = u32s[off+3] as i32;
                            samples.push(format!("[{}]={{x0:{},y0:{},x1:{},y1:{},flags:{},tx:{}}}",
                                p, x0, y0, x1, y1, u32s[off+4], u32s[off+5]));
                        }
                        log::warn!("DIAG path_bbox_buf (stride=24): {}", samples.join(", "));
                    }
                }
                if *name == "vello.seg_counts_buf" {
                    // SegmentCount = {uint line_ix, uint counts} = 8 bytes
                    let sz = buf.size().min(160) as usize; // first 20 entries
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        // At stride 8 (2 u32s per entry): entries are [line_ix, counts] pairs
                        let mut samples_s8 = Vec::new();
                        for i in 0..(u32s.len() / 2).min(10) {
                            let off = i * 2;
                            samples_s8.push(format!("[{}]={{line:{},cnt:{}}}", i, u32s[off], u32s[off+1]));
                        }
                        // At stride 4 (1 u32 per entry): alternating line_ix and counts
                        let raw: Vec<String> = u32s.iter().take(20).map(|v| format!("{v}")).collect();
                        log::warn!("DIAG seg_counts (stride=8): {}", samples_s8.join(", "));
                        log::warn!("DIAG seg_counts raw: [{}]", raw.join(", "));
                    }
                }
                if *name == "vello.lines_buf" {
                    // LineSoup = {path_ix: u32, _pad: u32, p0: [f32;2], p1: [f32;2]} = 24 bytes
                    let line_stride = 6; // 6 u32s per LineSoup
                    let num_lines = 10;
                    let sz = buf.size().min((num_lines * line_stride * 4) as u64) as usize;
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let mut samples = Vec::new();
                        for i in 0..(u32s.len() / line_stride).min(num_lines) {
                            let off = i * line_stride;
                            let path_ix = u32s[off];
                            let pad = u32s[off+1];
                            let p0x = f32::from_bits(u32s[off+2]);
                            let p0y = f32::from_bits(u32s[off+3]);
                            let p1x = f32::from_bits(u32s[off+4]);
                            let p1y = f32::from_bits(u32s[off+5]);
                            samples.push(format!("[{}]={{path:{},pad:{},p0:({:.2},{:.2}),p1:({:.2},{:.2})}}",
                                i, path_ix, pad, p0x, p0y, p1x, p1y));
                        }
                        log::warn!("DIAG lines_buf (stride=24): total={}, {}", buf.size() / 24, samples.join(", "));
                    }
                }
                if *name == "vello.ptcl_buf" {
                    // Read first 256 u32s of ptcl to verify coarse output
                    let sz = buf.size().min(1024) as usize;
                    let mut data = vec![0u8; sz];
                    if buf.read_to_cpu(device, &mut data).is_ok() {
                        let u32s: Vec<u32> = data.chunks(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        let nonzero = u32s.iter().filter(|&&v| v != 0).count();
                        let preview: Vec<String> = u32s.iter().take(40).map(|v| format!("{v}")).collect();
                        log::warn!("DIAG ptcl_buf: {} nonzero in first {} u32s, data: [{}]",
                            nonzero, u32s.len().min(40), preview.join(", "));
                    }
                }
            }
        }

        // Downloads must happen before frees, since a recording may download
        // and then free the same buffer.
        for buf_proxy in pending_downloads {
            if let Some((buf, _)) = self.bind_map.get_buf(buf_proxy.id) {
                let size = buf.size() as usize;
                let mut output = vec![0u8; size];
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

    /// Get downloaded buffer data, if the recording contained a Download command for it.
    pub fn get_download(&self, buf: BufferProxy) -> Option<&[u8]> {
        self.downloads.get(&buf.id).map(|v| v.as_slice())
    }

    /// Free a downloaded buffer from the engine's storage.
    pub fn free_download(&mut self, buf: BufferProxy) {
        self.downloads.remove(&buf.id);
    }

    /// Clear all transient resources (buffers, images, downloads) between retry attempts.
    /// Shaders and the pool are preserved.
    pub fn clear_transients(&mut self) {
        self.bind_map.buf_map.clear();
        self.bind_map.image_map.clear();
        self.downloads.clear();
    }
}

fn image_format_to_goldy(_format: crate::recording::ImageFormat) -> TextureFormat {
    TextureFormat::Rgba8Unorm
}

impl BindMap {
    fn insert_buf(&mut self, id: ResourceId, buf: Buffer, name: &'static str) {
        self.buf_map.insert(id, (buf, name));
    }

    fn get_buf(&self, id: ResourceId) -> Option<&(Buffer, &'static str)> {
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
