// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Direct GPU resource handles (no bind-map / proxies).

use std::mem::size_of;

use goldy::task_graph::NodeAccess;
use goldy::types::{BufferFlags, ResourceAccess, TextureFlags, TextureKind};
use goldy::{Buffer, BufferKind, Texture, TextureFormat};

use crate::goldy_renderer::FrameRecorder;
use crate::resource_proxy::{BindType, ImageFormat};
use crate::{Error, RenderParams, Result};
use ekrano_encoding::{BumpAllocators, CoverageMask, Images, Ramps, RenderConfig};

/// Shader binding helper for pipeline [`Buffer`] handles.
pub(crate) trait PipelineBuffer {
    fn as_binding(&self) -> GpuBinding<'_>;
}

impl PipelineBuffer for Buffer {
    fn as_binding(&self) -> GpuBinding<'_> {
        GpuBinding::Buf(self)
    }
}

pub(crate) enum GpuBinding<'a> {
    Buf(&'a Buffer),
    Tex(&'a Texture),
    /// A GPU sampler represented by its pre-resolved bindless index.
    /// We store the index directly (not a reference) so `fine_resources` doesn't
    /// borrow from `recorder.persistent`, which would conflict with `recorder.dispatch()`.
    Sampler(u32),
    /// A persistent (pre-initialized) buffer represented by its pre-resolved bindless
    /// index.  Like `Sampler`, the index is stored directly so `fine_resources` can be
    /// built without holding a live `&Buffer` reference across a `recorder.dispatch()`
    /// call.  Use this for buffers that are uploaded exactly once (e.g. static LUTs
    /// stored in `PersistentState`) and are guaranteed to be GPU-readable on every
    /// frame after their first upload, without any additional `WriteBuffer` nodes.
    PersistentBuf(u32),
}

impl<'a> GpuBinding<'a> {
    pub(crate) fn bindless_slot(&self, is_read_only: bool) -> Result<u32, Error> {
        let idx = match self {
            GpuBinding::Buf(buf) => {
                if is_read_only {
                    buf.resource_index(ResourceAccess::Read)
                } else {
                    buf.resource_index(ResourceAccess::Write)
                }
            }
            GpuBinding::Tex(tex) => {
                if is_read_only {
                    tex.resource_index(ResourceAccess::Read)
                } else {
                    tex.resource_index(ResourceAccess::Write)
                }
            }
            GpuBinding::Sampler(idx) | GpuBinding::PersistentBuf(idx) => return Ok(*idx),
        };
        idx.ok_or_else(|| Error::Shader("bindless index missing for shader resource binding".into()))
    }
}

fn is_pool_exempt(name: &'static str) -> bool {
    matches!(name, "ekrano.bump_buf" | "ekrano.indirect_dispatch")
}

fn image_fmt_goldy(f: ImageFormat) -> TextureFormat {
    match f {
        ImageFormat::Rgba8 => TextureFormat::Rgba8Unorm,
        ImageFormat::Bgra8 => TextureFormat::Bgra8Unorm,
    }
}

pub(crate) fn alloc_pipeline_buffer(
    recorder: &mut FrameRecorder<'_>,
    size: u64,
    stride: u32,
    name: &'static str,
    flags: BufferFlags,
) -> Result<Buffer, Error> {
    let buf = recorder.persistent.pool.get_buf_with_stride(
        recorder.device(),
        recorder.context(),
        size,
        name,
        BufferKind::Scattered,
        Some(stride),
        flags,
    )?;
    // Pre-clear pool-exempt buffers (bump needs zeroing each frame; indirect
    // dispatch counts must be 0 before GPU pipelines them). Other pipeline
    // buffers are always overwritten by GPU dispatches before first read.
    if is_pool_exempt(name) {
        recorder.graph().clear_buffer(&buf, 0, size);
    }
    Ok(buf)
}

pub(crate) fn record_upload_bytes(
    recorder: &mut FrameRecorder<'_>,
    name: &'static str,
    element_stride: u32,
    bytes: &[u8],
) -> Result<Buffer, Error> {
    let buf = recorder.persistent.pool.get_buf_with_stride(
        recorder.device(),
        recorder.context(),
        bytes.len() as u64,
        name,
        BufferKind::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    recorder.graph().write_buffer(&buf, 0, bytes.to_vec());
    Ok(buf)
}

/// Like [`record_upload_bytes`] but takes ownership of the byte vector, avoiding
/// the redundant `to_vec()` copy when the caller already holds an owned `Vec<u8>`.
pub(crate) fn record_upload_bytes_owned(
    recorder: &mut FrameRecorder<'_>,
    name: &'static str,
    element_stride: u32,
    bytes: Vec<u8>,
) -> Result<Buffer, Error> {
    let buf = recorder.persistent.pool.get_buf_with_stride(
        recorder.device(),
        recorder.context(),
        bytes.len() as u64,
        name,
        BufferKind::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    recorder.graph().write_buffer(&buf, 0, bytes);
    Ok(buf)
}

pub(crate) fn record_upload_image(
    recorder: &mut FrameRecorder<'_>,
    width: u32,
    height: u32,
    format: ImageFormat,
    bytes: &[u8],
) -> Result<Texture, Error> {
    let format = image_fmt_goldy(format);
    let texture = recorder
        .persistent
        .tex_pool
        .acquire(
            recorder.device(),
            width,
            height,
            format,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )
        .map_err(|e| Error::Shader(e.to_string()))?;
    recorder
        .graph()
        .write_texture(&texture, bytes.to_vec())
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(texture)
}

pub(crate) fn write_image_region(
    recorder: &mut FrameRecorder<'_>,
    tex: &Texture,
    x: u32,
    y: u32,
    image_data: &peniko::ImageData,
) -> Result<(), Error> {
    if image_data.data.is_empty() && image_data.width != 0 && image_data.height != 0 {
        return Err(Error::InvalidImage {
            id: image_data.data.id(),
            reason: "image has non-zero dimensions but no pixel data; \
                     it may have been registered to a different renderer \
                     or unregistered before this render was submitted",
        });
    }
    let raw_bytes = image_data.data.data();

    // The atlas is always sampled with hardware bilinear, which requires premultiplied-alpha
    // texels to avoid fringing on transparent edges. Straight-alpha images (ImageAlphaType::Alpha)
    // are converted to premultiplied on the CPU before upload; premultiplied sources are used
    // as-is.  Callers' ImageData is never mutated.
    let premul_storage;
    let bytes: &[u8] = if image_data.alpha_type == peniko::ImageAlphaType::Alpha {
        premul_storage = premultiply_rgba8(raw_bytes);
        &premul_storage
    } else {
        raw_bytes
    };

    recorder
        .graph()
        .write_texture_region(tex, x, y, image_data.width, image_data.height, bytes.to_vec())
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(())
}

/// Premultiply every RGBA8 pixel: `(r, g, b, a)` → `(r*a/255, g*a/255, b*a/255, a)`.
///
/// Uses integer arithmetic to match the GPU's 8-bit rounding behaviour precisely.
fn premultiply_rgba8(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for chunk in out.chunks_exact_mut(4) {
        let a = chunk[3] as u32;
        chunk[0] = ((chunk[0] as u32 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u32 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u32 * a + 127) / 255) as u8;
    }
    out
}

pub(crate) fn acquire_texture_rgba(
    recorder: &mut FrameRecorder<'_>,
    width: u32,
    height: u32,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    recorder
        .persistent
        .tex_pool
        .acquire(
            recorder.device(),
            width,
            height,
            TextureFormat::Rgba8Unorm,
            access,
            flags,
        )
        .map_err(|e| Error::Shader(e.to_string()))
}

pub(crate) fn clear_gpu_buf(
    recorder: &mut FrameRecorder<'_>,
    buf: &Buffer,
    off: u64,
    size: Option<u64>,
) -> Result<(), Error> {
    let sz = size.unwrap_or_else(|| buf.size().saturating_sub(off));
    recorder.graph().clear_buffer(buf, off, sz);
    Ok(())
}

/// Reuse a cached buffer if one is available, or allocate a fresh one from the pool.
fn al_cached_opt(
    recorder: &mut FrameRecorder<'_>,
    cached: Option<Buffer>,
    size: u64,
    stride: u32,
    name: &'static str,
) -> Result<Buffer, Error> {
    match cached {
        Some(buf) => Ok(buf),
        None => alloc_pipeline_buffer(recorder, size, stride, name, BufferFlags::empty()),
    }
}

/// The seven large pipeline buffers whose sizes are fixed or change only on coarse
/// config changes — "retained-shaped": allocate once, reuse in place across frames.
/// See `resource-pool.md §1` for the rationale behind this split from [`ScratchPipelineBuffers`].
pub(crate) struct StablePipelineBuffers {
    pub info_bin_data: Buffer,
    pub tile: Buffer,
    pub segments: Buffer,
    pub ptcl: Buffer,
    /// Used only by fine, but allocated in the shared pre-flush phase to avoid
    /// splitting `prepare` into two phases.
    pub blend_spill: Buffer,
    pub lines: Buffer,
    pub seg_counts: Buffer,
}

impl StablePipelineBuffers {
    fn alloc(
        recorder: &mut FrameRecorder<'_>,
        cached: Option<Self>,
        bs: &ekrano_encoding::BufferSizes,
    ) -> Result<Self, Error> {
        let (c_ibd, c_tile, c_seg, c_ptcl, c_bs, c_lines, c_sc) = match cached {
            Some(c) => (
                Some(c.info_bin_data),
                Some(c.tile),
                Some(c.segments),
                Some(c.ptcl),
                Some(c.blend_spill),
                Some(c.lines),
                Some(c.seg_counts),
            ),
            None => (None, None, None, None, None, None, None),
        };
        Ok(Self {
            info_bin_data: al_cached_opt(recorder, c_ibd, bs.bin_data.size_in_bytes() as u64, 4, "ekrano.info_bin_data_buf")?,
            tile: al_cached_opt(recorder, c_tile, bs.tiles.size_in_bytes().into(), 8, "ekrano.tile_buf")?,
            segments: al_cached_opt(recorder, c_seg, bs.segments.size_in_bytes().into(), 24, "ekrano.segments_buf")?,
            ptcl: al_cached_opt(recorder, c_ptcl, bs.ptcl.size_in_bytes().into(), 4, "ekrano.ptcl_buf")?,
            blend_spill: al_cached_opt(recorder, c_bs, bs.blend_spill.size_in_bytes().into(), size_of::<u32>() as u32, "ekrano.blend_spill")?,
            lines: al_cached_opt(recorder, c_lines, bs.lines.size_in_bytes().into(), 24, "ekrano.lines_buf")?,
            seg_counts: al_cached_opt(recorder, c_sc, bs.seg_counts.size_in_bytes().into(), 8, "ekrano.seg_counts_buf")?,
        })
    }

    fn return_to_pool(self, recorder: &mut FrameRecorder<'_>) {
        let pool = &mut recorder.persistent.pool;
        pool.return_buf(self.info_bin_data, "ekrano.info_bin_data_buf");
        pool.return_buf(self.tile, "ekrano.tile_buf");
        pool.return_buf(self.segments, "ekrano.segments_buf");
        pool.return_buf(self.ptcl, "ekrano.ptcl_buf");
        pool.return_buf(self.blend_spill, "ekrano.blend_spill");
        pool.return_buf(self.lines, "ekrano.lines_buf");
        pool.return_buf(self.seg_counts, "ekrano.seg_counts_buf");
    }

    pub(crate) fn defer_to(self, recorder: &mut FrameRecorder<'_>) {
        recorder.defer_owned_buffer(self.info_bin_data, "ekrano.info_bin_data_buf");
        recorder.defer_owned_buffer(self.tile, "ekrano.tile_buf");
        recorder.defer_owned_buffer(self.segments, "ekrano.segments_buf");
        recorder.defer_owned_buffer(self.ptcl, "ekrano.ptcl_buf");
        recorder.defer_owned_buffer(self.blend_spill, "ekrano.blend_spill");
        recorder.defer_owned_buffer(self.lines, "ekrano.lines_buf");
        recorder.defer_owned_buffer(self.seg_counts, "ekrano.seg_counts_buf");
    }
}

/// The fourteen count-derived scratch buffers whose sizes track scene complexity.
/// These stay in [`crate::goldy_renderer::ResourcePool`], recycled by
/// `{size, access, name, flags}`.
/// See `resource-pool.md §1` for the rationale behind this split from [`StablePipelineBuffers`].
pub(crate) struct ScratchPipelineBuffers {
    pub reduced: Buffer,
    pub reduced2: Buffer,
    pub reduced_scan: Buffer,
    pub tagmonoid: Buffer,
    pub path_bbox: Buffer,
    pub draw_reduced: Buffer,
    pub draw_monoid: Buffer,
    pub clip_inp: Buffer,
    pub clip_el: Buffer,
    pub clip_bic: Buffer,
    pub clip_bbox: Buffer,
    pub draw_bbox: Buffer,
    pub bin_header: Buffer,
    pub path: Buffer,
}

impl ScratchPipelineBuffers {
    fn alloc(
        recorder: &mut FrameRecorder<'_>,
        cached: Option<Self>,
        bs: &ekrano_encoding::BufferSizes,
    ) -> Result<Self, Error> {
        let (
            c_red, c_red2, c_reds, c_tag, c_pbbox,
            c_dred, c_dmon, c_ci, c_ce, c_cb,
            c_cbb, c_db, c_bh, c_path,
        ) = match cached {
            Some(c) => (
                Some(c.reduced), Some(c.reduced2), Some(c.reduced_scan), Some(c.tagmonoid),
                Some(c.path_bbox), Some(c.draw_reduced), Some(c.draw_monoid),
                Some(c.clip_inp), Some(c.clip_el), Some(c.clip_bic),
                Some(c.clip_bbox), Some(c.draw_bbox), Some(c.bin_header), Some(c.path),
            ),
            None => (
                None, None, None, None, None, None, None,
                None, None, None, None, None, None, None,
            ),
        };
        Ok(Self {
            reduced: al_cached_opt(recorder, c_red, bs.path_reduced.size_in_bytes().into(), 20, "ekrano.reduced_buf")?,
            reduced2: al_cached_opt(recorder, c_red2, bs.path_reduced2.size_in_bytes().into(), 20, "ekrano.reduced2_buf")?,
            reduced_scan: al_cached_opt(recorder, c_reds, bs.path_reduced_scan.size_in_bytes().into(), 20, "ekrano.reduced_scan_buf")?,
            tagmonoid: al_cached_opt(recorder, c_tag, bs.path_monoids.size_in_bytes().into(), 20, "ekrano.tagmonoid_buf")?,
            path_bbox: al_cached_opt(recorder, c_pbbox, bs.path_bboxes.size_in_bytes().into(), 24, "ekrano.path_bbox_buf")?,
            draw_reduced: al_cached_opt(recorder, c_dred, bs.draw_reduced.size_in_bytes().into(), 16, "ekrano.draw_reduced_buf")?,
            draw_monoid: al_cached_opt(recorder, c_dmon, bs.draw_monoids.size_in_bytes().into(), 16, "ekrano.draw_monoid_buf")?,
            clip_inp: al_cached_opt(recorder, c_ci, bs.clip_inps.size_in_bytes().into(), 8, "ekrano.clip_inp_buf")?,
            clip_el: al_cached_opt(recorder, c_ce, bs.clip_els.size_in_bytes().into(), 32, "ekrano.clip_el_buf")?,
            clip_bic: al_cached_opt(recorder, c_cb, bs.clip_bics.size_in_bytes().into(), 8, "ekrano.clip_bic_buf")?,
            clip_bbox: al_cached_opt(recorder, c_cbb, bs.clip_bboxes.size_in_bytes().into(), 16, "ekrano.clip_bbox_buf")?,
            draw_bbox: al_cached_opt(recorder, c_db, bs.draw_bboxes.size_in_bytes().into(), 16, "ekrano.draw_bbox_buf")?,
            bin_header: al_cached_opt(recorder, c_bh, bs.bin_headers.size_in_bytes().into(), 8, "ekrano.bin_header_buf")?,
            path: al_cached_opt(recorder, c_path, bs.paths.size_in_bytes().into(), 32, "ekrano.path_buf")?,
        })
    }

    fn return_to_pool(self, recorder: &mut FrameRecorder<'_>) {
        let pool = &mut recorder.persistent.pool;
        pool.return_buf(self.reduced, "ekrano.reduced_buf");
        pool.return_buf(self.reduced2, "ekrano.reduced2_buf");
        pool.return_buf(self.reduced_scan, "ekrano.reduced_scan_buf");
        pool.return_buf(self.tagmonoid, "ekrano.tagmonoid_buf");
        pool.return_buf(self.path_bbox, "ekrano.path_bbox_buf");
        pool.return_buf(self.draw_reduced, "ekrano.draw_reduced_buf");
        pool.return_buf(self.draw_monoid, "ekrano.draw_monoid_buf");
        pool.return_buf(self.clip_inp, "ekrano.clip_inp_buf");
        pool.return_buf(self.clip_el, "ekrano.clip_el_buf");
        pool.return_buf(self.clip_bic, "ekrano.clip_bic_buf");
        pool.return_buf(self.clip_bbox, "ekrano.clip_bbox_buf");
        pool.return_buf(self.draw_bbox, "ekrano.draw_bbox_buf");
        pool.return_buf(self.bin_header, "ekrano.bin_header_buf");
        pool.return_buf(self.path, "ekrano.path_buf");
    }

    pub(crate) fn defer_to(self, recorder: &mut FrameRecorder<'_>) {
        recorder.defer_owned_buffer(self.reduced, "ekrano.reduced_buf");
        recorder.defer_owned_buffer(self.reduced2, "ekrano.reduced2_buf");
        recorder.defer_owned_buffer(self.reduced_scan, "ekrano.reduced_scan_buf");
        recorder.defer_owned_buffer(self.tagmonoid, "ekrano.tagmonoid_buf");
        recorder.defer_owned_buffer(self.path_bbox, "ekrano.path_bbox_buf");
        recorder.defer_owned_buffer(self.draw_reduced, "ekrano.draw_reduced_buf");
        recorder.defer_owned_buffer(self.draw_monoid, "ekrano.draw_monoid_buf");
        recorder.defer_owned_buffer(self.clip_inp, "ekrano.clip_inp_buf");
        recorder.defer_owned_buffer(self.clip_el, "ekrano.clip_el_buf");
        recorder.defer_owned_buffer(self.clip_bic, "ekrano.clip_bic_buf");
        recorder.defer_owned_buffer(self.clip_bbox, "ekrano.clip_bbox_buf");
        recorder.defer_owned_buffer(self.draw_bbox, "ekrano.draw_bbox_buf");
        recorder.defer_owned_buffer(self.bin_header, "ekrano.bin_header_buf");
        recorder.defer_owned_buffer(self.path, "ekrano.path_buf");
    }
}

/// Cached GPU buffers that survive across frames when `buffer_sizes` is stable.
///
/// At depth=1 the previous frame's GPU work is complete by the time `begin_frame`
/// returns, so these buffers are safe to rebind immediately. All handles are
/// persistent `ResourcePool` allocations with stable bindless indices.
pub(crate) struct CachedPipeline {
    pub stable: StablePipelineBuffers,
    pub scratch: ScratchPipelineBuffers,
    pub buffer_sizes: ekrano_encoding::BufferSizes,
}

pub(crate) struct PipelineResources {
    pub gradient: Texture,
    pub image_atlas: Texture,
    pub mask_atlas: Texture,
    pub scene: Buffer,
    pub config: Buffer,
    pub indirect: Option<Buffer>,
    pub stable: StablePipelineBuffers,
    pub scratch: ScratchPipelineBuffers,
    pub bump: Buffer,
    pub out_image: Texture,
    pub filter_layers: [Texture; 4],
    /// Buffer sizes used this frame, stored for cache-key comparison next frame.
    pub buffer_sizes: ekrano_encoding::BufferSizes,
    /// The `ConfigUniform` value uploaded to `config`, stored so that
    /// `schedule_pipeline_cleanup` can stash the buffer back into
    /// `PersistentState::cached_config_uniform` without re-reading GPU memory.
    pub config_uniform_value: ekrano_encoding::ConfigUniform,
}

impl PipelineResources {
    pub(crate) fn prepare(
        recorder: &mut FrameRecorder<'_>,
        coverage_mask: Option<&CoverageMask>,
        mut packed: Vec<u8>,
        ramps: Ramps<'_>,
        images: Images<'_>,
        params: &RenderParams,
        config: &RenderConfig,
        out_image_format: TextureFormat,
    ) -> Result<Self, Error> {
        if packed.is_empty() {
            packed.resize(size_of::<u32>(), u8::MAX);
        }

        let gpu_progress = recorder.context().gpu_progress();
        log::debug!("[RT-CACHE] gpu_progress={gpu_progress} at prepare entry");

        let mut cpu_config_owned = *config;
        if coverage_mask.is_some() {
            cpu_config_owned.gpu.mask_active = 1;
        }
        if let Some(m) = coverage_mask {
            assert_eq!(m.width, params.width, "coverage_mask width must match render width");
            assert_eq!(m.height, params.height, "coverage_mask height must match render height");
        }

        let gradient = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.gradient");
            if ramps.height == 0 {
                acquire_texture_rgba(recorder, 1, 1, TextureKind::Interpolated, TextureFlags::COPY_DST)?
            } else {
                let data: &[u8] = bytemuck::cast_slice(ramps.data);
                record_upload_image(recorder, ramps.width, ramps.height, ImageFormat::Rgba8, data)?
            }
        };

        let (image_atlas, _) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.image_atlas");
            if images.images.is_empty() {
                let t = acquire_texture_rgba(
                    recorder,
                    1,
                    1,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )?;
                (t, (1_u32, 1_u32))
            } else {
                let t = acquire_texture_rgba(
                    recorder,
                    images.width,
                    images.height,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )?;
                for image in images.images {
                    write_image_region(recorder, &t, image.1, image.2, &image.0)?;
                }
                (t, (images.width, images.height))
            }
        };

        let mask_atlas = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.mask_atlas");
            match coverage_mask {
                Some(m) => {
                    let mut rgba = Vec::with_capacity(m.data.len() * 4);
                    for &b in m.data.iter() {
                        rgba.extend_from_slice(&[b, b, b, 255]);
                    }
                    record_upload_image(recorder, m.width, m.height, ImageFormat::Rgba8, &rgba)?
                }
                None => record_upload_image(recorder, 1, 1, ImageFormat::Rgba8, &[255, 255, 255, 255])?,
            }
        };

        // Move `packed` directly into the graph write node — avoids the redundant
        // `to_vec()` copy that `record_upload_bytes` would perform on a borrow.
        let scene = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload");
            record_upload_bytes_owned(recorder, "ekrano.scene", 4, packed)?
        };

        let config_uniform_value = cpu_config_owned.gpu;

        // Cache check: reuse the previous frame's GPU config buffer when the value is
        // identical (steady state after bump estimates converge). On a cache hit no
        // WriteBuffer node is added to the graph, eliminating a staging-belt round-trip.
        let config = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload");
            let cache_hit = recorder
                .persistent
                .cached_config_uniform
                .as_ref()
                .is_some_and(|(v, _)| v == &config_uniform_value);
            log::trace!("ConfigUniform cache {}", if cache_hit { "HIT" } else { "MISS" });
            if cache_hit {
                recorder.persistent.cached_config_uniform.take().unwrap().1
            } else if let Some((_, existing_buf)) = recorder.persistent.cached_config_uniform.take() {
                // Buffer size is constant (sizeof ConfigUniform); reuse the allocation
                // and just overwrite with the new value.
                recorder
                    .graph()
                    .write_buffer(&existing_buf, 0, bytemuck::bytes_of(&config_uniform_value).to_vec());
                existing_buf
            } else {
                record_upload_bytes(
                    recorder,
                    "ekrano.config",
                    size_of::<ekrano_encoding::ConfigUniform>() as u32,
                    bytemuck::bytes_of(&config_uniform_value),
                )?
            }
        };

        let buffer_sizes = cpu_config_owned.buffer_sizes;

        // Try to reuse cached pipeline buffers from the previous frame.
        // At depth=1, begin_frame blocks until the previous frame's GPU work is complete,
        // so these buffers are safe to rebind immediately without any fence check.
        let (cached_stable, cached_scratch) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.pipeline_cache");
            match recorder.persistent.take_cached_pipeline(gpu_progress) {
                Some(c) if c.buffer_sizes == buffer_sizes => (Some(c.stable), Some(c.scratch)),
                Some(c) => {
                    // Sizes changed: return stale buffers to pool before discarding.
                    log::debug!("[PIPE-CACHE] buffer_sizes mismatch — returning stale buffers to pool");
                    c.stable.return_to_pool(recorder);
                    c.scratch.return_to_pool(recorder);
                    (None, None)
                }
                None => (None, None),
            }
        }; // end ekrano.prepare.pipeline_cache zone

        let _tz_alloc = goldy::tracy_zone!("ekrano.prepare.alloc_buffers");
        // Reuse from cache when sizes match (no ResourcePool round-trip). These buffers
        // are fully GPU-overwritten before first read.
        let stable = StablePipelineBuffers::alloc(recorder, cached_stable, &buffer_sizes)?;
        let scratch = ScratchPipelineBuffers::alloc(recorder, cached_scratch, &buffer_sizes)?;
        let bump = alloc_pipeline_buffer(
            recorder,
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            size_of::<BumpAllocators>() as u32,
            "ekrano.bump_buf",
            BufferFlags::CPU_READABLE,
        )?;
        clear_gpu_buf(recorder, &bump, 0, None)?;

        // Try to reuse cached render targets from the previous frame (avoids TexturePool
        // round-trips when render dimensions are stable across frames).
        let (out_image, filter_layers) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.render_targets");
            if let Some((cached_out, cached_layers)) = recorder.persistent.take_cached_render_targets(
                gpu_progress,
                params.width,
                params.height,
                out_image_format,
            ) {
                (cached_out, cached_layers)
            } else {
                let _tz2 = goldy::tracy_zone!("ekrano.prepare.render_targets.ALLOC");
                let out = recorder
                    .persistent
                    .tex_pool
                    .acquire(
                        recorder.device(),
                        params.width,
                        params.height,
                        out_image_format,
                        TextureKind::Direct,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )
                    .map_err(|e| Error::Shader(e.to_string()))?;
                let layers = std::array::from_fn(|_| {
                    acquire_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )
                    .expect("filter layer")
                });
                (out, layers)
            }
        };

        Ok(Self {
            gradient,
            image_atlas,
            mask_atlas,
            scene,
            config,
            indirect: None,
            stable,
            scratch,
            bump,
            out_image,
            filter_layers,
            buffer_sizes,
            config_uniform_value,
        })
    }
}

/// Fill `out` with bindless slot indices for `bindings`, reusing its existing allocation.
///
/// Clears `out` before filling so the caller may pass a scratch `Vec<u32>` that already
/// has capacity from a previous call, avoiding a heap allocation per dispatch.
pub(crate) fn collect_bindless_indices_into(
    out: &mut Vec<u32>,
    bindings: &[GpuBinding<'_>],
    bind_types: &[BindType],
    max_slots: usize,
) -> Result<(), Error> {
    out.clear();
    for (i, binding) in bindings.iter().enumerate() {
        let is_read_only = matches!(bind_types.get(i), Some(BindType::BufReadOnly));
        let is_sampled_image = matches!(bind_types.get(i), Some(BindType::ImageRead(_)));
        let idx = match binding {
            GpuBinding::Buf(_) => binding.bindless_slot(is_read_only)?,
            GpuBinding::Tex(tex) if is_sampled_image => tex
                .resource_index(ResourceAccess::Read)
                .or_else(|| {
                    // Direct storage images have no separate sampled slot; use the primary
                    // storage index (legacy bindless_sampled_index().or_else(bindless_index)).
                    tex.resource_index(ResourceAccess::Write)
                        .or_else(|| tex.resource_index(ResourceAccess::ReadWrite))
                })
                .ok_or_else(|| Error::Shader("resource sampled index missing for ImageRead texture binding".into()))?,
            GpuBinding::Tex(_) => binding.bindless_slot(false)?,
            GpuBinding::Sampler(idx) | GpuBinding::PersistentBuf(idx) => *idx,
        };
        out.push(idx);
    }
    if out.len() > max_slots {
        return Err(Error::Shader(format!(
            "shader requires {} bindless slots, exceeds limit of {}",
            out.len(),
            max_slots
        )));
    }
    Ok(())
}

pub(crate) fn bind_type_to_node_access(bt: BindType) -> NodeAccess {
    match bt {
        BindType::Buffer | BindType::Image(_) => NodeAccess::ReadWrite,
        BindType::BufReadOnly | BindType::Uniform | BindType::ImageRead(_) => NodeAccess::Read,
        BindType::Sampler => NodeAccess::Read,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helpers that return a fixed bindless index without touching the GPU.
    fn sampler_binding(idx: u32) -> GpuBinding<'static> {
        GpuBinding::Sampler(idx)
    }

    #[test]
    fn collect_into_sampler_indices() {
        let bindings = [sampler_binding(7), sampler_binding(3)];
        let bind_types = [BindType::Sampler, BindType::Sampler];

        let mut out = Vec::new();
        collect_bindless_indices_into(&mut out, &bindings, &bind_types, 16).unwrap();

        assert_eq!(out, [7, 3]);
    }

    #[test]
    fn collect_into_clears_previous_contents() {
        let bindings = [sampler_binding(1)];
        let bind_types = [BindType::Sampler];

        let mut out = vec![99_u32; 5];
        collect_bindless_indices_into(&mut out, &bindings, &bind_types, 16).unwrap();

        assert_eq!(out, [1]);
    }

    #[test]
    fn collect_into_reuses_capacity() {
        let mut out: Vec<u32> = Vec::with_capacity(16);

        for _ in 0..3 {
            let bindings = [sampler_binding(5), sampler_binding(6)];
            let bind_types = [BindType::Sampler, BindType::Sampler];
            collect_bindless_indices_into(&mut out, &bindings, &bind_types, 16).unwrap();
            assert_eq!(out, [5, 6]);
        }
    }

    #[test]
    fn collect_into_exceeds_max_slots_returns_err() {
        let bindings = [sampler_binding(0), sampler_binding(1), sampler_binding(2)];
        let bind_types = [BindType::Sampler; 3];

        let mut out = Vec::new();
        let result = collect_bindless_indices_into(&mut out, &bindings, &bind_types, 2);

        assert!(result.is_err(), "expected Err when bindings exceed max_slots");
    }

    #[test]
    fn collect_into_empty_bindings() {
        let mut out = vec![99_u32];
        collect_bindless_indices_into(&mut out, &[], &[], 16).unwrap();
        assert!(out.is_empty());
    }
}
