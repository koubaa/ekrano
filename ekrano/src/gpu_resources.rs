// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Direct GPU resource handles (no bind-map / proxies).

use std::mem::size_of;

use goldy::task_graph::NodeAccess;
use goldy::types::{BufferFlags, ResourceAccess, TextureFlags, TextureKind};
use goldy::{Buffer, BufferKind, Context, Device, TaskGraph, Texture, TextureFormat};

use crate::goldy_renderer::PersistentState;
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
        idx.ok_or_else(|| {
            Error::Shader("bindless index missing for shader resource binding".into())
        })
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
    device: &Device,
    ctx: &Context,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    size: u64,
    stride: u32,
    name: &'static str,
    flags: BufferFlags,
) -> Result<Buffer, Error> {
    let buf = persistent.pool.get_buf_with_stride(
        device,
        ctx,
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
        graph.clear_buffer(&buf, 0, size);
    }
    Ok(buf)
}

pub(crate) fn record_upload_bytes(
    device: &Device,
    ctx: &Context,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    name: &'static str,
    element_stride: u32,
    bytes: &[u8],
) -> Result<Buffer, Error> {
    let buf = persistent.pool.get_buf_with_stride(
        device,
        ctx,
        bytes.len() as u64,
        name,
        BufferKind::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    graph.write_buffer(&buf, 0, bytes.to_vec());
    Ok(buf)
}

/// Like [`record_upload_bytes`] but takes ownership of the byte vector, avoiding
/// the redundant `to_vec()` copy when the caller already holds an owned `Vec<u8>`.
pub(crate) fn record_upload_bytes_owned(
    device: &Device,
    ctx: &Context,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    name: &'static str,
    element_stride: u32,
    bytes: Vec<u8>,
) -> Result<Buffer, Error> {
    let buf = persistent.pool.get_buf_with_stride(
        device,
        ctx,
        bytes.len() as u64,
        name,
        BufferKind::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    graph.write_buffer(&buf, 0, bytes);
    Ok(buf)
}

pub(crate) fn record_upload_image(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    width: u32,
    height: u32,
    format: ImageFormat,
    bytes: &[u8],
) -> Result<Texture, Error> {
    let format = image_fmt_goldy(format);
    let texture = persistent
        .tex_pool
        .acquire(
            device,
            width,
            height,
            format,
            TextureKind::Interpolated,
            TextureFlags::COPY_DST,
        )
        .map_err(|e| Error::Shader(e.to_string()))?;
    graph
        .write_texture(&texture, bytes.to_vec())
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(texture)
}

pub(crate) fn write_image_region(
    graph: &mut TaskGraph,
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
    device: &Device,
    persistent: &mut PersistentState,
    width: u32,
    height: u32,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    persistent
        .tex_pool
        .acquire(
            device,
            width,
            height,
            TextureFormat::Rgba8Unorm,
            access,
            flags,
        )
        .map_err(|e| Error::Shader(e.to_string()))
}

pub(crate) fn clear_gpu_buf(
    graph: &mut TaskGraph,
    buf: &Buffer,
    off: u64,
    size: Option<u64>,
) -> Result<(), Error> {
    let sz = size.unwrap_or_else(|| buf.size().saturating_sub(off));
    graph.clear_buffer(buf, off, sz);
    Ok(())
}

/// Cached GPU buffers that survive across frames when `buffer_sizes` is stable.
///
/// At depth=1 the previous frame's GPU work is complete by the time `begin_frame`
/// returns, so these buffers are safe to rebind immediately. All handles are
/// persistent `ResourcePool` allocations with stable bindless indices.
pub(crate) struct CachedPipeline {
    pub info_bin_data: Buffer,
    pub tile: Buffer,
    pub segments: Buffer,
    pub ptcl: Buffer,
    pub blend_spill: Buffer,
    pub reduced: Buffer,
    pub reduced2: Buffer,
    pub reduced_scan: Buffer,
    pub tagmonoid: Buffer,
    pub path_bbox: Buffer,
    pub lines: Buffer,
    pub draw_reduced: Buffer,
    pub draw_monoid: Buffer,
    pub clip_inp: Buffer,
    pub clip_el: Buffer,
    pub clip_bic: Buffer,
    pub clip_bbox: Buffer,
    pub draw_bbox: Buffer,
    pub bin_header: Buffer,
    pub path: Buffer,
    pub seg_counts: Buffer,
    pub buffer_sizes: ekrano_encoding::BufferSizes,
}

pub(crate) struct PipelineResources {
    pub gradient: Texture,
    pub image_atlas: Texture,
    pub mask_atlas: Texture,
    pub scene: Buffer,
    pub config: Buffer,
    pub indirect: Option<Buffer>,
    pub info_bin_data: Buffer,
    pub tile: Buffer,
    pub segments: Buffer,
    pub ptcl: Buffer,
    pub reduced: Buffer,
    pub reduced2: Buffer,
    pub reduced_scan: Buffer,
    pub tagmonoid: Buffer,
    pub path_bbox: Buffer,
    pub bump: Buffer,
    pub lines: Buffer,
    pub draw_reduced: Buffer,
    pub draw_monoid: Buffer,
    pub clip_inp: Buffer,
    pub clip_el: Buffer,
    pub clip_bic: Buffer,
    pub clip_bbox: Buffer,
    pub draw_bbox: Buffer,
    pub bin_header: Buffer,
    pub path: Buffer,
    pub seg_counts: Buffer,
    pub blend_spill: Buffer,
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
    #[allow(
        clippy::too_many_arguments,
        reason = "Single setup function threads every pipeline buffer and texture from resolve data"
    )]
    pub(crate) fn prepare(
        device: &Device,
        ctx: &Context,
        graph: &mut TaskGraph,
        persistent: &mut PersistentState,
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

        let gpu_progress = ctx.gpu_progress();
        log::debug!("[RT-CACHE] gpu_progress={gpu_progress} at prepare entry");

        let mut cpu_config_owned = *config;
        if coverage_mask.is_some() {
            cpu_config_owned.gpu.mask_active = 1;
        }
        if let Some(m) = coverage_mask {
            assert_eq!(
                m.width, params.width,
                "coverage_mask width must match render width"
            );
            assert_eq!(
                m.height, params.height,
                "coverage_mask height must match render height"
            );
        }

        let gradient = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.gradient");
            if ramps.height == 0 {
                acquire_texture_rgba(
                    device,
                    persistent,
                    1,
                    1,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST,
                )?
            } else {
                let data: &[u8] = bytemuck::cast_slice(ramps.data);
                record_upload_image(
                    device,
                    graph,
                    persistent,
                    ramps.width,
                    ramps.height,
                    ImageFormat::Rgba8,
                    data,
                )?
            }
        };

        let (image_atlas, _) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.image_atlas");
            if images.images.is_empty() {
                let t = acquire_texture_rgba(
                    device,
                    persistent,
                    1,
                    1,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )?;
                (t, (1_u32, 1_u32))
            } else {
                let t = acquire_texture_rgba(
                    device,
                    persistent,
                    images.width,
                    images.height,
                    TextureKind::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )?;
                for image in images.images {
                    write_image_region(graph, &t, image.1, image.2, &image.0)?;
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
                    record_upload_image(
                        device,
                        graph,
                        persistent,
                        m.width,
                        m.height,
                        ImageFormat::Rgba8,
                        &rgba,
                    )?
                }
                None => record_upload_image(
                    device,
                    graph,
                    persistent,
                    1,
                    1,
                    ImageFormat::Rgba8,
                    &[255, 255, 255, 255],
                )?,
            }
        };

        // Move `packed` directly into the graph write node — avoids the redundant
        // `to_vec()` copy that `record_upload_bytes` would perform on a borrow.
        let scene = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload");
            record_upload_bytes_owned(device, ctx, graph, persistent, "ekrano.scene", 4, packed)?
        };

        let config_uniform_value = cpu_config_owned.gpu;

        // Cache check: reuse the previous frame's GPU config buffer when the value is
        // identical (steady state after bump estimates converge). On a cache hit no
        // WriteBuffer node is added to the graph, eliminating a staging-belt round-trip.
        let config = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload");
            let cache_hit = persistent
                .cached_config_uniform
                .as_ref()
                .is_some_and(|(v, _)| v == &config_uniform_value);
            log::trace!(
                "ConfigUniform cache {}",
                if cache_hit { "HIT" } else { "MISS" }
            );
            if cache_hit {
                persistent.cached_config_uniform.take().unwrap().1
            } else if let Some((_, existing_buf)) = persistent.cached_config_uniform.take() {
                // Buffer size is constant (sizeof ConfigUniform); reuse the allocation
                // and just overwrite with the new value.
                graph.write_buffer(
                    &existing_buf,
                    0,
                    bytemuck::bytes_of(&config_uniform_value).to_vec(),
                );
                existing_buf
            } else {
                record_upload_bytes(
                    device,
                    ctx,
                    graph,
                    persistent,
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
        struct CachedOwnedBuffers {
            info_bin_data: Option<Buffer>,
            tile: Option<Buffer>,
            segments: Option<Buffer>,
            ptcl: Option<Buffer>,
            blend_spill: Option<Buffer>,
            reduced: Option<Buffer>,
            reduced2: Option<Buffer>,
            reduced_scan: Option<Buffer>,
            tagmonoid: Option<Buffer>,
            path_bbox: Option<Buffer>,
            lines: Option<Buffer>,
            draw_reduced: Option<Buffer>,
            draw_monoid: Option<Buffer>,
            clip_inp: Option<Buffer>,
            clip_el: Option<Buffer>,
            clip_bic: Option<Buffer>,
            clip_bbox: Option<Buffer>,
            draw_bbox: Option<Buffer>,
            bin_header: Option<Buffer>,
            path: Option<Buffer>,
            seg_counts: Option<Buffer>,
        }
        let cached = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.pipeline_cache");
            match persistent.take_cached_pipeline(gpu_progress) {
                Some(c) if c.buffer_sizes == buffer_sizes => CachedOwnedBuffers {
                    info_bin_data: Some(c.info_bin_data),
                    tile: Some(c.tile),
                    segments: Some(c.segments),
                    ptcl: Some(c.ptcl),
                    blend_spill: Some(c.blend_spill),
                    reduced: Some(c.reduced),
                    reduced2: Some(c.reduced2),
                    reduced_scan: Some(c.reduced_scan),
                    tagmonoid: Some(c.tagmonoid),
                    path_bbox: Some(c.path_bbox),
                    lines: Some(c.lines),
                    draw_reduced: Some(c.draw_reduced),
                    draw_monoid: Some(c.draw_monoid),
                    clip_inp: Some(c.clip_inp),
                    clip_el: Some(c.clip_el),
                    clip_bic: Some(c.clip_bic),
                    clip_bbox: Some(c.clip_bbox),
                    draw_bbox: Some(c.draw_bbox),
                    bin_header: Some(c.bin_header),
                    path: Some(c.path),
                    seg_counts: Some(c.seg_counts),
                },
                Some(c) => {
                    // Sizes changed: return stale buffers to pool before discarding.
                    persistent
                        .pool
                        .return_buf(c.info_bin_data, "ekrano.info_bin_data_buf");
                    persistent.pool.return_buf(c.tile, "ekrano.tile_buf");
                    persistent
                        .pool
                        .return_buf(c.segments, "ekrano.segments_buf");
                    persistent.pool.return_buf(c.ptcl, "ekrano.ptcl_buf");
                    persistent
                        .pool
                        .return_buf(c.blend_spill, "ekrano.blend_spill");
                    persistent.pool.return_buf(c.reduced, "ekrano.reduced_buf");
                    persistent
                        .pool
                        .return_buf(c.reduced2, "ekrano.reduced2_buf");
                    persistent
                        .pool
                        .return_buf(c.reduced_scan, "ekrano.reduced_scan_buf");
                    persistent
                        .pool
                        .return_buf(c.tagmonoid, "ekrano.tagmonoid_buf");
                    persistent
                        .pool
                        .return_buf(c.path_bbox, "ekrano.path_bbox_buf");
                    persistent.pool.return_buf(c.lines, "ekrano.lines_buf");
                    persistent
                        .pool
                        .return_buf(c.draw_reduced, "ekrano.draw_reduced_buf");
                    persistent
                        .pool
                        .return_buf(c.draw_monoid, "ekrano.draw_monoid_buf");
                    persistent
                        .pool
                        .return_buf(c.clip_inp, "ekrano.clip_inp_buf");
                    persistent.pool.return_buf(c.clip_el, "ekrano.clip_el_buf");
                    persistent
                        .pool
                        .return_buf(c.clip_bic, "ekrano.clip_bic_buf");
                    persistent
                        .pool
                        .return_buf(c.clip_bbox, "ekrano.clip_bbox_buf");
                    persistent
                        .pool
                        .return_buf(c.draw_bbox, "ekrano.draw_bbox_buf");
                    persistent
                        .pool
                        .return_buf(c.bin_header, "ekrano.bin_header_buf");
                    persistent.pool.return_buf(c.path, "ekrano.path_buf");
                    persistent
                        .pool
                        .return_buf(c.seg_counts, "ekrano.seg_counts_buf");
                    CachedOwnedBuffers {
                        info_bin_data: None,
                        tile: None,
                        segments: None,
                        ptcl: None,
                        blend_spill: None,
                        reduced: None,
                        reduced2: None,
                        reduced_scan: None,
                        tagmonoid: None,
                        path_bbox: None,
                        lines: None,
                        draw_reduced: None,
                        draw_monoid: None,
                        clip_inp: None,
                        clip_el: None,
                        clip_bic: None,
                        clip_bbox: None,
                        draw_bbox: None,
                        bin_header: None,
                        path: None,
                        seg_counts: None,
                    }
                }
                None => CachedOwnedBuffers {
                    info_bin_data: None,
                    tile: None,
                    segments: None,
                    ptcl: None,
                    blend_spill: None,
                    reduced: None,
                    reduced2: None,
                    reduced_scan: None,
                    tagmonoid: None,
                    path_bbox: None,
                    lines: None,
                    draw_reduced: None,
                    draw_monoid: None,
                    clip_inp: None,
                    clip_el: None,
                    clip_bic: None,
                    clip_bbox: None,
                    draw_bbox: None,
                    bin_header: None,
                    path: None,
                    seg_counts: None,
                },
            }
        }; // end ekrano.prepare.pipeline_cache zone

        let _tz_alloc = goldy::tracy_zone!("ekrano.prepare.alloc_buffers");
        // Reuse from cache when sizes match (no ResourcePool round-trip). These buffers
        // are fully GPU-overwritten before first read.
        macro_rules! al_cached {
            ($cached_opt:expr, $sz:expr, $stride:expr, $name:expr) => {
                match $cached_opt {
                    Some(buf) => buf,
                    None => alloc_pipeline_buffer(
                        device,
                        ctx,
                        graph,
                        persistent,
                        $sz,
                        $stride,
                        $name,
                        BufferFlags::empty(),
                    )?,
                }
            };
        }

        // Shared: written by coarse, read by fine.
        let info_bin_data = al_cached!(
            cached.info_bin_data,
            buffer_sizes.bin_data.size_in_bytes() as u64,
            4,
            "ekrano.info_bin_data_buf"
        );
        let tile = al_cached!(
            cached.tile,
            buffer_sizes.tiles.size_in_bytes().into(),
            8,
            "ekrano.tile_buf"
        );
        let segments = al_cached!(
            cached.segments,
            buffer_sizes.segments.size_in_bytes().into(),
            24,
            "ekrano.segments_buf"
        );
        let ptcl = al_cached!(
            cached.ptcl,
            buffer_sizes.ptcl.size_in_bytes().into(),
            4,
            "ekrano.ptcl_buf"
        );
        let reduced = al_cached!(
            cached.reduced,
            buffer_sizes.path_reduced.size_in_bytes().into(),
            20,
            "ekrano.reduced_buf"
        );
        let reduced2 = al_cached!(
            cached.reduced2,
            buffer_sizes.path_reduced2.size_in_bytes().into(),
            20,
            "ekrano.reduced2_buf"
        );
        let reduced_scan = al_cached!(
            cached.reduced_scan,
            buffer_sizes.path_reduced_scan.size_in_bytes().into(),
            20,
            "ekrano.reduced_scan_buf"
        );
        let tagmonoid = al_cached!(
            cached.tagmonoid,
            buffer_sizes.path_monoids.size_in_bytes().into(),
            20,
            "ekrano.tagmonoid_buf"
        );
        let path_bbox = al_cached!(
            cached.path_bbox,
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            24,
            "ekrano.path_bbox_buf"
        );
        let bump = alloc_pipeline_buffer(
            device,
            ctx,
            graph,
            persistent,
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            size_of::<BumpAllocators>() as u32,
            "ekrano.bump_buf",
            BufferFlags::CPU_READABLE,
        )?;
        clear_gpu_buf(graph, &bump, 0, None)?;
        let lines = al_cached!(
            cached.lines,
            buffer_sizes.lines.size_in_bytes().into(),
            24,
            "ekrano.lines_buf"
        );
        let draw_reduced = al_cached!(
            cached.draw_reduced,
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            16,
            "ekrano.draw_reduced_buf"
        );
        let draw_monoid = al_cached!(
            cached.draw_monoid,
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            16,
            "ekrano.draw_monoid_buf"
        );
        let clip_inp = al_cached!(
            cached.clip_inp,
            buffer_sizes.clip_inps.size_in_bytes().into(),
            8,
            "ekrano.clip_inp_buf"
        );
        let clip_el = al_cached!(
            cached.clip_el,
            buffer_sizes.clip_els.size_in_bytes().into(),
            32,
            "ekrano.clip_el_buf"
        );
        let clip_bic = al_cached!(
            cached.clip_bic,
            buffer_sizes.clip_bics.size_in_bytes().into(),
            8,
            "ekrano.clip_bic_buf"
        );
        let clip_bbox = al_cached!(
            cached.clip_bbox,
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            16,
            "ekrano.clip_bbox_buf"
        );
        let draw_bbox = al_cached!(
            cached.draw_bbox,
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            16,
            "ekrano.draw_bbox_buf"
        );
        let bin_header = al_cached!(
            cached.bin_header,
            buffer_sizes.bin_headers.size_in_bytes().into(),
            8,
            "ekrano.bin_header_buf"
        );
        let path = al_cached!(
            cached.path,
            buffer_sizes.paths.size_in_bytes().into(),
            32,
            "ekrano.path_buf"
        );
        let seg_counts = al_cached!(
            cached.seg_counts,
            buffer_sizes.seg_counts.size_in_bytes().into(),
            8,
            "ekrano.seg_counts_buf"
        );
        // blend_spill is used only by fine, but allocating it as Shared (pre-flush)
        // avoids the need to split prepare() into two phases.
        let blend_spill = al_cached!(
            cached.blend_spill,
            buffer_sizes.blend_spill.size_in_bytes().into(),
            size_of::<u32>() as u32,
            "ekrano.blend_spill"
        );

        // Try to reuse cached render targets from the previous frame (avoids TexturePool
        // round-trips when render dimensions are stable across frames).
        let (out_image, filter_layers) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.render_targets");
            if let Some((cached_out, cached_layers)) = persistent.take_cached_render_targets(
                gpu_progress,
                params.width,
                params.height,
                out_image_format,
            ) {
                (cached_out, cached_layers)
            } else {
                let _tz2 = goldy::tracy_zone!("ekrano.prepare.render_targets.ALLOC");
                let out = persistent
                    .tex_pool
                    .acquire(
                        device,
                        params.width,
                        params.height,
                        out_image_format,
                        TextureKind::Direct,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )
                    .map_err(|e| Error::Shader(e.to_string()))?;
                let layers = std::array::from_fn(|_| {
                    acquire_texture_rgba(
                        device,
                        persistent,
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
                .ok_or_else(|| {
                    Error::Shader(
                        "resource sampled index missing for ImageRead texture binding".into(),
                    )
                })?,
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

        assert!(
            result.is_err(),
            "expected Err when bindings exceed max_slots"
        );
    }

    #[test]
    fn collect_into_empty_bindings() {
        let mut out = vec![99_u32];
        collect_bindless_indices_into(&mut out, &[], &[], 16).unwrap();
        assert!(out.is_empty());
    }
}
