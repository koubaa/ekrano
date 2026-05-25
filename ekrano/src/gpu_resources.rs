// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Direct GPU resource handles (no bind-map / proxies).

use std::mem::size_of;

use goldy::task_graph::{NodeAccess, TransientId};
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags};
use goldy::{
    Buffer, BufferView, DataAccess, Device, DeviceType, TaskGraph, Texture, TextureFormat,
};

/// Sentinel bindless index for transient buffers whose real slot is resolved at
/// flush time after graph coloring. Must not collide with valid slot indices.
pub(crate) const TRANSIENT_SLOT_PLACEHOLDER: u32 = u32::MAX;

use crate::goldy_renderer::PersistentState;
use crate::resource_proxy::{BindType, ImageFormat};
use crate::{Error, RenderParams, Result};
use ekrano_encoding::{BumpAllocators, Encoding, Images, IndirectCount, Ramps, RenderConfig};

pub(crate) enum GpuBuf {
    Owned(Buffer),
    Pooled(BufferView),
    /// Graph-scoped transient: physical allocation is deferred until graph flush,
    /// when wave-lifetime coloring packs non-overlapping buffers into the same offset.
    Transient(TransientId),
}

impl GpuBuf {
    pub(crate) fn as_indirect_buffer(&self) -> Option<&Buffer> {
        match self {
            Self::Owned(b) => Some(b),
            Self::Pooled(_) | Self::Transient(_) => None,
        }
    }

    pub(crate) fn as_binding(&self) -> GpuBinding<'_> {
        match self {
            Self::Owned(b) => GpuBinding::Buf(b),
            Self::Pooled(v) => GpuBinding::View(v),
            Self::Transient(id) => GpuBinding::Transient(*id),
        }
    }
}

pub(crate) enum GpuBinding<'a> {
    Buf(&'a Buffer),
    View(&'a BufferView),
    Tex(&'a Texture),
    /// Deferred: physical allocation happens at graph flush after coloring.
    Transient(TransientId),
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
                    buf.bindless_srv_index()
                } else {
                    buf.bindless_index()
                }
            }
            GpuBinding::View(view) => {
                if is_read_only {
                    view.bindless_srv_index()
                } else {
                    view.bindless_index()
                }
            }
            GpuBinding::Tex(tex) => tex.bindless_index(),
            GpuBinding::Transient(_) => return Ok(TRANSIENT_SLOT_PLACEHOLDER),
            GpuBinding::Sampler(idx) | GpuBinding::PersistentBuf(idx) => return Ok(*idx),
        };
        idx.ok_or_else(|| {
            Error::Shader("bindless index missing for shader resource binding".into())
        })
    }
}

fn use_pool(device: &Device) -> bool {
    device.device_type() != DeviceType::Cpu
}

fn is_pool_exempt(name: &'static str) -> bool {
    matches!(
        name,
        "ekrano.bump_buf" | "ekrano.indirect_count" | "ekrano.indirect_dispatch"
    )
}

/// Controls how a pipeline buffer is allocated.
///
/// `CoarseOnly` buffers use graph-transient aliasing (via wave-interval coloring) when
/// the `FrameStrategy` enables graph coloring (depth > 1), enabling inter-frame VRAM
/// reuse across pipelined frames. At `LowLatency` (depth=1) they are promoted to
/// persistent `ResourcePool` buffers so their bindless indices are stable — a
/// prerequisite for command buffer retention.
/// `Shared`/`OwnedShared` buffers are always real GPU handles since they span coarse→fine.
#[derive(Clone, Copy)]
pub(crate) enum BufferLifetime {
    /// Consumed entirely within the coarse wave. May alias with other coarse transients
    /// via placement-heap interval coloring.
    CoarseOnly,
    /// Written by coarse, read by fine. Allocated as a `BufferView` sub-range from the
    /// `TransientAllocator` — cheap per-frame, but requires per-frame descriptor writes.
    Shared,
    /// Like `Shared`, but always backed by a real `Buffer` from the `ResourcePool`.
    /// Avoids per-frame `BufferView::create_view` (Metal argument-buffer writes) at the
    /// cost of a separate GPU allocation per buffer. Preferred at `MAX_CLEANUP_DEPTH=1`
    /// where transient-allocator packing provides no benefit.
    OwnedShared,
}

fn image_fmt_goldy(f: ImageFormat) -> TextureFormat {
    match f {
        ImageFormat::Rgba8 => TextureFormat::Rgba8Unorm,
        ImageFormat::Bgra8 => TextureFormat::Bgra8Unorm,
    }
}

pub(crate) fn alloc_pipeline_buffer(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    size: u64,
    stride: u32,
    name: &'static str,
    flags: BufferFlags,
    lifetime: BufferLifetime,
) -> Result<GpuBuf, Error> {
    let use_graph_coloring = use_pool(device) && !is_pool_exempt(name);

    // CoarseOnly → graph transient (wave-interval coloring) when graph coloring
    // is active per the FrameStrategy. At LowLatency (depth=1) all CoarseOnly
    // buffers are promoted to persistent OwnedShared so their bindless indices are
    // stable — a prerequisite for command buffer retention.
    if use_graph_coloring
        && persistent.strategy.use_graph_coloring()
        && matches!(lifetime, BufferLifetime::CoarseOnly)
    {
        let tid = graph.transient_buffer_with_stride(size, stride);
        return Ok(GpuBuf::Transient(tid));
    }

    // OwnedShared → ResourcePool (avoids per-frame BufferView::create_view /
    // Metal argument-buffer writes; preferable at MAX_CLEANUP_DEPTH=1).
    // Also used for pool-exempt names (bump, indirect, etc.) regardless of lifetime.
    // CoarseOnly at depth=1: promoted to this path for stable bindless indices.
    if matches!(
        lifetime,
        BufferLifetime::OwnedShared | BufferLifetime::CoarseOnly
    ) || is_pool_exempt(name)
    {
        let buf = persistent.pool.get_buf_with_stride(
            device,
            size,
            name,
            DataAccess::Scattered,
            Some(stride),
            flags,
        )?;
        // Pre-clear pool-exempt buffers (bump needs zeroing each frame; indirect
        // dispatch counts must be 0 before GPU pipelines them). OwnedShared buffers
        // are always overwritten by GPU dispatches before first read, so skip the clear.
        if is_pool_exempt(name) {
            graph.clear_buffer(&buf, 0, size);
        }
        return Ok(GpuBuf::Owned(buf));
    }

    // Shared → TransientAllocator sub-range (BufferView into a pooled backing buffer).
    if use_pool(device) {
        let allocator = persistent
            .storage_allocator_mut()
            .ok_or_else(|| Error::Shader("storage allocator not prepared".into()))?;
        let view = allocator
            .alloc(device, size, Some(stride))
            .map_err(|e| Error::Shader(e.to_string()))?;
        return Ok(GpuBuf::Pooled(view));
    }

    // CPU / WARP device fallback: Owned buffer, no pooling.
    let buf = persistent.pool.get_buf_with_stride(
        device,
        size,
        name,
        DataAccess::Scattered,
        Some(stride),
        flags,
    )?;
    graph.clear_buffer(&buf, 0, size);
    Ok(GpuBuf::Owned(buf))
}

pub(crate) fn record_upload_bytes(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    name: &'static str,
    element_stride: u32,
    bytes: &[u8],
) -> Result<GpuBuf, Error> {
    let buf = persistent.pool.get_buf_with_stride(
        device,
        bytes.len() as u64,
        name,
        DataAccess::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    graph.write_buffer(&buf, 0, bytes.to_vec());
    Ok(GpuBuf::Owned(buf))
}

/// Like [`record_upload_bytes`] but takes ownership of the byte vector, avoiding
/// the redundant `to_vec()` copy when the caller already holds an owned `Vec<u8>`.
pub(crate) fn record_upload_bytes_owned(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    name: &'static str,
    element_stride: u32,
    bytes: Vec<u8>,
) -> Result<GpuBuf, Error> {
    let buf = persistent.pool.get_buf_with_stride(
        device,
        bytes.len() as u64,
        name,
        DataAccess::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    graph.write_buffer(&buf, 0, bytes);
    Ok(GpuBuf::Owned(buf))
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
            SpatialAccess::Interpolated,
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
    access: SpatialAccess,
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
    buf: &GpuBuf,
    off: u64,
    size: Option<u64>,
) -> Result<(), Error> {
    match buf {
        GpuBuf::Owned(b) => {
            let sz = size.unwrap_or_else(|| b.size().saturating_sub(off));
            graph.clear_buffer(b, off, sz);
        }
        GpuBuf::Pooled(v) => {
            let sz = size.unwrap_or_else(|| v.size().saturating_sub(off));
            graph.clear_buffer_view(v, off, sz);
        }
        GpuBuf::Transient(_) => {}
    }
    Ok(())
}

/// Cached GPU buffers that survive across frames when `buffer_sizes` is stable.
///
/// At `MAX_CLEANUP_DEPTH=1` the previous frame's GPU work is complete by the time
/// `begin_frame` returns, so these buffers are safe to rebind immediately.
///
/// The six `OwnedShared` buffers (`info_bin_data`, `tile`, `segments`, `ptcl`,
/// `blend_spill`, `fallback_indirect`) have always lived here.
///
/// At `LowLatency` (depth=1), the twelve `CoarseOnly` buffers are also promoted to
/// persistent owned handles. This eliminates graph-coloring transient IDs and gives
/// them stable bindless indices — a prerequisite for command buffer retention.
pub(crate) struct CachedPipeline {
    // OwnedShared: written coarse, read fine.
    pub info_bin_data: Buffer,
    pub tile: Buffer,
    pub segments: Buffer,
    pub ptcl: Buffer,
    pub blend_spill: Buffer,
    pub fallback_indirect: Buffer,
    // CoarseOnly (depth=1 only): consumed within the coarse wave; cached for stable bindless indices.
    pub reduced: Option<Buffer>,
    pub reduced2: Option<Buffer>,
    pub reduced_scan: Option<Buffer>,
    pub tagmonoid: Option<Buffer>,
    pub path_bbox: Option<Buffer>,
    pub lines: Option<Buffer>,
    pub draw_reduced: Option<Buffer>,
    pub draw_monoid: Option<Buffer>,
    pub clip_inp: Option<Buffer>,
    pub clip_el: Option<Buffer>,
    pub clip_bic: Option<Buffer>,
    pub clip_bbox: Option<Buffer>,
    pub draw_bbox: Option<Buffer>,
    pub bin_header: Option<Buffer>,
    pub path: Option<Buffer>,
    pub seg_counts: Option<Buffer>,
    pub buffer_sizes: ekrano_encoding::BufferSizes,
}

pub(crate) struct PipelineResources {
    pub gradient: Texture,
    pub image_atlas: Texture,
    pub mask_atlas: Texture,
    pub scene: GpuBuf,
    pub config: GpuBuf,
    pub wg_counts: Option<GpuBuf>,
    pub indirect: Option<GpuBuf>,
    pub fallback_indirect: GpuBuf,
    pub info_bin_data: GpuBuf,
    pub tile: GpuBuf,
    pub segments: GpuBuf,
    pub ptcl: GpuBuf,
    pub reduced: GpuBuf,
    pub reduced2: GpuBuf,
    pub reduced_scan: GpuBuf,
    pub tagmonoid: GpuBuf,
    pub path_bbox: GpuBuf,
    pub bump: GpuBuf,
    pub lines: GpuBuf,
    pub draw_reduced: GpuBuf,
    pub draw_monoid: GpuBuf,
    pub clip_inp: GpuBuf,
    pub clip_el: GpuBuf,
    pub clip_bic: GpuBuf,
    pub clip_bbox: GpuBuf,
    pub draw_bbox: GpuBuf,
    pub bin_header: GpuBuf,
    pub path: GpuBuf,
    pub seg_counts: GpuBuf,
    pub blend_spill: GpuBuf,
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
        graph: &mut TaskGraph,
        persistent: &mut PersistentState,
        encoding: &Encoding,
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

        let gpu_progress = device.gpu_progress();

        let mut cpu_config_owned = *config;
        if encoding.coverage_mask.is_some() {
            cpu_config_owned.gpu.mask_active = 1;
        }
        if let Some(ref m) = encoding.coverage_mask {
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
                    SpatialAccess::Interpolated,
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
                    SpatialAccess::Interpolated,
                    TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                )?;
                (t, (1_u32, 1_u32))
            } else {
                let t = acquire_texture_rgba(
                    device,
                    persistent,
                    images.width,
                    images.height,
                    SpatialAccess::Interpolated,
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
            match &encoding.coverage_mask {
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
            record_upload_bytes_owned(device, graph, persistent, "ekrano.scene", 4, packed)?
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
                GpuBuf::Owned(persistent.cached_config_uniform.take().unwrap().1)
            } else if let Some((_, existing_buf)) = persistent.cached_config_uniform.take() {
                // Buffer size is constant (sizeof ConfigUniform); reuse the allocation
                // and just overwrite with the new value.
                graph.write_buffer(
                    &existing_buf,
                    0,
                    bytemuck::bytes_of(&config_uniform_value).to_vec(),
                );
                GpuBuf::Owned(existing_buf)
            } else {
                record_upload_bytes(
                    device,
                    graph,
                    persistent,
                    "ekrano.config",
                    size_of::<ekrano_encoding::ConfigUniform>() as u32,
                    bytemuck::bytes_of(&config_uniform_value),
                )?
            }
        };

        let buffer_sizes = cpu_config_owned.buffer_sizes;

        // Try to reuse cached OwnedShared + CoarseOnly buffers from the previous frame.
        // At MAX_CLEANUP_DEPTH=1, begin_frame blocks until the previous frame's GPU work
        // is complete, so these buffers are safe to rebind immediately without any fence check.
        // Cache hit eliminates ResourcePool HashMap lookups and (at depth=1) graph-coloring
        // transient IDs — keeping bindless indices stable for command buffer retention.
        struct CachedOwnedBuffers {
            info_bin_data: Option<Buffer>,
            tile: Option<Buffer>,
            segments: Option<Buffer>,
            ptcl: Option<Buffer>,
            blend_spill: Option<Buffer>,
            fallback_indirect: Option<Buffer>,
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
                    fallback_indirect: Some(c.fallback_indirect),
                    reduced: c.reduced,
                    reduced2: c.reduced2,
                    reduced_scan: c.reduced_scan,
                    tagmonoid: c.tagmonoid,
                    path_bbox: c.path_bbox,
                    lines: c.lines,
                    draw_reduced: c.draw_reduced,
                    draw_monoid: c.draw_monoid,
                    clip_inp: c.clip_inp,
                    clip_el: c.clip_el,
                    clip_bic: c.clip_bic,
                    clip_bbox: c.clip_bbox,
                    draw_bbox: c.draw_bbox,
                    bin_header: c.bin_header,
                    path: c.path,
                    seg_counts: c.seg_counts,
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
                    persistent
                        .pool
                        .return_buf(c.fallback_indirect, "ekrano.indirect_count");
                    macro_rules! return_coarse {
                        ($field:expr, $name:expr) => {
                            if let Some(b) = $field {
                                persistent.pool.return_buf(b, $name);
                            }
                        };
                    }
                    return_coarse!(c.reduced, "ekrano.reduced_buf");
                    return_coarse!(c.reduced2, "ekrano.reduced2_buf");
                    return_coarse!(c.reduced_scan, "ekrano.reduced_scan_buf");
                    return_coarse!(c.tagmonoid, "ekrano.tagmonoid_buf");
                    return_coarse!(c.path_bbox, "ekrano.path_bbox_buf");
                    return_coarse!(c.lines, "ekrano.lines_buf");
                    return_coarse!(c.draw_reduced, "ekrano.draw_reduced_buf");
                    return_coarse!(c.draw_monoid, "ekrano.draw_monoid_buf");
                    return_coarse!(c.clip_inp, "ekrano.clip_inp_buf");
                    return_coarse!(c.clip_el, "ekrano.clip_el_buf");
                    return_coarse!(c.clip_bic, "ekrano.clip_bic_buf");
                    return_coarse!(c.clip_bbox, "ekrano.clip_bbox_buf");
                    return_coarse!(c.draw_bbox, "ekrano.draw_bbox_buf");
                    return_coarse!(c.bin_header, "ekrano.bin_header_buf");
                    return_coarse!(c.path, "ekrano.path_buf");
                    return_coarse!(c.seg_counts, "ekrano.seg_counts_buf");
                    CachedOwnedBuffers {
                        info_bin_data: None,
                        tile: None,
                        segments: None,
                        ptcl: None,
                        blend_spill: None,
                        fallback_indirect: None,
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
                    fallback_indirect: None,
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

        // fallback_indirect: pool-exempt, must be zeroed before GPU use.
        let _tz_alloc = goldy::tracy_zone!("ekrano.prepare.alloc_buffers");
        let fallback_indirect = match cached.fallback_indirect {
            Some(buf) => {
                graph.clear_buffer(&buf, 0, size_of::<IndirectCount>() as u64);
                GpuBuf::Owned(buf)
            }
            None => alloc_pipeline_buffer(
                device,
                graph,
                persistent,
                size_of::<IndirectCount>() as u64,
                size_of::<IndirectCount>() as u32,
                "ekrano.indirect_count",
                BufferFlags::empty(),
                // pool-exempt: always GpuBuf::Owned regardless of lifetime tag
                BufferLifetime::CoarseOnly,
            )?,
        };

        // For OwnedShared buffers: reuse from cache when sizes match (no ResourcePool
        // round-trip). These buffers are fully GPU-overwritten before first read.
        macro_rules! al_shared_cached {
            ($cached_opt:expr, $sz:expr, $stride:expr, $name:expr) => {
                match $cached_opt {
                    Some(buf) => GpuBuf::Owned(buf),
                    None => alloc_pipeline_buffer(
                        device,
                        graph,
                        persistent,
                        $sz,
                        $stride,
                        $name,
                        BufferFlags::empty(),
                        BufferLifetime::OwnedShared,
                    )?,
                }
            };
        }

        // For CoarseOnly buffers: reuse from cache when available (depth=1 promoted path),
        // otherwise allocate with CoarseOnly lifetime (graph transient at depth>1, owned at depth=1).
        macro_rules! al_coarse_cached {
            ($cached_opt:expr, $sz:expr, $stride:expr, $name:expr) => {
                match $cached_opt {
                    Some(buf) => GpuBuf::Owned(buf),
                    None => alloc_pipeline_buffer(
                        device,
                        graph,
                        persistent,
                        $sz,
                        $stride,
                        $name,
                        BufferFlags::empty(),
                        BufferLifetime::CoarseOnly,
                    )?,
                }
            };
        }

        // Shared: written by coarse, read by fine.
        let info_bin_data = al_shared_cached!(
            cached.info_bin_data,
            buffer_sizes.bin_data.size_in_bytes() as u64,
            4,
            "ekrano.info_bin_data_buf"
        );
        let tile = al_shared_cached!(
            cached.tile,
            buffer_sizes.tiles.size_in_bytes().into(),
            8,
            "ekrano.tile_buf"
        );
        let segments = al_shared_cached!(
            cached.segments,
            buffer_sizes.segments.size_in_bytes().into(),
            24,
            "ekrano.segments_buf"
        );
        let ptcl = al_shared_cached!(
            cached.ptcl,
            buffer_sizes.ptcl.size_in_bytes().into(),
            4,
            "ekrano.ptcl_buf"
        );
        // CoarseOnly: consumed entirely within the coarse wave.
        // At LowLatency these are promoted to OwnedShared (stable bindless indices);
        // at higher depths they remain graph transients (wave-interval coloring for VRAM).
        let reduced = al_coarse_cached!(
            cached.reduced,
            buffer_sizes.path_reduced.size_in_bytes().into(),
            20,
            "ekrano.reduced_buf"
        );
        let reduced2 = al_coarse_cached!(
            cached.reduced2,
            buffer_sizes.path_reduced2.size_in_bytes().into(),
            20,
            "ekrano.reduced2_buf"
        );
        let reduced_scan = al_coarse_cached!(
            cached.reduced_scan,
            buffer_sizes.path_reduced_scan.size_in_bytes().into(),
            20,
            "ekrano.reduced_scan_buf"
        );
        let tagmonoid = al_coarse_cached!(
            cached.tagmonoid,
            buffer_sizes.path_monoids.size_in_bytes().into(),
            20,
            "ekrano.tagmonoid_buf"
        );
        let path_bbox = al_coarse_cached!(
            cached.path_bbox,
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            24,
            "ekrano.path_bbox_buf"
        );
        // bump is pool-exempt (CPU_READABLE) → always GpuBuf::Owned.
        let bump = alloc_pipeline_buffer(
            device,
            graph,
            persistent,
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            size_of::<BumpAllocators>() as u32,
            "ekrano.bump_buf",
            BufferFlags::CPU_READABLE,
            BufferLifetime::Shared,
        )?;
        clear_gpu_buf(graph, &bump, 0, None)?;
        let lines = al_coarse_cached!(
            cached.lines,
            buffer_sizes.lines.size_in_bytes().into(),
            24,
            "ekrano.lines_buf"
        );
        let draw_reduced = al_coarse_cached!(
            cached.draw_reduced,
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            16,
            "ekrano.draw_reduced_buf"
        );
        let draw_monoid = al_coarse_cached!(
            cached.draw_monoid,
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            16,
            "ekrano.draw_monoid_buf"
        );
        let clip_inp = al_coarse_cached!(
            cached.clip_inp,
            buffer_sizes.clip_inps.size_in_bytes().into(),
            8,
            "ekrano.clip_inp_buf"
        );
        let clip_el = al_coarse_cached!(
            cached.clip_el,
            buffer_sizes.clip_els.size_in_bytes().into(),
            32,
            "ekrano.clip_el_buf"
        );
        let clip_bic = al_coarse_cached!(
            cached.clip_bic,
            buffer_sizes.clip_bics.size_in_bytes().into(),
            8,
            "ekrano.clip_bic_buf"
        );
        let clip_bbox = al_coarse_cached!(
            cached.clip_bbox,
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            16,
            "ekrano.clip_bbox_buf"
        );
        let draw_bbox = al_coarse_cached!(
            cached.draw_bbox,
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            16,
            "ekrano.draw_bbox_buf"
        );
        let bin_header = al_coarse_cached!(
            cached.bin_header,
            buffer_sizes.bin_headers.size_in_bytes().into(),
            8,
            "ekrano.bin_header_buf"
        );
        let path = al_coarse_cached!(
            cached.path,
            buffer_sizes.paths.size_in_bytes().into(),
            32,
            "ekrano.path_buf"
        );
        let seg_counts = al_coarse_cached!(
            cached.seg_counts,
            buffer_sizes.seg_counts.size_in_bytes().into(),
            8,
            "ekrano.seg_counts_buf"
        );
        // blend_spill is used only by fine, but allocating it as Shared (pre-flush)
        // avoids the need to split prepare() into two phases.
        let blend_spill = al_shared_cached!(
            cached.blend_spill,
            buffer_sizes.blend_spill.size_in_bytes().into(),
            size_of::<u32>() as u32,
            "ekrano.blend_spill"
        );

        // How many filter layer textures must be full-size for this frame.
        // Layers beyond this count are rendered as 1×1 stubs to save VRAM — the fine
        // shader still receives all 4 bindings but never samples from unused slots.
        let needed_filter_layers = encoding
            .layer_filter_effects
            .iter()
            .map(|e| e.layer_index as usize + 1)
            .max()
            .unwrap_or(0)
            .min(4);

        // Try to reuse cached render targets from the previous frame (avoids TexturePool
        // round-trips when render dimensions are stable across frames).
        let (out_image, filter_layers) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.render_targets");
            if let Some((cached_out, cached_layers)) = persistent.take_cached_render_targets(
                gpu_progress,
                params.width,
                params.height,
                out_image_format,
                needed_filter_layers,
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
                        SpatialAccess::Direct,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )
                    .map_err(|e| Error::Shader(e.to_string()))?;
                let layers = std::array::from_fn(|i| {
                    // Allocate full-size only for layers the encoding actually uses;
                    // unused slots get a 1×1 stub (the fine shader binds but never reads them).
                    let (w, h) = if i < needed_filter_layers {
                        (params.width, params.height)
                    } else {
                        (1, 1)
                    };
                    let result = acquire_texture_rgba(
                        device,
                        persistent,
                        w,
                        h,
                        SpatialAccess::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    );
                    result.expect("filter layer")
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
            wg_counts: None,
            indirect: None,
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
            GpuBinding::Buf(_) | GpuBinding::View(_) => binding.bindless_slot(is_read_only)?,
            GpuBinding::Tex(tex) if is_sampled_image => tex
                .bindless_sampled_index()
                .or_else(|| tex.bindless_index())
                .ok_or_else(|| {
                    Error::Shader(
                        "bindless sampled index missing for ImageRead texture binding".into(),
                    )
                })?,
            GpuBinding::Tex(_) => binding.bindless_slot(false)?,
            GpuBinding::Transient(_) => TRANSIENT_SLOT_PLACEHOLDER,
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

    fn transient_binding() -> GpuBinding<'static> {
        use goldy::task_graph::TransientId;
        GpuBinding::Transient(TransientId(42))
    }

    #[test]
    fn collect_into_sampler_and_transient() {
        let bindings = [sampler_binding(7), transient_binding(), sampler_binding(3)];
        let bind_types = [BindType::Sampler, BindType::Buffer, BindType::Sampler];

        let mut out = Vec::new();
        collect_bindless_indices_into(&mut out, &bindings, &bind_types, 16).unwrap();

        assert_eq!(out, [7, TRANSIENT_SLOT_PLACEHOLDER, 3]);
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
