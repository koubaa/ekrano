// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scheme-backend GPU resource helpers.

use std::mem::size_of;

use goldy::types::{BufferFlags, TextureFlags, TextureKind};
use goldy::{
    Buffer, BufferKind, DepositTransaction, DispatchShape, Init, MemoryExchange, NodeAccess, Parcel, PresentLease,
    Sampler, Texture, TextureFormat, ordinal,
};

use crate::resource_proxy::BindType;
use crate::scheme_renderer::SchemeRecorder;
use crate::worker_retention::{note_scene_bucket_crossing, scene_size_bucket};
use crate::{Error, RenderParams, Result};
use ekrano_encoding::{
    BumpAllocators, CoverageMask, Images, N_INDIRECT_STAGES, Ramps, RenderConfig, STAGE_PATH_COUNT, STAGE_PATH_TILING,
    WorkgroupCountsGpu,
};

/// Record GPU-orderable reuse epochs on `scheme` for a buffer that will be overwritten.
fn record_buffer_reuse(scheme: &mut goldy::Scheme, buf: &Buffer) {
    scheme.record_reuse_buffer(buf);
}

/// Record GPU-orderable reuse epochs on `scheme` for a texture that will be overwritten.
fn record_texture_reuse(scheme: &mut goldy::Scheme, tex: &Texture) {
    scheme.record_reuse_parcel(tex.whole());
}

/// Return a buffer to the transient pool for epoch-gated retirement (nonblocking path).
fn defer_buffer_until_retired(ctx: &goldy::Context, buf: Buffer) {
    ctx.return_transient_buffer(buf);
}

/// Acquire a sticky (cross-frame) RGBA texture deed from the retained pool.
fn acquire_retained_texture(
    recorder: &mut SchemeRecorder<'_>,
    width: u32,
    height: u32,
    format: TextureFormat,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    recorder
        .persistent
        .retained_pool
        .acquire_texture(width, height, format, access, flags, None)
        .map_err(|e| Error::Gpu(format!("{e:#}")))
}

fn acquire_retained_texture_rgba(
    recorder: &mut SchemeRecorder<'_>,
    width: u32,
    height: u32,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    acquire_retained_texture(recorder, width, height, TextureFormat::Rgba8Unorm, access, flags)
}

/// Relinquish a sticky texture deed into the context transient pool (epoch-gated).
fn release_retained_texture(recorder: &mut SchemeRecorder<'_>, tex: Texture) {
    recorder
        .persistent
        .retained_pool
        .release_texture(recorder.context(), tex);
}

/// Host-visible staging write into a destination-bound [`DepositTransaction`] (never waits).
fn write_deposit(
    recorder: &mut SchemeRecorder<'_>,
    deposit: DepositTransaction,
    bytes: &[u8],
    what: &'static str,
) -> Result<(), Error> {
    deposit.write(recorder.upload_scheme(), 0, bytes).map_err(|e| {
        Error::Gpu(format!(
            "{e} (what={what}, deposit_id={}, bytes={}, needs_record={})",
            deposit.id(),
            bytes.len(),
            recorder.upload_needs_record,
        ))
    })
}

/// Pack tightly-packed RGBA rows into a footprint-pitched staging layout.
fn pack_rgba_to_pitch(src: &[u8], width: u32, height: u32, row_pitch: u32) -> Vec<u8> {
    let tight = (width as usize).saturating_mul(4);
    let pitch = row_pitch as usize;
    let mut out = vec![0_u8; pitch.saturating_mul(height as usize)];
    for y in 0..height as usize {
        let src_off = y * tight;
        let dst_off = y * pitch;
        let end = src_off + tight;
        if end <= src.len() && dst_off + tight <= out.len() {
            out[dst_off..dst_off + tight].copy_from_slice(&src[src_off..end]);
        }
    }
    out
}

/// Shader binding helper for pipeline [`Buffer`] handles.
pub(crate) trait PipelineBuffer {
    fn as_binding(&self) -> GpuBinding<'_>;
}

impl PipelineBuffer for Buffer {
    fn as_binding(&self) -> GpuBinding<'_> {
        GpuBinding::Buf(self)
    }
}

impl PipelineBuffer for Parcel {
    fn as_binding(&self) -> GpuBinding<'_> {
        GpuBinding::Parcel(self)
    }
}

pub(crate) enum GpuBinding<'a> {
    Buf(&'a Buffer),
    Parcel(&'a Parcel),
    Tex(&'a Texture),
    /// A GPU sampler from [`crate::goldy_renderer::PersistentState`].
    Sampler(&'a Sampler),
    /// Swapchain drawable bound at submit time via a present-lease placeholder slot.
    Present(&'a PresentLease, NodeAccess),
}

pub(crate) fn alloc_pipeline_buffer(
    recorder: &mut SchemeRecorder<'_>,
    size: u64,
    stride: u32,
    _name: &'static str,
    flags: BufferFlags,
) -> Result<Buffer, Error> {
    recorder
        .context
        .acquire_transient_buffer(size, BufferKind::Scattered, flags, Some(stride))
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// Allocate or reuse a composite indirect buffer for the scheme path.
///
/// One [`goldy::RetainedPool::acquire_record`] buffer holds `N_INDIRECT_STAGES` ordinal
/// [`goldy::DispatchShape`] parcels. CPU-known stages are initialised at allocation via
/// [`Init::data`]; GPU-written stages ([`STAGE_PATH_COUNT`], [`STAGE_PATH_TILING`])
/// use [`Init::reserve`] and are written each frame by setup shaders.
///
/// Indexed via [`Buffer::unit`]: `buf.unit(STAGE_FOO as usize)`.
pub(crate) fn alloc_or_reuse_scheme_indirect(
    recorder: &mut SchemeRecorder<'_>,
    wg_counts_gpu: &WorkgroupCountsGpu,
) -> Result<Buffer, Error> {
    if let Some((cached_wg, buf)) = recorder.persistent.cached_scheme_indirect.take() {
        if &cached_wg == wg_counts_gpu {
            record_buffer_reuse(recorder.scheme(), &buf);
            return Ok(buf);
        }
        // Partitioned acquire_record buffer: not binneable via return_transient_buffer.
        // Release through the retained pool (epoch-gated drop); same as other stable buffers.
        let ctx = recorder.context();
        recorder.persistent.retained_pool.release_buffer(ctx, buf);
    }
    let fields: Vec<_> = (0..N_INDIRECT_STAGES as usize)
        .map(|i| {
            if i != STAGE_PATH_COUNT as usize && i != STAGE_PATH_TILING as usize {
                let e = wg_counts_gpu.entries[i];
                ordinal(Init::data(&[DispatchShape {
                    x: e[0],
                    y: e[1],
                    z: e[2],
                }]))
            } else {
                ordinal(Init::reserve::<DispatchShape>(1))
            }
        })
        .collect();
    recorder
        .persistent
        .retained_pool
        .acquire_record(fields)
        .map_err(|e| Error::Gpu(e.to_string()))
}

#[cfg(feature = "debug_layers")]
pub(crate) fn record_upload_bytes_owned(
    recorder: &mut SchemeRecorder<'_>,
    _name: &'static str,
    element_stride: u32,
    bytes: Vec<u8>,
) -> Result<Buffer, Error> {
    let buf = recorder
        .context
        .acquire_transient_buffer(
            bytes.len() as u64,
            BufferKind::Scattered,
            BufferFlags::empty(),
            Some(element_stride),
        )
        .map_err(|e| Error::Gpu(e.to_string()))?;
    let deposit = MemoryExchange::new(recorder.context())
        .bind_deposit_buffer(recorder.upload_scheme(), &buf, bytes.len() as u64)
        .map_err(Error::from)?;
    write_deposit(recorder, deposit, &bytes, "record_upload_bytes_owned")?;
    Ok(buf)
}

pub(crate) fn write_image_region(
    recorder: &mut SchemeRecorder<'_>,
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

    // Fine samples the atlas with an explicit 4-tap bilinear (or nearest Load), which still
    // requires premultiplied-alpha texels to avoid fringing on transparent edges.
    // Straight-alpha images (ImageAlphaType::Alpha) are converted to premultiplied on the CPU
    // before upload; premultiplied sources are used as-is. Callers' ImageData is never mutated.
    let premul_storage;
    let bytes: &[u8] = if image_data.alpha_type == peniko::ImageAlphaType::Alpha {
        premul_storage = premultiply_rgba8(raw_bytes);
        &premul_storage
    } else {
        raw_bytes
    };

    upload_texture_region(recorder, tex, x, y, image_data.width, image_data.height, bytes)
}

/// Premultiply every RGBA8 pixel: `(r, g, b, a)` → `(r*a/255, g*a/255, b*a/255, a)`.
///
/// Uses integer arithmetic to match the GPU's 8-bit rounding behaviour precisely.
fn premultiply_rgba8(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for chunk in out.as_chunks_mut::<4>().0 {
        let a = chunk[3] as u32;
        chunk[0] = ((chunk[0] as u32 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u32 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u32 * a + 127) / 255) as u8;
    }
    out
}

pub(crate) fn acquire_texture_rgba(
    recorder: &mut SchemeRecorder<'_>,
    width: u32,
    height: u32,
    access: TextureKind,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    recorder
        .context()
        .acquire_transient_texture(width, height, TextureFormat::Rgba8Unorm, access, flags)
        .map_err(|e| Error::Gpu(format!("{e:#}")))
}

pub(crate) fn clear_gpu_buf(
    recorder: &mut SchemeRecorder<'_>,
    buf: &Buffer,
    off: u64,
    size: Option<u64>,
) -> Result<(), Error> {
    if !recorder.upload_needs_record {
        return Ok(());
    }
    let sz = size.unwrap_or_else(|| buf.byte_size().saturating_sub(off));
    recorder
        .upload_scheme()
        .clear_parcel(buf, off, sz)
        .map_err(Error::from)?;
    Ok(())
}

/// Allocate or reuse a stable scene buffer (bucketed capacity).
pub(crate) fn alloc_or_reuse_scene(recorder: &mut SchemeRecorder<'_>, live_bytes: usize) -> Result<Buffer, Error> {
    let bucket = scene_size_bucket(live_bytes);
    if let Some((cached_bucket, buf)) = recorder.persistent.cached_scene.take() {
        if cached_bucket >= bucket {
            record_buffer_reuse(recorder.upload_scheme(), &buf);
            return Ok(buf);
        }
        note_scene_bucket_crossing(&mut recorder.persistent.scene_growth, cached_bucket, bucket, live_bytes);
        defer_buffer_until_retired(recorder.context(), buf);
    }
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(bucket, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// Allocate or reuse a destination-bound scene deposit (bucketed capacity).
fn alloc_or_reuse_scene_deposit(
    recorder: &mut SchemeRecorder<'_>,
    scene: &Buffer,
    live_bytes: usize,
) -> Result<DepositTransaction, Error> {
    let bucket = scene_size_bucket(live_bytes);
    if !recorder.upload_needs_record
        && let Some((cached_bucket, deposit)) = recorder.persistent.cached_scene_deposit
        && cached_bucket >= bucket
    {
        return Ok(deposit);
    }
    let deposit = MemoryExchange::new(recorder.context())
        .bind_deposit_buffer(recorder.upload_scheme(), scene.whole(), bucket)
        .map_err(|e| Error::Gpu(e.to_string()))?;
    recorder.persistent.cached_scene_deposit = Some((bucket, deposit));
    Ok(deposit)
}

/// Allocate or reuse a destination-bound config uniform deposit.
fn alloc_or_reuse_config_deposit(
    recorder: &mut SchemeRecorder<'_>,
    config: &Buffer,
) -> Result<DepositTransaction, Error> {
    let size = size_of::<ekrano_encoding::ConfigUniform>() as u64;
    if !recorder.upload_needs_record
        && let Some(deposit) = recorder.persistent.cached_config_deposit
    {
        return Ok(deposit);
    }
    let deposit = MemoryExchange::new(recorder.context())
        .bind_deposit_buffer(recorder.upload_scheme(), config.whole(), size)
        .map_err(|e| Error::Gpu(e.to_string()))?;
    recorder.persistent.cached_config_deposit = Some(deposit);
    Ok(deposit)
}

/// Write scene bytes into a destination-bound deposit (copy topology recorded at bind).
pub(crate) fn stage_scene_bytes(recorder: &mut SchemeRecorder<'_>, scene: &Buffer, bytes: &[u8]) -> Result<(), Error> {
    let deposit = {
        let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload.deposit");
        alloc_or_reuse_scene_deposit(recorder, scene, bytes.len())?
    };
    let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload.write");
    write_deposit(recorder, deposit, bytes, "scene")
}

/// Write config uniform bytes into a destination-bound deposit (copy topology recorded at bind).
pub(crate) fn stage_config_bytes(
    recorder: &mut SchemeRecorder<'_>,
    config: &Buffer,
    bytes: &[u8],
) -> Result<(), Error> {
    let deposit = {
        let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload.deposit");
        alloc_or_reuse_config_deposit(recorder, config)?
    };
    let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload.write");
    write_deposit(recorder, deposit, bytes, "config")
}

/// Allocate or reuse a stable bump buffer for the retained worker.
pub(crate) fn alloc_or_reuse_bump(recorder: &mut SchemeRecorder<'_>, size: u64) -> Result<Buffer, Error> {
    if let Some((cached_size, buf)) = recorder.persistent.cached_bump.take() {
        if cached_size == size {
            record_buffer_reuse(recorder.scheme(), &buf);
            return Ok(buf);
        }
        recorder.persistent.cached_bump_withdraw = None;
        defer_buffer_until_retired(recorder.context(), buf);
    }
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(
            size,
            BufferKind::Scattered,
            Some(size_of::<BumpAllocators>() as u32),
            BufferFlags::empty(),
            None,
        )
        .map_err(|e| Error::Gpu(e.to_string()))
}

enum AtlasCache {
    Gradient,
    Image,
    Mask,
}

fn acquire_or_reuse_retained_atlas(
    recorder: &mut SchemeRecorder<'_>,
    which: AtlasCache,
    width: u32,
    height: u32,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    let cached = match which {
        AtlasCache::Gradient => &mut recorder.persistent.cached_gradient,
        AtlasCache::Image => &mut recorder.persistent.cached_image_atlas,
        AtlasCache::Mask => &mut recorder.persistent.cached_mask_atlas,
    };
    if let Some((cw, ch, tex)) = cached.as_ref()
        && *cw == width
        && *ch == height
    {
        return Ok(tex.borrow());
    }
    let stale = cached.take().map(|(_, _, tex)| tex);
    if let Some(stale) = stale {
        release_retained_texture(recorder, stale);
    }
    let tex = acquire_retained_texture_rgba(recorder, width, height, TextureKind::Interpolated, flags)?;
    let cached = match which {
        AtlasCache::Gradient => &mut recorder.persistent.cached_gradient,
        AtlasCache::Image => &mut recorder.persistent.cached_image_atlas,
        AtlasCache::Mask => &mut recorder.persistent.cached_mask_atlas,
    };
    cached.replace((width, height, tex));
    Ok(cached.as_ref().unwrap().2.borrow())
}

fn alloc_or_reuse_full_texture_deposit(
    recorder: &mut SchemeRecorder<'_>,
    cached: &mut Option<(u32, u32, u64, DepositTransaction)>,
    texture: &Texture,
    width: u32,
    height: u32,
    staging_bytes: u64,
    src_row_pitch: u32,
) -> Result<DepositTransaction, Error> {
    let need = staging_bytes.max(4);
    if !recorder.upload_needs_record
        && let Some((cw, ch, cap, deposit)) = *cached
        && cw >= width
        && ch >= height
        && cap >= need
    {
        return Ok(deposit);
    }
    let deposit = MemoryExchange::new(recorder.context())
        .bind_deposit_texture(
            recorder.upload_scheme(),
            texture,
            0,
            0,
            width,
            height,
            need,
            src_row_pitch,
        )
        .map_err(|e| Error::Gpu(e.to_string()))?;
    *cached = Some((width, height, need, deposit));
    Ok(deposit)
}

fn take_region_texture_deposit(
    recorder: &mut SchemeRecorder<'_>,
    key: (u32, u32, u32, u32),
) -> Option<DepositTransaction> {
    if recorder.upload_needs_record {
        return None;
    }
    recorder
        .persistent
        .cached_image_region_deposits
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, deposit)| *deposit)
}

fn alloc_or_reuse_region_texture_deposit(
    recorder: &mut SchemeRecorder<'_>,
    texture: &Texture,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    staging_bytes: u64,
    src_row_pitch: u32,
) -> Result<DepositTransaction, Error> {
    let key = (x, y, width, height);
    if let Some(deposit) = take_region_texture_deposit(recorder, key) {
        return Ok(deposit);
    }
    let need = staging_bytes.max(4);
    let deposit = MemoryExchange::new(recorder.context())
        .bind_deposit_texture(
            recorder.upload_scheme(),
            texture,
            x,
            y,
            width,
            height,
            need,
            src_row_pitch,
        )
        .map_err(|e| Error::Gpu(e.to_string()))?;
    recorder.persistent.cached_image_region_deposits.push((key, deposit));
    Ok(deposit)
}

/// Stage RGBA bytes into a destination-bound texture deposit.
///
/// `declare_deposit` selects/allocates the deposit given the pitched byte size and row pitch.
fn stage_texture_bytes(
    recorder: &mut SchemeRecorder<'_>,
    texture: &Texture,
    _x: u32,
    _y: u32,
    width: u32,
    height: u32,
    bytes: &[u8],
    what: &'static str,
    declare_deposit: impl FnOnce(&mut SchemeRecorder<'_>, u64, u32) -> Result<DepositTransaction, Error>,
) -> Result<(), Error> {
    record_texture_reuse(recorder.upload_scheme(), texture);
    let layout = recorder
        .device()
        .texture_copy_footprint(width, height, texture.format())
        .map_err(|e| Error::Gpu(e.to_string()))?;
    let pitched;
    let staged: &[u8] = if layout.row_pitch == layout.tight_row_bytes() {
        bytes
    } else {
        pitched = pack_rgba_to_pitch(bytes, width, height, layout.row_pitch);
        &pitched
    };
    let staging_bytes = layout.staging_bytes.max(staged.len() as u64);
    let deposit = declare_deposit(recorder, staging_bytes, layout.row_pitch)?;
    write_deposit(recorder, deposit, staged, what)
}

fn stage_texture_full(
    recorder: &mut SchemeRecorder<'_>,
    cached_deposit: &mut Option<(u32, u32, u64, DepositTransaction)>,
    texture: &Texture,
    bytes: &[u8],
) -> Result<(), Error> {
    let width = texture.width();
    let height = texture.height();
    stage_texture_bytes(
        recorder,
        texture,
        0,
        0,
        width,
        height,
        bytes,
        "texture_full",
        |recorder, staging_bytes, src_row_pitch| {
            alloc_or_reuse_full_texture_deposit(
                recorder,
                cached_deposit,
                texture,
                width,
                height,
                staging_bytes,
                src_row_pitch,
            )
        },
    )
}

fn stage_texture_region(
    recorder: &mut SchemeRecorder<'_>,
    texture: &Texture,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Result<(), Error> {
    stage_texture_bytes(
        recorder,
        texture,
        x,
        y,
        width,
        height,
        bytes,
        "texture_region",
        |recorder, staging_bytes, src_row_pitch| {
            alloc_or_reuse_region_texture_deposit(recorder, texture, x, y, width, height, staging_bytes, src_row_pitch)
        },
    )
}

fn upload_texture_full(
    recorder: &mut SchemeRecorder<'_>,
    cached_deposit: &mut Option<(u32, u32, u64, DepositTransaction)>,
    texture: &Texture,
    bytes: &[u8],
) -> Result<(), Error> {
    stage_texture_full(recorder, cached_deposit, texture, bytes)
}

fn upload_texture_region(
    recorder: &mut SchemeRecorder<'_>,
    texture: &Texture,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Result<(), Error> {
    stage_texture_region(recorder, texture, x, y, width, height, bytes)
}

/// Reuse a cached buffer if one is available, or allocate a fresh one from the pool.
fn al_cached_opt(
    recorder: &mut SchemeRecorder<'_>,
    cached: Option<Buffer>,
    size: u64,
    stride: u32,
    name: &'static str,
) -> Result<Buffer, Error> {
    match cached {
        Some(buf) => {
            record_buffer_reuse(recorder.scheme(), &buf);
            Ok(buf)
        }
        None => alloc_pipeline_buffer(recorder, size, stride, name, BufferFlags::empty()),
    }
}

/// The seven large pipeline buffers whose sizes are fixed or change only on coarse
/// config changes — retained parcels in [`crate::goldy_renderer::PersistentState::retained_pool`].
/// Split from [`ScratchPipelineBuffers`]: stable buffers live in the retained pool
/// and are reused while `buffer_sizes` match; scratch buffers use the transient pool.
///
/// Cross-frame reuse is ordered by [`goldy::Scheme::record_reuse_buffer`] on the worker
/// scheme (DX12/Vulkan/Metal) or by the frame-orchestrator `begin_frame` wait (backends
/// without `host_sidecar_on_submit_worker`). If pipeline
/// depth is raised so the next frame may record while the prior frame's GPU work is still in
/// flight without those gates, a single retained deed is not enough — use double-buffered
/// parcels or a transient pool instead.
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
        recorder: &mut SchemeRecorder<'_>,
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
            info_bin_data: alloc_stable_buffer(recorder, c_ibd, bs.bin_data.size_in_bytes() as u64, 4)?,
            tile: alloc_stable_buffer(recorder, c_tile, bs.tiles.size_in_bytes().into(), 8)?,
            segments: alloc_stable_buffer(recorder, c_seg, bs.segments.size_in_bytes().into(), 24)?,
            ptcl: alloc_stable_buffer(recorder, c_ptcl, bs.ptcl.size_in_bytes().into(), 4)?,
            blend_spill: alloc_stable_buffer(
                recorder,
                c_bs,
                bs.blend_spill.size_in_bytes().into(),
                size_of::<u32>() as u32,
            )?,
            lines: alloc_stable_buffer(recorder, c_lines, bs.lines.size_in_bytes().into(), 24)?,
            seg_counts: alloc_stable_buffer(recorder, c_sc, bs.seg_counts.size_in_bytes().into(), 8)?,
        })
    }
}

fn alloc_stable_buffer(
    recorder: &mut SchemeRecorder<'_>,
    cached: Option<Buffer>,
    size: u64,
    stride: u32,
) -> Result<Buffer, Error> {
    if let Some(buffer) = cached {
        record_buffer_reuse(recorder.scheme(), &buffer);
        return Ok(buffer);
    }
    if std::env::var_os("EKRANO_LOG_PIPELINE_RESIZE").is_some() {
        log::info!("[PIPE-RESIZE] acquire stable buffer size={size} stride={stride}");
    }
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(size, BufferKind::Scattered, Some(stride), BufferFlags::empty(), None)
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// The fourteen count-derived scratch buffers whose sizes track scene complexity.
/// Acquired via [`goldy::Context::acquire_transient_buffer`]; sticky across frames in
/// [`crate::goldy_renderer::PersistentState::cached_pipeline`] when sizes match.
/// Split from [`StablePipelineBuffers`]: scratch is transient-pooled; stable is retained.
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
        recorder: &mut SchemeRecorder<'_>,
        cached: Option<Self>,
        bs: &ekrano_encoding::BufferSizes,
    ) -> Result<Self, Error> {
        let (c_red, c_red2, c_reds, c_tag, c_pbbox, c_dred, c_dmon, c_ci, c_ce, c_cb, c_cbb, c_db, c_bh, c_path) =
            match cached {
                Some(c) => (
                    Some(c.reduced),
                    Some(c.reduced2),
                    Some(c.reduced_scan),
                    Some(c.tagmonoid),
                    Some(c.path_bbox),
                    Some(c.draw_reduced),
                    Some(c.draw_monoid),
                    Some(c.clip_inp),
                    Some(c.clip_el),
                    Some(c.clip_bic),
                    Some(c.clip_bbox),
                    Some(c.draw_bbox),
                    Some(c.bin_header),
                    Some(c.path),
                ),
                None => (
                    None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                ),
            };
        Ok(Self {
            reduced: al_cached_opt(
                recorder,
                c_red,
                bs.path_reduced.size_in_bytes().into(),
                20,
                "ekrano.reduced_buf",
            )?,
            reduced2: al_cached_opt(
                recorder,
                c_red2,
                bs.path_reduced2.size_in_bytes().into(),
                20,
                "ekrano.reduced2_buf",
            )?,
            reduced_scan: al_cached_opt(
                recorder,
                c_reds,
                bs.path_reduced_scan.size_in_bytes().into(),
                20,
                "ekrano.reduced_scan_buf",
            )?,
            tagmonoid: al_cached_opt(
                recorder,
                c_tag,
                bs.path_monoids.size_in_bytes().into(),
                20,
                "ekrano.tagmonoid_buf",
            )?,
            path_bbox: al_cached_opt(
                recorder,
                c_pbbox,
                bs.path_bboxes.size_in_bytes().into(),
                24,
                "ekrano.path_bbox_buf",
            )?,
            draw_reduced: al_cached_opt(
                recorder,
                c_dred,
                bs.draw_reduced.size_in_bytes().into(),
                16,
                "ekrano.draw_reduced_buf",
            )?,
            draw_monoid: al_cached_opt(
                recorder,
                c_dmon,
                bs.draw_monoids.size_in_bytes().into(),
                16,
                "ekrano.draw_monoid_buf",
            )?,
            clip_inp: al_cached_opt(
                recorder,
                c_ci,
                bs.clip_inps.size_in_bytes().into(),
                8,
                "ekrano.clip_inp_buf",
            )?,
            clip_el: al_cached_opt(
                recorder,
                c_ce,
                bs.clip_els.size_in_bytes().into(),
                32,
                "ekrano.clip_el_buf",
            )?,
            clip_bic: al_cached_opt(
                recorder,
                c_cb,
                bs.clip_bics.size_in_bytes().into(),
                8,
                "ekrano.clip_bic_buf",
            )?,
            clip_bbox: al_cached_opt(
                recorder,
                c_cbb,
                bs.clip_bboxes.size_in_bytes().into(),
                16,
                "ekrano.clip_bbox_buf",
            )?,
            draw_bbox: al_cached_opt(
                recorder,
                c_db,
                bs.draw_bboxes.size_in_bytes().into(),
                16,
                "ekrano.draw_bbox_buf",
            )?,
            bin_header: al_cached_opt(
                recorder,
                c_bh,
                bs.bin_headers.size_in_bytes().into(),
                8,
                "ekrano.bin_header_buf",
            )?,
            path: al_cached_opt(recorder, c_path, bs.paths.size_in_bytes().into(), 32, "ekrano.path_buf")?,
        })
    }
}

/// Cached stable + scratch pipeline buffers from the previous frame.
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
    /// Composite indirect buffer (one ordinal `DispatchShape` parcel per stage).
    /// Cache key and buffer live in [`crate::goldy_renderer::PersistentState::cached_scheme_indirect`].
    pub indirect: Option<(WorkgroupCountsGpu, Buffer)>,
    pub stable: StablePipelineBuffers,
    pub scratch: ScratchPipelineBuffers,
    pub bump: Buffer,
    /// Full-frame output texture. `None` when swapchain direct-present writes to the lease.
    pub out_image: Option<Texture>,
    pub filter_layers: [Texture; 4],
    pub frame_width: u32,
    pub frame_height: u32,
    /// Buffer sizes used this frame, stored for cache-key comparison next frame.
    pub buffer_sizes: ekrano_encoding::BufferSizes,
    /// The `ConfigUniform` value last written to `config`, stored so
    /// `schedule_pipeline_cleanup` can stash `(value, buffer)` together.
    pub config_uniform_value: ekrano_encoding::ConfigUniform,
}

impl PipelineResources {
    pub(crate) fn prepare(
        recorder: &mut SchemeRecorder<'_>,
        coverage_mask: Option<&CoverageMask>,
        mut packed: Vec<u8>,
        ramps: Ramps<'_>,
        images: Images<'_>,
        params: &RenderParams,
        config: &RenderConfig,
        out_image_format: TextureFormat,
        direct_present: bool,
    ) -> Result<Self, Error> {
        if packed.is_empty() {
            packed.resize(size_of::<u32>(), u8::MAX);
        }

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
            let (gw, gh) = crate::worker_retention::normalize_gradient_atlas(ramps.width, ramps.height);
            let tex = acquire_or_reuse_retained_atlas(recorder, AtlasCache::Gradient, gw, gh, TextureFlags::COPY_DST)?;
            if ramps.height != 0 {
                let mut deposit_slot = std::mem::take(&mut recorder.persistent.cached_gradient_deposit);
                upload_texture_full(recorder, &mut deposit_slot, &tex, bytemuck::cast_slice(ramps.data))?;
                recorder.persistent.cached_gradient_deposit = deposit_slot;
            }
            tex
        };

        let (image_atlas, _) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.image_atlas");
            if recorder.upload_needs_record {
                recorder.persistent.cached_image_region_deposits.clear();
            }
            let (aw, ah) =
                crate::worker_retention::normalize_image_atlas(images.images.len(), images.width, images.height);
            let t = acquire_or_reuse_retained_atlas(
                recorder,
                AtlasCache::Image,
                aw,
                ah,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )?;
            for image in images.images {
                write_image_region(recorder, &t, image.1, image.2, &image.0)?;
            }
            (t, (aw, ah))
        };

        let mask_atlas = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.mask_atlas");
            let (mw, mh) = crate::worker_retention::normalize_mask_atlas(coverage_mask.map(|m| (m.width, m.height)));
            let tex = {
                let _tz = goldy::tracy_zone!("ekrano.prepare.mask_atlas.acquire");
                acquire_or_reuse_retained_atlas(recorder, AtlasCache::Mask, mw, mh, TextureFlags::COPY_DST)?
            };
            let mut deposit_slot = std::mem::take(&mut recorder.persistent.cached_mask_deposit);
            match coverage_mask {
                Some(m) => {
                    let rgba = {
                        let _tz = goldy::tracy_zone!("ekrano.prepare.mask_atlas.expand");
                        let mut rgba = Vec::with_capacity(m.data.len() * 4);
                        for &b in m.data.iter() {
                            rgba.extend_from_slice(&[b, b, b, 255]);
                        }
                        rgba
                    };
                    let _tz = goldy::tracy_zone!("ekrano.prepare.mask_atlas.upload");
                    upload_texture_full(recorder, &mut deposit_slot, &tex, &rgba)?;
                }
                None => {
                    let _tz = goldy::tracy_zone!("ekrano.prepare.mask_atlas.upload");
                    upload_texture_full(recorder, &mut deposit_slot, &tex, &[255, 255, 255, 255])?;
                }
            }
            recorder.persistent.cached_mask_deposit = deposit_slot;
            tex
        };

        let scene = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload");
            let scene_buf = {
                let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload.alloc");
                alloc_or_reuse_scene(recorder, packed.len())?
            };
            stage_scene_bytes(recorder, &scene_buf, &packed)?;
            scene_buf
        };

        let config_uniform_value = cpu_config_owned.gpu;

        let config = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload");
            let config_buf = {
                let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload.acquire");
                if let Some((_, buf)) = recorder.persistent.cached_config_uniform.take() {
                    record_buffer_reuse(recorder.upload_scheme(), &buf);
                    buf
                } else {
                    recorder
                        .persistent
                        .retained_pool
                        .acquire_buffer(
                            size_of::<ekrano_encoding::ConfigUniform>() as u64,
                            BufferKind::Scattered,
                            Some(size_of::<ekrano_encoding::ConfigUniform>() as u32),
                            BufferFlags::empty(),
                            None,
                        )
                        .map_err(|e| Error::Gpu(e.to_string()))?
                }
            };
            // Always write: the retained upload scheme keeps a config deposit copy node, and
            // Goldy requires every referenced deposit to be written each submit. Content-keyed
            // skip fails with an unstaged deposit error. (The cached value is still stored for
            // diagnostics / a possible future Init-style path.)
            stage_config_bytes(recorder, &config_buf, bytemuck::bytes_of(&config_uniform_value))?;
            config_buf
        };

        let buffer_sizes = cpu_config_owned.buffer_sizes;

        // Try to reuse cached pipeline buffers from the previous frame.
        // On DX12/Vulkan/Metal, ordering is via submit-side reuse epochs / deferred host writes.
        // On backends without host_sidecar_on_submit_worker, begin_frame still retires
        // the prior frame before reuse.
        let (cached_stable, cached_scratch) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.pipeline_cache");
            match recorder.persistent.take_cached_pipeline() {
                Some(c) if c.buffer_sizes == buffer_sizes => (Some(c.stable), Some(c.scratch)),
                Some(c) => {
                    if std::env::var_os("EKRANO_LOG_PIPELINE_RESIZE").is_some() {
                        log::info!("[PIPE-RESIZE] buffer_sizes mismatch — releasing stable parcels");
                    }
                    log::debug!("[PIPE-CACHE] buffer_sizes mismatch — releasing stable parcels");
                    let ctx = recorder.context();
                    let pool = &mut recorder.persistent.retained_pool;
                    for buffer in [
                        c.stable.info_bin_data,
                        c.stable.tile,
                        c.stable.segments,
                        c.stable.ptcl,
                        c.stable.blend_spill,
                        c.stable.lines,
                        c.stable.seg_counts,
                    ] {
                        pool.release_buffer(ctx, buffer);
                    }
                    let scratch_returns = [
                        c.scratch.reduced,
                        c.scratch.reduced2,
                        c.scratch.reduced_scan,
                        c.scratch.tagmonoid,
                        c.scratch.path_bbox,
                        c.scratch.draw_reduced,
                        c.scratch.draw_monoid,
                        c.scratch.clip_inp,
                        c.scratch.clip_el,
                        c.scratch.clip_bic,
                        c.scratch.clip_bbox,
                        c.scratch.draw_bbox,
                        c.scratch.bin_header,
                        c.scratch.path,
                    ];
                    for buf in scratch_returns {
                        ctx.return_transient_buffer(buf);
                    }
                    (None, None)
                }
                None => (None, None),
            }
        }; // end ekrano.prepare.pipeline_cache zone

        let _tz_alloc = goldy::tracy_zone!("ekrano.prepare.alloc_buffers");
        // Reuse from cache when sizes match (no transient-pool round-trip). These buffers
        // are fully GPU-overwritten before first read.
        let stable = StablePipelineBuffers::alloc(recorder, cached_stable, &buffer_sizes)?;
        let scratch = ScratchPipelineBuffers::alloc(recorder, cached_scratch, &buffer_sizes)?;
        let bump_size = buffer_sizes.bump_alloc.size_in_bytes().into();
        let bump = alloc_or_reuse_bump(recorder, bump_size)?;
        clear_gpu_buf(recorder, &bump, 0, None)?;

        // Try to reuse cached render targets from the previous frame (avoids retained-pool
        // reallocation when render dimensions are stable across frames).
        let (out_image, filter_layers) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.render_targets");
            if let Some((cached_out, cached_layers)) = recorder.persistent.take_scheme_render_targets(
                recorder.context(),
                params.width,
                params.height,
                out_image_format,
                direct_present,
            ) {
                if let Some(ref out) = cached_out {
                    record_texture_reuse(recorder.scheme(), out);
                }
                for layer in &cached_layers {
                    record_texture_reuse(recorder.scheme(), layer);
                }
                (cached_out, cached_layers)
            } else {
                let _tz2 = goldy::tracy_zone!("ekrano.prepare.render_targets.ALLOC");
                let out = if direct_present {
                    None
                } else {
                    Some(acquire_retained_texture(
                        recorder,
                        params.width,
                        params.height,
                        out_image_format,
                        TextureKind::Direct,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?)
                };
                let layers = [
                    acquire_retained_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                    acquire_retained_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                    acquire_retained_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                    acquire_retained_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                ];
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
            frame_width: params.width,
            frame_height: params.height,
            buffer_sizes,
            config_uniform_value,
        })
    }
}

pub(crate) fn bind_type_to_node_access(bt: BindType) -> NodeAccess {
    match bt {
        BindType::Buffer | BindType::Image(_) => NodeAccess::ReadWrite,
        BindType::BufReadOnly | BindType::Uniform | BindType::ImageRead(_) => NodeAccess::Read,
        BindType::Sampler => NodeAccess::Read,
    }
}
