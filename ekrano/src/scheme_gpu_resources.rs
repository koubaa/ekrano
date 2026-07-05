// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scheme-backend GPU resource helpers.

use std::mem::size_of;

use goldy::types::{BufferFlags, TextureFlags, TextureKind};
use goldy::{
    Buffer, BufferKind, DispatchShape, Init, Parcel, Sampler, Texture, TextureCopyFootprint, TextureFormat,
    ordinal,
};

use crate::goldy_renderer::{CacheScheduleOutcome, PersistentState, defer_buffer_until_retired};
use crate::resource_proxy::BindType;
use crate::scheme_renderer::SchemeRecorder;

fn record_worker_reuse(recorder: &mut SchemeRecorder<'_>, buf: &Buffer) {
    recorder
        .scheme()
        .record_reuse_epochs(&buf.last_referenced());
}
use crate::worker_retention::scene_size_bucket;
use crate::{Error, RenderParams, Result};
use ekrano_encoding::{
    BumpAllocators, ConfigUniform, CoverageMask, Images, N_INDIRECT_STAGES, Ramps, RenderConfig, STAGE_PATH_COUNT,
    STAGE_PATH_TILING, WorkgroupCountsGpu,
};

fn config_uniform_without_layout_eq(a: &ConfigUniform, b: &ConfigUniform) -> bool {
    a.width_in_tiles == b.width_in_tiles
        && a.height_in_tiles == b.height_in_tiles
        && a.target_width == b.target_width
        && a.target_height == b.target_height
        && a.base_color == b.base_color
        && a.lines_size == b.lines_size
        && a.binning_size == b.binning_size
        && a.tiles_size == b.tiles_size
        && a.seg_counts_size == b.seg_counts_size
        && a.segments_size == b.segments_size
        && a.blend_size == b.blend_size
        && a.ptcl_size == b.ptcl_size
        && a.tile_y_offset == b.tile_y_offset
        && a.flatten_thread_base == b.flatten_thread_base
        && a.mask_active == b.mask_active
}

enum ConfigUniformCacheOutcome {
    Hit(Buffer),
    /// Packed scene unchanged since last config cache; only layout metadata drifted — refresh GPU uniform without reuse wait.
    LayoutRefresh(Buffer),
    MissReuse(Buffer),
    MissAlloc,
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
    /// A persistent (pre-initialized) buffer from [`crate::goldy_renderer::PersistentState`].
    /// Use for buffers uploaded exactly once (e.g. static LUTs) that are GPU-readable on
    /// every frame after their first upload, without additional write nodes.
    PersistentBuf(&'a Buffer),
}

pub(crate) fn alloc_pipeline_buffer(
    recorder: &mut SchemeRecorder<'_>,
    size: u64,
    stride: u32,
    name: &'static str,
    flags: BufferFlags,
) -> Result<Buffer, Error> {
    let ctx = recorder.context;
    let persistent = &mut recorder.persistent;
    let buf = persistent.pool.get_buf_with_stride(
        &mut persistent.retained_pool,
        ctx,
        size,
        name,
        BufferKind::Scattered,
        Some(stride),
        flags,
    )?;
    // Pipeline buffers are always overwritten by GPU dispatches before first read.
    // Per-frame bump clear is recorded once on the worker scheme (retained GPU clear node).
    Ok(buf)
}

/// Allocate or reuse a composite indirect buffer for the scheme path.
///
/// One `RetainedPool::acquire_record` buffer holds `N_INDIRECT_STAGES` ordinal
/// `DispatchShape` parcels. CPU-known stages are initialised at allocation via
/// [`Init::data`]; GPU-written stages ([`STAGE_PATH_COUNT`], [`STAGE_PATH_TILING`])
/// use [`Init::reserve`] and are written each frame by setup shaders.
///
/// Indexed via [`Buffer::unit`]: `buf.unit(STAGE_FOO as usize)`.
pub(crate) fn alloc_or_reuse_scheme_indirect(
    recorder: &mut SchemeRecorder<'_>,
    wg_counts_gpu: &WorkgroupCountsGpu,
) -> Result<Buffer, Error> {
    if let Some((cached_wg, buf)) = recorder.persistent.cached_scheme_indirect.take() {
        record_worker_reuse(recorder, &buf);
        if &cached_wg == wg_counts_gpu {
            return Ok(buf);
        }
        // WorkgroupCountsGpu changed (resize / topology change): drop the stale
        // composite buffer after the parcel reuse gate clears in-flight GPU work.
        defer_buffer_until_retired(recorder.context(), buf);
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

pub(crate) fn record_upload_bytes(
    recorder: &mut SchemeRecorder<'_>,
    name: &'static str,
    element_stride: u32,
    bytes: &[u8],
) -> Result<Buffer, Error> {
    let ctx = recorder.context;
    let persistent = &mut recorder.persistent;
    let buf = persistent.pool.get_buf_with_stride(
        &mut persistent.retained_pool,
        ctx,
        bytes.len() as u64,
        name,
        BufferKind::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    recorder
        .upload_scheme()
        .commit_write_parcel(&buf, 0, bytes.to_vec())
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(buf)
}

/// Like [`record_upload_bytes`] but takes ownership of the byte vector, avoiding
/// the redundant `to_vec()` copy when the caller already holds an owned `Vec<u8>`.
pub(crate) fn record_upload_bytes_owned(
    recorder: &mut SchemeRecorder<'_>,
    name: &'static str,
    element_stride: u32,
    bytes: Vec<u8>,
) -> Result<Buffer, Error> {
    let ctx = recorder.context;
    let persistent = &mut recorder.persistent;
    let buf = persistent.pool.get_buf_with_stride(
        &mut persistent.retained_pool,
        ctx,
        bytes.len() as u64,
        name,
        BufferKind::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    recorder
        .upload_scheme()
        .commit_write_parcel(&buf, 0, bytes)
        .map_err(|e| Error::Shader(e.to_string()))?;
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

    upload_texture_region(recorder, tex, x, y, image_data.width, image_data.height, bytes.to_vec())
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
    recorder: &mut SchemeRecorder<'_>,
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

pub(crate) fn clear_gpu_buf_on_worker(
    recorder: &mut SchemeRecorder<'_>,
    buf: &Buffer,
    off: u64,
    size: Option<u64>,
) -> Result<(), Error> {
    let sz = size.unwrap_or_else(|| buf.byte_size().saturating_sub(off));
    recorder
        .scheme()
        .commit_clear_parcel(buf, off, sz)
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(())
}

/// Allocate or reuse a stable scene buffer (bucketed capacity).
pub(crate) fn alloc_or_reuse_scene(recorder: &mut SchemeRecorder<'_>, live_bytes: usize) -> Result<Buffer, Error> {
    let bucket = scene_size_bucket(live_bytes);
    if let Some((cached_bucket, buf)) = recorder.persistent.cached_scene.take() {
        record_worker_reuse(recorder, &buf);
        if cached_bucket >= bucket {
            return Ok(buf);
        }
        defer_buffer_until_retired(recorder.context(), buf);
    }
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(bucket, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// Allocate or reuse the stable device buffer for the fine-pass config uniform.
///
/// Written each frame by a GPU `CopyBuffer` from `coarse_config`; never CPU-written.
fn alloc_or_reuse_fine_config(recorder: &mut SchemeRecorder<'_>) -> Result<Buffer, Error> {
    if let Some(buf) = recorder.persistent.cached_fine_config.take() {
        return Ok(buf);
    }
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(
            size_of::<ConfigUniform>() as u64,
            BufferKind::Scattered,
            Some(size_of::<ConfigUniform>() as u32),
            BufferFlags::empty(),
            None,
        )
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// Re-supply scene bytes to the worker scheme via the host-write sidecar.
///
/// Unlike [`crate::scheme::Scheme::commit_write_parcel`], this does not append an IR node
/// or dirty the scheme, so the worker topology stays retained across frames while the bytes
/// are refreshed on every submit. The reuse reference table gates the write on the CPU so
/// the submit worker only overwrites the buffer after the prior GPU reader has retired.
pub(crate) fn defer_scene_write_on_worker(recorder: &mut SchemeRecorder<'_>, scene: &Buffer, bytes: &[u8]) {
    recorder.persistent.config_scene_dirty = true;
    let refs = scene.last_referenced();
    recorder
        .scheme()
        .defer_host_write(&refs, scene, 0, bytes.to_vec().into_boxed_slice());
}

/// Re-supply config uniform bytes to the worker scheme via the host-write sidecar.
///
/// See [`defer_scene_write_on_worker`]; keeps the worker IR structurally clean.
pub(crate) fn defer_config_write_on_worker(recorder: &mut SchemeRecorder<'_>, config: &Buffer, bytes: &[u8]) {
    let refs = config.last_referenced();
    recorder
        .scheme()
        .defer_host_write(&refs, config, 0, bytes.to_vec().into_boxed_slice());
}

/// Allocate or reuse a stable bump buffer for the retained worker.
pub(crate) fn alloc_or_reuse_bump(recorder: &mut SchemeRecorder<'_>, size: u64) -> Result<Buffer, Error> {
    if let Some((cached_size, buf)) = recorder.persistent.cached_bump.take() {
        if cached_size == size {            record_worker_reuse(recorder, &buf);
            return Ok(buf);
        }        recorder.persistent.cached_bump_grant = None;
        defer_buffer_until_retired(recorder.context(), buf);
    }    recorder
        .persistent
        .retained_pool
        .acquire_buffer(
            size,
            BufferKind::Scattered,
            Some(size_of::<BumpAllocators>() as u32),
            BufferFlags::CPU_READABLE,
            None,
        )
        .map_err(|e| Error::Gpu(e.to_string()))
}

fn take_cached_texture(
    cached: &mut Option<(u32, u32, Texture)>,
    width: u32,
    height: u32,
) -> Result<Texture, Box<Option<Texture>>> {
    let Some((cw, ch, tex)) = cached.as_ref() else {
        return Err(Box::new(None));
    };
    if *cw == width && *ch == height {
        Ok(tex.borrow())
    } else {
        Err(Box::new(Some(cached.take().unwrap().2)))
    }
}

fn install_cached_texture(cached: &mut Option<(u32, u32, Texture)>, width: u32, height: u32, tex: Texture) -> Texture {
    cached.replace((width, height, tex));
    cached.as_ref().unwrap().2.borrow()
}

enum TextureStagingCache {
    Gradient,
    Mask,
}

/// Returns the footprint for a 2D texture upload (handles platform row-pitch differences).
///
/// The returned [`TextureCopyFootprint`] tells callers how to allocate and write the staging
/// buffer so the DX12 backend can skip the intermediate repack step.  On Vulkan/Metal the
/// footprint is always tight (`row_pitch == width * 4`), so no extra padding is needed.
fn query_upload_footprint(
    recorder: &SchemeRecorder<'_>,
    width: u32,
    height: u32,
) -> Result<TextureCopyFootprint, Error> {
    recorder
        .device()
        .texture_copy_footprint(width, height, TextureFormat::Rgba8Unorm)
        .map_err(|e| Error::Gpu(e.to_string()))
}

fn alloc_or_reuse_full_texture_staging(
    recorder: &mut SchemeRecorder<'_>,
    cached: &mut Option<(u32, u32, Buffer)>,
    width: u32,
    height: u32,
    min_size: u64,
) -> Result<Buffer, Error> {
    if let Some((cw, ch, buf)) = cached.take() {
        if cw >= width && ch >= height && buf.byte_size() >= min_size {
            return Ok(buf);
        }
        defer_buffer_until_retired(recorder.context(), buf);
    }
    let size = min_size.max(4);
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(size, BufferKind::Scattered, Some(4), BufferFlags::CPU_WRITABLE, None)
        .map_err(|e| Error::Gpu(e.to_string()))
}

fn take_region_texture_staging(recorder: &mut SchemeRecorder<'_>, key: (u32, u32, u32, u32)) -> Option<Buffer> {
    if let Some(idx) = recorder
        .persistent
        .cached_image_region_stagings
        .iter()
        .position(|(k, _)| *k == key)
    {
        Some(recorder.persistent.cached_image_region_stagings.remove(idx).1)
    } else {
        None
    }
}

fn alloc_or_reuse_region_texture_staging(
    recorder: &mut SchemeRecorder<'_>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    min_size: u64,
) -> Result<Buffer, Error> {
    let key = (x, y, width, height);
    if let Some(buf) = take_region_texture_staging(recorder, key) {
        if buf.byte_size() >= min_size {
            return Ok(buf);
        }
        defer_buffer_until_retired(recorder.context(), buf);
    }
    let size = min_size.max(4);
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(size, BufferKind::Scattered, Some(4), BufferFlags::CPU_WRITABLE, None)
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// Write `bytes` (tight `width*height*4` layout) into `buf` using the row pitch
/// from `footprint`, so the DX12 backend can use the buffer directly without repacking.
///
/// When `footprint.row_pitch == footprint.tight_row_bytes()` (Vulkan, Metal, or a
/// perfectly aligned DX12 case) the bytes are written in a single contiguous call.
fn write_pitched(buf: &Buffer, bytes: &[u8], footprint: &TextureCopyFootprint) -> Result<(), Error> {
    let tight_row = footprint.tight_row_bytes() as usize;
    let row_pitch = footprint.row_pitch as u64;
    let base = footprint.footprint_offset;
    if footprint.row_pitch == footprint.tight_row_bytes() {
        buf.write(base, bytes).map_err(|e| Error::Gpu(e.to_string()))
    } else {
        for row in 0..footprint.height {
            let src = &bytes[row as usize * tight_row..(row as usize + 1) * tight_row];
            let dst_offset = base + row as u64 * row_pitch;
            buf.write(dst_offset, src).map_err(|e| Error::Gpu(e.to_string()))?;
        }
        Ok(())
    }
}

fn stage_texture_full(
    recorder: &mut SchemeRecorder<'_>,
    cached_staging: &mut Option<(u32, u32, Buffer)>,
    texture: &Texture,
    bytes: &[u8],
) -> Result<(), Error> {
    let width = texture.width();
    let height = texture.height();
    let footprint = query_upload_footprint(recorder, width, height)?;
    let staging =
        alloc_or_reuse_full_texture_staging(recorder, cached_staging, width, height, footprint.staging_bytes)?;
    write_pitched(&staging, bytes, &footprint)?;
    if recorder.upload_needs_record {
        recorder
            .upload_scheme()
            .copy_buffer_to_texture_parcel(
                staging.whole(),
                footprint.footprint_offset,
                footprint.row_pitch,
                texture,
                0,
                0,
                width,
                height,
            )
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
    *cached_staging = Some((width, height, staging));
    Ok(())
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
    let key = (x, y, width, height);
    let footprint = query_upload_footprint(recorder, width, height)?;
    let staging = alloc_or_reuse_region_texture_staging(recorder, x, y, width, height, footprint.staging_bytes)?;
    write_pitched(&staging, bytes, &footprint)?;
    if recorder.upload_needs_record {
        recorder
            .upload_scheme()
            .copy_buffer_to_texture_parcel(
                staging.whole(),
                footprint.footprint_offset,
                footprint.row_pitch,
                texture,
                x,
                y,
                width,
                height,
            )
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
    recorder.persistent.cached_image_region_stagings.push((key, staging));
    Ok(())
}

fn upload_texture_full(
    recorder: &mut SchemeRecorder<'_>,
    cache: TextureStagingCache,
    texture: &Texture,
    bytes: &[u8],
) -> Result<(), Error> {
    let mut slot = match cache {
        TextureStagingCache::Gradient => std::mem::take(&mut recorder.persistent.cached_gradient_staging),
        TextureStagingCache::Mask => std::mem::take(&mut recorder.persistent.cached_mask_staging),
    };
    stage_texture_full(recorder, &mut slot, texture, bytes)?;
    match cache {
        TextureStagingCache::Gradient => recorder.persistent.cached_gradient_staging = slot,
        TextureStagingCache::Mask => recorder.persistent.cached_mask_staging = slot,
    }
    Ok(())
}

fn upload_texture_region(
    recorder: &mut SchemeRecorder<'_>,
    texture: &Texture,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    bytes: Vec<u8>,
) -> Result<(), Error> {
    stage_texture_region(recorder, texture, x, y, width, height, &bytes)
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
            record_worker_reuse(recorder, &buf);
            Ok(buf)
        }
        None => alloc_pipeline_buffer(recorder, size, stride, name, BufferFlags::empty()),
    }
}

/// The seven large pipeline buffers whose sizes are fixed or change only on coarse
/// config changes — retained parcels in [`crate::goldy_renderer::PersistentState::retained_pool`].
/// See `resource-pool.md §1` for the rationale behind this split from [`ScratchPipelineBuffers`].
///
/// Reuse dependencies are recorded via [`record_worker_reuse`] at take time in
/// [`alloc_stable_buffer`]; enforcement runs on the submission worker at execute time.
/// prior frame's GPU work is still in flight on these buffers, a single retained deed is not
/// enough — use double-buffered parcels or a transient pool instead; do not keep them in
/// [`goldy::RetainedPool`] under inter-frame overlap.
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
        record_worker_reuse(recorder, &buffer);
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

pub(crate) struct PipelineResources {
    pub gradient: Texture,
    pub image_atlas: Texture,
    pub mask_atlas: Texture,
    pub scene: Buffer,
    pub coarse_config: Buffer,
    pub fine_config: Buffer,
    /// Composite indirect buffer (one ordinal `DispatchShape` parcel per stage).
    /// Cache key and buffer live in [`crate::goldy_renderer::PersistentState::cached_scheme_indirect`].
    pub indirect: Option<(WorkgroupCountsGpu, Buffer)>,
    pub stable: StablePipelineBuffers,
    pub scratch: ScratchPipelineBuffers,
    pub bump: Buffer,
    pub out_image: Texture,
    pub filter_layers: [Texture; 4],
    /// Buffer sizes used this frame, stored for cache-key comparison next frame.
    pub buffer_sizes: ekrano_encoding::BufferSizes,
    /// The `ConfigUniform` value uploaded to `coarse_config`, stored so that
    /// `schedule_pipeline_cleanup` can stash the buffer back into
    /// `PersistentState::cached_config_uniform` without re-reading GPU memory.
    pub config_uniform_value: ConfigUniform,
    /// Packed scene bytes last uploaded with `PersistentState::cached_config_uniform`.
    pub packed_scene_len: usize,
}

/// Move per-frame pipeline handles into [`PersistentState`] for cross-frame reuse.
///
/// Called at the start of the next frame (`SchemeRenderer::flush_pending_pipeline_cleanup`),
/// not on the post-`alloc_buffers` critical path.
pub(crate) fn install_scheme_pipeline_cache(
    persistent: &mut PersistentState,
    pipeline: PipelineResources,
) -> CacheScheduleOutcome {
    let _tz = goldy::tracy_zone!("ekrano.schedule_pipeline_cleanup");
    let outcome = CacheScheduleOutcome::default();
    let PipelineResources {
        gradient,
        image_atlas,
        mask_atlas,
        scene,
        coarse_config,
        fine_config,
        indirect,
        stable,
        scratch,
        bump,
        out_image,
        filter_layers,
        buffer_sizes,
        config_uniform_value,
        packed_scene_len,
    } = pipeline;

    let _ = (gradient, image_atlas, mask_atlas);
    persistent.cached_scene = Some((scene.byte_size(), scene));
    persistent.cached_config_uniform = Some((config_uniform_value, coarse_config));
    persistent.cached_fine_config = Some(fine_config);
    persistent.config_scene_dirty = false;
    persistent.cached_config_packed_len = packed_scene_len;
    if let Some((wg_counts_gpu, indirect_buf)) = indirect {
        persistent.cached_scheme_indirect = Some((wg_counts_gpu, indirect_buf));
    }
    persistent.cached_bump = Some((bump.byte_size(), bump));
    let pipeline_cache = crate::graph_gpu_resources::CachedPipeline {
        stable: crate::graph_gpu_resources::StablePipelineBuffers {
            info_bin_data: stable.info_bin_data,
            tile: stable.tile,
            segments: stable.segments,
            ptcl: stable.ptcl,
            blend_spill: stable.blend_spill,
            lines: stable.lines,
            seg_counts: stable.seg_counts,
        },
        scratch: crate::graph_gpu_resources::ScratchPipelineBuffers {
            reduced: scratch.reduced,
            reduced2: scratch.reduced2,
            reduced_scan: scratch.reduced_scan,
            tagmonoid: scratch.tagmonoid,
            path_bbox: scratch.path_bbox,
            draw_reduced: scratch.draw_reduced,
            draw_monoid: scratch.draw_monoid,
            clip_inp: scratch.clip_inp,
            clip_el: scratch.clip_el,
            clip_bic: scratch.clip_bic,
            clip_bbox: scratch.clip_bbox,
            draw_bbox: scratch.draw_bbox,
            bin_header: scratch.bin_header,
            path: scratch.path,
        },
        buffer_sizes,
    };
    assert!(
        persistent.cached_pipeline.is_none(),
        "cached_pipeline must be empty at install (prepare should have taken it)"
    );
    persistent.cached_pipeline = Some(pipeline_cache);
    log::debug!("[PIPE-CACHE] schedule: cached");
    persistent.store_scheme_render_targets(out_image, filter_layers);
    log::debug!("[RT-CACHE] schedule: scheme out_image stored (single slot)");
    outcome
}

/// Scene bytes written to GPU; graph uses a sentinel when the CPU pack is empty.
pub(crate) fn scene_upload_bytes(packed: &[u8]) -> std::borrow::Cow<'_, [u8]> {
    if packed.is_empty() {
        std::borrow::Cow::Owned(vec![u8::MAX; size_of::<u32>()])
    } else {
        std::borrow::Cow::Borrowed(packed)
    }
}

impl PipelineResources {
    pub(crate) fn prepare(
        recorder: &mut SchemeRecorder<'_>,
        coverage_mask: Option<&CoverageMask>,
        packed: &[u8],
        ramps: Ramps<'_>,
        images: Images<'_>,
        params: &RenderParams,
        config: &RenderConfig,
        out_image_format: TextureFormat,
    ) -> Result<Self, Error> {
        let scene_bytes = scene_upload_bytes(packed);

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

        // Resolve render targets first: only needs frame dimensions — no scene/config/buffer
        // dependencies. Cross-frame WAR on `out_image` is ordered by the submission-worker
        // FIFO at worker submit (fine write after prior present-copy read).
        let (out_image, filter_layers, _) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.render_targets");
            if let Some((cached_out, cached_layers)) = recorder.persistent.take_scheme_render_targets(
                params.width,
                params.height,
                out_image_format,
            )
            {
                (cached_out, cached_layers, true)
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
                (out, layers, false)
            }
        };

        let gradient = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.gradient");
            if ramps.height == 0 {
                match take_cached_texture(&mut recorder.persistent.cached_gradient, 1, 1) {
                    Ok(tex) => tex,
                    Err(stale) => {
                        if let Some(tex) = *stale {
                            recorder.persistent.tex_pool.release(tex);
                        }
                        let tex =
                            acquire_texture_rgba(recorder, 1, 1, TextureKind::Interpolated, TextureFlags::COPY_DST)?;
                        install_cached_texture(&mut recorder.persistent.cached_gradient, 1, 1, tex)
                    }
                }
            } else {
                let tex = match take_cached_texture(&mut recorder.persistent.cached_gradient, ramps.width, ramps.height)
                {
                    Ok(tex) => tex,
                    Err(stale) => {
                        if let Some(tex) = *stale {
                            recorder.persistent.tex_pool.release(tex);
                        }
                        let tex = acquire_texture_rgba(
                            recorder,
                            ramps.width,
                            ramps.height,
                            TextureKind::Interpolated,
                            TextureFlags::COPY_DST,
                        )?;
                        install_cached_texture(&mut recorder.persistent.cached_gradient, ramps.width, ramps.height, tex)
                    }
                };
                upload_texture_full(
                    recorder,
                    TextureStagingCache::Gradient,
                    &tex,
                    bytemuck::cast_slice(ramps.data),
                )?;
                tex
            }
        };

        let (image_atlas, _) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.image_atlas");
            if recorder.upload_needs_record {
                recorder.persistent.cached_image_region_stagings.clear();
            }
            if images.images.is_empty() {
                let t = match take_cached_texture(&mut recorder.persistent.cached_image_atlas, 1, 1) {
                    Ok(tex) => tex,
                    Err(stale) => {
                        if let Some(tex) = *stale {
                            recorder.persistent.tex_pool.release(tex);
                        }
                        let tex = acquire_texture_rgba(
                            recorder,
                            1,
                            1,
                            TextureKind::Interpolated,
                            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                        )?;
                        install_cached_texture(&mut recorder.persistent.cached_image_atlas, 1, 1, tex)
                    }
                };
                (t, (1_u32, 1_u32))
            } else {
                let t =
                    match take_cached_texture(&mut recorder.persistent.cached_image_atlas, images.width, images.height)
                    {
                        Ok(tex) => tex,
                        Err(stale) => {
                            if let Some(tex) = *stale {
                                recorder.persistent.tex_pool.release(tex);
                            }
                            let tex = acquire_texture_rgba(
                                recorder,
                                images.width,
                                images.height,
                                TextureKind::Interpolated,
                                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                            )?;
                            install_cached_texture(
                                &mut recorder.persistent.cached_image_atlas,
                                images.width,
                                images.height,
                                tex,
                            )
                        }
                    };
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
                    let tex = match take_cached_texture(&mut recorder.persistent.cached_mask_atlas, m.width, m.height) {
                        Ok(tex) => tex,
                        Err(stale) => {
                            if let Some(tex) = *stale {
                                recorder.persistent.tex_pool.release(tex);
                            }
                            let tex = acquire_texture_rgba(
                                recorder,
                                m.width,
                                m.height,
                                TextureKind::Interpolated,
                                TextureFlags::COPY_DST,
                            )?;
                            install_cached_texture(&mut recorder.persistent.cached_mask_atlas, m.width, m.height, tex)
                        }
                    };
                    let mut rgba = Vec::with_capacity(m.data.len() * 4);
                    for &b in m.data.iter() {
                        rgba.extend_from_slice(&[b, b, b, 255]);
                    }
                    upload_texture_full(recorder, TextureStagingCache::Mask, &tex, &rgba)?;
                    tex
                }
                None => {
                    let tex = match take_cached_texture(&mut recorder.persistent.cached_mask_atlas, 1, 1) {
                        Ok(tex) => tex,
                        Err(stale) => {
                            if let Some(tex) = *stale {
                                recorder.persistent.tex_pool.release(tex);
                            }
                            let tex = acquire_texture_rgba(
                                recorder,
                                1,
                                1,
                                TextureKind::Interpolated,
                                TextureFlags::COPY_DST,
                            )?;
                            install_cached_texture(&mut recorder.persistent.cached_mask_atlas, 1, 1, tex)
                        }
                    };
                    upload_texture_full(recorder, TextureStagingCache::Mask, &tex, &[255, 255, 255, 255])?;
                    tex
                }
            }
        };

        let config_uniform_value = cpu_config_owned.gpu;
        let packed_scene_len = scene_bytes.len();

        let coarse_config = {
            let outcome = {
                let _tz = goldy::tracy_zone!("ekrano.prepare.config_cache");
                let cached_snapshot = recorder.persistent.cached_config_uniform.as_ref().map(|(v, _)| *v);
                if let Some(cv) = cached_snapshot {
                    if cv == config_uniform_value {
                        ConfigUniformCacheOutcome::Hit(recorder.persistent.cached_config_uniform.take().unwrap().1)
                    } else if packed_scene_len > 0
                        && !recorder.persistent.config_scene_dirty
                        && packed_scene_len == recorder.persistent.cached_config_packed_len
                        && config_uniform_without_layout_eq(&cv, &config_uniform_value)
                    {
                        ConfigUniformCacheOutcome::LayoutRefresh(
                            recorder.persistent.cached_config_uniform.take().unwrap().1,
                        )
                    } else if let Some((_, buf)) = recorder.persistent.cached_config_uniform.take() {
                        ConfigUniformCacheOutcome::MissReuse(buf)
                    } else {
                        ConfigUniformCacheOutcome::MissAlloc
                    }
                } else {
                    ConfigUniformCacheOutcome::MissAlloc
                }
            };
            let branch = match &outcome {
                ConfigUniformCacheOutcome::Hit(_) => "hit",
                ConfigUniformCacheOutcome::LayoutRefresh(_) => "layout_refresh",
                ConfigUniformCacheOutcome::MissReuse(_) => "miss_reuse",
                ConfigUniformCacheOutcome::MissAlloc => "miss_alloc",
            };
            log::trace!("ConfigUniform cache {branch}");
            match outcome {
                ConfigUniformCacheOutcome::Hit(buf) => {
                    let _tz = goldy::tracy_zone!("ekrano.prepare.config_hit");
                    buf
                }
                ConfigUniformCacheOutcome::LayoutRefresh(buf) => {
                    let _tz = goldy::tracy_zone!("ekrano.prepare.config_layout_refresh");
                    buf
                }
                ConfigUniformCacheOutcome::MissReuse(buf) => {
                    record_worker_reuse(recorder, &buf);
                    buf
                }
                ConfigUniformCacheOutcome::MissAlloc => {
                    let config_buf = {
                        let _tz = goldy::tracy_zone!("ekrano.prepare.config_alloc");
                        recorder
                            .persistent
                            .retained_pool
                            .acquire_buffer(
                                size_of::<ConfigUniform>() as u64,
                                BufferKind::Scattered,
                                Some(size_of::<ConfigUniform>() as u32),
                                BufferFlags::empty(),
                                None,
                            )
                            .map_err(|e| Error::Gpu(e.to_string()))?
                    };
                    config_buf
                }
            }
        };

        let fine_config = alloc_or_reuse_fine_config(recorder)?;

        let scene = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload");
            let scene_buf = alloc_or_reuse_scene(recorder, scene_bytes.len())?;
            recorder.persistent.config_scene_dirty = true;
            scene_buf
        };

        let buffer_sizes = cpu_config_owned.buffer_sizes;

        // Try to reuse cached pipeline buffers from the previous frame.
        // Reuse gates live in alloc_stable_buffer / al_cached_opt (parcel ledger waits).
        let (cached_stable, cached_scratch) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.pipeline_cache");
            match recorder.persistent.take_cached_pipeline() {
                Some(c) if c.buffer_sizes == buffer_sizes => {
                    let stable = StablePipelineBuffers {
                        info_bin_data: c.stable.info_bin_data,
                        tile: c.stable.tile,
                        segments: c.stable.segments,
                        ptcl: c.stable.ptcl,
                        blend_spill: c.stable.blend_spill,
                        lines: c.stable.lines,
                        seg_counts: c.stable.seg_counts,
                    };
                    let scratch = ScratchPipelineBuffers {
                        reduced: c.scratch.reduced,
                        reduced2: c.scratch.reduced2,
                        reduced_scan: c.scratch.reduced_scan,
                        tagmonoid: c.scratch.tagmonoid,
                        path_bbox: c.scratch.path_bbox,
                        draw_reduced: c.scratch.draw_reduced,
                        draw_monoid: c.scratch.draw_monoid,
                        clip_inp: c.scratch.clip_inp,
                        clip_el: c.scratch.clip_el,
                        clip_bic: c.scratch.clip_bic,
                        clip_bbox: c.scratch.clip_bbox,
                        draw_bbox: c.scratch.draw_bbox,
                        bin_header: c.scratch.bin_header,
                        path: c.scratch.path,
                    };
                    (Some(stable), Some(scratch))
                }
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
                    let ppool = &mut recorder.persistent.pool;
                    ppool.return_buf(c.scratch.reduced, "ekrano.reduced_buf");
                    ppool.return_buf(c.scratch.reduced2, "ekrano.reduced2_buf");
                    ppool.return_buf(c.scratch.reduced_scan, "ekrano.reduced_scan_buf");
                    ppool.return_buf(c.scratch.tagmonoid, "ekrano.tagmonoid_buf");
                    ppool.return_buf(c.scratch.path_bbox, "ekrano.path_bbox_buf");
                    ppool.return_buf(c.scratch.draw_reduced, "ekrano.draw_reduced_buf");
                    ppool.return_buf(c.scratch.draw_monoid, "ekrano.draw_monoid_buf");
                    ppool.return_buf(c.scratch.clip_inp, "ekrano.clip_inp_buf");
                    ppool.return_buf(c.scratch.clip_el, "ekrano.clip_el_buf");
                    ppool.return_buf(c.scratch.clip_bic, "ekrano.clip_bic_buf");
                    ppool.return_buf(c.scratch.clip_bbox, "ekrano.clip_bbox_buf");
                    ppool.return_buf(c.scratch.draw_bbox, "ekrano.draw_bbox_buf");
                    ppool.return_buf(c.scratch.bin_header, "ekrano.bin_header_buf");
                    ppool.return_buf(c.scratch.path, "ekrano.path_buf");
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
        let bump_size = buffer_sizes.bump_alloc.size_in_bytes().into();
        let bump = alloc_or_reuse_bump(recorder, bump_size)?;
        // Bump clear is recorded on the worker scheme before coarse/fine (see worker_stage).

        Ok(Self {
            gradient,
            image_atlas,
            mask_atlas,
            scene,
            coarse_config,
            fine_config,
            indirect: None,
            stable,
            scratch,
            bump,
            out_image,
            filter_layers,
            buffer_sizes,
            config_uniform_value,
            packed_scene_len,
        })
    }
}

pub(crate) fn bind_type_to_node_access(bt: BindType) -> goldy::task_graph::NodeAccess {
    match bt {
        BindType::Buffer | BindType::Image(_) => goldy::task_graph::NodeAccess::ReadWrite,
        BindType::BufReadOnly | BindType::Uniform | BindType::ImageRead(_) => goldy::task_graph::NodeAccess::Read,
        BindType::Sampler => goldy::task_graph::NodeAccess::Read,
    }
}
