// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scheme-backend GPU resource helpers.

use std::mem::size_of;

use goldy::types::{BufferFlags, TextureFlags, TextureKind};
use goldy::{Buffer, BufferKind, DispatchShape, Init, Parcel, Sampler, Texture, TextureFormat, ordinal};

use crate::resource_proxy::BindType;
use crate::scheme_renderer::SchemeRecorder;
use crate::worker_retention::scene_size_bucket;
use crate::{Error, RenderParams, Result};
use ekrano_encoding::{
    BumpAllocators, CoverageMask, Images, N_INDIRECT_STAGES, Ramps, RenderConfig, STAGE_PATH_COUNT, STAGE_PATH_TILING,
    WorkgroupCountsGpu,
};

/// Record GPU-orderable reuse epochs on `scheme` for a buffer that will be overwritten.
fn record_buffer_reuse(scheme: &mut goldy::Scheme, buf: &Buffer) {
    let refs = buf.last_referenced();
    if !refs.is_empty() {
        scheme.record_reuse_epochs(&refs);
    }
}

/// Record GPU-orderable reuse epochs on `scheme` for a texture that will be overwritten.
fn record_texture_reuse(scheme: &mut goldy::Scheme, tex: &Texture) {
    let refs = tex.last_referenced();
    if !refs.is_empty() {
        scheme.record_reuse_epochs(&refs);
    }
}

/// Defer buffer drop until its last GPU reference retires (nonblocking path).
fn defer_buffer_until_retired(ctx: &goldy::Context, buf: Buffer) {
    let epoch = buf.last_referenced().iter().map(|(_, tv)| tv).max().unwrap_or(0);
    if epoch == 0 {
        drop(buf);
        return;
    }
    let mut payload = goldy::DeferredPayload::new();
    payload.push(buf);
    ctx.defer_release(epoch, payload);
}

/// Return a texture to the texture pool immediately, or after its GPU references retire.
fn release_or_defer_pooled_texture(recorder: &mut SchemeRecorder<'_>, tex: Texture) {
    if !recorder.nonblocking_reuse {
        recorder.persistent.tex_pool.release(tex);
        return;
    }
    let epoch = tex.last_referenced().iter().map(|(_, tv)| tv).max().unwrap_or(0);
    if epoch == 0 {
        recorder.persistent.tex_pool.release(tex);
        return;
    }
    let mut payload = goldy::DeferredPayload::new();
    let generation = std::sync::Arc::clone(&recorder.persistent.texture_return_generation);
    let created_generation = generation.load(std::sync::atomic::Ordering::Relaxed);
    payload.push(DeferredPoolTextureReturn {
        pending: std::sync::Arc::clone(&recorder.persistent.pending_texture_returns),
        generation,
        created_generation,
        tex: Some(tex),
    });
    recorder.context().defer_release(epoch, payload);
}

struct DeferredPoolTextureReturn {
    pending: std::sync::Arc<std::sync::Mutex<Vec<Texture>>>,
    generation: std::sync::Arc<std::sync::atomic::AtomicU64>,
    created_generation: u64,
    tex: Option<Texture>,
}

impl Drop for DeferredPoolTextureReturn {
    fn drop(&mut self) {
        if self.generation.load(std::sync::atomic::Ordering::Relaxed) != self.created_generation {
            // Metal resize purge invalidated this generation — drop without re-pooling.
            return;
        }
        if let Some(tex) = self.tex.take()
            && let Ok(mut pending) = self.pending.lock()
        {
            pending.push(tex);
        }
    }
}

/// Host-visible staging write into a scheme [`goldy::UploadBuffer`] (never waits).
fn stage_upload(
    recorder: &mut SchemeRecorder<'_>,
    upload: goldy::UploadBuffer,
    bytes: &[u8],
    what: &'static str,
) -> Result<(), Error> {
    recorder
        .upload_scheme()
        .stage_upload_buffer(&upload, 0, bytes)
        .map_err(|e| {
            Error::Gpu(format!(
                "{e} (what={what}, upload_id={}, bytes={}, needs_record={})",
                upload.id(),
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
    // Per-frame clears (e.g. bump) go through `clear_gpu_buf` on the upload scheme.
    Ok(buf)
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
        // WorkgroupCountsGpu changed: defer drop until prior GPU use retires.
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
        .commit_clear_parcel(buf, off, sz)
        .map_err(|e| Error::Shader(e.to_string()))?;
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
        defer_buffer_until_retired(recorder.context(), buf);
    }
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(bucket, BufferKind::Scattered, Some(4), BufferFlags::empty(), None)
        .map_err(|e| Error::Gpu(e.to_string()))
}

/// Allocate or reuse a logical upload buffer for scene bytes.
fn alloc_or_reuse_scene_upload(
    recorder: &mut SchemeRecorder<'_>,
    live_bytes: usize,
) -> Result<goldy::UploadBuffer, Error> {
    let bucket = scene_size_bucket(live_bytes);
    if !recorder.upload_needs_record
        && let Some((cached_bucket, ub)) = recorder.persistent.cached_scene_upload
        && cached_bucket >= bucket
    {
        return Ok(ub);
    }
    let ub = recorder
        .upload_scheme()
        .declare_upload_buffer(bucket)
        .map_err(|e| Error::Gpu(e.to_string()))?;
    recorder.persistent.cached_scene_upload = Some((bucket, ub));
    Ok(ub)
}

/// Allocate or reuse a logical upload buffer for the config uniform.
fn alloc_or_reuse_config_upload(recorder: &mut SchemeRecorder<'_>) -> Result<goldy::UploadBuffer, Error> {
    let size = size_of::<ekrano_encoding::ConfigUniform>() as u64;
    if !recorder.upload_needs_record
        && let Some(ub) = recorder.persistent.cached_config_upload
    {
        return Ok(ub);
    }
    let ub = recorder
        .upload_scheme()
        .declare_upload_buffer(size)
        .map_err(|e| Error::Gpu(e.to_string()))?;
    recorder.persistent.cached_config_upload = Some(ub);
    Ok(ub)
}

/// Write scene bytes into staging and copy into the device scene buffer when recording upload topology.
pub(crate) fn stage_scene_bytes(recorder: &mut SchemeRecorder<'_>, scene: &Buffer, bytes: &[u8]) -> Result<(), Error> {
    let bucket = scene_size_bucket(bytes.len());
    let upload = alloc_or_reuse_scene_upload(recorder, bytes.len())?;
    stage_upload(recorder, upload, bytes, "scene")?;
    if recorder.upload_needs_record {
        // Copy the full staging buffer (bucket-sized) rather than bytes.len() bytes.
        // On retained frames the packed content varies within the bucket; the GPU reads
        // up to config.scene_size bytes (always ≤ bucket), so the tail is never accessed.
        recorder
            .upload_scheme()
            .copy_upload_buffer(&upload, 0, scene.whole(), 0, bucket)
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
    Ok(())
}

/// Write config uniform bytes into staging and copy into the device config buffer when recording.
pub(crate) fn stage_config_bytes(
    recorder: &mut SchemeRecorder<'_>,
    config: &Buffer,
    bytes: &[u8],
) -> Result<(), Error> {
    let upload = alloc_or_reuse_config_upload(recorder)?;
    stage_upload(recorder, upload, bytes, "config")?;
    if recorder.upload_needs_record {
        recorder
            .upload_scheme()
            .copy_upload_buffer(&upload, 0, config.whole(), 0, bytes.len() as u64)
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
    Ok(())
}

/// Allocate or reuse a stable bump buffer for the retained worker.
pub(crate) fn alloc_or_reuse_bump(recorder: &mut SchemeRecorder<'_>, size: u64) -> Result<Buffer, Error> {
    if let Some((cached_size, buf)) = recorder.persistent.cached_bump.take() {
        if cached_size == size {
            record_buffer_reuse(recorder.scheme(), &buf);
            return Ok(buf);
        }
        recorder.persistent.cached_bump_grant = None;
        defer_buffer_until_retired(recorder.context(), buf);
    }
    recorder
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

fn alloc_or_reuse_full_texture_upload(
    recorder: &mut SchemeRecorder<'_>,
    cached: &mut Option<(u32, u32, u64, goldy::UploadBuffer)>,
    width: u32,
    height: u32,
    staging_bytes: u64,
) -> Result<goldy::UploadBuffer, Error> {
    let need = staging_bytes.max(4);
    if !recorder.upload_needs_record
        && let Some((cw, ch, cap, ub)) = *cached
        && cw >= width
        && ch >= height
        && cap >= need
    {
        return Ok(ub);
    }
    let ub = recorder
        .upload_scheme()
        .declare_upload_buffer(need)
        .map_err(|e| Error::Gpu(e.to_string()))?;
    *cached = Some((width, height, need, ub));
    Ok(ub)
}

fn take_region_texture_upload(
    recorder: &mut SchemeRecorder<'_>,
    key: (u32, u32, u32, u32),
) -> Option<goldy::UploadBuffer> {
    if recorder.upload_needs_record {
        return None;
    }
    recorder
        .persistent
        .cached_image_region_uploads
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, ub)| *ub)
}

fn alloc_or_reuse_region_texture_upload(
    recorder: &mut SchemeRecorder<'_>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    staging_bytes: u64,
) -> Result<goldy::UploadBuffer, Error> {
    let key = (x, y, width, height);
    if let Some(ub) = take_region_texture_upload(recorder, key) {
        return Ok(ub);
    }
    let ub = recorder
        .upload_scheme()
        .declare_upload_buffer(staging_bytes.max(4))
        .map_err(|e| Error::Gpu(e.to_string()))?;
    recorder.persistent.cached_image_region_uploads.push((key, ub));
    Ok(ub)
}

fn stage_texture_full(
    recorder: &mut SchemeRecorder<'_>,
    cached_upload: &mut Option<(u32, u32, u64, goldy::UploadBuffer)>,
    texture: &Texture,
    bytes: &[u8],
) -> Result<(), Error> {
    let width = texture.width();
    let height = texture.height();
    record_texture_reuse(recorder.upload_scheme(), texture);
    let layout = recorder
        .device()
        .texture_copy_footprint(width, height, texture.format())
        .map_err(|e| Error::Gpu(e.to_string()))?;
    let pitched = if layout.row_pitch == layout.tight_row_bytes() {
        bytes.to_vec()
    } else {
        pack_rgba_to_pitch(bytes, width, height, layout.row_pitch)
    };
    let staging_bytes = layout.staging_bytes.max(pitched.len() as u64);
    let upload = alloc_or_reuse_full_texture_upload(recorder, cached_upload, width, height, staging_bytes)?;
    stage_upload(recorder, upload, &pitched, "texture_full")?;
    if recorder.upload_needs_record {
        recorder
            .upload_scheme()
            .copy_upload_buffer_to_texture(
                &upload,
                layout.footprint_offset,
                layout.row_pitch,
                texture,
                0,
                0,
                width,
                height,
            )
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
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
    record_texture_reuse(recorder.upload_scheme(), texture);
    let layout = recorder
        .device()
        .texture_copy_footprint(width, height, texture.format())
        .map_err(|e| Error::Gpu(e.to_string()))?;
    let pitched = if layout.row_pitch == layout.tight_row_bytes() {
        bytes.to_vec()
    } else {
        pack_rgba_to_pitch(bytes, width, height, layout.row_pitch)
    };
    let staging_bytes = layout.staging_bytes.max(pitched.len() as u64);
    let upload = alloc_or_reuse_region_texture_upload(recorder, x, y, width, height, staging_bytes)?;
    stage_upload(recorder, upload, &pitched, "texture_region")?;
    if recorder.upload_needs_record {
        recorder
            .upload_scheme()
            .copy_upload_buffer_to_texture(
                &upload,
                layout.footprint_offset,
                layout.row_pitch,
                texture,
                x,
                y,
                width,
                height,
            )
            .map_err(|e| Error::Shader(e.to_string()))?;
    }
    Ok(())
}

fn upload_texture_full(
    recorder: &mut SchemeRecorder<'_>,
    cache: TextureStagingCache,
    texture: &Texture,
    bytes: &[u8],
) -> Result<(), Error> {
    let mut slot = match cache {
        TextureStagingCache::Gradient => std::mem::take(&mut recorder.persistent.cached_gradient_upload),
        TextureStagingCache::Mask => std::mem::take(&mut recorder.persistent.cached_mask_upload),
    };
    stage_texture_full(recorder, &mut slot, texture, bytes)?;
    match cache {
        TextureStagingCache::Gradient => recorder.persistent.cached_gradient_upload = slot,
        TextureStagingCache::Mask => recorder.persistent.cached_mask_upload = slot,
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
            record_buffer_reuse(recorder.scheme(), &buf);
            Ok(buf)
        }
        None => alloc_pipeline_buffer(recorder, size, stride, name, BufferFlags::empty()),
    }
}

/// The seven large pipeline buffers whose sizes are fixed or change only on coarse
/// config changes — retained parcels in [`crate::goldy_renderer::PersistentState::retained_pool`].
/// See `resource-pool.md §1` for the rationale behind this split from [`ScratchPipelineBuffers`].
///
/// Cross-frame reuse is ordered by [`goldy::Scheme::record_reuse_epochs`] on the worker
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
    pub config: Buffer,
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
    /// The `ConfigUniform` value uploaded to `config`, stored so that
    /// `schedule_pipeline_cleanup` can stash the buffer back into
    /// `PersistentState::cached_config_uniform` without re-reading GPU memory.
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
                match take_cached_texture(&mut recorder.persistent.cached_gradient, 1, 1) {
                    Ok(tex) => tex,
                    Err(stale) => {
                        if let Some(tex) = *stale {
                            release_or_defer_pooled_texture(recorder, tex);
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
                            release_or_defer_pooled_texture(recorder, tex);
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
                recorder.persistent.cached_image_region_uploads.clear();
            }
            if images.images.is_empty() {
                let t = match take_cached_texture(&mut recorder.persistent.cached_image_atlas, 1, 1) {
                    Ok(tex) => tex,
                    Err(stale) => {
                        if let Some(tex) = *stale {
                            release_or_defer_pooled_texture(recorder, tex);
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
                                release_or_defer_pooled_texture(recorder, tex);
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
                                release_or_defer_pooled_texture(recorder, tex);
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
                                release_or_defer_pooled_texture(recorder, tex);
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

        let scene = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.scene_upload");
            let scene_buf = alloc_or_reuse_scene(recorder, packed.len())?;
            stage_scene_bytes(recorder, &scene_buf, &packed)?;
            scene_buf
        };

        let config_uniform_value = cpu_config_owned.gpu;

        let config = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.config_upload");
            let config_buf = if let Some((_, buf)) = recorder.persistent.cached_config_uniform.take() {
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
            };
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
                        // RetainedPool adopts with ready_after = last_referenced.
                        pool.release_buffer(ctx, buffer);
                    }
                    let scratch_returns = vec![
                        (c.scratch.reduced, "ekrano.reduced_buf"),
                        (c.scratch.reduced2, "ekrano.reduced2_buf"),
                        (c.scratch.reduced_scan, "ekrano.reduced_scan_buf"),
                        (c.scratch.tagmonoid, "ekrano.tagmonoid_buf"),
                        (c.scratch.path_bbox, "ekrano.path_bbox_buf"),
                        (c.scratch.draw_reduced, "ekrano.draw_reduced_buf"),
                        (c.scratch.draw_monoid, "ekrano.draw_monoid_buf"),
                        (c.scratch.clip_inp, "ekrano.clip_inp_buf"),
                        (c.scratch.clip_el, "ekrano.clip_el_buf"),
                        (c.scratch.clip_bic, "ekrano.clip_bic_buf"),
                        (c.scratch.clip_bbox, "ekrano.clip_bbox_buf"),
                        (c.scratch.draw_bbox, "ekrano.draw_bbox_buf"),
                        (c.scratch.bin_header, "ekrano.bin_header_buf"),
                        (c.scratch.path, "ekrano.path_buf"),
                    ];
                    if recorder.nonblocking_reuse {
                        let mut epoch = 0_u64;
                        for (buf, _) in &scratch_returns {
                            for (_, tv) in buf.last_referenced().iter() {
                                epoch = epoch.max(tv);
                            }
                        }
                        if epoch > 0 {
                            crate::goldy_renderer::defer_frame_gpu_resources(
                                ctx,
                                recorder.persistent,
                                epoch,
                                Vec::new(),
                                scratch_returns,
                            );
                        } else {
                            let ppool = &mut recorder.persistent.pool;
                            for (buf, name) in scratch_returns {
                                ppool.return_buf(buf, name);
                            }
                        }
                    } else {
                        let ppool = &mut recorder.persistent.pool;
                        for (buf, name) in scratch_returns {
                            ppool.return_buf(buf, name);
                        }
                    }
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
        clear_gpu_buf(recorder, &bump, 0, None)?;

        // Try to reuse cached render targets from the previous frame (avoids TexturePool
        // round-trips when render dimensions are stable across frames).
        let (out_image, filter_layers) = {
            let _tz = goldy::tracy_zone!("ekrano.prepare.render_targets");
            if let Some((cached_out, cached_layers)) = recorder.persistent.take_scheme_render_targets(
                recorder.context(),
                params.width,
                params.height,
                out_image_format,
            ) {
                record_texture_reuse(recorder.scheme(), &cached_out);
                for layer in &cached_layers {
                    record_texture_reuse(recorder.scheme(), layer);
                }
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
                let layers = [
                    acquire_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                    acquire_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                    acquire_texture_rgba(
                        recorder,
                        params.width,
                        params.height,
                        TextureKind::DirectInterpolated,
                        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
                    )?,
                    acquire_texture_rgba(
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
            buffer_sizes,
            config_uniform_value,
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
