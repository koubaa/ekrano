// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scheme-backend scene recording.

use crate::scheme_renderer::SchemeRecorder;
use crate::scheme_gpu_resources::{GpuBinding, PipelineBuffer, PipelineResources};
use crate::shaders::FullShaders;
use crate::{AaConfig, RenderParams};

use std::mem::size_of;

use ekrano_encoding::{
    FilterPrimitive, FilterUniform, IndirectCount, LayerFilterEffect, WorkgroupCountsGpu, WorkgroupSize, make_mask_lut,
    make_mask_lut_16,
};
use goldy::types::{BufferFlags, TextureFlags, TextureKind};
use goldy::{Buffer, Texture};
use peniko::color::{PremulColor, Srgb};

use ekrano_encoding::{
    STAGE_BACKDROP, STAGE_BBOX_CLEAR, STAGE_BINNING, STAGE_CLIP_LEAF, STAGE_CLIP_REDUCE, STAGE_COARSE, STAGE_DRAW_LEAF,
    STAGE_DRAW_REDUCE, STAGE_FLATTEN, STAGE_PATH_COUNT, STAGE_PATH_TILING, STAGE_PATHTAG_REDUCE, STAGE_PATHTAG_REDUCE2,
    STAGE_PATHTAG_SCAN, STAGE_PATHTAG_SCAN_LARGE, STAGE_PATHTAG_SCAN1, STAGE_TILE_ALLOC,
};

/// State for a render in progress.
pub struct Render {
    fine_wg_count: Option<WorkgroupSize>,
    aa_config: AaConfig,

    #[cfg(feature = "debug_layers")]
    captured_buffers: Option<CapturedBuffers>,
}

#[cfg(feature = "debug_layers")]
impl Drop for Render {
    fn drop(&mut self) {
        if self.captured_buffers.is_some() {
            unreachable!("Render captured buffers without freeing them");
        }
    }
}

/// Placeholder for a future CPU/debug capture path (direct resources).
#[cfg(feature = "debug_layers")]
pub struct CapturedBuffers {
    pub sizes: ekrano_encoding::BufferSizes,
}

/// Max flatten workgroups per queue submit. Large single dispatches can exceed the
/// Windows ~2s GPU timeout (TDR) on stressed dashed paths.
const MAX_FLATTEN_WG_PER_SUBMIT: u32 = 8;
/// Must match `FLATTEN_WG` in `ekrano_encoding` (threads per flatten workgroup).
const FLATTEN_THREADS_PER_GROUP: u32 = 256;

fn dispatch_stage(
    recorder: &mut SchemeRecorder<'_>,
    indirect: &Buffer,
    shader: crate::ShaderId,
    stage: u32,
    stride: u64,
    bindings: &[GpuBinding<'_>],
) {
    recorder.dispatch_indirect(shader, indirect, u64::from(stage) * stride, bindings);
}

impl Default for Render {
    fn default() -> Self {
        Self::new()
    }
}

impl Render {
    pub fn new() -> Self {
        Self {
            fine_wg_count: None,
            aa_config: AaConfig::Area,
            #[cfg(feature = "debug_layers")]
            captured_buffers: None,
        }
    }

    /// Execute the coarse rasterization phase.
    pub(crate) fn run_coarse(
        &mut self,
        pipeline: &mut PipelineResources,
        shaders: &FullShaders,
        params: &RenderParams,
        robust: bool,
        config: &ekrano_encoding::RenderConfig,
        recorder: &mut SchemeRecorder<'_>,
    ) {
        // HACK: The coarse workgroup counts is the number of active bins.
        if (config.workgroup_counts.coarse.0 * config.workgroup_counts.coarse.1 * config.workgroup_counts.coarse.2)
            > 256
        {
            log::warn!(
                "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
                params.width,
                params.height
            );
        }
        let buffer_sizes = &config.buffer_sizes;
        let wg_counts = &config.workgroup_counts;

        const INDIRECT_STRIDE: u64 = size_of::<IndirectCount>() as u64;

        let wg_counts_gpu = WorkgroupCountsGpu::from(wg_counts);

        // Cache hit check: compare before taking ownership.
        let cache_hit = recorder
            .persistent
            .cached_wg_counts
            .as_ref()
            .is_some_and(|(v, _)| v == &wg_counts_gpu);
        log::trace!("wg_counts cache {}", if cache_hit { "HIT" } else { "MISS" });

        // Take the buffer out of the cache (or upload fresh on miss).
        // Buffer::Drop destroys the GPU resource, so we must NOT drop the cached buffer on
        // a hit — it will be restored into the cache after the dispatch below.
        let wg_buf = if cache_hit {
            recorder.persistent.cached_wg_counts.take().unwrap().1
        } else {
            if let Some((_, old_buf)) = recorder.persistent.cached_wg_counts.take() {
                // Return evicted buffer to pool so its GPU memory can be reused.
                recorder.persistent.pool.return_buf(old_buf, "ekrano.wg_counts");
            }
            recorder.upload_typed("ekrano.wg_counts", &wg_counts_gpu)
        };

        let indirect = recorder
            .alloc_pipeline_buffer_named(
                buffer_sizes.indirect_count.size_in_bytes().into(),
                size_of::<IndirectCount>() as u32,
                "ekrano.indirect_dispatch",
                BufferFlags::empty(),
            )
            .expect("indirect buffer");
        recorder.dispatch(
            shaders.pipeline_setup,
            (1, 1, 1),
            &[GpuBinding::Buf(&wg_buf), indirect.as_binding()],
        );

        // Restore buffer to persistent cache; its lifetime is now managed by
        // PersistentState. Do NOT call defer_owned_buffer here.
        recorder.persistent.cached_wg_counts = Some((wg_counts_gpu, wg_buf));
        pipeline.indirect = Some(indirect);

        let indirect_buf = pipeline
            .indirect
            .as_ref()
            .expect("pipeline_setup must produce an indirect dispatch buffer");

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.pathtag_reduce,
            STAGE_PATHTAG_REDUCE,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.reduced.as_binding(),
            ],
        );
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.pathtag_reduce2,
            STAGE_PATHTAG_REDUCE2,
            INDIRECT_STRIDE,
            &[
                pipeline.scratch.reduced.as_binding(),
                pipeline.scratch.reduced2.as_binding(),
            ],
        );
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.pathtag_scan1,
            STAGE_PATHTAG_SCAN1,
            INDIRECT_STRIDE,
            &[
                pipeline.scratch.reduced.as_binding(),
                pipeline.scratch.reduced2.as_binding(),
                pipeline.scratch.reduced_scan.as_binding(),
            ],
        );
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.pathtag_scan,
            STAGE_PATHTAG_SCAN,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.reduced.as_binding(),
                pipeline.scratch.tagmonoid.as_binding(),
            ],
        );
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.pathtag_scan_large,
            STAGE_PATHTAG_SCAN_LARGE,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.reduced_scan.as_binding(),
                pipeline.scratch.tagmonoid.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.bbox_clear,
            STAGE_BBOX_CLEAR,
            INDIRECT_STRIDE,
            &[pipeline.config.as_binding(), pipeline.scratch.path_bbox.as_binding()],
        );

        let flatten_bindings = [
            pipeline.config.as_binding(),
            pipeline.scene.as_binding(),
            pipeline.scratch.tagmonoid.as_binding(),
            pipeline.scratch.path_bbox.as_binding(),
            pipeline.bump.as_binding(),
            pipeline.stable.lines.as_binding(),
        ];
        let flat_wg_x = wg_counts.flatten.0;
        if flat_wg_x > MAX_FLATTEN_WG_PER_SUBMIT {
            let mut base_wg = 0_u32;
            while base_wg < flat_wg_x {
                let chunk = (flat_wg_x - base_wg).min(MAX_FLATTEN_WG_PER_SUBMIT);
                let thread_base = base_wg * FLATTEN_THREADS_PER_GROUP;
                recorder.dispatch_with_push_tail(shaders.flatten, (chunk, 1, 1), &flatten_bindings, &[thread_base]);
                base_wg += chunk;
            }
        } else {
            dispatch_stage(
                recorder,
                indirect_buf,
                shaders.flatten,
                STAGE_FLATTEN,
                INDIRECT_STRIDE,
                &flatten_bindings,
            );
        }

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.draw_reduce,
            STAGE_DRAW_REDUCE,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.draw_reduced.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.draw_leaf,
            STAGE_DRAW_LEAF,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.draw_reduced.as_binding(),
                pipeline.scratch.path_bbox.as_binding(),
                pipeline.scratch.draw_monoid.as_binding(),
                pipeline.stable.info_bin_data.as_binding(),
                pipeline.scratch.clip_inp.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.clip_reduce,
            STAGE_CLIP_REDUCE,
            INDIRECT_STRIDE,
            &[
                pipeline.scratch.clip_inp.as_binding(),
                pipeline.scratch.path_bbox.as_binding(),
                pipeline.scratch.clip_bic.as_binding(),
                pipeline.scratch.clip_el.as_binding(),
            ],
        );
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.clip_leaf,
            STAGE_CLIP_LEAF,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scratch.clip_inp.as_binding(),
                pipeline.scratch.path_bbox.as_binding(),
                pipeline.scratch.clip_bic.as_binding(),
                pipeline.scratch.clip_el.as_binding(),
                pipeline.scratch.draw_monoid.as_binding(),
                pipeline.scratch.clip_bbox.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.binning,
            STAGE_BINNING,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.draw_monoid.as_binding(),
                pipeline.scratch.path_bbox.as_binding(),
                pipeline.scratch.clip_bbox.as_binding(),
                pipeline.scratch.draw_bbox.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.stable.info_bin_data.as_binding(),
                pipeline.scratch.bin_header.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.tile_alloc,
            STAGE_TILE_ALLOC,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.draw_bbox.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.scratch.path.as_binding(),
                pipeline.stable.tile.as_binding(),
            ],
        );

        recorder.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            &[pipeline.bump.as_binding(), GpuBinding::Buf(indirect_buf)],
        );

        recorder.dispatch_indirect(
            shaders.path_count,
            indirect_buf,
            u64::from(STAGE_PATH_COUNT) * INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.stable.lines.as_binding(),
                pipeline.scratch.path.as_binding(),
                pipeline.stable.tile.as_binding(),
                pipeline.stable.seg_counts.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.backdrop,
            STAGE_BACKDROP,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.scratch.path.as_binding(),
                pipeline.stable.tile.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.coarse,
            STAGE_COARSE,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.draw_monoid.as_binding(),
                pipeline.scratch.bin_header.as_binding(),
                pipeline.stable.info_bin_data.as_binding(),
                pipeline.scratch.path.as_binding(),
                pipeline.stable.tile.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.stable.ptcl.as_binding(),
            ],
        );

        recorder.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            &[
                pipeline.bump.as_binding(),
                GpuBinding::Buf(indirect_buf),
                pipeline.stable.ptcl.as_binding(),
            ],
        );

        recorder.dispatch_indirect(
            shaders.path_tiling,
            indirect_buf,
            u64::from(STAGE_PATH_TILING) * INDIRECT_STRIDE,
            &[
                pipeline.bump.as_binding(),
                pipeline.stable.seg_counts.as_binding(),
                pipeline.stable.lines.as_binding(),
                pipeline.scratch.path.as_binding(),
                pipeline.stable.tile.as_binding(),
                pipeline.stable.segments.as_binding(),
            ],
        );

        self.fine_wg_count = Some(wg_counts.fine);
        self.aa_config = params.antialiasing_method;

        #[cfg(feature = "debug_layers")]
        if robust {
            self.captured_buffers = Some(CapturedBuffers {
                sizes: config.buffer_sizes,
            });
        }
        #[cfg(not(feature = "debug_layers"))]
        let _ = robust;
    }

    /// Run fine rasterization assuming the coarse phase succeeded.
    pub(crate) fn record_fine(
        &mut self,
        layer_filter_effects: &[LayerFilterEffect],
        shaders: &FullShaders,
        pipeline: &PipelineResources,
        output_texture: Option<&Texture>,
        recorder: &mut SchemeRecorder<'_>,
    ) {
        let fine_wg_count = self.fine_wg_count.take().expect("fine_wg_count");
        let width_in_tiles = fine_wg_count.0;
        let height_in_tiles = fine_wg_count.1;

        let out_tex = output_texture.unwrap_or(&pipeline.out_image);

        let shader = match self.aa_config {
            AaConfig::Area => shaders
                .fine_area
                .expect("shaders not configured to support AA mode: area"),
            AaConfig::Msaa16 => shaders
                .fine_msaa16
                .expect("shaders not configured to support AA mode: msaa16"),
            AaConfig::Msaa8 => shaders
                .fine_msaa8
                .expect("shaders not configured to support AA mode: msaa8"),
        };

        // Obtain a persistent mask LUT buffer for MSAA modes.  The LUT is static
        // (does not depend on scene content), so it is uploaded exactly once and
        // reused across frames without re-recording a WriteBuffer node.  This
        // keeps the fine command list free of staging-belt CopyBufferRegion
        // commands, allowing it to be safely retained and resubmitted.
        //
        // On first MSAA use: upload the LUT (adds a WriteBuffer node → fine CL
        // will not be retained this frame due to the `has_write_buffer()` guard).
        // On subsequent frames: the slot is `Some`, so we skip the upload and
        // only record the dispatch; the retained CL can then be safely reused.
        //
        // The check and upload are separated into two steps to satisfy the borrow
        // checker: `upload_strided` needs `&mut recorder`, so we cannot hold a
        // `&mut recorder.persistent.stable_mask_lut_*` across the call.
        let uses_mask_lut = matches!(self.aa_config, AaConfig::Msaa16 | AaConfig::Msaa8);
        if uses_mask_lut {
            let needs_upload = match self.aa_config {
                AaConfig::Msaa16 => recorder.persistent.stable_mask_lut_msaa16.is_none(),
                AaConfig::Msaa8 => recorder.persistent.stable_mask_lut_msaa8.is_none(),
                _ => false,
            };
            if needs_upload {
                let lut_data = match self.aa_config {
                    AaConfig::Msaa16 => make_mask_lut_16(),
                    AaConfig::Msaa8 => make_mask_lut(),
                    _ => unreachable!(),
                };
                let buf = recorder.upload_strided("ekrano.mask_lut", 4, lut_data);
                match self.aa_config {
                    AaConfig::Msaa16 => recorder.persistent.stable_mask_lut_msaa16 = Some(buf),
                    AaConfig::Msaa8 => recorder.persistent.stable_mask_lut_msaa8 = Some(buf),
                    _ => unreachable!(),
                }
            }
        }

        let width_px = out_tex.width();
        let height_px = out_tex.height();
        if !layer_filter_effects.is_empty() && width_px > 0 && height_px > 0 {
            if let Some(fs) = shaders.filter_pass {
                let wg = (width_px.div_ceil(16), height_px.div_ceil(16), 1);
                let u_clear = FilterUniform::clear_transparent(width_px, height_px);
                for fl in &pipeline.filter_layers {
                    filter_dispatch(recorder, fs, &u_clear, wg, out_tex, fl);
                }
            } else {
                log::warn!("filter_pass shader unavailable; cannot clear filter layer textures");
            }
        }

        let persistent = &*recorder.persistent;
        let mut fine_resources: Vec<GpuBinding<'_>> = vec![
            pipeline.config.as_binding(),
            pipeline.stable.segments.as_binding(),
            pipeline.stable.ptcl.as_binding(),
            pipeline.stable.info_bin_data.as_binding(),
            pipeline.stable.blend_spill.as_binding(),
            GpuBinding::Tex(out_tex),
            GpuBinding::Tex(&pipeline.gradient),
            GpuBinding::Tex(&pipeline.image_atlas),
            GpuBinding::Tex(&pipeline.mask_atlas),
        ];
        if uses_mask_lut {
            let lut = match self.aa_config {
                AaConfig::Msaa16 => persistent.stable_mask_lut_msaa16.as_ref(),
                AaConfig::Msaa8 => persistent.stable_mask_lut_msaa8.as_ref(),
                _ => None,
            };
            if let Some(lut) = lut {
                fine_resources.push(GpuBinding::PersistentBuf(lut));
            }
        }
        for fl in &pipeline.filter_layers {
            fine_resources.push(GpuBinding::Tex(fl));
        }
        // Hardware samplers for gradient ramps and image atlas (slots 13–14 / 14–15).
        fine_resources.push(GpuBinding::Sampler(
            persistent
                .linear_clamp_sampler
                .as_ref()
                .expect("linear_clamp_sampler must be initialised before fine pass"),
        ));
        fine_resources.push(GpuBinding::Sampler(
            persistent
                .nearest_clamp_sampler
                .as_ref()
                .expect("nearest_clamp_sampler must be initialised before fine pass"),
        ));

        SchemeRecorder::record_dispatch(
            recorder.scheme,
            recorder.shaders,
            shader,
            (width_in_tiles, height_in_tiles, 1),
            &fine_resources,
            &[],
        );
    }

    #[cfg(feature = "debug_layers")]
    pub fn take_captured_buffers(&mut self) -> Option<CapturedBuffers> {
        self.captured_buffers.take()
    }
}

fn premul_srgb_u32(c: PremulColor<Srgb>) -> u32 {
    c.to_rgba8().to_u32()
}

fn filter_dispatch(
    recorder: &mut SchemeRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    src: &Texture,
    dst: &Texture,
) {
    let slot = recorder.filter_dispatch_slot;
    recorder.filter_dispatch_slot += 1;

    // Take the cached entry for this slot (leaves None in its place temporarily).
    let cached = recorder
        .persistent
        .cached_filter_uniforms
        .get_mut(slot)
        .and_then(|e| e.take());

    // Produce the buffer: reuse cached on hit, upload fresh on miss.
    let buf = match cached {
        Some((ref val, buf)) if val == uniform => buf,
        Some((_, old_buf)) => {
            // Value changed: return stale buffer to pool, upload fresh.
            recorder.persistent.pool.return_buf(old_buf, "ekrano.filter_uniform");
            recorder.upload_typed("ekrano.filter_uniform", uniform)
        }
        None => recorder.upload_typed("ekrano.filter_uniform", uniform),
    };

    let persistent = &*recorder.persistent;
    let bindings = [
        GpuBinding::Buf(&buf),
        GpuBinding::Tex(src), // src_sampled (Interpolated — SRV)
        GpuBinding::Tex(src), // src (DirectSpatial — UAV)
        GpuBinding::Tex(dst), // dst (DirectSpatial — UAV)
        GpuBinding::Sampler(
            persistent
                .linear_clamp_sampler
                .as_ref()
                .expect("linear_clamp_sampler must be initialised before filter pass"),
        ),
    ];
    SchemeRecorder::record_dispatch(recorder.scheme, recorder.shaders, shader, wg, &bindings, &[]);

    // Restore buffer to persistent cache. Do NOT call defer_owned_buffer.
    let cache = &mut recorder.persistent.cached_filter_uniforms;
    if slot < cache.len() {
        cache[slot] = Some((*uniform, buf));
    } else {
        while cache.len() < slot {
            cache.push(None);
        }
        cache.push(Some((*uniform, buf)));
    }
}

/// Like `filter_dispatch` but uses `sampled_src` for the SRV slot and `uav_src` for the UAV slot.
/// Used by pyramid shadow composite `pass_kinds` (13/14) where the pre-blurred source and the
/// original foreground layer are different textures.
fn filter_dispatch_two_src(
    recorder: &mut SchemeRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    sampled_src: &Texture,
    uav_src: &Texture,
    dst: &Texture,
) {
    let slot = recorder.filter_dispatch_slot;
    recorder.filter_dispatch_slot += 1;

    let cached = recorder
        .persistent
        .cached_filter_uniforms
        .get_mut(slot)
        .and_then(|e| e.take());

    let buf = match cached {
        Some((ref val, buf)) if val == uniform => buf,
        Some((_, old_buf)) => {
            recorder.persistent.pool.return_buf(old_buf, "ekrano.filter_uniform");
            recorder.upload_typed("ekrano.filter_uniform", uniform)
        }
        None => recorder.upload_typed("ekrano.filter_uniform", uniform),
    };

    let persistent = &*recorder.persistent;
    let bindings = [
        GpuBinding::Buf(&buf),
        GpuBinding::Tex(sampled_src), // src_sampled (Interpolated — SRV)
        GpuBinding::Tex(uav_src),     // src (DirectSpatial — UAV)
        GpuBinding::Tex(dst),         // dst (DirectSpatial — UAV)
        GpuBinding::Sampler(
            persistent
                .linear_clamp_sampler
                .as_ref()
                .expect("linear_clamp_sampler must be initialised before filter pass"),
        ),
    ];
    SchemeRecorder::record_dispatch(recorder.scheme, recorder.shaders, shader, wg, &bindings, &[]);

    let cache = &mut recorder.persistent.cached_filter_uniforms;
    if slot < cache.len() {
        cache[slot] = Some((*uniform, buf));
    } else {
        while cache.len() < slot {
            cache.push(None);
        }
        cache.push(Some((*uniform, buf)));
    }
}

/// Apply a 2D pyramid blur on `src`, writing the blurred full-resolution result into `dst`.
///
/// The pyramid:
/// 1. Downsample `src` `num_levels` times (hardware bilinear, each level at half the previous).
/// 2. Blur the bottom level with the separable Gaussian (σ = `std_dev / 2^num_levels`).
/// 3. Upsample back `num_levels` times (hardware bilinear).
///
/// `levels` is clamped to a maximum of 6 (64× downsample). Allocates and releases transient
/// `DirectInterpolated` textures for each intermediate level.
fn pyramid_blur(
    shader: crate::ShaderId,
    recorder: &mut SchemeRecorder<'_>,
    src: &Texture,
    dst: &Texture,
    std_dev: f32,
    edge_mode: ekrano_encoding::FilterEdgeMode,
    width: u32,
    height: u32,
) {
    let levels = (std_dev.log2().floor() as u32).clamp(1, 6) as usize;
    let sigma_residual = std_dev / (1_u32 << levels) as f32;

    // Allocate pyramid textures (level 0 = half full-res, level `levels-1` = bottom).
    let pyramid: Vec<Texture> = (0..levels)
        .map(|l| {
            let lw = (width >> (l + 1)).max(1);
            let lh = (height >> (l + 1)).max(1);
            recorder
                .acquire_texture_rgba(lw, lh, TextureKind::DirectInterpolated, TextureFlags::empty())
                .expect("pyramid level texture")
        })
        .collect();

    // Downsample: src → pyramid[0] → pyramid[1] → ... → pyramid[levels-1]
    let mut prev_src = src;
    for (l, level) in pyramid.iter().enumerate() {
        let lw = (width >> (l + 1)).max(1);
        let lh = (height >> (l + 1)).max(1);
        let wg = (lw.div_ceil(16), lh.div_ceil(16), 1);
        let u = FilterUniform::downsample(lw, lh);
        filter_dispatch(recorder, shader, &u, wg, prev_src, level);
        prev_src = level;
    }

    // Blur at bottom level (1D separable Gaussian with σ_residual).
    let bottom = &pyramid[levels - 1];
    let bw = (width >> levels).max(1);
    let bh = (height >> levels).max(1);
    let wg_b = (bw.div_ceil(16), bh.div_ceil(16), 1);

    // Allocate a transient scratch for the H-blur result at the bottom level.
    let bottom_scratch = recorder
        .acquire_texture_rgba(bw, bh, TextureKind::DirectInterpolated, TextureFlags::empty())
        .expect("pyramid bottom scratch");

    let u_h = FilterUniform::gaussian_blur(bw, bh, true, sigma_residual, edge_mode);
    filter_dispatch(recorder, shader, &u_h, wg_b, bottom, &bottom_scratch);
    let u_v = FilterUniform::gaussian_blur(bw, bh, false, sigma_residual, edge_mode);
    filter_dispatch(recorder, shader, &u_v, wg_b, &bottom_scratch, bottom);

    // Upsample: pyramid[levels-1] → ... → pyramid[0] → dst
    for l in (0..levels).rev() {
        let (dst_tex, uw, uh): (&Texture, u32, u32) = if l == 0 {
            (dst, width, height)
        } else {
            (&pyramid[l - 1], (width >> l).max(1), (height >> l).max(1))
        };
        let wg = (uw.div_ceil(16), uh.div_ceil(16), 1);
        let u = FilterUniform::upsample(uw, uh);
        filter_dispatch(recorder, shader, &u, wg, &pyramid[l], dst_tex);
    }

    // Release transient textures.
    for tex in pyramid {
        recorder.defer_texture(tex);
    }
    recorder.defer_texture(bottom_scratch);
}

/// Per-layer filter chain for layer filter effects after fine rasterization.
pub(crate) fn record_filter_effects(
    layer_filter_effects: &[LayerFilterEffect],
    shaders: &FullShaders,
    recorder: &mut SchemeRecorder<'_>,
    pipeline: &PipelineResources,
    output_override: Option<&Texture>,
) {
    let width = pipeline.out_image.width();
    let height = pipeline.out_image.height();
    let dest = output_override.unwrap_or(&pipeline.out_image);

    if width == 0 || height == 0 {
        return;
    }
    if layer_filter_effects.is_empty() {
        return;
    }
    let Some(shader) = shaders.filter_pass else {
        log::warn!("filter_pass shader unavailable; skipping layer_filter_effects");
        return;
    };

    let scratch = recorder
        .acquire_texture_rgba(
            width,
            height,
            TextureKind::DirectInterpolated,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )
        .expect("filter scratch texture");

    let wg = (width.div_ceil(16), height.div_ceil(16), 1);
    let filter_layers = &pipeline.filter_layers;

    for effect in layer_filter_effects {
        let idx = (effect.layer_index as usize).min(3);
        let ft = &filter_layers[idx];
        match &effect.primitive {
            FilterPrimitive::GaussianBlur { std_dev, edge_mode } => {
                let u_h = FilterUniform::gaussian_blur(width, height, true, *std_dev, *edge_mode);
                filter_dispatch(recorder, shader, &u_h, wg, ft, &scratch);
                let u_v = FilterUniform::gaussian_blur(width, height, false, *std_dev, *edge_mode);
                filter_dispatch(recorder, shader, &u_v, wg, &scratch, ft);
            }
            FilterPrimitive::Offset { dx, dy } => {
                let edge = ekrano_encoding::FilterEdgeMode::Duplicate;
                let u = FilterUniform::offset(width, height, *dx, *dy, edge);
                filter_dispatch(recorder, shader, &u, wg, ft, &scratch);
                let u_copy = FilterUniform::copy(width, height);
                filter_dispatch(recorder, shader, &u_copy, wg, &scratch, ft);
            }
            FilterPrimitive::Flood { color, clip_rect } => {
                let u = FilterUniform::flood(width, height, premul_srgb_u32(*color), *clip_rect);
                filter_dispatch(recorder, shader, &u, wg, ft, &scratch);
                let u_copy = FilterUniform::copy(width, height);
                filter_dispatch(recorder, shader, &u_copy, wg, &scratch, ft);
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_dev,
                color,
                edge_mode,
            } => {
                // For large radii use the pyramid path; small radii use the one-shot path.
                const PYRAMID_THRESHOLD: f32 = 16.0;
                if *std_dev > PYRAMID_THRESHOLD {
                    // Allocate a full-res DirectInterpolated scratch for the blurred alpha.
                    let blur_dst = recorder
                        .acquire_texture_rgba(width, height, TextureKind::DirectInterpolated, TextureFlags::empty())
                        .expect("pyramid blur_dst");

                    if effect.is_nested {
                        let inner_idx = (effect.layer_index as usize).saturating_sub(1).min(3);
                        let inner_ft = &filter_layers[inner_idx];
                        pyramid_blur(
                            shader, recorder, inner_ft, &blur_dst, *std_dev, *edge_mode, width, height,
                        );
                        let u_comp = FilterUniform::shadow_composite_preblurred_nested(
                            width,
                            height,
                            *dx,
                            *dy,
                            premul_srgb_u32(*color),
                        );
                        filter_dispatch_two_src(recorder, shader, &u_comp, wg, &blur_dst, inner_ft, ft);
                    } else {
                        pyramid_blur(shader, recorder, ft, &blur_dst, *std_dev, *edge_mode, width, height);
                        let u_comp = FilterUniform::shadow_composite_preblurred(
                            width,
                            height,
                            *dx,
                            *dy,
                            premul_srgb_u32(*color),
                        );
                        filter_dispatch_two_src(recorder, shader, &u_comp, wg, &blur_dst, ft, &scratch);
                        let u_copy = FilterUniform::copy(width, height);
                        filter_dispatch(recorder, shader, &u_copy, wg, &scratch, ft);
                    }
                    recorder.defer_texture(blur_dst);
                } else if effect.is_nested {
                    let inner_idx = (effect.layer_index as usize).saturating_sub(1).min(3);
                    let inner_ft = &filter_layers[inner_idx];
                    let u = FilterUniform::drop_shadow_nested(
                        width,
                        height,
                        *dx,
                        *dy,
                        *std_dev,
                        premul_srgb_u32(*color),
                        *edge_mode,
                    );
                    filter_dispatch(recorder, shader, &u, wg, inner_ft, &scratch);
                    let u_copy = FilterUniform::copy(width, height);
                    filter_dispatch(recorder, shader, &u_copy, wg, &scratch, ft);
                    continue;
                } else {
                    let u = FilterUniform::drop_shadow(
                        width,
                        height,
                        *dx,
                        *dy,
                        *std_dev,
                        premul_srgb_u32(*color),
                        *edge_mode,
                    );
                    filter_dispatch(recorder, shader, &u, wg, ft, &scratch);
                    let u_copy = FilterUniform::copy(width, height);
                    filter_dispatch(recorder, shader, &u_copy, wg, &scratch, ft);
                }
            }
        }
    }

    for effect in layer_filter_effects.iter().filter(|e| e.is_nested) {
        let idx = (effect.layer_index as usize).min(3);
        let ft = &filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch(recorder, shader, &u_comp, wg, ft, dest);
    }
    for effect in layer_filter_effects.iter().filter(|e| !e.is_nested) {
        let idx = (effect.layer_index as usize).min(3);
        let ft = &filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch(recorder, shader, &u_comp, wg, ft, dest);
    }

    recorder.defer_texture(scratch);
}
