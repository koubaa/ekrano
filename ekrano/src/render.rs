// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Take an encoded scene and create a graph to render it

use crate::goldy_renderer::FrameRecorder;
use crate::gpu_resources::{GpuBinding, PipelineResources};
use crate::shaders::FullShaders;
use crate::{AaConfig, RenderParams};

use std::mem::size_of;

use ekrano_encoding::{
    Encoding, FilterPrimitive, FilterUniform, IndirectCount, WorkgroupCountsGpu, WorkgroupSize,
    make_mask_lut, make_mask_lut_16,
};
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags};
use goldy::{Buffer, Texture};
use peniko::color::{PremulColor, Srgb};

use ekrano_encoding::{
    STAGE_BACKDROP, STAGE_BBOX_CLEAR, STAGE_BINNING, STAGE_CLIP_LEAF, STAGE_CLIP_REDUCE,
    STAGE_COARSE, STAGE_DRAW_LEAF, STAGE_DRAW_REDUCE, STAGE_FLATTEN, STAGE_PATH_COUNT,
    STAGE_PATH_TILING, STAGE_PATHTAG_REDUCE, STAGE_PATHTAG_REDUCE2, STAGE_PATHTAG_SCAN,
    STAGE_PATHTAG_SCAN_LARGE, STAGE_PATHTAG_SCAN1, STAGE_TILE_ALLOC,
};

/// State for a render in progress.
pub struct Render {
    fine_wg_count: Option<WorkgroupSize>,
    aa_config: AaConfig,
    /// MSAA subpixel mask LUT (uploaded once, reused while this [`Render`] lives).
    mask_lut_buf: Option<Buffer>,

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
    recorder: &mut FrameRecorder<'_>,
    use_indirect: bool,
    indirect: &Buffer,
    shader: crate::ShaderId,
    stage: u32,
    wg: WorkgroupSize,
    stride: u64,
    bindings: &[GpuBinding<'_>],
) {
    if use_indirect {
        recorder.dispatch_indirect(shader, indirect, u64::from(stage) * stride, bindings);
    } else {
        recorder.dispatch(shader, wg, bindings);
    }
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
            mask_lut_buf: None,
            #[cfg(feature = "debug_layers")]
            captured_buffers: None,
        }
    }

    /// Execute the coarse rasterization phase.
    pub(crate) fn run_coarse(
        &mut self,
        _encoding: &Encoding,
        pipeline: &mut PipelineResources,
        shaders: &FullShaders,
        params: &RenderParams,
        robust: bool,
        config: &ekrano_encoding::RenderConfig,
        recorder: &mut FrameRecorder<'_>,
    ) {
        // HACK: The coarse workgroup counts is the number of active bins.
        if (config.workgroup_counts.coarse.0
            * config.workgroup_counts.coarse.1
            * config.workgroup_counts.coarse.2)
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

        let use_indirect = shaders.pipeline_setup.is_some();
        const INDIRECT_STRIDE: u64 = size_of::<IndirectCount>() as u64;

        if use_indirect {
            let setup = shaders
                .pipeline_setup
                .expect("pipeline_setup when use_indirect");
            let wg_counts_gpu = WorkgroupCountsGpu::from(wg_counts);
            let wg_buf = recorder.upload_typed("ekrano.wg_counts", &wg_counts_gpu);
            let indirect = recorder
                .alloc_pipeline_buffer_named(
                    buffer_sizes.indirect_count.size_in_bytes().into(),
                    size_of::<IndirectCount>() as u32,
                    "ekrano.indirect_dispatch",
                    BufferFlags::empty(),
                )
                .expect("indirect buffer");
            recorder.dispatch(
                setup,
                (1, 1, 1),
                &[GpuBinding::Buf(&wg_buf), indirect.as_binding()],
            );
            recorder.defer_owned_buffer(wg_buf, "ekrano.wg_counts");
            pipeline.indirect = Some(indirect);
        }

        let use_large_path_scan = wg_counts.use_large_path_scan && !shaders.pathtag_is_cpu;

        let indirect_buf = pipeline
            .indirect
            .as_ref()
            .unwrap_or(&pipeline.fallback_indirect)
            .as_indirect_buffer()
            .expect("indirect buffer must be a `Buffer` for dispatch_indirect");
        // Decouple the borrow: indirect_buf only references `indirect` or
        // `fallback_indirect`, neither of which we free mid-pipeline.
        let indirect_buf: &Buffer = unsafe { &*(indirect_buf as *const Buffer) };

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.pathtag_reduce,
            STAGE_PATHTAG_REDUCE,
            wg_counts.path_reduce,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.reduced.as_binding(),
            ],
        );
        let mut pathtag_parent = &pipeline.reduced;
        if use_indirect {
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_reduce2,
                STAGE_PATHTAG_REDUCE2,
                wg_counts.path_reduce2,
                INDIRECT_STRIDE,
                &[
                    pipeline.reduced.as_binding(),
                    pipeline.reduced2.as_binding(),
                ],
            );
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_scan1,
                STAGE_PATHTAG_SCAN1,
                wg_counts.path_scan1,
                INDIRECT_STRIDE,
                &[
                    pipeline.reduced.as_binding(),
                    pipeline.reduced2.as_binding(),
                    pipeline.reduced_scan.as_binding(),
                ],
            );
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_scan,
                STAGE_PATHTAG_SCAN,
                wg_counts.path_scan,
                INDIRECT_STRIDE,
                &[
                    pipeline.config.as_binding(),
                    pipeline.scene.as_binding(),
                    pipeline.reduced.as_binding(),
                    pipeline.tagmonoid.as_binding(),
                ],
            );
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_scan_large,
                STAGE_PATHTAG_SCAN_LARGE,
                wg_counts.path_scan,
                INDIRECT_STRIDE,
                &[
                    pipeline.config.as_binding(),
                    pipeline.scene.as_binding(),
                    pipeline.reduced_scan.as_binding(),
                    pipeline.tagmonoid.as_binding(),
                ],
            );
        } else if use_large_path_scan {
            recorder.dispatch(
                shaders.pathtag_reduce2,
                wg_counts.path_reduce2,
                &[
                    pipeline.reduced.as_binding(),
                    pipeline.reduced2.as_binding(),
                ],
            );
            recorder.dispatch(
                shaders.pathtag_scan1,
                wg_counts.path_scan1,
                &[
                    pipeline.reduced.as_binding(),
                    pipeline.reduced2.as_binding(),
                    pipeline.reduced_scan.as_binding(),
                ],
            );
            pathtag_parent = &pipeline.reduced_scan;
            recorder.dispatch(
                shaders.pathtag_scan_large,
                wg_counts.path_scan,
                &[
                    pipeline.config.as_binding(),
                    pipeline.scene.as_binding(),
                    pathtag_parent.as_binding(),
                    pipeline.tagmonoid.as_binding(),
                ],
            );
        } else {
            recorder.dispatch(
                shaders.pathtag_scan,
                wg_counts.path_scan,
                &[
                    pipeline.config.as_binding(),
                    pipeline.scene.as_binding(),
                    pathtag_parent.as_binding(),
                    pipeline.tagmonoid.as_binding(),
                ],
            );
        }

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.bbox_clear,
            STAGE_BBOX_CLEAR,
            wg_counts.bbox_clear,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.path_bbox.as_binding(),
            ],
        );

        let flatten_bindings = [
            pipeline.config.as_binding(),
            pipeline.scene.as_binding(),
            pipeline.tagmonoid.as_binding(),
            pipeline.path_bbox.as_binding(),
            pipeline.bump.as_binding(),
            pipeline.lines.as_binding(),
        ];
        let flat_wg_x = wg_counts.flatten.0;
        if flat_wg_x > MAX_FLATTEN_WG_PER_SUBMIT {
            let mut base_wg = 0_u32;
            while base_wg < flat_wg_x {
                let chunk = (flat_wg_x - base_wg).min(MAX_FLATTEN_WG_PER_SUBMIT);
                let thread_base = base_wg * FLATTEN_THREADS_PER_GROUP;
                recorder.dispatch_with_push_tail(
                    shaders.flatten,
                    (chunk, 1, 1),
                    &flatten_bindings,
                    &[thread_base],
                );
                base_wg += chunk;
            }
        } else if use_indirect {
            dispatch_stage(
                recorder,
                true,
                indirect_buf,
                shaders.flatten,
                STAGE_FLATTEN,
                wg_counts.flatten,
                INDIRECT_STRIDE,
                &flatten_bindings,
            );
        } else {
            recorder.dispatch_with_push_tail(
                shaders.flatten,
                wg_counts.flatten,
                &flatten_bindings,
                &[0],
            );
        }

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.draw_reduce,
            STAGE_DRAW_REDUCE,
            wg_counts.draw_reduce,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.draw_reduced.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.draw_leaf,
            STAGE_DRAW_LEAF,
            wg_counts.draw_leaf,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.draw_reduced.as_binding(),
                pipeline.path_bbox.as_binding(),
                pipeline.draw_monoid.as_binding(),
                pipeline.info_bin_data.as_binding(),
                pipeline.clip_inp.as_binding(),
            ],
        );

        if use_indirect || wg_counts.clip_reduce.0 > 0 {
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.clip_reduce,
                STAGE_CLIP_REDUCE,
                wg_counts.clip_reduce,
                INDIRECT_STRIDE,
                &[
                    pipeline.clip_inp.as_binding(),
                    pipeline.path_bbox.as_binding(),
                    pipeline.clip_bic.as_binding(),
                    pipeline.clip_el.as_binding(),
                ],
            );
        }
        if use_indirect || wg_counts.clip_leaf.0 > 0 {
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.clip_leaf,
                STAGE_CLIP_LEAF,
                wg_counts.clip_leaf,
                INDIRECT_STRIDE,
                &[
                    pipeline.config.as_binding(),
                    pipeline.clip_inp.as_binding(),
                    pipeline.path_bbox.as_binding(),
                    pipeline.clip_bic.as_binding(),
                    pipeline.clip_el.as_binding(),
                    pipeline.draw_monoid.as_binding(),
                    pipeline.clip_bbox.as_binding(),
                ],
            );
        }

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.binning,
            STAGE_BINNING,
            wg_counts.binning,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.draw_monoid.as_binding(),
                pipeline.path_bbox.as_binding(),
                pipeline.clip_bbox.as_binding(),
                pipeline.draw_bbox.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.info_bin_data.as_binding(),
                pipeline.bin_header.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.tile_alloc,
            STAGE_TILE_ALLOC,
            wg_counts.tile_alloc,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.draw_bbox.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.path.as_binding(),
                pipeline.tile.as_binding(),
            ],
        );

        recorder.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            &[pipeline.bump.as_binding(), GpuBinding::Buf(indirect_buf)],
        );

        let path_count_offset = if use_indirect {
            u64::from(STAGE_PATH_COUNT) * INDIRECT_STRIDE
        } else {
            0
        };
        recorder.dispatch_indirect(
            shaders.path_count,
            indirect_buf,
            path_count_offset,
            &[
                pipeline.config.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.lines.as_binding(),
                pipeline.path.as_binding(),
                pipeline.tile.as_binding(),
                pipeline.seg_counts.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.backdrop,
            STAGE_BACKDROP,
            wg_counts.backdrop,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.path.as_binding(),
                pipeline.tile.as_binding(),
            ],
        );

        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.coarse,
            STAGE_COARSE,
            wg_counts.coarse,
            INDIRECT_STRIDE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.draw_monoid.as_binding(),
                pipeline.bin_header.as_binding(),
                pipeline.info_bin_data.as_binding(),
                pipeline.path.as_binding(),
                pipeline.tile.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.ptcl.as_binding(),
            ],
        );

        recorder.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            &[
                pipeline.bump.as_binding(),
                GpuBinding::Buf(indirect_buf),
                pipeline.ptcl.as_binding(),
            ],
        );

        let path_tiling_offset = if use_indirect {
            u64::from(STAGE_PATH_TILING) * INDIRECT_STRIDE
        } else {
            0
        };
        recorder.dispatch_indirect(
            shaders.path_tiling,
            indirect_buf,
            path_tiling_offset,
            &[
                pipeline.bump.as_binding(),
                pipeline.seg_counts.as_binding(),
                pipeline.lines.as_binding(),
                pipeline.path.as_binding(),
                pipeline.tile.as_binding(),
                pipeline.segments.as_binding(),
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
        encoding: &Encoding,
        shaders: &FullShaders,
        pipeline: &PipelineResources,
        output_override: Option<&Texture>,
        recorder: &mut FrameRecorder<'_>,
    ) {
        let fine_wg_count = self.fine_wg_count.take().expect("fine_wg_count");
        let width_in_tiles = fine_wg_count.0;
        let height_in_tiles = fine_wg_count.1;

        let out_tex = output_override.unwrap_or(&pipeline.out_image);

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

        if matches!(self.aa_config, AaConfig::Msaa16 | AaConfig::Msaa8)
            && self.mask_lut_buf.is_none()
        {
            let mask_lut = match self.aa_config {
                AaConfig::Msaa16 => make_mask_lut_16(),
                AaConfig::Msaa8 => make_mask_lut(),
                _ => unreachable!(),
            };
            let buf = recorder.upload_strided("ekrano.mask_lut", 4, mask_lut);
            self.mask_lut_buf = Some(buf);
        }

        let mut fine_resources: Vec<GpuBinding<'_>> = vec![
            pipeline.config.as_binding(),
            pipeline.segments.as_binding(),
            pipeline.ptcl.as_binding(),
            pipeline.info_bin_data.as_binding(),
            pipeline.blend_spill.as_binding(),
            GpuBinding::Tex(out_tex),
            GpuBinding::Tex(&pipeline.gradient),
            GpuBinding::Tex(&pipeline.image_atlas),
            GpuBinding::Tex(&pipeline.mask_atlas),
        ];
        if let Some(mask) = self.mask_lut_buf.as_ref() {
            fine_resources.push(GpuBinding::Buf(mask));
        }
        for fl in &pipeline.filter_layers {
            fine_resources.push(GpuBinding::Tex(fl));
        }
        // Hardware samplers for gradient ramps and image atlas (slots 13–14 / 14–15).
        fine_resources.push(GpuBinding::Sampler(
            recorder
                .persistent
                .linear_clamp_sampler
                .as_ref()
                .expect("linear_clamp_sampler must be initialised before fine pass")
                .bindless_index()
                .expect("linear_clamp_sampler has no bindless index"),
        ));
        fine_resources.push(GpuBinding::Sampler(
            recorder
                .persistent
                .nearest_clamp_sampler
                .as_ref()
                .expect("nearest_clamp_sampler must be initialised before fine pass")
                .bindless_index()
                .expect("nearest_clamp_sampler has no bindless index"),
        ));

        let width_px = out_tex.width();
        let height_px = out_tex.height();
        if !encoding.layer_filter_effects.is_empty() && width_px > 0 && height_px > 0 {
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

        recorder.dispatch(
            shader,
            (width_in_tiles, height_in_tiles, 1),
            &fine_resources,
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

fn linear_clamp_sampler_index(recorder: &FrameRecorder<'_>) -> u32 {
    recorder
        .persistent
        .linear_clamp_sampler
        .as_ref()
        .expect("linear_clamp_sampler must be initialised before filter pass")
        .bindless_index()
        .expect("linear_clamp_sampler has no bindless index")
}

fn filter_dispatch(
    recorder: &mut FrameRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    src: &Texture,
    dst: &Texture,
) {
    let sampler_idx = linear_clamp_sampler_index(recorder);
    let buf = recorder.upload_typed("ekrano.filter_uniform", uniform);
    recorder.dispatch(
        shader,
        wg,
        &[
            GpuBinding::Buf(&buf),
            GpuBinding::Tex(src),  // src_sampled (Interpolated — SRV)
            GpuBinding::Tex(src),  // src (DirectSpatial — UAV)
            GpuBinding::Tex(dst),  // dst (DirectSpatial — UAV)
            GpuBinding::Sampler(sampler_idx),
        ],
    );
    recorder.defer_owned_buffer(buf, "ekrano.filter_uniform");
}

/// Like `filter_dispatch` but uses `sampled_src` for the SRV slot and `uav_src` for the UAV slot.
/// Used by pyramid shadow composite pass_kinds (13/14) where the pre-blurred source and the
/// original foreground layer are different textures.
fn filter_dispatch_two_src(
    recorder: &mut FrameRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    sampled_src: &Texture,
    uav_src: &Texture,
    dst: &Texture,
) {
    let sampler_idx = linear_clamp_sampler_index(recorder);
    let buf = recorder.upload_typed("ekrano.filter_uniform", uniform);
    recorder.dispatch(
        shader,
        wg,
        &[
            GpuBinding::Buf(&buf),
            GpuBinding::Tex(sampled_src), // src_sampled (Interpolated — SRV)
            GpuBinding::Tex(uav_src),     // src (DirectSpatial — UAV)
            GpuBinding::Tex(dst),         // dst (DirectSpatial — UAV)
            GpuBinding::Sampler(sampler_idx),
        ],
    );
    recorder.defer_owned_buffer(buf, "ekrano.filter_uniform");
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
    recorder: &mut FrameRecorder<'_>,
    src: &Texture,
    dst: &Texture,
    std_dev: f32,
    edge_mode: ekrano_encoding::FilterEdgeMode,
    width: u32,
    height: u32,
) {
    let levels = (std_dev.log2().floor() as u32).clamp(1, 6) as usize;
    let sigma_residual = std_dev / (1u32 << levels) as f32;

    // Allocate pyramid textures (level 0 = half full-res, level `levels-1` = bottom).
    let pyramid: Vec<Texture> = (0..levels)
        .map(|l| {
            let lw = (width >> (l + 1)).max(1);
            let lh = (height >> (l + 1)).max(1);
            crate::gpu_resources::acquire_texture_rgba(
                recorder.device,
                recorder.persistent,
                lw,
                lh,
                SpatialAccess::DirectInterpolated,
                TextureFlags::empty(),
            )
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
    let bottom_scratch = crate::gpu_resources::acquire_texture_rgba(
        recorder.device,
        recorder.persistent,
        bw,
        bh,
        SpatialAccess::DirectInterpolated,
        TextureFlags::empty(),
    )
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

/// Per-layer filter chain for [`Encoding::layer_filter_effects`] after fine rasterization.
pub(crate) fn record_filter_effects(
    encoding: &Encoding,
    shaders: &FullShaders,
    recorder: &mut FrameRecorder<'_>,
    pipeline: &PipelineResources,
    output_override: Option<&Texture>,
) {
    let width = pipeline.out_image.width();
    let height = pipeline.out_image.height();
    let dest = output_override.unwrap_or(&pipeline.out_image);

    if width == 0 || height == 0 {
        return;
    }
    if encoding.layer_filter_effects.is_empty() {
        return;
    }
    let Some(shader) = shaders.filter_pass else {
        log::warn!("filter_pass shader unavailable; skipping layer_filter_effects");
        return;
    };

    let scratch = crate::gpu_resources::acquire_texture_rgba(
        recorder.device,
        recorder.persistent,
        width,
        height,
        SpatialAccess::DirectInterpolated,
        TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
    )
    .expect("filter scratch texture");

    let wg = (width.div_ceil(16), height.div_ceil(16), 1);
    let filter_layers = &pipeline.filter_layers;

    for effect in &encoding.layer_filter_effects {
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
                    let blur_dst = crate::gpu_resources::acquire_texture_rgba(
                        recorder.device,
                        recorder.persistent,
                        width,
                        height,
                        SpatialAccess::DirectInterpolated,
                        TextureFlags::empty(),
                    )
                    .expect("pyramid blur_dst");

                    if effect.is_nested {
                        let inner_idx = (effect.layer_index as usize).saturating_sub(1).min(3);
                        let inner_ft = &filter_layers[inner_idx];
                        pyramid_blur(
                            shader, recorder, inner_ft, &blur_dst, *std_dev, *edge_mode,
                            width, height,
                        );
                        let u_comp = FilterUniform::shadow_composite_preblurred_nested(
                            width, height, *dx, *dy, premul_srgb_u32(*color),
                        );
                        filter_dispatch_two_src(
                            recorder, shader, &u_comp, wg, &blur_dst, inner_ft, ft,
                        );
                    } else {
                        pyramid_blur(
                            shader, recorder, ft, &blur_dst, *std_dev, *edge_mode,
                            width, height,
                        );
                        let u_comp = FilterUniform::shadow_composite_preblurred(
                            width, height, *dx, *dy, premul_srgb_u32(*color),
                        );
                        filter_dispatch_two_src(
                            recorder, shader, &u_comp, wg, &blur_dst, ft, &scratch,
                        );
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

    for effect in encoding.layer_filter_effects.iter().filter(|e| e.is_nested) {
        let idx = (effect.layer_index as usize).min(3);
        let ft = &filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch(recorder, shader, &u_comp, wg, ft, dest);
    }
    for effect in encoding
        .layer_filter_effects
        .iter()
        .filter(|e| !e.is_nested)
    {
        let idx = (effect.layer_index as usize).min(3);
        let ft = &filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch(recorder, shader, &u_comp, wg, ft, dest);
    }

    recorder.defer_texture(scratch);
}
