// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scheme-backend scene recording.

use crate::scheme_gpu_resources::{GpuBinding, PipelineBuffer, PipelineResources, alloc_or_reuse_scheme_indirect};
use crate::scheme_renderer::SchemeRecorder;
use crate::shaders::FullShaders;
use crate::{AaConfig, RenderParams};

use ekrano_encoding::{
    FilterPrimitive, FilterUniform, LayerFilterEffect, WorkgroupCountsGpu, WorkgroupSize, make_mask_lut,
    make_mask_lut_16,
};
use goldy::types::{BufferFlags, BufferKind, TextureFlags, TextureKind};
use goldy::{Buffer, NodeAccess, PresentLease, Texture};
use peniko::color::{PremulColor, Srgb};
use std::mem::size_of;

use ekrano_encoding::{
    STAGE_BACKDROP, STAGE_BBOX_CLEAR, STAGE_BINNING, STAGE_CLIP_LEAF, STAGE_CLIP_REDUCE, STAGE_COARSE, STAGE_DRAW_LEAF,
    STAGE_DRAW_REDUCE, STAGE_FLATTEN, STAGE_PATH_COUNT, STAGE_PATH_TILING, STAGE_PATHTAG_REDUCE, STAGE_PATHTAG_REDUCE2,
    STAGE_PATHTAG_SCAN, STAGE_PATHTAG_SCAN_LARGE, STAGE_PATHTAG_SCAN1, STAGE_TILE_ALLOC,
};

/// Fine/filter final destination: offscreen texture or swapchain present lease.
#[derive(Clone, Copy)]
pub(crate) enum RenderOutput<'a> {
    Texture(&'a Texture),
    Present(&'a PresentLease),
}

impl<'a> RenderOutput<'a> {
    fn width(self, pipeline: &PipelineResources) -> u32 {
        match self {
            Self::Texture(tex) => tex.width(),
            Self::Present(_) => pipeline.frame_width,
        }
    }

    fn height(self, pipeline: &PipelineResources) -> u32 {
        match self {
            Self::Texture(tex) => tex.height(),
            Self::Present(_) => pipeline.frame_height,
        }
    }

    fn fine_binding(self) -> GpuBinding<'a> {
        match self {
            Self::Texture(tex) => GpuBinding::Tex(tex),
            Self::Present(lease) => GpuBinding::Present(lease, NodeAccess::Write),
        }
    }

    fn filter_dst_binding(self) -> GpuBinding<'a> {
        match self {
            Self::Texture(tex) => GpuBinding::Tex(tex),
            Self::Present(lease) => GpuBinding::Present(lease, NodeAccess::ReadWrite),
        }
    }
}

/// Resolve the render output target for fine/filter compositing.
pub(crate) fn resolve_render_output<'a>(
    pipeline: &'a PipelineResources,
    output_texture: Option<&'a Texture>,
    present_lease: Option<&'a PresentLease>,
) -> RenderOutput<'a> {
    if let Some(tex) = output_texture {
        RenderOutput::Texture(tex)
    } else if let Some(lease) = present_lease {
        RenderOutput::Present(lease)
    } else {
        RenderOutput::Texture(
            pipeline
                .out_image
                .as_ref()
                .expect("render output requires out_image or present lease"),
        )
    }
}

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
    #[allow(dead_code, reason = "reserved for future scheme-path debug capture")]
    pub sizes: ekrano_encoding::BufferSizes,
}

/// Flatten uses a single dispatch (`ConfigUniform::flatten_thread_base` stays 0).
/// Chunking via push-constant `thread_base` under-counted work on DX12.

fn dispatch_stage(
    recorder: &mut SchemeRecorder<'_>,
    indirect: &Buffer,
    shader: crate::ShaderId,
    stage: u32,
    bindings: &[GpuBinding<'_>],
) {
    recorder.dispatch_shape(shader, indirect.unit(stage as usize), bindings);
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
        let _ = buffer_sizes; // used in orig path; kept to avoid dead-code warnings on the field

        let wg_counts_gpu = WorkgroupCountsGpu::from(wg_counts);

        let indirect_composite =
            alloc_or_reuse_scheme_indirect(recorder, &wg_counts_gpu).expect("alloc_or_reuse_scheme_indirect");
        pipeline.indirect = Some((wg_counts_gpu, indirect_composite));

        let (_, indirect_buf) = pipeline
            .indirect
            .as_ref()
            .expect("alloc_or_reuse_scheme_indirect must produce indirect buffer");

        if wg_counts.use_large_path_scan {
            // First-level reduce only fills `path_tag_wgs` slots; `reduced` is aligned up
            // to a multiple of 256 for reduce2. Zero padding so unused slots are identity.
            recorder
                .scheme()
                .clear_parcel(&pipeline.scratch.reduced, 0, 0)
                .expect("clear path_reduced before large pathtag scan");
        }
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.pathtag_reduce,
            STAGE_PATHTAG_REDUCE,
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.reduced.as_binding(),
            ],
        );
        if wg_counts.use_large_path_scan {
            dispatch_stage(
                recorder,
                indirect_buf,
                shaders.pathtag_reduce2,
                STAGE_PATHTAG_REDUCE2,
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
                &[
                    pipeline.scratch.reduced.as_binding(),
                    pipeline.scratch.reduced2.as_binding(),
                    pipeline.scratch.reduced_scan.as_binding(),
                ],
            );
            dispatch_stage(
                recorder,
                indirect_buf,
                shaders.pathtag_scan_large,
                STAGE_PATHTAG_SCAN_LARGE,
                &[
                    pipeline.config.as_binding(),
                    pipeline.scene.as_binding(),
                    pipeline.scratch.reduced_scan.as_binding(),
                    pipeline.scratch.tagmonoid.as_binding(),
                ],
            );
        } else {
            dispatch_stage(
                recorder,
                indirect_buf,
                shaders.pathtag_scan,
                STAGE_PATHTAG_SCAN,
                &[
                    pipeline.config.as_binding(),
                    pipeline.scene.as_binding(),
                    pipeline.scratch.reduced.as_binding(),
                    pipeline.scratch.tagmonoid.as_binding(),
                ],
            );
        }

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.bbox_clear,
            STAGE_BBOX_CLEAR,
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
        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.flatten,
            STAGE_FLATTEN,
            &flatten_bindings,
        );

        dispatch_stage(
            recorder,
            indirect_buf,
            shaders.draw_reduce,
            STAGE_DRAW_REDUCE,
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
            &[
                pipeline.config.as_binding(),
                pipeline.scene.as_binding(),
                pipeline.scratch.draw_bbox.as_binding(),
                pipeline.bump.as_binding(),
                pipeline.scratch.path.as_binding(),
                pipeline.stable.tile.as_binding(),
            ],
        );

        // path_count_setup_scheme writes the workgroup count for path_count into
        // the per-stage DispatchShape buffer.
        recorder.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            &[
                pipeline.bump.as_binding(),
                GpuBinding::Parcel(indirect_buf.unit(STAGE_PATH_COUNT as usize)),
            ],
        );

        // Indirect dispatch driven by the GPU-written path_count shape buffer.
        recorder.dispatch_shape(
            shaders.path_count,
            indirect_buf.unit(STAGE_PATH_COUNT as usize),
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

        // path_tiling_setup_scheme writes the workgroup count for path_tiling
        // into the per-stage DispatchShape buffer.
        recorder.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            &[
                pipeline.bump.as_binding(),
                GpuBinding::Parcel(indirect_buf.unit(STAGE_PATH_TILING as usize)),
                pipeline.stable.ptcl.as_binding(),
            ],
        );

        // Indirect dispatch driven by the GPU-written path_tiling shape buffer.
        recorder.dispatch_shape(
            shaders.path_tiling,
            indirect_buf.unit(STAGE_PATH_TILING as usize),
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
        render_output: RenderOutput<'_>,
        recorder: &mut SchemeRecorder<'_>,
    ) {
        let fine_wg_count = self.fine_wg_count.take().expect("fine_wg_count");
        let width_in_tiles = fine_wg_count.0;
        let height_in_tiles = fine_wg_count.1;

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

        // Obtain a persistent mask LUT buffer for MSAA modes. The LUT is static
        // (does not depend on scene content), so it is acquired once from the
        // retained pool with init data and reused across frames — no per-frame
        // WriteBuffer / staging copy on the fine command list.
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
                let buf = recorder
                    .persistent
                    .retained_pool
                    .acquire_buffer(
                        lut_data.len() as u64,
                        BufferKind::Scattered,
                        Some(4),
                        BufferFlags::empty(),
                        Some(&lut_data),
                    )
                    .expect("mask_lut retained acquire failed");
                match self.aa_config {
                    AaConfig::Msaa16 => recorder.persistent.stable_mask_lut_msaa16 = Some(buf),
                    AaConfig::Msaa8 => recorder.persistent.stable_mask_lut_msaa8 = Some(buf),
                    _ => unreachable!(),
                }
            }
        }

        let width_px = render_output.width(pipeline);
        let height_px = render_output.height(pipeline);
        if !layer_filter_effects.is_empty() && width_px > 0 && height_px > 0 {
            if let Some(fs) = shaders.filter_pass {
                let wg = (width_px.div_ceil(16), height_px.div_ceil(16), 1);
                let u_clear = FilterUniform::clear_transparent(width_px, height_px);
                let clear_src = pipeline.out_image.as_ref().unwrap_or(&pipeline.filter_layers[0]);
                for fl in &pipeline.filter_layers {
                    filter_dispatch(recorder, fs, &u_clear, wg, clear_src, fl);
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
            render_output.fine_binding(),
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
                fine_resources.push(GpuBinding::Buf(lut));
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

/// Acquire a sticky filter-uniform deed from the retained pool (init upload at alloc).
fn acquire_retained_filter_uniform(recorder: &mut SchemeRecorder<'_>, uniform: &FilterUniform) -> Buffer {
    let bytes = bytemuck::bytes_of(uniform);
    recorder
        .persistent
        .retained_pool
        .acquire_buffer(
            bytes.len() as u64,
            BufferKind::Scattered,
            Some(size_of::<FilterUniform>() as u32),
            BufferFlags::empty(),
            Some(bytes),
        )
        .expect("filter_uniform retained acquire failed")
}

/// Content-keyed filter-uniform buffer: reuse on hit; release old deed and reacquire on miss.
fn take_or_refresh_filter_uniform(
    recorder: &mut SchemeRecorder<'_>,
    cached: Option<(FilterUniform, Buffer)>,
    uniform: &FilterUniform,
) -> Buffer {
    match cached {
        Some((ref val, buf)) if val == uniform => buf,
        Some((_, old_buf)) => {
            recorder
                .persistent
                .retained_pool
                .release_buffer(recorder.context(), old_buf);
            acquire_retained_filter_uniform(recorder, uniform)
        }
        None => acquire_retained_filter_uniform(recorder, uniform),
    }
}

/// Destination binding for a filter pass (texture UAV or present lease).
enum FilterDst<'a> {
    Tex(&'a Texture),
    Output(RenderOutput<'a>),
}

impl<'a> FilterDst<'a> {
    fn binding(self) -> GpuBinding<'a> {
        match self {
            Self::Tex(tex) => GpuBinding::Tex(tex),
            Self::Output(out) => out.filter_dst_binding(),
        }
    }
}

/// Shared filter dispatch: slot cache, uniform buffer, sampler, and record.
fn filter_dispatch_impl(
    recorder: &mut SchemeRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    sampled_src: &Texture,
    uav_src: &Texture,
    dst: FilterDst<'_>,
) {
    let slot = recorder.filter_dispatch_slot;
    recorder.filter_dispatch_slot += 1;

    let cached = recorder
        .persistent
        .cached_filter_uniforms
        .get_mut(slot)
        .and_then(|e| e.take());

    let buf = take_or_refresh_filter_uniform(recorder, cached, uniform);

    let persistent = &*recorder.persistent;
    let bindings = [
        GpuBinding::Buf(&buf),
        GpuBinding::Tex(sampled_src), // src_sampled (Interpolated — SRV)
        GpuBinding::Tex(uav_src),     // src (DirectSpatial — UAV)
        dst.binding(),
        GpuBinding::Sampler(
            persistent
                .linear_clamp_sampler
                .as_ref()
                .expect("linear_clamp_sampler must be initialised before filter pass"),
        ),
    ];
    SchemeRecorder::record_dispatch(recorder.scheme, recorder.shaders, shader, wg, &bindings, &[]);

    // Restore buffer to persistent cache. Do NOT defer — sticky retained deed.
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

fn filter_dispatch_to_output(
    recorder: &mut SchemeRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    src: &Texture,
    dst: RenderOutput<'_>,
) {
    filter_dispatch_impl(recorder, shader, uniform, wg, src, src, FilterDst::Output(dst));
}

fn filter_dispatch(
    recorder: &mut SchemeRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    src: &Texture,
    dst: &Texture,
) {
    filter_dispatch_impl(recorder, shader, uniform, wg, src, src, FilterDst::Tex(dst));
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
    filter_dispatch_impl(recorder, shader, uniform, wg, sampled_src, uav_src, FilterDst::Tex(dst));
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
    render_output: RenderOutput<'_>,
) {
    let width = pipeline.frame_width;
    let height = pipeline.frame_height;

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
        filter_dispatch_to_output(recorder, shader, &u_comp, wg, ft, render_output);
    }
    for effect in layer_filter_effects.iter().filter(|e| !e.is_nested) {
        let idx = (effect.layer_index as usize).min(3);
        let ft = &filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch_to_output(recorder, shader, &u_comp, wg, ft, render_output);
    }

    recorder.defer_texture(scratch);
}
