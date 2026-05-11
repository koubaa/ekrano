// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Take an encoded scene and create a graph to render it

use crate::goldy_renderer::FrameRecorder;
use crate::resource_proxy::{BufferProxy, ImageFormat, ImageProxy, ResourceProxy};
use crate::shaders::FullShaders;
use crate::{AaConfig, RenderParams};

use std::mem::size_of;

use ekrano_encoding::{
    BumpAllocators, Encoding, FilterPrimitive, FilterUniform, Images, IndirectCount, Ramps,
    WorkgroupCountsGpu, WorkgroupSize, make_mask_lut, make_mask_lut_16,
};
use peniko::color::{PremulColor, Srgb};

use ekrano_encoding::{
    STAGE_BACKDROP, STAGE_BBOX_CLEAR, STAGE_BINNING, STAGE_CLIP_LEAF, STAGE_CLIP_REDUCE,
    STAGE_COARSE, STAGE_DRAW_LEAF, STAGE_DRAW_REDUCE, STAGE_FLATTEN, STAGE_PATH_COUNT,
    STAGE_PATH_TILING, STAGE_PATHTAG_REDUCE, STAGE_PATHTAG_REDUCE2, STAGE_PATHTAG_SCAN,
    STAGE_PATHTAG_SCAN_LARGE, STAGE_PATHTAG_SCAN1, STAGE_TILE_ALLOC,
};
use goldy::types::BufferFlags;

/// State for a render in progress.
pub struct Render {
    fine_wg_count: Option<WorkgroupSize>,
    fine_resources: Option<FineResources>,
    mask_buf: Option<ResourceProxy>,

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

/// Resources produced by pipeline, needed for fine rasterization.
struct FineResources {
    aa_config: AaConfig,
    /// When Some (Goldy indirect path), fine stage uses `dispatch_indirect`.
    indirect_buf: Option<BufferProxy>,
    config_buf: ResourceProxy,
    bump_buf: ResourceProxy,
    tile_buf: ResourceProxy,
    segments_buf: ResourceProxy,
    ptcl_buf: ResourceProxy,
    gradient_image: ResourceProxy,
    info_bin_data_buf: ResourceProxy,
    image_atlas: ResourceProxy,
    /// R8-equivalent RGBA (`.r` channel) full-frame mask; 1×1 white when unused.
    mask_atlas: ResourceProxy,
    blend_spill_buf: ResourceProxy,

    out_image: ImageProxy,
    /// Premultiplied snapshots for up to four isolated filter layers (see `fine.slang`).
    filter_layers: [ImageProxy; 4],
}

/// A collection of internal buffers that are used for debug visualization when the
/// `debug_layers` feature is enabled. The contents of these buffers remain GPU resident
/// and must be freed directly by the caller.
///
/// Some of these buffers are also scheduled for a download to allow their contents to be
/// processed for CPU-side validation. These buffers are documented as such.
#[cfg(feature = "debug_layers")]
pub struct CapturedBuffers {
    pub sizes: ekrano_encoding::BufferSizes,

    /// Buffers that remain GPU-only
    pub path_bboxes: BufferProxy,

    /// Buffers scheduled for download
    pub lines: BufferProxy,
}

#[cfg(feature = "debug_layers")]
impl CapturedBuffers {
    pub(crate) fn release_buffers(self, recorder: &mut FrameRecorder<'_>) {
        recorder.free_buffer(self.path_bboxes);
        recorder.free_buffer(self.lines);
    }
}

/// Max flatten workgroups per queue submit. Large single dispatches can exceed the
/// Windows ~2s GPU timeout (TDR) on stressed dashed paths.
const MAX_FLATTEN_WG_PER_SUBMIT: u32 = 8;
/// Must match `FLATTEN_WG` in `ekrano_encoding` (threads per flatten workgroup).
const FLATTEN_THREADS_PER_GROUP: u32 = 256;

fn dispatch_stage(
    recorder: &mut FrameRecorder<'_>,
    use_indirect: bool,
    indirect_buf: Option<BufferProxy>,
    shader: crate::ShaderId,
    stage: u32,
    wg: WorkgroupSize,
    stride: u64,
    resources: impl IntoIterator<Item = impl Into<ResourceProxy>>,
) {
    let r: Vec<_> = resources.into_iter().map(|x| x.into()).collect();
    if use_indirect {
        recorder.dispatch_indirect(shader, indirect_buf.unwrap(), stage as u64 * stride, r);
    } else {
        recorder.dispatch(shader, wg, r);
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
            fine_resources: None,
            mask_buf: None,
            #[cfg(feature = "debug_layers")]
            captured_buffers: None,
        }
    }

    /// Execute the coarse rasterization phase.
    ///
    /// `packed`, `ramps`, and `images` must come from a single
    /// `Resolver::resolve` call on `encoding` (performed by the caller, in
    /// [`GoldyRenderer::run_frame`]). Accepting pre-resolved data avoids a
    /// second resolve that would duplicate scene packing work.
    ///
    /// The `robust` flag enables GPU bump-allocator readback so the next frame
    /// can grow buffers when an overflow is detected.
    pub(crate) fn run_coarse(
        &mut self,
        encoding: &Encoding,
        mut packed: Vec<u8>,
        ramps: Ramps<'_>,
        images: Images<'_>,
        shaders: &FullShaders,
        params: &RenderParams,
        robust: bool,
        config: &ekrano_encoding::RenderConfig,
        recorder: &mut FrameRecorder<'_>,
    ) {
        if packed.is_empty() {
            packed.resize(size_of::<u32>(), u8::MAX);
        }
        let gradient_image = if ramps.height == 0 {
            ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramps.data);
            ResourceProxy::Image(recorder.upload_image(
                ramps.width,
                ramps.height,
                ImageFormat::Rgba8,
                data,
            ))
        };
        let image_atlas = if images.images.is_empty() {
            ImageProxy::new(1, 1, ImageFormat::Rgba8)
        } else {
            ImageProxy::new(images.width, images.height, ImageFormat::Rgba8)
        };
        for image in images.images {
            recorder.write_image(image_atlas, image.1, image.2, image.0.clone());
        }
        let mask_atlas = ResourceProxy::Image(match &encoding.coverage_mask {
            Some(m) => {
                let mut rgba = Vec::with_capacity(m.data.len() * 4);
                for &b in m.data.iter() {
                    rgba.extend_from_slice(&[b, b, b, 255]);
                }
                recorder.upload_image(m.width, m.height, ImageFormat::Rgba8, rgba)
            }
            None => recorder.upload_image(1, 1, ImageFormat::Rgba8, vec![255, 255, 255, 255]),
        });
        let mut cpu_config_owned = *config;
        if encoding.coverage_mask.is_some() {
            cpu_config_owned.gpu.mask_active = 1;
        }
        let cpu_config = &cpu_config_owned;
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
        // HACK: The coarse workgroup counts is the number of active bins.
        if (cpu_config.workgroup_counts.coarse.0
            * cpu_config.workgroup_counts.coarse.1
            * cpu_config.workgroup_counts.coarse.2)
            > 256
        {
            log::warn!(
                "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
                params.width,
                params.height
            );
        }
        let buffer_sizes = &cpu_config.buffer_sizes;
        let wg_counts = &cpu_config.workgroup_counts;

        let scene_buf = ResourceProxy::Buffer(recorder.upload_strided("ekrano.scene", 4, packed));
        let config_buf_proxy = recorder.upload_typed("ekrano.config", &cpu_config.gpu);
        let config_buf = ResourceProxy::Buffer(config_buf_proxy);
        const INDIRECT_STRIDE: u64 = size_of::<IndirectCount>() as u64;

        let use_indirect = shaders.pipeline_setup.is_some();
        let indirect_buf = if use_indirect {
            let wg_counts_gpu = WorkgroupCountsGpu::from(wg_counts);
            let wg_counts_buf_proxy = recorder.upload_typed("ekrano.wg_counts", &wg_counts_gpu);
            let wg_counts_buf = ResourceProxy::Buffer(wg_counts_buf_proxy);
            let indirect_buf = BufferProxy::with_stride(
                buffer_sizes.indirect_count.size_in_bytes().into(),
                "ekrano.indirect_dispatch",
                size_of::<IndirectCount>() as u32,
            );
            recorder.dispatch(
                shaders.pipeline_setup.unwrap(),
                (1, 1, 1),
                [wg_counts_buf, indirect_buf.into()],
            );
            recorder.free_buffer(wg_counts_buf_proxy);
            Some(indirect_buf)
        } else {
            None
        };
        let info_bin_data_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.bin_data.size_in_bytes() as u64,
            "ekrano.info_bin_data_buf",
            4,
        );
        let tile_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.tiles.size_in_bytes().into(),
            "ekrano.tile_buf",
            8,
        );
        let segments_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.segments.size_in_bytes().into(),
            "ekrano.segments_buf",
            24,
        );
        let ptcl_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.ptcl.size_in_bytes().into(),
            "ekrano.ptcl_buf",
            4,
        );
        let reduced_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.path_reduced.size_in_bytes().into(),
            "ekrano.reduced_buf",
            20,
        );
        let reduced2_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.path_reduced2.size_in_bytes().into(),
            "ekrano.reduced2_buf",
            20,
        );
        let reduced_scan_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.path_reduced_scan.size_in_bytes().into(),
            "ekrano.reduced_scan_buf",
            20,
        );
        let tagmonoid_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.path_monoids.size_in_bytes().into(),
            "ekrano.tagmonoid_buf",
            20,
        );
        let use_large_path_scan = wg_counts.use_large_path_scan && !shaders.pathtag_is_cpu;

        // TODO: really only need pathtag_wgs - 1
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.pathtag_reduce,
            STAGE_PATHTAG_REDUCE,
            wg_counts.path_reduce,
            INDIRECT_STRIDE,
            [config_buf, scene_buf, reduced_buf],
        );
        let mut pathtag_parent = reduced_buf;
        if use_indirect {
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_reduce2,
                STAGE_PATHTAG_REDUCE2,
                wg_counts.path_reduce2,
                INDIRECT_STRIDE,
                [reduced_buf, reduced2_buf],
            );
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_scan1,
                STAGE_PATHTAG_SCAN1,
                wg_counts.path_scan1,
                INDIRECT_STRIDE,
                [reduced_buf, reduced2_buf, reduced_scan_buf],
            );
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_scan,
                STAGE_PATHTAG_SCAN,
                wg_counts.path_scan,
                INDIRECT_STRIDE,
                [config_buf, scene_buf, reduced_buf, tagmonoid_buf],
            );
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.pathtag_scan_large,
                STAGE_PATHTAG_SCAN_LARGE,
                wg_counts.path_scan,
                INDIRECT_STRIDE,
                [config_buf, scene_buf, reduced_scan_buf, tagmonoid_buf],
            );
        } else if use_large_path_scan {
            recorder.dispatch(
                shaders.pathtag_reduce2,
                wg_counts.path_reduce2,
                [reduced_buf, reduced2_buf],
            );
            recorder.dispatch(
                shaders.pathtag_scan1,
                wg_counts.path_scan1,
                [reduced_buf, reduced2_buf, reduced_scan_buf],
            );
            pathtag_parent = reduced_scan_buf;
            recorder.dispatch(
                shaders.pathtag_scan_large,
                wg_counts.path_scan,
                [config_buf, scene_buf, pathtag_parent, tagmonoid_buf],
            );
        } else {
            recorder.dispatch(
                shaders.pathtag_scan,
                wg_counts.path_scan,
                [config_buf, scene_buf, pathtag_parent, tagmonoid_buf],
            );
        }
        recorder.free_resource(reduced_buf);
        recorder.free_resource(reduced2_buf);
        recorder.free_resource(reduced_scan_buf);

        let path_bbox_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            "ekrano.path_bbox_buf",
            24,
        );
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.bbox_clear,
            STAGE_BBOX_CLEAR,
            wg_counts.bbox_clear,
            INDIRECT_STRIDE,
            [config_buf, path_bbox_buf],
        );
        let bump_buf = BufferProxy::with_stride_and_flags(
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            "ekrano.bump_buf",
            size_of::<BumpAllocators>() as u32,
            BufferFlags::CPU_READABLE,
        );
        recorder.clear_all(bump_buf);
        let bump_buf = ResourceProxy::Buffer(bump_buf);
        let lines_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.lines.size_in_bytes().into(),
            "ekrano.lines_buf",
            24,
        );
        let flatten_bindings = [
            config_buf,
            scene_buf,
            tagmonoid_buf,
            path_bbox_buf,
            bump_buf,
            lines_buf,
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
                    flatten_bindings,
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
                flatten_bindings,
            );
        } else {
            recorder.dispatch_with_push_tail(
                shaders.flatten,
                wg_counts.flatten,
                flatten_bindings,
                &[0],
            );
        }
        let draw_reduced_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            "ekrano.draw_reduced_buf",
            16,
        );
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.draw_reduce,
            STAGE_DRAW_REDUCE,
            wg_counts.draw_reduce,
            INDIRECT_STRIDE,
            [config_buf, scene_buf, draw_reduced_buf],
        );
        let draw_monoid_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            "ekrano.draw_monoid_buf",
            16,
        );
        let clip_inp_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.clip_inps.size_in_bytes().into(),
            "ekrano.clip_inp_buf",
            8,
        );
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.draw_leaf,
            STAGE_DRAW_LEAF,
            wg_counts.draw_leaf,
            INDIRECT_STRIDE,
            [
                config_buf,
                scene_buf,
                draw_reduced_buf,
                path_bbox_buf,
                draw_monoid_buf,
                info_bin_data_buf,
                clip_inp_buf,
            ],
        );
        recorder.free_resource(draw_reduced_buf);
        let clip_el_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.clip_els.size_in_bytes().into(),
            "ekrano.clip_el_buf",
            32,
        );
        let clip_bic_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.clip_bics.size_in_bytes().into(),
            "ekrano.clip_bic_buf",
            8,
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
                [clip_inp_buf, path_bbox_buf, clip_bic_buf, clip_el_buf],
            );
        }
        let clip_bbox_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            "ekrano.clip_bbox_buf",
            16,
        );
        if use_indirect || wg_counts.clip_leaf.0 > 0 {
            dispatch_stage(
                recorder,
                use_indirect,
                indirect_buf,
                shaders.clip_leaf,
                STAGE_CLIP_LEAF,
                wg_counts.clip_leaf,
                INDIRECT_STRIDE,
                [
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                    draw_monoid_buf,
                    clip_bbox_buf,
                ],
            );
        }
        recorder.free_resource(clip_inp_buf);
        recorder.free_resource(clip_bic_buf);
        recorder.free_resource(clip_el_buf);
        let draw_bbox_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            "ekrano.draw_bbox_buf",
            16,
        );
        let bin_header_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.bin_headers.size_in_bytes().into(),
            "ekrano.bin_header_buf",
            8,
        );
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.binning,
            STAGE_BINNING,
            wg_counts.binning,
            INDIRECT_STRIDE,
            [
                config_buf,
                scene_buf,
                draw_monoid_buf,
                path_bbox_buf,
                clip_bbox_buf,
                draw_bbox_buf,
                bump_buf,
                info_bin_data_buf,
                bin_header_buf,
            ],
        );
        recorder.free_resource(draw_monoid_buf);
        recorder.free_resource(clip_bbox_buf);
        let path_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.paths.size_in_bytes().into(),
            "ekrano.path_buf",
            32,
        );
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.tile_alloc,
            STAGE_TILE_ALLOC,
            wg_counts.tile_alloc,
            INDIRECT_STRIDE,
            [
                config_buf,
                scene_buf,
                draw_bbox_buf,
                bump_buf,
                path_buf,
                tile_buf,
            ],
        );
        recorder.free_resource(draw_bbox_buf);
        recorder.free_resource(tagmonoid_buf);

        let path_indirect_buf = indirect_buf.unwrap_or_else(|| {
            BufferProxy::with_stride(
                size_of::<IndirectCount>() as u64,
                "ekrano.indirect_count",
                size_of::<IndirectCount>() as u32,
            )
        });
        recorder.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            [bump_buf, path_indirect_buf.into()],
        );
        let seg_counts_buf = ResourceProxy::new_buf_with_stride(
            buffer_sizes.seg_counts.size_in_bytes().into(),
            "ekrano.seg_counts_buf",
            8,
        );
        let path_count_offset = if use_indirect {
            STAGE_PATH_COUNT as u64 * INDIRECT_STRIDE
        } else {
            0
        };
        recorder.dispatch_indirect(
            shaders.path_count,
            path_indirect_buf,
            path_count_offset,
            [
                config_buf,
                bump_buf,
                lines_buf,
                path_buf,
                tile_buf,
                seg_counts_buf,
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
            [config_buf, bump_buf, path_buf, tile_buf],
        );
        dispatch_stage(
            recorder,
            use_indirect,
            indirect_buf,
            shaders.coarse,
            STAGE_COARSE,
            wg_counts.coarse,
            INDIRECT_STRIDE,
            [
                config_buf,
                scene_buf,
                draw_monoid_buf,
                bin_header_buf,
                info_bin_data_buf,
                path_buf,
                tile_buf,
                bump_buf,
                ptcl_buf,
            ],
        );
        recorder.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            [bump_buf, path_indirect_buf.into(), ptcl_buf],
        );
        let path_tiling_offset = if use_indirect {
            STAGE_PATH_TILING as u64 * INDIRECT_STRIDE
        } else {
            0
        };
        recorder.dispatch_indirect(
            shaders.path_tiling,
            path_indirect_buf,
            path_tiling_offset,
            [
                bump_buf,
                seg_counts_buf,
                lines_buf,
                path_buf,
                tile_buf,
                segments_buf,
            ],
        );
        if !use_indirect {
            recorder.free_buffer(path_indirect_buf);
        }
        recorder.free_resource(seg_counts_buf);
        recorder.free_resource(scene_buf);
        recorder.free_resource(draw_monoid_buf);
        recorder.free_resource(bin_header_buf);
        recorder.free_resource(path_buf);
        let out_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
        let filter_layers = [
            ImageProxy::new(params.width, params.height, ImageFormat::Rgba8),
            ImageProxy::new(params.width, params.height, ImageFormat::Rgba8),
            ImageProxy::new(params.width, params.height, ImageFormat::Rgba8),
            ImageProxy::new(params.width, params.height, ImageFormat::Rgba8),
        ];
        let blend_spill_buf = BufferProxy::with_stride(
            buffer_sizes.blend_spill.size_in_bytes().into(),
            "ekrano.blend_spill",
            size_of::<u32>() as u32,
        );
        self.fine_wg_count = Some(wg_counts.fine);
        self.fine_resources = Some(FineResources {
            aa_config: params.antialiasing_method,
            indirect_buf: if use_indirect {
                Some(path_indirect_buf)
            } else {
                None
            },
            config_buf,
            bump_buf,
            tile_buf,
            segments_buf,
            ptcl_buf,
            gradient_image,
            info_bin_data_buf,
            blend_spill_buf: ResourceProxy::Buffer(blend_spill_buf),
            image_atlas: ResourceProxy::Image(image_atlas),
            mask_atlas,
            out_image,
            filter_layers,
        });
        if robust {
            recorder.download(*bump_buf.as_buf().unwrap());
        }
        recorder.free_resource(bump_buf);

        #[cfg(feature = "debug_layers")]
        {
            if robust {
                let path_bboxes = *path_bbox_buf.as_buf().unwrap();
                let lines = *lines_buf.as_buf().unwrap();
                recorder.download(lines);

                self.captured_buffers = Some(CapturedBuffers {
                    sizes: cpu_config.buffer_sizes,
                    path_bboxes,
                    lines,
                });
            } else {
                recorder.free_resource(path_bbox_buf);
                recorder.free_resource(lines_buf);
            }
        }
        #[cfg(not(feature = "debug_layers"))]
        {
            recorder.free_resource(path_bbox_buf);
            recorder.free_resource(lines_buf);
        }
    }

    /// Run fine rasterization assuming the coarse phase succeeded.
    ///
    /// When `height_in_tiles > 1`, splits the fine dispatch into per-row
    /// dispatches to avoid GPU TDR on large workloads. Each row gets a
    /// separate config buffer with `tile_y_offset` set, so the fine shader
    /// knows which tile row to process.
    /// `encoding` is used to clear per-layer filter textures before fine when filters are present.
    pub(crate) fn record_fine(
        &mut self,
        encoding: &Encoding,
        shaders: &FullShaders,
        recorder: &mut FrameRecorder<'_>,
    ) {
        let fine_wg_count = self.fine_wg_count.take().unwrap();
        let mut fine = self.fine_resources.take().unwrap();
        let width_in_tiles = fine_wg_count.0;
        let height_in_tiles = fine_wg_count.1;

        if let Some(indirect_buf) = fine.indirect_buf.take() {
            recorder.free_buffer(indirect_buf);
        }

        let base_resources = [
            fine.config_buf,
            fine.segments_buf,
            fine.ptcl_buf,
            fine.info_bin_data_buf,
            fine.blend_spill_buf,
            ResourceProxy::Image(fine.out_image),
            fine.gradient_image,
            fine.image_atlas,
            fine.mask_atlas,
        ];

        let shader = match fine.aa_config {
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

        let msaa_mask_buf = match fine.aa_config {
            AaConfig::Msaa16 | AaConfig::Msaa8 => {
                if self.mask_buf.is_none() {
                    let mask_lut = match fine.aa_config {
                        AaConfig::Msaa16 => make_mask_lut_16(),
                        AaConfig::Msaa8 => make_mask_lut(),
                        _ => unreachable!(),
                    };
                    let buf = recorder.upload_strided("ekrano.mask_lut", 4, mask_lut);
                    self.mask_buf = Some(buf.into());
                }
                self.mask_buf
            }
            _ => None,
        };

        let mut fine_resources: Vec<ResourceProxy> = base_resources.to_vec();
        if let Some(mask) = msaa_mask_buf {
            fine_resources.push(mask);
        }
        for fl in &fine.filter_layers {
            fine_resources.push(ResourceProxy::Image(*fl));
        }

        let width_px = fine.out_image.width;
        let height_px = fine.out_image.height;
        if !encoding.layer_filter_effects.is_empty() && width_px > 0 && height_px > 0 {
            if let Some(fs) = shaders.filter_pass {
                let wg = (width_px.div_ceil(16), height_px.div_ceil(16), 1);
                let u_clear = FilterUniform::clear_transparent(width_px, height_px);
                for fl in &fine.filter_layers {
                    filter_dispatch(recorder, fs, &u_clear, wg, fine.out_image, *fl);
                }
            } else {
                log::warn!("filter_pass shader unavailable; cannot clear filter layer textures");
            }
        }

        recorder.dispatch(
            shader,
            (width_in_tiles, height_in_tiles, 1),
            fine_resources.iter().cloned(),
        );

        recorder.free_resource(fine.config_buf);
        recorder.free_resource(fine.tile_buf);
        recorder.free_resource(fine.segments_buf);
        recorder.free_resource(fine.ptcl_buf);
        recorder.free_resource(fine.gradient_image);
        recorder.free_resource(fine.image_atlas);
        recorder.free_resource(fine.mask_atlas);
        recorder.free_resource(fine.info_bin_data_buf);
        recorder.free_resource(fine.blend_spill_buf);
        if let Some(mask_buf) = self.mask_buf.take() {
            recorder.free_resource(mask_buf);
        }
    }

    /// Get the output image.
    ///
    /// This is going away, as the caller will add the output image to the bind
    /// map.
    pub fn out_image(&self) -> ImageProxy {
        self.fine_resources.as_ref().unwrap().out_image
    }

    /// Per-layer filter snapshot textures (same size as [`Self::out_image`]).
    ///
    /// Only meaningful after coarse rasterization and before fine rasterization completes;
    /// the pipeline clears these proxies once the fine pass runs.
    pub fn filter_layer_textures(&self) -> [ImageProxy; 4] {
        self.fine_resources.as_ref().unwrap().filter_layers
    }

    pub fn bump_buf(&self) -> BufferProxy {
        *self
            .fine_resources
            .as_ref()
            .unwrap()
            .bump_buf
            .as_buf()
            .unwrap()
    }

    pub fn tile_buf(&self) -> BufferProxy {
        *self
            .fine_resources
            .as_ref()
            .unwrap()
            .tile_buf
            .as_buf()
            .unwrap()
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
    recorder: &mut FrameRecorder<'_>,
    shader: crate::ShaderId,
    uniform: &FilterUniform,
    wg: (u32, u32, u32),
    src: ImageProxy,
    dst: ImageProxy,
) {
    let buf = recorder.upload_typed("ekrano.filter_uniform", uniform);
    recorder.dispatch(
        shader,
        wg,
        [
            ResourceProxy::Buffer(buf),
            ResourceProxy::Image(src),
            ResourceProxy::Image(dst),
        ],
    );
    recorder.free_buffer(buf);
}

/// Per-layer filter chain for [`Encoding::layer_filter_effects`] after fine rasterization.
///
/// Each [`ekrano_encoding::LayerFilterEffect`] runs its [`FilterPrimitive`] on the corresponding
/// entry in `filter_layers` (premultiplied snapshot written during fine), then composites onto
/// `out_image` using the layer blend mode. Uses a scratch buffer the size of the target.
pub(crate) fn record_filter_effects(
    encoding: &Encoding,
    shaders: &FullShaders,
    recorder: &mut FrameRecorder<'_>,
    width: u32,
    height: u32,
    filter_layers: &[ImageProxy; 4],
    out_image: ImageProxy,
) {
    let free_filter_images = |recorder: &mut FrameRecorder<'_>| {
        for fl in filter_layers {
            recorder.free_image(*fl);
        }
    };

    if width == 0 || height == 0 {
        return;
    }
    if encoding.layer_filter_effects.is_empty() {
        free_filter_images(recorder);
        return;
    }
    let Some(shader) = shaders.filter_pass else {
        log::warn!("filter_pass shader unavailable; skipping layer_filter_effects");
        free_filter_images(recorder);
        return;
    };
    let wg = (width.div_ceil(16), height.div_ceil(16), 1);
    let scratch = ImageProxy::new(width, height, ImageFormat::Rgba8);

    for effect in &encoding.layer_filter_effects {
        let idx = (effect.layer_index as usize).min(3);
        let ft = filter_layers[idx];
        match &effect.primitive {
            FilterPrimitive::GaussianBlur { std_dev, edge_mode } => {
                let u_h = FilterUniform::gaussian_blur(width, height, true, *std_dev, *edge_mode);
                filter_dispatch(recorder, shader, &u_h, wg, ft, scratch);
                let u_v = FilterUniform::gaussian_blur(width, height, false, *std_dev, *edge_mode);
                filter_dispatch(recorder, shader, &u_v, wg, scratch, ft);
            }
            FilterPrimitive::Offset { dx, dy } => {
                let edge = ekrano_encoding::FilterEdgeMode::Duplicate;
                let u = FilterUniform::offset(width, height, *dx, *dy, edge);
                filter_dispatch(recorder, shader, &u, wg, ft, scratch);
                let u_copy = FilterUniform::copy(width, height);
                filter_dispatch(recorder, shader, &u_copy, wg, scratch, ft);
            }
            FilterPrimitive::Flood { color, clip_rect } => {
                let u = FilterUniform::flood(width, height, premul_srgb_u32(*color), *clip_rect);
                filter_dispatch(recorder, shader, &u, wg, ft, scratch);
                let u_copy = FilterUniform::copy(width, height);
                filter_dispatch(recorder, shader, &u_copy, wg, scratch, ft);
            }
            FilterPrimitive::DropShadow {
                dx,
                dy,
                std_dev,
                color,
                edge_mode,
            } => {
                let u = if effect.is_nested {
                    let inner_idx = (effect.layer_index as usize).saturating_sub(1).min(3);
                    let inner_ft = filter_layers[inner_idx];
                    let u = FilterUniform::drop_shadow_nested(
                        width,
                        height,
                        *dx,
                        *dy,
                        *std_dev,
                        premul_srgb_u32(*color),
                        *edge_mode,
                    );
                    filter_dispatch(recorder, shader, &u, wg, inner_ft, scratch);
                    let u_copy = FilterUniform::copy(width, height);
                    filter_dispatch(recorder, shader, &u_copy, wg, scratch, ft);
                    continue;
                } else {
                    FilterUniform::drop_shadow(
                        width,
                        height,
                        *dx,
                        *dy,
                        *std_dev,
                        premul_srgb_u32(*color),
                        *edge_mode,
                    )
                };
                filter_dispatch(recorder, shader, &u, wg, ft, scratch);
                let u_copy = FilterUniform::copy(width, height);
                filter_dispatch(recorder, shader, &u_copy, wg, scratch, ft);
            }
        }
    }

    for effect in encoding.layer_filter_effects.iter().filter(|e| e.is_nested) {
        let idx = (effect.layer_index as usize).min(3);
        let ft = filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch(recorder, shader, &u_comp, wg, ft, out_image);
    }
    for effect in encoding
        .layer_filter_effects
        .iter()
        .filter(|e| !e.is_nested)
    {
        let idx = (effect.layer_index as usize).min(3);
        let ft = filter_layers[idx];
        let u_comp = FilterUniform::composite_filtered_layer(width, height, effect.layer_blend);
        filter_dispatch(recorder, shader, &u_comp, wg, ft, out_image);
    }

    recorder.free_image(scratch);
    free_filter_images(recorder);
}
