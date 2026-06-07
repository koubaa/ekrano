// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Load rendering shaders.

use crate::ShaderId;
use crate::goldy_renderer::GoldyRenderer;
use crate::{
    Error,
    resource_proxy::{BindType, ImageFormat},
};

// Shaders for the full pipeline
pub struct FullShaders {
    /// Initializes the indirect dispatch buffer.
    pub pipeline_setup: ShaderId,
    pub pathtag_reduce: ShaderId,
    pub pathtag_reduce2: ShaderId,
    pub pathtag_scan1: ShaderId,
    pub pathtag_scan: ShaderId,
    pub pathtag_scan_large: ShaderId,
    pub bbox_clear: ShaderId,
    pub flatten: ShaderId,
    pub draw_reduce: ShaderId,
    pub draw_leaf: ShaderId,
    pub clip_reduce: ShaderId,
    pub clip_leaf: ShaderId,
    pub binning: ShaderId,
    pub tile_alloc: ShaderId,
    pub backdrop: ShaderId,
    pub path_count_setup: ShaderId,
    pub path_count: ShaderId,
    pub coarse: ShaderId,
    pub path_tiling_setup: ShaderId,
    pub path_tiling: ShaderId,
    pub fine_area: Option<ShaderId>,
    pub fine_msaa8: Option<ShaderId>,
    pub fine_msaa16: Option<ShaderId>,
    /// Full-frame filter chain after fine raster (optional).
    pub filter_pass: Option<ShaderId>,
}

impl FullShaders {
    /// Create a placeholder with zeroed shader IDs; used during `GoldyRenderer` construction
    /// before shaders are compiled.
    pub(crate) fn empty() -> Self {
        Self {
            pipeline_setup: ShaderId(0),
            pathtag_reduce: ShaderId(0),
            pathtag_reduce2: ShaderId(0),
            pathtag_scan1: ShaderId(0),
            pathtag_scan: ShaderId(0),
            pathtag_scan_large: ShaderId(0),
            bbox_clear: ShaderId(0),
            flatten: ShaderId(0),
            draw_reduce: ShaderId(0),
            draw_leaf: ShaderId(0),
            clip_reduce: ShaderId(0),
            clip_leaf: ShaderId(0),
            binning: ShaderId(0),
            tile_alloc: ShaderId(0),
            backdrop: ShaderId(0),
            path_count_setup: ShaderId(0),
            path_count: ShaderId(0),
            coarse: ShaderId(0),
            path_tiling_setup: ShaderId(0),
            path_tiling: ShaderId(0),
            fine_area: None,
            fine_msaa8: None,
            fine_msaa16: None,
            filter_pass: None,
        }
    }
}

pub(crate) fn goldy_full_shaders(renderer: &mut GoldyRenderer) -> Result<FullShaders, Error> {
    use BindType::*;

    let search_path = ekrano_shaders::slang::slang_search_path();
    let search_path_str = search_path.to_string_lossy();
    let search_paths = [search_path_str.as_ref()];

    let sw_opt = goldy::OptimizationLevel::Default;

    let pipeline_setup = renderer.add_compute_shader(
        "pipeline_setup",
        ekrano_shaders::slang::PIPELINE_SETUP,
        &[BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_reduce = renderer.add_compute_shader(
        "pathtag_reduce",
        ekrano_shaders::slang::PATHTAG_REDUCE,
        &[BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_reduce2 = renderer.add_compute_shader(
        "pathtag_reduce2",
        ekrano_shaders::slang::PATHTAG_REDUCE2,
        &[BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_scan1 = renderer.add_compute_shader(
        "pathtag_scan1",
        ekrano_shaders::slang::PATHTAG_SCAN1,
        &[BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_scan = renderer.add_compute_shader(
        "pathtag_scan_small",
        ekrano_shaders::slang::PATHTAG_SCAN_SMALL,
        &[BufReadOnly, BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_scan_large = renderer.add_compute_shader(
        "pathtag_scan_large",
        ekrano_shaders::slang::PATHTAG_SCAN_SMALL,
        &[BufReadOnly, BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let bbox_clear = renderer.add_compute_shader(
        "bbox_clear",
        ekrano_shaders::slang::BBOX_CLEAR,
        &[BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let flatten = renderer.add_compute_shader(
        "flatten",
        ekrano_shaders::slang::FLATTEN,
        &[
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let draw_reduce = renderer.add_compute_shader(
        "draw_reduce",
        ekrano_shaders::slang::DRAW_REDUCE,
        &[BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let draw_leaf = renderer.add_compute_shader(
        "draw_leaf",
        ekrano_shaders::slang::DRAW_LEAF,
        &[
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let clip_reduce = renderer.add_compute_shader(
        "clip_reduce",
        ekrano_shaders::slang::CLIP_REDUCE,
        &[BufReadOnly, BufReadOnly, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let clip_leaf = renderer.add_compute_shader(
        "clip_leaf",
        ekrano_shaders::slang::CLIP_LEAF,
        &[
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let binning = renderer.add_compute_shader(
        "binning",
        ekrano_shaders::slang::BINNING,
        &[
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let tile_alloc = renderer.add_compute_shader(
        "tile_alloc",
        ekrano_shaders::slang::TILE_ALLOC,
        &[
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let path_count_setup = renderer.add_compute_shader(
        "path_count_setup",
        ekrano_shaders::slang::PATH_COUNT_SETUP,
        &[Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let path_count = renderer.add_compute_shader(
        "path_count",
        ekrano_shaders::slang::PATH_COUNT,
        &[
            BufReadOnly,
            Buffer,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let backdrop = renderer.add_compute_shader(
        "backdrop_dyn",
        ekrano_shaders::slang::BACKDROP_DYN,
        &[BufReadOnly, Buffer, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let coarse = renderer.add_compute_shader_with_options(
        "coarse",
        ekrano_shaders::slang::COARSE,
        &[
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ],
        &search_paths,
        &[],
        sw_opt,
    )?;
    let path_tiling_setup = renderer.add_compute_shader(
        "path_tiling_setup",
        ekrano_shaders::slang::PATH_TILING_SETUP,
        &[Buffer, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let path_tiling = renderer.add_compute_shader(
        "path_tiling",
        ekrano_shaders::slang::PATH_TILING,
        &[
            Buffer,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
        ],
        &search_paths,
        &[],
    )?;
    let fine_resources = [
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8), // mask_atlas
        Image(ImageFormat::Rgba8),
        Image(ImageFormat::Rgba8),
        Image(ImageFormat::Rgba8),
        Image(ImageFormat::Rgba8),
        Sampler, // linear_clamp
        Sampler, // nearest_clamp
    ];
    let fine_msaa_resources = [
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8), // mask_atlas
        BufReadOnly,                   // mask_lut
        Image(ImageFormat::Rgba8),
        Image(ImageFormat::Rgba8),
        Image(ImageFormat::Rgba8),
        Image(ImageFormat::Rgba8),
        Sampler, // linear_clamp
        Sampler, // nearest_clamp
    ];
    let fine_area = Some(renderer.add_compute_shader_with_options(
        "fine_area",
        ekrano_shaders::slang::FINE,
        &fine_resources,
        &search_paths,
        &[],
        sw_opt,
    )?);
    let fine_msaa8 = renderer
        .add_compute_shader_with_options(
            "fine_msaa8",
            ekrano_shaders::slang::FINE,
            &fine_msaa_resources,
            &search_paths,
            &[("msaa", "1"), ("msaa8", "1")],
            sw_opt,
        )
        .ok();
    let fine_msaa16 = renderer
        .add_compute_shader_with_options(
            "fine_msaa16",
            ekrano_shaders::slang::FINE,
            &fine_msaa_resources,
            &search_paths,
            &[("msaa", "1"), ("msaa16", "1")],
            sw_opt,
        )
        .ok();

    let filter_pass = match renderer.add_compute_shader(
        "filter_pass",
        ekrano_shaders::slang::FILTER_PASS,
        &[
            BufReadOnly,                   // uniforms_buf (BufRO<FilterUniform>)
            ImageRead(ImageFormat::Rgba8), // src_sampled (Interpolated<float4>)
            Image(ImageFormat::Rgba8),     // src (DirectSpatial<float4>)
            Image(ImageFormat::Rgba8),     // dst (DirectSpatial<float4>)
            Sampler,                       // linear_clamp (Filter)
        ],
        &search_paths,
        &[],
    ) {
        Ok(id) => Some(id),
        Err(e) => {
            log::error!("filter_pass shader compilation failed: {e}");
            None
        }
    };

    Ok(FullShaders {
        pipeline_setup,
        pathtag_reduce,
        pathtag_reduce2,
        pathtag_scan1,
        pathtag_scan,
        pathtag_scan_large,
        bbox_clear,
        flatten,
        draw_reduce,
        draw_leaf,
        clip_reduce,
        clip_leaf,
        binning,
        tile_alloc,
        path_count_setup,
        path_count,
        backdrop,
        coarse,
        path_tiling_setup,
        path_tiling,
        fine_area,
        fine_msaa8,
        fine_msaa16,
        filter_pass,
    })
}
