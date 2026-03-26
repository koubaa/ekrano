// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Load rendering shaders.

use crate::ShaderId;
use crate::goldy_engine::GoldyEngine;
use crate::{
    Error,
    recording::{BindType, ImageFormat},
};

// Shaders for the full pipeline
pub struct FullShaders {
    /// Present for indirect dispatch.
    pub pipeline_setup: Option<ShaderId>,
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
    // 2-level dispatch works for CPU pathtag scan even for large
    // inputs, 3-level is not yet implemented.
    pub pathtag_is_cpu: bool,
}

pub(crate) fn goldy_full_shaders(
    device: &goldy::Device,
    engine: &mut GoldyEngine,
) -> Result<FullShaders, Error> {
    use BindType::*;

    let search_path = ekrano_shaders::slang::slang_search_path();
    let search_path_str = search_path.to_string_lossy();
    let search_paths = [search_path_str.as_ref()];

    let pipeline_setup = engine.add_compute_shader(
        device,
        "pipeline_setup",
        ekrano_shaders::slang::PIPELINE_SETUP,
        &[BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_reduce = engine.add_compute_shader(
        device,
        "pathtag_reduce",
        ekrano_shaders::slang::PATHTAG_REDUCE,
        &[Uniform, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_reduce2 = engine.add_compute_shader(
        device,
        "pathtag_reduce2",
        ekrano_shaders::slang::PATHTAG_REDUCE2,
        &[BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_scan1 = engine.add_compute_shader(
        device,
        "pathtag_scan1",
        ekrano_shaders::slang::PATHTAG_SCAN1,
        &[BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_scan = engine.add_compute_shader(
        device,
        "pathtag_scan_small",
        ekrano_shaders::slang::PATHTAG_SCAN_SMALL,
        &[Uniform, BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let pathtag_scan_large = engine.add_compute_shader(
        device,
        "pathtag_scan_large",
        ekrano_shaders::slang::PATHTAG_SCAN_SMALL,
        &[Uniform, BufReadOnly, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let bbox_clear = engine.add_compute_shader(
        device,
        "bbox_clear",
        ekrano_shaders::slang::BBOX_CLEAR,
        &[Uniform, Buffer],
        &search_paths,
        &[],
    )?;
    let flatten = engine.add_compute_shader(
        device,
        "flatten",
        ekrano_shaders::slang::FLATTEN,
        &[Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let draw_reduce = engine.add_compute_shader(
        device,
        "draw_reduce",
        ekrano_shaders::slang::DRAW_REDUCE,
        &[Uniform, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let draw_leaf = engine.add_compute_shader(
        device,
        "draw_leaf",
        ekrano_shaders::slang::DRAW_LEAF,
        &[
            Uniform,
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
    let clip_reduce = engine.add_compute_shader(
        device,
        "clip_reduce",
        ekrano_shaders::slang::CLIP_REDUCE,
        &[BufReadOnly, BufReadOnly, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let clip_leaf = engine.add_compute_shader(
        device,
        "clip_leaf",
        ekrano_shaders::slang::CLIP_LEAF,
        &[
            Uniform,
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
    let binning = engine.add_compute_shader(
        device,
        "binning",
        ekrano_shaders::slang::BINNING,
        &[
            Uniform,
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
    let tile_alloc = engine.add_compute_shader(
        device,
        "tile_alloc",
        ekrano_shaders::slang::TILE_ALLOC,
        &[Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let path_count_setup = engine.add_compute_shader(
        device,
        "path_count_setup",
        ekrano_shaders::slang::PATH_COUNT_SETUP,
        &[Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let path_count = engine.add_compute_shader(
        device,
        "path_count",
        ekrano_shaders::slang::PATH_COUNT,
        &[Uniform, Buffer, BufReadOnly, BufReadOnly, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let backdrop = engine.add_compute_shader(
        device,
        "backdrop_dyn",
        ekrano_shaders::slang::BACKDROP_DYN,
        &[Uniform, Buffer, BufReadOnly, Buffer],
        &search_paths,
        &[],
    )?;
    let coarse = engine.add_compute_shader(
        device,
        "coarse",
        ekrano_shaders::slang::COARSE,
        &[
            Uniform,
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
    )?;
    let path_tiling_setup = engine.add_compute_shader(
        device,
        "path_tiling_setup",
        ekrano_shaders::slang::PATH_TILING_SETUP,
        &[Buffer, Buffer, Buffer],
        &search_paths,
        &[],
    )?;
    let path_tiling = engine.add_compute_shader(
        device,
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
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_msaa_resources = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly, // mask_lut at slot 8
    ];
    let fine_area = Some(engine.add_compute_shader(
        device,
        "fine_area",
        ekrano_shaders::slang::FINE,
        &fine_resources,
        &search_paths,
        &[],
    )?);
    let fine_msaa8 = engine
        .add_compute_shader(
            device,
            "fine_msaa8",
            ekrano_shaders::slang::FINE,
            &fine_msaa_resources,
            &search_paths,
            &[("msaa", "1"), ("msaa8", "1")],
        )
        .ok();
    let fine_msaa16 = engine
        .add_compute_shader(
            device,
            "fine_msaa16",
            ekrano_shaders::slang::FINE,
            &fine_msaa_resources,
            &search_paths,
            &[("msaa", "1"), ("msaa16", "1")],
        )
        .ok();

    Ok(FullShaders {
        pipeline_setup: Some(pipeline_setup),
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
        pathtag_is_cpu: false,
    })
}
