// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Slang shader sources for the Goldy backend.
//!
//! These are used when building with the `goldy` feature.

use std::path::PathBuf;

/// Path to the slang directory for Slang compiler search paths.
/// Required for `__include "ekrano_shared"` to resolve.
pub fn slang_search_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("slang")
}

macro_rules! include_slang {
    ($name:ident, $file:literal) => {
        pub const $name: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/slang/", $file));
    };
}

include_slang!(EKRANO_SHARED, "ekrano_shared.slang");
include_slang!(BBOX_CLEAR, "bbox_clear.slang");
include_slang!(PIPELINE_SETUP, "pipeline_setup.slang");
include_slang!(PATH_COUNT_SETUP, "path_count_setup.slang");
include_slang!(PATH_COUNT_SETUP_SCHEME, "path_count_setup_scheme.slang");
include_slang!(PATH_TILING_SETUP, "path_tiling_setup.slang");
include_slang!(PATH_TILING_SETUP_SCHEME, "path_tiling_setup_scheme.slang");
include_slang!(PATHTAG_REDUCE, "pathtag_reduce.slang");
include_slang!(PATHTAG_REDUCE2, "pathtag_reduce2.slang");
include_slang!(PATHTAG_SCAN1, "pathtag_scan1.slang");
include_slang!(PATHTAG_SCAN_SMALL, "pathtag_scan_small.slang");
include_slang!(PATHTAG_SCAN_LARGE, "pathtag_scan_large.slang");
include_slang!(DRAW_REDUCE, "draw_reduce.slang");
include_slang!(CLIP_REDUCE, "clip_reduce.slang");
include_slang!(CLIP_LEAF, "clip_leaf.slang");
include_slang!(DRAW_LEAF, "draw_leaf.slang");
include_slang!(BINNING, "binning.slang");
include_slang!(TILE_ALLOC, "tile_alloc.slang");
include_slang!(PATH_COUNT, "path_count.slang");
include_slang!(BACKDROP, "backdrop.slang");
include_slang!(BACKDROP_DYN, "backdrop_dyn.slang");
include_slang!(COARSE, "coarse.slang");
include_slang!(PATH_TILING, "path_tiling.slang");
include_slang!(FLATTEN, "flatten.slang");
include_slang!(FINE, "fine.slang");
include_slang!(FILTER_PASS, "filter_pass.slang");
