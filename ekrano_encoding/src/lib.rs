// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Raw scene encoding.

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
// Need to allow instead of expect until Rust 1.83 https://github.com/rust-lang/rust/pull/130025
#![allow(missing_docs, reason = "We have many as-yet undocumented items.")]
#![expect(
    missing_debug_implementations,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    reason = "Deferred"
)]
#![allow(unnameable_types, reason = "Deferred, only apply in some feature sets so not expect")]

mod binning;
mod clip;
mod config;
mod coverage_mask;
mod draw;
mod encoding;
#[cfg(feature = "bump_estimate")]
mod estimate;
mod filter;
mod glyph;
mod glyph_cache;
mod image_cache;
mod mask;
pub mod math;
mod monoid;
mod path;
mod ramp_cache;
mod resolve;

pub use binning::BinHeader;
pub use clip::{Clip, ClipBbox, ClipBic, ClipElement};
pub use config::{
    BufferSize, BufferSizes, BumpAllocatorMemory, BumpAllocators, ConfigUniform, IndirectCount, N_INDIRECT_STAGES,
    RenderConfig, STAGE_BACKDROP, STAGE_BBOX_CLEAR, STAGE_BINNING, STAGE_CLIP_LEAF, STAGE_CLIP_REDUCE, STAGE_COARSE,
    STAGE_DRAW_LEAF, STAGE_DRAW_REDUCE, STAGE_FINE, STAGE_FLATTEN, STAGE_PATH_COUNT, STAGE_PATH_COUNT_SETUP,
    STAGE_PATH_TILING, STAGE_PATH_TILING_SETUP, STAGE_PATHTAG_REDUCE, STAGE_PATHTAG_REDUCE2, STAGE_PATHTAG_SCAN,
    STAGE_PATHTAG_SCAN_LARGE, STAGE_PATHTAG_SCAN1, STAGE_TILE_ALLOC, WorkgroupCounts, WorkgroupCountsGpu,
    WorkgroupSize,
};
pub use coverage_mask::CoverageMask;
pub use draw::{
    DRAW_INFO_FLAGS_FILL_RULE_BIT, DrawBbox, DrawBeginClip, DrawBlurRoundedRect, DrawColor, DrawImage, DrawImageTinted,
    DrawLinearGradient, DrawMonoid, DrawRadialGradient, DrawSweepGradient, DrawTag, Tint, TintMode,
};
pub use encoding::{Encoding, Resources, StreamOffsets};
pub use filter::{Filter, FilterEdgeMode, FilterPrimitive, FilterUniform, LayerFilterEffect};
pub use glyph::{FontEmbolden, Glyph, GlyphRun};
pub use image_cache::Images;
pub use mask::{make_mask_lut, make_mask_lut_16};
pub use math::Transform;
pub use monoid::Monoid;
pub use path::{
    Cubic, LineSoup, Path, PathBbox, PathEncoder, PathMonoid, PathSegment, PathSegmentType, PathTag, SegmentCount,
    Style, Tile,
};
pub use ramp_cache::Ramps;
pub use resolve::{LIVE_IMAGE_BIT, Layout, Patch, Resolver, resolve_solid_paths_only};

#[cfg(feature = "bump_estimate")]
pub use estimate::BumpEstimator;

/// A normalized variation coordinate (for variable fonts) in 2.14 fixed point format.
///
/// In most cases, this can be [cast](bytemuck::cast_slice) from the
/// normalised coords provided by your text layout library.
///
/// Equivalent to [`skrifa::instance::NormalizedCoord`], but defined
/// in Vello so that Skrifa is not part of Vello's public API.
/// This allows Vello to update its Skrifa in a patch release, and limits
/// the need for updates only to align Skrifa versions.
pub type NormalizedCoord = i16;

#[cfg(test)]
mod tests {
    const _NORMALISED_COORD_SIZE_MATCHES: () =
        assert!(size_of::<skrifa::prelude::NormalizedCoord>() == size_of::<crate::NormalizedCoord>());
}
