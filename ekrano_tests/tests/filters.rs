// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for filter effects (Gaussian blur, drop shadow, flood, offset).
//!
//! Ported from `vello_sparse_tests/tests/filter.rs`, translating the stateful
//! `vello_cpu::Renderer` API to ekrano's `Scene` API.
//! Test names are shared with `vello_sparse_tests`, but the reference PNGs in
//! `ekrano_tests/snapshots/` are **ekrano-specific** baselines rather than the
//! vello_sparse golden images — the two renderers accumulate small numerical
//! rounding differences (GPU vs CPU, different AA approaches, premul rounding)
//! that make direct reuse impractical.  Divergence from vello_sparse is expected
//! and acceptable; what matters is consistency across ekrano builds.
//!
//! ## Background colour
//! vello_sparse initialises its render surface to opaque **white** before each
//! test; the filter shaders therefore operate on white-backed content.  We
//! replicate this via two complementary mechanisms:
//!
//! 1. `params.base_color = Some(WHITE)` — tells the snapshot-comparison code to
//!    composite both the rendered and reference images over white, making
//!    transparent "background" regions match the white-backed reference.
//!    Filter layers are rasterized in isolation over transparent black; post-fine
//!    passes then composite back onto the surface.

use ekrano::{
    Scene,
    kurbo::{Affine, BezPath, Point, Rect, Stroke, Vec2},
    peniko::{Color, Fill, color::palette::css::*},
};
use ekrano_encoding::{Filter, FilterEdgeMode, FilterPrimitive};
use ekrano_tests::{TestParams, snapshot_test_sync};

/// Build a star polygon centered at `center` with `n` points, alternating
/// between `inner` and `outer` radii.
fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
    let mut path = BezPath::new();
    let start_angle = -std::f64::consts::FRAC_PI_2;
    path.move_to(center + outer * Vec2::from_angle(start_angle));
    for i in 1..n * 2 {
        let th = start_angle + i as f64 * std::f64::consts::PI / n as f64;
        let r = if i % 2 == 0 { outer } else { inner };
        path.line_to(center + r * Vec2::from_angle(th));
    }
    path.close_path();
    path
}

/// Full-viewport rect.
fn vp(w: f64, h: f64) -> Rect {
    Rect::new(0.0, 0.0, w, h)
}

// ─── Gaussian blur ──────────────────────────────────────────────────────────

/// Gaussian blur with small radius (`std_dev` = 2.0, no decimation).
#[test]
fn filter_gaussian_blur_no_decimation() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 2.0,
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &Rect::new(20.0, 20.0, 80.0, 80.0));
    scene.pop_layer();
    let mut params = TestParams::new("filter_gaussian_blur_no_decimation", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Gaussian blur with larger radius (`std_dev` = 4.0, uses decimation).
#[test]
fn filter_gaussian_blur_with_decimation() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 4.0,
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &Rect::new(20.0, 20.0, 80.0, 80.0));
    scene.pop_layer();
    let mut params = TestParams::new("filter_gaussian_blur_with_decimation", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Zero blur acts as identity (no-op).
#[test]
fn filter_gaussian_blur_zero() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 0.0,
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &Rect::new(25.0, 25.0, 75.0, 75.0));
    scene.pop_layer();
    let mut params = TestParams::new("filter_gaussian_blur_zero", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Blur with very large `std_dev` (= 20.0) — shape barely visible.
#[test]
fn filter_extreme_blur() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 20.0,
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &Rect::new(25.0, 25.0, 75.0, 75.0));
    scene.pop_layer();
    let mut params = TestParams::new("filter_extreme_blur", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Blur on semi-transparent shapes — fully-opaque (left) and 50%-transparent (right).
#[test]
fn filter_transparent_shapes() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 3.0,
        edge_mode: FilterEdgeMode::None,
    });

    scene.push_filter_layer(filter.clone(), Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &Rect::new(10.0, 25.0, 40.0, 75.0));
    scene.pop_layer();

    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgba8(150, 100, 200, 128), None, &Rect::new(60.0, 25.0, 90.0, 75.0));
    scene.pop_layer();

    let mut params = TestParams::new("filter_transparent_shapes", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

// ─── Gaussian blur edge modes ─────────────────────────────────────────────────
//
// The colored bands fill the entire canvas, so the blur happens on top of an
// already-opaque image.  No white-inside-filter fill is needed here, but the
// edge modes may differ slightly between GPU (ekrano) and CPU (vello_sparse),
// so a slightly looser threshold is used.

fn blur_with_edge_mode(edge_mode: FilterEdgeMode) -> Scene {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur { std_dev: 6.0, edge_mode });
    let step = 256.0 / 3.0;

    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(256.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, RED,   None, &Rect::new(0.0,        0.0, step,        100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, BLUE,  None, &Rect::new(step,       0.0, 2.0 * step,  100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, GREEN, None, &Rect::new(2.0 * step, 0.0, 3.0 * step,  100.0));
    scene.pop_layer();
    scene
}

#[test]
fn filter_gaussian_blur_edge_mode_duplicate() {
    let mut params = TestParams::new("filter_gaussian_blur_edge_mode_duplicate", 256, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(blur_with_edge_mode(FilterEdgeMode::Duplicate), &params)
        .unwrap().assert_mean_less_than(0.04);
}

#[test]
fn filter_gaussian_blur_edge_mode_wrap() {
    let mut params = TestParams::new("filter_gaussian_blur_edge_mode_wrap", 256, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(blur_with_edge_mode(FilterEdgeMode::Wrap), &params)
        .unwrap().assert_mean_less_than(0.06);
}

#[test]
fn filter_gaussian_blur_edge_mode_mirror() {
    let mut params = TestParams::new("filter_gaussian_blur_edge_mode_mirror", 256, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(blur_with_edge_mode(FilterEdgeMode::Mirror), &params)
        .unwrap().assert_mean_less_than(0.04);
}

// ─── Flood filter ─────────────────────────────────────────────────────────────

/// Flood filter: replaces layer content with a solid color.
///
/// The filter layer's clip is `drawn_rect`, which tells `push_filter_layer` to
/// record those bounds in `FilterPrimitive::Flood::clip_rect`.  The shader then
/// restricts the flood to that rect, matching vello_sparse's auto-bounded
/// per-layer-pixmap semantics.
#[test]
fn filter_flood() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::Flood { color: TOMATO.premultiply(), clip_rect: [0; 4] });
    let drawn_rect = Rect::new(0.0, 8.0, 256.0, 32.0);
    // The filter layer clip IS drawn_rect — push_filter_layer computes its bounding box
    // and stores it in FilterPrimitive::Flood::clip_rect for the shader.
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &drawn_rect);
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &drawn_rect);
    scene.pop_layer();
    let mut params = TestParams::new("filter_flood", 256, 40);
    params.base_color = Some(WHITE);
    // Render with transparent clear so the flood shader can detect drawn pixels via src.a.
    params.render_clear_color = Some(Color::TRANSPARENT);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Flood filter on a star-shaped fill (no extra clip wrapper).
#[test]
fn filter_flood_star() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::Flood { color: TOMATO.premultiply(), clip_rect: [0; 4] });
    let star_path = circular_star(Point::new(50.0, 50.0), 5, 20.0, 40.0);

    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &star_path);
    scene.pop_layer();

    let mut params = TestParams::new("filter_flood_star", 100, 100);
    params.base_color = Some(WHITE);
    // Render with transparent clear so s.a carries the star's AA coverage, which the
    // flood shader then copies into the output alpha — matching vello_sparse edge behaviour.
    params.render_clear_color = Some(Color::TRANSPARENT);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

// ─── Drop shadow ─────────────────────────────────────────────────────────────

/// Drop shadow with sub-pixel offsets.
#[test]
fn filter_drop_shadow_fractional_offset() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::DropShadow {
        dx: 2.5,
        dy: 3.7,
        std_dev: 1.0,
        color: Color::from_rgba8(0, 0, 0, 180).premultiply(),
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, ROYAL_BLUE, None, &Rect::new(30.0, 30.0, 70.0, 70.0));
    scene.pop_layer();
    let mut params = TestParams::new("filter_drop_shadow_fractional_offset", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Drop shadow with zero offset (shadow directly behind).
#[test]
fn filter_drop_shadow_zero_offset() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::DropShadow {
        dx: 0.0,
        dy: 0.0,
        std_dev: 4.0,
        color: Color::from_rgba8(0, 0, 0, 180).premultiply(),
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, ROYAL_BLUE, None, &Rect::new(30.0, 30.0, 70.0, 70.0));
    scene.pop_layer();
    let mut params = TestParams::new("filter_drop_shadow_zero_offset", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

// ─── Offset filter ────────────────────────────────────────────────────────────

/// Offset filter shifts content within a filter layer (no clipping to original bounds).
///
/// Reference stroke + marker use a zero-blur filter layer as an identity pass-through
/// (ekrano does not render top-level draws outside a layer).
#[test]
fn filter_offset() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::Offset { dx: 18.0, dy: -12.0 });
    let star_path = circular_star(Point::new(50.0, 50.0), 7, 10.0, 22.0);
    let marker = Rect::new(49.0, 27.0, 53.0, 31.0);

    // Draw unfiltered reference stroke + marker via an identity (zero-blur) filter layer
    // so they appear at their original positions over a white background.
    let identity = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 0.0,
        edge_mode: FilterEdgeMode::None,
    });
    scene.push_filter_layer(identity, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.stroke(&Stroke::new(1.5), Affine::IDENTITY, ROYAL_BLUE, None, &star_path);
    scene.fill(Fill::NonZero, Affine::IDENTITY, SEA_GREEN, None, &marker);
    scene.pop_layer();

    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, TOMATO, None, &star_path);
    scene.stroke(&Stroke::new(1.5), Affine::IDENTITY, BLACK, None, &star_path);
    scene.fill(Fill::NonZero, Affine::IDENTITY, VIOLET, None, &marker);
    scene.pop_layer();

    let mut params = TestParams::new("filter_offset", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

// ─── Layer structure tests ───────────────────────────────────────────────────

/// Nested filter layers: Gaussian blur inside a drop shadow.
///
/// TODO: nested filter layers require multi-pass fine, not yet implemented.
#[test]
fn filter_nested_layers() {
    let mut scene = Scene::new();
    let blur = Filter(FilterPrimitive::GaussianBlur { std_dev: 2.0, edge_mode: FilterEdgeMode::None });
    let shadow = Filter(FilterPrimitive::DropShadow {
        dx: 12.0,
        dy: 12.0,
        std_dev: 4.0,
        color: Color::from_rgba8(0, 0, 0, 180).premultiply(),
        edge_mode: FilterEdgeMode::None,
    });

    scene.push_filter_layer(shadow, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.push_filter_layer(blur, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.fill(Fill::NonZero, Affine::IDENTITY, REBECCA_PURPLE, None, &Rect::new(25.0, 25.0, 75.0, 75.0));
    scene.pop_layer();
    scene.pop_layer();

    let mut params = TestParams::new("filter_nested_layers", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}

/// Three nested filter layers with no content drawn — white background is all that shows.
///
/// TODO: nested filter layers require multi-pass fine, not yet implemented.
#[test]
fn filter_empty_layers() {
    let mut scene = Scene::new();
    let filter = Filter(FilterPrimitive::GaussianBlur { std_dev: 4.0, edge_mode: FilterEdgeMode::None });

    scene.push_filter_layer(filter.clone(), Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.push_filter_layer(filter.clone(), Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    scene.push_filter_layer(filter, Fill::NonZero, Affine::IDENTITY, &vp(100.0, 100.0));
    // Draw nothing.
    scene.pop_layer();
    scene.pop_layer();
    scene.pop_layer();

    let mut params = TestParams::new("filter_empty_layers", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.0095);
}
