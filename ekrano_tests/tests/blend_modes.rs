// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for blend modes and non-isolated (per-draw) blending.
//!
//! Ported from `vello_sparse_tests/tests/mix.rs`, translating the stateful
//! `vello_cpu::Renderer` API to ekrano's `Scene` API.
//! Test names and reference PNGs match those in `vello_sparse_tests`.
//!
//! ## Top-level draws
//! In ekrano, `scene.fill()` and similar calls placed *outside* any layer are not
//! rendered.  All drawing must happen inside `push_layer`, `push_filter_layer`, or
//! `push_clip_layer`.  For tests that need an opaque white background we set
//! `params.base_color = Some(WHITE)` instead.
//!
//! For `mix_modes_non_gradient_test_matrix` the whole scene is wrapped in an outer
//! `push_layer(SrcOver)` so that the dark-grey background fill and the per-cell
//! fills are inside a real layer and therefore get rendered.

use ekrano::{
    Scene,
    kurbo::{Affine, Rect},
    peniko::{BlendMode, Color, Compose, Fill, Mix, color::palette::css::*},
};
use ekrano_tests::{TestParams, snapshot_test_sync};

/// Helper: full-viewport rect for use as a layer clip.
fn viewport(width: f64, height: f64) -> Rect {
    Rect::new(0.0, 0.0, width, height)
}

// ─── Non-isolated (per-draw) blend modes ─────────────────────────────────────
//
// These tests use `scene.set_blend_mode` to apply a blend mode to individual
// draw calls without creating a layer, which is the "non-isolated blending"
// feature added to ekrano.

fn non_isolated_blend(mix: Mix) -> Scene {
    let mut scene = Scene::new();
    let vp = viewport(100.0, 100.0);

    // Just to isolate from the white background (comment from vello_sparse source).
    scene.push_layer(
        Fill::NonZero,
        BlendMode::new(Mix::Normal, Compose::SrcOver),
        1.0,
        Affine::IDENTITY,
        &vp,
    );

    let rect1 = Rect::new(10.5, 10.5, 70.5, 70.5);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        BLUE.with_alpha(0.5),
        None,
        &rect1,
    );

    // Non-isolated blend: the second draw blends directly into the accumulated
    // layer content (not into an isolated group).
    scene.set_blend_mode(BlendMode::new(mix, Compose::SrcOver));
    let rect2 = Rect::new(30.5, 30.5, 90.5, 90.5);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        LIME.with_alpha(0.5),
        None,
        &rect2,
    );
    scene.reset_draw_blend_mode();

    scene.pop_layer();
    scene
}

/// Non-isolated `Difference` blend: lime overlaps blue with difference compositing.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn mix_non_isolated_difference() {
    let mut params = TestParams::new("mix_non_isolated_difference", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(non_isolated_blend(Mix::Difference), &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}

/// Non-isolated `SoftLight` blend.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn mix_non_isolated_soft_light() {
    let mut params = TestParams::new("mix_non_isolated_soft_light", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(non_isolated_blend(Mix::SoftLight), &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}

/// Non-isolated `ColorDodge` blend.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn mix_non_isolated_color_dodge() {
    let mut params = TestParams::new("mix_non_isolated_color_dodge", 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(non_isolated_blend(Mix::ColorDodge), &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}

// ─── Layer blend mode matrix ─────────────────────────────────────────────────
//
// A 16 × 8 grid of cells, one per blend mode × base color.
// Each cell draws a destination rect then a blend layer with two overlapping
// source shapes.  Uses solid colors only (no gradients) so that the test works
// independently of any gradient rendering.
//
// All scene content is wrapped in an outer `push_layer(SrcOver)` so that the
// top-level dark-grey fill and per-cell fills are inside a layer and get rendered
// (ekrano ignores bare `scene.fill()` calls that are outside any layer).

/// 16 × 8 grid exercising all standard `Mix` blend modes against 8 base colors.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn mix_modes_non_gradient_test_matrix() {
    let mut scene = Scene::new();
    let vp = viewport(80.0, 160.0);

    let mix_modes = [
        Mix::Normal,
        Mix::Multiply,
        Mix::Screen,
        Mix::Overlay,
        Mix::Darken,
        Mix::Lighten,
        Mix::ColorDodge,
        Mix::ColorBurn,
        Mix::HardLight,
        Mix::SoftLight,
        Mix::Difference,
        Mix::Exclusion,
        Mix::Hue,
        Mix::Saturation,
        Mix::Color,
        Mix::Luminosity,
    ];

    let base_colors: [Color; 8] = [
        RED,
        Color::from_rgb8(10, 230, 10),
        BLUE,
        YELLOW,
        MAGENTA,
        Color::from_rgb8(10, 230, 230),
        Color::from_rgb8(128, 128, 128),
        Color::from_rgb8(64, 64, 64),
    ];

    let cell_size = 10.0_f64;

    // Dark background layer — ekrano requires fills to be inside a layer,
    // so we give the background its own push/pop.
    scene.push_layer(Fill::NonZero, BlendMode::new(Mix::Normal, Compose::SrcOver), 1.0, Affine::IDENTITY, &vp);
    scene.fill(Fill::NonZero, Affine::IDENTITY, Color::from_rgb8(30, 30, 30), None, &vp);
    scene.pop_layer();

    for (row, mix_mode) in mix_modes.iter().enumerate() {
        for (col, base_color) in base_colors.iter().enumerate() {
            let x = col as f64 * cell_size;
            let y = row as f64 * cell_size;

            let cell = Rect::new(x, y, x + cell_size, y + cell_size);
            let blend_rect = Rect::new(x, y, x + cell_size * 0.7, y + cell_size * 0.7);
            let white_rect = Rect::new(
                x + cell_size * 0.3,
                y + cell_size * 0.3,
                x + cell_size,
                y + cell_size,
            );

            // Destination (base color) in its own layer, so fills are rendered.
            // Direct scene.fill() calls outside any layer are not rendered by ekrano.
            scene.push_layer(Fill::NonZero, BlendMode::new(Mix::Normal, Compose::SrcOver), 1.0, Affine::IDENTITY, &cell);
            scene.fill(Fill::NonZero, Affine::IDENTITY, *base_color, None, &cell);
            scene.pop_layer();

            // Source via blend layer.
            scene.push_layer(
                Fill::NonZero,
                BlendMode::new(*mix_mode, Compose::SrcOver),
                1.0,
                Affine::IDENTITY,
                &cell,
            );
            scene.fill(Fill::NonZero, Affine::IDENTITY, ORANGE.with_alpha(0.7), None, &blend_rect);
            scene.fill(Fill::NonZero, Affine::IDENTITY, Color::WHITE.with_alpha(0.5), None, &white_rect);
            scene.pop_layer();
        }
    }

    let params = TestParams::new("mix_modes_non_gradient_test_matrix", 80, 160);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}
