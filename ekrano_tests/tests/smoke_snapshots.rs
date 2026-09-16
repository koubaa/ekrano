// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to validate our snapshot testing ability

#[path = "common/submission.rs"]
mod submission;

use ekrano::{
    Scene,
    kurbo::{Affine, Circle, Rect},
    peniko::{Brush, Fill, Gradient, color::palette},
};
use ekrano_tests::{TestParams, smoke_snapshot_test_sync};
use scenes::SimpleText;

fn filled_square() {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Rect::from_center_size((10., 10.), (6., 6.)),
    );
    let params = TestParams::new("filled_square", 20, 20);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn filled_circle() {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Circle::new((10., 10.), 7.),
    );
    let params = TestParams::new("filled_circle", 20, 20);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn two_emoji() {
    let mut scene = Scene::new();
    let mut text = SimpleText::new();
    text.add_colr_emoji_run(&mut scene, 24., Affine::translate((0., 24.)), None, Fill::NonZero, "🤠");
    text.add_bitmap_emoji_run(
        &mut scene,
        24.,
        Affine::translate((30., 24.)),
        None,
        Fill::NonZero,
        "🤠",
    );
    let params = TestParams::new("two_emoji", 60, 30);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn glyph_gradient_brush_transform() {
    let mut scene = Scene::new();
    let mut text = SimpleText::new();
    // The gradient starts to the right of the text. Without a brush transform,
    // pad extension clamps the whole run to red; with the transform below, the
    // gradient is translated over the glyphs and becomes visibly red-lime-blue.
    let gradient = Gradient::new_linear((200.0, 0.0), (320.0, 0.0)).with_stops([
        palette::css::RED,
        palette::css::LIME,
        palette::css::BLUE,
    ]);

    text.add_run(
        &mut scene,
        None,
        40.0,
        &gradient,
        Affine::translate((8.0, 38.0)),
        None,
        None,
        Fill::NonZero,
        "GRAD",
    );
    text.add_run(
        &mut scene,
        None,
        40.0,
        &gradient,
        Affine::translate((8.0, 82.0)),
        None,
        Some(Affine::translate((-200.0, 0.0))),
        Fill::NonZero,
        "GRAD",
    );

    let params = TestParams::new("glyph_gradient_brush_transform", 150, 92);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("filled_square", || {
            filled_square();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("filled_circle", || {
            filled_circle();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("two_emoji", || {
            two_emoji();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("glyph_gradient_brush_transform", || {
            glyph_gradient_brush_transform();
            Ok(())
        })
        .with_ignored_flag(false),
    );

    let args = libtest_mimic::Arguments::from_args();
    submission::run_gpu_snapshot_trials(args, trials);
}
