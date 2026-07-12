// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to validate our snapshot testing ability

use ekrano::{
    Scene,
    kurbo::{Affine, Circle, Rect},
    peniko::{Brush, Fill, color::palette},
};
use ekrano_tests::{TestBackend, TestParams, smoke_snapshot_test_sync};
use scenes::SimpleText;

fn filled_square_body(backend: TestBackend) {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Rect::from_center_size((10., 10.), (6., 6.)),
    );
    let params = TestParams::new("filled_square", 20, 20).with_backend(backend);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}
#[test]
fn filled_square() {
    filled_square_body(TestBackend::Classic);
}

#[test]
fn scheme_filled_square() {
    filled_square_body(TestBackend::Scheme);
}

fn filled_circle_body(backend: TestBackend) {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Circle::new((10., 10.), 7.),
    );
    let params = TestParams::new("filled_circle", 20, 20).with_backend(backend);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}
#[test]
fn filled_circle() {
    filled_circle_body(TestBackend::Classic);
}

#[test]
fn scheme_filled_circle() {
    filled_circle_body(TestBackend::Scheme);
}

fn two_emoji_body(backend: TestBackend) {
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
    let params = TestParams::new("two_emoji", 60, 30).with_backend(backend);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}
#[test]
fn two_emoji() {
    two_emoji_body(TestBackend::Classic);
}

#[test]
fn scheme_two_emoji() {
    two_emoji_body(TestBackend::Scheme);
}
