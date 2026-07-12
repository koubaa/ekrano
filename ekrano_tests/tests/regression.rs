// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to ensure that certain issues which don't deserve a test scene don't regress

use ekrano::{
    AaConfig, Scene,
    kurbo::{Affine, Rect, RoundedRect, Stroke},
    peniko::{Extend, ImageQuality, color::palette},
};
use ekrano_tests::{TestBackend, TestParams, smoke_snapshot_test_sync, snapshot_test_sync};
use scenes::ImageCache;
use scenes::SimpleText;

/// Test created from <https://github.com/linebender/vello/issues/616>
fn rounded_rectangle_watertight_body(backend: TestBackend) {
    let mut scene = Scene::new();
    let rect = RoundedRect::new(60.0, 10.0, 80.0, 30.0, 10.0);
    let stroke = Stroke::new(2.0);
    scene.stroke(&stroke, Affine::IDENTITY, palette::css::WHITE, None, &rect);
    let mut params = TestParams::new("rounded_rectangle_watertight", 70, 30).with_backend(backend);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[test]
fn rounded_rectangle_watertight() {
    rounded_rectangle_watertight_body(TestBackend::Classic);
}

#[test]
fn scheme_rounded_rectangle_watertight() {
    rounded_rectangle_watertight_body(TestBackend::Scheme);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
fn test_data_image_roundtrip_extend_pad_body(backend: TestBackend) {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Pad);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params =
        TestParams::new("data_image_roundtrip", image.image.width, image.image.height).with_backend(backend);
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
#[test]
fn test_data_image_roundtrip_extend_pad() {
    test_data_image_roundtrip_extend_pad_body(TestBackend::Classic);
}

#[test]
fn scheme_test_data_image_roundtrip_extend_pad() {
    test_data_image_roundtrip_extend_pad_body(TestBackend::Scheme);
}

/// Test created from <https://github.com/linebender/vello/issues/662>
fn stroke_width_zero_body(backend: TestBackend) {
    let mut scene = Scene::new();
    let stroke = Stroke::new(0.0);
    let rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    let rect_stroke_color = palette::css::PEACH_PUFF;
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);
    let mut params = TestParams::new("stroke_width_zero", 50, 50).with_backend(backend);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[test]
fn stroke_width_zero() {
    stroke_width_zero_body(TestBackend::Classic);
}

#[test]
fn scheme_stroke_width_zero() {
    stroke_width_zero_body(TestBackend::Scheme);
}

#[expect(clippy::cast_possible_truncation, reason = "Test code")]
fn text_stroke_width_zero_body(backend: TestBackend) {
    let font_size = 12.;
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    simple_text.add_run(
        &mut scene,
        None,
        font_size,
        palette::css::WHITE,
        Affine::translate((0., f64::from(font_size))),
        None,
        &Stroke::new(0.),
        "Testing text",
    );
    let params = TestParams::new(
        "text_stroke_width_zero",
        (font_size * 6.) as _,
        (font_size * 1.25).ceil() as _,
    )
    .with_backend(backend);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[test]
fn text_stroke_width_zero() {
    text_stroke_width_zero_body(TestBackend::Classic);
}

#[test]
fn scheme_text_stroke_width_zero() {
    text_stroke_width_zero_body(TestBackend::Scheme);
}
