// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to ensure that certain issues which don't deserve a test scene don't regress

#[path = "common/submission.rs"]
mod submission;

use ekrano::{
    AaConfig, FontEmbolden, Scene,
    kurbo::{Affine, Diagonal2, Rect, RoundedRect, Stroke},
    peniko::{Color, ColorStop, Extend, Gradient, ImageQuality, InterpolationAlphaSpace, color::palette},
};
use ekrano_tests::{TestParams, shared_test_device, smoke_snapshot_test_sync, snapshot_test_sync};
use scenes::ImageCache;
use scenes::SimpleText;

/// Test created from <https://github.com/linebender/vello/issues/616>
fn rounded_rectangle_watertight() {
    let mut scene = Scene::new();
    let rect = RoundedRect::new(60.0, 10.0, 80.0, 30.0, 10.0);
    let stroke = Stroke::new(2.0);
    scene.stroke(&stroke, Affine::IDENTITY, palette::css::WHITE, None, &rect);
    let mut params = TestParams::new("rounded_rectangle_watertight", 70, 30);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
fn test_data_image_roundtrip_extend_pad() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Pad);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new("data_image_roundtrip", image.image.width, image.image.height);
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
fn test_gradient_color_alpha_premultiplied() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Premultiplied),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha_premultiplied", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
fn test_gradient_color_alpha_unpremultiplied() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha_unpremultiplied", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test created from <https://github.com/linebender/vello/issues/662>
fn stroke_width_zero() {
    let mut scene = Scene::new();
    let stroke = Stroke::new(0.0);
    let rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    let rect_stroke_color = palette::css::PEACH_PUFF;
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);
    let mut params = TestParams::new("stroke_width_zero", 50, 50);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}

#[expect(clippy::cast_possible_truncation, reason = "Test code")]
fn text_stroke_width_zero() {
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
        None,
        &Stroke::new(0.),
        "Testing text",
    );
    let params = TestParams::new(
        "text_stroke_width_zero",
        (font_size * 6.) as _,
        (font_size * 1.25).ceil() as _,
    );
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}

/// Honesty gate: Linebender `main` sparse LFS `glyphs_emboldened.png` (Vello #1628).
fn glyphs_emboldened() {
    let font_size = 44_f32;
    let text = "this is regular and emboldened text";
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    let paint = palette::css::REBECCA_PURPLE.with_alpha(0.5);
    simple_text.add_var_run(
        &mut scene,
        None,
        font_size,
        &[],
        &paint,
        Affine::translate((0., f64::from(font_size))),
        None,
        None,
        ekrano::peniko::Fill::NonZero,
        text,
        true,
        FontEmbolden::default(),
    );
    simple_text.add_var_run(
        &mut scene,
        None,
        font_size,
        &[],
        &paint,
        Affine::translate((0., f64::from(font_size) + 58.0)),
        None,
        None,
        ekrano::peniko::Fill::NonZero,
        text,
        true,
        FontEmbolden::new(Diagonal2::new(1.0, 1.0)),
    );
    let mut params = TestParams::new("glyphs_emboldened", 760, 140);
    params.base_color = Some(palette::css::WHITE);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("rounded_rectangle_watertight", || {
            rounded_rectangle_watertight();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("test_data_image_roundtrip_extend_pad", || {
            test_data_image_roundtrip_extend_pad();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("test_gradient_color_alpha_premultiplied", || {
            test_gradient_color_alpha_premultiplied();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("test_gradient_color_alpha_unpremultiplied", || {
            test_gradient_color_alpha_unpremultiplied();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("stroke_width_zero", || {
            stroke_width_zero();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("text_stroke_width_zero", || {
            text_stroke_width_zero();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("glyphs_emboldened", || {
            glyphs_emboldened();
            Ok(())
        })
        .with_ignored_flag(false),
    );

    let mut args = libtest_mimic::Arguments::from_args();
    if let Some(device) = shared_test_device() {
        submission::clamp_test_threads(&mut args, device);
    }
    libtest_mimic::run(&args, trials).exit()
}
