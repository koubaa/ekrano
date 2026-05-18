// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple property tests of rendered Vello scenes.

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::missing_assert_message,
    clippy::allow_attributes_without_reason
)]

use ekrano::Scene;
use ekrano::kurbo::{Affine, Rect};
use ekrano::peniko::color::palette::css::TRANSPARENT;
use ekrano::peniko::{Brush, Color, ImageFormat, color::palette};
use ekrano::peniko::{ImageAlphaType, ImageData, ImageSampler};
use ekrano_tests::TestParams;

fn simple_square() {
    let mut scene = Scene::new();
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Rect::from_center_size((100., 100.), (50., 50.)),
    );
    let params = TestParams::new("simple_square", 150, 150);
    let image = ekrano_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, ImageFormat::Rgba8);
    let mut red_count = 0;
    let mut black_count = 0;
    for pixel in image.data.data().chunks_exact(4) {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let is_red = r == 255 && g == 0 && b == 0 && a == 255;
        let is_black = r == 0 && g == 0 && b == 0 && a == 255;
        if !is_red && !is_black {
            panic!("{pixel:?}");
        }
        match (is_red, is_black) {
            (true, true) => unreachable!(),
            (true, false) => red_count += 1,
            (false, true) => black_count += 1,
            (false, false) => panic!("Got unexpected pixel {pixel:?}"),
        }
    }
    assert_eq!(red_count, 50 * 50);
    assert_eq!(black_count, 150 * 150 - 50 * 50);
}

fn empty_scene() {
    let scene = Scene::new();

    // Adding an alpha factor here changes the resulting color *slightly*,
    // presumably due to pre-multiplied alpha.
    // We just assume that alpha scenarios work fine
    let color = palette::css::PLUM;
    let mut params = TestParams::new("simple_square", 150, 150);
    params.base_color = Some(color);
    let image = ekrano_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, ImageFormat::Rgba8);
    for pixel in image.data.data().chunks_exact(4) {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let image_color = Color::from_rgba8(r, g, b, a);
        if image_color.premultiply().difference(color.premultiply()) > 1e-4 {
            panic!("Got {image_color:?}, expected clear color {color:?}");
        }
    }
}

#[test]
fn simple_square_test() {
    simple_square();
}

#[test]
fn tiny_red_2x2_test() {
    let mut scene = Scene::new();
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Rect::from_origin_size((0., 0.), (2., 2.)),
    );
    let params = TestParams::new("tiny_red_2x2", 2, 2);
    let image = ekrano_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, ImageFormat::Rgba8);
    let mut red_count = 0;
    for pixel in image.data.data().chunks_exact(4) {
        let &[r, g, b, a] = pixel else { unreachable!() };
        if r == 255 && g == 0 && b == 0 && a == 255 {
            red_count += 1;
        } else {
            eprintln!("pixel: [{r}, {g}, {b}, {a}]");
        }
    }
    assert_eq!(
        red_count, 4,
        "expected 4 red pixels in 2x2, got {red_count}"
    );
}

#[test]
fn empty_scene_test() {
    empty_scene();
}

#[test]
fn bgra_image() {
    let mut scene = Scene::new();
    let colors = [
        palette::css::RED,
        palette::css::BLUE,
        palette::css::LIME,
        palette::css::WHITE,
    ];
    let blob: Vec<u8> = colors
        .iter()
        .flat_map(|c| {
            let [r, g, b, a] = c.to_rgba8().to_u8_array();
            [b, g, r, a]
        })
        .collect();
    let image = ekrano::peniko::ImageBrush {
        image: ImageData {
            data: blob.into(),
            format: ImageFormat::Bgra8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::Alpha,
        },
        sampler: ImageSampler {
            quality: ekrano::peniko::ImageQuality::Low,
            ..Default::default()
        },
    };
    scene.draw_image(&image, Affine::IDENTITY);
    let scene_image =
        ekrano_tests::render_then_debug_sync(&scene, &TestParams::new("bgra", 2, 2)).unwrap();
    assert_eq!(scene_image.format, ImageFormat::Rgba8);
    for (i, pixel) in scene_image.data.data().chunks_exact(4).enumerate() {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let image_color = Color::from_rgba8(r, g, b, a);
        let color = colors[i];
        if image_color.premultiply().difference(color.premultiply()) > 1e-4 {
            panic!("Got {image_color:?}, expected color {color:?}");
        }
    }
}

#[test]
fn premultiplied_image() {
    let mut scene = Scene::new();
    let colors = [
        palette::css::RED.with_alpha(0.5).premultiply(),
        palette::css::BLUE.with_alpha(0.5).premultiply(),
        palette::css::LIME.with_alpha(0.5).premultiply(),
        palette::css::WHITE.with_alpha(0.5).premultiply(),
    ];
    let blob: Vec<u8> = colors
        .iter()
        .flat_map(|c| c.to_rgba8().to_u8_array())
        .collect();
    let image = ekrano::peniko::ImageBrush {
        image: ImageData {
            data: blob.into(),
            format: ImageFormat::Rgba8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::AlphaPremultiplied,
        },
        sampler: ImageSampler {
            quality: ekrano::peniko::ImageQuality::Low,
            ..Default::default()
        },
    };
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new("bgra", 2, 2);
    params.base_color = Some(TRANSPARENT);
    let scene_image = ekrano_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(scene_image.format, ImageFormat::Rgba8);
    for (i, pixel) in scene_image.data.data().chunks_exact(4).enumerate() {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let image_color = Color::from_rgba8(r, g, b, a).premultiply();
        let color = colors[i];
        if image_color.difference(color) > 1e-2 {
            panic!("Got {image_color:?}, expected color {color:?}");
        }
    }
}

/// Confirms that [`ImageAlphaType::Alpha`] (straight-alpha) and
/// [`ImageAlphaType::AlphaPremultiplied`] produce pixel-equivalent output when the
/// source data describes the same visual colours.  Since we now premultiply straight-alpha
/// images on CPU before uploading to the atlas, the GPU path treats both identically.
#[test]
fn straight_alpha_equals_premultiplied() {
    use ekrano::peniko::{ImageBrush, ImageQuality, ImageSampler};

    let opaque_colors: [Color; 4] = [
        palette::css::RED,
        palette::css::BLUE,
        palette::css::LIME,
        palette::css::WHITE,
    ];
    let alpha: u32 = 128;

    let straight_blob: Vec<u8> = opaque_colors
        .iter()
        .flat_map(|c| {
            let [r, g, b, _] = c.to_rgba8().to_u8_array();
            [r, g, b, alpha as u8]
        })
        .collect();

    let premul_blob: Vec<u8> = opaque_colors
        .iter()
        .flat_map(|c| {
            let [r, g, b, _] = c.to_rgba8().to_u8_array();
            let pm = |ch: u8| ((ch as u32 * alpha + 127) / 255) as u8;
            [pm(r), pm(g), pm(b), alpha as u8]
        })
        .collect();

    let make_brush = |data: Vec<u8>, at: ImageAlphaType| -> ImageBrush {
        ImageBrush {
            image: ImageData {
                data: data.into(),
                format: ImageFormat::Rgba8,
                width: 2,
                height: 2,
                alpha_type: at,
            },
            sampler: ImageSampler {
                quality: ImageQuality::Low,
                ..Default::default()
            },
        }
    };

    let render = |brush: ImageBrush| {
        let mut scene = Scene::new();
        scene.draw_image(&brush, Affine::IDENTITY);
        let mut params = TestParams::new("premul_equiv", 2, 2);
        params.base_color = Some(TRANSPARENT);
        ekrano_tests::render_then_debug_sync(&scene, &params).unwrap()
    };

    let out_straight = render(make_brush(straight_blob, ImageAlphaType::Alpha));
    let out_premul = render(make_brush(premul_blob, ImageAlphaType::AlphaPremultiplied));

    assert_eq!(
        out_straight.data.data(),
        out_premul.data.data(),
        "straight-alpha and premultiplied-alpha must produce byte-identical output \
         after atlas premultiplication"
    );
}

/// Confirms that fully-opaque straight-alpha images are rendered correctly after
/// the CPU premultiplication pass (premul is a no-op when a=255).
#[test]
fn fully_opaque_straight_alpha_unchanged() {
    let colors = [
        palette::css::RED,
        palette::css::BLUE,
        palette::css::LIME,
        palette::css::WHITE,
    ];
    let blob: Vec<u8> = colors
        .iter()
        .flat_map(|c| c.to_rgba8().to_u8_array())
        .collect();
    let image = ekrano::peniko::ImageBrush {
        image: ImageData {
            data: blob.into(),
            format: ImageFormat::Rgba8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::Alpha,
        },
        sampler: ImageSampler {
            quality: ekrano::peniko::ImageQuality::Low,
            ..Default::default()
        },
    };
    let mut scene = Scene::new();
    scene.draw_image(&image, Affine::IDENTITY);
    let result = ekrano_tests::render_then_debug_sync(
        &scene,
        &TestParams::new("fully_opaque_straight", 2, 2),
    )
    .unwrap();
    assert_eq!(result.format, ImageFormat::Rgba8);
    for (i, pixel) in result.data.data().chunks_exact(4).enumerate() {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let image_color = Color::from_rgba8(r, g, b, a);
        let expected = colors[i];
        assert!(
            image_color.premultiply().difference(expected.premultiply()) < 1e-3,
            "pixel {i}: got {image_color:?}, expected {expected:?}"
        );
    }
}
