// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Live-texture property tests.

#[path = "common/submission.rs"]
mod submission;

use std::sync::Arc;

use ekrano::kurbo::Affine;
use ekrano::peniko::color::palette::css::TRANSPARENT;
use ekrano::peniko::{Blob, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality, ImageSampler};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use ekrano_tests::{TestParams, render_then_debug_sync, shared_test_device};

fn image_brush(data: ImageData) -> ImageBrush {
    ImageBrush {
        image: data,
        sampler: ImageSampler {
            quality: ImageQuality::Low,
            ..Default::default()
        },
    }
}

fn solid_rgba(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    rgba.repeat((width * height) as usize)
}

fn cpu_image(width: u32, height: u32, rgba: Vec<u8>) -> ImageData {
    ImageData {
        data: Blob::new(Arc::new(rgba)),
        format: ImageFormat::Rgba8,
        width,
        height,
        alpha_type: ImageAlphaType::Alpha,
    }
}

fn publish_live_texture(renderer: &mut GoldyRenderer, id: ekrano::LiveTextureId, rgba: &[u8]) {
    let exchange = renderer.live_textures_mut();
    let slot = exchange
        .begin_publish(id)
        .expect("begin_publish")
        .expect("available live slot");
    let texture = exchange.slot_texture(id, slot).expect("slot texture").borrow();
    #[allow(deprecated, reason = "write is the current Goldy CPU upload path")]
    texture.write(rgba).expect("write live texture bytes");
    exchange
        .complete_publish_ready(id, slot)
        .expect("complete_publish_ready");
    exchange.sync_sample_mirror(id).expect("sync_sample_mirror");
}

fn live_render_params(width: u32, height: u32) -> RenderParams {
    RenderParams {
        base_color: TRANSPARENT,
        width,
        height,
        antialiasing_method: AaConfig::Area,
        robust: true,
    }
}

fn live_texture_matches_cpu_image() {
    let Some(device) = shared_test_device() else {
        return;
    };

    let mut renderer = GoldyRenderer::new(device).expect("GoldyRenderer::new");
    let rgba = vec![
        255, 0, 0, 255, 0, 255, 0, 255, //
        0, 0, 255, 255, 255, 255, 255, 255,
    ];
    let (id, live_image) = renderer.alloc_live_texture(2, 2).expect("alloc_live_texture");
    publish_live_texture(&mut renderer, id, &rgba);

    let mut live_scene = Scene::new();
    live_scene.draw_image(&image_brush(live_image), Affine::IDENTITY);
    let live_pixels = renderer
        .render_to_buffer(&live_scene, &live_render_params(2, 2))
        .expect("render live scene");

    let mut cpu_scene = Scene::new();
    cpu_scene.draw_image(&image_brush(cpu_image(2, 2, rgba)), Affine::IDENTITY);
    let mut params = TestParams::new("live_texture_matches_cpu_image", 2, 2);
    params.base_color = Some(TRANSPARENT);
    let expected = render_then_debug_sync(&cpu_scene, &params).expect("render expected CPU image");

    assert_eq!(
        live_pixels,
        expected.data.data(),
        "live texture pixels should match CPU image"
    );
}

fn multiple_live_textures_pack_linearly() {
    let Some(device) = shared_test_device() else {
        return;
    };

    let mut renderer = GoldyRenderer::new(device).expect("GoldyRenderer::new");
    let left_rgba = solid_rgba(1, 1, [200, 40, 20, 255]);
    let right_rgba = solid_rgba(1, 1, [20, 80, 220, 255]);
    let (left_id, left_live) = renderer.alloc_live_texture(1, 1).expect("alloc left live texture");
    let (right_id, right_live) = renderer.alloc_live_texture(1, 1).expect("alloc right live texture");
    publish_live_texture(&mut renderer, left_id, &left_rgba);
    publish_live_texture(&mut renderer, right_id, &right_rgba);

    let mut live_scene = Scene::new();
    live_scene.draw_image(&image_brush(left_live), Affine::IDENTITY);
    live_scene.draw_image(&image_brush(right_live), Affine::translate((1.0, 0.0)));
    let live_pixels = renderer
        .render_to_buffer(&live_scene, &live_render_params(2, 1))
        .expect("render multi-live scene");

    let mut cpu_scene = Scene::new();
    cpu_scene.draw_image(&image_brush(cpu_image(1, 1, left_rgba)), Affine::IDENTITY);
    cpu_scene.draw_image(&image_brush(cpu_image(1, 1, right_rgba)), Affine::translate((1.0, 0.0)));
    let mut params = TestParams::new("multiple_live_textures_pack_linearly", 2, 1);
    params.base_color = Some(TRANSPARENT);
    let expected = render_then_debug_sync(&cpu_scene, &params).expect("render expected CPU scene");

    assert_eq!(
        live_pixels,
        expected.data.data(),
        "packed live textures should match CPU images"
    );
}

fn empty_unregistered_live_blob_errors() {
    let mut scene = Scene::new();
    scene.draw_image(
        &image_brush(ImageData {
            data: Blob::new(Arc::new(Vec::<u8>::new())),
            format: ImageFormat::Rgba8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::Alpha,
        }),
        Affine::IDENTITY,
    );

    let mut params = TestParams::new("empty_unregistered_live_blob_errors", 2, 2);
    params.base_color = Some(TRANSPARENT);
    let err = render_then_debug_sync(&scene, &params).expect_err("empty blob must not upload");
    let message = format!("{err:#}");
    assert!(
        message.contains("Invalid empty image"),
        "expected InvalidImage error, got: {message}"
    );
}

fn live_texture_drops_publishes_when_slots_stay_busy() {
    let Some(device) = shared_test_device() else {
        return;
    };

    let mut renderer = GoldyRenderer::new(device).expect("GoldyRenderer::new");
    let (id, _) = renderer.alloc_live_texture(1, 1).expect("alloc_live_texture");
    let exchange = renderer.live_textures_mut();
    let mut reserved = Vec::new();
    while let Some(slot) = exchange.begin_publish(id).expect("begin_publish") {
        reserved.push(slot);
    }

    assert!(
        exchange.dropped_publishes > 0,
        "mailbox should report dropped publishes when every slot is reserved"
    );

    for slot in reserved {
        exchange.cancel_publish(id, slot);
    }
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("live_texture_matches_cpu_image", || {
            live_texture_matches_cpu_image();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("multiple_live_textures_pack_linearly", || {
            multiple_live_textures_pack_linearly();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("empty_unregistered_live_blob_errors", || {
            empty_unregistered_live_blob_errors();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("live_texture_drops_publishes_when_slots_stay_busy", || {
            live_texture_drops_publishes_when_slots_stay_busy();
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
