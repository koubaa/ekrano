// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reproductions for known bugs, to allow test driven development

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::missing_assert_message,
    clippy::should_panic_without_expect,
    clippy::allow_attributes_without_reason
)]

use ekrano::{
    AaConfig, Scene,
    kurbo::{Affine, Rect, Triangle},
    peniko::{Color, ColorStop, Extend, Gradient, ImageFormat, ImageQuality, Mix, color::palette},
};
use ekrano_tests::{TestParams, smoke_snapshot_test_sync, snapshot_test_sync};
use scenes::ImageCache;

/// A reproduction of <https://github.com/linebender/vello/issues/680>
///
/// # Test status:
/// Previously flaky on DX12 (intermittent 16,384-pixel / 64-tile deficit).
/// Fixed by the `TaskGraph` refactor — see below. The Vulkan backend was
/// already stable after fixes 1–3 below.
///
/// ## Root cause (fixed)
///
/// The DX12 flake was a missing UAV barrier between the pool `ClearBuffer`
/// and the first wave of compute dispatches.
///
/// On Vulkan it was masked because the Vulkan backend inserts a global
/// `COMPUTE_SHADER write → COMPUTE_SHADER read|write` memory barrier before
/// **every** dispatch (so `Barrier` / `ResourceBarrier` commands are no-ops
/// for correctness purposes).
///
/// On DX12 there is no per-dispatch barrier — the backend relies on
/// `ResourceBarrier` commands emitted by the graph scheduler at wave
/// boundaries. Previously, `ClearBuffer` lived in `ComputeGraph::prelude`,
/// which bypassed dependency analysis. Wave 0 had no barrier before it, so
/// the GPU could start executing wave-0 shaders before
/// `ClearUnorderedAccessViewUint` finished zeroing the pool buffer.
///
/// ## Fix (TaskGraph refactor)
///
/// `ComputeGraph` has been renamed to `TaskGraph` and the `pub prelude`
/// escape hatch removed. Pool clears and buffer writes are now first-class
/// graph nodes with `NodeAccess::Write`. The dependency analyzer sees them
/// and inserts the required `ResourceBarrier` before any downstream reader,
/// including the DX12 `ClearUnorderedAccessViewUint → compute` boundary.
///
/// # Test design:
/// Draws a large red rectangle across a 4352x4352 viewport (17x17 bins, 272x272 tiles).
/// The coarse shader caps `bin_ix` at 256, so 256 of the 289 bins should render red.
///
/// # Fixes applied (Vulkan):
///
/// 1. **binning.slang**: Added bounds checks (`bin_ix < N_TILE`) on `sh_bitmaps` writes
///    to prevent out-of-bounds shared memory access when `bin_ix >= 256`.
///
/// 2. **coarse.slang**: Added missing `GroupMemoryBarrierWithGroupSync()` between
///    `sh_part_count[local_id.x] = part_start_ix + count` and
///    `ready_ix = sh_part_count[WG_SIZE - 1]`. Without the barrier, subgroup 0 could
///    read `sh_part_count[255]` before subgroup 7 had written its prefix-sum result,
///    causing `ready_ix = 0` instead of 1 for a non-deterministic subset of bins.
///    The tile-count prefix sum already had the equivalent barrier; this one was missing
///    only on the partition-count prefix sum. Confirmed via SPIR-V disassembly.
///
/// 3. **goldy vulkan buffer.rs**: Added `TRANSFER_WRITE → COMPUTE_SHADER` memory
///    barriers after `cmd_fill_buffer` and `cmd_copy_buffer` in `Buffer::clear`.
///
fn many_bins() {
    let mut scene = Scene::new();
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        palette::css::RED,
        None,
        &Rect::new(-5., -5., 256. * 20., 256. * 20.),
    );
    let params = TestParams::new("many_bins", 256 * 17, 256 * 17);
    // To view, use EKRANO_DEBUG_TEST=many_bins
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

    // When #680 is fully fixed, this will become:
    // let drawn_bins = 17 /* x bins */ * 17 /* y bins*/;

    // The coarse shader caps bin_ix at 256 (see vello #680), so at most 256 of the
    // 289 bins render.  With the binning OOB fix all 256 should be correct.
    const MIN_RED_COUNT: u32 = 256 * 256 * 256;
    assert!(
        red_count >= MIN_RED_COUNT,
        "expected at least {MIN_RED_COUNT} red pixels, got {red_count}"
    );
    assert!(black_count > 0);
}

#[test]
fn many_bins_test() {
    many_bins();
}

/// Regression test for <https://github.com/linebender/vello/issues/1061>
/// (Fixed in ekrano by the `END_CLIP` draw-data encoding fix.)
#[test]
fn test_layer_size() {
    let mut scene = Scene::new();
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        ekrano::peniko::color::AlphaColor::from_rgb8(0, 255, 0),
        None,
        &Rect::from_origin_size((0.0, 0.0), (60., 60.)),
    );
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        ekrano::peniko::color::AlphaColor::from_rgb8(255, 0, 0),
        None,
        &Rect::from_origin_size((20.0, 20.0), (20., 20.)),
    );
    scene.push_layer(
        ekrano::peniko::Fill::NonZero,
        ekrano::peniko::Compose::Clear,
        1.0,
        Affine::IDENTITY,
        &Rect::from_origin_size((20.0, 20.0), (20., 20.)),
    );
    scene.pop_layer();
    // Compose::Clear makes the layer region transparent; compositing over white
    // makes the hole visible as white.  The reference was generated on a white-
    // surface renderer (vello-sparse), so we match that here.
    let mut params = TestParams::new("layer_size", 60, 60);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[ignore = "CI runs these tests on a CPU, leading to them having unrealistic precision"] // Uncomment below line when removing this.
#[should_panic]
fn test_data_image_roundtrip_extend_reflect() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Reflect);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[ignore = "CI runs these tests on a CPU, leading to them having unrealistic precision"] // Uncomment below line when removing this.
#[should_panic]
fn test_data_image_roundtrip_extend_repeat() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Repeat);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[test]
fn test_gradient_color_alpha() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.)).with_stops([
            ColorStop {
                offset: 0.,
                color: Color::from_rgba8(255, 255, 0, 0).into(),
            },
            ColorStop {
                offset: 1.,
                color: Color::from_rgba8(0, 0, 255, 255).into(),
            },
        ]),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// See <https://github.com/linebender/vello/issues/1198>
#[test]
fn clip_blends() {
    let mut scene = Scene::new();

    scene.fill(
        ekrano::peniko::Fill::EvenOdd,
        Affine::IDENTITY,
        palette::css::BLUE,
        None,
        &Rect::from_origin_size((0., 0.), (100., 100.)),
    );
    let layer_shape = Triangle::from_coords((50., 0.), (0., 100.), (100., 100.));
    scene.push_clip_layer(
        ekrano::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &layer_shape,
    );
    scene.push_layer(
        ekrano::peniko::Fill::NonZero,
        Mix::Multiply,
        1.0,
        Affine::IDENTITY,
        &layer_shape,
    );
    scene.fill(
        ekrano::peniko::Fill::EvenOdd,
        Affine::IDENTITY,
        palette::css::AQUAMARINE,
        None,
        &Rect::from_origin_size((0., 0.), (100., 100.)),
    );
    scene.pop_layer();
    scene.pop_layer();

    let params = TestParams::new("clip_blends", 100, 100);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
