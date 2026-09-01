// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests using the test scenes from [`scenes`].

#[path = "common/submission.rs"]
mod submission;

use ekrano_tests::{TestParams, encode_test_scene, shared_test_device, snapshot_test_sync};
use scenes::{ExampleScene, test_scenes};

/// Snapshot each scene against the LFS reference PNG.
fn snapshot_test_scene(test_scene: ExampleScene, mut params: TestParams) {
    let scene = encode_test_scene(test_scene, &mut params);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}

fn snapshot_splash() {
    let test_scene = test_scenes::splash_with_tiger();
    let params = TestParams::new("splash", 300, 300);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_funky_paths() {
    let test_scene = test_scenes::funky_paths();
    let params = TestParams::new("funky_paths", 600, 600);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_stroke_styles() {
    let test_scene = test_scenes::stroke_styles();
    let params = TestParams::new("stroke_styles", 600, 425);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_stroke_styles_non_uniform() {
    let test_scene = test_scenes::stroke_styles_non_uniform();
    let params = TestParams::new("stroke_styles_non_uniform", 600, 425);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_stroke_styles_skew() {
    let test_scene = test_scenes::stroke_styles_skew();
    let params = TestParams::new("stroke_styles_skew", 600, 425);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_tricky_strokes() {
    let test_scene = test_scenes::tricky_strokes();
    let params = TestParams::new("tricky_strokes", 600, 425);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_fill_types() {
    let test_scene = test_scenes::fill_types();
    let params = TestParams::new("fill_types", 700, 350);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_deep_blend() {
    let test_scene = test_scenes::deep_blend();
    let params = TestParams::new("deep_blend", 200, 200);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_gradient_extend() {
    let test_scene = test_scenes::gradient_extend();
    let params = TestParams::new("gradient_extend", 200, 200);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_many_clips() {
    let test_scene = test_scenes::many_clips();
    let params = TestParams::new("many_clips", 200, 200);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_clip_test() {
    let test_scene = test_scenes::clip_test();
    let params = TestParams::new("clip_test", 512, 768);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_blurred_rounded_rect() {
    let test_scene = test_scenes::blurred_rounded_rect();
    let params = TestParams::new("blurred_rounded_rect", 400, 400);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_longpathdash_butt() {
    let test_scene = test_scenes::longpathdash_butt();
    let params = TestParams::new("longpathdash_butt", 440, 80);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_dashed_curves() {
    let test_scene = test_scenes::dashed_curves();
    let params = TestParams::new("dashed_curves", 480, 240);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_image_sampling() {
    let test_scene = test_scenes::image_sampling();
    let params = TestParams::new("image_sampling", 400, 400);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_image_sampling_bicubic() {
    let test_scene = test_scenes::image_sampling_bicubic();
    let params = TestParams::new("image_sampling_bicubic", 520, 336);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_image_extend_modes_bilinear() {
    let test_scene = test_scenes::image_extend_modes_bilinear();
    let params = TestParams::new("image_extend_modes_bilinear", 400, 400);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_image_extend_modes_nearest_neighbor() {
    let test_scene = test_scenes::image_extend_modes_nearest_neighbor();
    let params = TestParams::new("image_extend_modes_nearest_neighbor", 400, 400);
    snapshot_test_scene(test_scene, params);
}

fn snapshot_luminance_mask() {
    let test_scene = test_scenes::luminance_mask();
    // This has been manually validated to match the example in
    // https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/mask-type
    let params = TestParams::new("luminance_mask", 55, 55);
    snapshot_test_scene(test_scene, params);
}

fn image_luminance_mask() {
    let test_scene = test_scenes::image_luminance_mask();
    let params = TestParams::new("image_luminance_mask", 350, 250);
    snapshot_test_scene(test_scene, params);
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("snapshot_splash", || {
            snapshot_splash();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_funky_paths", || {
            snapshot_funky_paths();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_stroke_styles", || {
            snapshot_stroke_styles();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_stroke_styles_non_uniform", || {
            snapshot_stroke_styles_non_uniform();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_stroke_styles_skew", || {
            snapshot_stroke_styles_skew();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_tricky_strokes", || {
            snapshot_tricky_strokes();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_fill_types", || {
            snapshot_fill_types();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_deep_blend", || {
            snapshot_deep_blend();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_gradient_extend", || {
            snapshot_gradient_extend();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_many_clips", || {
            snapshot_many_clips();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_clip_test", || {
            snapshot_clip_test();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_blurred_rounded_rect", || {
            snapshot_blurred_rounded_rect();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_longpathdash_butt", || {
            snapshot_longpathdash_butt();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_dashed_curves", || {
            snapshot_dashed_curves();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_image_sampling", || {
            snapshot_image_sampling();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_image_sampling_bicubic", || {
            snapshot_image_sampling_bicubic();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_image_extend_modes_bilinear", || {
            snapshot_image_extend_modes_bilinear();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_image_extend_modes_nearest_neighbor", || {
            snapshot_image_extend_modes_nearest_neighbor();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("snapshot_luminance_mask", || {
            snapshot_luminance_mask();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("image_luminance_mask", || {
            image_luminance_mask();
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
