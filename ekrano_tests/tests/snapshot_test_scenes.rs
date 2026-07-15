// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests using the test scenes from [`scenes`].

#[path = "common/submission.rs"]
mod submission;

use ekrano_tests::{TestBackend, TestParams, encode_test_scene, snapshot_test_sync, shared_test_device};
use scenes::{ExampleScene, test_scenes};

/// Snapshot each scene against the LFS reference PNG.
fn snapshot_test_scene(test_scene: ExampleScene, mut params: TestParams) {
    let scene = encode_test_scene(test_scene, &mut params);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.0095);
}

fn snapshot_splash_body(backend: TestBackend) {
    let test_scene = test_scenes::splash_with_tiger();
    let params = TestParams::new("splash", 300, 300).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_splash() {
    snapshot_splash_body(TestBackend::Classic);
}

fn scheme_snapshot_splash() {
    snapshot_splash_body(TestBackend::Scheme);
}

fn snapshot_funky_paths_body(backend: TestBackend) {
    let test_scene = test_scenes::funky_paths();
    let params = TestParams::new("funky_paths", 600, 600).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_funky_paths() {
    snapshot_funky_paths_body(TestBackend::Classic);
}

fn scheme_snapshot_funky_paths() {
    snapshot_funky_paths_body(TestBackend::Scheme);
}

fn snapshot_stroke_styles_body(backend: TestBackend) {
    let test_scene = test_scenes::stroke_styles();
    let params = TestParams::new("stroke_styles", 600, 425).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_stroke_styles() {
    snapshot_stroke_styles_body(TestBackend::Classic);
}

fn scheme_snapshot_stroke_styles() {
    snapshot_stroke_styles_body(TestBackend::Scheme);
}

fn snapshot_stroke_styles_non_uniform_body(backend: TestBackend) {
    let test_scene = test_scenes::stroke_styles_non_uniform();
    let params = TestParams::new("stroke_styles_non_uniform", 600, 425).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_stroke_styles_non_uniform() {
    snapshot_stroke_styles_non_uniform_body(TestBackend::Classic);
}

fn scheme_snapshot_stroke_styles_non_uniform() {
    snapshot_stroke_styles_non_uniform_body(TestBackend::Scheme);
}

fn snapshot_stroke_styles_skew_body(backend: TestBackend) {
    let test_scene = test_scenes::stroke_styles_skew();
    let params = TestParams::new("stroke_styles_skew", 600, 425).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_stroke_styles_skew() {
    snapshot_stroke_styles_skew_body(TestBackend::Classic);
}

fn scheme_snapshot_stroke_styles_skew() {
    snapshot_stroke_styles_skew_body(TestBackend::Scheme);
}

fn snapshot_tricky_strokes_body(backend: TestBackend) {
    let test_scene = test_scenes::tricky_strokes();
    let params = TestParams::new("tricky_strokes", 600, 425).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_tricky_strokes() {
    snapshot_tricky_strokes_body(TestBackend::Classic);
}

fn scheme_snapshot_tricky_strokes() {
    snapshot_tricky_strokes_body(TestBackend::Scheme);
}

fn snapshot_fill_types_body(backend: TestBackend) {
    let test_scene = test_scenes::fill_types();
    let params = TestParams::new("fill_types", 700, 350).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_fill_types() {
    snapshot_fill_types_body(TestBackend::Classic);
}

fn scheme_snapshot_fill_types() {
    snapshot_fill_types_body(TestBackend::Scheme);
}

fn snapshot_deep_blend_body(backend: TestBackend) {
    let test_scene = test_scenes::deep_blend();
    let params = TestParams::new("deep_blend", 200, 200).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_deep_blend() {
    snapshot_deep_blend_body(TestBackend::Classic);
}

fn scheme_snapshot_deep_blend() {
    snapshot_deep_blend_body(TestBackend::Scheme);
}

fn snapshot_gradient_extend_body(backend: TestBackend) {
    let test_scene = test_scenes::gradient_extend();
    let params = TestParams::new("gradient_extend", 200, 200).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_gradient_extend() {
    snapshot_gradient_extend_body(TestBackend::Classic);
}

fn scheme_snapshot_gradient_extend() {
    snapshot_gradient_extend_body(TestBackend::Scheme);
}

fn snapshot_many_clips_body(backend: TestBackend) {
    let test_scene = test_scenes::many_clips();
    let params = TestParams::new("many_clips", 200, 200).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_many_clips() {
    snapshot_many_clips_body(TestBackend::Classic);
}

fn scheme_snapshot_many_clips() {
    snapshot_many_clips_body(TestBackend::Scheme);
}

fn snapshot_clip_test_body(backend: TestBackend) {
    let test_scene = test_scenes::clip_test();
    let params = TestParams::new("clip_test", 512, 768).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_clip_test() {
    snapshot_clip_test_body(TestBackend::Classic);
}

fn scheme_snapshot_clip_test() {
    snapshot_clip_test_body(TestBackend::Scheme);
}

fn snapshot_blurred_rounded_rect_body(backend: TestBackend) {
    let test_scene = test_scenes::blurred_rounded_rect();
    let params = TestParams::new("blurred_rounded_rect", 400, 400).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_blurred_rounded_rect() {
    snapshot_blurred_rounded_rect_body(TestBackend::Classic);
}

fn scheme_snapshot_blurred_rounded_rect() {
    snapshot_blurred_rounded_rect_body(TestBackend::Scheme);
}

fn snapshot_longpathdash_butt_body(backend: TestBackend) {
    let test_scene = test_scenes::longpathdash_butt();
    let params = TestParams::new("longpathdash_butt", 440, 80).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_longpathdash_butt() {
    snapshot_longpathdash_butt_body(TestBackend::Classic);
}

fn scheme_snapshot_longpathdash_butt() {
    snapshot_longpathdash_butt_body(TestBackend::Scheme);
}

fn snapshot_image_sampling_body(backend: TestBackend) {
    let test_scene = test_scenes::image_sampling();
    let params = TestParams::new("image_sampling", 400, 400).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_image_sampling() {
    snapshot_image_sampling_body(TestBackend::Classic);
}

fn scheme_snapshot_image_sampling() {
    snapshot_image_sampling_body(TestBackend::Scheme);
}

fn snapshot_image_extend_modes_bilinear_body(backend: TestBackend) {
    let test_scene = test_scenes::image_extend_modes_bilinear();
    let params = TestParams::new("image_extend_modes_bilinear", 400, 400).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_image_extend_modes_bilinear() {
    snapshot_image_extend_modes_bilinear_body(TestBackend::Classic);
}

fn scheme_snapshot_image_extend_modes_bilinear() {
    snapshot_image_extend_modes_bilinear_body(TestBackend::Scheme);
}

fn snapshot_image_extend_modes_nearest_neighbor_body(backend: TestBackend) {
    let test_scene = test_scenes::image_extend_modes_nearest_neighbor();
    let params = TestParams::new("image_extend_modes_nearest_neighbor", 400, 400).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_image_extend_modes_nearest_neighbor() {
    snapshot_image_extend_modes_nearest_neighbor_body(TestBackend::Classic);
}

fn scheme_snapshot_image_extend_modes_nearest_neighbor() {
    snapshot_image_extend_modes_nearest_neighbor_body(TestBackend::Scheme);
}

fn snapshot_luminance_mask_body(backend: TestBackend) {
    let test_scene = test_scenes::luminance_mask();
    // This has been manually validated to match the example in
    // https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/mask-type
    let params = TestParams::new("luminance_mask", 55, 55).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn snapshot_luminance_mask() {
    snapshot_luminance_mask_body(TestBackend::Classic);
}

fn scheme_snapshot_luminance_mask() {
    snapshot_luminance_mask_body(TestBackend::Scheme);
}

fn image_luminance_mask_body(backend: TestBackend) {
    let test_scene = test_scenes::image_luminance_mask();
    let params = TestParams::new("image_luminance_mask", 350, 250).with_backend(backend);
    snapshot_test_scene(test_scene, params);
}
fn image_luminance_mask() {
    image_luminance_mask_body(TestBackend::Classic);
}

fn scheme_image_luminance_mask() {
    image_luminance_mask_body(TestBackend::Scheme);
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
        libtest_mimic::Trial::test("scheme_snapshot_splash", || {
            scheme_snapshot_splash();
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
        libtest_mimic::Trial::test("scheme_snapshot_funky_paths", || {
            scheme_snapshot_funky_paths();
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
        libtest_mimic::Trial::test("scheme_snapshot_stroke_styles", || {
            scheme_snapshot_stroke_styles();
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
        libtest_mimic::Trial::test("scheme_snapshot_stroke_styles_non_uniform", || {
            scheme_snapshot_stroke_styles_non_uniform();
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
        libtest_mimic::Trial::test("scheme_snapshot_stroke_styles_skew", || {
            scheme_snapshot_stroke_styles_skew();
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
        libtest_mimic::Trial::test("scheme_snapshot_tricky_strokes", || {
            scheme_snapshot_tricky_strokes();
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
        libtest_mimic::Trial::test("scheme_snapshot_fill_types", || {
            scheme_snapshot_fill_types();
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
        libtest_mimic::Trial::test("scheme_snapshot_deep_blend", || {
            scheme_snapshot_deep_blend();
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
        libtest_mimic::Trial::test("scheme_snapshot_gradient_extend", || {
            scheme_snapshot_gradient_extend();
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
        libtest_mimic::Trial::test("scheme_snapshot_many_clips", || {
            scheme_snapshot_many_clips();
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
        libtest_mimic::Trial::test("scheme_snapshot_clip_test", || {
            scheme_snapshot_clip_test();
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
        libtest_mimic::Trial::test("scheme_snapshot_blurred_rounded_rect", || {
            scheme_snapshot_blurred_rounded_rect();
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
        libtest_mimic::Trial::test("scheme_snapshot_longpathdash_butt", || {
            scheme_snapshot_longpathdash_butt();
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
        libtest_mimic::Trial::test("scheme_snapshot_image_sampling", || {
            scheme_snapshot_image_sampling();
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
        libtest_mimic::Trial::test("scheme_snapshot_image_extend_modes_bilinear", || {
            scheme_snapshot_image_extend_modes_bilinear();
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
        libtest_mimic::Trial::test("scheme_snapshot_image_extend_modes_nearest_neighbor", || {
            scheme_snapshot_image_extend_modes_nearest_neighbor();
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
        libtest_mimic::Trial::test("scheme_snapshot_luminance_mask", || {
            scheme_snapshot_luminance_mask();
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
    trials.push(
        libtest_mimic::Trial::test("scheme_image_luminance_mask", || {
            scheme_image_luminance_mask();
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
