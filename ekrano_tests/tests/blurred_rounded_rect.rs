// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Inverse blurred rounded rect (Vello #1715 / classic #1718).
//!
//! Honesty gate: Linebender `main` sparse LFS PNGs
//! (`sparse_strips/vello_sparse_tests/snapshots/inverse_blurred_rounded_rect_*`).
//! Those goldens still use the #1715 inflated clip (`2.5σ` around the rect), not
//! the #1718 box clip. These tests match that clip via `_in`.
//!
//! GPU analytic kernel vs `vello_cpu` u16/AA: FLIP mean ~0.05–0.06 on these 100×100
//! frames (not self-rendered). 0.0095 is too tight; 0.07 covers the residual.

#[path = "common/submission.rs"]
mod submission;

use ekrano::{
    Scene,
    kurbo::{Affine, Point, Rect},
    peniko::color::palette::css::{REBECCA_PURPLE, WHITE},
};
use ekrano_tests::{TestParams, snapshot_test_sync};

fn inverse_rect_with(radius: f64, std_dev: f64, affine: Affine, name: &str) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    let kernel_size = 2.5 * std_dev;
    let shape = rect.inflate(kernel_size, kernel_size);
    let mut scene = Scene::new();
    scene.draw_blurred_rounded_rect_in(&shape, affine, rect, REBECCA_PURPLE, radius, std_dev, true);
    let mut params = TestParams::new(name, 100, 100);
    params.base_color = Some(WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.07);
}

fn inverse_blurred_rounded_rect_small_std_dev() {
    inverse_rect_with(0.0, 5.0, Affine::IDENTITY, "inverse_blurred_rounded_rect_small_std_dev");
}

fn inverse_blurred_rounded_rect_medium_std_dev() {
    inverse_rect_with(
        0.0,
        10.0,
        Affine::IDENTITY,
        "inverse_blurred_rounded_rect_medium_std_dev",
    );
}

fn inverse_blurred_rounded_rect_large_std_dev() {
    inverse_rect_with(
        0.0,
        20.0,
        Affine::IDENTITY,
        "inverse_blurred_rounded_rect_large_std_dev",
    );
}

fn inverse_blurred_rounded_rect_with_radius() {
    inverse_rect_with(10.0, 10.0, Affine::IDENTITY, "inverse_blurred_rounded_rect_with_radius");
}

fn inverse_blurred_rounded_rect_with_large_radius() {
    inverse_rect_with(
        30.0,
        10.0,
        Affine::IDENTITY,
        "inverse_blurred_rounded_rect_with_large_radius",
    );
}

fn inverse_blurred_rounded_rect_with_transform() {
    inverse_rect_with(
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        "inverse_blurred_rounded_rect_with_transform",
    );
}

fn main() {
    let mut trials = Vec::new();
    macro_rules! case {
        ($name:literal, $body:expr) => {{
            trials.push(libtest_mimic::Trial::test($name, move || {
                $body;
                Ok(())
            }));
        }};
    }

    case!(
        "inverse_blurred_rounded_rect_small_std_dev",
        inverse_blurred_rounded_rect_small_std_dev()
    );
    case!(
        "inverse_blurred_rounded_rect_medium_std_dev",
        inverse_blurred_rounded_rect_medium_std_dev()
    );
    case!(
        "inverse_blurred_rounded_rect_large_std_dev",
        inverse_blurred_rounded_rect_large_std_dev()
    );
    case!(
        "inverse_blurred_rounded_rect_with_radius",
        inverse_blurred_rounded_rect_with_radius()
    );
    case!(
        "inverse_blurred_rounded_rect_with_large_radius",
        inverse_blurred_rounded_rect_with_large_radius()
    );
    case!(
        "inverse_blurred_rounded_rect_with_transform",
        inverse_blurred_rounded_rect_with_transform()
    );

    let args = libtest_mimic::Arguments::from_args();
    submission::run_gpu_snapshot_trials(args, trials);
}
