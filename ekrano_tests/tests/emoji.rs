// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for Emoji [`scenes`].

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(clippy::cast_possible_truncation, clippy::allow_attributes_without_reason)]

#[cfg(target_os = "macos")]
use ekrano::peniko::color::palette;
#[cfg(target_os = "macos")]
use ekrano::peniko::{Blob, Brush, FontData};
use ekrano::{Scene, kurbo::Affine, peniko::Fill};
use ekrano_tests::{TestBackend, TestParams, snapshot_test_sync};
use scenes::SimpleText;
#[cfg(target_os = "macos")]
use std::sync::Arc;

fn encode_noto_colr(text: &str, font_size: f32) -> Scene {
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    simple_text.add_colr_emoji_run(
        &mut scene,
        font_size,
        Affine::translate((0., f64::from(font_size))),
        None,
        Fill::EvenOdd,
        text,
    );
    scene
}

fn encode_noto_bitmap(text: &str, font_size: f32) -> Scene {
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    simple_text.add_bitmap_emoji_run(
        &mut scene,
        font_size,
        Affine::translate((0., f64::from(font_size))),
        None,
        Fill::EvenOdd,
        text,
    );
    scene
}

#[cfg(target_os = "macos")]
fn encode_apple_bitmap(text: &str, font_size: f32) -> Scene {
    let font = FontData::new(
        Blob::new(Arc::new(
            std::fs::read("/System/Library/Fonts/Apple Color Emoji.ttc").unwrap(),
        )),
        0,
    );
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    simple_text.add_var_run(
        &mut scene,
        Some(&font),
        font_size,
        &[],
        // This should be unused
        &Brush::Solid(palette::css::WHITE),
        Affine::translate((0., f64::from(font_size))),
        None,
        Fill::EvenOdd,
        text,
        false,
    );
    scene
}

/// The Emoji supported by our font subset.
const TEXT: &str = "✅👀🎉🤠";

fn big_colr_body(backend: TestBackend) {
    let font_size = 48.;
    let scene = encode_noto_colr(TEXT, font_size);
    let params = TestParams::new(
        "big_colr",
        (font_size * 10.) as _,
        // Noto Emoji seem to be about 25% bigger than the actual font_size suggests
        (font_size * 1.25).ceil() as _,
    )
    .with_backend(backend);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.002);
}
#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn big_colr() {
    big_colr_body(TestBackend::Classic);
}

#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn scheme_big_colr() {
    big_colr_body(TestBackend::Scheme);
}

fn little_colr_body(backend: TestBackend) {
    let font_size = 10.;
    let scene = encode_noto_colr(TEXT, font_size);
    let params = TestParams::new("little_colr", (font_size * 10.) as _, (font_size * 1.25).ceil() as _)
        .with_backend(backend);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.005);
}
#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn little_colr() {
    little_colr_body(TestBackend::Classic);
}

#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn scheme_little_colr() {
    little_colr_body(TestBackend::Scheme);
}

fn colr_undef_body(backend: TestBackend) {
    let font_size = 10.;
    // This emoji isn't in the subset we have made
    let scene = encode_noto_colr("🤷", font_size);
    let params = TestParams::new("colr_undef", (font_size * 10.) as _, (font_size * 1.25).ceil() as _)
        .with_backend(backend);
    // TODO: Work out why the undef glyph is nothing - is it an issue with our font subset or with our renderer?
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn colr_undef() {
    colr_undef_body(TestBackend::Classic);
}

#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn scheme_colr_undef() {
    colr_undef_body(TestBackend::Scheme);
}

// Bitmap emoji compositing can differ slightly between DX12 WARP and hardware GPUs
// (±1–2 RGB on opaque glyph pixels; FLIP mean ~0.0011 on WARP vs ~0.0009 on hardware).
// CI runs on WARP; regenerate the snapshot with `EKRANO_TEST_UPDATE=big_bitmap` on WARP if needed.
fn big_bitmap_body(backend: TestBackend) {
    let font_size = 48.;
    let scene = encode_noto_bitmap(TEXT, font_size);
    let params = TestParams::new("big_bitmap", (font_size * 10.) as _, (font_size * 1.25).ceil() as _)
        .with_backend(backend);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn big_bitmap() {
    big_bitmap_body(TestBackend::Classic);
}

#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn scheme_big_bitmap() {
    big_bitmap_body(TestBackend::Scheme);
}

#[cfg(target_os = "macos")]
fn big_bitmap_apple_body(backend: TestBackend) {
    let font_size = 48.;
    let scene = encode_apple_bitmap(TEXT, font_size);
    let params = TestParams::new(
        "big_bitmap_apple",
        (font_size * 10.) as _,
        (font_size * 1.25).ceil() as _,
    )
    .with_backend(backend);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[cfg(target_os = "macos")]
#[test]
fn big_bitmap_apple() {
    big_bitmap_apple_body(TestBackend::Classic);
}

#[cfg(target_os = "macos")]
#[test]
fn scheme_big_bitmap_apple() {
    big_bitmap_apple_body(TestBackend::Scheme);
}

fn little_bitmap_body(backend: TestBackend) {
    let font_size = 10.;
    let scene = encode_noto_bitmap(TEXT, font_size);
    let params = TestParams::new("little_bitmap", (font_size * 10.) as _, (font_size * 1.25).ceil() as _)
        .with_backend(backend);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn little_bitmap() {
    little_bitmap_body(TestBackend::Classic);
}

#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn scheme_little_bitmap() {
    little_bitmap_body(TestBackend::Scheme);
}

fn bitmap_undef_body(backend: TestBackend) {
    let font_size = 10.;
    // This emoji isn't in the subset we have made
    let scene = encode_noto_bitmap("🤷", font_size);
    let params = TestParams::new("bitmap_undef", (font_size * 10.) as _, (font_size * 1.25).ceil() as _)
        .with_backend(backend);
    // TODO: Work out why the undef glyph is nothing - is it an issue with our font subset or with our renderer?
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.001);
}
#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn bitmap_undef() {
    bitmap_undef_body(TestBackend::Classic);
}

#[cfg_attr(skip_slow_tests, ignore)]
#[test]
fn scheme_bitmap_undef() {
    bitmap_undef_body(TestBackend::Scheme);
}
