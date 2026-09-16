// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text decorations with skip-ink (Vello #1592).
//!
//! Honesty gate: Linebender `main` sparse LFS PNGs
//! (`sparse_strips/vello_sparse_tests/snapshots/glyphs_decoration_*`, uncached).
//! GPU vs `vello_cpu` glyph AA; not self-rendered.

#![allow(clippy::cast_possible_truncation, clippy::allow_attributes_without_reason)]

#[path = "common/submission.rs"]
mod submission;

use std::f64::consts::FRAC_PI_4;

use ekrano::{
    Glyph, Scene,
    kurbo::Affine,
    peniko::{Fill, FontData, color::palette::css::REBECCA_PURPLE},
};
use ekrano_tests::{TestParams, snapshot_test_sync};
use scenes::SimpleText;

fn layout_roboto(text: &SimpleText, content: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    let font = text.roboto().clone();
    let glyphs = SimpleText::layout_glyphs(&font, font_size, &[], content);
    (font, glyphs)
}

fn render_decorated_text(
    scene: &mut Scene,
    text: &SimpleText,
    content: &str,
    font_size: f32,
    transform: Affine,
    glyph_transform: Option<Affine>,
    offset: f32,
    size: f32,
    buffer: f32,
) {
    let (font, glyphs) = layout_roboto(text, content, font_size);
    let paint = REBECCA_PURPLE;

    let mut fill_builder = scene
        .draw_glyphs(&font)
        .font_size(font_size)
        .transform(transform)
        .hint(false)
        .brush(paint);
    if let Some(gt) = glyph_transform {
        fill_builder = fill_builder.glyph_transform(Some(gt));
    }
    fill_builder.draw(Fill::NonZero, glyphs.iter().copied());

    let x_end = glyphs.last().map_or(0.0, |g| g.x + font_size * 0.6);
    let mut deco_builder = scene
        .draw_glyphs(&font)
        .font_size(font_size)
        .transform(transform)
        .hint(glyph_transform.is_none())
        .brush(paint);
    if let Some(gt) = glyph_transform {
        deco_builder = deco_builder.glyph_transform(Some(gt));
    }
    deco_builder.render_decoration(glyphs.into_iter(), 0.0..=x_end, 0.0, offset, size, buffer);
}

fn gate(name: &str, width: u32, height: u32, build: impl FnOnce(&mut Scene, &SimpleText)) {
    let mut scene = Scene::new();
    let text = SimpleText::new();
    build(&mut scene, &text);
    let mut params = TestParams::new(name, width, height);
    params.base_color = Some(ekrano::peniko::color::palette::css::WHITE);
    snapshot_test_sync(scene, &params).unwrap().assert_mean_less_than(0.01);
}

fn glyphs_decoration_offset_values() {
    gate("glyphs_decoration_offset_values", 300, 180, |scene, text| {
        let font_size = 30.0_f32;
        for (i, offset) in [-6.0_f32, -2.0, 0.0, 8.0, 15.0].iter().enumerate() {
            let y = 30.0 + (i as f64) * 32.0;
            render_decorated_text(
                scene,
                text,
                "Happy joyful",
                font_size,
                Affine::translate((0., y)),
                None,
                *offset,
                1.5,
                1.5,
            );
        }
    });
}

fn glyphs_decoration_size_values() {
    gate("glyphs_decoration_size_values", 180, 180, |scene, text| {
        let font_size = 30.0_f32;
        for (i, size) in [0.5_f32, 1.0, 2.0, 4.0].iter().enumerate() {
            let y = 30.0 + (i as f64) * 38.0;
            render_decorated_text(
                scene,
                text,
                "Happy joyful",
                font_size,
                Affine::translate((0., y)),
                None,
                -2.0,
                *size,
                1.5,
            );
        }
    });
}

fn glyphs_decoration_no_descenders() {
    gate("glyphs_decoration_no_descenders", 180, 70, |scene, text| {
        render_decorated_text(
            scene,
            text,
            "HELLO",
            50.0,
            Affine::translate((0., 50.)),
            None,
            -2.0,
            2.0,
            1.5,
        );
    });
}

fn glyphs_decoration_transformed() {
    gate("glyphs_decoration_transformed", 100, 150, |scene, text| {
        let content = "Happy";
        let rows: [(Affine, f32, Option<Affine>, f64); 4] = [
            (Affine::scale(2.0), 12.0, None, 30.0),
            (Affine::IDENTITY, 10.0, Some(Affine::scale(1.2)), 40.0),
            (
                Affine::scale_non_uniform(1.0, -1.0) * Affine::translate((0.0, 20.0)),
                20.0,
                None,
                10.0,
            ),
            (Affine::rotate(FRAC_PI_4), 12.0, None, 40.0),
        ];
        let mut y = 30.0;
        for (run_transform, font_size, glyph_transform, buffer) in rows {
            render_decorated_text(
                scene,
                text,
                content,
                font_size,
                Affine::translate((16.0, y)) * run_transform,
                glyph_transform,
                -1.0,
                1.0,
                1.0,
            );
            y += buffer;
        }
    });
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

    case!("glyphs_decoration_offset_values", glyphs_decoration_offset_values());
    case!("glyphs_decoration_size_values", glyphs_decoration_size_values());
    case!("glyphs_decoration_no_descenders", glyphs_decoration_no_descenders());
    case!("glyphs_decoration_transformed", glyphs_decoration_transformed());

    let args = libtest_mimic::Arguments::from_args();
    submission::run_gpu_snapshot_trials(args, trials);
}
