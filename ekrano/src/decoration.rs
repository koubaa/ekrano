// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Skip-ink text decorations (Vello #1592 / Glifo `render_decoration`).
//!
//! Classic Ekrano does not depend on Glifo. This is the encode-time analog:
//! intersect glyph outlines with a decoration band and emit filled rects.

use std::ops::RangeInclusive;

use ekrano_encoding::{FontEmbolden, Glyph, NormalizedCoord, Transform};
use peniko::{
    FontData,
    kurbo::{Affine, BezPath, Line, ParamCurve as _, PathSeg, Point, Rect, Shape},
};
use skrifa::{
    GlyphId, MetadataProvider,
    instance::Size,
    outline::{DrawSettings, HintingInstance, HintingOptions, OutlinePen},
};

/// Same hinting options as the `ekrano_encoding` glyph cache, so skip-ink gaps
/// line up with hinted outline glyphs.
const HINTING_OPTIONS: HintingOptions = HintingOptions {
    engine: skrifa::outline::Engine::AutoFallback,
    target: skrifa::outline::Target::Smooth {
        mode: skrifa::outline::SmoothMode::Lcd,
        symmetric_rendering: false,
        preserve_linear_metrics: true,
    },
};

struct BezPathOutline(BezPath);

impl OutlinePen for BezPathOutline {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to(Point::new(x.into(), y.into()));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(Point::new(x.into(), y.into()));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0
            .quad_to(Point::new(cx0.into(), cy0.into()), Point::new(x.into(), y.into()));
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            Point::new(cx0.into(), cy0.into()),
            Point::new(cx1.into(), cy1.into()),
            Point::new(x.into(), y.into()),
        );
    }

    fn close(&mut self) {
        self.0.close_path();
    }
}

/// Rectangles for a skip-ink decoration in glyph-run (layout) space.
///
/// Apply the run transform when filling, matching Glifo's `fill_rect` under the
/// scene transform.
pub(crate) fn decoration_rects(
    font: &FontData,
    font_size: f32,
    font_embolden: FontEmbolden,
    glyph_transform: Option<Affine>,
    mut hint: bool,
    run_transform: Transform,
    coords: &[NormalizedCoord],
    glyphs: impl Iterator<Item = Glyph>,
    x_range: RangeInclusive<f32>,
    baseline_y: f32,
    offset: f32,
    size: f32,
    buffer: f32,
) -> Vec<Rect> {
    let mut outline_size = font_size;
    let mut outline_to_layout = 1.0_f64;
    if hint {
        // Match resolve: vertical-only hinting only for uniform scale, no skew.
        if run_transform.matrix[0] == run_transform.matrix[3]
            && run_transform.matrix[1] == 0.0
            && run_transform.matrix[2] == 0.0
            && run_transform.matrix[0] > 0.0
        {
            outline_to_layout = 1.0 / f64::from(run_transform.matrix[0]);
            outline_size *= run_transform.matrix[0];
        } else {
            hint = false;
        }
    }

    let Ok(font_ref) = skrifa::FontRef::from_index(font.data.as_ref(), font.index) else {
        return Vec::new();
    };
    let outlines = font_ref.outline_glyphs();
    let skrifa_coords: &[skrifa::instance::NormalizedCoord] = bytemuck::cast_slice(coords);

    let hinter = if hint {
        HintingInstance::new(&outlines, Size::new(outline_size), skrifa_coords, HINTING_OPTIONS).ok()
    } else {
        None
    };

    let outline_transform =
        glyph_transform.unwrap_or(Affine::IDENTITY) * Affine::FLIP_Y * Affine::scale(outline_to_layout);

    let buffer = f64::from(buffer);
    let x0 = f64::from(*x_range.start());
    let x1 = f64::from(*x_range.end());
    let layout_y0 = f64::from(-offset);
    let layout_y1 = f64::from(-offset + size);

    let mut exclusions = Vec::new();
    let mut path_buf = BezPathOutline(BezPath::new());

    for glyph in glyphs {
        let Some(outline) = outlines.get(GlyphId::new(glyph.id)) else {
            continue;
        };

        path_buf.0.truncate(0);
        let draw_settings = if let Some(hinter) = hinter.as_ref() {
            DrawSettings::hinted(hinter, false)
        } else {
            DrawSettings::unhinted(Size::new(outline_size), skrifa_coords)
        };
        if outline.draw(draw_settings, &mut path_buf).is_err() {
            continue;
        }
        let expanded = (font_embolden.amount != peniko::kurbo::Diagonal2::new(0.0, 0.0)).then(|| {
            peniko::kurbo::expand_path(
                &path_buf.0,
                font_embolden.amount,
                font_embolden.join,
                font_embolden.miter_limit,
                font_embolden.tolerance,
            )
        });
        let path = expanded.as_ref().unwrap_or(&path_buf.0);
        let bbox = path.bounding_box();

        let [_, b, _, d, _, f] = outline_transform.as_coeffs();
        let (y_min, y_max) = {
            let bx0 = b * bbox.x0;
            let bx1 = b * bbox.x1;
            let dy0 = d * bbox.y0;
            let dy1 = d * bbox.y1;
            (f + bx0.min(bx1) + dy0.min(dy1), f + bx0.max(bx1) + dy0.max(dy1))
        };
        if y_max < layout_y0 || y_min > layout_y1 {
            continue;
        }

        let mut rect = Rect {
            x0: f64::INFINITY,
            x1: f64::NEG_INFINITY,
            y0: layout_y0,
            y1: layout_y1,
        };
        for seg in path.segments() {
            expand_rect_with_segment(&mut rect, outline_transform * seg, layout_y0..=layout_y1);
        }

        let excl_start = (rect.x0 + f64::from(glyph.x) - buffer).max(x0);
        let excl_end = (rect.x1 + f64::from(glyph.x) + buffer).min(x1);
        if excl_start >= excl_end {
            continue;
        }
        insert_and_merge_range(&mut exclusions, excl_start, excl_end);
    }

    let y0 = f64::from(baseline_y) + layout_y0;
    let y1 = f64::from(baseline_y) + layout_y1;
    let mut rects = Vec::new();
    let mut current_x = x0;
    for (excl_start, excl_end) in exclusions {
        let rect = Rect::new(current_x, y0, excl_start, y1);
        if rect.width() > 0.0 {
            rects.push(rect);
        }
        current_x = excl_end;
    }
    let trailing = Rect::new(current_x, y0, x1, y1);
    if trailing.width() > 0.0 {
        rects.push(trailing);
    }
    rects
}

fn insert_and_merge_range(ranges: &mut Vec<(f64, f64)>, start: f64, end: f64) {
    let insert_pos = ranges.iter().rposition(|r| r.0 <= start).map_or(0, |i| i + 1);
    let merge_start = insert_pos
        .checked_sub(1)
        .filter(|&i| ranges[i].1 >= start)
        .unwrap_or(insert_pos);
    let new_end = ranges[merge_start..]
        .iter()
        .take_while(|(s, _)| *s <= end)
        .fold(end, |acc, (_, e)| acc.max(*e));
    let merge_end = merge_start + ranges[merge_start..].iter().take_while(|(s, _)| *s <= new_end).count();
    if merge_start < merge_end {
        let new_start = start.min(ranges[merge_start].0);
        ranges.splice(merge_start..merge_end, [(new_start, new_end)]);
    } else {
        ranges.insert(insert_pos, (start, end));
    }
}

fn expand_rect_with_segment(rect: &mut Rect, seg: PathSeg, y_span: RangeInclusive<f64>) {
    let (mut x_bounds, y_bounds) = match seg {
        PathSeg::Line(line) => (
            (line.p0.x.min(line.p1.x), line.p0.x.max(line.p1.x)),
            (line.p0.y.min(line.p1.y), line.p0.y.max(line.p1.y)),
        ),
        PathSeg::Quad(quad) => (
            (
                quad.p0.x.min(quad.p1.x).min(quad.p2.x),
                quad.p0.x.max(quad.p1.x).max(quad.p2.x),
            ),
            (
                quad.p0.y.min(quad.p1.y).min(quad.p2.y),
                quad.p0.y.max(quad.p1.y).max(quad.p2.y),
            ),
        ),
        PathSeg::Cubic(cubic) => (
            (
                cubic.p0.x.min(cubic.p1.x).min(cubic.p2.x).min(cubic.p3.x),
                cubic.p0.x.max(cubic.p1.x).max(cubic.p2.x).max(cubic.p3.x),
            ),
            (
                cubic.p0.y.min(cubic.p1.y).min(cubic.p2.y).min(cubic.p3.y),
                cubic.p0.y.max(cubic.p1.y).max(cubic.p2.y).max(cubic.p3.y),
            ),
        ),
    };
    if y_bounds.1 < *y_span.start() || y_bounds.0 > *y_span.end() {
        return;
    }

    x_bounds.0 -= 1.0;
    x_bounds.1 += 1.0;
    let top_line = Line::new((x_bounds.0, *y_span.start()), (x_bounds.1, *y_span.start()));
    let bottom_line = Line::new((x_bounds.0, *y_span.end()), (x_bounds.1, *y_span.end()));

    for intersection in seg.intersect_line(top_line) {
        let point = top_line.eval(intersection.line_t);
        rect.x0 = rect.x0.min(point.x);
        rect.x1 = rect.x1.max(point.x);
    }
    for intersection in seg.intersect_line(bottom_line) {
        let point = bottom_line.eval(intersection.line_t);
        rect.x0 = rect.x0.min(point.x);
        rect.x1 = rect.x1.max(point.x);
    }

    let (seg_start, seg_end) = match seg {
        PathSeg::Line(line) => (line.p0, line.p1),
        PathSeg::Quad(quad) => (quad.p0, quad.p2),
        PathSeg::Cubic(cubic) => (cubic.p0, cubic.p3),
    };
    for point in [seg_start, seg_end] {
        if (*y_span.start()..=*y_span.end()).contains(&point.y) {
            rect.x0 = rect.x0.min(point.x);
            rect.x1 = rect.x1.max(point.x);
        }
    }
}
