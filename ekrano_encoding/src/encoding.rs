// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{DrawBeginClip, FilterPrimitive, LayerFilterEffect};

use super::{
    CoverageMask, DrawBlurRoundedRect, DrawColor, DrawImage, DrawLinearGradient,
    DrawRadialGradient, DrawSweepGradient, DrawTag, Glyph, GlyphRun, NormalizedCoord, Patch,
    PathEncoder, PathTag, Style, Transform,
};

use peniko::color::{DynamicColor, palette};
use peniko::kurbo::{Shape, Stroke};
use peniko::{
    BlendMode, BrushRef, ColorStop, Extend, Fill, GradientKind, ImageBrushRef, ImageSampler,
    LinearGradientPosition, RadialGradientPosition, SweepGradientPosition,
};

/// Encoded data streams for a scene.
///
/// # Invariants
///
/// * At least one transform and style must be encoded before any path data
///   or draw object.
#[derive(Clone, Default)]
pub struct Encoding {
    /// The path tag stream.
    pub path_tags: Vec<PathTag>,
    /// The path data stream.
    /// Stores all coordinates on paths.
    /// Stored as `u32` as all comparisons are performed bitwise.
    pub path_data: Vec<u32>,
    /// The draw tag stream.
    pub draw_tags: Vec<DrawTag>,
    /// The draw data stream.
    pub draw_data: Vec<u32>,
    /// The transform stream.
    pub transforms: Vec<Transform>,
    /// The style stream
    pub styles: Vec<Style>,
    /// Late bound resource data.
    pub resources: Resources,
    /// Number of encoded paths.
    pub n_paths: u32,
    /// Number of encoded path segments.
    pub n_path_segments: u32,
    /// Number of encoded clips/layers.
    pub n_clips: u32,
    /// Number of unclosed clips/layers.
    pub n_open_clips: u32,
    /// Flags that capture the current state of the encoding.
    pub flags: u32,
    /// Optional full-frame mask sampled during fine rasterization (must match render size).
    pub coverage_mask: Option<CoverageMask>,
    /// If set, the next [`Self::encode_begin_clip`] associates this filter with that layer.
    pending_layer_filter: Option<FilterPrimitive>,
    /// Stack parallel to open clips: `Some` when the matching `BEGIN_CLIP` had a pending filter.
    clip_filter_stack: Vec<Option<FilterPrimitive>>,
    /// Parallel to open clips. Stores `(parameters, layer_idx_slot)` for each open
    /// `BEGIN_CLIP`; `layer_idx_slot` is the `u32` index into [`Self::draw_data`] where
    /// `encode_end_clip` backfills the matching filter's layer index (zeroed until then).
    begin_clip_stack: Vec<(DrawBeginClip, usize)>,
    /// Filters recorded when a filtered layer ends (`encode_end_clip`).
    ///
    /// Applied after fine: each entry is filtered in isolation, then composited back (see `ekrano`).
    pub layer_filter_effects: Vec<LayerFilterEffect>,
}

impl Encoding {
    /// Forces encoding of the next transform even if it matches
    /// the current transform in the stream.
    pub const FORCE_NEXT_TRANSFORM: u32 = 1;

    /// Forces encoding of the next style even if it matches
    /// the current style in the stream.
    pub const FORCE_NEXT_STYLE: u32 = 2;

    /// Creates a new encoding.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.path_tags.is_empty()
    }

    #[doc(alias = "clear")]
    // This is not called "clear" because "clear" has other implications
    // in graphics contexts.
    /// Clears the encoding.
    pub fn reset(&mut self) {
        self.transforms.clear();
        self.path_tags.clear();
        self.path_data.clear();
        self.styles.clear();
        self.draw_data.clear();
        self.draw_tags.clear();
        self.n_paths = 0;
        self.n_path_segments = 0;
        self.n_clips = 0;
        self.n_open_clips = 0;
        self.flags = 0;
        self.coverage_mask = None;
        self.pending_layer_filter = None;
        self.clip_filter_stack.clear();
        self.begin_clip_stack.clear();
        self.layer_filter_effects.clear();
        self.resources.reset();
    }

    /// Appends another encoding to this one with an optional transform.
    pub fn append(&mut self, other: &Self, transform: &Option<Transform>) {
        let glyph_runs_base = {
            let offsets = self.stream_offsets();
            let stops_base = self.resources.color_stops.len();
            let glyph_runs_base = self.resources.glyph_runs.len();
            let glyphs_base = self.resources.glyphs.len();
            let coords_base = self.resources.normalized_coords.len();
            self.resources
                .glyphs
                .extend_from_slice(&other.resources.glyphs);
            self.resources
                .normalized_coords
                .extend_from_slice(&other.resources.normalized_coords);
            self.resources
                .glyph_runs
                .extend(other.resources.glyph_runs.iter().cloned().map(|mut run| {
                    run.glyphs.start += glyphs_base;
                    run.glyphs.end += glyphs_base;
                    run.normalized_coords.start += coords_base;
                    run.normalized_coords.end += coords_base;
                    run.stream_offsets.path_tags += offsets.path_tags;
                    run.stream_offsets.path_data += offsets.path_data;
                    run.stream_offsets.draw_tags += offsets.draw_tags;
                    run.stream_offsets.draw_data += offsets.draw_data;
                    run.stream_offsets.transforms += offsets.transforms;
                    run.stream_offsets.styles += offsets.styles;
                    run
                }));
            self.resources
                .patches
                .extend(other.resources.patches.iter().map(|patch| match patch {
                    Patch::Ramp {
                        draw_data_offset: offset,
                        stops,
                        extend,
                    } => {
                        let stops = stops.start + stops_base..stops.end + stops_base;
                        Patch::Ramp {
                            draw_data_offset: offset + offsets.draw_data,
                            stops,
                            extend: *extend,
                        }
                    }
                    Patch::GlyphRun { index } => Patch::GlyphRun {
                        index: index + glyph_runs_base,
                    },
                    Patch::Image {
                        image,
                        draw_data_offset,
                    } => Patch::Image {
                        image: image.clone(),
                        draw_data_offset: *draw_data_offset + offsets.draw_data,
                    },
                }));
            self.resources
                .color_stops
                .extend_from_slice(&other.resources.color_stops);
            glyph_runs_base
        };
        self.path_tags.extend_from_slice(&other.path_tags);
        self.path_data.extend_from_slice(&other.path_data);
        self.draw_tags.extend_from_slice(&other.draw_tags);
        self.draw_data.extend_from_slice(&other.draw_data);
        self.n_paths += other.n_paths;
        self.n_path_segments += other.n_path_segments;
        self.n_clips += other.n_clips;
        self.n_open_clips += other.n_open_clips;
        self.flags = other.flags;
        if other.coverage_mask.is_some() {
            self.coverage_mask = other.coverage_mask.clone();
        }
        self.pending_layer_filter = other.pending_layer_filter.clone();
        self.clip_filter_stack
            .extend_from_slice(&other.clip_filter_stack);
        self.begin_clip_stack
            .extend_from_slice(&other.begin_clip_stack);
        self.layer_filter_effects
            .extend_from_slice(&other.layer_filter_effects);
        if let Some(transform) = *transform {
            self.transforms
                .extend(other.transforms.iter().map(|x| transform * *x));
            for run in &mut self.resources.glyph_runs[glyph_runs_base..] {
                run.transform = transform * run.transform;
            }
        } else {
            self.transforms.extend_from_slice(&other.transforms);
        }
        self.styles.extend_from_slice(&other.styles);
    }

    /// Returns a snapshot of the current stream offsets.
    pub fn stream_offsets(&self) -> StreamOffsets {
        StreamOffsets {
            path_tags: self.path_tags.len(),
            path_data: self.path_data.len(),
            draw_tags: self.draw_tags.len(),
            draw_data: self.draw_data.len(),
            transforms: self.transforms.len(),
            styles: self.styles.len(),
        }
    }

    /// Encodes a fill style.
    pub fn encode_fill_style(&mut self, fill: Fill) {
        self.encode_style(Style::from_fill(fill));
    }

    /// Encodes a stroke style.
    ///
    /// Returns false if the stroke had zero width and so couldn't be encoded.
    #[must_use]
    pub fn encode_stroke_style(&mut self, stroke: &Stroke) -> bool {
        let style = Style::from_stroke(stroke);
        if let Some(style) = style {
            self.encode_style(style);
            true
        } else {
            false
        }
    }

    fn encode_style(&mut self, style: Style) {
        if self.flags & Self::FORCE_NEXT_STYLE != 0 || self.styles.last() != Some(&style) {
            self.path_tags.push(PathTag::STYLE);
            self.styles.push(style);
            self.flags &= !Self::FORCE_NEXT_STYLE;
        }
    }

    /// Encodes a transform.
    ///
    /// If the given transform is different from the current one, encodes it and
    /// returns true. Otherwise, encodes nothing and returns false.
    pub fn encode_transform(&mut self, transform: Transform) -> bool {
        if self.flags & Self::FORCE_NEXT_TRANSFORM != 0
            || self.transforms.last() != Some(&transform)
        {
            self.path_tags.push(PathTag::TRANSFORM);
            self.transforms.push(transform);
            self.flags &= !Self::FORCE_NEXT_TRANSFORM;
            true
        } else {
            false
        }
    }

    /// Returns an encoder for encoding a path. If `is_fill` is true, all subpaths will
    /// be automatically closed.
    pub fn encode_path(&mut self, is_fill: bool) -> PathEncoder<'_> {
        PathEncoder::new(
            &mut self.path_tags,
            &mut self.path_data,
            &mut self.n_path_segments,
            &mut self.n_paths,
            is_fill,
        )
    }

    /// Encodes a shape. If `is_fill` is true, all subpaths will be automatically closed.
    /// Returns `true` if a non-zero number of segments were encoded.
    pub fn encode_shape(&mut self, shape: &impl Shape, is_fill: bool) -> bool {
        let mut encoder = self.encode_path(is_fill);
        encoder.shape(shape);
        encoder.finish(true) != 0
    }

    /// Encode an empty path.
    ///
    /// This is useful for bookkeeping when a path is absolutely required (for example in
    /// pushing a clip layer). It is almost always the case, however, that an application
    /// can be optimized to not use this method.
    pub fn encode_empty_shape(&mut self) {
        let mut encoder = self.encode_path(true);
        encoder.empty_path();
        encoder.finish(true);
    }

    /// Encodes a path element iterator. If `is_fill` is true, all subpaths will be automatically
    /// closed. Returns `true` if a non-zero number of segments were encoded.
    pub fn encode_path_elements(
        &mut self,
        path: impl Iterator<Item = peniko::kurbo::PathEl>,
        is_fill: bool,
    ) -> bool {
        let mut encoder = self.encode_path(is_fill);
        encoder.path_elements(path);
        encoder.finish(true) != 0
    }

    /// Encodes a brush with an optional alpha modifier.
    #[expect(
        single_use_lifetimes,
        reason = "False positive: https://github.com/rust-lang/rust/issues/129255"
    )]
    pub fn encode_brush<'b>(&mut self, brush: impl Into<BrushRef<'b>>, alpha: f32) {
        use super::math::point_to_f32;
        match brush.into() {
            BrushRef::Solid(color) => {
                let color = if alpha != 1.0 {
                    color.multiply_alpha(alpha)
                } else {
                    color
                };
                self.encode_color(color);
            }
            BrushRef::Gradient(gradient) => match gradient.kind {
                GradientKind::Linear(LinearGradientPosition { start, end }) => {
                    self.encode_linear_gradient(
                        DrawLinearGradient {
                            index: 0,
                            p0: point_to_f32(start),
                            p1: point_to_f32(end),
                        },
                        gradient.stops.iter().copied(),
                        alpha,
                        gradient.extend,
                    );
                }
                GradientKind::Radial(RadialGradientPosition {
                    start_center,
                    start_radius,
                    end_center,
                    end_radius,
                }) => {
                    self.encode_radial_gradient(
                        DrawRadialGradient {
                            index: 0,
                            p0: point_to_f32(start_center),
                            p1: point_to_f32(end_center),
                            r0: start_radius,
                            r1: end_radius,
                        },
                        gradient.stops.iter().copied(),
                        alpha,
                        gradient.extend,
                    );
                }
                GradientKind::Sweep(SweepGradientPosition {
                    center,
                    start_angle,
                    end_angle,
                }) => {
                    use core::f32::consts::TAU;
                    self.encode_sweep_gradient(
                        DrawSweepGradient {
                            index: 0,
                            p0: point_to_f32(center),
                            t0: start_angle / TAU,
                            t1: end_angle / TAU,
                        },
                        gradient.stops.iter().copied(),
                        alpha,
                        gradient.extend,
                    );
                }
            },
            BrushRef::Image(image) => {
                self.encode_image(image, alpha);
            }
        }
    }

    /// Encodes a solid color brush.
    pub fn encode_color(&mut self, color: impl Into<DrawColor>) {
        let color = color.into();
        self.draw_tags.push(DrawTag::COLOR);
        let DrawColor { rgba } = color;
        self.draw_data.push(rgba);
    }

    /// Sets the blend mode for subsequent per-draw compositing (non-isolated blending).
    ///
    /// This inserts a draw object into the stream; it does not draw geometry by itself.
    /// Call before fills/strokes that should use this mode. Default is normal `SrcOver`.
    pub fn encode_set_blend_mode(&mut self, blend: BlendMode) {
        // #region agent log af15c3 - H1: SET_BLEND_MODE encoded with dummy PATH, confirm path_ix offset
        {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("C:\\Dev\\kob3\\debug-af15c3.log") {
                let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis();
                let _ = writeln!(f, r#"{{"sessionId":"af15c3","runId":"pre-fix","hypothesisId":"H1","timestamp":{ts},"location":"encoding.rs:encode_set_blend_mode","message":"SET_BLEND_MODE encoded, about to push dummy PathTag::PATH","data":{{"n_paths_before":{},"draw_tags_len_before":{}}}}}"#,
                    self.n_paths, self.draw_tags.len());
            }
        }
        // #endregion
        self.draw_tags.push(DrawTag::SET_BLEND_MODE);
        let packed = ((blend.mix as u32) << 8) | blend.compose as u32;
        self.draw_data.push(packed);
        // Every draw tag must have a corresponding path slot so that n_paths == draw_tags.len().
        // Without this, GPU shaders that iterate up to n_draw_objects (= n_paths) would skip
        // draw objects inserted after SET_BLEND_MODE, and the path[] array indexing would diverge
        // between tile_alloc (uses drawobj_ix) and flatten/path_count (uses path-tag prefix index).
        self.path_tags.push(PathTag::PATH);
        self.n_paths += 1;
    }

    /// Encodes a linear gradient brush.
    pub fn encode_linear_gradient(
        &mut self,
        gradient: DrawLinearGradient,
        color_stops: impl Iterator<Item = ColorStop>,
        alpha: f32,
        extend: Extend,
    ) {
        match self.add_ramp(color_stops, alpha, extend) {
            RampStops::Empty => self.encode_color(palette::css::TRANSPARENT),
            RampStops::One(color) => {
                self.encode_color(color);
            }
            RampStops::Many => {
                self.draw_tags.push(DrawTag::LINEAR_GRADIENT);
                self.draw_data
                    .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&gradient)));
            }
        }
    }

    /// Encodes a radial gradient brush.
    pub fn encode_radial_gradient(
        &mut self,
        gradient: DrawRadialGradient,
        color_stops: impl Iterator<Item = ColorStop>,
        alpha: f32,
        extend: Extend,
    ) {
        // Match Skia's epsilon for radii comparison
        const SKIA_EPSILON: f32 = 1.0 / (1 << 12) as f32;
        if gradient.p0 == gradient.p1 && (gradient.r0 - gradient.r1).abs() < SKIA_EPSILON {
            self.encode_color(palette::css::TRANSPARENT);
            return;
        }
        match self.add_ramp(color_stops, alpha, extend) {
            RampStops::Empty => self.encode_color(palette::css::TRANSPARENT),
            RampStops::One(color) => self.encode_color(color),
            RampStops::Many => {
                self.draw_tags.push(DrawTag::RADIAL_GRADIENT);
                self.draw_data
                    .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&gradient)));
            }
        }
    }

    /// Encodes a radial gradient brush.
    pub fn encode_sweep_gradient(
        &mut self,
        gradient: DrawSweepGradient,
        color_stops: impl Iterator<Item = ColorStop>,
        alpha: f32,
        extend: Extend,
    ) {
        const SKIA_DEGENERATE_THRESHOLD: f32 = 1.0 / (1 << 15) as f32;
        if (gradient.t0 - gradient.t1).abs() < SKIA_DEGENERATE_THRESHOLD {
            self.encode_color(palette::css::TRANSPARENT);
            return;
        }
        match self.add_ramp(color_stops, alpha, extend) {
            RampStops::Empty => self.encode_color(palette::css::TRANSPARENT),
            RampStops::One(color) => self.encode_color(color),
            RampStops::Many => {
                self.draw_tags.push(DrawTag::SWEEP_GRADIENT);
                self.draw_data
                    .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&gradient)));
            }
        }
    }

    /// Encodes an image brush.
    pub fn encode_image<'b>(&mut self, brush: impl Into<ImageBrushRef<'b>>, alpha: f32) {
        let brush: ImageBrushRef<'b> = brush.into();
        let ImageSampler {
            x_extend,
            y_extend,
            quality,
            alpha: global_alpha,
        } = brush.sampler;
        let alpha = (global_alpha * alpha * 255.0).round() as u8;
        // TODO: feed the alpha multiplier through the full pipeline for consistency
        // with other brushes?
        // Tracked in https://github.com/linebender/vello/issues/692
        self.resources.patches.push(Patch::Image {
            image: brush.image.clone(),
            draw_data_offset: self.draw_data.len(),
        });
        self.draw_tags.push(DrawTag::IMAGE);
        self.draw_data
            .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&DrawImage {
                xy: 0,
                width_height: (brush.image.width << 16) | (brush.image.height & 0xFFFF),
                sample_alpha: ((brush.image.format as u32) << 15
                    | (brush.image.alpha_type as u32) << 14
                    | (quality as u32) << 12
                    | ((x_extend as u32) << 10)
                    | ((y_extend as u32) << 8)
                    | alpha as u32),
            })));
    }

    // Encodes a blurred rounded rectangle brush.
    pub fn encode_blurred_rounded_rect(
        &mut self,
        color: impl Into<DrawColor>,
        width: f32,
        height: f32,
        radius: f32,
        std_dev: f32,
    ) {
        self.draw_tags.push(DrawTag::BLUR_RECT);
        self.draw_data
            .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(
                &DrawBlurRoundedRect {
                    color: color.into(),
                    width,
                    height,
                    radius,
                    std_dev,
                },
            )));
    }

    /// Encodes a begin clip command.
    pub fn encode_begin_clip(&mut self, parameters: DrawBeginClip) {
        self.clip_filter_stack
            .push(self.pending_layer_filter.take());
        self.draw_tags.push(DrawTag::BEGIN_CLIP);
        self.draw_data
            .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&parameters)));
        // Reserve a u32 `layer_idx` slot; `encode_end_clip` backfills it when this
        // clip turns out to be a filter layer. See the `DrawTag::BEGIN_CLIP` doc for
        // why the layer index lives on `BEGIN_CLIP` rather than `END_CLIP_FILTER`.
        let layer_idx_slot = self.draw_data.len();
        self.draw_data.push(0);
        self.begin_clip_stack.push((parameters, layer_idx_slot));
        self.n_clips += 1;
        self.n_open_clips += 1;
    }

    /// Associates [`FilterPrimitive`] with the next [`Self::encode_begin_clip`] (called by `ekrano::Scene::push_filter_layer`).
    pub fn set_pending_layer_filter(&mut self, filter: Option<FilterPrimitive>) {
        self.pending_layer_filter = filter;
    }

    /// Encodes an end clip command.
    pub fn encode_end_clip(&mut self) {
        if self.n_open_clips > 0 {
            let filter_for_layer = self.clip_filter_stack.pop();
            let (parameters, layer_idx_slot) = self
                .begin_clip_stack
                .pop()
                .expect("encode_end_clip without matching encode_begin_clip");
            // `pop()` is `Option<Option<FilterPrimitive>>`: inner None = regular layer (no filter).
            // Only `Some(Some(_))` is a real filter layer; do not use `pop().is_some()` on the outer option.
            let has_filter = matches!(&filter_for_layer, Some(Some(_)));
            let mut pushed_layer_index: u32 = 0;
            if let Some(f) = filter_for_layer.flatten() {
                let layer_index = self.layer_filter_effects.len() as u32;
                pushed_layer_index = layer_index;
                self.layer_filter_effects.push(LayerFilterEffect {
                    primitive: f,
                    layer_blend: parameters.blend_mode,
                    layer_alpha: parameters.alpha,
                    layer_index,
                });
            }
            // Backfill the `BEGIN_CLIP`'s reserved `layer_idx` slot. `coarse.slang`
            // reads this via `scene[dd + 2]` on `END_CLIP_FILTER` because `clip_leaf.slang`
            // has already rewritten `END_CLIP_FILTER`'s `scene_offset` to point at the
            // matching `BEGIN_CLIP`'s scene data.
            self.draw_data[layer_idx_slot] = pushed_layer_index;
            // Coarse/fine read `scene[dd]`… for END_CLIP / END_CLIP_FILTER.
            if has_filter {
                self.draw_tags.push(DrawTag::END_CLIP_FILTER);
                self.draw_data
                    .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&parameters)));
                // `END_CLIP_FILTER`'s own `scene_offset` is clobbered by `clip_leaf.slang`
                // (see [`DrawTag::BEGIN_CLIP`]), so this word is unused at coarse time, but
                // we still reserve it to keep the draw-tag stream's scene sizes accurate for
                // the prefix-sum-driven draw_monoids.
                self.draw_data.push(pushed_layer_index);
            } else {
                self.draw_tags.push(DrawTag::END_CLIP);
                self.draw_data
                    .extend_from_slice(bytemuck::cast_slice(bytemuck::bytes_of(&parameters)));
            }
            // This is a dummy path, and will go away with the new clip impl.
            self.path_tags.push(PathTag::PATH);
            self.n_paths += 1;
            self.n_clips += 1;
            self.n_open_clips -= 1;
        }
    }

    /// Forces the next transform and style to be encoded even if they match
    /// the current state.
    pub fn force_next_transform_and_style(&mut self) {
        self.flags |= Self::FORCE_NEXT_TRANSFORM | Self::FORCE_NEXT_STYLE;
    }

    // Swap the last two tags in the path tag stream; used for transformed
    // gradients.
    pub fn swap_last_path_tags(&mut self) {
        let len = self.path_tags.len();
        self.path_tags.swap(len - 1, len - 2);
    }

    fn add_ramp(
        &mut self,
        color_stops: impl Iterator<Item = ColorStop>,
        alpha: f32,
        extend: Extend,
    ) -> RampStops {
        let offset = self.draw_data.len();
        let stops_start = self.resources.color_stops.len();
        if alpha != 1.0 {
            self.resources
                .color_stops
                .extend(color_stops.map(|stop| stop.multiply_alpha(alpha)));
        } else {
            self.resources.color_stops.extend(color_stops);
        }
        let stops_end = self.resources.color_stops.len();
        match stops_end - stops_start {
            0 => RampStops::Empty,
            1 => RampStops::One(self.resources.color_stops.pop().unwrap().color),
            _ => {
                self.resources.patches.push(Patch::Ramp {
                    draw_data_offset: offset,
                    stops: stops_start..stops_end,
                    extend,
                });
                RampStops::Many
            }
        }
    }
}

/// Result for adding a sequence of color stops.
enum RampStops {
    /// Color stop sequence was empty.
    Empty,
    /// Contained a single color stop.
    One(DynamicColor),
    /// More than one color stop.
    Many,
}

/// Encoded data for late bound resources.
#[derive(Clone, Default)]
pub struct Resources {
    /// Draw data patches for late bound resources.
    pub patches: Vec<Patch>,
    /// Color stop collection for gradients.
    pub color_stops: Vec<ColorStop>,
    /// Positioned glyph buffer.
    pub glyphs: Vec<Glyph>,
    /// Sequences of glyphs.
    pub glyph_runs: Vec<GlyphRun>,
    /// Normalized coordinate buffer for variable fonts.
    pub normalized_coords: Vec<NormalizedCoord>,
}

impl Resources {
    #[doc(alias = "clear")]
    // This is not called "clear" because "clear" has other implications
    // in graphics contexts.
    fn reset(&mut self) {
        self.patches.clear();
        self.color_stops.clear();
        self.glyphs.clear();
        self.glyph_runs.clear();
        self.normalized_coords.clear();
    }
}

/// Snapshot of offsets for encoded streams.
#[derive(Copy, Clone, Default, Debug)]
pub struct StreamOffsets {
    /// Current length of path tag stream.
    pub path_tags: usize,
    /// Current length of path data stream.
    pub path_data: usize,
    /// Current length of draw tag stream.
    pub draw_tags: usize,
    /// Current length of draw data stream.
    pub draw_data: usize,
    /// Current length of transform stream.
    pub transforms: usize,
    /// Current length of style stream.
    pub styles: usize,
}

impl StreamOffsets {
    pub(crate) fn add(&mut self, other: &Self) {
        self.path_tags += other.path_tags;
        self.path_data += other.path_data;
        self.draw_tags += other.draw_tags;
        self.draw_data += other.draw_data;
        self.transforms += other.transforms;
        self.styles += other.styles;
    }
}

#[cfg(test)]
mod tests {
    use peniko::{Extend, ImageQuality};

    #[test]
    fn ensure_image_quality_values() {
        assert_eq!(ImageQuality::Low as u32, 0);
        assert_eq!(ImageQuality::Medium as u32, 1);
        assert_eq!(ImageQuality::High as u32, 2);
        // exhaustive match to catch new variants
        match ImageQuality::Low {
            ImageQuality::Low | ImageQuality::Medium | ImageQuality::High => {}
        }
    }

    #[test]
    fn ensure_extend_values() {
        assert_eq!(Extend::Pad as u32, 0);
        assert_eq!(Extend::Repeat as u32, 1);
        assert_eq!(Extend::Reflect as u32, 2);
        // exhaustive match to catch new variants
        match Extend::Pad {
            Extend::Pad | Extend::Repeat | Extend::Reflect => {}
        }
    }
}
