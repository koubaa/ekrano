// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter primitives for multi-pass compositing (blur, shadow, etc.).
//!
//! GPU scheduling applies [`crate::Encoding::layer_filter_effects`] after the fine rasterizer:
//! each entry is processed in order on that layer’s snapshot texture, then composited back
//! onto the main output.

use bytemuck::{Pod, Zeroable};
use peniko::color::PremulColor;

/// Edge handling for filters that sample outside the layer bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterEdgeMode {
    None,
    Duplicate,
    Wrap,
    Mirror,
}

/// One filter primitive (single-primitive graphs only, as in `vello_cpu` today).
#[derive(Clone, Debug)]
pub enum FilterPrimitive {
    Flood {
        color: PremulColor<peniko::color::Srgb>,
        /// Pixel-coordinate bounding rect `[x0, y0, x1, y1]` that constrains the flood.
        ///
        /// Derived from the `push_filter_layer` clip shape's bounding box. The flood shader
        /// only writes within this rect; pixels outside are copied unchanged from `src`.
        /// Use `[0, 0, u32::MAX, u32::MAX]` to flood the entire frame (legacy behaviour).
        clip_rect: [u32; 4],
    },
    GaussianBlur {
        std_dev: f32,
        edge_mode: FilterEdgeMode,
    },
    DropShadow {
        dx: f32,
        dy: f32,
        std_dev: f32,
        color: PremulColor<peniko::color::Srgb>,
        edge_mode: FilterEdgeMode,
    },
    Offset {
        dx: f32,
        dy: f32,
    },
}

/// A filter graph placeholder (single-primitive only for parity with `vello_cpu`).
#[derive(Clone, Debug)]
pub struct Filter(pub FilterPrimitive);

/// One filtered layer: primitive to run after fine, plus layer compositing parameters.
#[derive(Clone, Debug)]
pub struct LayerFilterEffect {
    pub primitive: FilterPrimitive,
    /// Packed blend from [`crate::draw::DrawBeginClip`] (`mix << 8 | compose`).
    pub layer_blend: u32,
    pub layer_alpha: f32,
    /// Index into per-layer filter textures (0..N-1), assigned when the layer ends.
    pub layer_index: u32,
    /// True when this filter layer is enclosed by another filter layer.
    ///
    /// For nested drop-shadow layers the shadow must be rendered without including the
    /// source foreground pixels (`fg`) in the output — the inner layer's filtered result
    /// is composited on top separately. Non-nested (standalone) filters are unaffected.
    pub is_nested: bool,
}

/// GPU uniform for simple filter compute passes (matches `filter_*.slang` `FilterUniform`).
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct FilterUniform {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `FilterEdgeMode` as `u32`.
    pub edge_mode: u32,
    /// Pass: `0` = horizontal blur, `1` = vertical blur, `2` = offset, `3` = flood, `4` = drop shadow.
    pub pass_kind: u32,
    pub std_dev: f32,
    pub dx: f32,
    pub dy: f32,
    /// Packed premultiplied RGBA (`peniko` order: rgba8 as u32).
    pub color: u32,
    pub _pad: u32,
}

impl FilterEdgeMode {
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Duplicate => 1,
            Self::Wrap => 2,
            Self::Mirror => 3,
        }
    }
}

impl FilterUniform {
    /// Builds params for a separable Gaussian blur pass (`pass_kind` 0 or 1).
    pub fn gaussian_blur(
        width: u32,
        height: u32,
        horizontal: bool,
        std_dev: f32,
        edge: FilterEdgeMode,
    ) -> Self {
        Self {
            width,
            height,
            edge_mode: edge.as_u32(),
            pass_kind: if horizontal { 0 } else { 1 },
            std_dev,
            dx: 0.0,
            dy: 0.0,
            color: 0,
            _pad: 0,
        }
    }

    /// Offset pass (`pass_kind` 2).
    pub fn offset(width: u32, height: u32, dx: f32, dy: f32, edge: FilterEdgeMode) -> Self {
        Self {
            width,
            height,
            edge_mode: edge.as_u32(),
            pass_kind: 2,
            std_dev: 0.0,
            dx,
            dy,
            color: 0,
            _pad: 0,
        }
    }

    /// Flood fill (`pass_kind` 3).
    ///
    /// `clip` is `[x0, y0, x1, y1]` in pixels.  The shader only floods within that rect;
    /// pixels outside are copied from `src` unchanged.  Pass `[0, 0, width, height]` to
    /// flood the full frame.  The four clip values are packed into the otherwise-unused
    /// `edge_mode`, `std_dev`, `dx`, `dy` fields (as bit-cast u32s) so the GPU struct
    /// layout does not change.
    pub fn flood(width: u32, height: u32, color_rgba: u32, clip: [u32; 4]) -> Self {
        let [x0, y0, x1, y1] = clip;
        Self {
            width,
            height,
            // Repurpose unused fields to carry clip bounds for the flood shader.
            edge_mode: x0,
            pass_kind: 3,
            std_dev: f32::from_bits(y0),
            dx: f32::from_bits(x1),
            dy: f32::from_bits(y1),
            color: color_rgba,
            _pad: 0,
        }
    }

    /// Drop shadow composite (`pass_kind` 4): blurred, colored shadow under `src`.
    pub fn drop_shadow(
        width: u32,
        height: u32,
        dx: f32,
        dy: f32,
        std_dev: f32,
        shadow_rgba: u32,
        edge: FilterEdgeMode,
    ) -> Self {
        Self {
            width,
            height,
            edge_mode: edge.as_u32(),
            pass_kind: 4,
            std_dev,
            dx,
            dy,
            color: shadow_rgba,
            _pad: 0,
        }
    }

    /// Drop shadow for a **nested** filter layer: emits only the shadow pixels (no foreground).
    ///
    /// Used when the drop-shadow layer wraps an inner filter layer whose filtered content is
    /// composited separately on top, so that the inner layer's soft/blurred edges are preserved.
    pub fn drop_shadow_nested(
        width: u32,
        height: u32,
        dx: f32,
        dy: f32,
        std_dev: f32,
        shadow_rgba: u32,
        edge: FilterEdgeMode,
    ) -> Self {
        Self {
            width,
            height,
            edge_mode: edge.as_u32(),
            pass_kind: 8,
            std_dev,
            dx,
            dy,
            color: shadow_rgba,
            _pad: 0,
        }
    }

    /// Identity copy (`pass_kind` 5).
    pub fn copy(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edge_mode: 0,
            pass_kind: 5,
            std_dev: 0.0,
            dx: 0.0,
            dy: 0.0,
            color: 0,
            _pad: 0,
        }
    }

    /// Composite filtered premultiplied layer over straight-alpha output (`pass_kind` 6).
    /// `layer_blend_packed` is `DrawBeginClip`-style `(mix << 8) | compose`.
    pub fn composite_filtered_layer(width: u32, height: u32, layer_blend_packed: u32) -> Self {
        Self {
            width,
            height,
            edge_mode: layer_blend_packed,
            pass_kind: 6,
            std_dev: 0.0,
            dx: 0.0,
            dy: 0.0,
            color: 0,
            _pad: 0,
        }
    }

    /// Clear RGBA texture to transparent (`pass_kind` 7).
    pub fn clear_transparent(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edge_mode: 0,
            pass_kind: 7,
            std_dev: 0.0,
            dx: 0.0,
            dy: 0.0,
            color: 0,
            _pad: 0,
        }
    }

    /// 2× downsample: read `src_sampled` (2× larger) via hardware bilinear, write to dst at
    /// (width, height) (`pass_kind` 9).
    pub fn downsample(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edge_mode: 0,
            pass_kind: 9,
            std_dev: 0.0,
            dx: 0.0,
            dy: 0.0,
            color: 0,
            _pad: 0,
        }
    }

    /// 2× upsample: read `src_sampled` (2× smaller) via hardware bilinear, write to dst at
    /// (width, height), overwriting the existing contents (`pass_kind` 11).
    pub fn upsample(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edge_mode: 0,
            pass_kind: 11,
            std_dev: 0.0,
            dx: 0.0,
            dy: 0.0,
            color: 0,
            _pad: 0,
        }
    }

    /// Shadow composite from a pre-blurred source (`pass_kind` 13).
    ///
    /// Reads the pre-blurred alpha from `src_sampled` at `(p - offset)`, colorises it with
    /// `shadow_rgba`, and composites the shadow under the unblurred foreground from `src` (UAV).
    ///
    /// Used for the pyramid drop-shadow where the blur and composite are separated.
    pub fn shadow_composite_preblurred(
        width: u32,
        height: u32,
        dx: f32,
        dy: f32,
        shadow_rgba: u32,
    ) -> Self {
        Self {
            width,
            height,
            edge_mode: 0,
            pass_kind: 13,
            std_dev: 0.0,
            dx,
            dy,
            color: shadow_rgba,
            _pad: 0,
        }
    }

    /// Nested shadow composite from a pre-blurred source — shadow only, no foreground
    /// (`pass_kind` 14).
    pub fn shadow_composite_preblurred_nested(
        width: u32,
        height: u32,
        dx: f32,
        dy: f32,
        shadow_rgba: u32,
    ) -> Self {
        Self {
            width,
            height,
            edge_mode: 0,
            pass_kind: 14,
            std_dev: 0.0,
            dx,
            dy,
            color: shadow_rgba,
            _pad: 0,
        }
    }
}
