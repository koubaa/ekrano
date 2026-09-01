// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use bytemuck::{Pod, Zeroable};
use peniko::{
    BlendMode, Color,
    color::{AlphaColor, ColorSpace, DynamicColor, OpaqueColor, PremulColor, Srgb},
};

use super::Monoid;

/// Draw tag representation.
#[derive(Copy, Clone, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct DrawTag(pub u32);

impl DrawTag {
    /// No operation.
    pub const NOP: Self = Self(0);

    /// Color fill.
    pub const COLOR: Self = Self(0x44);

    /// Linear gradient fill.
    pub const LINEAR_GRADIENT: Self = Self(0x114);

    /// Radial gradient fill.
    pub const RADIAL_GRADIENT: Self = Self(0x29c);

    /// Sweep gradient fill.
    pub const SWEEP_GRADIENT: Self = Self(0x254);

    /// Image fill.
    pub const IMAGE: Self = Self(0x28C); // info: 10, scene: 3

    /// Tinted image fill.
    pub const IMAGE_TINTED: Self = Self(0x2D0); // info: 11, scene: 4

    /// Blurred rounded rectangle.
    pub const BLUR_RECT: Self = Self(0x2d4); // info: 11, scene: 5 (DrawBlurRoundedRect)

    /// Begin layer/clip.
    ///
    /// Scene payload: [`DrawBeginClip`] words + a `u32` "filter layer index" slot that's
    /// populated by [`Encoding::encode_end_clip`](crate::encoding::Encoding::encode_end_clip) when the layer has a filter. The extra
    /// word exists to work around `clip_leaf.slang`'s scene-offset rewrite: it copies
    /// `BEGIN_CLIP`'s `scene_offset` to the matching `END_CLIP[_FILTER]`'s monoid, so any
    /// data `END_CLIP_FILTER` needs at coarse time (`scene[dd + 2]`) has to live in the
    /// matching `BEGIN_CLIP`'s scene slot. `(tag >> 2) & 7 == 3`.
    pub const BEGIN_CLIP: Self = Self(0x4D);

    /// End layer/clip.
    /// Scene payload: duplicate [`DrawBeginClip`] words (blend + alpha); `(tag >> 2) & 7 == 2`.
    pub const END_CLIP: Self = Self(0x09);

    /// End a filter layer (same as [`Self::END_CLIP`] for compositing params, plus layer index).
    /// Scene payload: [`DrawBeginClip`] words + `u32` filter layer index; `(tag >> 2) & 7 == 3`.
    pub const END_CLIP_FILTER: Self = Self(0x0D);

    /// Set per-draw blend mode for subsequent fills (non-isolated blending).
    /// Scene data: one `u32` packed like [`DrawBeginClip::new`](DrawBeginClip::new).
    pub const SET_BLEND_MODE: Self = Self(0x04);
}

impl DrawTag {
    /// Returns the size of the info buffer (in u32s) used by this tag.
    pub const fn info_size(self) -> u32 {
        (self.0 >> 6) & 0xf
    }
}

/// The first word of each draw info stream entry contains the flags.
///
/// This is not part of the draw object stream but gets used after the draw
/// objects get reduced on the GPU. `0` represents a non-zero fill.
/// `1` represents an even-odd fill.
pub const DRAW_INFO_FLAGS_FILL_RULE_BIT: u32 = 1;

/// Draw object bounding box.
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
#[repr(C)]
pub struct DrawBbox {
    pub bbox: [f32; 4],
}

/// Draw data for a solid color.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawColor {
    /// Packed little-endian RGBA premultiplied color with the red component in the low byte, i.e.,
    /// with `r` the least significant byte and `a` the most significant.
    pub rgba: u32,
}

impl<CS: ColorSpace> From<AlphaColor<CS>> for DrawColor {
    fn from(color: AlphaColor<CS>) -> Self {
        Self {
            rgba: color.convert::<Srgb>().premultiply().to_rgba8().to_u32(),
        }
    }
}

impl From<DynamicColor> for DrawColor {
    fn from(color: DynamicColor) -> Self {
        Self {
            rgba: color.to_alpha_color::<Srgb>().premultiply().to_rgba8().to_u32(),
        }
    }
}

impl<CS: ColorSpace> From<OpaqueColor<CS>> for DrawColor {
    fn from(color: OpaqueColor<CS>) -> Self {
        Self {
            rgba: color.convert::<Srgb>().with_alpha(1.).premultiply().to_rgba8().to_u32(),
        }
    }
}

impl<CS: ColorSpace> From<PremulColor<CS>> for DrawColor {
    fn from(color: PremulColor<CS>) -> Self {
        Self {
            rgba: color.convert::<Srgb>().to_rgba8().to_u32(),
        }
    }
}

/// Draw data for a linear gradient.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawLinearGradient {
    /// Ramp index.
    pub index: u32,
    /// Start point.
    pub p0: [f32; 2],
    /// End point.
    pub p1: [f32; 2],
}

/// Draw data for a radial gradient.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawRadialGradient {
    /// Ramp index.
    pub index: u32,
    /// Start point.
    pub p0: [f32; 2],
    /// End point.
    pub p1: [f32; 2],
    /// Start radius.
    pub r0: f32,
    /// End radius.
    pub r1: f32,
}

/// Draw data for a sweep gradient.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawSweepGradient {
    /// Ramp index.
    pub index: u32,
    /// Center point.
    pub p0: [f32; 2],
    /// Normalized start angle.
    pub t0: f32,
    /// Normalized end angle.
    pub t1: f32,
}

/// Draw data for an image.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawImage {
    /// Packed atlas coordinates.
    pub xy: u32,
    /// Packed image dimensions.
    pub width_height: u32,
    /// Packed quality, extend mode and 8-bit alpha (bits `qqxxyyaaaaaaaa`,
    /// 18 unused prefix bits).
    pub sample_alpha: u32,
}

/// Draw data for a tinted image.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawImageTinted {
    /// Packed atlas coordinates.
    pub xy: u32,
    /// Packed image dimensions.
    pub width_height: u32,
    /// Packed quality, extend mode, tint mode and 8-bit alpha.
    pub sample_alpha: u32,
    /// Premultiplied tint color packed as RGBA8.
    pub tint_rgba: u32,
}

/// How an image tint is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TintMode {
    /// Replace the source RGB with the tint and use source alpha as coverage.
    AlphaMask = 1,
    /// Component-wise multiply the premultiplied source and tint colors.
    Multiply = 2,
}

/// A tint applied to image paints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tint {
    /// The tint color.
    pub color: Color,
    /// How the tint is applied.
    pub mode: TintMode,
}

/// Draw data for a blurred rounded rectangle.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawBlurRoundedRect {
    /// Solid color brush.
    pub color: DrawColor,
    /// Rectangle width.
    pub width: f32,
    /// Rectangle height.
    pub height: f32,
    /// Rectangle corner radius.
    pub radius: f32,
    /// Standard deviation of gaussian filter.
    ///
    /// The sign bit encodes invert (`1 - alpha` coverage) for inset box-shadows.
    /// Magnitude is always the non-negative σ.
    pub std_dev: f32,
}

/// Draw data for a clip or layer.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct DrawBeginClip {
    /// Blend mode.
    pub blend_mode: u32,
    /// Group alpha.
    pub alpha: f32,
}

impl DrawBeginClip {
    /// The `blend_mode` used to indicate that a layer should be
    /// treated as a luminance mask.
    ///
    /// The least significant 16 bits are reserved for Mix + Compose
    /// combinations.
    pub const LUMINANCE_MASK_BLEND_MODE: u32 = 0x10000;
    /// The `blend_mode` used to indicate that a layer should be
    /// treated as a clip.
    ///
    /// This is equivalent to `Compose::SrcOver` with a `Mix` of 128,
    /// for legacy reasons.
    /// We expect this to change in the future.
    pub const CLIP_BLEND_MODE: u32 = 0x8003;

    /// Creates new clip draw data for a Porter-Duff blend mode.
    pub fn new(blend_mode: BlendMode, alpha: f32) -> Self {
        Self {
            blend_mode: ((blend_mode.mix as u32) << 8) | blend_mode.compose as u32,
            alpha,
        }
    }

    /// Creates a new clip draw data for a luminance mask.
    pub fn luminance_mask(alpha: f32) -> Self {
        Self {
            blend_mode: Self::LUMINANCE_MASK_BLEND_MODE,
            alpha,
        }
    }

    /// Creates the clip draw data for a clip-only layer.
    pub fn clip() -> Self {
        Self {
            blend_mode: Self::CLIP_BLEND_MODE,
            alpha: 1.0,
        }
    }
}

/// Monoid for the draw tag stream.
#[derive(Copy, Clone, PartialEq, Eq, Pod, Zeroable, Default, Debug)]
#[repr(C)]
pub struct DrawMonoid {
    // The number of paths preceding this draw object.
    pub path_ix: u32,
    // The number of clip operations preceding this draw object.
    pub clip_ix: u32,
    // The offset of the encoded draw object in the scene (u32s).
    pub scene_offset: u32,
    // The offset of the associated info.
    pub info_offset: u32,
}

impl Monoid for DrawMonoid {
    type SourceValue = DrawTag;

    fn new(tag: DrawTag) -> Self {
        Self {
            // SET_BLEND_MODE has a dummy PathTag::PATH on the Rust side, so it counts as a path.
            path_ix: match tag {
                DrawTag::NOP => 0,
                _ => 1,
            },
            clip_ix: tag.0 & 1,
            scene_offset: (tag.0 >> 2) & 0x7,
            info_offset: (tag.0 >> 6) & 0xf,
        }
    }

    fn combine(&self, other: &Self) -> Self {
        Self {
            path_ix: self.path_ix + other.path_ix,
            clip_ix: self.clip_ix + other.clip_ix,
            scene_offset: self.scene_offset + other.scene_offset,
            info_offset: self.info_offset + other.info_offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use peniko::Color;

    use super::DrawColor;

    #[test]
    fn draw_color_endianness() {
        // `DrawColor` should be packed little-endian with red the least significant byte.
        //
        // If this changes intentionally, the `DrawColor` docs also need updating.
        let c = Color::from_rgba8(0x00, 0xca, 0xfe, 0xff);
        assert_eq!(bytemuck::bytes_of(&DrawColor::from(c)), [0x00, 0xca, 0xfe, 0xff]);
    }

    #[test]
    fn draw_color_premultiplied() {
        // If this changes intentionally, the `DrawColor` docs also need updating.
        let c = Color::from_rgba8(0x00, 0xca, 0xfe, 0x00);
        assert_eq!(DrawColor::from(c).rgba, 0);
    }
}
