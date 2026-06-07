// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

//! Utility types

use ekrano_encoding::ConfigUniform;
use std::ops::Mul;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Self) -> f32 {
        (self.x * other.y) - (self.y * other.x)
    }

    pub fn length(self) -> f32 {
        self.x.hypot(self.y)
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    pub fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn from_array(a: [f32; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    pub fn mix(self, other: Self, t: f32) -> Self {
        let x = self.x + (other.x - self.x) * t;
        let y = self.y + (other.y - self.y) * t;
        Self { x, y }
    }

    pub fn normalize(self) -> Self {
        self / self.length()
    }

    pub fn atan2(self) -> f32 {
        self.y.atan2(self.x)
    }

    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }
}

#[derive(Clone)]
pub(crate) struct Transform(pub(crate) [f32; 6]);

impl Transform {
    pub(crate) fn identity() -> Self {
        Self([1., 0., 0., 1., 0., 0.])
    }

    pub(crate) fn apply(&self, p: Vec2) -> Vec2 {
        let z = self.0;
        let x = z[0] * p.x + z[2] * p.y + z[4];
        let y = z[1] * p.x + z[3] * p.y + z[5];
        Vec2 { x, y }
    }

    pub(crate) fn inverse(&self) -> Self {
        let z = self.0;
        let inv_det = (z[0] * z[3] - z[1] * z[2]).recip();
        let inv_mat = [z[3] * inv_det, -z[1] * inv_det, -z[2] * inv_det, z[0] * inv_det];
        Self([
            inv_mat[0],
            inv_mat[1],
            inv_mat[2],
            inv_mat[3],
            -(inv_mat[0] * z[4] + inv_mat[2] * z[5]),
            -(inv_mat[1] * z[4] + inv_mat[3] * z[5]),
        ])
    }

    pub(crate) fn read(transform_base: u32, ix: u32, data: &[u32]) -> Self {
        let mut z = [0.0; 6];
        let base = (transform_base + ix * 6) as usize;
        for i in 0..6 {
            z[i] = f32::from_bits(data[base + i]);
        }
        Self(z)
    }
}

impl Mul for Transform {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self([
            self.0[0] * other.0[0] + self.0[2] * other.0[1],
            self.0[1] * other.0[0] + self.0[3] * other.0[1],
            self.0[0] * other.0[2] + self.0[2] * other.0[3],
            self.0[1] * other.0[2] + self.0[3] * other.0[3],
            self.0[0] * other.0[4] + self.0[2] * other.0[5] + self.0[4],
            self.0[1] * other.0[4] + self.0[3] * other.0[5] + self.0[5],
        ])
    }
}

pub(crate) fn span(a: f32, b: f32) -> u32 {
    (a.max(b).ceil() - a.min(b).floor()).max(1.0) as u32
}

const DRAWTAG_NOP: u32 = 0;

/// Read draw tag, guarded by number of draw objects.
///
/// The `ix` argument is allowed to exceed the number of draw objects,
/// in which case a NOP is returned.
pub(crate) fn read_draw_tag_from_scene(config: &ConfigUniform, scene: &[u32], ix: u32) -> u32 {
    if ix < config.layout.n_draw_objects {
        let tag_ix = config.layout.draw_tag_base + ix;
        scene[tag_ix as usize]
    } else {
        DRAWTAG_NOP
    }
}

/// The largest floating point value strictly less than 1.
///
/// This value is used to limit the value of b so that its floor is strictly less
/// than 1. That guarantees that floor(a * i + b) == 0 for i == 0, which lands on
/// the correct first tile.
pub(crate) const ONE_MINUS_ULP: f32 = 0.99999994;

/// An epsilon to be applied in path numerical robustness.
///
/// When floor(a * (n - 1) + b) does not match the expected value (the width in
/// grid cells minus one), this delta is applied to a to push it in the correct
/// direction. The theory is that a is not off by more than a few ulp, and it's
/// always in the range of 0..1.
pub(crate) const ROBUST_EPSILON: f32 = 2e-7;

// ---- Morton / Z-curve helpers (CPU mirror of ekrano_shared.slang) ----

#[inline(always)]
fn morton_expand(mut x: u32) -> u32 {
    x &= 0x0000_ffff;
    x = (x ^ (x << 8)) & 0x00ff_00ff;
    x = (x ^ (x << 4)) & 0x0f0f_0f0f;
    x = (x ^ (x << 2)) & 0x3333_3333;
    x = (x ^ (x << 1)) & 0x5555_5555;
    x
}

#[inline(always)]
fn morton_compact(mut x: u32) -> u32 {
    x &= 0x5555_5555;
    x = (x ^ (x >> 1)) & 0x3333_3333;
    x = (x ^ (x >> 2)) & 0x0f0f_0f0f;
    x = (x ^ (x >> 4)) & 0x00ff_00ff;
    x = (x ^ (x >> 8)) & 0x0000_ffff;
    x
}

/// Encode (x, y) into a Z-curve (Morton) index.
#[inline(always)]
pub(crate) fn morton_encode_2d(x: u32, y: u32) -> u32 {
    morton_expand(x) | (morton_expand(y) << 1)
}

/// Decode a Z-curve (Morton) index back to (x, y).
#[allow(dead_code, reason = "only referenced from #[cfg(test)] round-trip tests")]
#[inline(always)]
pub(crate) fn morton_decode_2d(z: u32) -> (u32, u32) {
    (morton_compact(z), morton_compact(z >> 1))
}

/// Return the side-length of the smallest power-of-two square that fits a
/// `width × height` tile bbox for Morton allocation.  Returns 0 for empty bboxes.
#[inline(always)]
pub(crate) fn morton_tile_dim(width: u32, height: u32) -> u32 {
    if width == 0 || height == 0 {
        return 0;
    }
    let d = width.max(height);
    d.next_power_of_two()
}

#[cfg(test)]
mod morton_tests {
    use super::*;

    // --- morton_tile_dim ---

    #[test]
    fn tile_dim_zero_for_empty_bbox() {
        assert_eq!(morton_tile_dim(0, 5), 0);
        assert_eq!(morton_tile_dim(5, 0), 0);
        assert_eq!(morton_tile_dim(0, 0), 0);
    }

    #[test]
    fn tile_dim_exact_powers_of_two() {
        assert_eq!(morton_tile_dim(1, 1), 1);
        assert_eq!(morton_tile_dim(2, 2), 2);
        assert_eq!(morton_tile_dim(4, 4), 4);
        assert_eq!(morton_tile_dim(8, 8), 8);
        assert_eq!(morton_tile_dim(16, 16), 16);
        assert_eq!(morton_tile_dim(64, 64), 64);
    }

    #[test]
    fn tile_dim_rounds_up_to_next_pow2() {
        assert_eq!(morton_tile_dim(3, 1), 4); // max=3 → 4
        assert_eq!(morton_tile_dim(1, 3), 4);
        assert_eq!(morton_tile_dim(5, 3), 8); // max=5 → 8
        assert_eq!(morton_tile_dim(7, 7), 8);
        assert_eq!(morton_tile_dim(9, 1), 16); // max=9 → 16
        assert_eq!(morton_tile_dim(1, 9), 16);
    }

    #[test]
    fn tile_dim_non_square_uses_max_axis() {
        assert_eq!(morton_tile_dim(1, 64), 64); // narrow tall strip
        assert_eq!(morton_tile_dim(64, 1), 64); // wide flat strip
        assert_eq!(morton_tile_dim(2, 4), 4);
        assert_eq!(morton_tile_dim(4, 2), 4);
    }

    // --- morton_expand / morton_compact round-trip ---

    #[test]
    fn expand_compact_roundtrip() {
        for x in [0_u32, 1, 2, 3, 7, 15, 255, 65535] {
            let expanded = morton_expand(x);
            let compacted = morton_compact(expanded);
            assert_eq!(compacted, x, "roundtrip failed for x={x}");
        }
    }

    #[test]
    fn expand_known_values() {
        // morton_expand(0b0000_0001) = 0b0000_0001  (bit 0 → bit 0)
        assert_eq!(morton_expand(0b01), 0b01);
        // morton_expand(0b0000_0010) = 0b0000_0100  (bit 1 → bit 2)
        assert_eq!(morton_expand(0b10), 0b0100);
        // morton_expand(0b0000_0011) = 0b0000_0101  (bits 0,1 → bits 0,2)
        assert_eq!(morton_expand(0b11), 0b0101);
        // All 16 low bits set → alternating bits 0,2,4,...,30
        assert_eq!(morton_expand(0xffff), 0x5555_5555);
    }

    // --- morton_encode_2d / morton_decode_2d round-trip ---

    #[test]
    fn encode_decode_roundtrip() {
        let cases = [(0, 0), (1, 0), (0, 1), (1, 1), (3, 5), (15, 12), (63, 42), (255, 255)];
        for (x, y) in cases {
            let z = morton_encode_2d(x, y);
            let (dx, dy) = morton_decode_2d(z);
            assert_eq!((dx, dy), (x, y), "roundtrip failed for ({x},{y})");
        }
    }

    #[test]
    fn encode_known_values() {
        // (0,0) → 0
        assert_eq!(morton_encode_2d(0, 0), 0);
        // (1,0) → 0b01 (x in even bits)
        assert_eq!(morton_encode_2d(1, 0), 0b01);
        // (0,1) → 0b10 (y in odd bits)
        assert_eq!(morton_encode_2d(0, 1), 0b10);
        // (1,1) → 0b11
        assert_eq!(morton_encode_2d(1, 1), 0b11);
        // (2,0) → 0b0100
        assert_eq!(morton_encode_2d(2, 0), 0b0100);
        // (0,2) → 0b1000
        assert_eq!(morton_encode_2d(0, 2), 0b1000);
        // (2,2) → 0b1100
        assert_eq!(morton_encode_2d(2, 2), 0b1100);
        // (3,3) → 0b1111 = 15
        assert_eq!(morton_encode_2d(3, 3), 15);
    }

    // --- Critical property: all indices within dim² for any W×H bbox ---
    //
    // For every (x,y) with 0 ≤ x < width, 0 ≤ y < height,
    // morton_encode_2d(x, y) must be < dim² where dim = morton_tile_dim(width, height).
    // This is the core correctness requirement for tile allocation.

    fn assert_no_overflow(width: u32, height: u32) {
        let dim = morton_tile_dim(width, height);
        let limit = dim * dim;
        for y in 0..height {
            for x in 0..width {
                let idx = morton_encode_2d(x, y);
                assert!(
                    idx < limit,
                    "morton_encode_2d({x},{y}) = {idx} >= dim²={limit} (dim={dim}, w={width}, h={height})"
                );
            }
        }
    }

    #[test]
    fn no_index_overflow_square_bboxes() {
        for dim in [1_u32, 2, 4, 8, 16] {
            assert_no_overflow(dim, dim);
        }
    }

    #[test]
    fn no_index_overflow_non_square_bboxes() {
        assert_no_overflow(1, 2);
        assert_no_overflow(2, 1);
        assert_no_overflow(1, 4);
        assert_no_overflow(4, 1);
        assert_no_overflow(2, 4);
        assert_no_overflow(3, 5);
        assert_no_overflow(5, 3);
        assert_no_overflow(1, 64); // narrow tall strip — worst case for memory
        assert_no_overflow(64, 1);
        assert_no_overflow(10, 7);
    }

    // --- No collisions: all (x,y) pairs within a bbox map to distinct indices ---

    fn assert_no_collisions(width: u32, height: u32) {
        let dim = morton_tile_dim(width, height);
        let limit = (dim * dim) as usize;
        let mut seen = vec![false; limit];
        for y in 0..height {
            for x in 0..width {
                let idx = morton_encode_2d(x, y) as usize;
                assert!(
                    !seen[idx],
                    "collision at index {idx} for ({x},{y}) in {width}×{height} bbox"
                );
                seen[idx] = true;
            }
        }
    }

    #[test]
    fn no_collisions_various_bboxes() {
        for (w, h) in [
            (1, 1),
            (2, 2),
            (4, 4),
            (8, 8),
            (2, 4),
            (4, 2),
            (3, 5),
            (1, 8),
            (8, 1),
            (7, 6),
        ] {
            assert_no_collisions(w, h);
        }
    }
}
