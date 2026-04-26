// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Full-frame coverage mask for per-draw compositing (GPU samples in fine stage).

use std::sync::Arc;

/// Row-major alpha mask (`width * height` bytes), same dimensions as the render target.
#[derive(Clone, Debug)]
pub struct CoverageMask {
    pub width: u32,
    pub height: u32,
    pub data: Arc<[u8]>,
}

impl CoverageMask {
    /// Creates a mask from raw bytes (typically one byte per pixel).
    pub fn new(width: u32, height: u32, data: impl Into<Arc<[u8]>>) -> Option<Self> {
        let data = data.into();
        if width == 0 || height == 0 {
            return None;
        }
        let expected = (width as usize).saturating_mul(height as usize);
        if data.len() != expected {
            return None;
        }
        Some(Self {
            width,
            height,
            data,
        })
    }
}
