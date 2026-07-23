// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Ekrano is a 2d graphics rendering engine written in Rust, using Goldy.
//! It efficiently draws large 2d scenes with interactive or near-interactive performance.
//!
//! ## Getting started
//!
//! Ekrano renders scenes to GPU textures via [`GoldyRenderer`]. A typical usage looks like:
//!
//! ```ignore
//! use ekrano::{GoldyRenderer, Scene, RenderParams, AaConfig};
//!
//! let device: goldy::Device = /* obtain from Goldy */;
//! let mut renderer = GoldyRenderer::new(&device).expect("Failed to create renderer");
//!
//! let mut scene = Scene::new();
//! scene.fill(
//!    ekrano::peniko::Fill::NonZero,
//!    ekrano::peniko::kurbo::Affine::IDENTITY,
//!    ekrano::peniko::Color::from_rgb8(242, 140, 168),
//!    None,
//!    &ekrano::peniko::kurbo::Circle::new((420.0, 200.0), 120.0),
//! );
//!
//! let texture: goldy::Texture = /* allocate render target */;
//! renderer
//!    .render_to_texture(
//!       &scene,
//!       &texture,
//!       &RenderParams {
//!          base_color: ekrano::peniko::color::palette::css::BLACK,
//!          width: 800,
//!          height: 600,
//!          antialiasing_method: AaConfig::Area,
//!          robust: true,
//!       },
//!    )
//!    .expect("Failed to render to a texture");
//! ```

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
// Need to allow instead of expect until Rust 1.83 https://github.com/rust-lang/rust/pull/130025
#![allow(missing_docs, reason = "We have many as-yet undocumented items.")]
#![expect(missing_debug_implementations, clippy::cast_possible_truncation, reason = "Deferred")]
#![allow(
    clippy::todo,
    unreachable_pub,
    unnameable_types,
    reason = "Deferred, only apply in some feature sets so not expect"
)]

mod debug;
mod resource_proxy;
mod scene;
mod shaders;

mod goldy_renderer;
mod scheme_gpu_resources;
mod scheme_render;
mod scheme_renderer;
mod worker_retention;

pub mod low_level {
    //! Utilities which can be used to create an alternative renderer to [`GoldyRenderer`][crate::GoldyRenderer].
    //!
    //! These APIs have not been carefully designed, and might not be powerful enough for this use case.

    pub use crate::debug::DebugLayers;
    pub use crate::resource_proxy::{BindType, ImageFormat, ShaderId};
    pub use crate::scheme_render::Render;
    pub use crate::shaders::FullShaders;
    /// Temporary export, used in `with_winit` for stats
    pub use ekrano_encoding::BumpAllocators;
}
/// Styling and composition primitives.
pub use peniko;
/// 2D geometry, with a focus on curves.
pub use peniko::kurbo;

pub use goldy_renderer::{
    AllocatorStats, FrameStats, PreparedFrame, PresentToken, ResourcePoolStats, SceneGrowthStats,
};
pub use scheme_renderer::SchemeRenderer;
/// Goldy-based 2D renderer (retained-`Scheme` frame loop).
pub type GoldyRenderer = SchemeRenderer;

pub use ekrano_encoding::{Glyph, NormalizedCoord};
pub use scene::{DrawGlyphs, Scene};

use low_level::ShaderId;
use thiserror::Error;

/// Represents the anti-aliasing method to use during a render pass.
///
/// Can be configured for a render operation by setting [`RenderParams::antialiasing_method`].
/// Each value of this can only be used if the corresponding field on [`AaSupport`] was used.
///
/// This can be converted into an `AaSupport` using [`Iterator::collect`],
/// as `AaSupport` implements `FromIterator`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AaConfig {
    /// Area anti-aliasing, where the alpha value for a pixel is computed from integrating
    /// the winding number over its square area.
    ///
    /// This technique produces very accurate values when the shape has winding number of 0 or 1
    /// everywhere, but can result in conflation artifacts otherwise.
    /// It generally has better performance than the multi-sampling methods.
    ///
    /// Can only be used if [enabled][AaSupport::area] for the `Renderer`.
    Area,
    /// 8x Multisampling
    ///
    /// Can only be used if [enabled][AaSupport::msaa8] for the `Renderer`.
    Msaa8,
    /// 16x Multisampling
    ///
    /// Can only be used if [enabled][AaSupport::msaa16] for the `Renderer`.
    Msaa16,
}

/// Represents the set of anti-aliasing configurations to enable during pipeline creation.
///
/// This is configured when creating a renderer by selecting which AA methods to support.
///
/// This can be created from a set of `AaConfig` using [`Iterator::collect`],
/// as `AaSupport` implements `FromIterator`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AaSupport {
    /// Support [`AaConfig::Area`].
    pub area: bool,
    /// Support [`AaConfig::Msaa8`].
    pub msaa8: bool,
    /// Support [`AaConfig::Msaa16`].
    pub msaa16: bool,
}

impl AaSupport {
    /// Support every anti-aliasing method.
    ///
    /// This might increase startup time, as more shader variations must be compiled.
    pub fn all() -> Self {
        Self {
            area: true,
            msaa8: true,
            msaa16: true,
        }
    }

    /// Support only [`AaConfig::Area`].
    ///
    /// This should be the default choice for most users.
    pub fn area_only() -> Self {
        Self {
            area: true,
            msaa8: false,
            msaa16: false,
        }
    }
}

impl FromIterator<AaConfig> for AaSupport {
    fn from_iter<T: IntoIterator<Item = AaConfig>>(iter: T) -> Self {
        let mut result = Self {
            area: false,
            msaa8: false,
            msaa16: false,
        };
        for config in iter {
            match config {
                AaConfig::Area => result.area = true,
                AaConfig::Msaa8 => result.msaa8 = true,
                AaConfig::Msaa16 => result.msaa16 = true,
            }
        }
        result
    }
}

/// Errors that can occur in Ekrano.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// Goldy backend shader or GPU operation error.
    #[error("Shader/GPU error: {0}")]
    Shader(String),
    /// An image was submitted for rendering but contained no pixel data despite having non-zero
    /// dimensions. This typically means the image was registered with a different renderer
    /// instance or was unregistered before the render was submitted.
    #[error("Invalid empty image (id: {id}): {reason}")]
    InvalidImage { id: u64, reason: &'static str },
    /// A GPU resource (texture, buffer) could not be created or configured.
    #[error("GPU resource error: {0}")]
    Gpu(String),
    /// Reading rendered pixel data back to the CPU failed.
    #[error("CPU readback error: {0}")]
    Readback(String),
}

pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// Parameters used in a single render that are configurable by the client.
///
/// These are used in [`GoldyRenderer::render_to_texture`].
pub struct RenderParams {
    /// The background color applied to the target. This value is only applicable to the full
    /// pipeline.
    pub base_color: peniko::Color,

    /// Dimensions of the rasterization target
    pub width: u32,
    pub height: u32,

    /// The anti-aliasing algorithm. The selected algorithm must have been initialized while
    /// constructing the `Renderer`.
    pub antialiasing_method: AaConfig,

    /// Enable robust dynamic memory: download the bump allocator after each
    /// frame so overflows can be detected and buffers grown automatically.
    ///
    /// Turning this off eliminates the per-frame GPU→CPU readback and the
    /// pipelining sync point it imposes, which can significantly improve
    /// throughput at the cost of silently producing incomplete output if the
    /// bump allocator overflows.
    ///
    /// Override at runtime with `EKRANO_ROBUST=0` (disable) or `EKRANO_ROBUST=1`
    /// (force enable) for benchmarking.
    ///
    /// Defaults to `true`.
    pub robust: bool,
}
