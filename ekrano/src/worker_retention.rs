// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained worker-scheme invalidation (no per-frame hashing).

use goldy::types::{ResourceHandle, TextureFormat};
#[cfg(debug_assertions)]
use goldy::{Buffer, Texture};

use crate::goldy_renderer::PersistentState;
use crate::AaConfig;
use ekrano_encoding::{
    BufferSizes, FilterPrimitive, LayerFilterEffect, RenderConfig,
};

/// Inputs that change the compute/present node graph — not per-frame payload bytes.
///
/// `scene_bucket` is included because the scene buffer is bound directly in the worker's
/// recorded dispatch nodes: if the bucket grows and the buffer is reallocated (new
/// `ResourceHandle`), the retained worker would reference a stale handle.
///
/// `mask_atlas_width/height` are included for the same reason: `has_coverage_mask` alone
/// only detects None↔Some transitions; dimension changes within a non-None mask also
/// reallocate the texture with a new handle.
#[derive(Clone, PartialEq)]
pub(crate) struct WorkerTopology {
    pub aa: AaConfig,
    pub robust: bool,
    pub out_format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub buffer_sizes: BufferSizes,
    pub has_coverage_mask: bool,
    pub ramps_width: u32,
    pub ramps_height: u32,
    pub images_width: u32,
    pub images_height: u32,
    pub image_count: usize,
    pub swapchain_present: bool,
    /// Scene byte bucket the worker was recorded against; change = new scene buffer handle.
    pub scene_bucket: u64,
    /// Normalized mask atlas dims (1×1 when no coverage mask).
    pub mask_atlas_width: u32,
    pub mask_atlas_height: u32,
}

pub(crate) fn worker_topology(
    params: &crate::RenderParams,
    config: &RenderConfig,
    out_format: TextureFormat,
    has_coverage_mask: bool,
    ramps_width: u32,
    ramps_height: u32,
    images_width: u32,
    images_height: u32,
    image_count: usize,
    swapchain_present: bool,
    scene_bucket: u64,
    coverage_mask_dims: Option<(u32, u32)>,
) -> WorkerTopology {
    let (mask_atlas_width, mask_atlas_height) = coverage_mask_dims.unwrap_or((1, 1));
    WorkerTopology {
        aa: params.antialiasing_method,
        robust: params.robust,
        out_format,
        width: params.width,
        height: params.height,
        buffer_sizes: config.buffer_sizes,
        has_coverage_mask,
        ramps_width,
        ramps_height,
        images_width,
        images_height,
        image_count,
        swapchain_present,
        scene_bucket,
        mask_atlas_width,
        mask_atlas_height,
    }
}

/// All resources the upload scheme writes into, keyed by the dimensions that determine
/// their GPU handle identity.  A change in any field means at least one upload copy node
/// references a stale `ResourceHandle` and the scheme must be re-recorded.
///
/// Dimensions are normalised to the actual texture sizes allocated by
/// `prepare_pipeline_resources` (e.g. 1×1 for an empty atlas) so that the key matches
/// what is stored in `PersistentState::cached_gradient` etc.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct UploadKey {
    pub scene_bucket: u64,
    pub gradient_width: u32,
    pub gradient_height: u32,
    pub image_atlas_width: u32,
    pub image_atlas_height: u32,
    pub mask_atlas_width: u32,
    pub mask_atlas_height: u32,
}

/// Build an [`UploadKey`] from the per-frame inputs visible before `prepare_pipeline_resources`.
///
/// `ramps_height == 0` → 1×1 gradient atlas.  
/// `image_count == 0` → 1×1 image atlas.  
/// `coverage_mask_dims == None` → 1×1 mask atlas.
pub(crate) fn upload_key(
    scene_bucket: u64,
    ramps_width: u32,
    ramps_height: u32,
    image_count: usize,
    images_width: u32,
    images_height: u32,
    coverage_mask_dims: Option<(u32, u32)>,
) -> UploadKey {
    let (gradient_width, gradient_height) =
        if ramps_height == 0 { (1, 1) } else { (ramps_width, ramps_height) };
    let (image_atlas_width, image_atlas_height) =
        if image_count == 0 { (1, 1) } else { (images_width, images_height) };
    let (mask_atlas_width, mask_atlas_height) = coverage_mask_dims.unwrap_or((1, 1));
    UploadKey {
        scene_bucket,
        gradient_width,
        gradient_height,
        image_atlas_width,
        image_atlas_height,
        mask_atlas_width,
        mask_atlas_height,
    }
}

/// GPU handles the retained worker was recorded against (debug invariant checks only).
#[cfg(debug_assertions)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorkerResourceHandles {
    pub scene: ResourceHandle,
    pub bump: ResourceHandle,
    pub gradient: ResourceHandle,
    pub image_atlas: ResourceHandle,
    pub mask_atlas: ResourceHandle,
    pub out_image: ResourceHandle,
}

#[cfg(debug_assertions)]
pub(crate) fn worker_resource_handles(
    scene: &Buffer,
    bump: &Buffer,
    gradient: &Texture,
    image_atlas: &Texture,
    mask_atlas: &Texture,
    out_image: &Texture,
) -> WorkerResourceHandles {
    WorkerResourceHandles {
        scene: scattered_buffer_handle(scene),
        bump: scattered_buffer_handle(bump),
        gradient: sampled_texture_handle(gradient),
        image_atlas: sampled_texture_handle(image_atlas),
        mask_atlas: sampled_texture_handle(mask_atlas),
        out_image: out_image
            .handle(ResourceAccess::Write)
            .expect("out_image must be writable"),
    }
}

#[cfg(debug_assertions)]
fn scattered_buffer_handle(buf: &Buffer) -> ResourceHandle {
    buf.handle(ResourceAccess::ReadWrite)
        .expect("retained worker buffer must expose a UAV handle")
}

#[cfg(debug_assertions)]
fn sampled_texture_handle(tex: &Texture) -> ResourceHandle {
    tex.handle(ResourceAccess::Read)
        .expect("retained worker sampled texture must expose an SRV handle")
}

/// True when the retained worker must be replaced and re-recorded.
///
/// The topology comparison covers both graph-structure inputs (AA, resolution, …) *and*
/// resource-identity inputs (`scene_bucket`, `mask_atlas_width/height`) that, if changed,
/// mean the worker's recorded dispatch nodes bind stale `ResourceHandle`s.
pub(crate) fn worker_stale(
    persistent: &PersistentState,
    topology: &WorkerTopology,
    filter_effects: &[LayerFilterEffect],
    out_image: ResourceHandle,
) -> bool {
    persistent.cached_worker_out_image != Some(out_image)
        || persistent.cached_worker_topology.as_ref() != Some(topology)
        || !layer_filter_effects_eq(&persistent.cached_worker_filter_effects, filter_effects)
}

/// Retained resubmit assumes worker-bound resources keep the same GPU handles.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_retained_worker_resources(
    recorded: &WorkerResourceHandles,
    current: &WorkerResourceHandles,
) {
    debug_assert_eq!(
        recorded.scene, current.scene,
        "retained worker scene buffer handle changed without worker invalidation"
    );
    debug_assert_eq!(
        recorded.bump, current.bump,
        "retained worker bump buffer handle changed without worker invalidation"
    );
    debug_assert_eq!(
        recorded.gradient, current.gradient,
        "retained worker gradient texture handle changed without worker invalidation"
    );
    debug_assert_eq!(
        recorded.image_atlas, current.image_atlas,
        "retained worker image atlas handle changed without worker invalidation"
    );
    debug_assert_eq!(
        recorded.mask_atlas, current.mask_atlas,
        "retained worker mask atlas handle changed without worker invalidation"
    );
    debug_assert_eq!(
        recorded.out_image, current.out_image,
        "retained worker out_image handle changed without worker invalidation"
    );
}

fn layer_filter_effects_eq(a: &[LayerFilterEffect], b: &[LayerFilterEffect]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(left, right)| {
            left.layer_index == right.layer_index
                && left.is_nested == right.is_nested
                && left.layer_blend == right.layer_blend
                && filter_primitive_eq(&left.primitive, &right.primitive)
        })
}

fn filter_primitive_eq(a: &FilterPrimitive, b: &FilterPrimitive) -> bool {
    use FilterPrimitive::*;
    match (a, b) {
        (
            Flood {
                color: color_a,
                clip_rect: rect_a,
            },
            Flood {
                color: color_b,
                clip_rect: rect_b,
            },
        ) => rect_a == rect_b && premul_color_eq(color_a, color_b),
        (
            GaussianBlur {
                std_dev: std_a,
                edge_mode: edge_a,
            },
            GaussianBlur {
                std_dev: std_b,
                edge_mode: edge_b,
            },
        ) => edge_a == edge_b && (std_a - std_b).abs() <= f32::EPSILON,
        (
            DropShadow {
                dx: dx_a,
                dy: dy_a,
                std_dev: std_a,
                color: color_a,
                edge_mode: edge_a,
            },
            DropShadow {
                dx: dx_b,
                dy: dy_b,
                std_dev: std_b,
                color: color_b,
                edge_mode: edge_b,
            },
        ) => {
            edge_a == edge_b
                && (dx_a - dx_b).abs() <= f32::EPSILON
                && (dy_a - dy_b).abs() <= f32::EPSILON
                && (std_a - std_b).abs() <= f32::EPSILON
                && premul_color_eq(color_a, color_b)
        }
        (Offset { dx: dx_a, dy: dy_a }, Offset { dx: dx_b, dy: dy_b }) => {
            (dx_a - dx_b).abs() <= f32::EPSILON && (dy_a - dy_b).abs() <= f32::EPSILON
        }
        _ => false,
    }
}

fn premul_color_eq(
    a: &peniko::color::PremulColor<peniko::color::Srgb>,
    b: &peniko::color::PremulColor<peniko::color::Srgb>,
) -> bool {
    a.to_rgba8().to_u32() == b.to_rgba8().to_u32()
}

/// True when the upload scheme must be replaced and its copy/upload nodes re-recorded.
///
/// The key covers every resource the upload scheme copies *into* — scene buffer, gradient
/// atlas, image atlas, and mask atlas.  A change in any dimension means the target
/// `ResourceHandle` was reallocated; the retained upload scheme would then copy into a
/// resource that no longer backs the current frame's pipeline.
pub(crate) fn upload_stale(persistent: &PersistentState, key: &UploadKey) -> bool {
    persistent.cached_upload_key.as_ref() != Some(key)
}

/// Round `bytes` up to the next power of two (minimum 4) for stable scene buffer reuse.
///
/// The scene buffer is bound by ResourceHandle in both the worker and upload schemes.
/// Bucketing prevents churn on minor scene-size fluctuations while still producing a
/// stable handle across frames that stay within the same bucket.
///
/// Note: `bump_alloc = BufferSize::new(1)` is constant regardless of scene complexity,
/// so the bump buffer handle never changes between frames — the upload scheme's clear
/// node always targets the same allocation.
pub(crate) fn scene_size_bucket(bytes: usize) -> u64 {
    bytes.max(4).next_power_of_two() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use peniko::color::palette::css;

    use peniko::color::{AlphaColor, Srgb};
    use ekrano_encoding::FilterEdgeMode;

    fn premul_srgb(color: AlphaColor<Srgb>) -> peniko::color::PremulColor<Srgb> {
        color.premultiply()
    }

    fn base_upload_key() -> UploadKey {
        upload_key(256, 8, 4, 0, 0, 0, None)
    }

    // -----------------------------------------------------------------------
    // upload_key normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn upload_key_empty_gradient_normalises_to_1x1() {
        let k = upload_key(64, 32, 0, 0, 0, 0, None);
        assert_eq!((k.gradient_width, k.gradient_height), (1, 1));
    }

    #[test]
    fn upload_key_nonempty_gradient_uses_raw_dims() {
        let k = upload_key(64, 32, 4, 0, 0, 0, None);
        assert_eq!((k.gradient_width, k.gradient_height), (32, 4));
    }

    #[test]
    fn upload_key_empty_image_atlas_normalises_to_1x1() {
        let k = upload_key(64, 8, 4, 0, 512, 512, None);
        assert_eq!((k.image_atlas_width, k.image_atlas_height), (1, 1));
    }

    #[test]
    fn upload_key_nonempty_image_atlas_uses_raw_dims() {
        let k = upload_key(64, 8, 4, 3, 512, 256, None);
        assert_eq!((k.image_atlas_width, k.image_atlas_height), (512, 256));
    }

    #[test]
    fn upload_key_no_coverage_mask_normalises_to_1x1() {
        let k = upload_key(64, 8, 4, 0, 0, 0, None);
        assert_eq!((k.mask_atlas_width, k.mask_atlas_height), (1, 1));
    }

    #[test]
    fn upload_key_coverage_mask_uses_mask_dims() {
        let k = upload_key(64, 8, 4, 0, 0, 0, Some((128, 64)));
        assert_eq!((k.mask_atlas_width, k.mask_atlas_height), (128, 64));
    }

    // -----------------------------------------------------------------------
    // upload_stale predicate
    // -----------------------------------------------------------------------

    #[test]
    fn upload_stale_false_when_key_unchanged() {
        let mut p = PersistentState::new_test_only();
        let k = base_upload_key();
        p.cached_upload_key = Some(k.clone());
        assert!(!upload_stale(&p, &k));
    }

    #[test]
    fn upload_stale_true_when_no_cached_key() {
        let p = PersistentState::new_test_only();
        let k = base_upload_key();
        assert!(upload_stale(&p, &k));
    }

    #[test]
    fn upload_stale_true_on_scene_bucket_growth() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None);
        let k2 = upload_key(512, 8, 4, 0, 0, 0, None);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_gradient_dim_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None);
        let k2 = upload_key(256, 8, 16, 0, 0, 0, None);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_when_gradient_appears() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 0, 0, 0, 0, 0, None);
        let k2 = upload_key(256, 8, 4, 0, 0, 0, None);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_image_atlas_dim_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 3, 512, 256, None);
        let k2 = upload_key(256, 8, 4, 3, 512, 512, None);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_when_image_atlas_appears() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None);
        let k2 = upload_key(256, 8, 4, 2, 256, 256, None);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_mask_atlas_dim_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, Some((64, 64)));
        let k2 = upload_key(256, 8, 4, 0, 0, 0, Some((128, 64)));
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_when_mask_appears() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None);
        let k2 = upload_key(256, 8, 4, 0, 0, 0, Some((64, 64)));
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    // -----------------------------------------------------------------------
    // filter_primitive equality (retained from original tests)
    // -----------------------------------------------------------------------

    #[test]
    fn filter_effects_eq_matches_identical() {
        let a = LayerFilterEffect {
            primitive: FilterPrimitive::GaussianBlur {
                std_dev: 2.0,
                edge_mode: FilterEdgeMode::Duplicate,
            },
            layer_blend: 1,
            layer_alpha: 1.0,
            layer_index: 0,
            is_nested: false,
        };
        let b = a.clone();
        assert!(layer_filter_effects_eq(&[a], &[b]));
    }

    #[test]
    fn filter_effects_eq_detects_color_change() {
        let a = LayerFilterEffect {
            primitive: FilterPrimitive::Flood {
                color: premul_srgb(css::RED),
                clip_rect: [0, 0, 10, 10],
            },
            layer_blend: 1,
            layer_alpha: 1.0,
            layer_index: 0,
            is_nested: false,
        };
        let mut b = a.clone();
        if let FilterPrimitive::Flood { color, .. } = &mut b.primitive {
            *color = premul_srgb(css::BLUE);
        }
        assert!(!layer_filter_effects_eq(&[a], &[b]));
    }
}
