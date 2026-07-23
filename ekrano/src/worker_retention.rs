// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained worker-scheme invalidation (no per-frame hashing).

use goldy::types::{ResourceAccess, ResourceHandle, TextureFormat};
#[cfg(debug_assertions)]
use goldy::{Buffer, Texture};

use crate::AaConfig;
use crate::goldy_renderer::{PersistentState, SceneGrowthStats};
use ekrano_encoding::{BufferSizes, FilterPrimitive, LayerFilterEffect, RenderConfig};

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
    /// When true, fine/filter composites write directly to the present lease (no `out_image`).
    pub direct_present: bool,
    /// Scene byte bucket the worker was recorded against; change = new scene buffer handle.
    pub scene_bucket: u64,
    /// Normalized mask atlas dims (1×1 when no coverage mask).
    pub mask_atlas_width: u32,
    pub mask_atlas_height: u32,
}

/// Normalized atlas / scene dimensions shared by retention keys and prepare.
///
/// Empty atlases are represented as 1×1 so keys match the textures actually allocated.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct ResourceDims {
    pub scene_bucket: u64,
    pub gradient_width: u32,
    pub gradient_height: u32,
    pub image_atlas_width: u32,
    pub image_atlas_height: u32,
    pub mask_atlas_width: u32,
    pub mask_atlas_height: u32,
    pub image_count: usize,
    /// Sorted unique `(x, y, width, height)` region copy keys for the image atlas.
    pub image_regions: Vec<(u32, u32, u32, u32)>,
    pub ramps_width: u32,
    pub ramps_height: u32,
    pub images_width: u32,
    pub images_height: u32,
    pub has_coverage_mask: bool,
}

/// `ramps_height == 0` → 1×1 gradient atlas (prepare sentinel).
pub(crate) fn normalize_gradient_atlas(width: u32, height: u32) -> (u32, u32) {
    if height == 0 { (1, 1) } else { (width, height) }
}

/// `image_count == 0` → 1×1 image atlas (prepare sentinel).
pub(crate) fn normalize_image_atlas(image_count: usize, width: u32, height: u32) -> (u32, u32) {
    if image_count == 0 { (1, 1) } else { (width, height) }
}

/// `None` coverage mask → 1×1 mask atlas (prepare sentinel).
pub(crate) fn normalize_mask_atlas(coverage_mask_dims: Option<(u32, u32)>) -> (u32, u32) {
    coverage_mask_dims.unwrap_or((1, 1))
}

/// Build normalized resource dimensions from per-frame resolve outputs.
pub(crate) fn resource_dims(
    scene_bucket: u64,
    ramps_width: u32,
    ramps_height: u32,
    image_count: usize,
    images_width: u32,
    images_height: u32,
    coverage_mask_dims: Option<(u32, u32)>,
    image_regions: &[(u32, u32, u32, u32)],
) -> ResourceDims {
    let (gradient_width, gradient_height) = normalize_gradient_atlas(ramps_width, ramps_height);
    let (image_atlas_width, image_atlas_height) = normalize_image_atlas(image_count, images_width, images_height);
    let (mask_atlas_width, mask_atlas_height) = normalize_mask_atlas(coverage_mask_dims);
    let mut image_regions = image_regions.to_vec();
    image_regions.sort_unstable();
    image_regions.dedup();
    ResourceDims {
        scene_bucket,
        gradient_width,
        gradient_height,
        image_atlas_width,
        image_atlas_height,
        mask_atlas_width,
        mask_atlas_height,
        image_count,
        image_regions,
        ramps_width,
        ramps_height,
        images_width,
        images_height,
        has_coverage_mask: coverage_mask_dims.is_some(),
    }
}

pub(crate) fn worker_topology(
    params: &crate::RenderParams,
    config: &RenderConfig,
    out_format: TextureFormat,
    dims: &ResourceDims,
    swapchain_present: bool,
    direct_present: bool,
) -> WorkerTopology {
    WorkerTopology {
        aa: params.antialiasing_method,
        robust: params.robust,
        out_format,
        width: params.width,
        height: params.height,
        buffer_sizes: config.buffer_sizes,
        has_coverage_mask: dims.has_coverage_mask,
        ramps_width: dims.ramps_width,
        ramps_height: dims.ramps_height,
        images_width: dims.images_width,
        images_height: dims.images_height,
        image_count: dims.image_count,
        swapchain_present,
        direct_present,
        scene_bucket: dims.scene_bucket,
        mask_atlas_width: dims.mask_atlas_width,
        mask_atlas_height: dims.mask_atlas_height,
    }
}

/// All resources the upload scheme writes into, keyed by the dimensions that determine
/// their GPU handle identity.  A change in any field means at least one upload copy node
/// references a stale `ResourceHandle` and the scheme must be re-recorded.
///
/// Dimensions are normalised to the actual texture sizes allocated by
/// `prepare_pipeline_resources` (e.g. 1×1 for an empty atlas) so that the key matches
/// what is stored in `PersistentState::cached_gradient` etc.
///
/// `image_regions` captures every region copy shape so retained upload topology cannot
/// silently mismatch when atlas dims stay fixed but region layout changes.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct UploadKey {
    pub scene_bucket: u64,
    pub gradient_width: u32,
    pub gradient_height: u32,
    pub image_atlas_width: u32,
    pub image_atlas_height: u32,
    pub mask_atlas_width: u32,
    pub mask_atlas_height: u32,
    /// Sorted unique `(x, y, width, height)` region copy keys for the image atlas.
    pub image_regions: Vec<(u32, u32, u32, u32)>,
}

/// Build an [`UploadKey`] from normalized [`ResourceDims`].
pub(crate) fn upload_key_from(dims: &ResourceDims) -> UploadKey {
    UploadKey {
        scene_bucket: dims.scene_bucket,
        gradient_width: dims.gradient_width,
        gradient_height: dims.gradient_height,
        image_atlas_width: dims.image_atlas_width,
        image_atlas_height: dims.image_atlas_height,
        mask_atlas_width: dims.mask_atlas_width,
        mask_atlas_height: dims.mask_atlas_height,
        image_regions: dims.image_regions.clone(),
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
    pub out_image: Option<ResourceHandle>,
}

#[cfg(debug_assertions)]
pub(crate) fn worker_resource_handles(
    scene: &Buffer,
    bump: &Buffer,
    gradient: &Texture,
    image_atlas: &Texture,
    mask_atlas: &Texture,
    out_image: Option<&Texture>,
) -> WorkerResourceHandles {
    WorkerResourceHandles {
        scene: scattered_buffer_handle(scene),
        bump: scattered_buffer_handle(bump),
        gradient: sampled_texture_handle(gradient),
        image_atlas: sampled_texture_handle(image_atlas),
        mask_atlas: sampled_texture_handle(mask_atlas),
        out_image: out_image.and_then(|tex| tex.handle(ResourceAccess::Write)),
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
///
/// Non-empty filter effects always force a re-record: `record_filter_effects` binds
/// per-frame scratch textures that are returned to the transient pool after submit.
/// Those scratches are true one-shot deeds — retaining the worker would resubmit a
/// scheme whose stamps were retired on return (`GoldyError::StaleResource`).
pub(crate) fn worker_stale_reasons(
    persistent: &PersistentState,
    topology: &WorkerTopology,
    filter_effects: &[LayerFilterEffect],
    out_image: Option<ResourceHandle>,
    output_texture: Option<goldy::TextureHandle>,
) -> bool {
    // Filter scratches outrank retention: cannot keep a scheme that bound returned deeds.
    if !filter_effects.is_empty() {
        return true;
    }
    let out_image_mismatch = if topology.direct_present {
        false
    } else {
        persistent.cached_worker_out_image != out_image
    };
    let output_texture_mismatch = persistent.cached_worker_output_texture != output_texture;
    let topology_mismatch = persistent.cached_worker_topology.as_ref() != Some(topology);
    let filter_effects_mismatch = !layer_filter_effects_eq(&persistent.cached_worker_filter_effects, filter_effects);
    out_image_mismatch || output_texture_mismatch || topology_mismatch || filter_effects_mismatch
}

/// Predict worker staleness *before* `prepare_pipeline_resources` acquires resources.
///
/// Peeks the cached scheme `out_image` handle when dimensions still match; a cache miss
/// means prepare will allocate a new RT (new handle) and the worker is therefore stale.
/// Used on the Metal fused path to skip a throwaway first prepare that would consume
/// `cached_pipeline` / RTs and force a duplicate allocation spike on re-prepare.
pub(crate) fn predict_worker_stale(
    persistent: &PersistentState,
    topology: &WorkerTopology,
    filter_effects: &[LayerFilterEffect],
    output_texture: Option<goldy::TextureHandle>,
    width: u32,
    height: u32,
    out_format: TextureFormat,
) -> bool {
    // Same rule as `worker_stale_reasons`: filter frames always re-record.
    if !filter_effects.is_empty() {
        return true;
    }
    if persistent.cached_worker_topology.as_ref() != Some(topology) {
        return true;
    }
    if !layer_filter_effects_eq(&persistent.cached_worker_filter_effects, filter_effects) {
        return true;
    }
    if persistent.cached_worker_output_texture != output_texture {
        return true;
    }
    if topology.direct_present {
        return false;
    }
    match &persistent.cached_scheme_rt {
        Some((Some(out), _)) if out.width() == width && out.height() == height && out.format() == out_format => {
            let handle = out
                .handle(ResourceAccess::Write)
                .expect("cached scheme out_image must be writable");
            persistent.cached_worker_out_image != Some(handle)
        }
        Some((None, _)) => false,
        _ => true,
    }
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
        ) => rect_a == rect_b && premul_color_eq(*color_a, *color_b),
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
                && premul_color_eq(*color_a, *color_b)
        }
        (Offset { dx: dx_a, dy: dy_a }, Offset { dx: dx_b, dy: dy_b }) => {
            (dx_a - dx_b).abs() <= f32::EPSILON && (dy_a - dy_b).abs() <= f32::EPSILON
        }
        _ => false,
    }
}

fn premul_color_eq(
    a: peniko::color::PremulColor<peniko::color::Srgb>,
    b: peniko::color::PremulColor<peniko::color::Srgb>,
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
/// The scene buffer is bound by `ResourceHandle` in both the worker and upload schemes.
/// Bucketing prevents churn on minor scene-size fluctuations while still producing a
/// stable handle across frames that stay within the same bucket.
///
/// Note: `bump_alloc = BufferSize::new(1)` is constant regardless of scene complexity,
/// so the bump buffer handle never changes between frames — the upload scheme's clear
/// node always targets the same allocation.
pub(crate) fn scene_size_bucket(bytes: usize) -> u64 {
    bytes.max(4).next_power_of_two() as u64
}

/// Update per-frame scene growth counters.
pub(crate) fn note_scene_growth_frame(stats: &mut SceneGrowthStats, live_bytes: usize, scene_bucket: u64) {
    stats.frames += 1;
    stats.current_scene_bucket = scene_bucket;
    stats.peak_scene_bucket = stats.peak_scene_bucket.max(scene_bucket);
    stats.peak_live_scene_bytes = stats.peak_live_scene_bytes.max(live_bytes as u64);
}

/// Log and count a scene buffer bucket crossing (physical reallocation).
pub(crate) fn note_scene_bucket_crossing(
    stats: &mut SceneGrowthStats,
    old_bucket: u64,
    new_bucket: u64,
    live_bytes: usize,
) {
    stats.scene_bucket_crossings += 1;
    log::info!(
        target: "ekrano::scene_growth",
        "scene buffer bucket crossing: old_bucket={old_bucket} new_bucket={new_bucket} live_bytes={live_bytes}"
    );
}

/// Log and count a worker topology invalidation driven by scene bucket growth.
pub(crate) fn note_worker_rerecord_scene_bucket(stats: &mut SceneGrowthStats, old_bucket: u64, new_bucket: u64) {
    stats.worker_rerecord_scene_bucket += 1;
    log::info!(
        target: "ekrano::scene_growth",
        "worker topology invalidation (scene bucket): old_bucket={old_bucket} new_bucket={new_bucket}"
    );
}

/// Log and count an upload topology invalidation driven by scene bucket growth.
pub(crate) fn note_upload_rerecord_scene_bucket(stats: &mut SceneGrowthStats, old_bucket: u64, new_bucket: u64) {
    stats.upload_rerecord_scene_bucket += 1;
    log::info!(
        target: "ekrano::scene_growth",
        "upload topology invalidation (scene bucket): old_bucket={old_bucket} new_bucket={new_bucket}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use peniko::color::palette::css;

    use ekrano_encoding::BufferSizes;
    use ekrano_encoding::FilterEdgeMode;
    use peniko::color::{AlphaColor, Srgb};

    fn premul_srgb(color: AlphaColor<Srgb>) -> peniko::color::PremulColor<Srgb> {
        color.premultiply()
    }

    fn upload_key(
        scene_bucket: u64,
        ramps_width: u32,
        ramps_height: u32,
        image_count: usize,
        images_width: u32,
        images_height: u32,
        coverage_mask_dims: Option<(u32, u32)>,
        image_regions: &[(u32, u32, u32, u32)],
    ) -> UploadKey {
        upload_key_from(&resource_dims(
            scene_bucket,
            ramps_width,
            ramps_height,
            image_count,
            images_width,
            images_height,
            coverage_mask_dims,
            image_regions,
        ))
    }

    fn base_upload_key() -> UploadKey {
        upload_key(256, 8, 4, 0, 0, 0, None, &[])
    }

    // -----------------------------------------------------------------------
    // upload_key normalisation
    // -----------------------------------------------------------------------

    #[test]
    fn upload_key_empty_gradient_normalises_to_1x1() {
        let k = upload_key(64, 32, 0, 0, 0, 0, None, &[]);
        assert_eq!((k.gradient_width, k.gradient_height), (1, 1));
    }

    #[test]
    fn upload_key_nonempty_gradient_uses_raw_dims() {
        let k = upload_key(64, 32, 4, 0, 0, 0, None, &[]);
        assert_eq!((k.gradient_width, k.gradient_height), (32, 4));
    }

    #[test]
    fn upload_key_empty_image_atlas_normalises_to_1x1() {
        let k = upload_key(64, 8, 4, 0, 512, 512, None, &[]);
        assert_eq!((k.image_atlas_width, k.image_atlas_height), (1, 1));
    }

    #[test]
    fn upload_key_nonempty_image_atlas_uses_raw_dims() {
        let k = upload_key(64, 8, 4, 3, 512, 256, None, &[]);
        assert_eq!((k.image_atlas_width, k.image_atlas_height), (512, 256));
    }

    #[test]
    fn upload_key_no_coverage_mask_normalises_to_1x1() {
        let k = upload_key(64, 8, 4, 0, 0, 0, None, &[]);
        assert_eq!((k.mask_atlas_width, k.mask_atlas_height), (1, 1));
    }

    #[test]
    fn upload_key_coverage_mask_uses_mask_dims() {
        let k = upload_key(64, 8, 4, 0, 0, 0, Some((128, 64)), &[]);
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
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None, &[]);
        let k2 = upload_key(512, 8, 4, 0, 0, 0, None, &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_gradient_dim_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None, &[]);
        let k2 = upload_key(256, 8, 16, 0, 0, 0, None, &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_when_gradient_appears() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 0, 0, 0, 0, 0, None, &[]);
        let k2 = upload_key(256, 8, 4, 0, 0, 0, None, &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_image_atlas_dim_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 3, 512, 256, None, &[]);
        let k2 = upload_key(256, 8, 4, 3, 512, 512, None, &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_when_image_atlas_appears() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None, &[]);
        let k2 = upload_key(256, 8, 4, 2, 256, 256, None, &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_mask_atlas_dim_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, Some((64, 64)), &[]);
        let k2 = upload_key(256, 8, 4, 0, 0, 0, Some((128, 64)), &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_when_mask_appears() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 0, 0, 0, None, &[]);
        let k2 = upload_key(256, 8, 4, 0, 0, 0, Some((64, 64)), &[]);
        p.cached_upload_key = Some(k1);
        assert!(upload_stale(&p, &k2));
    }

    #[test]
    fn upload_stale_true_on_image_region_layout_change() {
        let mut p = PersistentState::new_test_only();
        let k1 = upload_key(256, 8, 4, 2, 256, 256, None, &[(0, 0, 32, 32)]);
        let k2 = upload_key(256, 8, 4, 2, 256, 256, None, &[(0, 0, 32, 32), (40, 0, 16, 16)]);
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

    fn sample_topology(direct_present: bool) -> WorkerTopology {
        WorkerTopology {
            aa: AaConfig::Area,
            robust: false,
            out_format: TextureFormat::Rgba8Unorm,
            width: 64,
            height: 64,
            buffer_sizes: BufferSizes::default(),
            has_coverage_mask: false,
            ramps_width: 1,
            ramps_height: 1,
            images_width: 1,
            images_height: 1,
            image_count: 0,
            swapchain_present: true,
            direct_present,
            scene_bucket: 256,
            mask_atlas_width: 1,
            mask_atlas_height: 1,
        }
    }

    #[test]
    fn worker_stale_when_direct_present_mode_changes() {
        let mut p = PersistentState::new_test_only();
        let copy_topo = sample_topology(false);
        let direct_topo = sample_topology(true);
        p.cached_worker_topology = Some(copy_topo.clone());
        assert!(worker_stale_reasons(&p, &direct_topo, &[], None, None,));
        p.cached_worker_topology = Some(direct_topo.clone());
        assert!(!worker_stale_reasons(&p, &direct_topo, &[], None, None,));
    }

    #[test]
    fn worker_stale_when_filter_effects_non_empty() {
        let mut p = PersistentState::new_test_only();
        let topo = sample_topology(false);
        p.cached_worker_topology = Some(topo.clone());
        let effect = LayerFilterEffect {
            primitive: FilterPrimitive::GaussianBlur {
                std_dev: 2.0,
                edge_mode: FilterEdgeMode::Duplicate,
            },
            layer_blend: 1,
            layer_alpha: 1.0,
            layer_index: 0,
            is_nested: false,
        };
        // Cached effects match — descriptor equality alone is not enough to retain.
        p.cached_worker_filter_effects = vec![effect.clone()];
        assert!(
            worker_stale_reasons(&p, &topo, &[effect], None, None),
            "filter frames bind per-submit scratches; worker must not be retained"
        );
        assert!(predict_worker_stale(
            &p,
            &topo,
            &[LayerFilterEffect {
                primitive: FilterPrimitive::GaussianBlur {
                    std_dev: 2.0,
                    edge_mode: FilterEdgeMode::Duplicate,
                },
                layer_blend: 1,
                layer_alpha: 1.0,
                layer_index: 0,
                is_nested: false,
            }],
            None,
            64,
            64,
            TextureFormat::Rgba8Unorm,
        ));
    }

    #[test]
    fn predict_worker_stale_false_in_direct_present_without_out_image() {
        let mut p = PersistentState::new_test_only();
        let topo = sample_topology(true);
        p.cached_worker_topology = Some(topo.clone());
        assert!(!predict_worker_stale(
            &p,
            &topo,
            &[],
            None,
            64,
            64,
            TextureFormat::Rgba8Unorm,
        ));
    }
}
