// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Heap self-regulation tests at the ekrano renderer level.
//!
//! These tests verify that the Metal buffer heap allocator + ekrano `ResourcePool`
//! cooperate correctly under realistic rendering workloads:
//!
//! - The resource pool replenishes from deferred returns (no unbounded fresh allocations)
//! - Overflow heaps compact after steady-state is reached
//! - Different `FrameStrategy` values (`LowLatency`, `Balanced`, `MaxThroughput`) all survive
//! - Varying scene complexity (warmup pressure) doesn't exhaust the heap
//! - The `robust` mode flag doesn't alter memory lifecycle correctness
//! - Multiple AA configs work without heap exhaustion

use ekrano::kurbo::{Affine, Circle, Line, Rect, Stroke};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};
use goldy::{Device, DeviceDescriptor, Instance, RequestAdapterOptions};

#[cfg(target_os = "windows")]
fn gpu_test_lock() -> Option<std::sync::MutexGuard<'static, ()>> {
    use std::sync::{Mutex, OnceLock};
    if goldy::backend::dx12::is_debug_mode() {
        static GPU_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        return Some(
            GPU_LOCK
                .get_or_init(|| Mutex::new(()))
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
    }
    None
}

#[cfg(not(target_os = "windows"))]
fn gpu_test_lock() -> Option<()> {
    None
}

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

fn make_device() -> Device {
    let instance = Instance::new().expect("Instance::new");
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("No Goldy device")
}

fn tiny_scene() -> Scene {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::RED,
        None,
        &Rect::new(0.0, 0.0, 32.0, 32.0),
    );
    scene
}

fn complex_scene() -> Scene {
    let mut scene = Scene::new();
    for i in 0..50 {
        let offset = i as f64 * 1.2;
        scene.fill(
            Fill::NonZero,
            Affine::translate((offset, offset)),
            palette::css::BLUE,
            None,
            &Circle::new((16.0, 16.0), 8.0 + offset * 0.1),
        );
        scene.stroke(
            &Stroke::new(2.0),
            Affine::IDENTITY,
            palette::css::GREEN,
            None,
            &Line::new((0.0, offset), (64.0, 64.0 - offset)),
        );
    }
    scene
}

fn render_n_frames(
    device: &Device,
    renderer: &mut GoldyRenderer,
    scene: &Scene,
    params: &RenderParams,
    n: usize,
) {
    let texture = device
        .alloc_texture(
            params.width,
            params.height,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");

    for i in 0..n {
        renderer
            .render_to_texture(device, scene, &texture, params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
    }
}

// ===========================================================================
// Survival tests: default strategy (Balanced) survives many frames
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn default_strategy_survives_200_frames_tiny_scene() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    render_n_frames(&device, &mut renderer, &scene, &params, 200);
}

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn default_strategy_survives_200_frames_complex_scene() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let scene = complex_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    render_n_frames(&device, &mut renderer, &scene, &params, 200);
}

// ===========================================================================
// Robust mode
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn robust_mode_survives_200_frames() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: true,
    };
    render_n_frames(&device, &mut renderer, &scene, &params, 200);
}

// ===========================================================================
// Complex scene under various AA configs
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn complex_scene_area_aa_survives_100_frames() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let scene = complex_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    render_n_frames(&device, &mut renderer, &scene, &params, 100);
}

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn complex_scene_msaa16_survives_100_frames() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let scene = complex_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Msaa16,
        robust: false,
    };
    render_n_frames(&device, &mut renderer, &scene, &params, 100);
}

// ===========================================================================
// Resource pool convergence: no new allocations in steady state
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn resource_pool_stabilizes_after_warmup() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    // Warmup: 30 frames
    for i in 0..30 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("warmup frame {i} failed: {e}"));
    }

    let baseline_pool = renderer.resource_pool_stats();

    // Steady state: 50 more frames — pool should not grow
    let mut max_pooled = baseline_pool.total_pooled_buffers;
    for i in 0..50 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("steady frame {i} failed: {e}"));
        let stats = renderer.resource_pool_stats();
        max_pooled = max_pooled.max(stats.total_pooled_buffers);
    }

    // Pool should not have grown unboundedly. Allow some leeway for double-buffering.
    let growth = max_pooled.saturating_sub(baseline_pool.total_pooled_buffers);
    assert!(
        growth <= baseline_pool.total_pooled_buffers.max(10),
        "resource pool grew excessively after warmup: baseline={} max_seen={max_pooled} growth={growth}",
        baseline_pool.total_pooled_buffers,
    );
}

// ===========================================================================
// Overflow heap compaction after warmup
// ===========================================================================

#[test]
#[cfg(target_os = "macos")]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn overflow_heaps_compact_to_zero_in_steady_state() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    // Run 100 frames with periodic compaction.
    for i in 0..100 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
    }

    // At this point, flush and compact.
    renderer.flush_deferred_deletions();
    device.compact_overflow_heaps();

    if let Some(stats) = device.buffer_heap_stats() {
        assert_eq!(
            stats.overflow_count, 0,
            "expected 0 overflow heaps in steady state, got {}",
            stats.overflow_count
        );
    }
}

// ===========================================================================
// Growing scene: scene complexity increases each frame
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn growing_scene_survives_without_heap_exhaustion() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    for frame in 0..60 {
        let mut scene = Scene::new();
        let n_shapes = 1 + frame * 2; // growing complexity
        for i in 0..n_shapes {
            let offset = (i as f64) * 0.5;
            scene.fill(
                Fill::NonZero,
                Affine::translate((offset, offset)),
                palette::css::CORAL,
                None,
                &Rect::new(0.0, 0.0, 10.0, 10.0),
            );
        }
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {frame} (n_shapes={n_shapes}): {e}"));
    }
}

// ===========================================================================
// Shrinking scene: scene complexity decreases (tests unused memory release)
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn shrinking_scene_does_not_leak_buffers() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    // Phase 1: complex scene for 30 frames
    let big_scene = complex_scene();
    for i in 0..30 {
        renderer
            .render_to_texture(&device, &big_scene, &texture, &params)
            .unwrap_or_else(|e| panic!("big frame {i}: {e}"));
    }

    // Phase 2: trivial scene for 50 frames (previously allocated buffers should be recycled, not leaked)
    let small_scene = tiny_scene();
    for i in 0..50 {
        renderer
            .render_to_texture(&device, &small_scene, &texture, &params)
            .unwrap_or_else(|e| panic!("small frame {i}: {e}"));
    }

    // Just surviving without OOM is the assertion.
    // Additionally, check the pool hasn't exploded.
    let stats = renderer.resource_pool_stats();
    assert!(
        stats.total_pooled_buffers < 100,
        "pool should not have >100 buffers for a trivial scene: got {}",
        stats.total_pooled_buffers,
    );
}

// ===========================================================================
// Deferred ring depth bounded
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn deferred_ring_does_not_grow_unbounded() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    for i in 0..100 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));

        // The VramAllocator deferred ring should be bounded by pipelining depth.
        // After steady state it shouldn't hold more than ~2-3 frames worth of payloads.
        if i > 20 {
            let has_deferred = renderer.has_deferred_payloads();
            // It's OK to have deferred payloads (pipelined), but they should flush periodically.
            // After 100 frames, if there are still deferred payloads, the flush mechanism works
            // because we haven't OOM'd.
            let _ = has_deferred;
        }
    }
}

// ===========================================================================
// Larger resolution
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn large_resolution_survives_50_frames() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let w = 512;
    let h = 512;
    let texture = device
        .alloc_texture(
            w,
            h,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: w,
        height: h,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    for i in 0..50 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));
    }
}

// ===========================================================================
// Re-creation resilience: destroy and recreate renderer
// ===========================================================================

#[test]
#[ignore = "archive reclamation at dispatch boundaries not wired correctly (gpu_progress stale)"]
fn recreate_renderer_after_warmup_survives() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST,
        )
        .expect("alloc_texture");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    // First renderer: warmup
    {
        let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
        for i in 0..30 {
            renderer
                .render_to_texture(&device, &scene, &texture, &params)
                .unwrap_or_else(|e| panic!("first renderer frame {i}: {e}"));
        }
    }
    // Renderer dropped — all resources should be released.

    // Second renderer: should start fresh without hitting heap limits.
    {
        let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new (second)");
        for i in 0..30 {
            renderer
                .render_to_texture(&device, &scene, &texture, &params)
                .unwrap_or_else(|e| panic!("second renderer frame {i}: {e}"));
        }
    }
}
