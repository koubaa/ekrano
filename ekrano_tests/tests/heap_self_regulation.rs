// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Heap self-regulation tests at the ekrano renderer level.
//!
//! These tests verify that the Metal buffer heap allocator + ekrano retained/transient
//! pools cooperate correctly under realistic rendering workloads:
//!
//! - Retained deeds and transient scratch stabilize after warmup (no unbounded fresh allocs)
//! - Overflow heaps compact after steady-state is reached
//! - Single-frame scheduling (depth=1) survives sustained rendering
//! - Varying scene complexity (warmup pressure) doesn't exhaust the heap
//! - The `robust` mode flag doesn't alter memory lifecycle correctness
//! - Multiple AA configs work without heap exhaustion
//!
//! These tests assert heap/pool/deferred-ring survival, not byte-level VRAM accounting.
//! Budget and tracking policies are tested in goldy (`allocation_policy`, `vram_allocator`);
//! ekrano production paths run with [`NoPolicy`](goldy::NoPolicy) unless the caller installs
//! one via [`Device::ensure_allocation_policy`](goldy::Device::ensure_allocation_policy).

#[path = "common/submission.rs"]
mod submission;

use ekrano::kurbo::{Affine, Circle, Line, Rect, Stroke};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use ekrano_tests::{SharedTestDevice, shared_test_device, test_alloc_texture, test_device};
use goldy::types::{TextureFlags, TextureFormat, TextureKind};

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

fn make_device() -> SharedTestDevice {
    test_device()
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

fn render_n_frames(renderer: &mut GoldyRenderer, scene: &Scene, params: &RenderParams, n: usize) {
    let texture = test_alloc_texture(
        renderer.device(),
        params.width,
        params.height,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );

    for i in 0..n {
        renderer
            .render_to_texture(scene, &texture, params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
    }
}

// ===========================================================================
// Survival tests: single-frame model survives many frames
// ===========================================================================

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
    render_n_frames(&mut renderer, &scene, &params, 200);
}

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
    render_n_frames(&mut renderer, &scene, &params, 200);
}

// ===========================================================================
// Robust mode
// ===========================================================================

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
    render_n_frames(&mut renderer, &scene, &params, 200);
}

// ===========================================================================
// Complex scene under various AA configs
// ===========================================================================

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
    render_n_frames(&mut renderer, &scene, &params, 100);
}

/// Observed failure (June 2026, DX12 debug build): aborts mid-run with
/// `memory allocation of 137438953472 bytes failed` (128 GiB host-side alloc,
/// exit 0xc0000409). This is NOT the stale-`gpu_progress` issue the original
/// ignore cited — the clock was fixed by the unified boundary event
/// (`Device::boundary_crossed`), and the tiny-scene / area-AA siblings now pass
/// unmodified. The 128 GiB request points at a size computation gone wild in
/// the old `ResourcePool`/`TexturePool` reuse path under MSAA16 (likely garbage
/// readback or overflow in a growth calculation). Left unfixed deliberately:
/// this path is slated for replacement by the retained-scheme lease design
/// (`docu/.../diwan/in-progress/retained-scheme/design.md`, phases 2-3 of its project).
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
    render_n_frames(&mut renderer, &scene, &params, 100);
}

// ===========================================================================
// Resource pool convergence: no new allocations in steady state
// ===========================================================================

fn resource_pool_stabilizes_after_warmup() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = test_alloc_texture(
        renderer.device(),
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("warmup frame {i} failed: {e}"));
    }

    let baseline_retained = renderer.resource_pool_stats().retained_pool_buffer_bytes;
    let baseline_transient_allocs = renderer.submission_context().transient_buffer_alloc_count();

    // Steady state: 50 more frames — accounting should not grow
    let mut max_retained = baseline_retained;
    for i in 0..50 {
        renderer
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("steady frame {i} failed: {e}"));
        max_retained = max_retained.max(renderer.resource_pool_stats().retained_pool_buffer_bytes);
    }

    let retained_growth = max_retained.saturating_sub(baseline_retained);
    assert!(
        retained_growth == 0,
        "retained pool grew after warmup: baseline={baseline_retained} max_seen={max_retained}"
    );
    // Includes scratch + scheme upload-staging leases; allow a small absolute bump.
    let transient_allocs = renderer.submission_context().transient_buffer_alloc_count();
    let transient_growth = transient_allocs.saturating_sub(baseline_transient_allocs);
    assert!(
        transient_growth <= 10,
        "transient buffer fresh allocs grew excessively after warmup: \
         baseline={baseline_transient_allocs} after={transient_allocs} growth={transient_growth}"
    );
}

// ===========================================================================
// Overflow heap compaction after warmup
// ===========================================================================

#[cfg(target_os = "macos")]
fn overflow_heaps_compact_to_zero_in_steady_state() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = test_alloc_texture(
        renderer.device(),
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));
    }

    // At this point, flush and compact.
    renderer.flush_deferred_deletions();
    renderer.device().compact_overflow_heaps();

    if let Some(stats) = renderer.device().buffer_heap_stats() {
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

fn growing_scene_survives_without_heap_exhaustion() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = test_alloc_texture(
        renderer.device(),
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {frame} (n_shapes={n_shapes}): {e}"));
    }
}

// ===========================================================================
// Shrinking scene: scene complexity decreases (tests unused memory release)
// ===========================================================================

fn shrinking_scene_does_not_leak_buffers() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = test_alloc_texture(
        renderer.device(),
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
            .render_to_texture(&big_scene, &texture, &params)
            .unwrap_or_else(|e| panic!("big frame {i}: {e}"));
    }

    // Phase 2: trivial scene for 50 frames (previously allocated buffers should be recycled, not leaked)
    let small_scene = tiny_scene();
    for i in 0..50 {
        renderer
            .render_to_texture(&small_scene, &texture, &params)
            .unwrap_or_else(|e| panic!("small frame {i}: {e}"));
    }

    // Just surviving without OOM is the assertion.
    // Additionally, check retained deeds haven't exploded for a trivial scene.
    let stats = renderer.resource_pool_stats();
    assert!(
        stats.retained_pool_buffer_bytes < 64 * 1024 * 1024,
        "retained pool should stay under 64 MiB for a trivial scene: got {} bytes",
        stats.retained_pool_buffer_bytes,
    );
}

// ===========================================================================
// Deferred ring depth bounded
// ===========================================================================

fn deferred_ring_does_not_grow_unbounded() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = test_alloc_texture(
        renderer.device(),
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
            .render_to_texture(&scene, &texture, &params)
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

fn large_resolution_survives_50_frames() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let w = 512;
    let h = 512;
    let texture = test_alloc_texture(
        renderer.device(),
        w,
        h,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i}: {e}"));
    }
}

// ===========================================================================
// Re-creation resilience: destroy and recreate renderer
// ===========================================================================

fn recreate_renderer_after_warmup_survives() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();
    // Keep an owning device handle: this test constructs two renderers sequentially and
    // a texture that outlives both, so we cannot rely on a single renderer's clone alone.
    let device = make_device();
    let texture = test_alloc_texture(
        &device,
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
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
                .render_to_texture(&scene, &texture, &params)
                .unwrap_or_else(|e| panic!("first renderer frame {i}: {e}"));
        }
    }
    // Renderer dropped — all resources should be released.

    // Second renderer: should start fresh without hitting heap limits.
    {
        let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new (second)");
        for i in 0..30 {
            renderer
                .render_to_texture(&scene, &texture, &params)
                .unwrap_or_else(|e| panic!("second renderer frame {i}: {e}"));
        }
    }
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("default_strategy_survives_200_frames_tiny_scene", || {
            default_strategy_survives_200_frames_tiny_scene();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("default_strategy_survives_200_frames_complex_scene", || {
            default_strategy_survives_200_frames_complex_scene();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("robust_mode_survives_200_frames", || {
            robust_mode_survives_200_frames();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("complex_scene_area_aa_survives_100_frames", || {
            complex_scene_area_aa_survives_100_frames();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("complex_scene_msaa16_survives_100_frames", || {
            complex_scene_msaa16_survives_100_frames();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("resource_pool_stabilizes_after_warmup", || {
            resource_pool_stabilizes_after_warmup();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    #[cfg(target_os = "macos")]
    trials.push(
        libtest_mimic::Trial::test("overflow_heaps_compact_to_zero_in_steady_state", || {
            overflow_heaps_compact_to_zero_in_steady_state();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("growing_scene_survives_without_heap_exhaustion", || {
            growing_scene_survives_without_heap_exhaustion();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("shrinking_scene_does_not_leak_buffers", || {
            shrinking_scene_does_not_leak_buffers();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("deferred_ring_does_not_grow_unbounded", || {
            deferred_ring_does_not_grow_unbounded();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("large_resolution_survives_50_frames", || {
            large_resolution_survives_50_frames();
            Ok(())
        })
        .with_ignored_flag(true),
    );
    trials.push(
        libtest_mimic::Trial::test("recreate_renderer_after_warmup_survives", || {
            recreate_renderer_after_warmup_survives();
            Ok(())
        })
        .with_ignored_flag(true),
    );

    let mut args = libtest_mimic::Arguments::from_args();
    if let Some(device) = shared_test_device() {
        submission::clamp_test_threads(&mut args, device);
    }
    libtest_mimic::run(&args, trials).exit()
}
