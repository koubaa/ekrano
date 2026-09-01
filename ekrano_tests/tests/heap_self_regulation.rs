// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renderer-level memory lifecycle tests (pools, scene growth, AA / robust paths).
//!
//! Metal heap overflow compact, deferred-ring reclaim, and multi-frame alloc
//! survival without a scene live in Goldy (`heap_tests`). Sustained
//! `render_to_texture` ring/pool stability lives in `pipelined_memory`.

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
    if goldy::dx12_debug_mode() {
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
/// exit 0xc0000409). This is not heap overflow reclaim — the clock was fixed by
/// the unified boundary event (`Device::boundary_crossed`). The 128 GiB request
/// points at a size computation gone wild in the old `ResourcePool`/`TexturePool`
/// reuse path under MSAA16. Left unfixed deliberately: this path is slated for
/// replacement by the retained-scheme lease design
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

    for i in 0..30 {
        renderer
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("warmup frame {i} failed: {e}"));
    }

    let baseline_retained = renderer.resource_pool_stats().retained_pool_buffer_bytes;
    let baseline_transient_allocs = renderer.submission_context().transient_buffer_alloc_count();

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
    let transient_allocs = renderer.submission_context().transient_buffer_alloc_count();
    let transient_growth = transient_allocs.saturating_sub(baseline_transient_allocs);
    assert!(
        transient_growth <= 10,
        "transient buffer fresh allocs grew excessively after warmup: \
         baseline={baseline_transient_allocs} after={transient_allocs} growth={transient_growth}"
    );
}

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
        let n_shapes = 1 + frame * 2;
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

    let big_scene = complex_scene();
    for i in 0..30 {
        renderer
            .render_to_texture(&big_scene, &texture, &params)
            .unwrap_or_else(|e| panic!("big frame {i}: {e}"));
    }

    let small_scene = tiny_scene();
    for i in 0..50 {
        renderer
            .render_to_texture(&small_scene, &texture, &params)
            .unwrap_or_else(|e| panic!("small frame {i}: {e}"));
    }

    let stats = renderer.resource_pool_stats();
    assert!(
        stats.retained_pool_buffer_bytes < 64 * 1024 * 1024,
        "retained pool should stay under 64 MiB for a trivial scene: got {} bytes",
        stats.retained_pool_buffer_bytes,
    );
}

fn main() {
    let mut trials = Vec::new();
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

    let mut args = libtest_mimic::Arguments::from_args();
    if let Some(device) = shared_test_device() {
        submission::clamp_test_threads(&mut args, device);
    }
    libtest_mimic::run(&args, trials).exit()
}
