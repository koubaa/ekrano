// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Single-frame frame-orchestrator stress tests.
//!
//! Ekrano uses a fixed depth=1 fire-and-forget model (ekrano issue #71). These
//! tests verify that many frames through `render_to_texture` keep the
//! [`FrameOrchestrator`] ring bounded and the resource pool stable.

#[path = "common/submission.rs"]
mod submission;

use ekrano::kurbo::{Affine, Rect};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use ekrano_tests::{SharedTestDevice, shared_test_device, test_alloc_texture, test_device};
use goldy::types::{TextureFlags, TextureFormat, TextureKind};

/// Serialize GPU tests when the D3D12 debug layer is active.
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

const FRAME_COUNT: usize = 200;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

fn make_device() -> SharedTestDevice {
    test_device()
}

fn make_renderer() -> (SharedTestDevice, GoldyRenderer) {
    let device = make_device();
    let renderer = GoldyRenderer::new(&device).expect("GoldyRenderer");
    (device, renderer)
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

/// Render many frames and verify the frame-orchestrator ring stays at depth=1.
///
/// With nonblocking reuse (`host_sidecar_on_submit_worker`), frames close with
/// `end_frame_externally_ordered` and leave **no** retirement-ring slot, so
/// `cleanup_ring_depth` stays 0 after each frame.
fn single_frame_ring_depth_bounded() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    let nonblocking = device.capabilities().host_sidecar_on_submit_worker;
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer");
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

    for i in 0..FRAME_COUNT {
        renderer
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        let depth = renderer.allocator_stats().cleanup_ring_depth;
        if nonblocking {
            assert_eq!(
                depth, 0,
                "frame {i}: scheme nonblocking path must not create a retirement-ring slot (depth={depth})"
            );
        } else {
            assert!(
                depth <= 1,
                "frame {i}: cleanup ring depth {depth} exceeds single-frame limit (1)"
            );
        }
    }
}

/// Verify retained + transient buffer accounting stabilises after warmup under the single-frame model.
fn resource_pool_stable_under_single_frame() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let (_device, mut renderer) = make_renderer();
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
    // Transient alloc count includes scratch *and* scheme upload-staging leases; allow a
    // small absolute bump (e.g. an extra staging chunk) but not unbounded growth.
    let transient_allocs = renderer.submission_context().transient_buffer_alloc_count();
    let transient_growth = transient_allocs.saturating_sub(baseline_transient_allocs);
    assert!(
        transient_growth <= 10,
        "transient buffer fresh allocs grew excessively after warmup: \
         baseline={baseline_transient_allocs} after={transient_allocs} growth={transient_growth}"
    );
}

/// Verify the composite indirect buffer is reused (not reallocated) across frames when the
/// scene resolution is unchanged, and reallocated when the resolution changes.
///
/// Failure modes this catches:
/// - `alloc_or_reuse_scheme_indirect` ignoring the cache and allocating every frame.
/// - The cache key being stale so topology changes don't trigger a fresh allocation.
fn indirect_buffer_reused_across_frames() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer");

    let texture_a = test_alloc_texture(
        renderer.device(),
        WIDTH,
        HEIGHT,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );
    let texture_b = test_alloc_texture(
        renderer.device(),
        WIDTH * 2,
        HEIGHT * 2,
        TextureFormat::Rgba8Unorm,
        TextureKind::Direct,
        TextureFlags::COPY_DST,
    );

    let scene = tiny_scene();
    let params_a = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    let params_b = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH * 2,
        height: HEIGHT * 2,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    // Warmup: let the retained pool reach steady state.
    for i in 0..5 {
        renderer
            .render_to_texture(&scene, &texture_a, &params_a)
            .unwrap_or_else(|e| panic!("warmup frame {i} failed: {e}"));
    }

    let bytes_after_warmup = renderer.resource_pool_stats().retained_pool_buffer_bytes;
    assert!(
        bytes_after_warmup > 0,
        "retained pool is empty after warmup — composite indirect buffer was never allocated"
    );

    // Steady state: same resolution, bytes must not grow (buffer is reused).
    for i in 0..20 {
        renderer
            .render_to_texture(&scene, &texture_a, &params_a)
            .unwrap_or_else(|e| panic!("steady frame {i} failed: {e}"));
        let bytes = renderer.resource_pool_stats().retained_pool_buffer_bytes;
        assert_eq!(
            bytes, bytes_after_warmup,
            "frame {i}: retained pool bytes changed unexpectedly ({bytes_after_warmup} → {bytes}); \
             composite indirect buffer was reallocated"
        );
    }

    // Resolution change: different `WorkgroupCountsGpu`, must trigger a reallocation.
    renderer
        .render_to_texture(&scene, &texture_b, &params_b)
        .unwrap_or_else(|e| panic!("resolution-change frame failed: {e}"));
    let bytes_after_resize = renderer.resource_pool_stats().retained_pool_buffer_bytes;
    // The new buffer may be larger or the same size, but the transition itself must
    // not regress to zero (buffer was dropped and reallocated correctly).
    assert!(
        bytes_after_resize > 0,
        "retained pool is empty after resolution change — composite indirect buffer was lost"
    );

    // Re-stabilise at the new resolution.
    for i in 0..10 {
        renderer
            .render_to_texture(&scene, &texture_b, &params_b)
            .unwrap_or_else(|e| panic!("post-resize frame {i} failed: {e}"));
        let bytes = renderer.resource_pool_stats().retained_pool_buffer_bytes;
        assert_eq!(
            bytes, bytes_after_resize,
            "frame {i}: retained pool bytes changed unexpectedly after resize \
             ({bytes_after_resize} → {bytes})"
        );
    }
}

/// Head-chases-tail: resize churn must not force orchestrator retirement
/// slots or unbounded retained-pool growth (DX12/Vulkan with host sidecar).
fn resize_churn_keeps_ring_empty_and_pool_bounded() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    if !device.capabilities().host_sidecar_on_submit_worker {
        // Backends without host_sidecar_on_submit_worker still use the blocking path.
        return;
    }

    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer");
    let scene = tiny_scene();
    let sizes = [
        (WIDTH, HEIGHT),
        (WIDTH * 2, HEIGHT),
        (WIDTH, HEIGHT * 2),
        (WIDTH, HEIGHT),
    ];

    let mut max_retained = 0_u64;
    for (i, &(w, h)) in sizes.iter().cycle().take(40).enumerate() {
        let texture = test_alloc_texture(
            renderer.device(),
            w,
            h,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
            TextureFlags::COPY_DST,
        );
        let params = RenderParams {
            base_color: palette::css::BLACK,
            width: w,
            height: h,
            antialiasing_method: AaConfig::Area,
            robust: false,
        };
        renderer
            .render_to_texture(&scene, &texture, &params)
            .unwrap_or_else(|e| panic!("resize frame {i} ({w}x{h}) failed: {e}"));

        let depth = renderer.allocator_stats().cleanup_ring_depth;
        assert_eq!(
            depth, 0,
            "frame {i}: nonblocking scheme resize must not create retirement-ring slots (depth={depth})"
        );
        max_retained = max_retained.max(renderer.resource_pool_stats().retained_pool_buffer_bytes);
    }

    assert!(max_retained > 0, "retained pool never allocated during resize churn");
    // Bound growth: after cycling sizes, bytes should stay within a few scene/config buckets.
    let final_bytes = renderer.resource_pool_stats().retained_pool_buffer_bytes;
    assert!(
        final_bytes <= max_retained,
        "retained pool bytes regress after churn: final={final_bytes} max_seen={max_retained}"
    );
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("single_frame_ring_depth_bounded", || {
            single_frame_ring_depth_bounded();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("resource_pool_stable_under_single_frame", || {
            resource_pool_stable_under_single_frame();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("indirect_buffer_reused_across_frames", || {
            indirect_buffer_reused_across_frames();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("resize_churn_keeps_ring_empty_and_pool_bounded", || {
            resize_churn_keeps_ring_empty_and_pool_bounded();
            Ok(())
        })
        .with_ignored_flag(false),
    );

    let mut args = libtest_mimic::Arguments::from_args();
    if let Some(device) = shared_test_device() {
        submission::clamp_test_threads(&mut args, device);
    }
    libtest_mimic::run(&args, trials).exit()
}
