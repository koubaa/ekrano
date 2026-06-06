// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Single-frame frame-orchestrator stress tests.
//!
//! Ekrano uses a fixed depth=1 fire-and-forget model (ekrano issue #71). These
//! tests verify that many frames through `render_to_texture` keep the
//! [`FrameOrchestrator`] ring bounded and the resource pool stable.

use ekrano::kurbo::{Affine, Rect};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use goldy::types::{TextureFlags, TextureFormat, TextureKind};
use goldy::{Device, DeviceDescriptor, Instance, RequestAdapterOptions};

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

/// Render many frames and verify the frame-orchestrator ring stays at depth=1.
#[test]
fn single_frame_ring_depth_bounded() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
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

    for i in 0..FRAME_COUNT {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        let depth = renderer
            .allocator_stats()
            .expect("allocator stats available after first frame")
            .cleanup_ring_depth;
        assert!(
            depth <= 1,
            "frame {i}: cleanup ring depth {depth} exceeds single-frame limit (1)"
        );
    }
}

/// Verify the resource pool stabilises after warmup under the single-frame model.
#[test]
fn resource_pool_stable_under_single_frame() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(
            WIDTH,
            HEIGHT,
            TextureFormat::Rgba8Unorm,
            TextureKind::Direct,
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

    for i in 0..30 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("warmup frame {i} failed: {e}"));
    }

    let baseline = renderer.resource_pool_stats();
    let mut max_pooled = baseline.total_pooled_buffers;

    for i in 0..50 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("steady frame {i} failed: {e}"));
        max_pooled = max_pooled.max(renderer.resource_pool_stats().total_pooled_buffers);
    }

    let growth = max_pooled.saturating_sub(baseline.total_pooled_buffers);
    assert!(
        growth <= baseline.total_pooled_buffers.max(10),
        "resource pool grew excessively after warmup: baseline={} max_seen={max_pooled} growth={growth}",
        baseline.total_pooled_buffers,
    );
}
