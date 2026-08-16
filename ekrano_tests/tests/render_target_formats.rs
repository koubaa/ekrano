// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Offscreen render-target format coverage (no window / swapchain).
//!
//! Exercises the same binding mix CUDA swapchain present hits: fine writes a
//! `Rgba32Float` destination while filter layers stay `Rgba8Unorm`.

#[path = "common/submission.rs"]
mod submission;

use ekrano::kurbo::{Affine, Circle, Rect};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use ekrano_encoding::{Filter, FilterEdgeMode, FilterPrimitive};
use ekrano_tests::{SharedTestDevice, shared_test_device, test_alloc_texture, test_device};
use goldy::types::{TextureFlags, TextureFormat, TextureKind};
use goldy::{MemoryExchange, Scheme};

/// Serialize GPU tests when the D3D12 debug layer is active.
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

fn make_device() -> SharedTestDevice {
    test_device()
}

fn backend_supports_rgba32float(device: &goldy::Device) -> bool {
    device
        .capabilities()
        .supported_render_target_formats
        .contains(&TextureFormat::Rgba32Float)
}

fn blurred_circle_scene(width: f64, height: f64) -> Scene {
    let mut scene = Scene::new();
    let blur = Filter(FilterPrimitive::GaussianBlur {
        std_dev: 2.0,
        edge_mode: FilterEdgeMode::Duplicate,
    });
    scene.push_filter_layer(
        blur,
        Fill::NonZero,
        Affine::IDENTITY,
        &Rect::new(0.0, 0.0, width, height),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::ORANGE,
        None,
        &Circle::new((width * 0.5, height * 0.5), width.min(height) * 0.25),
    );
    scene.pop_layer();
    scene
}

/// `render_to_texture` into `Rgba32Float` with active filter layers.
///
/// Without a display this still binds fine as float `output` + rgba8
/// `filter_tex*`, matching CUDA swapchain scratch + filter snapshots.
fn render_to_rgba32float_with_filters() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    if !backend_supports_rgba32float(&device) {
        eprintln!(
            "skipping render_to_rgba32float_with_filters: {:?} has no Rgba32Float render targets",
            device.backend_type()
        );
        return;
    }

    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer");
    let width = 64_u32;
    let height = 64_u32;
    let texture = test_alloc_texture(
        renderer.device(),
        width,
        height,
        TextureFormat::Rgba32Float,
        TextureKind::Direct,
        TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
    );
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width,
        height,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    let scene = blurred_circle_scene(width as f64, height as f64);

    renderer
        .render_to_texture(&scene, &texture, &params)
        .unwrap_or_else(|e| panic!("float+filter render failed (CUDA mix regression?): {e:#}"));

    // Second frame: retention / specialization cache must survive the mix.
    renderer
        .render_to_texture(&scene, &texture, &params)
        .unwrap_or_else(|e| panic!("float+filter second frame failed: {e:#}"));

    let ctx = renderer.submission_context();
    let mut scheme = Scheme::new(&ctx);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &texture)
        .expect("withdraw float RT");
    let mut frame = scheme.submit().expect("submit readback");
    let loan = grant.claim(&mut frame).expect("claim").consume().expect("read");
    assert_eq!(
        loan.len(),
        (width * height * 4 * 4) as usize,
        "Rgba32Float readback byte count"
    );

    // Blurred orange circle over black: some texels must be non-zero.
    let nonzero = loan.chunks_exact(4).any(|b| {
        let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        v > 1e-3
    });
    assert!(nonzero, "Rgba32Float filter render produced an all-zero image");
}

/// Plain (no filter) float target still works — identity specialization path.
fn render_to_rgba32float_plain() {
    env_logger::try_init().ok();
    let _gpu_guard = gpu_test_lock();

    let device = make_device();
    if !backend_supports_rgba32float(&device) {
        eprintln!(
            "skipping render_to_rgba32float_plain: {:?} has no Rgba32Float render targets",
            device.backend_type()
        );
        return;
    }

    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer");
    let width = 32_u32;
    let height = 32_u32;
    let texture = test_alloc_texture(
        renderer.device(),
        width,
        height,
        TextureFormat::Rgba32Float,
        TextureKind::Direct,
        TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
    );
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width,
        height,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::RED,
        None,
        &Rect::new(0.0, 0.0, 16.0, 16.0),
    );
    renderer
        .render_to_texture(&scene, &texture, &params)
        .unwrap_or_else(|e| panic!("plain float render failed: {e:#}"));
}

fn main() {
    let mut trials = Vec::new();
    trials.push(
        libtest_mimic::Trial::test("render_to_rgba32float_with_filters", || {
            render_to_rgba32float_with_filters();
            Ok(())
        })
        .with_ignored_flag(false),
    );
    trials.push(
        libtest_mimic::Trial::test("render_to_rgba32float_plain", || {
            render_to_rgba32float_plain();
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
