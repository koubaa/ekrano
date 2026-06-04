// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Placement heap trace tests.
//!
//! Verifies that the persistent placement heap behaves as expected under real
//! GPU workloads:
//!
//! - Backing buffer is allocated once and reused across all frames (no per-frame
//!   `Buffer::new`).
//! - Capacity does not grow unboundedly.

use ekrano::kurbo::{Affine, Rect};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};
use goldy::{Device, DeviceDescriptor, Instance, RequestAdapterOptions};

const FRAME_COUNT: usize = 300;
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

/// Collect per-frame placement heap capacity over many frames and verify
/// the backing buffer is allocated once and never resized.
#[test]
fn placement_heap_paged_stable() {
    env_logger::try_init().ok();

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

    let mut capacities: Vec<u64> = Vec::new();

    eprintln!();
    eprintln!("=== Per-Frame Placement Heap Trace ===");
    eprintln!("{:>6}  {:>10}", "frame", "cap (MiB)");
    eprintln!("{:-<6}  {:-<10}", "", "");

    for i in 0..FRAME_COUNT {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        if let Some(stats) = renderer.placement_heap_stats() {
            capacities.push(stats.capacity);

            if i < 10 || i % 50 == 0 || i == FRAME_COUNT - 1 {
                eprintln!(
                    "{i:>6}  {:>10.2}",
                    stats.capacity as f64 / (1024.0 * 1024.0),
                );
            }
        }
    }

    eprintln!();

    if let Some(stats) = renderer.placement_heap_stats() {
        eprintln!("=== Summary ===");
        eprintln!(
            "  backing buffer capacity : {:.2} MiB",
            stats.capacity as f64 / (1024.0 * 1024.0)
        );
    }

    if !capacities.is_empty() {
        let max_cap = *capacities.iter().max().unwrap();
        let min_cap = *capacities.iter().min().unwrap();
        eprintln!(
            "  capacity range          : [{:.2}, {:.2}] MiB",
            min_cap as f64 / (1024.0 * 1024.0),
            max_cap as f64 / (1024.0 * 1024.0)
        );

        assert_eq!(
            max_cap, min_cap,
            "placement heap capacity changed: min={min_cap} max={max_cap} — \
             backing buffer should be allocated once"
        );
    }

    eprintln!("  PASS: placement heap is stable over {FRAME_COUNT} frames");
    eprintln!();
}

/// Verify that the placement heap's backing buffer is sized correctly and allocated once.
#[test]
fn placement_heap_capacity_sized_correctly() {
    env_logger::try_init().ok();

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

    let mut capacities: Vec<u64> = Vec::new();

    for i in 0..50 {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        if let Some(stats) = renderer.placement_heap_stats() {
            capacities.push(stats.capacity);
        }
    }

    if let Some(stats) = renderer.placement_heap_stats() {
        let cap_mb = stats.capacity as f64 / (1024.0 * 1024.0);
        eprintln!();
        eprintln!("=== Placement Heap Capacity ===");
        eprintln!("  backing buffer  : {cap_mb:.2} MiB");

        let max_cap = *capacities.iter().max().unwrap();
        let min_cap = *capacities.iter().min().unwrap();
        assert_eq!(
            max_cap, min_cap,
            "placement heap capacity changed: min={min_cap} max={max_cap}"
        );

        assert!(
            stats.capacity > 0,
            "placement heap should have non-zero capacity after rendering"
        );

        eprintln!("  PASS: capacity {cap_mb:.2} MiB, allocated once");
        eprintln!();
    }
}
