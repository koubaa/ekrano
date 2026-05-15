// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Multi-frame pipelined memory stress tests.
//!
//! These tests exercise the `HeapTransientAllocator` lifecycle across many
//! frames using the pipelined `render_to_texture` path (not the synchronous
//! `render_to_buffer` path used by snapshot tests). They verify that:
//!
//! - The allocator capacity stabilises (no unbounded growth).
//! - `used` bytes return to a per-frame baseline once the cleanup ring drains.
//! - The cleanup ring depth stays within `MAX_CLEANUP_DEPTH`.

use ekrano::kurbo::{Affine, Rect};
use ekrano::peniko::{Fill, color::palette};
use ekrano::{AaConfig, GoldyRenderer, RenderParams, Scene};
use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};
use goldy::{Device, DeviceType, Instance};

const FRAME_COUNT: usize = 100;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

fn make_device() -> Device {
    let instance = Instance::new().expect("Instance::new");
    instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .or_else(|_| instance.create_device(DeviceType::Other))
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

/// Render `FRAME_COUNT` frames through the pipelined `render_to_texture` path
/// and verify that allocator capacity stabilises (no unbounded growth).
#[test]
fn pipelined_allocator_capacity_stable() {
    env_logger::try_init().ok();

    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(WIDTH, HEIGHT, TextureFormat::Rgba8Unorm, SpatialAccess::Direct, TextureFlags::COPY_DST)
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

    for i in 0..FRAME_COUNT {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        if let Some(stats) = renderer.allocator_stats() {
            capacities.push(stats.capacity);
        }
    }

    // After pipeline saturation the capacity must stop growing. Check the last
    // 50 frames: if capacity is still increasing, the allocator is leaking.
    let tail = &capacities[capacities.len().saturating_sub(50)..];
    let max_cap = *tail.iter().max().unwrap();
    let min_cap = *tail.iter().min().unwrap();
    assert_eq!(
        max_cap, min_cap,
        "allocator capacity is still growing in the last 50 frames: \
         min={min_cap} max={max_cap} (delta={})",
        max_cap - min_cap,
    );

    let stats = renderer
        .allocator_stats()
        .expect("allocator should be initialised after rendering");
    assert!(
        stats.cleanup_ring_depth <= 3,
        "cleanup ring depth {} exceeds MAX_CLEANUP_DEPTH (3)",
        stats.cleanup_ring_depth,
    );
}

/// Verify that `used` bytes converge to a per-frame steady state, proving
/// that freed ranges are actually reclaimed across frames.
#[test]
fn pipelined_allocator_used_converges() {
    env_logger::try_init().ok();

    let device = make_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new");
    let texture = device
        .alloc_texture(WIDTH, HEIGHT, TextureFormat::Rgba8Unorm, SpatialAccess::Direct, TextureFlags::COPY_DST)
        .expect("alloc_texture");
    let scene = tiny_scene();
    let params = RenderParams {
        base_color: palette::css::BLACK,
        width: WIDTH,
        height: HEIGHT,
        antialiasing_method: AaConfig::Area,
        robust: false,
    };

    let mut used_samples: Vec<u64> = Vec::new();

    for i in 0..FRAME_COUNT {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        if let Some(stats) = renderer.allocator_stats() {
            used_samples.push(stats.used);
        }
    }

    // After warmup the last 50 samples should be bounded: not monotonically
    // increasing, i.e. freeing is working.
    let tail = &used_samples[used_samples.len().saturating_sub(50)..];
    let max_tail = *tail.iter().max().unwrap();
    let min_tail = *tail.iter().min().unwrap();

    // The range of used-bytes in steady state should be small (within one
    // frame's allocation volume). If used keeps growing linearly, max - min
    // would be ~50 × per_frame_alloc, which is huge.
    let first_frame_used = used_samples.first().copied().unwrap_or(0);
    let tolerance = first_frame_used.max(1) * 5;
    assert!(
        max_tail - min_tail <= tolerance,
        "used bytes did not converge: range [{min_tail}, {max_tail}] over last 50 frames \
         (first frame used = {first_frame_used}, tolerance = {tolerance})"
    );
}
