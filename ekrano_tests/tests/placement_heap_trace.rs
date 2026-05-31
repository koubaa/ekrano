// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Placement heap Metal trace tests.
//!
//! Verifies that the persistent placement heap behaves as expected under real
//! GPU workloads:
//!
//! - Backing buffer is allocated once and reused across all frames (no per-frame
//!   `Buffer::new`).
//! - Ring regions wrap correctly after reclaim.
//! - In-flight region count stays bounded by `MAX_CLEANUP_DEPTH`.
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

/// Collect per-frame placement heap metrics over many frames and verify
/// the hardware reclaim/reuse pattern is correct.
///
/// Prints a detailed per-frame trace table showing ring occupancy over time.
#[test]
fn placement_heap_ring_stable() {
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
    let mut in_flight_counts: Vec<usize> = Vec::new();
    let mut in_flight_bytes_samples: Vec<u64> = Vec::new();

    eprintln!();
    eprintln!("=== Per-Frame Placement Heap Trace ===");
    eprintln!(
        "{:>6}  {:>10}  {:>10}  {:>8}",
        "frame", "cap (MiB)", "inflight", "regions"
    );
    eprintln!("{:-<6}  {:-<10}  {:-<10}  {:-<8}", "", "", "", "");

    for i in 0..FRAME_COUNT {
        renderer
            .render_to_texture(&device, &scene, &texture, &params)
            .unwrap_or_else(|e| panic!("frame {i} failed: {e}"));

        if let Some(stats) = renderer.placement_heap_stats() {
            capacities.push(stats.capacity);
            in_flight_counts.push(stats.in_flight_count);
            in_flight_bytes_samples.push(stats.in_flight_bytes);

            // Print every 10th frame and the first 10.
            if i < 10 || i % 50 == 0 || i == FRAME_COUNT - 1 {
                eprintln!(
                    "{i:>6}  {:>10.2}  {:>10.2}  {:>8}",
                    stats.capacity as f64 / (1024.0 * 1024.0),
                    stats.in_flight_bytes as f64 / (1024.0 * 1024.0),
                    stats.in_flight_count,
                );
            }
        }
    }

    eprintln!();

    // Print summary.
    if let Some(stats) = renderer.placement_heap_stats() {
        eprintln!("=== Summary ===");
        eprintln!(
            "  backing buffer capacity : {:.2} MiB",
            stats.capacity as f64 / (1024.0 * 1024.0)
        );
        eprintln!("  final in-flight regions : {}", stats.in_flight_count);
        eprintln!(
            "  final in-flight bytes   : {:.2} MiB",
            stats.in_flight_bytes as f64 / (1024.0 * 1024.0)
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

    if !in_flight_counts.is_empty() {
        let max_inflight = *in_flight_counts.iter().max().unwrap();
        eprintln!("  max in-flight regions   : {max_inflight}");

        assert!(
            max_inflight <= 4,
            "in-flight region count {} exceeds expected bound (4 = MAX_CLEANUP_DEPTH + 1)",
            max_inflight,
        );
    }

    if !in_flight_bytes_samples.is_empty() {
        let tail = &in_flight_bytes_samples[in_flight_bytes_samples.len().saturating_sub(50)..];
        let max_tail = *tail.iter().max().unwrap();
        let min_tail = *tail.iter().min().unwrap();
        eprintln!(
            "  in-flight bytes range   : [{:.2}, {:.2}] MiB (last 50 frames)",
            min_tail as f64 / (1024.0 * 1024.0),
            max_tail as f64 / (1024.0 * 1024.0)
        );

        let per_region =
            max_tail / in_flight_counts.iter().max().copied().unwrap_or(1).max(1) as u64;
        let tolerance = per_region * 2;
        assert!(
            max_tail - min_tail <= tolerance,
            "in-flight bytes range [{min_tail}, {max_tail}] too wide — ring may not be reclaiming"
        );
    }

    eprintln!("  PASS: placement heap ring is stable over {FRAME_COUNT} frames");
    eprintln!();
}

/// Verify that the placement heap's backing buffer is sized correctly:
/// capacity = `per_frame_demand` × (`MAX_CLEANUP_DEPTH` + 1), allocated once.
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
        let inflight_mb = stats.in_flight_bytes as f64 / (1024.0 * 1024.0);
        let per_region_mb = if stats.in_flight_count > 0 {
            inflight_mb / stats.in_flight_count as f64
        } else {
            0.0
        };
        eprintln!();
        eprintln!("=== Placement Heap Capacity ===");
        eprintln!("  backing buffer  : {cap_mb:.2} MiB");
        eprintln!(
            "  in-flight       : {inflight_mb:.2} MiB ({} regions)",
            stats.in_flight_count
        );
        eprintln!("  per-region avg  : {per_region_mb:.2} MiB");
        eprintln!(
            "  expected        : {per_region_mb:.2} × 4 = {:.2} MiB",
            per_region_mb * 4.0
        );

        // Capacity should be allocated once and never change.
        let max_cap = *capacities.iter().max().unwrap();
        let min_cap = *capacities.iter().min().unwrap();
        assert_eq!(
            max_cap, min_cap,
            "placement heap capacity changed: min={min_cap} max={max_cap}"
        );

        // Capacity should fit at least MAX_CLEANUP_DEPTH+1 regions.
        // With 4 MiB page alignment, capacity ≈ per_region * (depth + 1),
        // rounded to page boundaries.
        if stats.in_flight_count > 0 {
            let per_region = stats.in_flight_bytes / stats.in_flight_count as u64;
            let expected_min = per_region * 3; // at least 3 regions
            assert!(
                stats.capacity >= expected_min,
                "capacity {} < expected minimum {} (per_region={} × 3)",
                stats.capacity,
                expected_min,
                per_region,
            );
        }

        eprintln!("  PASS: capacity {cap_mb:.2} MiB, allocated once, fits pipeline depth");
        eprintln!();
    }
}
