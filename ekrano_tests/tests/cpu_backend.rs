// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy CPU compute backend (`GOLDY_BACKEND=cpu`, issue #114).
//!
//! Isolated binary so the env override cannot race GPU tests. Area fine writes
//! a packed RGBA8 pixmap; [`ekrano::GoldyRenderer::render_to_buffer`] withdraws
//! it through [`goldy::PixelExchange`].

use std::mem::size_of;
use std::sync::Arc;

use ekrano::kurbo::{Affine, Rect};
use ekrano::peniko::color::palette;
use ekrano::peniko::{Brush, Fill};
use ekrano::{GoldyRenderer, RenderParams, Scene};
use ekrano_encoding::{ConfigUniform, PathBbox};
use goldy::{
    BufferKind, ComputePipeline, DeviceDescriptor, HostPixelSink, Instance, MemoryExchange, NodeAccess,
    RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, TextureFormat, types::BufferFlags,
};

fn select_cpu_backend() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
}

fn cpu_device() -> goldy::Device {
    let instance = Instance::new().expect("CPU instance");
    assert_eq!(
        instance.backend_type(),
        goldy::BackendType::Cpu,
        "GOLDY_BACKEND=cpu must select the host-callable device"
    );
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("CPU adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("CPU device")
}

fn area_params(width: u32, height: u32) -> RenderParams {
    RenderParams {
        base_color: palette::css::BLACK,
        width,
        height,
        antialiasing_method: ekrano::AaConfig::Area,
        robust: false,
    }
}

/// Dispatch Ekrano `bbox_clear` on host parcels (Goldy #292 acceptance: one real stage).
#[test]
fn bbox_clear_runs_on_cpu() {
    select_cpu_backend();
    let device = cpu_device();
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));

    let mut config = ConfigUniform::default();
    config.layout.n_paths = 2;
    let config_bytes = bytemuck::bytes_of(&config);
    let config_buf = pool
        .acquire_buffer(
            config_bytes.len() as u64,
            BufferKind::Scattered,
            Some(u32::try_from(size_of::<ConfigUniform>()).expect("ConfigUniform stride fits u32")),
            BufferFlags::empty(),
            Some(config_bytes),
        )
        .expect("config");

    let bboxes = [PathBbox::default(), PathBbox::default()];
    let bbox_bytes = bytemuck::bytes_of(&bboxes);
    let bbox_buf = pool
        .acquire_buffer(
            bbox_bytes.len() as u64,
            BufferKind::Scattered,
            Some(u32::try_from(size_of::<PathBbox>()).expect("PathBbox stride fits u32")),
            BufferFlags::empty(),
            Some(bbox_bytes),
        )
        .expect("path_bboxes");

    let search = ekrano_shaders::slang::slang_search_path();
    let search_str = search.to_string_lossy();
    let shader =
        ShaderModule::from_slang_with_paths(&device, ekrano_shaders::slang::BBOX_CLEAR, &[search_str.as_ref()])
            .expect("compile bbox_clear");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("bbox_clear", &pipeline)
        .with_parcel(&config_buf, NodeAccess::Read)
        .with_parcel(&bbox_buf, NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, &bbox_buf)
        .expect("withdraw");
    let mut frame = scheme.submit().expect("submit");
    let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
    let out: &[PathBbox] = bytemuck::cast_slice(&bytes);
    assert_eq!(out.len(), 2, "withdraw should return both path bboxes");
    for (i, bbox) in out.iter().enumerate() {
        assert_eq!(bbox.x0, i32::MAX, "x0[{i}]");
        assert_eq!(bbox.y0, i32::MAX, "y0[{i}]");
        assert_eq!(bbox.x1, i32::MIN, "x1[{i}]");
        assert_eq!(bbox.y1, i32::MIN, "y1[{i}]");
    }
}

/// Area fine writes packed RGBA8; PixelExchange copies it to a host sink.
#[test]
fn render_to_buffer_solid_rect() {
    select_cpu_backend();
    let device = cpu_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new on CPU");
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Rect::new(4.0, 4.0, 12.0, 12.0),
    );
    let params = area_params(16, 16);
    let bytes = renderer
        .render_to_buffer(&scene, &params)
        .expect("CPU render_to_buffer");
    assert_eq!(bytes.len(), 16 * 16 * 4);
    assert_eq!(&bytes[0..4], &[0, 0, 0, 255], "base color at (0,0)");
    let interior = ((8 * 16) + 8) * 4;
    assert_eq!(&bytes[interior..interior + 4], &[255, 0, 0, 255], "filled red at (8,8)");
}

/// Same pixmap path via an explicit [`HostPixelSink`].
#[test]
fn render_to_pixel_sink_solid_rect() {
    select_cpu_backend();
    let device = cpu_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new on CPU");
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::LIME),
        None,
        &Rect::new(2.0, 2.0, 6.0, 6.0),
    );
    let params = area_params(8, 8);
    let sink = Arc::new(HostPixelSink::new(8, 8, TextureFormat::Rgba8Unorm).expect("sink"));
    renderer
        .render_to_pixel_sink(&scene, &params, sink.clone())
        .expect("CPU render_to_pixel_sink");
    let bytes = sink.pixels();
    assert_eq!(bytes.len(), 8 * 8 * 4);
    assert_eq!(&bytes[0..4], &[0, 0, 0, 255], "base color at (0,0)");
    let interior = ((4 * 8) + 4) * 4;
    assert_eq!(
        &bytes[interior..interior + 4],
        &[0, 255, 0, 255],
        "filled lime at (4,4)"
    );
}

#[test]
fn cpu_fine_rejects_msaa() {
    select_cpu_backend();
    let device = cpu_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new on CPU");
    let err = renderer
        .render_to_buffer(
            &Scene::new(),
            &RenderParams {
                base_color: palette::css::BLACK,
                width: 8,
                height: 8,
                antialiasing_method: ekrano::AaConfig::Msaa8,
                robust: false,
            },
        )
        .expect_err("CPU fine is Area-only");
    let detail = err.detail();
    assert!(detail.contains("Area"), "expected Area-only error, got {detail}");
}
