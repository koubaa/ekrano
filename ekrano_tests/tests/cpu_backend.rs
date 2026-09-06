// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy CPU compute backend (`GOLDY_BACKEND=cpu`, issue #114).
//!
//! Isolated binary so the env override cannot race GPU tests. Fine raster
//! needs textures and stays GPU-only; this covers buffer-stage JIT.

use std::mem::size_of;
use std::sync::Arc;

use ekrano::{GoldyRenderer, RenderParams, Scene};
use ekrano_encoding::{ConfigUniform, PathBbox};
use goldy::{
    BufferKind, ComputePipeline, DeviceDescriptor, Instance, MemoryExchange, NodeAccess, RequestAdapterOptions,
    RetainedPool, Scheme, ShaderModule, types::BufferFlags,
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

/// Buffer-only stages compile; full `render_to_buffer` still needs textures.
#[test]
fn renderer_constructs_and_rejects_texture_render() {
    select_cpu_backend();
    let device = cpu_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new on CPU");
    let err = renderer
        .render_to_buffer(
            &Scene::new(),
            &RenderParams {
                base_color: ekrano::peniko::color::palette::css::BLACK,
                width: 16,
                height: 16,
                antialiasing_method: ekrano::AaConfig::Area,
                robust: false,
            },
        )
        .expect_err("CPU backend cannot rasterize");
    let detail = err.detail();
    assert!(
        detail.contains("compute-only"),
        "expected compute-only error, got {detail}"
    );
}
