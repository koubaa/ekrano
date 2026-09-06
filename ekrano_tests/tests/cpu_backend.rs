// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy CPU compute backend (`GOLDY_BACKEND=cpu`, issue #114).
//!
//! Isolated binary so the env override cannot race GPU tests. Area fine
//! (`fine_cpu.slang`) writes a packed RGBA8 pixmap; `PixelExchange` withdraws it.
//! Coarse stages that use `GroupMemoryBarrierWithGroupSync` do not compile on
//! Slang's host-callable target, so a full `GoldyRenderer` frame errors until
//! Goldy implements workgroup barriers on CPU.

use std::mem::size_of;
use std::sync::Arc;

use ekrano::peniko::color::palette;
use ekrano::{GoldyRenderer, RenderParams, Scene};
use ekrano_encoding::{ConfigUniform, PathBbox};
use goldy::{
    BufferKind, ComputePipeline, DeviceDescriptor, HostPixelSink, Instance, MemoryExchange, NodeAccess, PixelExchange,
    PixmapLayout, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, TextureFormat, types::BufferFlags,
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

/// `fine_cpu.slang` writes packed RGBA8; `PixelExchange` copies it into a host sink.
#[test]
fn fine_cpu_writes_packed_pixmap() {
    select_cpu_backend();
    let device = cpu_device();
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));

    let config = ConfigUniform {
        width_in_tiles: 1,
        height_in_tiles: 1,
        target_width: 16,
        target_height: 16,
        base_color: palette::css::RED.premultiply().to_rgba8().to_u32(),
        ..ConfigUniform::default()
    };
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

    let zeros4 = [0_u32; 4];
    let segments = pool
        .acquire_buffer(
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::empty(),
            Some(bytemuck::bytes_of(&zeros4)),
        )
        .expect("segments");
    let ptcl_words = [0_u32; 64];
    let ptcl = pool
        .acquire_buffer(
            (ptcl_words.len() * 4) as u64,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::empty(),
            Some(bytemuck::bytes_of(&ptcl_words)),
        )
        .expect("ptcl");
    let info = pool
        .acquire_buffer(
            4,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::empty(),
            Some(bytemuck::bytes_of(&0_u32)),
        )
        .expect("info");
    let blend_spill = pool
        .acquire_buffer(
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::empty(),
            Some(bytemuck::bytes_of(&[0_u32; 4])),
        )
        .expect("blend_spill");
    let pixmap_words = [0_u32; 16 * 16];
    let pixmap = pool
        .acquire_buffer(
            (pixmap_words.len() * 4) as u64,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::empty(),
            Some(bytemuck::bytes_of(&pixmap_words)),
        )
        .expect("pixmap");

    let search = ekrano_shaders::slang::slang_search_path();
    let search_str = search.to_string_lossy();
    let shader = ShaderModule::from_slang_with_paths(&device, ekrano_shaders::slang::FINE_CPU, &[search_str.as_ref()])
        .expect("compile fine_cpu");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fine_cpu", &pipeline)
        .with_parcel(&config_buf, NodeAccess::Read)
        .with_parcel(&segments, NodeAccess::Read)
        .with_parcel(&ptcl, NodeAccess::Read)
        .with_parcel(&info, NodeAccess::Read)
        .with_parcel(&blend_spill, NodeAccess::ReadWrite)
        .with_parcel(&pixmap, NodeAccess::ReadWrite)
        .dispatch(1, 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("wait");

    let sink = Arc::new(HostPixelSink::new(16, 16, TextureFormat::Rgba8Unorm).expect("sink"));
    let exchange = PixelExchange::new(&ctx, sink.clone());
    let mut readback = Scheme::new(&ctx);
    let layout = PixmapLayout::tight(16, 16, TextureFormat::Rgba8Unorm);
    let tx = exchange
        .bind_source(&mut readback, pixmap.whole(), layout)
        .expect("bind_source");
    let mut submission = readback.submit().expect("submit");
    tx.claim(&mut submission).expect("claim").consume().expect("consume");
    let bytes = sink.pixels();
    assert_eq!(bytes.len(), 16 * 16 * 4);
    assert_eq!(&bytes[0..4], &[255, 0, 0, 255], "base color at (0,0)");
    let interior = ((8 * 16) + 8) * 4;
    assert_eq!(&bytes[interior..interior + 4], &[255, 0, 0, 255], "base color at (8,8)");
}

#[test]
fn renderer_constructs_and_full_frame_needs_barriers() {
    select_cpu_backend();
    let device = cpu_device();
    let mut renderer = GoldyRenderer::new(&device).expect("GoldyRenderer::new on CPU");
    let err = renderer
        .render_to_buffer(&Scene::new(), &area_params(16, 16))
        .expect_err("full CPU frame needs workgroup barriers");
    let detail = err.detail();
    assert!(
        detail.contains("GroupMemoryBarrierWithGroupSync") || detail.contains("workgroup-barrier"),
        "expected barrier-stage error, got {detail}"
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
