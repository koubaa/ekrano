// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello tests.

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    missing_debug_implementations,
    unreachable_pub,
    missing_docs,
    clippy::missing_assert_message,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::allow_attributes_without_reason
)]

use std::env;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::{Arc, Once};

/// D3D12 Agility SDK opt-in: the D3D12 loader calls GetProcAddress on the
/// running .exe to find these two symbols. CI deploys D3D12Core.dll into
/// `target\debug\deps\D3D12\` so this path resolves correctly for nextest.
#[cfg(target_os = "windows")]
mod _d3d12_agility_sdk {
    #[no_mangle]
    #[used]
    pub static D3D12SDKVersion: u32 = 614;

    #[no_mangle]
    #[used]
    pub static D3D12SDKPath: *const std::ffi::c_char =
        b".\\D3D12\\\0".as_ptr().cast::<std::ffi::c_char>();
}

use log as _;

use anyhow::{Result, anyhow, bail};
use ekrano::kurbo::{Affine, Vec2};
use ekrano::peniko::{Blob, Color, ImageFormat, color::palette};
use ekrano::peniko::{ImageAlphaType, ImageData};
use ekrano::{AaConfig, Scene};
use scenes::{ExampleScene, ImageCache, SceneParams, SimpleText};

mod snapshot;

pub use snapshot::{
    Snapshot, SnapshotDirectory, smoke_snapshot_test_sync, snapshot_test, snapshot_test_sync,
};

pub struct TestParams {
    pub width: u32,
    pub height: u32,
    pub base_color: Option<Color>,
    pub name: String,
    pub anti_aliasing: AaConfig,
}

impl TestParams {
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            base_color: None,
            name: name.into(),
            anti_aliasing: AaConfig::Area,
        }
    }
}

pub fn render_then_debug_sync(scene: &Scene, params: &TestParams) -> Result<ImageData> {
    render_then_debug(scene, params)
}

pub fn render_then_debug(scene: &Scene, params: &TestParams) -> Result<ImageData> {
    let image = get_scene_image(params, scene)?;
    let name = params.name.clone();
    let out_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("debug_outputs")
        .join(name)
        .with_extension("png");
    if env_var_relates_to("EKRANO_DEBUG_TEST", &params.name) {
        write_png_to_file(params, &out_path, &image, None, false)?;
        let (width, height) = (image.width, image.height);
        println!("Wrote debug result ({width}x{height}) to {out_path:?}");
    } else {
        match std::fs::remove_file(&out_path) {
            Ok(()) => (),
            Err(e) if e.kind() == ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(image)
}

pub fn get_scene_image(params: &TestParams, scene: &Scene) -> Result<ImageData, anyhow::Error> {
    static INIT_LOGGER: Once = Once::new();
    INIT_LOGGER.call_once(|| {
        env_logger::init();
    });

    use ekrano::{GoldyRenderer, RenderParams};
    use goldy::{DeviceType, Instance};

    let instance = Instance::new()?;

    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .or_else(|_| instance.create_device(DeviceType::Other))
        .map_err(|e| anyhow!("No Goldy device: {e}"))?;

    let mut renderer = GoldyRenderer::new(&device)?;
    let width = params.width;
    let height = params.height;
    let render_params = RenderParams {
        base_color: params.base_color.unwrap_or(palette::css::BLACK),
        width,
        height,
        antialiasing_method: params.anti_aliasing,
    };

    let pixels = renderer.render_to_buffer(&device, scene, &render_params)?;
    let data = Blob::new(Arc::new(pixels));
    Ok(ImageData {
        data,
        format: ImageFormat::Rgba8,
        width,
        height,
        alpha_type: ImageAlphaType::Alpha,
    })
}

pub fn write_png_to_file(
    params: &TestParams,
    out_path: &Path,
    image: &ImageData,
    max_size_in_bytes: Option<u64>,
    optimise: bool,
) -> Result<(), anyhow::Error> {
    if image.format != ImageFormat::Rgba8 {
        unimplemented!();
    }
    if image.alpha_type != ImageAlphaType::Alpha {
        unimplemented!()
    }
    let width = params.width;
    let height = params.height;
    let mut data = Vec::new();
    let mut encoder = png::Encoder::new(&mut data, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.data.data())?;
    writer.finish()?;
    if optimise {
        data = oxipng::optimize_from_memory(&data, &oxipng::Options::max_compression()).unwrap();
    }

    let size = data.len();
    std::fs::write(out_path, &data)?;
    let oversized_path = out_path.with_extension("oversized.png");
    if max_size_in_bytes
        .is_some_and(|max_size_in_bytes| u64::try_from(size).unwrap() > max_size_in_bytes)
    {
        std::fs::rename(out_path, &oversized_path)?;
        bail!(
            "File was oversized, expected {} bytes, got {size} bytes. New file written to {to}",
            max_size_in_bytes.unwrap(),
            to = oversized_path.display()
        );
    } else {
        // Intentionally do not handle errors here
        drop(std::fs::remove_file(oversized_path));
    }
    Ok(())
}

/// Determine whether the value of the environment variable `env_var`
/// includes a specific test.
/// This is used when updating tests, or dumping the debug output
fn env_var_relates_to(env_var: &'static str, name: &str) -> bool {
    if let Ok(val) = env::var(env_var) {
        if val.eq_ignore_ascii_case("all") {
            return true;
        }
        for test in val.split(',') {
            let test_name = test.trim();
            if test_name.eq_ignore_ascii_case(name) {
                return true;
            }
        }
    }
    false
}

pub fn encode_test_scene(mut test_scene: ExampleScene, test_params: &mut TestParams) -> Scene {
    let mut inner_scene = Scene::new();
    let mut image_cache = ImageCache::new();
    let mut text = SimpleText::new();
    let mut scene_params = SceneParams {
        base_color: None,
        complexity: 100,
        time: 0.,
        images: &mut image_cache,
        interactive: false,
        resolution: None,
        text: &mut text,
    };
    test_scene
        .function
        .render(&mut inner_scene, &mut scene_params);
    if test_params.base_color.is_none() {
        test_params.base_color = scene_params.base_color;
    }
    if let Some(resolution) = scene_params.resolution {
        // Automatically scale the rendering to fill as much of the window as possible
        let factor = Vec2::new(test_params.width as f64, test_params.height as f64);
        let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
        let mut outer_scene = Scene::new();
        outer_scene.append(&inner_scene, Some(Affine::scale(scale_factor)));
        outer_scene
    } else {
        inner_scene
    }
}
