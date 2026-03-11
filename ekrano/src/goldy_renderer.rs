// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy-based renderer for offscreen and texture rendering.
//!
//! Use this when building with `--no-default-features --features goldy`.

use goldy::{Device, Texture};
use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};

use crate::{
    goldy_engine::GoldyEngine,
    recording::Recording,
    render::Render,
    shaders::{self, FullShaders},
    Error, RenderParams, Result, Scene,
};
use ekrano_encoding::{BumpAllocators, Resolver};

const MAX_BUMP_RETRIES: usize = 2;

/// Goldy-based 2D renderer.
///
/// Renders scenes to textures using the Goldy GPU backend with Slang shaders.
pub struct GoldyRenderer {
    engine: GoldyEngine,
    shaders: FullShaders,
    resolver: Resolver,
}

impl GoldyRenderer {
    /// Create a new renderer for the given device.
    pub fn new(device: &Device) -> Result<Self> {
        let mut engine = GoldyEngine::new();
        let shaders = shaders::goldy_full_shaders(device, &mut engine)?;
        Ok(Self {
            engine,
            shaders,
            resolver: Resolver::new(),
        })
    }

    /// Render a scene to the given texture.
    ///
    /// Uses robust rendering: runs the coarse pass, reads back the bump allocator,
    /// and retries with larger buffers if any stage overflowed.
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        scene: &Scene,
        texture: &Texture,
        params: &RenderParams,
    ) -> Result<()> {
        let encoding = scene.encoding();
        let mut retry_config: Option<ekrano_encoding::RenderConfig> = None;

        for attempt in 0..=MAX_BUMP_RETRIES {
            let mut render = Render::new();
            let coarse_recording = if let Some(ref cfg) = retry_config {
                render.render_encoding_coarse_with_config(
                    encoding,
                    &mut self.resolver,
                    &self.shaders,
                    params,
                    true,
                    cfg,
                )
            } else {
                render.render_encoding_coarse(
                    encoding,
                    &mut self.resolver,
                    &self.shaders,
                    params,
                    true,
                )
            };
            let bump_buf = render.bump_buf();

            self.engine
                .run_recording(device, &coarse_recording, None, "coarse")?;

            let bump = self.read_bump(&bump_buf)?;
            self.engine.free_download(bump_buf);

            if bump.failed == 0 || attempt == MAX_BUMP_RETRIES {
                if bump.failed != 0 {
                    log::warn!(
                        "Bump allocator overflow after {} retries (failed stages: 0x{:x}). \
                         Rendering may be incomplete.",
                        MAX_BUMP_RETRIES,
                        bump.failed,
                    );
                }
                let out_image = render.out_image();
                let mut fine_recording = Recording::default();
                render.record_fine(&self.shaders, &mut fine_recording);
                return self.engine.run_recording(
                    device,
                    &fine_recording,
                    Some((&out_image, texture)),
                    "fine",
                );
            }

            // Build a new config with grown buffer sizes for retry.
            // On first retry, base off the default config that render_encoding_coarse
            // would have created internally.
            let base = retry_config.take().unwrap_or_else(|| {
                let mut packed = vec![];
                let (layout, _, _) = self.resolver.resolve(encoding, &mut packed);
                ekrano_encoding::RenderConfig::new(
                    &layout,
                    params.width,
                    params.height,
                    &params.base_color,
                )
            });
            let new_config = base.with_bump_estimates(&bump);
            log::info!(
                "Bump overflow on attempt {} (failed: 0x{:x}), retrying with larger buffers",
                attempt + 1,
                bump.failed,
            );
            retry_config = Some(new_config);
            self.engine.clear_transients();
        }
        unreachable!()
    }

    /// Render a scene and return the pixel data as RGBA bytes.
    pub fn render_to_buffer(
        &mut self,
        device: &Device,
        scene: &Scene,
        params: &RenderParams,
    ) -> Result<Vec<u8>> {
        let width = params.width;
        let height = params.height;
        let texture = Texture::new(
            device,
            width,
            height,
            TextureFormat::Rgba8Unorm,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )
        .map_err(|e| Error::Shader(e.to_string()))?;

        self.render_to_texture(device, scene, &texture, params)?;

        let mut output = vec![0u8; texture.byte_size()];
        texture
            .read_to_cpu(&mut output)
            .map_err(|e| Error::Shader(e.to_string()))?;
        Ok(output)
    }

    fn read_bump(&self, bump_buf: &crate::low_level::BufferProxy) -> Result<BumpAllocators> {
        let data = self.engine.get_download(*bump_buf).ok_or_else(|| {
            Error::Shader("bump buffer download not available".into())
        })?;
        Ok(bytemuck::pod_read_unaligned::<BumpAllocators>(data))
    }
}
