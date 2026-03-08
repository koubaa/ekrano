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
    recording::ResourceProxy,
    shaders::{self, FullShaders},
    Error, RenderParams, Result, Scene,
};
use ekrano_encoding::Resolver;

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
    /// The texture must have the same dimensions as `params.width` x `params.height`
    /// and use Rgba8Unorm format with storage/COPY_DST capability.
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        scene: &Scene,
        texture: &Texture,
        params: &RenderParams,
    ) -> Result<()> {
        let (recording, target) = crate::render::render_full(
            scene,
            &mut self.resolver,
            &self.shaders,
            params,
        );
        let target_proxy = match &target {
            ResourceProxy::Image(p) => p,
            _ => return Err(Error::Shader("expected image target".into())),
        };
        self.engine.run_recording(
            device,
            &recording,
            Some((target_proxy, texture)),
            "render_to_texture",
        )
    }

    /// Render a scene and return the pixel data as RGBA bytes.
    ///
    /// Creates an internal texture, renders to it, then reads back via Texture::read_to_cpu.
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
}
