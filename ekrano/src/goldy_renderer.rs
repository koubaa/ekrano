// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy-based renderer for offscreen and texture rendering.
//!
//! Use this when building with `--no-default-features --features goldy`.

use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};
use goldy::{BufferPool, Device, Texture};

use crate::{
    Error, RenderParams, Result, Scene,
    goldy_engine::GoldyEngine,
    render::Render,
    shaders::{self, FullShaders},
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
    /// Uses robust rendering with deferred bump validation: coarse and fine
    /// passes execute back-to-back on the GPU in a single submission (no CPU
    /// readback stall between them). The bump allocator is checked *after*
    /// both passes complete; if any stage overflowed, the frame is re-rendered
    /// with larger buffers.
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
            let config = retry_config.take().unwrap_or_else(|| {
                let mut packed = vec![];
                let (layout, _, _) = self.resolver.resolve(encoding, &mut packed);
                ekrano_encoding::RenderConfig::new(
                    &layout,
                    params.width,
                    params.height,
                    &params.base_color,
                )
            });
            let base = BufferPool::padded_size(&config.buffer_sizes.pool_allocs());
            // Safety margin: runtime allocation order may differ from pool_allocs order,
            // causing extra alignment padding. Add 256K to absorb ordering variance.
            let pool_size = base.saturating_add(262144);

            self.engine.prepare_storage_pool(device, pool_size)?;

            let mut render = Render::new();
            let mut recording = render.render_encoding_coarse_with_config(
                encoding,
                &mut self.resolver,
                &self.shaders,
                params,
                true,
                &config,
            );
            let bump_buf = render.bump_buf();
            let out_image = render.out_image();

            render.record_fine(&self.shaders, &mut recording);

            #[cfg(feature = "debug_layers")]
            if let Some(captured) = render.take_captured_buffers() {
                captured.release_buffers(&mut recording);
            }

            self.engine.run_recording(
                device,
                &recording,
                Some((&out_image, texture)),
                "coarse+fine",
            )?;

            let bump = self.read_bump(&bump_buf)?;
            self.engine.free_download(bump_buf);

            log::debug!(
                "[BUMP] lines={}, seg_counts={}, segments={}, tile={}, failed=0x{:x}",
                bump.lines,
                bump.seg_counts,
                bump.segments,
                bump.tile,
                bump.failed
            );
            if bump.failed == 0 || attempt == MAX_BUMP_RETRIES {
                if bump.failed != 0 {
                    log::warn!(
                        "Bump allocator overflow after {} retries (failed stages: 0x{:x}). \
                         Rendering may be incomplete.",
                        MAX_BUMP_RETRIES,
                        bump.failed,
                    );
                }
                return Ok(());
            }

            retry_config = Some(config.with_bump_estimates(&bump));
            log::info!(
                "Bump overflow on attempt {} (failed: 0x{:x}), retrying with larger buffers",
                attempt + 1,
                bump.failed,
            );
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

        // Free pool and transient buffer memory before readback so the staging
        // buffer allocation doesn't fail on memory-constrained workloads.
        self.engine.release_pool();

        let mut output = vec![0_u8; texture.byte_size()];
        texture
            .read_to_cpu(&mut output)
            .map_err(|e| Error::Shader(e.to_string()))?;
        Ok(output)
    }

    fn read_bump(&self, bump_buf: &crate::low_level::BufferProxy) -> Result<BumpAllocators> {
        let data = self
            .engine
            .get_download(*bump_buf)
            .ok_or_else(|| Error::Shader("bump buffer download not available".into()))?;
        Ok(bytemuck::pod_read_unaligned::<BumpAllocators>(data))
    }
}
