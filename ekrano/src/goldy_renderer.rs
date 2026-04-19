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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const MAX_BUMP_RETRIES: usize = 2;

/// Upper bound applied to observed bump counters before they're fed into
/// `RenderConfig::with_bump_estimates`. Legitimate scenes need far less than
/// this (the tiger hits ~13K segments, paris-30k a few hundred thousand),
/// but a stale/corrupt read of `vello.bump_buf` — which we've observed
/// intermittently — can make a counter look like a billion, at which point
/// `grow()` rounds it up to `next_power_of_two` and the pool allocation goes
/// to multiple GB, exhausts Metal's heap, and puts the app in an infinite
/// retry loop. 16M entries for any single counter covers even paris-30k
/// plus an order of magnitude of headroom; anything larger is treated as
/// garbage and clamped.
const BUMP_SANITY_CAP: u32 = 16 * 1024 * 1024;

/// Clamp any counter that looks implausibly large back to the default.
/// `with_bump_estimates` ignores fields that are below the current buffer
/// size, so substituting 0 for an absurd value is equivalent to "stay at
/// default" without risking an allocation blow-up.
fn sanitize_bump(bump: &BumpAllocators) -> BumpAllocators {
    let clamp = |v: u32| if v > BUMP_SANITY_CAP { 0 } else { v };
    BumpAllocators {
        failed: bump.failed,
        binning: clamp(bump.binning),
        ptcl: clamp(bump.ptcl),
        tile: clamp(bump.tile),
        seg_counts: clamp(bump.seg_counts),
        segments: clamp(bump.segments),
        blend: clamp(bump.blend),
        lines: clamp(bump.lines),
    }
}

/// Rolling frame counter, used purely to rate-limit per-frame logging so a
/// long run doesn't flood the terminal and truncate the scrollback. Shared
/// across all `GoldyRenderer` instances in the process; that's fine because
/// in practice there's only one.
static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);
/// Tracks whether the previous frame produced zero geometry so we can log a
/// single, loud line the moment the scene transitions between empty and
/// non-empty (the interesting event) rather than per-frame spam.
static LAST_FRAME_EMPTY: AtomicBool = AtomicBool::new(false);

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
        use std::time::Instant;
        let frame_start = Instant::now();

        let encoding = scene.encoding();
        let mut retry_config: Option<ekrano_encoding::RenderConfig> = None;

        for attempt in 0..=MAX_BUMP_RETRIES {
            let t0 = Instant::now();
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
            let t_resolve = t0.elapsed();

            let base = BufferPool::padded_size(&config.buffer_sizes.pool_allocs());
            let pool_size = base.saturating_add(262144);

            let t1 = Instant::now();
            self.engine.prepare_storage_pool(device, pool_size)?;
            let t_pool = t1.elapsed();

            let t2 = Instant::now();
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
            let t_coarse = t2.elapsed();

            let t3 = Instant::now();
            render.record_fine(&self.shaders, &mut recording);
            let t_fine_record = t3.elapsed();

            #[cfg(feature = "debug_layers")]
            if let Some(captured) = render.take_captured_buffers() {
                captured.release_buffers(&mut recording);
            }

            let t4 = Instant::now();
            self.engine.run_recording(
                device,
                &recording,
                Some((&out_image, texture)),
                "coarse+fine",
            )?;
            let t_gpu = t4.elapsed();

            let t5 = Instant::now();
            let bump = self.read_bump(&bump_buf)?;
            self.engine.free_download(bump_buf);
            let t_bump = t5.elapsed();

            let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
            let is_empty =
                bump.lines == 0 && bump.segments == 0 && bump.seg_counts == 0 && bump.tile == 0;
            let was_empty = LAST_FRAME_EMPTY.swap(is_empty, Ordering::Relaxed);

            // Loudly flag the transition so a black-screen event leaves a
            // single grep-able line in the log instead of vanishing into
            // thousands of identical per-frame entries.
            if is_empty != was_empty {
                log::warn!(
                    "[SCENE] frame {} transitioned to {} (lines={}, seg_counts={}, segments={}, tile={})",
                    frame_num,
                    if is_empty { "EMPTY" } else { "NON-EMPTY" },
                    bump.lines,
                    bump.seg_counts,
                    bump.segments,
                    bump.tile,
                );
            }

            // [PERF]/[BUMP] lines are debug-only so default `ekrano=info`
            // produces no per-frame stdout traffic — required for clean FPS
            // comparisons, since even once-per-second writes through `tee`
            // show up as jitter in the heartbeat. Opt in with
            // `RUST_LOG=ekrano=debug` when you actually want the rhythm.
            // Transition and overflow-retry events remain at warn/info so a
            // real problem still surfaces.
            log::debug!(
                "[PERF] frame={} resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms gpu={:.2}ms bump_read={:.2}ms total={:.2}ms",
                frame_num,
                t_resolve.as_secs_f64() * 1000.0,
                t_pool.as_secs_f64() * 1000.0,
                t_coarse.as_secs_f64() * 1000.0,
                t_fine_record.as_secs_f64() * 1000.0,
                t_gpu.as_secs_f64() * 1000.0,
                t_bump.as_secs_f64() * 1000.0,
                frame_start.elapsed().as_secs_f64() * 1000.0,
            );
            log::debug!(
                "[BUMP] frame={} lines={}, seg_counts={}, segments={}, tile={}, failed=0x{:x}",
                frame_num,
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

            log::info!(
                "Bump overflow on attempt {} (failed: 0x{:x}), retrying with larger buffers",
                attempt + 1,
                bump.failed,
            );
            // Re-enable alongside the matching diagnostic in binning.slang's
            // overflow path when chasing a STAGE_FLATTEN retry cascade. The
            // shader stashes its observed `bump.lines` / `config.lines_size`
            // into the unused `binning` / `ptcl` counters; if those differ
            // from the CPU-side values below, the GPU is reading a stale
            // config uniform rather than genuinely overflowing.
            //   log::info!(
            //       "  cpu(bump.lines={} config.lines_size={} bump.tile={} config.tiles={}) \
            //        gpu(observed_lines={} observed_lines_size={})",
            //       bump.lines,
            //       config.buffer_sizes.lines.len(),
            //       bump.tile,
            //       config.buffer_sizes.tiles.len(),
            //       bump.binning,
            //       bump.ptcl,
            //   );
            retry_config = Some(config.with_bump_estimates(&sanitize_bump(&bump)));
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
