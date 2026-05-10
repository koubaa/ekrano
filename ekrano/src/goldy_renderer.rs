// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Goldy-based renderer for offscreen and texture rendering.
//!
//! Use this when building with `--no-default-features --features goldy`.

use goldy::types::{SpatialAccess, TextureFlags, TextureFormat};
use goldy::{BufferPool, Device, Frame, Texture, TimelineValue};

use crate::{
    Error, RenderParams, Result, Scene,
    goldy_engine::GoldyEngine,
    low_level::{ImageProxy, Recording},
    render::{self, Render},
    shaders::{self, FullShaders},
};
use ekrano_encoding::{BumpAllocators, Resolver};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const MAX_BUMP_RETRIES: usize = 2;

/// Extra space added to the storage pool beyond the exact allocation required.
///
/// Provides headroom for sub-allocation rounding and small over-allocations
/// from the bump allocator growth path without triggering a full pool realloc.
const POOL_SIZE_SLACK: u64 = 256 * 1024;

/// Per-frame render statistics returned by [`GoldyRenderer::render_to_texture`].
///
/// Non-zero `bump_retries` means the GPU bump allocator overflowed at least once
/// and the frame was re-rendered with larger buffers.  Callers that want to surface
/// this to the user (e.g. to detect scenes that are too complex for the default
/// buffer estimates) can print a warning when `bump_retries > 0`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameStats {
    /// Number of times the bump allocator overflowed and the frame was retried.
    /// Zero on a clean frame.
    pub bump_retries: u32,
}

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

/// Output of [`GoldyRenderer::prepare_frame_recording`], carrying everything
/// needed by the two public render entry points after the shared pipeline runs.
struct PreparedFrame {
    stats: FrameStats,
    recording: Recording,
    out_image: ImageProxy,
    t_drain: Duration,
    t_resolve: Duration,
    t_pool: Duration,
    t_coarse: Duration,
    t_fine_record: Duration,
}

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
    /// **Pipelined:** drains the *previous* frame's GPU work at the start,
    /// then submits the current frame and returns without waiting.  The bump
    /// allocator check uses the previous frame's data — if it overflowed,
    /// buffers are grown for the current frame.  This gives one frame of
    /// CPU/GPU overlap and eliminates the per-frame synchronization stall
    /// that previously bottlenecked throughput.
    ///
    /// Returns [`FrameStats`] on success. Check [`FrameStats::bump_retries`] to detect
    /// scenes that required buffer reallocation (e.g. to print a warning to stdout).
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        scene: &Scene,
        texture: &Texture,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        use std::time::Instant;
        let frame_start = Instant::now();

        let PreparedFrame {
            stats,
            recording,
            out_image,
            t_drain,
            t_resolve,
            t_pool,
            t_coarse,
            t_fine_record,
        } = self.prepare_frame_recording(device, scene, params)?;

        let t4 = Instant::now();
        self.engine.run_recording(
            device,
            &recording,
            Some((&out_image, texture)),
            "coarse+fine",
        )?;
        let t_submit = t4.elapsed();

        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        log::debug!(
            "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms",
            frame_num,
            t_drain.as_secs_f64() * 1000.0,
            t_resolve.as_secs_f64() * 1000.0,
            t_pool.as_secs_f64() * 1000.0,
            t_coarse.as_secs_f64() * 1000.0,
            t_fine_record.as_secs_f64() * 1000.0,
            t_submit.as_secs_f64() * 1000.0,
            frame_start.elapsed().as_secs_f64() * 1000.0,
        );

        Ok(stats)
    }

    /// Like [`Self::render_to_texture`], but records compute into a swapchain [`Frame`] via
    /// [`Frame::submit_compute`] instead of standalone [`Device::submit`].
    ///
    /// After this returns, call [`goldy::Frame::present`] and then [`Self::note_frame_presented`]
    /// with the returned [`TimelineValue`] before the next `begin` / render call.
    pub fn render_to_frame(
        &mut self,
        device: &Device,
        scene: &Scene,
        frame: &Frame,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        use std::time::Instant;
        let frame_start = Instant::now();

        let PreparedFrame {
            stats,
            recording,
            out_image,
            t_drain,
            t_resolve,
            t_pool,
            t_coarse,
            t_fine_record,
        } = self.prepare_frame_recording(device, scene, params)?;

        let t4 = Instant::now();
        self.engine
            .run_recording_to_frame(device, &recording, &out_image, frame, "coarse+fine")?;
        let t_submit = t4.elapsed();

        let frame_num = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
        log::debug!(
            "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms (surface)",
            frame_num,
            t_drain.as_secs_f64() * 1000.0,
            t_resolve.as_secs_f64() * 1000.0,
            t_pool.as_secs_f64() * 1000.0,
            t_coarse.as_secs_f64() * 1000.0,
            t_fine_record.as_secs_f64() * 1000.0,
            t_submit.as_secs_f64() * 1000.0,
            frame_start.elapsed().as_secs_f64() * 1000.0,
        );

        Ok(stats)
    }

    /// Acknowledge a swapchain frame after [`goldy::Frame::present`]; required after [`Self::render_to_frame`].
    pub fn note_frame_presented(&mut self, tv: TimelineValue) -> Result<()> {
        self.engine.after_surface_present(tv)
    }

    /// Render a scene and return the pixel data as RGBA bytes.
    ///
    /// Unlike [`render_to_texture`](Self::render_to_texture), this path is
    /// **synchronous**: it waits for GPU completion and retries on bump
    /// overflow to guarantee correct output for screenshots / headless
    /// rendering.
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
        .map_err(|e| Error::Gpu(e.to_string()))?;

        // render_to_texture is pipelined (returns before GPU finishes), so
        // force a synchronous drain + bump check here with retries.
        for _attempt in 0..=MAX_BUMP_RETRIES {
            self.render_to_texture(device, scene, &texture, params)?;
            self.engine.finish_frame_for_readback(device)?;

            match self.engine.last_drained_bump() {
                Some(bump) if bump.failed != 0 => {
                    log::info!(
                        "Bump overflow in render_to_buffer (0x{:x}), retrying",
                        bump.failed,
                    );
                    // Next render_to_texture will see the overflow via
                    // take_last_drained_bump and grow buffers automatically.
                }
                _ => break,
            }
        }

        // Warn if bump overflows persisted through all retries — the output may be incomplete.
        if let Some(bump) = self.engine.last_drained_bump()
            && bump.failed != 0
        {
            log::warn!(
                "render_to_buffer: bump overflow (0x{:x}) persisted after {} retries; \
                 output may be incomplete",
                bump.failed,
                MAX_BUMP_RETRIES
            );
        }

        // Free pool and transient buffer memory before readback so the staging
        // buffer allocation doesn't fail on memory-constrained workloads.
        self.engine.release_pool(device)?;

        let mut output = vec![0_u8; texture.byte_size()];
        texture
            .read_to_cpu(&mut output)
            .map_err(|e| Error::Readback(e.to_string()))?;
        Ok(output)
    }

    /// Shared pipeline for both [`render_to_texture`](Self::render_to_texture) and
    /// [`render_to_frame`](Self::render_to_frame).
    ///
    /// Drains the previous frame, resolves the scene encoding, builds the render config,
    /// prepares the storage pool, and records the coarse + fine pass. Returns the
    /// completed [`Recording`], the output image proxy, per-phase timing, and frame stats.
    fn prepare_frame_recording(
        &mut self,
        device: &Device,
        scene: &Scene,
        params: &RenderParams,
    ) -> Result<PreparedFrame> {
        use std::time::Instant;

        let encoding = scene.encoding();
        let mut stats = FrameStats::default();

        // ---- Pipelined drain: wait for the PREVIOUS frame's GPU work ----
        // By now the GPU has been executing the previous frame concurrently
        // with CPU event-loop processing, so this wait is often near-zero.
        let t_drain_start = Instant::now();
        self.engine.finish_frame_for_readback(device)?;
        let t_drain = t_drain_start.elapsed();

        // Check if the previous frame's bump allocators overflowed.
        // If so, grow buffers for THIS frame.
        let prev_bump = self.engine.take_last_drained_bump();

        if let Some(ref bump) = prev_bump {
            let frame_num = FRAME_COUNTER.load(Ordering::Relaxed);
            log::debug!(
                "[BUMP] frame={} lines={}, seg_counts={}, segments={}, tile={}, failed=0x{:x}",
                frame_num,
                bump.lines,
                bump.seg_counts,
                bump.segments,
                bump.tile,
                bump.failed,
            );
        }

        let t0 = Instant::now();
        let config = {
            let mut packed = vec![];
            let (layout, _, _) = self.resolver.resolve(encoding, &mut packed);
            let base = ekrano_encoding::RenderConfig::new(
                &layout,
                params.width,
                params.height,
                &params.base_color,
            );
            if let Some(ref bump) = prev_bump
                && bump.failed != 0
            {
                stats.bump_retries += 1;
                log::info!(
                    "Previous frame bump overflow (0x{:x}), growing buffers",
                    bump.failed,
                );
                base.with_bump_estimates(&sanitize_bump(bump))
            } else {
                base
            }
        };
        let t_resolve = t0.elapsed();

        let base = BufferPool::padded_size(&config.buffer_sizes.pool_allocs());
        let pool_size = base.saturating_add(POOL_SIZE_SLACK);

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
        let out_image = render.out_image();
        let filter_layers = render.filter_layer_textures();
        let t_coarse = t2.elapsed();

        let t3 = Instant::now();
        render.record_fine(encoding, &self.shaders, &mut recording);
        render::record_filter_effects(
            encoding,
            &self.shaders,
            &mut recording,
            params.width,
            params.height,
            &filter_layers,
            out_image,
        );
        let t_fine_record = t3.elapsed();

        #[cfg(feature = "debug_layers")]
        if let Some(captured) = render.take_captured_buffers() {
            captured.release_buffers(&mut recording);
        }

        Ok(PreparedFrame {
            stats,
            recording,
            out_image,
            t_drain,
            t_resolve,
            t_pool,
            t_coarse,
            t_fine_record,
        })
    }
}
