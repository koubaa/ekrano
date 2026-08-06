// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained-`Scheme`-based renderer.
//!
//! Shares infrastructure types from [`crate::goldy_renderer`] (`PersistentState`,
//! `PreparedFrame`, pools, etc.).
//!
//! # Surface rendering
//!
//! `SchemeRenderer` uses the scheme-native present mechanism via [`goldy::SurfaceExchange`]:
//!
//! 1. [`goldy::SurfaceExchange`] is passed by the caller each frame.
//! 2. **Direct present (Metal):** [`goldy::SurfaceExchange::bind_destination`] runs
//!    before coarse/fine recording; fine and filter composites use the returned lease.
//!    **Copy path (other backends):** after recording, [`goldy::SurfaceExchange::bind`]
//!    records a copy from `out_image` to the surface destination.
//!    Either path returns a [`goldy::Transaction`].
//! 3. [`goldy::Scheme::submit`] submits non-present partitions first, then acquires the
//!    drawable when the present partition is about to run (deferred acquire); fall back
//!    to headless [`goldy::Scheme::submit`] only when no surface exchange is bound.
//! 4. [`goldy::Transaction::claim`] extracts a [`goldy::Claim`] from the submission.
//! 5. [`PresentToken::present`] performs scanout — synchronously in
//!    [`SchemeRenderer::render_to_swapchain`], or async on `TID_PRESENT` via
//!    [`SchemeRenderer::submit_to_swapchain`] + velato's `Presenter`.

use std::mem;
use std::sync::Arc;

#[cfg(feature = "debug_layers")]
use goldy::Buffer;
use goldy::types::{BackendType, ResourceAccess, TextureFlags, TextureFormat, TextureKind};
use goldy::{
    BudgetPolicy, ComputePipeline, Context, Device, FrameHandle, FrameOrchestrator, MemoryExchange, Scheme,
    ShaderModule, Signal, Texture,
};

#[cfg(feature = "debug_layers")]
use crate::scheme_gpu_resources::record_upload_bytes_owned;
#[cfg(debug_assertions)]
use crate::worker_retention::{debug_assert_retained_worker_resources, worker_resource_handles};
use crate::{
    Error, RenderParams, Result, Scene,
    goldy_renderer::{
        AllocatorStats, CacheScheduleOutcome, FRAME_COUNTER, FRAME_PIPELINE_DEPTH, FrameFinishOutcome, FrameStats,
        GoldyShader, MAX_BUMP_RETRIES, PersistentState, PreparedFrame, PresentToken, ResourcePoolStats,
        SceneGrowthStats, defer_frame_gpu_resources, env_robust_override, sanitize_bump,
    },
    resource_proxy::{BindType, ShaderId},
    scheme_gpu_resources::{GpuBinding, acquire_texture_rgba, bind_type_to_node_access},
    scheme_render::Render,
    shaders::{self, FullShaders},
    worker_retention::{
        note_scene_growth_frame, note_upload_rerecord_scene_bucket, note_worker_rerecord_scene_bucket,
        predict_worker_stale, resource_dims, upload_key_from, worker_stale_reasons, worker_topology,
    },
};
use ekrano_encoding::{BumpAllocators, Images, Layout, Ramps, RenderConfig, Resolver};

// -----------------------------------------------------------------------
// SchemeRenderer — Scheme-based renderer
// -----------------------------------------------------------------------

/// Goldy-based 2D renderer using the retained-[`Scheme`] backend.
///
/// All rendering is done via Goldy's [`Scheme`] command recording.
/// Surface presentation uses [`goldy::SurfaceExchange`] + [`goldy::Transaction`] /
/// [`goldy::Claim`] (scheme-native present mechanism).
pub struct SchemeRenderer {
    device: Device,
    context: Context,
    shaders: FullShaders,
    resolver: Resolver,
    engine_shaders: Vec<GoldyShader>,
    /// Cross-frame GPU resources: pools, texture cache, bump readback.
    persistent: PersistentState,
    /// Pipelined frame scheduling: depth enforcement and timeline tracking.
    frame_pipeline: FrameOrchestrator,
    /// When true (DX12/Vulkan/Metal), reuse ordering is enforced via scheme submit sidecars and
    /// frames close with [`FrameOrchestrator::end_frame_externally_ordered`] — no
    /// coarse `begin_frame` GPU wait. Backends without `host_sidecar_on_submit_worker`
    /// keep the blocking ring path.
    nonblocking_reuse: bool,
    /// Persistent bump estimates: running max across frames.
    persistent_bump: Option<BumpAllocators>,
    /// Frame counter for rate-limiting housekeeping operations.
    cleanup_frame_counter: u64,
    /// When true (Metal), upload declarations and copies are recorded on the worker
    /// scheme so one submit emits a single upload+compute command buffer.
    metal_fused_upload: bool,
    /// Retained worker scheme: compute topology recorded once per fingerprint.
    worker: Scheme,
    /// Persistent upload scheme: per-frame property writes without churning worker topology.
    upload: Scheme,
    /// Standalone scheme for headless texture withdraw (topology-invisible to the worker).
    readback: Scheme,
    /// Cumulative worker topology records across worker replacements (tests / diagnostics).
    #[cfg(test)]
    worker_record_epochs: u64,
    /// Cumulative upload scheme records across upload replacements (tests / diagnostics).
    #[cfg(test)]
    upload_record_epochs: u64,
}

impl SchemeRenderer {
    /// Create a new Scheme renderer for the given device.
    pub fn new(device: &Device) -> Result<Self> {
        let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new");

        let device = device.clone();

        device
            .ensure_allocation_policy(Arc::new(BudgetPolicy::new()))
            .map_err(|e| Error::Gpu(e.to_string()))?;

        let context = device.create_context().map_err(|e| Error::Gpu(e.to_string()))?;
        let frame_pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new.frame_orchestrator");
            FrameOrchestrator::new(&context, FRAME_PIPELINE_DEPTH)
        };
        let nonblocking_reuse = device.capabilities().host_sidecar_on_submit_worker;
        let metal_fused_upload = device.backend_type() == BackendType::Metal;
        let worker = Scheme::new(&context);
        let upload = Scheme::new(&context);
        let readback = Scheme::new(&context);
        let mut renderer = Self {
            device: device.clone(),
            context,
            shaders: FullShaders::empty(),
            resolver: Resolver::new(),
            engine_shaders: Vec::new(),
            persistent: PersistentState::new(&device),
            frame_pipeline,
            nonblocking_reuse,
            metal_fused_upload,
            persistent_bump: None,
            cleanup_frame_counter: 0,
            worker,
            upload,
            readback,
            #[cfg(test)]
            worker_record_epochs: 0,
            #[cfg(test)]
            upload_record_epochs: 0,
        };
        let shaders = {
            let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new.compile_shaders");
            shaders::goldy_full_shaders_scheme(&mut renderer)?
        };
        renderer.shaders = shaders;
        {
            let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new.release_compiler");
            device.release_idle_shader_compiler();
        }
        Ok(renderer)
    }
}

impl SchemeRenderer {
    // =======================================================================
    // Internal helpers — pool sizing & bump persistence
    // =======================================================================

    fn apply_bump_feedback(
        &mut self,
        prev_bump: Option<BumpAllocators>,
        layout: &Layout,
        params: &RenderParams,
        config: &mut RenderConfig,
        stats: &mut FrameStats,
    ) {
        let prev_bump = prev_bump.filter(|b| {
            let any_nonzero = b.lines > 0
                || b.seg_counts > 0
                || b.segments > 0
                || b.tile > 0
                || b.binning > 0
                || b.ptcl > 0
                || b.blend > 0;
            if !any_nonzero && b.failed == 0 {
                log::debug!("Ignoring all-zero bump readback (likely stale)");
                return false;
            }
            true
        });

        if let Some(ref bump) = prev_bump {
            let frame_num = FRAME_COUNTER.load(std::sync::atomic::Ordering::Relaxed);
            log::debug!(
                "[BUMP] frame={} lines={}, seg_counts={}, segments={}, tile={}, failed=0x{:x}",
                frame_num,
                bump.lines,
                bump.seg_counts,
                bump.segments,
                bump.tile,
                bump.failed,
            );
            self.update_persistent_bump(&sanitize_bump(bump));

            if bump.failed != 0 {
                stats.bump_retries += 1;
                log::info!("Previous frame bump overflow (0x{:x}), growing buffers", bump.failed);
                *config = RenderConfig::new(layout, params.width, params.height, &params.base_color)
                    .with_bump_estimates(&sanitize_bump(bump));
            }
        }
    }

    fn update_persistent_bump(&mut self, bump: &BumpAllocators) {
        let p = self.persistent_bump.get_or_insert(BumpAllocators {
            failed: 0,
            binning: 0,
            ptcl: 0,
            tile: 0,
            seg_counts: 0,
            segments: 0,
            blend: 0,
            lines: 0,
        });
        p.binning = p.binning.max(bump.binning);
        p.ptcl = p.ptcl.max(bump.ptcl);
        p.tile = p.tile.max(bump.tile);
        p.seg_counts = p.seg_counts.max(bump.seg_counts);
        p.segments = p.segments.max(bump.segments);
        p.blend = p.blend.max(bump.blend);
        p.lines = p.lines.max(bump.lines);
    }

    // =======================================================================
    // Public API
    // =======================================================================

    /// Returns a clone of the renderer's submission [`goldy::Context`].
    pub fn submission_context(&self) -> Context {
        self.context.clone()
    }

    /// Drain goldy signals and reclaim GPU resources tied to completed frames.
    pub fn poll_and_reclaim(&mut self) {
        for signal in self.context.poll_signals_and_service() {
            match signal {
                Signal::Oversubscribed { .. } => {
                    if self.frame_pipeline.wait_for_progress().is_err() {
                        break;
                    }
                }
                Signal::SwapchainReturned { .. } => {}
                Signal::SwapchainAcquired { .. } => {}
            }
        }
    }

    /// Renders a scene to a texture (offscreen; no swapchain).
    pub fn render_to_texture(&mut self, scene: &Scene, texture: &Texture, params: &RenderParams) -> Result<FrameStats> {
        self.poll_and_reclaim();
        self.run_frame(scene, params, Some(texture), None)
    }

    /// Render a scene to a swapchain using the scheme-native present mechanism.
    ///
    /// `surface` must have been created with the same [`Context`] as this renderer.
    /// Each call binds the rendered output to the surface, submits the scheme, and returns
    /// a [`PresentToken`] for scanout via [`PresentToken::present`].
    pub fn render_to_swapchain(
        &mut self,
        scene: &Scene,
        surface: &goldy::SurfaceExchange,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        let _tz = goldy::tracy_zone!("ekrano.render_to_swapchain");
        let prepared = self.prepare(scene, params)?;
        let (stats, token) = self.submit_to_swapchain(prepared, surface)?;
        self.present_to_swapchain(token)?;
        Ok(stats)
    }

    /// Phase 1: resolve scene encoding to CPU buffers.
    pub fn prepare(&mut self, scene: &Scene, params: &RenderParams) -> Result<PreparedFrame> {
        let _tz = goldy::tracy_zone!("ekrano.prepare");
        let encoding = scene.encoding();
        let mut params = RenderParams {
            base_color: params.base_color,
            width: params.width,
            height: params.height,
            antialiasing_method: params.antialiasing_method,
            robust: params.robust,
        };
        if let Some(robust) = env_robust_override() {
            params.robust = robust;
        }

        let mut resolver = mem::take(&mut self.resolver);
        let mut packed = vec![];
        let (layout, ramps, images) = {
            let _rz = goldy::tracy_zone!("ekrano.resolve");
            resolver.resolve(encoding, &mut packed)
        };

        let base_config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);
        let config = if let Some(ref persistent) = self.persistent_bump {
            base_config.with_bump_estimates(persistent)
        } else {
            base_config
        };
        Ok(PreparedFrame {
            packed,
            layout,
            ramps_data: ramps.data.to_vec(),
            ramps_width: ramps.width,
            ramps_height: ramps.height,
            images_width: images.width,
            images_height: images.height,
            image_entries: images.images.to_vec(),
            config,
            params,
            resolver,
            coverage_mask: encoding.coverage_mask.clone(),
            layer_filter_effects: encoding.layer_filter_effects.clone(),
        })
    }

    /// Phase 2: record GPU work and return frame stats plus a present token.
    ///
    /// Uses the scheme-native present mechanism: records a surface bind via
    /// [`goldy::SurfaceExchange::bind`], submits the scheme, and returns a
    /// [`PresentToken`] for scanout (does not call [`PresentToken::present`]).
    pub fn submit_to_swapchain(
        &mut self,
        prepared: PreparedFrame,
        surface: &goldy::SurfaceExchange,
    ) -> Result<(FrameStats, PresentToken)> {
        self.submit_to_swapchain_with(prepared, surface, || Ok(()))
    }

    /// Like [`Self::submit_to_swapchain`], but runs `pre_acquire` after the upload
    /// scheme is submitted and immediately before the worker scheme submits (including
    /// deferred swapchain acquire at the present partition). Callers use this to defer
    /// present-ack backpressure to that boundary so upload recording/submit overlaps the
    /// previous frame's present.
    pub fn submit_to_swapchain_with<F>(
        &mut self,
        prepared: PreparedFrame,
        surface: &goldy::SurfaceExchange,
        pre_acquire: F,
    ) -> Result<(FrameStats, PresentToken)>
    where
        F: FnOnce() -> Result<()>,
    {
        let _tz = goldy::tracy_zone!("ekrano.submit_to_swapchain");
        self.poll_and_reclaim();
        let (stats, token) = self.run_frame_from_prepared(prepared, None, Some(surface), pre_acquire)?;
        token
            .ok_or_else(|| Error::Shader("missing present transaction for swapchain submit".into()))
            .map(|token| (stats, token))
    }

    /// Query frame-scheduling state for diagnostics or test assertions.
    pub fn allocator_stats(&self) -> AllocatorStats {
        AllocatorStats {
            cleanup_ring_depth: self.frame_pipeline.pending_frames(),
        }
    }

    /// Worker scheme retention counters (tests / diagnostics).
    #[cfg(test)]
    pub(crate) fn worker_replay_stats(&self) -> goldy::ReplayStats {
        self.worker.replay_stats()
    }

    /// Upload scheme retention counters (tests / diagnostics).
    #[cfg(test)]
    pub(crate) fn upload_replay_stats(&self) -> goldy::ReplayStats {
        if self.metal_fused_upload {
            self.worker.replay_stats()
        } else {
            self.upload.replay_stats()
        }
    }

    /// True when upload topology is recorded on the worker scheme (Metal fused path).
    #[cfg(test)]
    pub(crate) fn metal_fused_upload(&self) -> bool {
        self.metal_fused_upload
    }

    /// Worker topology records across worker replacements (tests / diagnostics).
    #[cfg(test)]
    pub(crate) fn worker_record_epochs(&self) -> u64 {
        self.worker_record_epochs
    }

    /// Upload scheme records across upload replacements (tests / diagnostics).
    #[cfg(test)]
    pub(crate) fn upload_record_epochs(&self) -> u64 {
        self.upload_record_epochs
    }

    #[cfg(test)]
    fn cached_scheme_has_out_image(&self) -> bool {
        self.persistent
            .cached_scheme_rt
            .as_ref()
            .is_some_and(|(out, _)| out.is_some())
    }

    #[cfg(test)]
    fn cached_scheme_out_image_format(&self) -> Option<TextureFormat> {
        self.persistent
            .cached_scheme_rt
            .as_ref()
            .and_then(|(out, _)| out.as_ref())
            .map(|t| t.format())
    }

    /// GPU device handle shared by this renderer.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Query retained-pool accounting for diagnostics or test assertions.
    pub fn resource_pool_stats(&self) -> ResourcePoolStats {
        let by_kind = self.persistent.retained_pool.bytes_by_kind();
        ResourcePoolStats {
            retained_pool_buffer_bytes: by_kind.buffer,
            retained_pool_texture_bytes: by_kind.texture,
        }
    }

    /// Scene capacity growth counters (bucket crossings / topology invalidations).
    pub fn scene_growth_stats(&self) -> SceneGrowthStats {
        self.persistent.scene_growth
    }

    /// Log a cumulative scene-growth summary (`RUST_LOG=ekrano::scene_growth=info`).
    pub fn log_scene_growth_summary(&self) {
        let stats = self.scene_growth_stats();
        log::info!(
            target: "ekrano::scene_growth",
            "scene growth summary: frames={} scene_bucket_crossings={} worker_rerecord_scene_bucket={} upload_rerecord_scene_bucket={} current_scene_bucket={} peak_scene_bucket={} peak_live_scene_bytes={}",
            stats.frames,
            stats.scene_bucket_crossings,
            stats.worker_rerecord_scene_bucket,
            stats.upload_rerecord_scene_bucket,
            stats.current_scene_bucket,
            stats.peak_scene_bucket,
            stats.peak_live_scene_bytes,
        );
    }

    /// Number of retained filter-uniform cache slots (tests / diagnostics).
    #[cfg(test)]
    pub(crate) fn filter_uniform_cache_len(&self) -> usize {
        self.persistent.cached_filter_uniforms.len()
    }

    /// Cached mask-atlas dimensions, if any (tests / diagnostics).
    #[cfg(test)]
    pub(crate) fn cached_mask_atlas_dims(&self) -> Option<(u32, u32)> {
        self.persistent.cached_mask_atlas.as_ref().map(|(w, h, _)| (*w, *h))
    }

    /// `true` if the submission context still holds unreclaimed deferred payloads.
    pub fn has_deferred_payloads(&self) -> bool {
        self.context.has_deferred_payloads()
    }

    /// Pull-side reclamation: drain the submission context's deferred-deletion ring.
    pub fn flush_deferred_deletions(&self) {
        self.context.flush_deferred_deletions();
    }

    /// Render a scene and return the pixel data as RGBA bytes (synchronous).
    pub fn render_to_buffer(&mut self, scene: &Scene, params: &RenderParams) -> Result<Vec<u8>> {
        for _attempt in 0..=MAX_BUMP_RETRIES {
            self.poll_and_reclaim();
            self.run_frame(scene, params, None, None)?;
            self.frame_pipeline
                .drain_all()
                .map_err(|e| Error::Shader(e.to_string()))?;
            // Must wait: with host-sidecar / nonblocking reuse the orchestrator ring
            // does not fence the scheme submission, so a poll-only drain skips bump
            // feedback and leaves overflowed frames unrecovered.
            self.persistent.wait_and_drain_bump_readbacks(&self.context)?;
            self.context.flush_deferred_deletions();

            match self.persistent.last_drained_bump() {
                Some(bump) if bump.failed != 0 => {
                    log::info!("Bump overflow in render_to_buffer (0x{:x}), retrying", bump.failed);
                    self.persistent.cached_scheme_rt = None;
                    self.persistent.cached_out_image_withdraw = None;
                }
                _ => break,
            }
        }

        if let Some(bump) = self.persistent.last_drained_bump()
            && bump.failed != 0
        {
            log::warn!(
                "render_to_buffer: bump overflow (0x{:x}) persisted after {} retries; \
                 output may be incomplete",
                bump.failed,
                MAX_BUMP_RETRIES
            );
        }

        let out_handle = self
            .persistent
            .cached_scheme_rt
            .as_ref()
            .and_then(|(t, _)| t.as_ref())
            .map(|t| t.gpu_handle())
            .ok_or_else(|| Error::Shader("render_to_buffer: missing scheme out_image".into()))?;
        let needs_rebind = self
            .persistent
            .cached_out_image_withdraw
            .as_ref()
            .map(|(h, _)| *h != out_handle)
            .unwrap_or(true);
        if needs_rebind {
            self.readback = Scheme::new(&self.context);
            let out_image = self
                .persistent
                .cached_scheme_rt
                .as_ref()
                .and_then(|(t, _)| t.as_ref())
                .expect("render_to_buffer: missing scheme out_image");
            let withdraw = MemoryExchange::new(&self.context)
                .bind_withdraw(&mut self.readback, out_image)
                .map_err(|e| Error::Readback(e.to_string()))?;
            self.persistent.cached_out_image_withdraw = Some((out_handle, withdraw));
        }
        let withdraw = self
            .persistent
            .cached_out_image_withdraw
            .as_ref()
            .map(|(_, w)| w.clone())
            .expect("render_to_buffer: out_image withdraw must be bound");
        let mut submission = self.readback.submit().map_err(|e| Error::Readback(e.to_string()))?;
        let bytes = withdraw
            .claim(&mut submission)
            .map_err(|e| Error::Readback(e.to_string()))?
            .consume()
            .map_err(|e| Error::Readback(e.to_string()))?;
        // Texture withdraws expose tight-row RGBA (`logical_bytes`). If Goldy ever returns
        // pitched footprint bytes instead, unpack here using the bind-time layout.
        Ok(match withdraw.texture_layout() {
            Some(layout)
                if bytes.len() as u64 == layout.staging_bytes && layout.row_pitch != layout.tight_row_bytes() =>
            {
                let row_bytes = layout.tight_row_bytes() as usize;
                let pitch = layout.row_pitch as usize;
                let mut output = vec![0_u8; layout.logical_bytes as usize];
                for row in 0..layout.height as usize {
                    let src_offset = layout.footprint_offset as usize + row * pitch;
                    let dst_offset = row * row_bytes;
                    output[dst_offset..dst_offset + row_bytes]
                        .copy_from_slice(&bytes[src_offset..src_offset + row_bytes]);
                }
                output
            }
            _ => bytes.into_vec(),
        })
    }

    fn drain_ready_bump_readbacks(&mut self) -> Result<()> {
        self.persistent.drain_ready_bump_readbacks(&self.context)
    }

    /// Present a swapchain token returned by [`Self::submit_to_swapchain`].
    ///
    /// On blocking backends this also stamps the frame-orchestrator ring and queues
    /// robust bump readback when configured.
    pub fn present_to_swapchain(&mut self, token: PresentToken) -> Result<()> {
        self.finish_present_token(token)
    }

    fn finish_present_token(&mut self, token: PresentToken) -> Result<()> {
        let PresentToken {
            claim,
            ring_note_submission,
        } = token;
        claim.consume().map_err(|e| Error::Shader(e.to_string()))?;
        if let Some(submission) = ring_note_submission {
            self.frame_pipeline.note_presented(&submission);
            if self.persistent.cached_bump_withdraw.is_some() {
                self.persistent.queue_bump_submission(submission);
            }
        }
        Ok(())
    }

    // =======================================================================
    // Frame execution (private)
    // =======================================================================

    fn run_frame(
        &mut self,
        scene: &Scene,
        params: &RenderParams,
        output_texture: Option<&Texture>,
        surface: Option<&goldy::SurfaceExchange>,
    ) -> Result<FrameStats> {
        let prepared = self.prepare(scene, params)?;
        let (stats, token) = self.run_frame_from_prepared(prepared, output_texture, surface, || Ok(()))?;
        if let Some(token) = token {
            self.present_to_swapchain(token)?;
        }
        Ok(stats)
    }

    /// GPU submission path for both texture and swapchain rendering.
    ///
    /// When `surface` is `Some`, records a surface bind via [`goldy::SurfaceExchange::bind`],
    /// submits, and returns a [`PresentToken`] (does not present).
    fn run_frame_from_prepared<F>(
        &mut self,
        prepared: PreparedFrame,
        output_texture: Option<&Texture>,
        surface: Option<&goldy::SurfaceExchange>,
        pre_acquire: F,
    ) -> Result<(FrameStats, Option<PresentToken>)>
    where
        F: FnOnce() -> Result<()>,
    {
        let _tz = goldy::tracy_zone!("ekrano.run_frame");
        use std::time::Instant;
        let frame_start = Instant::now();

        let packed = prepared.packed;
        let layout = prepared.layout;
        let ramps_data = prepared.ramps_data;
        let ramps_width = prepared.ramps_width;
        let ramps_height = prepared.ramps_height;
        let images_width = prepared.images_width;
        let images_height = prepared.images_height;
        let image_entries = prepared.image_entries;
        let mut config = prepared.config;
        let params = prepared.params;
        let resolver = prepared.resolver;
        let coverage_mask = prepared.coverage_mask;
        let layer_filter_effects = prepared.layer_filter_effects;
        let ramps = Ramps {
            data: &ramps_data,
            width: ramps_width,
            height: ramps_height,
        };
        let images = Images {
            width: images_width,
            height: images_height,
            images: &image_entries,
        };
        let mut stats = FrameStats::default();
        let t_resolve = frame_start.elapsed();

        // Headless default: Rgba8Unorm (matches render_to_buffer RGBA8 bytes).
        let out_image_format = surface.map(|s| s.format()).unwrap_or(TextureFormat::Rgba8Unorm);
        // Metal: write fine/filter output straight into the drawable. Other backends keep
        // the intermediate `out_image` + copy blit (DX12 flip-model cannot UAV the backbuffer).
        let direct_present =
            surface.is_some() && output_texture.is_none() && self.device.backend_type() == BackendType::Metal;
        if self.persistent.purge_render_target_cache_if_mismatch(
            &self.context,
            params.width,
            params.height,
            out_image_format,
            direct_present,
        ) {
            self.persistent.cached_out_image_withdraw = None;
            self.device.compact_overflow_heaps();
        }

        let t_drain_start = Instant::now();

        let _tz_begin = goldy::tracy_zone!("ekrano.begin_frame");
        let frame_handle = self
            .frame_pipeline
            .begin_frame()
            .map_err(|e| Error::Shader(e.to_string()))?;
        self.drain_ready_bump_readbacks()?;
        self.cleanup_frame_counter = self.cleanup_frame_counter.wrapping_add(1);
        if self.cleanup_frame_counter.is_multiple_of(64) {
            self.device.compact_overflow_heaps();
        }
        let t_drain = t_drain_start.elapsed();

        let prev_bump = self.persistent.take_last_drained_bump();
        self.apply_bump_feedback(prev_bump, &layout, &params, &mut config, &mut stats);

        let t1 = Instant::now();
        if self.persistent.linear_clamp_sampler.is_none() {
            self.persistent.linear_clamp_sampler =
                Some(goldy::Sampler::linear(&self.device).map_err(|e| Error::Gpu(e.to_string()))?);
        }
        if self.persistent.nearest_clamp_sampler.is_none() {
            self.persistent.nearest_clamp_sampler =
                Some(goldy::Sampler::nearest(&self.device).map_err(|e| Error::Gpu(e.to_string()))?);
        }
        self.context.flush_deferred_deletions();
        let t_pool = t1.elapsed();

        let t2 = Instant::now();
        let scene_bucket = crate::worker_retention::scene_size_bucket(packed.len());
        note_scene_growth_frame(&mut self.persistent.scene_growth, packed.len(), scene_bucket);
        let coverage_mask_dims = coverage_mask.as_ref().map(|m| (m.width, m.height));
        let image_regions: Vec<(u32, u32, u32, u32)> = image_entries
            .iter()
            .map(|(img, x, y)| (*x, *y, img.width, img.height))
            .collect();
        let dims = resource_dims(
            scene_bucket,
            ramps_width,
            ramps_height,
            image_entries.len(),
            images_width,
            images_height,
            coverage_mask_dims,
            &image_regions,
        );
        let topology = worker_topology(
            &params,
            &config,
            out_image_format,
            &dims,
            surface.is_some(),
            direct_present,
        );
        let upload_key = upload_key_from(&dims);
        let upload_needs_record = crate::worker_retention::upload_stale(&self.persistent, &upload_key);
        if upload_needs_record
            && let Some(old_key) = self.persistent.cached_upload_key.as_ref()
            && old_key.scene_bucket != upload_key.scene_bucket
        {
            note_upload_rerecord_scene_bucket(
                &mut self.persistent.scene_growth,
                old_key.scene_bucket,
                upload_key.scene_bucket,
            );
        }
        if upload_needs_record && !self.metal_fused_upload {
            self.upload = Scheme::new(&self.context);
            // Logical upload declarations live on the scheme; drop cached handles.
            self.persistent.clear_deposit_declarations();
            #[cfg(test)]
            {
                self.upload_record_epochs += 1;
            }
        }

        let output_tex_handle = output_texture.map(|t| t.gpu_handle());
        // Predict worker re-record *before* prepare whenever the worker must be replaced.
        // Filter frames always re-record (one-shot scratch deeds). Metal fused also
        // replaces early so prepare cannot consume cached_pipeline/RTs onto a throwaway
        // worker and then allocate a duplicate set on re-prepare.
        let worker_stale_predicted = predict_worker_stale(
            &self.persistent,
            &topology,
            &layer_filter_effects,
            output_tex_handle,
            params.width,
            params.height,
            out_image_format,
        );
        let replace_worker_before_prepare = worker_stale_predicted || (self.metal_fused_upload && upload_needs_record);
        let metal_fused_rerecord = self.metal_fused_upload && replace_worker_before_prepare;

        if replace_worker_before_prepare {
            let _tz = goldy::tracy_zone!("ekrano.worker_record");
            self.worker = Scheme::new(&self.context);
            self.persistent.cached_bump_withdraw = None;
            self.persistent.cached_present_tx = None;
            #[cfg(debug_assertions)]
            {
                self.persistent.cached_worker_resources = None;
            }
            if self.metal_fused_upload {
                // Upload topology lives on the worker scheme; drop cached declarations.
                self.persistent.clear_deposit_declarations();
                #[cfg(test)]
                {
                    self.upload_record_epochs += 1;
                }
            }
        }

        let (mut pipeline, out_image_handle, worker_stale) = {
            let mut recorder = SchemeRecorder::new(
                &self.device,
                &self.context,
                &mut self.worker,
                &mut self.upload,
                upload_needs_record || metal_fused_rerecord,
                self.metal_fused_upload,
                self.nonblocking_reuse,
                &mut self.frame_pipeline,
                frame_handle,
                &mut self.persistent,
                &self.engine_shaders,
            );
            let pipeline = {
                let _tz = goldy::tracy_zone!("ekrano.prepare");
                let pipeline_result = recorder.prepare_pipeline_resources(
                    coverage_mask.as_ref(),
                    packed,
                    ramps,
                    images,
                    &params,
                    &config,
                    out_image_format,
                    direct_present,
                );
                self.resolver = resolver;
                match pipeline_result {
                    Ok(p) => p,
                    Err(e) => {
                        recorder.dismiss();
                        return Err(e);
                    }
                }
            };
            let out_image_handle = pipeline
                .out_image
                .as_ref()
                .and_then(|t| t.handle(ResourceAccess::Write));
            recorder.dismiss();
            if upload_needs_record {
                self.persistent.cached_upload_key = Some(upload_key);
            }

            let worker_stale = if replace_worker_before_prepare {
                true
            } else {
                let _tz = goldy::tracy_zone!("ekrano.worker_stale_check");
                worker_stale_reasons(
                    &self.persistent,
                    &topology,
                    &layer_filter_effects,
                    out_image_handle,
                    output_tex_handle,
                )
            };
            debug_assert!(
                !(self.metal_fused_upload && worker_stale && !replace_worker_before_prepare),
                "Metal fused worker re-record should have been predicted before prepare"
            );
            (pipeline, out_image_handle, worker_stale)
        };

        #[cfg(debug_assertions)]
        if !worker_stale && let Some(recorded) = self.persistent.cached_worker_resources.as_ref() {
            let current = worker_resource_handles(
                &pipeline.scene,
                &pipeline.bump,
                &pipeline.gradient,
                &pipeline.image_atlas,
                &pipeline.mask_atlas,
                pipeline.out_image.as_ref(),
            );
            debug_assert_retained_worker_resources(recorded, &current);
        }

        #[cfg(debug_assertions)]
        let mut debug_recorded_resources = None;

        if worker_stale && !replace_worker_before_prepare {
            let _tz = goldy::tracy_zone!("ekrano.worker_record");
            self.worker = Scheme::new(&self.context);
            self.persistent.cached_bump_withdraw = None;
            self.persistent.cached_present_tx = None;
            #[cfg(debug_assertions)]
            {
                self.persistent.cached_worker_resources = None;
            }
        }

        if worker_stale
            && let Some(old_topo) = self.persistent.cached_worker_topology.as_ref()
            && old_topo.scene_bucket != topology.scene_bucket
        {
            note_worker_rerecord_scene_bucket(
                &mut self.persistent.scene_growth,
                old_topo.scene_bucket,
                topology.scene_bucket,
            );
        }

        let mut recorder = SchemeRecorder::new(
            &self.device,
            &self.context,
            &mut self.worker,
            &mut self.upload,
            upload_needs_record || metal_fused_rerecord,
            self.metal_fused_upload,
            self.nonblocking_reuse,
            &mut self.frame_pipeline,
            frame_handle,
            &mut self.persistent,
            &self.engine_shaders,
        );

        let mut render = Render::new();
        let mut worker_cache = None;

        let (t_coarse, t_fine_record) = if worker_stale {
            let mut present_bound_lease = None;
            let mut early_present_tx = None;
            if direct_present {
                let surface = surface.expect("direct present requires surface");
                let (lease, tx) = surface
                    .bind_destination(recorder.scheme())
                    .map_err(|e| Error::Shader(e.to_string()))?;
                present_bound_lease = Some(lease);
                early_present_tx = Some(tx);
            }
            let present_lease = present_bound_lease.as_ref();

            {
                let _tz = goldy::tracy_zone!("ekrano.coarse");
                render.run_coarse(
                    &mut pipeline,
                    &self.shaders,
                    &params,
                    params.robust,
                    &config,
                    &mut recorder,
                );
            }
            let t_coarse = t2.elapsed();

            let render_output = crate::scheme_render::resolve_render_output(&pipeline, output_texture, present_lease);

            let t3 = Instant::now();
            {
                let _tz = goldy::tracy_zone!("ekrano.fine");

                render.record_fine(
                    &layer_filter_effects,
                    &self.shaders,
                    &pipeline,
                    render_output,
                    &mut recorder,
                );
                #[cfg(feature = "debug_layers")]
                let _ = render.take_captured_buffers();
                crate::scheme_render::record_filter_effects(
                    &layer_filter_effects,
                    &self.shaders,
                    &mut recorder,
                    &pipeline,
                    render_output,
                );
                let keep = recorder.filter_dispatch_slot;
                recorder.persistent.trim_filter_uniform_cache(recorder.context(), keep);
            }
            let t_fine_record = t3.elapsed();

            let present_tx = if direct_present {
                early_present_tx
            } else if let Some(surface) = surface {
                Some(recorder.bind_surface(
                    surface,
                    pipeline.out_image.as_ref().expect("copy path requires out_image"),
                )?)
            } else {
                None
            };

            let bump_withdraw = if params.robust {
                Some(
                    MemoryExchange::new(recorder.context())
                        .bind_withdraw(recorder.scheme(), &pipeline.bump)
                        .map_err(|e| Error::Shader(e.to_string()))?,
                )
            } else {
                None
            };

            // Sticky worker retention only when the recorded graph has no one-shot
            // filter scratches. Filter frames always re-record; publishing a full
            // retention cache would claim a sticky worker while it still binds deeds
            // that are retired when scratches return to the transient pool.
            // Apply present/bump (and filter-effects bookkeeping) after `finish` so we
            // do not mutate `persistent` while `recorder` still borrows it.
            worker_cache = Some((
                present_tx,
                bump_withdraw,
                topology,
                layer_filter_effects.clone(),
                out_image_handle,
            ));
            #[cfg(debug_assertions)]
            {
                debug_recorded_resources = Some(worker_resource_handles(
                    &pipeline.scene,
                    &pipeline.bump,
                    &pipeline.gradient,
                    &pipeline.image_atlas,
                    &pipeline.mask_atlas,
                    pipeline.out_image.as_ref(),
                ));
            }
            #[cfg(test)]
            {
                self.worker_record_epochs += 1;
            }
            (t_coarse, t_fine_record)
        } else {
            let _tz = goldy::tracy_zone!("ekrano.worker_retained");
            // Worker retained: coarse/fine are not re-run this frame.
            // t_coarse and t_fine_record are reported as zero in FrameStats on the
            // hot retained path — the timings reflect only the recording cost, which
            // is zero by design when the COW bit is clean.
            (std::time::Duration::ZERO, std::time::Duration::ZERO)
        };

        let t4 = Instant::now();
        let cache_outcome = recorder.schedule_pipeline_cleanup(pipeline);
        let FrameFinishOutcome {
            deferred_textures,
            scheme_submission,
        } = {
            let _tz = goldy::tracy_zone!("ekrano.finish");
            recorder.finish(surface.is_some(), pre_acquire, None)?
        };

        if let Some((present, bump, topology, filter_effects, out_image)) = worker_cache {
            self.persistent.cached_present_tx = present;
            self.persistent.cached_bump_withdraw = bump;
            if filter_effects.is_empty() {
                self.persistent.cached_worker_topology = Some(topology);
                self.persistent.cached_worker_filter_effects = filter_effects;
                self.persistent.cached_worker_out_image = out_image;
                self.persistent.cached_worker_output_texture = output_texture.map(|t| t.gpu_handle());
                #[cfg(debug_assertions)]
                if let Some(resources) = debug_recorded_resources {
                    self.persistent.cached_worker_resources = Some(resources);
                }
            } else {
                // Remember filter presence for the next frame's staleness check when
                // filters are removed, without retaining the scratch-bound worker.
                self.persistent.cached_worker_filter_effects = filter_effects;
                self.persistent.cached_worker_topology = None;
                self.persistent.cached_worker_out_image = None;
                self.persistent.cached_worker_output_texture = None;
                #[cfg(debug_assertions)]
                {
                    self.persistent.cached_worker_resources = None;
                }
            }
        }

        let present_tx = surface
            .is_some()
            .then(|| self.persistent.cached_present_tx.clone())
            .flatten();

        let present_token = match (present_tx, scheme_submission) {
            (Some(tx), Some(mut submission)) => {
                let claim = tx.claim(&mut submission).map_err(|e| Error::Shader(e.to_string()))?;
                let queue_bump = params.robust && self.persistent.cached_bump_withdraw.is_some();
                let note_after_present = !self.nonblocking_reuse && surface.is_some();
                let ring_note_submission = if note_after_present {
                    Some(submission)
                } else {
                    if queue_bump {
                        self.persistent.queue_bump_submission(submission);
                    }
                    None
                };
                Some(PresentToken {
                    claim,
                    ring_note_submission,
                })
            }
            (None, Some(submission)) => {
                if params.robust && self.persistent.cached_bump_withdraw.is_some() {
                    self.persistent.queue_bump_submission(submission);
                }
                None
            }
            _ => None,
        };

        if cache_outcome.scheme_rt_stored {
            log::debug!("[RT-CACHE] scheme render targets stored for reuse");
        }
        defer_frame_gpu_resources(&self.context, &self.persistent, deferred_textures);
        // Filter scratches are returned above; their stamps are now dead on the worker.
        // Drop that scheme so the next frame cannot submit (or prepare against) retired
        // deeds — re-record must mint a fresh worker with new scratch identities.
        if !self.persistent.cached_worker_filter_effects.is_empty() {
            self.worker = Scheme::new(&self.context);
            self.persistent.cached_bump_withdraw = None;
            // Keep cached_present_tx: present claim for this frame may still be in flight.
            #[cfg(debug_assertions)]
            {
                self.persistent.cached_worker_resources = None;
            }
        }

        {
            let _tz = goldy::tracy_zone!("ekrano.run_frame.post_submit");
            self.context.flush_deferred_deletions();
            let t_submit = t4.elapsed();

            let frame_num = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let label = if surface.is_some() { "swapchain" } else { "" };

            let ring_depth = self.frame_pipeline.pending_frames();
            let rt_slots = self.persistent.cached_scheme_rt.is_some() as usize;
            let pipe_slots = self.persistent.cached_pipeline.is_some() as usize;

            log::debug!(
                "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms ring={} rt_slots={rt_slots} pipe_slots={pipe_slots} {label}",
                frame_num,
                t_drain.as_secs_f64() * 1000.0,
                t_resolve.as_secs_f64() * 1000.0,
                t_pool.as_secs_f64() * 1000.0,
                t_coarse.as_secs_f64() * 1000.0,
                t_fine_record.as_secs_f64() * 1000.0,
                t_submit.as_secs_f64() * 1000.0,
                frame_start.elapsed().as_secs_f64() * 1000.0,
                ring_depth,
            );
        }

        crate::goldy_renderer::maybe_log_gpu_memory(&self.device);

        Ok((stats, present_token))
    }

    // =======================================================================
    // Engine methods
    // =======================================================================

    /// Add a compute shader from Slang source.
    pub(crate) fn add_compute_shader(
        &mut self,
        label: &'static str,
        slang_source: &str,
        bindings: &[BindType],
        search_paths: &[&str],
        defines: &[(&str, &str)],
    ) -> Result<ShaderId> {
        self.add_compute_shader_with_options(
            label,
            slang_source,
            bindings,
            search_paths,
            defines,
            goldy::OptimizationLevel::Default,
        )
    }

    /// Add a compute shader with explicit optimization level.
    pub(crate) fn add_compute_shader_with_options(
        &mut self,
        label: &'static str,
        slang_source: &str,
        bindings: &[BindType],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: goldy::OptimizationLevel,
    ) -> Result<ShaderId> {
        let shader_module = {
            let _tz = goldy::tracy_zone!("ekrano.add_shader.slang", label);
            ShaderModule::from_slang_with_options(
                &self.device,
                slang_source,
                search_paths,
                defines,
                optimization_level,
                &[],
            )
            .map_err(|e| Error::Shader(format!("{:#}", e)))?
        };
        let pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.add_shader.pipeline", label);
            ComputePipeline::new_with_label(&self.device, &shader_module, Some(label))
                .map_err(|e| Error::Shader(format!("{:#}", e)))?
        };

        let id = ShaderId(self.engine_shaders.len());
        self.engine_shaders.push(GoldyShader {
            pipeline,
            bindings: bindings.to_vec(),
            label,
        });
        Ok(id)
    }
}
// -----------------------------------------------------------------------
// SchemeRecorder — direct-execution recorder that builds Scheme nodes
// -----------------------------------------------------------------------

pub(crate) struct SchemeRecorder<'a> {
    device: &'a Device,
    pub(crate) context: &'a Context,
    /// Retained worker scheme (compute + present topology).
    pub(crate) scheme: &'a mut Scheme,
    /// Per-frame upload scheme (property writes only); unused on Metal fused path.
    upload: &'a mut Scheme,
    /// When true (Metal), upload nodes are recorded on the worker scheme.
    metal_fused_upload: bool,
    /// When true, the upload scheme IR is empty and copy/upload nodes must be recorded this frame.
    pub(crate) upload_needs_record: bool,
    /// Nonblocking head-chases-tail path: deferred host writes + reuse epochs; no ring wait.
    pub(crate) nonblocking_reuse: bool,
    frame_pipeline: &'a mut FrameOrchestrator,
    frame_handle: FrameHandle,
    pub(crate) persistent: &'a mut PersistentState,
    pub(crate) shaders: &'a [GoldyShader],
    /// Set to `true` by `finish` or `abort`; the `Drop` impl aborts the open frame
    /// if the recorder is dropped without being properly completed (e.g. on a `?` return).
    finished: bool,
    #[cfg(feature = "debug_layers")]
    deferred_owned_buffers: Vec<Buffer>,
    deferred_textures: Vec<Texture>,
    /// Per-frame filter dispatch slot counter, incremented by each `filter_dispatch` call.
    /// Used to index into `PersistentState::cached_filter_uniforms` for cache lookup.
    /// Reset to 0 at the start of each frame.
    pub(crate) filter_dispatch_slot: usize,
}

impl<'a> SchemeRecorder<'a> {
    pub(crate) fn device(&self) -> &'a Device {
        self.device
    }

    pub(crate) fn context(&self) -> &'a Context {
        self.context
    }

    pub(crate) fn scheme(&mut self) -> &mut Scheme {
        self.scheme
    }

    pub(crate) fn upload_scheme(&mut self) -> &mut Scheme {
        if self.metal_fused_upload {
            self.scheme
        } else {
            self.upload
        }
    }

    pub(crate) fn bind_surface(
        &mut self,
        surface: &goldy::SurfaceExchange,
        source: &Texture,
    ) -> Result<goldy::Transaction> {
        surface
            .bind(self.scheme, source)
            .map_err(|e| Error::Shader(e.to_string()))
    }

    pub(crate) fn new(
        device: &'a Device,
        context: &'a Context,
        scheme: &'a mut Scheme,
        upload: &'a mut Scheme,
        upload_needs_record: bool,
        metal_fused_upload: bool,
        nonblocking_reuse: bool,
        frame_pipeline: &'a mut FrameOrchestrator,
        frame_handle: FrameHandle,
        persistent: &'a mut PersistentState,
        shaders: &'a [GoldyShader],
    ) -> Self {
        // Read capacity hints before persistent is moved into Self.
        let tex_cap = persistent.deferred_textures_cap_hint;

        Self {
            device,
            context,
            scheme,
            upload,
            upload_needs_record,
            metal_fused_upload,
            nonblocking_reuse,
            frame_pipeline,
            frame_handle,
            persistent,
            shaders,
            #[cfg(feature = "debug_layers")]
            deferred_owned_buffers: Vec::new(),
            deferred_textures: Vec::with_capacity(tex_cap),
            finished: false,
            filter_dispatch_slot: 0,
        }
    }

    /// End prepare-only use without aborting the open frame (upload scheme retains writes).
    pub(crate) fn dismiss(mut self) {
        self.finished = true;
    }

    pub(crate) fn acquire_texture_rgba(
        &mut self,
        width: u32,
        height: u32,
        access: TextureKind,
        flags: TextureFlags,
    ) -> Result<Texture, Error> {
        acquire_texture_rgba(self, width, height, access, flags)
    }

    pub(crate) fn prepare_pipeline_resources(
        &mut self,
        coverage_mask: Option<&ekrano_encoding::CoverageMask>,
        packed: Vec<u8>,
        ramps: Ramps<'_>,
        images: Images<'_>,
        params: &RenderParams,
        config: &RenderConfig,
        out_image_format: TextureFormat,
        direct_present: bool,
    ) -> Result<crate::scheme_gpu_resources::PipelineResources, Error> {
        crate::scheme_gpu_resources::PipelineResources::prepare(
            self,
            coverage_mask,
            packed,
            ramps,
            images,
            params,
            config,
            out_image_format,
            direct_present,
        )
    }

    // NOTE: There is no explicit mid-frame flush. The backend is free to split
    // a single Scheme submission into multiple command buffers as an
    // implementation detail (e.g. coarse vs fine partitions). Callers should
    // not assume or rely on intra-frame submission boundaries.

    pub(crate) fn defer_texture(&mut self, tex: Texture) {
        self.deferred_textures.push(tex);
    }

    pub(crate) fn schedule_pipeline_cleanup(
        &mut self,
        pipeline: crate::scheme_gpu_resources::PipelineResources,
    ) -> CacheScheduleOutcome {
        let mut outcome = CacheScheduleOutcome::default();
        let crate::scheme_gpu_resources::PipelineResources {
            gradient,
            image_atlas,
            mask_atlas,
            scene,
            config,
            indirect,
            stable,
            scratch,
            bump,
            out_image,
            filter_layers,
            buffer_sizes,
            config_uniform_value,
            frame_width: _,
            frame_height: _,
        } = pipeline;

        let _ = (gradient, image_atlas, mask_atlas);
        self.persistent.cached_scene = Some((scene.byte_size(), scene));
        self.persistent.cached_config_uniform = Some((config_uniform_value, config));
        if let Some((wg_counts_gpu, indirect_buf)) = indirect {
            self.persistent.cached_scheme_indirect = Some((wg_counts_gpu, indirect_buf));
        }
        self.persistent.cached_bump = Some((bump.byte_size(), bump));
        let pipeline_cache = crate::scheme_gpu_resources::CachedPipeline {
            stable,
            scratch,
            buffer_sizes,
        };
        assert!(
            self.persistent.cached_pipeline.is_none(),
            "cached_pipeline must be empty at schedule (prepare should have taken it)"
        );
        self.persistent.cached_pipeline = Some(pipeline_cache);
        log::debug!("[PIPE-CACHE] schedule: cached");
        self.persistent.store_scheme_render_targets(out_image, filter_layers);
        outcome.scheme_rt_stored = true;
        log::debug!("[RT-CACHE] schedule: scheme render targets stored");
        outcome
    }

    #[cfg(feature = "debug_layers")]
    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> Buffer {
        record_upload_bytes_owned(self, name, 1, data.into()).expect("upload failed")
    }

    #[cfg(feature = "debug_layers")]
    pub fn upload_strided(&mut self, name: &'static str, element_stride: u32, data: impl Into<Vec<u8>>) -> Buffer {
        record_upload_bytes_owned(self, name, element_stride, data.into()).expect("upload_strided failed")
    }

    pub fn dispatch(&mut self, shader: ShaderId, wg_size: (u32, u32, u32), bindings: &[GpuBinding<'_>]) {
        Self::record_dispatch(self.scheme, self.shaders, shader, wg_size, bindings, &[]);
    }

    #[allow(dead_code, reason = "kept for reintroducing config-based flatten chunking")]
    pub fn dispatch_with_push_tail(
        &mut self,
        shader: ShaderId,
        wg_size: (u32, u32, u32),
        bindings: &[GpuBinding<'_>],
        push_tail: &[u32],
    ) {
        Self::record_dispatch(self.scheme, self.shaders, shader, wg_size, bindings, push_tail);
    }

    /// Record a compute dispatch without holding `&mut SchemeRecorder`.
    ///
    /// Use this (with split field borrows from the recorder) when `bindings` may
    /// reference resources in `recorder.persistent`, which cannot coexist with
    /// `recorder.dispatch()`'s whole-recorder mutable borrow.
    pub(crate) fn record_dispatch(
        scheme: &mut Scheme,
        shaders: &[GoldyShader],
        shader_id: ShaderId,
        wg_size: (u32, u32, u32),
        bindings: &[GpuBinding<'_>],
        push_tail: &[u32],
    ) {
        Self::dispatch_inner(scheme, shaders, shader_id, wg_size, bindings, push_tail);
    }

    fn dispatch_inner(
        scheme: &mut Scheme,
        shaders: &[GoldyShader],
        shader_id: ShaderId,
        (x, y, z): (u32, u32, u32),
        bindings: &[GpuBinding<'_>],
        push_tail: &[u32],
    ) {
        if x == 0 || y == 0 || z == 0 {
            log::warn!(
                "Skipping Dispatch for shader {} with zero grid dimension ({x}, {y}, {z}); \
                 this may indicate a bug in the caller",
                shader_id.0
            );
            return;
        }
        let bind_types = &shaders[shader_id.0].bindings;
        let label = shaders[shader_id.0].label;
        debug_assert_eq!(
            bind_types.len(),
            bindings.len(),
            "shader {} bind metadata count ({}) must match runtime binding count ({})",
            label,
            bind_types.len(),
            bindings.len(),
        );

        let mut node = scheme.node(label, &shaders[shader_id.0].pipeline);
        for (i, binding) in bindings.iter().enumerate() {
            node = if let GpuBinding::Present(lease, present_access) = binding {
                node.with_present_access(lease, *present_access)
            } else {
                let access = bind_type_to_node_access(bind_types[i]);
                match binding {
                    GpuBinding::Buf(b) => node.with_parcel(*b, access),
                    GpuBinding::Parcel(p) => node.with_parcel(*p, access),
                    GpuBinding::Tex(t) => node.with_parcel(*t, access),
                    GpuBinding::Sampler(s) => node.with_parcel(*s, access),
                    GpuBinding::Present(..) => unreachable!(),
                }
            };
        }
        for &val in push_tail {
            node = node.with_param(val);
        }
        node.dispatch(x, y, z);
    }

    /// Issue an indirect compute dispatch using a [`goldy::DispatchShape`] buffer as the
    /// workgroup-count source.  The `shape` buffer must contain exactly one
    /// `DispatchShape` element; the scheme ordering engine automatically registers
    /// it as a read dependency so that any preceding write to the buffer
    /// (e.g. from `path_count_setup_scheme`) is correctly ordered before this node.
    pub fn dispatch_shape(&mut self, shader: ShaderId, shape: &goldy::Parcel, bindings: &[GpuBinding<'_>]) {
        let bind_types = &self.shaders[shader.0].bindings;
        let label = self.shaders[shader.0].label;
        debug_assert_eq!(
            bind_types.len(),
            bindings.len(),
            "shader {} bind metadata count ({}) must match runtime binding count ({})",
            label,
            bind_types.len(),
            bindings.len(),
        );
        let mut node = self.scheme.node(label, &self.shaders[shader.0].pipeline);
        for (i, binding) in bindings.iter().enumerate() {
            node = if let GpuBinding::Present(lease, present_access) = binding {
                node.with_present_access(lease, *present_access)
            } else {
                let access = bind_type_to_node_access(bind_types[i]);
                match binding {
                    GpuBinding::Buf(b) => node.with_parcel(*b, access),
                    GpuBinding::Parcel(p) => node.with_parcel(*p, access),
                    GpuBinding::Tex(t) => node.with_parcel(*t, access),
                    GpuBinding::Sampler(s) => node.with_parcel(*s, access),
                    GpuBinding::Present(..) => unreachable!(),
                }
            };
        }
        node.dispatch_shape_parcel(shape).expect("dispatch_shape_parcel failed");
    }

    /// Stub for debug-layer draw commands (not yet implemented in Goldy).
    #[cfg(feature = "debug_layers")]
    #[allow(dead_code, reason = "parity stub; scheme debug renderer not wired yet")]
    pub fn draw(&mut self, params: crate::resource_proxy::DrawParams) {
        if let Some(vb) = params.vertex_buffer {
            self.defer_owned_buffer(vb);
        }
        for b in params.resources {
            self.defer_owned_buffer(b);
        }
        if let Some(tex) = params.target {
            self.defer_texture(tex);
        }
    }

    #[cfg(feature = "debug_layers")]
    #[allow(dead_code, reason = "parity stub; scheme debug renderer not wired yet")]
    pub(crate) fn defer_owned_buffer(&mut self, buf: Buffer) {
        self.deferred_owned_buffers.push(buf);
    }

    /// Finish dispatch: flush the final graph and register a frame slot with
    /// the orchestrator.
    ///
    /// Returns the submit timeline and an optional surface frame awaiting present.
    ///
    /// Surface paths submit through [`goldy::SurfaceExchange`] and [`goldy::Transaction`]
    /// so the timeline is valid for cache stamping before present.
    /// render-to-texture paths call `finish` with `deferred_present: false`.
    pub(crate) fn finish<F>(
        mut self,
        deferred_present: bool,
        pre_acquire: F,
        early_present: Option<goldy::AcquiredPresent>,
    ) -> Result<FrameFinishOutcome>
    where
        F: FnOnce() -> Result<()>,
    {
        // Keep `finished` false until success so Drop aborts the orchestrator frame
        // if upload/submit/end_frame fails (otherwise begin_frame stays stuck open).
        self.persistent.deferred_textures_cap_hint = self.deferred_textures.capacity();

        let deferred_textures = mem::take(&mut self.deferred_textures);
        #[cfg(feature = "debug_layers")]
        {
            for buf in mem::take(&mut self.deferred_owned_buffers) {
                self.context.return_transient_buffer(buf);
            }
        }

        let frame_handle = self.frame_handle;

        if !self.metal_fused_upload {
            let _tz = goldy::tracy_zone!("ekrano.finish.upload_submit");
            self.upload.submit().map_err(|e| Error::Shader(e.to_string()))?;
        }

        // On the fused Metal path the upload blits share the worker scheme's single
        // command buffer. Run pre_acquire immediately before that submit so the
        // upload is not committed ahead of the drawable/present barrier.
        {
            let _tz = goldy::tracy_zone!("ekrano.finish.pre_acquire");
            pre_acquire()?;
        }

        let submission = {
            let _tz = goldy::tracy_zone!("ekrano.finish.worker_submit");
            match early_present {
                Some(claim) => self
                    .scheme
                    .submit_with_acquired_presents(vec![claim])
                    .map_err(|e| Error::Shader(e.to_string()))?,
                None => self.scheme.submit().map_err(|e| Error::Shader(e.to_string()))?,
            }
        };
        {
            let _tz = goldy::tracy_zone!("ekrano.finish.orchestrator");
            if self.nonblocking_reuse {
                // Ordering is enforced by reuse epochs / deferred host writes / present easement.
                self.frame_pipeline
                    .end_frame_externally_ordered(frame_handle)
                    .map_err(|e| Error::Shader(e.to_string()))?;
            } else if deferred_present {
                self.frame_pipeline
                    .end_frame_for_present(frame_handle, &submission)
                    .map_err(|e| Error::Shader(e.to_string()))?;
            } else {
                self.frame_pipeline
                    .end_frame_standalone(frame_handle, &submission)
                    .map_err(|e| Error::Shader(e.to_string()))?;
            }
        }
        self.finished = true;
        Ok(FrameFinishOutcome {
            deferred_textures,
            scheme_submission: Some(submission),
        })
    }
}

impl Drop for SchemeRecorder<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.frame_pipeline.abort_frame(self.frame_handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheme_gpu_resources::PipelineResources;
    use crate::{RenderParams, Scene};
    use ekrano_encoding::{RenderConfig, Resolver};
    use goldy::{FrameOrchestrator, Scheme};

    /// Regression: `PipelineResources::prepare` must honour the requested
    /// `out_image_format` and must not hard-code `Rgba8Unorm`.  The same
    /// channel-swap regression (velato tiger turning blue) can occur if
    /// `copy_texture_to_present` copies an RGBA `out_image` into a BGRA
    /// present lease.
    ///
    /// Formats unsupported by the active backend (e.g. `Bgra8Unorm` on CUDA) are
    /// skipped — the regression is about honouring the request, not inventing
    /// backend texture formats.
    #[test]
    fn prepare_out_image_format_matches_requested() {
        let Some((gpu, mut persistent)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let scene = Scene::new();
        let encoding = scene.encoding();

        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        let caps = gpu.capabilities();
        let formats: Vec<TextureFormat> = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm]
            .into_iter()
            .filter(|f| caps.supported_render_target_formats.contains(f))
            .collect();
        assert!(
            !formats.is_empty(),
            "backend must support Bgra8Unorm and/or Rgba8Unorm to exercise out_image format selection"
        );

        for &expected_format in &formats {
            let mut resolver = Resolver::new();
            let mut packed = Vec::new();
            let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
            let config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);

            let ctx = gpu.create_context().expect("context");
            let mut worker = Scheme::new(&ctx);
            let mut upload = Scheme::new(&ctx);
            let mut frame_pipeline = FrameOrchestrator::new(&ctx, FRAME_PIPELINE_DEPTH);
            let frame_handle = frame_pipeline.begin_frame().expect("begin_frame");
            let pipeline = {
                let mut recorder = SchemeRecorder::new(
                    &gpu,
                    &ctx,
                    &mut worker,
                    &mut upload,
                    true,
                    false,
                    false,
                    &mut frame_pipeline,
                    frame_handle,
                    &mut persistent,
                    &[],
                );
                PipelineResources::prepare(
                    &mut recorder,
                    encoding.coverage_mask.as_ref(),
                    packed,
                    ramps,
                    images,
                    &params,
                    &config,
                    expected_format,
                    false,
                )
                .unwrap_or_else(|e| panic!("PipelineResources::prepare({expected_format:?}) failed: {e}"))
            };

            assert_eq!(
                pipeline
                    .out_image
                    .as_ref()
                    .expect("headless prepare allocates out_image")
                    .format(),
                expected_format,
                "out_image must use the requested format {expected_format:?}; \
                 using Rgba8Unorm unconditionally would cause copy_texture_to_present \
                 to swap R and B when copying to a Bgra8Unorm present lease"
            );
        }
    }

    /// Worker scheme records once and resubmits on subsequent frames with stable topology.
    ///
    /// Headless `render_to_buffer` uses a topology-invisible texture withdraw on a separate
    /// readback scheme, so the worker is not dirtied by a foreign reader and records once.
    /// The upload scheme may still record twice from cross-scheme topology discovery:
    /// upload registers writer edges first, then the worker registers reader edges on shared
    /// parcels and Goldy dirties the upload scheme. A second upload record on frame 2 is
    /// correct today but an optimization target in Goldy (narrower foreign-scheme
    /// invalidation / partition-local retention).
    #[test]
    fn worker_scheme_retains_topology_across_frames() {
        // Goldy topology dirty + retention records require CB replay; ignore
        // GOLDY_DISABLE_CB_REUSE=1 from the developer shell.
        let _cb = goldy::test_support::CbReuseOverride::force_enabled();
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 24.0),
        );

        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        const FRAMES: u32 = 4;
        for _ in 0..FRAMES {
            renderer.render_to_buffer(&scene, &params).expect("render_to_buffer");
        }

        let stats = renderer.worker_replay_stats();
        assert_eq!(
            stats.records, 1,
            "render_to_buffer: topology-invisible withdraw does not dirty the worker"
        );
        assert_eq!(
            stats.topology_records, 0,
            "texture withdraw must not register a topology-visible foreign reader on out_image"
        );
        let upload_stats = renderer.upload_replay_stats();
        if renderer.metal_fused_upload() {
            assert_eq!(
                upload_stats.records, stats.records,
                "fused Metal path: upload and worker share one scheme"
            );
            assert_eq!(
                upload_stats.topology_records, stats.topology_records,
                "fused Metal path: upload and worker share one scheme"
            );
        } else {
            assert_eq!(
                upload_stats.records, 2,
                "bootstrap record + one topology-induced invalidation when worker registers \
                 reader edges on shared parcels (optimization target: should plateau at 1)"
            );
            assert_eq!(
                upload_stats.topology_records, 1,
                "foreign worker reader on shared parcels dirties upload topology once"
            );
        }
    }

    /// Resolution change invalidates worker retention and triggers exactly one re-record.
    #[test]
    fn worker_scheme_rerecords_on_topology_change() {
        let _cb = goldy::test_support::CbReuseOverride::force_enabled();
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 24.0),
        );

        let mut params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        for _ in 0..2 {
            renderer.render_to_buffer(&scene, &params).expect("render_to_buffer");
        }
        assert_eq!(renderer.worker_record_epochs(), 1);

        params.width = 128;
        params.height = 128;
        for _ in 0..2 {
            renderer.render_to_buffer(&scene, &params).expect("render_to_buffer");
        }
        assert_eq!(
            renderer.worker_record_epochs(),
            2,
            "resolution change must trigger exactly one additional worker record"
        );
        assert_eq!(
            renderer.worker_replay_stats().records,
            1,
            "render_to_buffer at new resolution: worker records once (withdraw is topology-invisible)"
        );
        assert_eq!(
            renderer.worker_replay_stats().topology_records,
            0,
            "texture withdraw must not dirty worker topology after resolution change"
        );
    }

    /// Without per-frame readback, the worker records once and resubmits.
    #[test]
    fn worker_scheme_render_to_texture_records_once() {
        let _cb = goldy::test_support::CbReuseOverride::force_enabled();
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 24.0),
        );

        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        let texture = {
            use goldy::RetainedPool;
            use std::sync::Arc;
            RetainedPool::new(Arc::new(gpu.clone()))
                .acquire_texture(
                    params.width,
                    params.height,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Direct,
                    TextureFlags::COPY_DST,
                    None,
                )
                .expect("output texture")
        };

        const FRAMES: u32 = 4;
        for _ in 0..FRAMES {
            renderer
                .render_to_texture(&scene, &texture, &params)
                .expect("render_to_texture");
        }

        let stats = renderer.worker_replay_stats();
        assert_eq!(renderer.worker_record_epochs(), 1);
        if renderer.metal_fused_upload() {
            assert!(
                stats.records >= 1,
                "fused Metal path: worker scheme records at least once; upload staging may \
                 trigger an additional partition record without ekrano re-recording coarse/fine"
            );
        } else {
            assert_eq!(
                stats.records, 1,
                "render_to_texture has no foreign topology-visible reader on out_image"
            );
        }
        assert_eq!(stats.topology_records, 0);
    }

    /// Upload scheme re-records exactly once when `scene_bucket` grows, and the worker also
    /// re-records because the scene buffer `ResourceHandle` changed.
    ///
    /// This is a regression guard for the `scene_bucket` gap in the original `worker_stale`
    /// predicate: the scene buffer is bound by handle in the worker's recorded dispatches,
    /// so a bucket change (new allocation) must invalidate the worker too.
    #[test]
    fn worker_and_upload_rerecord_on_scene_bucket_growth() {
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");

        let mut small_scene = Scene::new();
        small_scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 24.0),
        );

        // Build a scene whose packed bytes land in the next power-of-two bucket
        // above the small scene.  We add enough paths to reliably cross a bucket boundary
        // without pinning exact byte counts (the assertion is on record epochs, not bytes).
        let mut large_scene = Scene::new();
        for i in 0..200 {
            let r = (i % 256) as u8;
            large_scene.fill(
                peniko::Fill::NonZero,
                peniko::kurbo::Affine::IDENTITY,
                peniko::Color::from_rgb8(r, 100, 50),
                None,
                &peniko::kurbo::Circle::new((32.0, 32.0), 8.0),
            );
        }

        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        // Warm up with the small scene — establishes an initial bucket.
        for _ in 0..2 {
            renderer
                .render_to_buffer(&small_scene, &params)
                .expect("render_to_buffer small");
        }
        let worker_epochs_after_small = renderer.worker_record_epochs();
        let upload_epochs_after_small = renderer.upload_record_epochs();
        assert_eq!(worker_epochs_after_small, 1);
        assert_eq!(upload_epochs_after_small, 1);

        // Switch to a much larger scene.  The scene_bucket should grow and force both
        // the upload scheme (new staging copy topology) and the worker scheme (new scene
        // buffer handle) to re-record.
        renderer
            .render_to_buffer(&large_scene, &params)
            .expect("render_to_buffer large");

        assert!(
            renderer.upload_record_epochs() > upload_epochs_after_small,
            "upload scheme must re-record when scene byte bucket grows"
        );
        assert!(
            renderer.worker_record_epochs() > worker_epochs_after_small,
            "worker scheme must re-record when scene buffer is reallocated (bucket grew)"
        );

        // Stabilise: two more large frames must not trigger additional records.
        let worker_epochs_stable = renderer.worker_record_epochs();
        let upload_epochs_stable = renderer.upload_record_epochs();
        for _ in 0..2 {
            renderer
                .render_to_buffer(&large_scene, &params)
                .expect("render_to_buffer large stable");
        }
        assert_eq!(
            renderer.worker_record_epochs(),
            worker_epochs_stable,
            "worker must not re-record on repeated large-scene frames"
        );
        assert_eq!(
            renderer.upload_record_epochs(),
            upload_epochs_stable,
            "upload must not re-record on repeated large-scene frames"
        );
    }

    /// Upload scheme is stable when only worker topology changes (AA mode switch).
    ///
    /// A topology-only change (AA mode) forces a worker re-record but the upload key
    /// (scene bucket + atlas dims) is unchanged, so the upload scheme must NOT re-record.
    #[test]
    fn upload_scheme_stable_when_only_worker_topology_changes() {
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 24.0),
        );

        let mut params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        for _ in 0..2 {
            renderer.render_to_buffer(&scene, &params).expect("render_to_buffer");
        }
        let upload_epochs_after_area = renderer.upload_record_epochs();
        let worker_epochs_after_area = renderer.worker_record_epochs();
        assert_eq!(upload_epochs_after_area, 1);
        assert_eq!(worker_epochs_after_area, 1);

        // Switch AA — this changes WorkerTopology.aa but not the UploadKey.
        params.antialiasing_method = crate::AaConfig::Msaa16;
        renderer
            .render_to_buffer(&scene, &params)
            .expect("render_to_buffer with msaa16");

        if renderer.metal_fused_upload() {
            assert_eq!(
                renderer.upload_record_epochs(),
                upload_epochs_after_area + 1,
                "fused Metal path: worker replacement re-attaches upload topology"
            );
        } else {
            assert_eq!(
                renderer.upload_record_epochs(),
                upload_epochs_after_area,
                "upload scheme must NOT re-record when only AA mode changes"
            );
        }
        assert_eq!(
            renderer.worker_record_epochs(),
            worker_epochs_after_area + 1,
            "worker scheme must re-record on AA mode change"
        );
    }

    /// Image atlas region layout can change while atlas dimensions and image count stay
    /// fixed (default atlas is 1024²). That dirties `UploadKey.image_regions` without
    /// dirtying `WorkerTopology`.
    ///
    /// On the Metal fused path, upload copy nodes live on the worker scheme — without
    /// forcing a worker replacement, new region deposits would be appended while old
    /// deposit ids remain referenced but unwritten, poisoning submit permanently.
    #[test]
    fn fused_worker_rerecords_on_image_region_layout_change() {
        let _cb = goldy::test_support::CbReuseOverride::force_enabled();
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");

        fn solid_image(width: u32, height: u32, rgba: [u8; 4]) -> peniko::ImageBrush {
            let data = rgba.repeat((width * height) as usize);
            peniko::ImageBrush {
                image: peniko::ImageData {
                    data: data.into(),
                    format: peniko::ImageFormat::Rgba8,
                    width,
                    height,
                    alpha_type: peniko::ImageAlphaType::Alpha,
                },
                sampler: peniko::ImageSampler {
                    quality: peniko::ImageQuality::Low,
                    ..Default::default()
                },
            }
        }

        let mut scene_a = Scene::new();
        scene_a.draw_image(&solid_image(8, 8, [200, 100, 50, 255]), peniko::kurbo::Affine::IDENTITY);

        let mut scene_b = Scene::new();
        // Same atlas size (1024²) and image_count (1); different region (8×8 → 16×16).
        scene_b.draw_image(
            &solid_image(16, 16, [50, 100, 200, 255]),
            peniko::kurbo::Affine::IDENTITY,
        );

        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        for _ in 0..2 {
            renderer
                .render_to_buffer(&scene_a, &params)
                .expect("render_to_buffer scene_a");
        }
        let worker_epochs = renderer.worker_record_epochs();
        let upload_epochs = renderer.upload_record_epochs();
        assert_eq!(worker_epochs, 1);
        assert_eq!(upload_epochs, 1);

        // Must succeed (not poison the retained worker) and re-record upload topology.
        renderer
            .render_to_buffer(&scene_b, &params)
            .expect("render_to_buffer after image region layout change");

        assert!(
            renderer.upload_record_epochs() > upload_epochs,
            "upload topology must re-record when image region layout changes at fixed atlas dims"
        );
        if renderer.metal_fused_upload() {
            assert!(
                renderer.worker_record_epochs() > worker_epochs,
                "fused Metal path: worker must be replaced when upload topology changes"
            );
        } else {
            assert_eq!(
                renderer.worker_record_epochs(),
                worker_epochs,
                "non-fused path: image region layout is upload-only; worker topology stays fresh"
            );
        }

        // Stabilise: further frames with scene_b must not keep failing or re-recording.
        let worker_stable = renderer.worker_record_epochs();
        let upload_stable = renderer.upload_record_epochs();
        for _ in 0..2 {
            renderer
                .render_to_buffer(&scene_b, &params)
                .expect("render_to_buffer scene_b stable");
        }
        assert_eq!(
            renderer.upload_record_epochs(),
            upload_stable,
            "upload must not re-record on repeated frames with stable region layout"
        );
        assert_eq!(
            renderer.worker_record_epochs(),
            worker_stable,
            "worker must not re-record on repeated frames with stable region layout"
        );
    }

    #[test]
    fn non_metal_swapchain_uses_out_image_copy_path() {
        // MockBackend reports Vulkan; direct present is Metal-only.
        let device = goldy::test_support::mock_device();
        assert_ne!(
            device.backend_type(),
            BackendType::Metal,
            "mock backend must not report Metal"
        );
        let (_ctx, surface) = goldy::test_support::mock_surface_exchange(&device);
        let mut renderer = SchemeRenderer::new(&device).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(40, 80, 120),
            None,
            &peniko::kurbo::Rect::new(0.0, 0.0, 64.0, 64.0),
        );
        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };
        let prepared = renderer.prepare(&scene, &params).expect("prepare");
        let (_stats, _token) = renderer
            .submit_to_swapchain(prepared, &surface)
            .expect("submit_to_swapchain");
        assert!(
            renderer.cached_scheme_has_out_image(),
            "non-Metal swapchain path must keep intermediate out_image"
        );
        assert!(
            !renderer.metal_fused_upload(),
            "mock device must not enable Metal fused upload"
        );
    }

    #[test]
    fn swapchain_retains_worker_and_present_transaction() {
        let _cb = goldy::test_support::CbReuseOverride::force_enabled();
        let device = goldy::test_support::mock_device();
        let (_ctx, surface) = goldy::test_support::mock_surface_exchange(&device);
        let mut renderer = SchemeRenderer::new(&device).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(40, 80, 120),
            None,
            &peniko::kurbo::Rect::new(0.0, 0.0, 64.0, 64.0),
        );
        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };
        for _ in 0..2 {
            let prepared = renderer.prepare(&scene, &params).expect("prepare");
            renderer
                .submit_to_swapchain(prepared, &surface)
                .expect("submit_to_swapchain");
        }
        assert_eq!(
            renderer.worker_record_epochs(),
            1,
            "swapchain path must retain worker topology across frames"
        );
    }

    #[test]
    fn render_to_buffer_allocates_out_image() {
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 24.0),
        );
        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };
        renderer.render_to_buffer(&scene, &params).expect("render_to_buffer");
        assert!(
            renderer.cached_scheme_has_out_image(),
            "headless readback path must retain out_image"
        );
        assert_eq!(
            renderer.cached_scheme_out_image_format(),
            Some(TextureFormat::Rgba8Unorm),
            "headless out_image is Rgba8Unorm (including CUDA typed-UAV pack path)"
        );
    }

    /// Filter frames re-record each submit (scratches are one-shot). Removing filters
    /// must still trim the sticky filter-uniform cache so retained deeds are released.
    #[test]
    fn filter_uniform_cache_shrinks_when_filters_removed() {
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };
        let texture = {
            use goldy::RetainedPool;
            use std::sync::Arc;
            RetainedPool::new(Arc::new(gpu.clone()))
                .acquire_texture(
                    params.width,
                    params.height,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Direct,
                    TextureFlags::COPY_DST,
                    None,
                )
                .expect("tex")
        };

        let mut filtered = Scene::new();
        let blur = ekrano_encoding::Filter(ekrano_encoding::FilterPrimitive::GaussianBlur {
            std_dev: 2.0,
            edge_mode: ekrano_encoding::FilterEdgeMode::Duplicate,
        });
        filtered.push_filter_layer(
            blur,
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            &peniko::kurbo::Rect::new(0.0, 0.0, 64.0, 64.0),
        );
        filtered.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 16.0),
        );
        filtered.pop_layer();

        for _ in 0..2 {
            renderer
                .render_to_texture(&filtered, &texture, &params)
                .expect("filtered render");
        }
        assert!(
            renderer.filter_uniform_cache_len() > 0,
            "filtered scene must populate filter-uniform cache"
        );

        let mut plain = Scene::new();
        plain.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(40, 80, 120),
            None,
            &peniko::kurbo::Rect::new(0.0, 0.0, 32.0, 32.0),
        );
        renderer
            .render_to_texture(&plain, &texture, &params)
            .expect("plain render");
        assert_eq!(
            renderer.filter_uniform_cache_len(),
            0,
            "removing filters must trim retained filter-uniform cache to zero"
        );
    }

    #[test]
    fn mask_atlas_replacement_updates_dims_and_releases_old() {
        let Some((gpu, _)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
            return;
        };

        let mut renderer = SchemeRenderer::new(&gpu).expect("SchemeRenderer::new");
        let texture = {
            use goldy::RetainedPool;
            use std::sync::Arc;
            RetainedPool::new(Arc::new(gpu.clone()))
                .acquire_texture(
                    64,
                    64,
                    TextureFormat::Rgba8Unorm,
                    TextureKind::Direct,
                    TextureFlags::COPY_DST,
                    None,
                )
                .expect("tex")
        };
        let params = RenderParams {
            base_color: peniko::color::palette::css::BLACK,
            width: 64,
            height: 64,
            antialiasing_method: crate::AaConfig::Area,
            robust: false,
        };

        // No mask → 1×1 sentinel atlas.
        let mut scene = Scene::new();
        scene.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(200, 100, 50),
            None,
            &peniko::kurbo::Circle::new((32.0, 32.0), 16.0),
        );
        renderer.render_to_texture(&scene, &texture, &params).expect("no-mask");
        assert_eq!(renderer.cached_mask_atlas_dims(), Some((1, 1)));
        let tex_bytes_1x1 = renderer.resource_pool_stats().retained_pool_texture_bytes;

        // Full-frame coverage mask → 64×64 atlas; old 1×1 deed released from retained pool.
        let mask_data = vec![255_u8; 64 * 64];
        let mask = ekrano_encoding::CoverageMask::new(64, 64, mask_data).expect("mask");
        scene.set_coverage_mask(mask);
        renderer
            .render_to_texture(&scene, &texture, &params)
            .expect("with-mask");
        assert_eq!(renderer.cached_mask_atlas_dims(), Some((64, 64)));
        let tex_bytes_64 = renderer.resource_pool_stats().retained_pool_texture_bytes;
        assert!(
            tex_bytes_64 > tex_bytes_1x1,
            "64×64 mask atlas must increase retained texture bytes"
        );

        // Back to no mask → 1×1 again; retained texture accounting must not keep both atlases.
        let mut plain = Scene::new();
        plain.fill(
            peniko::Fill::NonZero,
            peniko::kurbo::Affine::IDENTITY,
            peniko::Color::from_rgb8(40, 80, 120),
            None,
            &peniko::kurbo::Rect::new(0.0, 0.0, 32.0, 32.0),
        );
        renderer
            .render_to_texture(&plain, &texture, &params)
            .expect("mask cleared");
        renderer.flush_deferred_deletions();
        assert_eq!(renderer.cached_mask_atlas_dims(), Some((1, 1)));
        let tex_bytes_back = renderer.resource_pool_stats().retained_pool_texture_bytes;
        assert!(
            tex_bytes_back < tex_bytes_64,
            "releasing 64×64 mask atlas must drop retained texture bytes (was {tex_bytes_64}, now {tex_bytes_back})"
        );
    }
}
