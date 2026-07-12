// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Retained-`Scheme`-based renderer — the Scheme backend.
//!
//! This file is intentionally self-contained: it contains only `Scheme` code.
//! The companion [`crate::graph_renderer`] module contains the `TaskGraph`-based
//! renderer; the two share infrastructure types from [`crate::goldy_renderer`]
//! but have no rendering logic in common.
//!
//! # Surface rendering
//!
//! Unlike the Classic backend which uses [`goldy::Surface`] + `TaskGraph::copy_texture_to_swapchain`,
//! `SchemeRenderer` uses the scheme-native present mechanism:
//!
//! 1. [`goldy::SwapchainPool`] is passed by the caller each frame.
//! 2. A [`goldy::PresentLease`] is acquired from the pool.
//! 3. After all compute nodes, [`goldy::Scheme::copy_texture_to_present`] records a copy
//!    from `out_image` to the present lease.
//! 4. [`goldy::Scheme::grant_present`] records the present easement (once per frame in
//!    Phase 2; retained in Phase 3).
//! 5. [`goldy::Scheme::submit`] submits all partitions, including the present-touching one.
//! 6. [`goldy::PresentGrant::consume`] performs scanout.
//!
//! # Phase 2 notes
//!
//! In this naive Phase 2 port the scheme is rebuilt from scratch every frame (same
//! node topology as the `TaskGraph` path). Phase 3 will retain the topology across
//! frames and only re-record when dirty.

use std::mem;
use std::mem::size_of;
use std::sync::Arc;

use goldy::task_graph::NodeAccess;
use goldy::types::{TextureFlags, TextureFormat, TextureKind};
use goldy::{
    BudgetPolicy, Buffer, ComputePipeline, Context, Device, FrameHandle, FrameOrchestrator, Grant, Scheme,
    ShaderModule, Signal, TaskGraph, Texture,
};

use crate::{
    Error, RenderParams, Result, Scene,
    goldy_renderer::{
        AllocatorStats, CacheScheduleOutcome, FRAME_COUNTER, FRAME_PIPELINE_DEPTH, FrameFinishOutcome, FrameStats,
        GoldyShader, MAX_BUMP_RETRIES, PersistentState, PreparedFrame, ResourcePoolStats, defer_frame_gpu_resources,
        env_robust_override, find_empty_cache_slot, sanitize_bump,
    },
    resource_proxy::{BindType, ShaderId},
    scheme_gpu_resources::{
        GpuBinding, acquire_texture_rgba, bind_type_to_node_access, record_upload_bytes, record_upload_bytes_owned,
    },
    scheme_render::Render,
    shaders::{self, FullShaders},
};
use ekrano_encoding::{BumpAllocators, Images, Layout, Ramps, RenderConfig, Resolver};

// -----------------------------------------------------------------------
// SchemeRenderer — Scheme-based renderer
// -----------------------------------------------------------------------

/// Goldy-based 2D renderer using the retained-[`Scheme`] backend.
///
/// All rendering is done via Goldy's [`Scheme`] command recording.
/// Surface presentation uses [`goldy::SwapchainPool`] + [`goldy::PresentGrant`]
/// (scheme-native present mechanism). For the `TaskGraph` path see
/// [`crate::graph_renderer::GraphRenderer`].
///
/// This struct is intentionally isolated: it contains no `TaskGraph` references and
/// no graph-path branches.
pub struct SchemeRenderer {
    device: Device,
    context: Context,
    shaders: FullShaders,
    resolver: Resolver,
    engine_shaders: Vec<GoldyShader>,
    /// Cross-frame GPU resources: pools, texture cache, bump readback.
    persistent: PersistentState,
    /// Pipelined frame scheduling: depth enforcement and timeline tracking.
    frame_pipeline: FrameOrchestrator<()>,
    /// Persistent bump estimates: running max across frames.
    persistent_bump: Option<BumpAllocators>,
    /// Frame counter for rate-limiting housekeeping operations.
    cleanup_frame_counter: u64,
    /// Retained scheme.
    ///
    /// Rebuilt from scratch each frame in Phase 2; Phase 3 retains topology
    /// across frames for the zero-rerecord path.
    scheme: Scheme,
    /// Separate scheme for `render_to_buffer` texture→CPU readback.
    readback: Scheme,
}

impl SchemeRenderer {
    /// Create a new Scheme renderer for the given device.
    pub fn new(device: &Device) -> Result<Self> {
        let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new");

        let device = device.clone();

        device
            .set_allocation_policy(Arc::new(BudgetPolicy::new()))
            .map_err(|e| Error::Gpu(e.to_string()))?;

        let context = device.create_context().map_err(|e| Error::Gpu(e.to_string()))?;
        let frame_pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new.frame_orchestrator");
            FrameOrchestrator::new(&context, FRAME_PIPELINE_DEPTH)
        };
        let scheme = Scheme::new(&context);
        let readback = Scheme::new(&context);
        let mut renderer = Self {
            device: device.clone(),
            context,
            shaders: FullShaders::empty(),
            resolver: Resolver::new(),
            engine_shaders: Vec::new(),
            persistent: PersistentState::new(&device),
            frame_pipeline,
            persistent_bump: None,
            cleanup_frame_counter: 0,
            scheme,
            readback,
        };
        let shaders = {
            let _tz = goldy::tracy_zone!("ekrano.SchemeRenderer::new.compile_shaders");
            shaders::goldy_full_shaders_scheme(&mut renderer)?
        };
        renderer.shaders = shaders;
        let pending_returns = renderer.persistent.pending_owned_returns.clone();
        renderer.persistent.pool.set_pending_returns(pending_returns);
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
                Signal::BoundaryCrossed { epoch } => {
                    self.persistent.drain_pending_returns();
                    self.frame_pipeline.note_presented(epoch);
                }
                Signal::Oversubscribed { .. } => {
                    if let Some(oldest) = self.context.peek_oldest_in_flight()
                        && self.context.wait_until(oldest).is_err()
                    {
                        break;
                    }
                    self.context.flush_deferred_deletions();
                    self.persistent.drain_pending_returns();
                }
                Signal::SwapchainReturned { image_index } => {
                    self.persistent.mark_rt_slot_returned(&self.context, image_index);
                }
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
    /// `pool` must have been created with the same [`Context`] as this renderer.
    /// Each call acquires a drawable from the pool, submits the scheme, and presents
    /// via [`goldy::PresentGrant::consume`].
    pub fn render_to_swapchain(
        &mut self,
        scene: &Scene,
        pool: &goldy::SwapchainPool,
        params: &RenderParams,
    ) -> Result<FrameStats> {
        let _tz = goldy::tracy_zone!("ekrano.render_to_swapchain");
        let prepared = self.prepare(scene, params)?;
        self.submit_to_swapchain(prepared, pool)
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

    /// Phase 2: record GPU work, present, and return frame stats.
    ///
    /// Uses the scheme-native present mechanism: records a `copy_texture_to_present`
    /// node followed by `grant_present`, submits the scheme, and calls
    /// [`goldy::PresentGrant::consume`] to perform scanout.
    pub fn submit_to_swapchain(&mut self, prepared: PreparedFrame, pool: &goldy::SwapchainPool) -> Result<FrameStats> {
        let _tz = goldy::tracy_zone!("ekrano.submit_to_swapchain");
        self.poll_and_reclaim();
        let (stats, _) = self.run_frame_from_prepared(prepared, None, Some(pool))?;
        Ok(stats)
    }

    /// Query frame-scheduling state for diagnostics or test assertions.
    pub fn allocator_stats(&self) -> AllocatorStats {
        AllocatorStats {
            cleanup_ring_depth: self.frame_pipeline.pending_frames(),
        }
    }

    /// GPU device handle shared by this renderer.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Query the resource pool's current state for diagnostics or test assertions.
    pub fn resource_pool_stats(&self) -> ResourcePoolStats {
        ResourcePoolStats {
            total_pooled_buffers: self.persistent.pool.total_pooled_buffers(),
            distinct_keys: self.persistent.pool.distinct_keys(),
            retained_pool_buffer_bytes: self.persistent.retained_pool.bytes_by_kind().buffer,
        }
    }

    /// `true` if the submission context still holds unreclaimed deferred payloads.
    pub fn has_deferred_payloads(&self) -> bool {
        self.context.has_deferred_payloads()
    }

    /// Pull-side reclamation: drain the submission context's deferred-deletion ring.
    pub fn flush_deferred_deletions(&self) {
        self.context.flush_deferred_deletions();
    }

    /// Query the render context's placement heap state for diagnostics / tests.
    pub fn placement_heap_stats(&self) -> Option<goldy::placement_heap::PlacementHeapStats> {
        self.context.placement_heap_stats()
    }

    /// Render a scene and return the pixel data as RGBA bytes (synchronous).
    pub fn render_to_buffer(&mut self, scene: &Scene, params: &RenderParams) -> Result<Vec<u8>> {
        let width = params.width;
        let height = params.height;
        let texture = self
            .persistent
            .tex_pool
            .acquire(
                &self.device,
                width,
                height,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .map_err(|e| Error::Gpu(e.to_string()))?;

        for _attempt in 0..=MAX_BUMP_RETRIES {
            self.render_to_texture(scene, &texture, params)?;
            self.frame_pipeline
                .drain_all(|_, _| Ok::<(), Error>(()))
                .map_err(|e| Error::Shader(e.to_string()))?;
            self.drain_ready_bump_readbacks()?;
            self.context.flush_deferred_deletions();

            match self.persistent.last_drained_bump() {
                Some(bump) if bump.failed != 0 => {
                    log::info!("Bump overflow in render_to_buffer (0x{:x}), retrying", bump.failed);
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

        let layout = texture.copy_layout();
        let host_buf = self
            .persistent
            .acquire_readback_host_buf(&self.context, layout.staging_bytes)
            .map_err(|e| Error::Readback(e.to_string()))?;
        self.readback
            .copy_texture(&texture, &host_buf)
            .map_err(|e| Error::Readback(e.to_string()))?;
        let submission = self.readback.submit().map_err(|e| Error::Readback(e.to_string()))?;
        submission
            .wait(&self.context)
            .map_err(|e| Error::Readback(e.to_string()))?;
        let mut padded = vec![0_u8; layout.staging_bytes as usize];
        host_buf
            .read_to_cpu(&self.device, &mut padded)
            .map_err(|e| Error::Readback(e.to_string()))?;
        let row_bytes = layout.tight_row_bytes() as usize;
        let pitch = layout.row_pitch as usize;
        let mut output = vec![0_u8; layout.logical_bytes as usize];
        for row in 0..layout.height as usize {
            let src_offset = layout.footprint_offset as usize + row * pitch;
            let dst_offset = row * row_bytes;
            output[dst_offset..dst_offset + row_bytes].copy_from_slice(&padded[src_offset..src_offset + row_bytes]);
        }
        self.persistent.store_readback_host_buf(host_buf, layout.staging_bytes);
        Ok(output)
    }

    fn drain_ready_bump_readbacks(&mut self) -> Result<()> {
        self.persistent.drain_ready_bump_readbacks(&self.device, &self.context)
    }

    // =======================================================================
    // Frame execution (private)
    // =======================================================================

    fn run_frame(
        &mut self,
        scene: &Scene,
        params: &RenderParams,
        output_texture: Option<&Texture>,
        pool: Option<&goldy::SwapchainPool>,
    ) -> Result<FrameStats> {
        let prepared = self.prepare(scene, params)?;
        let (stats, _) = self.run_frame_from_prepared(prepared, output_texture, pool)?;
        Ok(stats)
    }

    /// GPU submission path for both texture and swapchain rendering.
    ///
    /// When `pool` is `Some`, records `copy_texture_to_present` + `grant_present`
    /// into the scheme, submits, and consumes the present grant.
    fn run_frame_from_prepared(
        &mut self,
        prepared: PreparedFrame,
        output_texture: Option<&Texture>,
        pool: Option<&goldy::SwapchainPool>,
    ) -> Result<(FrameStats, ())> {
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

        self.persistent.drain_pending_returns();

        let out_image_format = pool.map(|p| p.format()).unwrap_or(TextureFormat::Rgba8Unorm);
        self.persistent.purge_render_target_cache_if_mismatch(
            &self.context,
            params.width,
            params.height,
            out_image_format,
        );

        let t_drain_start = Instant::now();

        let _tz_begin = goldy::tracy_zone!("ekrano.begin_frame");
        let frame_handle = self
            .frame_pipeline
            .begin_frame(|_, _| Ok::<(), Error>(()))
            .map_err(|e| Error::Shader(e.to_string()))?;
        self.drain_ready_bump_readbacks()?;
        self.cleanup_frame_counter = self.cleanup_frame_counter.wrapping_add(1);
        if self.cleanup_frame_counter.is_multiple_of(64) {
            self.persistent.pool.cap_pool_depth(12);
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
        // Reclaim owned buffers whose defer epoch just retired during the begin_frame wait.
        // Without this, prepare() pool-misses and allocates duplicates (scene+bump) while
        // the prior frame's buffers sit in pending_owned_returns.
        self.persistent.drain_pending_returns();
        let t_pool = t1.elapsed();

        let t2 = Instant::now();
        // Phase 2: rebuild the scheme from scratch each frame so recording starts fresh.
        // Phase 3 will retain the topology across frames for zero-rerecord paths.
        self.scheme = Scheme::new(&self.context);
        let mut recorder = SchemeRecorder::new(
            &self.device,
            &self.context,
            &mut self.scheme,
            &mut self.frame_pipeline,
            frame_handle,
            &mut self.persistent,
            &self.engine_shaders,
        );

        let mut pipeline = {
            let _tz = goldy::tracy_zone!("ekrano.prepare");
            let pipeline_result = recorder.prepare_pipeline_resources(
                coverage_mask.as_ref(),
                packed,
                ramps,
                images,
                &params,
                &config,
                out_image_format,
            );
            self.resolver = resolver;
            match pipeline_result {
                Ok(p) => p,
                Err(e) => {
                    drop(recorder);
                    return Err(e);
                }
            }
        };

        let mut render = Render::new();
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

        let t3 = Instant::now();
        {
            let _tz = goldy::tracy_zone!("ekrano.fine");

            render.record_fine(
                &layer_filter_effects,
                &self.shaders,
                &pipeline,
                output_texture,
                &mut recorder,
            );
            #[cfg(feature = "debug_layers")]
            let _ = render.take_captured_buffers();
            crate::scheme_render::record_filter_effects(
                &layer_filter_effects,
                &self.shaders,
                &mut recorder,
                &pipeline,
                output_texture,
            );
        }
        let t_fine_record = t3.elapsed();

        // Surface path: record copy_texture_to_present + grant_present before submit.
        let present_grant = if let Some(pool) = pool {
            let screen = pool.lease();
            recorder.copy_texture_to_present(&pipeline.out_image, &screen);
            Some(recorder.grant_present(&screen))
        } else {
            None
        };

        let t4 = Instant::now();
        let cache_outcome = recorder.schedule_pipeline_cleanup(pipeline, params.robust);
        let FrameFinishOutcome {
            timeline: frame_tv,
            surface_frame: _,
            bump_readback,
            deferred_textures,
            recyclable_owned,
            scheme_submission,
        } = {
            let _tz = goldy::tracy_zone!("ekrano.finish");
            recorder.finish()?
        };

        // Present using the scheme's native mechanism.
        if let (Some(grant), Some(submission)) = (present_grant, scheme_submission) {
            grant.consume(&submission).map_err(|e| Error::Shader(e.to_string()))?;
        }

        // On Metal, gate the orchestrator on the compute timeline to allow the
        // present partition to overlap the next frame's CPU recording.
        if self.device.backend_type() == goldy::BackendType::Metal {
            self.frame_pipeline.note_presented(frame_tv);
        }

        if let Some(i) = cache_outcome.cached_render_targets_slot {
            log::debug!(
                "[RT-CACHE] stamp slot={i} timeline={frame_tv} (prev={})",
                self.persistent.cached_rt_timelines[i],
            );
            self.persistent.cached_rt_timelines[i] = frame_tv;
        }
        if let Some(buf) = bump_readback {
            self.persistent.queue_bump_readback(frame_tv, buf);
        }
        defer_frame_gpu_resources(
            &self.context,
            &self.persistent,
            frame_tv,
            deferred_textures,
            recyclable_owned,
        );

        {
            let _tz = goldy::tracy_zone!("ekrano.run_frame.post_submit");
            self.context.flush_deferred_deletions();
            let t_submit = t4.elapsed();

            let frame_num = FRAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let label = if pool.is_some() { "swapchain" } else { "" };

            let ring_depth = self.frame_pipeline.pending_frames();
            let (transient_views, transient_textures) = self.context.transient_cache_counts();
            let rt_slots = self
                .persistent
                .cached_render_targets
                .iter()
                .filter(|s| s.is_some())
                .count();
            let pipe_slots = self.persistent.cached_pipeline.is_some() as usize;

            log::debug!(
                "[PERF] frame={} drain={:.2}ms resolve={:.2}ms pool={:.2}ms coarse_record={:.2}ms fine_record={:.2}ms submit={:.2}ms total={:.2}ms ring={} rt_slots={rt_slots} pipe_slots={pipe_slots} tv={} tt={} {label}",
                frame_num,
                t_drain.as_secs_f64() * 1000.0,
                t_resolve.as_secs_f64() * 1000.0,
                t_pool.as_secs_f64() * 1000.0,
                t_coarse.as_secs_f64() * 1000.0,
                t_fine_record.as_secs_f64() * 1000.0,
                t_submit.as_secs_f64() * 1000.0,
                frame_start.elapsed().as_secs_f64() * 1000.0,
                ring_depth,
                transient_views,
                transient_textures,
            );
        }

        Ok((stats, ()))
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
        _label: &'static str,
        slang_source: &str,
        bindings: &[BindType],
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: goldy::OptimizationLevel,
    ) -> Result<ShaderId> {
        let shader_module = {
            let _tz = goldy::tracy_zone!("ekrano.add_shader.slang", _label);
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
            let _tz = goldy::tracy_zone!("ekrano.add_shader.pipeline", _label);
            ComputePipeline::new(&self.device, &shader_module).map_err(|e| Error::Shader(format!("{:#}", e)))?
        };

        let id = ShaderId(self.engine_shaders.len());
        self.engine_shaders.push(GoldyShader {
            pipeline,
            bindings: bindings.to_vec(),
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
    pub(crate) scheme: &'a mut Scheme,
    frame_pipeline: &'a mut FrameOrchestrator<()>,
    frame_handle: FrameHandle,
    pub(crate) persistent: &'a mut PersistentState,
    pub(crate) shaders: &'a [GoldyShader],
    /// Set to `true` by `finish` or `abort`; the `Drop` impl aborts the open frame
    /// if the recorder is dropped without being properly completed (e.g. on a `?` return).
    finished: bool,
    /// The bump readback buffer for the current frame (`robust=true` only).
    /// Queued into `PersistentState::pending_bump_readback` after GPU submit.
    bump_buf_for_readback: Option<Buffer>,
    deferred_owned_buffers: Vec<(Buffer, &'static str)>,
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

    pub(crate) fn copy_texture_to_present(&mut self, src: &Texture, dst: &goldy::PresentLease) {
        self.scheme.copy_texture_to_present(src, dst);
    }

    pub(crate) fn grant_present(&mut self, dst: &goldy::PresentLease) -> goldy::PresentGrant {
        self.scheme.grant_present(dst)
    }

    pub(crate) fn new(
        device: &'a Device,
        context: &'a Context,
        scheme: &'a mut Scheme,
        frame_pipeline: &'a mut FrameOrchestrator<()>,
        frame_handle: FrameHandle,
        persistent: &'a mut PersistentState,
        shaders: &'a [GoldyShader],
    ) -> Self {
        // Read capacity hints before persistent is moved into Self.
        let owned_cap = persistent.deferred_owned_cap_hint;
        let tex_cap = persistent.deferred_textures_cap_hint;

        Self {
            device,
            context,
            scheme,
            frame_pipeline,
            frame_handle,
            persistent,
            shaders,
            bump_buf_for_readback: None,
            deferred_owned_buffers: Vec::with_capacity(owned_cap),
            deferred_textures: Vec::with_capacity(tex_cap),
            finished: false,
            filter_dispatch_slot: 0,
        }
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
        )
    }

    // NOTE: There is no explicit mid-frame flush. The backend is free to split
    // a single TaskGraph into multiple command buffers as an implementation
    // detail (e.g. coarse vs fine partitions). Callers should not assume or
    // rely on intra-frame submission boundaries.

    pub(crate) fn defer_texture(&mut self, tex: Texture) {
        self.deferred_textures.push(tex);
    }

    pub(crate) fn schedule_pipeline_cleanup(
        &mut self,
        pipeline: crate::scheme_gpu_resources::PipelineResources,
        bump_readback: bool,
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
        } = pipeline;

        self.defer_texture(gradient);
        self.defer_texture(image_atlas);
        self.defer_texture(mask_atlas);
        self.defer_owned_buffer(scene, "ekrano.scene");
        self.persistent.cached_config_uniform = Some((config_uniform_value, config));
        if let Some((wg_counts_gpu, indirect_buf)) = indirect {
            self.persistent.cached_scheme_indirect = Some((wg_counts_gpu, indirect_buf));
        }
        if bump_readback {
            self.bump_buf_for_readback = Some(bump);
        } else {
            self.defer_owned_buffer(bump, "ekrano.bump_buf");
        }
        let pipeline_cache = crate::graph_gpu_resources::CachedPipeline {
            stable: crate::graph_gpu_resources::StablePipelineBuffers {
                info_bin_data: stable.info_bin_data,
                tile: stable.tile,
                segments: stable.segments,
                ptcl: stable.ptcl,
                blend_spill: stable.blend_spill,
                lines: stable.lines,
                seg_counts: stable.seg_counts,
            },
            scratch: crate::graph_gpu_resources::ScratchPipelineBuffers {
                reduced: scratch.reduced,
                reduced2: scratch.reduced2,
                reduced_scan: scratch.reduced_scan,
                tagmonoid: scratch.tagmonoid,
                path_bbox: scratch.path_bbox,
                draw_reduced: scratch.draw_reduced,
                draw_monoid: scratch.draw_monoid,
                clip_inp: scratch.clip_inp,
                clip_el: scratch.clip_el,
                clip_bic: scratch.clip_bic,
                clip_bbox: scratch.clip_bbox,
                draw_bbox: scratch.draw_bbox,
                bin_header: scratch.bin_header,
                path: scratch.path,
            },
            buffer_sizes,
        };
        assert!(
            self.persistent.cached_pipeline.is_none(),
            "cached_pipeline must be empty at schedule (prepare should have taken it)"
        );
        self.persistent.cached_pipeline = Some(pipeline_cache);
        log::debug!("[PIPE-CACHE] schedule: cached");
        if let Some(i) = find_empty_cache_slot(&self.persistent.cached_render_targets) {
            self.persistent.cached_render_targets[i] = Some((out_image, filter_layers));
            outcome.cached_render_targets_slot = Some(i);
            log::debug!("[RT-CACHE] schedule: cached slot={i}");
        } else {
            self.defer_texture(out_image);
            for l in filter_layers {
                self.defer_texture(l);
            }
            log::debug!("[RT-CACHE] schedule: all slots full — deferred current frame RTs");
        }
        outcome
    }

    #[allow(dead_code, reason = "debug_layers uses GraphRecorder::upload, not SchemeRecorder")]
    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> Buffer {
        record_upload_bytes_owned(self, name, 1, data.into()).expect("upload failed")
    }

    pub fn upload_strided(&mut self, name: &'static str, element_stride: u32, data: impl Into<Vec<u8>>) -> Buffer {
        record_upload_bytes_owned(self, name, element_stride, data.into()).expect("upload_strided failed")
    }

    pub fn upload_typed<T: bytemuck::Pod>(&mut self, name: &'static str, data: &T) -> Buffer {
        record_upload_bytes(self, name, size_of::<T>() as u32, bytemuck::bytes_of(data)).expect("upload_typed failed")
    }

    pub fn dispatch(&mut self, shader: ShaderId, wg_size: (u32, u32, u32), bindings: &[GpuBinding<'_>]) {
        Self::record_dispatch(self.scheme, self.shaders, shader, wg_size, bindings, &[]);
    }

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

        let mut node = scheme.node("dispatch", &shaders[shader_id.0].pipeline);
        for (i, binding) in bindings.iter().enumerate() {
            let access = bind_types
                .get(i)
                .copied()
                .map(bind_type_to_node_access)
                .unwrap_or(NodeAccess::ReadWrite);
            node = match binding {
                GpuBinding::Buf(b) => node.with_parcel(*b, access),
                GpuBinding::Parcel(p) => node.with_parcel(*p, access),
                GpuBinding::Tex(t) => node.with_parcel(*t, access),
                GpuBinding::Sampler(s) => node.with_parcel(*s, access),
                GpuBinding::PersistentBuf(b) => node.with_parcel(*b, access),
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
        let mut node = self.scheme.node("dispatch_shape", &self.shaders[shader.0].pipeline);
        for (i, binding) in bindings.iter().enumerate() {
            let access = bind_types
                .get(i)
                .copied()
                .map(bind_type_to_node_access)
                .unwrap_or(NodeAccess::ReadWrite);
            node = match binding {
                GpuBinding::Buf(b) => node.with_parcel(*b, access),
                GpuBinding::Parcel(p) => node.with_parcel(*p, access),
                GpuBinding::Tex(t) => node.with_parcel(*t, access),
                GpuBinding::Sampler(s) => node.with_parcel(*s, access),
                GpuBinding::PersistentBuf(b) => node.with_parcel(*b, access),
            };
        }
        node.dispatch_shape(shape).expect("dispatch_shape failed");
    }

    /// Stub for debug-layer draw commands (not yet implemented in Goldy).
    #[cfg(feature = "debug_layers")]
    #[allow(dead_code, reason = "debug_layers uses GraphRecorder::draw, not SchemeRecorder")]
    pub fn draw(&mut self, params: crate::resource_proxy::DrawParams) {
        if let Some(vb) = params.vertex_buffer {
            self.defer_owned_buffer(vb, "ekrano.debug.vertex_buffer");
        }
        for b in params.resources {
            self.defer_owned_buffer(b, "ekrano.debug.resource");
        }
        if let Some(tex) = params.target {
            self.defer_texture(tex);
        }
    }

    pub(crate) fn defer_owned_buffer(&mut self, buf: Buffer, name: &'static str) {
        self.deferred_owned_buffers.push((buf, name));
    }

    /// Finish dispatch: flush the final graph and register a frame slot with
    /// the orchestrator.
    ///
    /// Returns the submit timeline and an optional surface frame awaiting present.
    ///
    /// Surface paths call [`goldy::Frame::submit_frame`] before returning so the
    /// timeline is valid for cache stamping before [`goldy::Frame::present`].
    pub(crate) fn finish(mut self) -> Result<FrameFinishOutcome> {
        self.finished = true;

        self.persistent.deferred_owned_cap_hint = self.deferred_owned_buffers.capacity();
        self.persistent.deferred_textures_cap_hint = self.deferred_textures.capacity();

        let deferred_textures = mem::take(&mut self.deferred_textures);
        let bump_readback = self.bump_buf_for_readback.take();
        let recyclable_owned = mem::take(&mut self.deferred_owned_buffers);

        let frame_handle = self.frame_handle;

        let submission = self.scheme.submit().map_err(|e| Error::Shader(e.to_string()))?;
        let scheme_tv = submission.timeline_value();
        let mut empty_graph = TaskGraph::new();
        let tv = self
            .frame_pipeline
            .end_frame_standalone(frame_handle, &mut empty_graph, Some(scheme_tv), ())
            .map_err(|e| Error::Shader(e.to_string()))?;
        Ok(FrameFinishOutcome {
            timeline: tv,
            surface_frame: None,
            bump_readback,
            deferred_textures,
            recyclable_owned,
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

    /// Scheme-backend counterpart of the Classic `prepare_out_image_format_matches_requested`
    /// test in [`crate::goldy_renderer::tests`].
    ///
    /// Verifies that `scheme_gpu_resources::PipelineResources::prepare` honours the
    /// `out_image_format` argument and does not hard-code `Rgba8Unorm`.  The same
    /// channel-swap regression (velato tiger turning blue) can occur on the Scheme
    /// path if `copy_texture_to_present` copies an RGBA `out_image` into a BGRA
    /// present lease.
    #[test]
    fn prepare_out_image_format_matches_requested() {
        let Some((device, mut persistent)) = crate::goldy_renderer::tests::make_device_and_persistent() else {
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

        for &expected_format in &[TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm] {
            let mut resolver = Resolver::new();
            let mut packed = Vec::new();
            let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
            let config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);

            let ctx = device.create_context().expect("context");
            let mut scheme = Scheme::new(&ctx);
            let mut frame_pipeline = FrameOrchestrator::new(&ctx, FRAME_PIPELINE_DEPTH);
            let frame_handle = frame_pipeline
                .begin_frame(|_, _| Ok::<(), Error>(()))
                .expect("begin_frame");
            let pipeline = {
                let mut recorder = SchemeRecorder::new(
                    &device,
                    &ctx,
                    &mut scheme,
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
                )
                .unwrap_or_else(|e| panic!("PipelineResources::prepare({expected_format:?}) failed: {e}"))
            };

            assert_eq!(
                pipeline.out_image.format(),
                expected_format,
                "out_image must use the requested format {expected_format:?}; \
                 using Rgba8Unorm unconditionally would cause copy_texture_to_present \
                 to swap R and B when copying to a Bgra8Unorm present lease"
            );
        }
    }
}
