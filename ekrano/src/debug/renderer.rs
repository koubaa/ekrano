// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Debug visualization renderer (Goldy port stub).
//!
//! The original wgpu-based render pipelines have been removed as part of the
//! wgpu -> Goldy migration. The `FrameRecorder::draw` method is currently a
//! no-op, so all draw calls emitted here are silently dropped.
//! The CPU-side validation logic (`validate_line_soup`) remains functional.

use super::DebugLayers;
use crate::{
    RenderParams,
    debug::validate::{LineEndpoint, validate_line_soup},
    goldy_renderer::FrameRecorder,
    render::CapturedBuffers,
    resource_proxy::{DrawParams, ShaderId},
};

use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use ekrano_encoding::BumpAllocators;
use peniko::color::{OpaqueColor, Srgb, palette};

/// CPU-side downloads of debug buffers.
///
/// Populate by calling `GoldyEngine::get_download` on the buffer proxies
/// from `CapturedBuffers` after the coarse pass completes.
pub struct DebugDownloads<'a> {
    /// Raw bytes of the downloaded line-soup buffer for CPU-side validation.
    pub lines: &'a [u8],
}

pub(crate) struct DebugRenderer {
    clear_tint: ShaderId,
    bboxes: ShaderId,
    linesoup: ShaderId,
    linesoup_points: ShaderId,
    unpaired_points: ShaderId,
}

impl DebugRenderer {
    /// Create a new debug renderer.
    ///
    /// Currently a no-op stub: `FrameRecorder::draw` is a no-op, so shader
    /// IDs are sentinels and no GPU resources are created here.
    pub fn new() -> Self {
        Self {
            clear_tint: ShaderId(usize::MAX),
            bboxes: ShaderId(usize::MAX),
            linesoup: ShaderId(usize::MAX),
            linesoup_points: ShaderId(usize::MAX),
            unpaired_points: ShaderId(usize::MAX),
        }
    }

    pub fn render(
        &self,
        recorder: &mut FrameRecorder<'_>,
        _captured: &CapturedBuffers,
        bump: &BumpAllocators,
        params: &RenderParams,
        downloads: &DebugDownloads<'_>,
        layers: DebugLayers,
    ) {
        if layers.is_empty() {
            return;
        }

        let (unpaired_pts_len, unpaired_pts_buf) = if layers.contains(DebugLayers::VALIDATION) {
            let unpaired_pts: Vec<LineEndpoint> =
                validate_line_soup(bytemuck::cast_slice(downloads.lines));
            if unpaired_pts.is_empty() {
                (0, None)
            } else {
                (
                    unpaired_pts.len(),
                    Some(recorder.upload(
                        "ekrano.debug.unpaired_points",
                        bytemuck::cast_slice(&unpaired_pts[..]),
                    )),
                )
            }
        } else {
            (0, None)
        };

        let uniforms = Uniforms {
            width: params.width,
            height: params.height,
        };
        let uniforms_buf = recorder.upload(
            "ekrano.debug_uniforms",
            bytemuck::bytes_of(&uniforms).to_vec(),
        );

        let linepoints_uniforms = [
            LinepointsUniforms::new(palette::css::DARK_CYAN.discard_alpha(), 10.),
            LinepointsUniforms::new(palette::css::RED.discard_alpha(), 80.),
        ];
        let linepoints_uniforms_buf = recorder.upload_strided(
            "ekrano.debug.linepoints_uniforms",
            size_of::<LinepointsUniforms>() as u32,
            bytemuck::bytes_of(&linepoints_uniforms).to_vec(),
        );

        recorder.draw(DrawParams {
            shader_id: self.clear_tint,
            instance_count: 1,
            vertex_count: 4,
            vertex_buffer: None,
            resources: vec![],
            target: None,
            clear_color: None,
        });
        if layers.contains(DebugLayers::BOUNDING_BOXES) {
            recorder.draw(DrawParams {
                shader_id: self.bboxes,
                instance_count: _captured.sizes.path_bboxes.len(),
                vertex_count: 5,
                vertex_buffer: None,
                resources: vec![],
                target: None,
                clear_color: None,
            });
        }
        if layers.contains(DebugLayers::LINESOUP_SEGMENTS) {
            recorder.draw(DrawParams {
                shader_id: self.linesoup,
                instance_count: bump.lines,
                vertex_count: 4,
                vertex_buffer: None,
                resources: vec![],
                target: None,
                clear_color: None,
            });
        }
        if layers.contains(DebugLayers::LINESOUP_POINTS) {
            recorder.draw(DrawParams {
                shader_id: self.linesoup_points,
                instance_count: bump.lines,
                vertex_count: 4,
                vertex_buffer: None,
                resources: vec![],
                target: None,
                clear_color: None,
            });
        }
        if let Some(unpaired_pts_buf) = unpaired_pts_buf {
            recorder.draw(DrawParams {
                shader_id: self.unpaired_points,
                instance_count: unpaired_pts_len.try_into().unwrap(),
                vertex_count: 4,
                vertex_buffer: Some(unpaired_pts_buf),
                resources: vec![],
                target: None,
                clear_color: None,
            });
        }

        recorder.defer_owned_buffer(uniforms_buf, "ekrano.debug_uniforms");
        recorder.defer_owned_buffer(linepoints_uniforms_buf, "ekrano.debug.linepoints_uniforms");
    }
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct Uniforms {
    width: u32,
    height: u32,
}

#[derive(Copy, Clone, Zeroable, Pod)]
#[repr(C)]
struct LinepointsUniforms {
    point_color: [f32; 3],
    point_size: f32,
    // Uniform parameters for individual SDF point draws are stored in a single buffer.
    // This 240 byte padding is here to bring the element offset alignment of 256 bytes.
    // (see https://www.w3.org/TR/webgpu/#dom-supported-limits-minuniformbufferoffsetalignment)
    _pad0: [u32; 30],
    _pad1: [u32; 30],
}

impl LinepointsUniforms {
    fn new(color: OpaqueColor<Srgb>, point_size: f32) -> Self {
        Self {
            point_color: color.components,
            point_size,
            _pad0: [0; 30],
            _pad1: [0; 30],
        }
    }
}
