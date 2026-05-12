// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared types for shader binding metadata and debug stubs.
//!
//! Historical note: this module used to define `ResourceProxy` / `BufferProxy` handles for a
//! bind-map recording layer; the render path now uses direct [`goldy::Buffer`] / [`goldy::Texture`]
//! handles (`crate::gpu_resources`).

/// Shader entry index in the renderer's pipeline table (matches Slang entry-point order).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShaderId(pub usize);

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    Rgba8,
    Bgra8,
}

/// The type of resource that will be bound to a slot in a shader.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BindType {
    /// A storage buffer with read/write access.
    Buffer,
    /// A storage buffer with read only access.
    BufReadOnly,
    /// A small storage buffer to be used as uniforms.
    Uniform,
    /// A storage image.
    Image(ImageFormat),
    /// A storage image with read only access.
    ImageRead(ImageFormat),
}

#[cfg(feature = "debug_layers")]
#[allow(
    dead_code,
    reason = "fields read by debug renderer; draw is a no-op stub"
)]
pub struct DrawParams {
    pub shader_id: ShaderId,
    pub instance_count: u32,
    pub vertex_count: u32,
    pub vertex_buffer: Option<goldy::Buffer>,
    pub resources: Vec<goldy::Buffer>,
    pub target: Option<goldy::Texture>,
    pub clear_color: Option<[f32; 4]>,
}
