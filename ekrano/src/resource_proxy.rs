// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Lightweight proxy handles for GPU resources.
//!
//! Proxies are cheap, `Copy` identifiers used by the rendering pipeline to
//! reference buffers and images before (and after) they are materialised into
//! actual Goldy resources. They carry metadata (size, format, stride) that the
//! [`FrameRecorder`](crate::goldy_renderer::FrameRecorder) uses to create the
//! real GPU objects on demand.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use goldy::types::BufferFlags;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShaderId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub NonZeroU64);

impl ResourceId {
    pub fn next() -> Self {
        static ID_COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(NonZeroU64::new(ID_COUNTER.fetch_add(1, Ordering::Relaxed)).unwrap())
    }
}

/// Proxy used as a handle to a buffer.
#[derive(Clone, Copy)]
pub struct BufferProxy {
    pub size: u64,
    pub id: ResourceId,
    pub name: &'static str,
    /// Structured buffer element stride in bytes.
    /// When set, the GPU backend uses this for `StructureByteStride` in SRV/UAV descriptors.
    /// When `None`, the engine falls back to a name-based lookup.
    pub element_stride: Option<u32>,
    /// Goldy buffer creation flags (e.g. [`BufferFlags::CPU_READABLE`] for host-readable bump data).
    pub buffer_flags: BufferFlags,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    Rgba8,
    Bgra8,
}

/// Proxy used as a handle to an image.
#[derive(Clone, Copy)]
pub struct ImageProxy {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub id: ResourceId,
}

#[derive(Clone, Copy)]
pub enum ResourceProxy {
    Buffer(BufferProxy),
    BufferRange {
        proxy: BufferProxy,
        offset: u64,
        size: u64,
    },
    Image(ImageProxy),
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
    pub vertex_buffer: Option<BufferProxy>,
    pub resources: Vec<ResourceProxy>,
    pub target: ImageProxy,
    pub clear_color: Option<[f32; 4]>,
}

impl BufferProxy {
    pub fn new(size: u64, name: &'static str) -> Self {
        let id = ResourceId::next();
        debug_assert!(size > 0);
        Self {
            id,
            size,
            name,
            element_stride: None,
            buffer_flags: BufferFlags::empty(),
        }
    }

    /// Create a proxy with an explicit structured buffer element stride.
    pub fn with_stride(size: u64, name: &'static str, element_stride: u32) -> Self {
        Self::with_stride_and_flags(size, name, element_stride, BufferFlags::empty())
    }

    /// [`Self::with_stride`] with Goldy [`BufferFlags`].
    pub fn with_stride_and_flags(
        size: u64,
        name: &'static str,
        element_stride: u32,
        buffer_flags: BufferFlags,
    ) -> Self {
        let id = ResourceId::next();
        debug_assert!(size > 0);
        debug_assert!(element_stride > 0);
        debug_assert!(
            size.is_multiple_of(element_stride as u64),
            "buffer size {size} not divisible by element stride {element_stride}"
        );
        Self {
            id,
            size,
            name,
            element_stride: Some(element_stride),
            buffer_flags,
        }
    }
}

impl ImageProxy {
    pub fn new(width: u32, height: u32, format: ImageFormat) -> Self {
        let id = ResourceId::next();
        Self {
            width,
            height,
            format,
            id,
        }
    }
}

impl ResourceProxy {
    pub fn new_buf(size: u64, name: &'static str) -> Self {
        Self::Buffer(BufferProxy::new(size, name))
    }

    pub fn new_image(width: u32, height: u32, format: ImageFormat) -> Self {
        Self::Image(ImageProxy::new(width, height, format))
    }

    pub fn as_buf(&self) -> Option<&BufferProxy> {
        match self {
            Self::Buffer(proxy) => Some(proxy),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageProxy> {
        match self {
            Self::Image(proxy) => Some(proxy),
            _ => None,
        }
    }
}

impl From<BufferProxy> for ResourceProxy {
    fn from(value: BufferProxy) -> Self {
        Self::Buffer(value)
    }
}

impl From<ImageProxy> for ResourceProxy {
    fn from(value: ImageProxy) -> Self {
        Self::Image(value)
    }
}
