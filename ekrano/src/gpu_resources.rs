// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Direct GPU resource handles (no bind-map / proxies).

use std::mem::size_of;

use goldy::task_graph::NodeAccess;
use goldy::types::{BufferFlags, SpatialAccess, TextureFlags};
use goldy::{
    Buffer, BufferView, DataAccess, Device, DeviceType, TaskGraph, Texture, TextureFormat,
};

use crate::goldy_renderer::PersistentState;
use crate::resource_proxy::{BindType, ImageFormat};
use crate::{Error, RenderParams, Result};
use ekrano_encoding::{BumpAllocators, Encoding, Images, IndirectCount, Ramps, RenderConfig};

pub(crate) enum GpuBuf {
    Owned(Buffer),
    Pooled(BufferView),
}

impl GpuBuf {
    pub(crate) fn size(&self) -> u64 {
        match self {
            Self::Owned(b) => b.size(),
            Self::Pooled(v) => v.size(),
        }
    }

    pub(crate) fn as_indirect_buffer(&self) -> Option<&Buffer> {
        match self {
            Self::Owned(b) => Some(b),
            Self::Pooled(_) => None,
        }
    }

    pub(crate) fn as_binding(&self) -> GpuBinding<'_> {
        match self {
            Self::Owned(b) => GpuBinding::Buf(b),
            Self::Pooled(v) => GpuBinding::View(v),
        }
    }
}

/// Extension trait so `Option<GpuBuf>` fields can be bound with `.binding()` in dispatch
/// call sites. Panics if the buffer has already been freed mid-pipeline.
pub(crate) trait OptGpuBufExt {
    fn binding(&self) -> GpuBinding<'_>;
}

impl OptGpuBufExt for Option<GpuBuf> {
    #[track_caller]
    fn binding(&self) -> GpuBinding<'_> {
        self.as_ref().expect("pipeline buffer already freed").as_binding()
    }
}

pub(crate) enum GpuBinding<'a> {
    Buf(&'a Buffer),
    View(&'a BufferView),
    Tex(&'a Texture),
}

impl<'a> GpuBinding<'a> {
    pub(crate) fn bindless_slot(&self, is_read_only: bool) -> Result<u32, Error> {
        let idx = match self {
            GpuBinding::Buf(buf) => {
                if is_read_only {
                    buf.bindless_srv_index()
                } else {
                    buf.bindless_index()
                }
            }
            GpuBinding::View(view) => {
                if is_read_only {
                    view.bindless_srv_index()
                } else {
                    view.bindless_index()
                }
            }
            GpuBinding::Tex(tex) => tex.bindless_index(),
        };
        idx.ok_or_else(|| {
            Error::Shader("bindless index missing for shader resource binding".into())
        })
    }
}

fn use_pool(device: &Device) -> bool {
    device.device_type() != DeviceType::Cpu
}

fn is_pool_exempt(name: &'static str) -> bool {
    matches!(
        name,
        "ekrano.bump_buf"
            | "ekrano.indirect_count"
            | "ekrano.indirect_dispatch"
    )
}

fn image_fmt_goldy(f: ImageFormat) -> TextureFormat {
    match f {
        ImageFormat::Rgba8 => TextureFormat::Rgba8Unorm,
        ImageFormat::Bgra8 => TextureFormat::Bgra8Unorm,
    }
}

pub(crate) fn alloc_pipeline_buffer(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    size: u64,
    stride: u32,
    name: &'static str,
    flags: BufferFlags,
) -> Result<GpuBuf, Error> {
    let pooled = use_pool(device) && !is_pool_exempt(name);
    if pooled {
        let allocator = persistent
            .storage_allocator_mut()
            .ok_or_else(|| Error::Shader("storage allocator not prepared".into()))?;
        let view = allocator
            .alloc(device, size, Some(stride))
            .map_err(|e| Error::Shader(e.to_string()))?;
        Ok(GpuBuf::Pooled(view))
    } else {
        let buf = persistent.pool.get_buf_with_stride(
            device,
            size,
            name,
            DataAccess::Scattered,
            Some(stride),
            flags,
        )?;
        graph.clear_buffer(&buf, 0, size);
        Ok(GpuBuf::Owned(buf))
    }
}

pub(crate) fn record_upload_bytes(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    name: &'static str,
    element_stride: u32,
    bytes: &[u8],
) -> Result<GpuBuf, Error> {
    let buf = persistent.pool.get_buf_with_stride(
        device,
        bytes.len() as u64,
        name,
        DataAccess::Scattered,
        Some(element_stride),
        BufferFlags::empty(),
    )?;
    graph.write_buffer(&buf, 0, bytes.to_vec());
    Ok(GpuBuf::Owned(buf))
}

pub(crate) fn record_upload_image(
    device: &Device,
    graph: &mut TaskGraph,
    persistent: &mut PersistentState,
    width: u32,
    height: u32,
    format: ImageFormat,
    bytes: &[u8],
) -> Result<Texture, Error> {
    let format = image_fmt_goldy(format);
    let texture = persistent
        .tex_pool
        .acquire(
            device,
            width,
            height,
            format,
            SpatialAccess::Interpolated,
            TextureFlags::COPY_DST,
        )
        .map_err(|e| Error::Shader(e.to_string()))?;
    graph
        .write_texture(&texture, bytes.to_vec())
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(texture)
}

pub(crate) fn write_image_region(
    graph: &mut TaskGraph,
    tex: &Texture,
    x: u32,
    y: u32,
    image_data: &peniko::ImageData,
) -> Result<(), Error> {
    if image_data.data.is_empty() && image_data.width != 0 && image_data.height != 0 {
        return Err(Error::InvalidImage {
            id: image_data.data.id(),
            reason: "image has non-zero dimensions but no pixel data; \
                     it may have been registered to a different renderer \
                     or unregistered before this render was submitted",
        });
    }
    let bytes = image_data.data.data();
    graph
        .write_texture_region(
            tex,
            x,
            y,
            image_data.width,
            image_data.height,
            bytes.to_vec(),
        )
        .map_err(|e| Error::Shader(e.to_string()))?;
    Ok(())
}

pub(crate) fn acquire_texture_rgba(
    device: &Device,
    persistent: &mut PersistentState,
    width: u32,
    height: u32,
    access: SpatialAccess,
    flags: TextureFlags,
) -> Result<Texture, Error> {
    persistent
        .tex_pool
        .acquire(
            device,
            width,
            height,
            TextureFormat::Rgba8Unorm,
            access,
            flags,
        )
        .map_err(|e| Error::Shader(e.to_string()))
}

pub(crate) fn clear_gpu_buf(
    graph: &mut TaskGraph,
    buf: &GpuBuf,
    off: u64,
    size: Option<u64>,
) -> Result<(), Error> {
    let sz = size.unwrap_or_else(|| buf.size().saturating_sub(off));
    match buf {
        GpuBuf::Owned(b) => graph.clear_buffer(b, off, sz),
        GpuBuf::Pooled(v) => graph.clear_buffer_view(v, off, sz),
    }
    Ok(())
}

pub(crate) struct PipelineResources {
    pub gradient: Texture,
    pub image_atlas: Texture,
    pub mask_atlas: Texture,
    pub scene: Option<GpuBuf>,
    pub config: GpuBuf,
    pub wg_counts: Option<GpuBuf>,
    pub indirect: Option<GpuBuf>,
    pub fallback_indirect: GpuBuf,
    pub info_bin_data: GpuBuf,
    pub tile: GpuBuf,
    pub segments: GpuBuf,
    pub ptcl: GpuBuf,
    pub reduced: Option<GpuBuf>,
    pub reduced2: Option<GpuBuf>,
    pub reduced_scan: Option<GpuBuf>,
    pub tagmonoid: Option<GpuBuf>,
    pub path_bbox: Option<GpuBuf>,
    pub bump: GpuBuf,
    pub lines: Option<GpuBuf>,
    pub draw_reduced: Option<GpuBuf>,
    pub draw_monoid: Option<GpuBuf>,
    pub clip_inp: Option<GpuBuf>,
    pub clip_el: Option<GpuBuf>,
    pub clip_bic: Option<GpuBuf>,
    pub clip_bbox: Option<GpuBuf>,
    pub draw_bbox: Option<GpuBuf>,
    pub bin_header: Option<GpuBuf>,
    pub path: Option<GpuBuf>,
    pub seg_counts: Option<GpuBuf>,
    pub blend_spill: GpuBuf,
    pub out_image: Texture,
    pub filter_layers: [Texture; 4],
}

impl PipelineResources {
    #[allow(
        clippy::too_many_arguments,
        reason = "Single setup function threads every pipeline buffer and texture from resolve data"
    )]
    pub(crate) fn prepare(
        device: &Device,
        graph: &mut TaskGraph,
        persistent: &mut PersistentState,
        encoding: &Encoding,
        mut packed: Vec<u8>,
        ramps: Ramps<'_>,
        images: Images<'_>,
        params: &RenderParams,
        config: &RenderConfig,
    ) -> Result<Self, Error> {
        if packed.is_empty() {
            packed.resize(size_of::<u32>(), u8::MAX);
        }

        let mut cpu_config_owned = *config;
        if encoding.coverage_mask.is_some() {
            cpu_config_owned.gpu.mask_active = 1;
        }
        if let Some(ref m) = encoding.coverage_mask {
            assert_eq!(
                m.width, params.width,
                "coverage_mask width must match render width"
            );
            assert_eq!(
                m.height, params.height,
                "coverage_mask height must match render height"
            );
        }

        let gradient = if ramps.height == 0 {
            acquire_texture_rgba(
                device,
                persistent,
                1,
                1,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST,
            )?
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramps.data);
            record_upload_image(
                device,
                graph,
                persistent,
                ramps.width,
                ramps.height,
                ImageFormat::Rgba8,
                data,
            )?
        };

        let (image_atlas, _) = if images.images.is_empty() {
            let t = acquire_texture_rgba(
                device,
                persistent,
                1,
                1,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )?;
            (t, (1_u32, 1_u32))
        } else {
            let t = acquire_texture_rgba(
                device,
                persistent,
                images.width,
                images.height,
                SpatialAccess::Interpolated,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )?;
            (t, (images.width, images.height))
        };
        for image in images.images {
            write_image_region(graph, &image_atlas, image.1, image.2, &image.0)?;
        }

        let mask_atlas = match &encoding.coverage_mask {
            Some(m) => {
                let mut rgba = Vec::with_capacity(m.data.len() * 4);
                for &b in m.data.iter() {
                    rgba.extend_from_slice(&[b, b, b, 255]);
                }
                record_upload_image(
                    device,
                    graph,
                    persistent,
                    m.width,
                    m.height,
                    ImageFormat::Rgba8,
                    &rgba,
                )?
            }
            None => record_upload_image(
                device,
                graph,
                persistent,
                1,
                1,
                ImageFormat::Rgba8,
                &[255, 255, 255, 255],
            )?,
        };

        let scene = record_upload_bytes(device, graph, persistent, "ekrano.scene", 4, &packed)?;
        let config = record_upload_bytes(
            device,
            graph,
            persistent,
            "ekrano.config",
            size_of::<ekrano_encoding::ConfigUniform>() as u32,
            bytemuck::bytes_of(&cpu_config_owned.gpu),
        )?;

        let buffer_sizes = &cpu_config_owned.buffer_sizes;

        let fallback_indirect = alloc_pipeline_buffer(
            device,
            graph,
            persistent,
            size_of::<IndirectCount>() as u64,
            size_of::<IndirectCount>() as u32,
            "ekrano.indirect_count",
            BufferFlags::empty(),
        )?;

        macro_rules! al {
            ($sz:expr, $stride:expr, $name:expr) => {
                alloc_pipeline_buffer(
                    device,
                    graph,
                    persistent,
                    $sz,
                    $stride,
                    $name,
                    BufferFlags::empty(),
                )?
            };
        }

        let info_bin_data = al!(
            buffer_sizes.bin_data.size_in_bytes() as u64,
            4,
            "ekrano.info_bin_data_buf"
        );
        let tile = al!(
            buffer_sizes.tiles.size_in_bytes().into(),
            8,
            "ekrano.tile_buf"
        );
        let segments = al!(
            buffer_sizes.segments.size_in_bytes().into(),
            24,
            "ekrano.segments_buf"
        );
        let ptcl = al!(
            buffer_sizes.ptcl.size_in_bytes().into(),
            4,
            "ekrano.ptcl_buf"
        );
        let reduced = al!(
            buffer_sizes.path_reduced.size_in_bytes().into(),
            20,
            "ekrano.reduced_buf"
        );
        let reduced2 = al!(
            buffer_sizes.path_reduced2.size_in_bytes().into(),
            20,
            "ekrano.reduced2_buf"
        );
        let reduced_scan = al!(
            buffer_sizes.path_reduced_scan.size_in_bytes().into(),
            20,
            "ekrano.reduced_scan_buf"
        );
        let tagmonoid = al!(
            buffer_sizes.path_monoids.size_in_bytes().into(),
            20,
            "ekrano.tagmonoid_buf"
        );
        let path_bbox = al!(
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            24,
            "ekrano.path_bbox_buf"
        );
        let bump = alloc_pipeline_buffer(
            device,
            graph,
            persistent,
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            size_of::<BumpAllocators>() as u32,
            "ekrano.bump_buf",
            BufferFlags::CPU_READABLE,
        )?;
        clear_gpu_buf(graph, &bump, 0, None)?;
        let lines = al!(
            buffer_sizes.lines.size_in_bytes().into(),
            24,
            "ekrano.lines_buf"
        );
        let draw_reduced = al!(
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            16,
            "ekrano.draw_reduced_buf"
        );
        let draw_monoid = al!(
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            16,
            "ekrano.draw_monoid_buf"
        );
        let clip_inp = al!(
            buffer_sizes.clip_inps.size_in_bytes().into(),
            8,
            "ekrano.clip_inp_buf"
        );
        let clip_el = al!(
            buffer_sizes.clip_els.size_in_bytes().into(),
            32,
            "ekrano.clip_el_buf"
        );
        let clip_bic = al!(
            buffer_sizes.clip_bics.size_in_bytes().into(),
            8,
            "ekrano.clip_bic_buf"
        );
        let clip_bbox = al!(
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            16,
            "ekrano.clip_bbox_buf"
        );
        let draw_bbox = al!(
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            16,
            "ekrano.draw_bbox_buf"
        );
        let bin_header = al!(
            buffer_sizes.bin_headers.size_in_bytes().into(),
            8,
            "ekrano.bin_header_buf"
        );
        let path = al!(
            buffer_sizes.paths.size_in_bytes().into(),
            32,
            "ekrano.path_buf"
        );
        let seg_counts = al!(
            buffer_sizes.seg_counts.size_in_bytes().into(),
            8,
            "ekrano.seg_counts_buf"
        );
        let blend_spill = al!(
            buffer_sizes.blend_spill.size_in_bytes().into(),
            size_of::<u32>() as u32,
            "ekrano.blend_spill"
        );

        let out_image = acquire_texture_rgba(
            device,
            persistent,
            params.width,
            params.height,
            SpatialAccess::Direct,
            TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
        )?;
        let filter_layers = std::array::from_fn(|_| {
            acquire_texture_rgba(
                device,
                persistent,
                params.width,
                params.height,
                SpatialAccess::Direct,
                TextureFlags::COPY_DST | TextureFlags::COPY_SRC,
            )
            .expect("filter layer")
        });

        Ok(Self {
            gradient,
            image_atlas,
            mask_atlas,
            scene: Some(scene),
            config,
            wg_counts: None,
            indirect: None,
            fallback_indirect,
            info_bin_data,
            tile,
            segments,
            ptcl,
            reduced: Some(reduced),
            reduced2: Some(reduced2),
            reduced_scan: Some(reduced_scan),
            tagmonoid: Some(tagmonoid),
            path_bbox: Some(path_bbox),
            bump,
            lines: Some(lines),
            draw_reduced: Some(draw_reduced),
            draw_monoid: Some(draw_monoid),
            clip_inp: Some(clip_inp),
            clip_el: Some(clip_el),
            clip_bic: Some(clip_bic),
            clip_bbox: Some(clip_bbox),
            draw_bbox: Some(draw_bbox),
            bin_header: Some(bin_header),
            path: Some(path),
            seg_counts: Some(seg_counts),
            blend_spill,
            out_image,
            filter_layers,
        })
    }
}

pub(crate) fn collect_bindless_indices_direct(
    bindings: &[GpuBinding<'_>],
    bind_types: &[BindType],
    force_uav: bool,
    max_slots: usize,
) -> Result<Vec<u32>, Error> {
    let mut indices = Vec::with_capacity(bindings.len());
    for (i, binding) in bindings.iter().enumerate() {
        let is_read_only = !force_uav && matches!(bind_types.get(i), Some(BindType::BufReadOnly));
        let idx = match binding {
            GpuBinding::Buf(_) | GpuBinding::View(_) => binding.bindless_slot(is_read_only)?,
            GpuBinding::Tex(_) => binding.bindless_slot(false)?,
        };
        indices.push(idx);
    }
    if indices.len() > max_slots {
        return Err(Error::Shader(format!(
            "shader requires {} bindless slots, exceeds limit of {}",
            indices.len(),
            max_slots
        )));
    }
    Ok(indices)
}

pub(crate) fn bind_type_to_node_access(bt: BindType) -> NodeAccess {
    match bt {
        BindType::Buffer | BindType::Image(_) => NodeAccess::ReadWrite,
        BindType::BufReadOnly | BindType::Uniform | BindType::ImageRead(_) => NodeAccess::Read,
    }
}
