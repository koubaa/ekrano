// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SegmentCount;

use super::{
    BinHeader, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox, DrawMonoid, Layout, LineSoup, Path, PathBbox,
    PathMonoid, PathSegment, Tile,
};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;

// TODO: Obtain these from the vello_shaders crate
pub(crate) const PATH_REDUCE_WG: u32 = 256;
const PATH_BBOX_WG: u32 = 256;
const FLATTEN_WG: u32 = 256;
const CLIP_REDUCE_WG: u32 = 256;

/// Counters for tracking dynamic allocation on the GPU.
///
/// This must be kept in sync with the struct in `ekrano_shaders/slang/ekrano_shared.slang` (`BumpAllocators`).
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct BumpAllocators {
    pub failed: u32,
    // Final needed dynamic size of the buffers. If any of these are larger
    // than the corresponding `_size` element reallocation needs to occur.
    pub binning: u32,
    pub ptcl: u32,
    pub tile: u32,
    pub seg_counts: u32,
    pub segments: u32,
    pub blend: u32,
    pub lines: u32,
    /// Count of tiles with non-empty PTCL (written by coarse, read by fine_setup).
    pub active_tiles: u32,
}

#[derive(Default)]
pub struct BumpAllocatorMemory {
    pub total: u32,
    pub binning: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
    pub tile: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub lines: BufferSize<LineSoup>,
}

impl BumpAllocators {
    pub fn memory(&self) -> BumpAllocatorMemory {
        let binning = BufferSize::new(self.binning);
        let ptcl = BufferSize::new(self.ptcl);
        let tile = BufferSize::new(self.tile);
        let seg_counts = BufferSize::new(self.seg_counts);
        let segments = BufferSize::new(self.segments);
        let lines = BufferSize::new(self.lines);
        BumpAllocatorMemory {
            total: binning.size_in_bytes()
                + ptcl.size_in_bytes()
                + tile.size_in_bytes()
                + seg_counts.size_in_bytes()
                + segments.size_in_bytes()
                + lines.size_in_bytes(),
            binning,
            ptcl,
            tile,
            seg_counts,
            segments,
            lines,
        }
    }
}

impl std::fmt::Display for BumpAllocatorMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n \
                 \tTotal:\t\t\t{} bytes ({:.2} KB | {:.2} MB)\n\
                 \tBinning\t\t\t{} elements ({} bytes)\n\
                 \tPTCL\t\t\t{} elements ({} bytes)\n\
                 \tTile:\t\t\t{} elements ({} bytes)\n\
                 \tSegment Counts:\t\t{} elements ({} bytes)\n\
                 \tSegments:\t\t{} elements ({} bytes)\n\
                 \tLines:\t\t\t{} elements ({} bytes)",
            self.total,
            self.total as f32 / (1 << 10) as f32,
            self.total as f32 / (1 << 20) as f32,
            self.binning.len(),
            self.binning.size_in_bytes(),
            self.ptcl.len(),
            self.ptcl.size_in_bytes(),
            self.tile.len(),
            self.tile.size_in_bytes(),
            self.seg_counts.len(),
            self.seg_counts.size_in_bytes(),
            self.segments.len(),
            self.segments.size_in_bytes(),
            self.lines.len(),
            self.lines.size_in_bytes()
        )
    }
}

/// Storage of indirect dispatch size values.
///
/// The original plan was to reuse [`BumpAllocators`], but the WebGPU compatible
/// usage list rules forbid that being used as indirect counts while also
/// bound as writable.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct IndirectCount {
    pub count_x: u32,
    pub count_y: u32,
    pub count_z: u32,
    pub pad0: u32,
}

/// Stage indices into the multi-entry indirect dispatch buffer.
/// Must match the order used by `pipeline_setup.slang`.
pub const STAGE_PATHTAG_REDUCE: u32 = 0;
pub const STAGE_PATHTAG_REDUCE2: u32 = 1;
pub const STAGE_PATHTAG_SCAN1: u32 = 2;
pub const STAGE_PATHTAG_SCAN: u32 = 3;
pub const STAGE_PATHTAG_SCAN_LARGE: u32 = 4;
pub const STAGE_BBOX_CLEAR: u32 = 5;
pub const STAGE_FLATTEN: u32 = 6;
pub const STAGE_DRAW_REDUCE: u32 = 7;
pub const STAGE_DRAW_LEAF: u32 = 8;
pub const STAGE_CLIP_REDUCE: u32 = 9;
pub const STAGE_CLIP_LEAF: u32 = 10;
pub const STAGE_BINNING: u32 = 11;
pub const STAGE_TILE_ALLOC: u32 = 12;
pub const STAGE_PATH_COUNT_SETUP: u32 = 13;
pub const STAGE_PATH_COUNT: u32 = 14;
pub const STAGE_BACKDROP: u32 = 15;
pub const STAGE_COARSE: u32 = 16;
pub const STAGE_PATH_TILING_SETUP: u32 = 17;
pub const STAGE_PATH_TILING: u32 = 18;
pub const STAGE_FINE: u32 = 19;

pub const N_INDIRECT_STAGES: u32 = 20;

/// GPU-uploadable workgroup counts for Phase 1 indirect dispatch.
/// Layout must match `ekrano_shared.slang` `WorkgroupCountsGpu`.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq, Eq)]
#[repr(C)]
pub struct WorkgroupCountsGpu {
    pub entries: [[u32; 4]; N_INDIRECT_STAGES as usize],
}

/// Uniform render configuration data used by all GPU stages.
///
/// This data structure must be kept in sync with the definition in
/// `ekrano_shaders/slang/ekrano_shared.slang` (Config).
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, PartialEq)]
#[repr(C)]
pub struct ConfigUniform {
    /// Width of the scene in tiles.
    pub width_in_tiles: u32,
    /// Height of the scene in tiles.
    pub height_in_tiles: u32,
    /// Width of the target in pixels.
    pub target_width: u32,
    /// Height of the target in pixels.
    pub target_height: u32,
    /// The base background color applied to the target before any blends.
    pub base_color: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
    /// Size of line soup buffer allocation (in [`LineSoup`]s)
    pub lines_size: u32,
    /// Size of binning buffer allocation (in `u32`s).
    pub binning_size: u32,
    /// Size of tile buffer allocation (in [`Tile`]s).
    pub tiles_size: u32,
    /// Size of segment count buffer allocation (in [`SegmentCount`]s).
    pub seg_counts_size: u32,
    /// Size of segment buffer allocation (in [`PathSegment`]s).
    pub segments_size: u32,
    /// Size of blend spill buffer (in `u32` pixels).
    // TODO: Maybe store in TILE_WIDTH * TILE_HEIGHT blocks of pixels instead?
    pub blend_size: u32,
    /// Size of per-tile command list buffer allocation (in `u32`s).
    pub ptcl_size: u32,
    /// Y-offset for fine pass row splitting (0 = full dispatch).
    pub tile_y_offset: u32,
    /// Global thread index offset for chunked flatten dispatches (0 = default).
    /// Must be a multiple of the flatten workgroup size (256).
    pub flatten_thread_base: u32,
    /// Non-zero if [`crate::encoding::Encoding::coverage_mask`] is set (fine stage samples `mask_atlas`).
    pub mask_active: u32,
}

/// CPU side setup and configuration.
#[derive(Clone, Copy, Default)]
pub struct RenderConfig {
    /// GPU side configuration.
    pub gpu: ConfigUniform,
    /// Workgroup counts for all compute pipelines.
    pub workgroup_counts: WorkgroupCounts,
    /// Sizes of all buffer resources.
    pub buffer_sizes: BufferSizes,
}

impl RenderConfig {
    pub fn new(layout: &Layout, width: u32, height: u32, base_color: &peniko::Color) -> Self {
        let new_width = width.next_multiple_of(TILE_WIDTH);
        let new_height = height.next_multiple_of(TILE_HEIGHT);
        let width_in_tiles = new_width / TILE_WIDTH;
        let height_in_tiles = new_height / TILE_HEIGHT;
        let n_path_tags = layout.path_tags_size();
        let workgroup_counts = WorkgroupCounts::new(layout, width_in_tiles, height_in_tiles, n_path_tags);
        let buffer_sizes = BufferSizes::new(layout, &workgroup_counts);
        Self {
            gpu: ConfigUniform {
                width_in_tiles,
                height_in_tiles,
                target_width: width,
                target_height: height,
                base_color: base_color.premultiply().to_rgba8().to_u32(),
                lines_size: buffer_sizes.lines.len(),
                binning_size: buffer_sizes.bin_data.len() - layout.bin_data_start,
                tiles_size: buffer_sizes.tiles.len(),
                seg_counts_size: buffer_sizes.seg_counts.len(),
                segments_size: buffer_sizes.segments.len(),
                blend_size: buffer_sizes.blend_spill.len(),
                ptcl_size: buffer_sizes.ptcl.len(),
                tile_y_offset: 0,
                flatten_thread_base: 0,
                mask_active: 0,
                layout: *layout,
            },
            workgroup_counts,
            buffer_sizes,
        }
    }

    /// Create a new config with bump-allocated buffers resized to accommodate
    /// actual GPU usage. The GPU config's `*_size` fields are updated to match.
    pub fn with_bump_estimates(&self, bump: &BumpAllocators) -> Self {
        let buffer_sizes = self.buffer_sizes.with_bump_estimates(bump);
        let layout = &self.gpu.layout;
        Self {
            gpu: ConfigUniform {
                lines_size: buffer_sizes.lines.len(),
                binning_size: buffer_sizes.bin_data.len() - layout.bin_data_start,
                tiles_size: buffer_sizes.tiles.len(),
                seg_counts_size: buffer_sizes.seg_counts.len(),
                segments_size: buffer_sizes.segments.len(),
                blend_size: buffer_sizes.blend_spill.len(),
                ptcl_size: buffer_sizes.ptcl.len(),
                ..self.gpu
            },
            buffer_sizes,
            workgroup_counts: self.workgroup_counts,
        }
    }
}

/// Type alias for a workgroup size.
pub type WorkgroupSize = (u32, u32, u32);

/// Computed sizes for all dispatches.
#[derive(Copy, Clone, Debug, Default)]
pub struct WorkgroupCounts {
    pub use_large_path_scan: bool,
    pub path_reduce: WorkgroupSize,
    pub path_reduce2: WorkgroupSize,
    pub path_scan1: WorkgroupSize,
    pub path_scan: WorkgroupSize,
    pub bbox_clear: WorkgroupSize,
    pub flatten: WorkgroupSize,
    pub draw_reduce: WorkgroupSize,
    pub draw_leaf: WorkgroupSize,
    pub clip_reduce: WorkgroupSize,
    pub clip_leaf: WorkgroupSize,
    pub binning: WorkgroupSize,
    pub tile_alloc: WorkgroupSize,
    pub path_count_setup: WorkgroupSize,
    // Note: `path_count` must use an indirect dispatch
    pub backdrop: WorkgroupSize,
    pub coarse: WorkgroupSize,
    pub path_tiling_setup: WorkgroupSize,
    // Note: `path_tiling` must use an indirect dispatch
    pub fine: WorkgroupSize,
}

impl WorkgroupCounts {
    pub fn new(layout: &Layout, width_in_tiles: u32, height_in_tiles: u32, n_path_tags: u32) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_padded = align_up(n_path_tags, 4 * PATH_REDUCE_WG);
        let path_tag_wgs = path_tag_padded / (4 * PATH_REDUCE_WG);
        let use_large_path_scan = path_tag_wgs > PATH_REDUCE_WG;
        let reduced_size = if use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let draw_object_wgs = n_draw_objects.div_ceil(PATH_BBOX_WG);
        let draw_monoid_wgs = draw_object_wgs.min(PATH_BBOX_WG);
        let flatten_wgs = n_path_tags.div_ceil(FLATTEN_WG);
        let clip_reduce_wgs = n_clips.saturating_sub(1) / CLIP_REDUCE_WG;
        let clip_wgs = n_clips.div_ceil(CLIP_REDUCE_WG);
        let path_wgs = n_paths.div_ceil(PATH_BBOX_WG);
        let width_in_bins = width_in_tiles.div_ceil(16);
        let height_in_bins = height_in_tiles.div_ceil(16);
        // coarse is capped to 256 in coarse.slang when bin_headers has only 256 slots (vello #680)
        Self {
            use_large_path_scan,
            path_reduce: (path_tag_wgs, 1, 1),
            path_reduce2: (PATH_REDUCE_WG, 1, 1),
            path_scan1: (reduced_size / PATH_REDUCE_WG, 1, 1),
            path_scan: (path_tag_wgs, 1, 1),
            bbox_clear: (draw_object_wgs, 1, 1),
            flatten: (flatten_wgs, 1, 1),
            draw_reduce: (draw_monoid_wgs, 1, 1),
            draw_leaf: (draw_monoid_wgs, 1, 1),
            clip_reduce: (clip_reduce_wgs, 1, 1),
            clip_leaf: (clip_wgs, 1, 1),
            binning: (draw_object_wgs, 1, 1),
            tile_alloc: (path_wgs, 1, 1),
            path_count_setup: (1, 1, 1),
            backdrop: (path_wgs, 1, 1),
            coarse: (width_in_bins, height_in_bins, 1),
            path_tiling_setup: (1, 1, 1),
            fine: (width_in_tiles, height_in_tiles, 1),
        }
    }
}

impl From<&WorkgroupCounts> for WorkgroupCountsGpu {
    fn from(wc: &WorkgroupCounts) -> Self {
        let to_u4 = |(x, y, z): WorkgroupSize| [x, y, z, 0];
        let zero = [0_u32; 4];
        let mut entries = [[0_u32; 4]; N_INDIRECT_STAGES as usize];
        entries[STAGE_PATHTAG_REDUCE as usize] = to_u4(wc.path_reduce);
        entries[STAGE_PATHTAG_REDUCE2 as usize] = if wc.use_large_path_scan {
            to_u4(wc.path_reduce2)
        } else {
            zero
        };
        entries[STAGE_PATHTAG_SCAN1 as usize] = if wc.use_large_path_scan {
            to_u4(wc.path_scan1)
        } else {
            zero
        };
        entries[STAGE_PATHTAG_SCAN as usize] = if wc.use_large_path_scan {
            zero
        } else {
            to_u4(wc.path_scan)
        };
        entries[STAGE_PATHTAG_SCAN_LARGE as usize] = if wc.use_large_path_scan {
            to_u4(wc.path_scan)
        } else {
            zero
        };
        entries[STAGE_BBOX_CLEAR as usize] = to_u4(wc.bbox_clear);
        entries[STAGE_FLATTEN as usize] = to_u4(wc.flatten);
        entries[STAGE_DRAW_REDUCE as usize] = to_u4(wc.draw_reduce);
        entries[STAGE_DRAW_LEAF as usize] = to_u4(wc.draw_leaf);
        entries[STAGE_CLIP_REDUCE as usize] = to_u4(wc.clip_reduce);
        entries[STAGE_CLIP_LEAF as usize] = to_u4(wc.clip_leaf);
        entries[STAGE_BINNING as usize] = to_u4(wc.binning);
        entries[STAGE_TILE_ALLOC as usize] = to_u4(wc.tile_alloc);
        entries[STAGE_PATH_COUNT_SETUP as usize] = to_u4(wc.path_count_setup);
        entries[STAGE_PATH_COUNT as usize] = zero; // Written by path_count_setup
        entries[STAGE_BACKDROP as usize] = to_u4(wc.backdrop);
        entries[STAGE_COARSE as usize] = to_u4(wc.coarse);
        entries[STAGE_PATH_TILING_SETUP as usize] = to_u4(wc.path_tiling_setup);
        entries[STAGE_PATH_TILING as usize] = zero; // Written by path_tiling_setup
        entries[STAGE_FINE as usize] = to_u4(wc.fine);
        Self { entries }
    }
}

/// Typed buffer size primitive.
#[derive(Copy, Clone, Eq, Default, Debug)]
pub struct BufferSize<T: Sized> {
    len: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Sized> BufferSize<T> {
    /// Creates a new buffer size from number of elements.
    pub const fn new(len: u32) -> Self {
        Self {
            // Each buffer binding must be large enough to hold at least one element to avoid
            // triggering validation errors.
            //
            // Note: not using `Ord::max` here because it doesn't support const eval yet (except
            // in nightly)
            len: if len > 0 { len } else { 1 },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new buffer size from size in bytes.
    pub const fn from_size_in_bytes(size: u32) -> Self {
        Self::new(size / size_of::<T>() as u32)
    }

    /// Returns the number of elements.
    #[expect(clippy::len_without_is_empty, reason = "The buffer can never be empty")]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Returns the size in bytes.
    pub const fn size_in_bytes(self) -> u32 {
        size_of::<T>() as u32 * self.len
    }

    /// Returns the size in bytes aligned up to the given value.
    pub const fn aligned_in_bytes(self, alignment: u32) -> u32 {
        align_up(self.size_in_bytes(), alignment)
    }
}

impl<T: Sized> PartialEq for BufferSize<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
    }
}

impl<T: Sized> PartialOrd for BufferSize<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.len.partial_cmp(&other.len)
    }
}

/// Computed sizes for all buffers.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct BufferSizes {
    // Known size buffers
    pub path_reduced: BufferSize<PathMonoid>,
    pub path_reduced2: BufferSize<PathMonoid>,
    pub path_reduced_scan: BufferSize<PathMonoid>,
    pub path_monoids: BufferSize<PathMonoid>,
    pub path_bboxes: BufferSize<PathBbox>,
    pub draw_reduced: BufferSize<DrawMonoid>,
    pub draw_monoids: BufferSize<DrawMonoid>,
    pub info: BufferSize<u32>,
    pub clip_inps: BufferSize<Clip>,
    pub clip_els: BufferSize<ClipElement>,
    pub clip_bics: BufferSize<ClipBic>,
    pub clip_bboxes: BufferSize<ClipBbox>,
    pub draw_bboxes: BufferSize<DrawBbox>,
    pub bump_alloc: BufferSize<BumpAllocators>,
    pub indirect_count: BufferSize<IndirectCount>,
    pub bin_headers: BufferSize<BinHeader>,
    pub paths: BufferSize<Path>,
    // Bump allocated buffers
    pub lines: BufferSize<LineSoup>,
    pub bin_data: BufferSize<u32>,
    pub tiles: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub blend_spill: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
    /// Compacted list of active tile indices (capacity = width_in_tiles * height_in_tiles).
    pub active_tile_ids: BufferSize<u32>,
}

impl BufferSizes {
    pub fn new(layout: &Layout, workgroups: &WorkgroupCounts) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_wgs = workgroups.path_reduce.0;
        let reduced_size = if workgroups.use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let path_reduced = BufferSize::new(reduced_size);
        let path_reduced2 = BufferSize::new(PATH_REDUCE_WG);
        let path_reduced_scan = BufferSize::new(reduced_size);
        let path_monoids = BufferSize::new(path_tag_wgs * PATH_REDUCE_WG);
        let path_bboxes = BufferSize::new(n_paths);
        let binning_wgs = workgroups.binning.0;
        let draw_monoid_wgs = workgroups.draw_reduce.0;
        let draw_reduced = BufferSize::new(draw_monoid_wgs);
        let draw_monoids = BufferSize::new(n_draw_objects);
        let info = BufferSize::new(layout.bin_data_start);
        let clip_inps = BufferSize::new(n_clips);
        let clip_els = BufferSize::new(n_clips);
        let clip_bics = BufferSize::new(n_clips / CLIP_REDUCE_WG);
        let clip_bboxes = BufferSize::new(n_clips);
        let draw_bboxes = BufferSize::new(n_paths);
        let bump_alloc = BufferSize::new(1);
        let indirect_count = BufferSize::new(N_INDIRECT_STAGES);
        let bin_headers = BufferSize::new(binning_wgs * 256);
        let n_paths_aligned = align_up(n_paths, 256);
        let paths = BufferSize::new(n_paths_aligned);

        // The following buffer sizes have been hand picked to accommodate the vello test scenes as
        // well as paris-30k. These should instead get derived from the scene layout using
        // reasonable heuristics.
        let bin_data = BufferSize::new(1 << 18);
        let tiles = BufferSize::new(1 << 21);
        let lines = BufferSize::new(1 << 21);
        let seg_counts = BufferSize::new(1 << 21);
        let segments = BufferSize::new(1 << 21);
        // 16 * 16 (1 << 8) is one blend spill, so this allows for 4096 spills.
        let blend_spill = BufferSize::new(1 << 20);
        let ptcl = BufferSize::new(1 << 23);
        let active_tile_ids = BufferSize::new(workgroups.fine.0 * workgroups.fine.1);
        Self {
            path_reduced,
            path_reduced2,
            path_reduced_scan,
            path_monoids,
            path_bboxes,
            draw_reduced,
            draw_monoids,
            info,
            clip_inps,
            clip_els,
            clip_bics,
            clip_bboxes,
            draw_bboxes,
            bump_alloc,
            indirect_count,
            lines,
            bin_headers,
            paths,
            bin_data,
            tiles,
            seg_counts,
            segments,
            blend_spill,
            ptcl,
            active_tile_ids,
        }
    }
}

impl BufferSizes {
    /// Pairs of (`element_count`, `element_stride`) for all buffers that go into the
    /// storage pool. Excludes `bump_alloc` and `indirect_count` (exempt from pooling).
    /// Used with `BufferPool::padded_size()` to compute total pool size.
    pub fn pool_allocs(&self) -> Vec<(usize, usize)> {
        vec![
            (self.path_reduced.len() as usize, size_of::<PathMonoid>()),
            (self.path_reduced2.len() as usize, size_of::<PathMonoid>()),
            (self.path_reduced_scan.len() as usize, size_of::<PathMonoid>()),
            (self.path_monoids.len() as usize, size_of::<PathMonoid>()),
            (self.path_bboxes.len() as usize, size_of::<PathBbox>()),
            (self.draw_reduced.len() as usize, size_of::<DrawMonoid>()),
            (self.draw_monoids.len() as usize, size_of::<DrawMonoid>()),
            (self.bin_data.len() as usize, size_of::<u32>()),
            (self.clip_inps.len() as usize, size_of::<Clip>()),
            (self.clip_els.len() as usize, size_of::<ClipElement>()),
            (self.clip_bics.len() as usize, size_of::<ClipBic>()),
            (self.clip_bboxes.len() as usize, size_of::<ClipBbox>()),
            (self.draw_bboxes.len() as usize, size_of::<DrawBbox>()),
            (self.bin_headers.len() as usize, size_of::<BinHeader>()),
            (self.paths.len() as usize, size_of::<Path>()),
            (self.tiles.len() as usize, size_of::<Tile>()),
            (self.segments.len() as usize, size_of::<PathSegment>()),
            (self.ptcl.len() as usize, size_of::<u32>()),
            (self.lines.len() as usize, size_of::<LineSoup>()),
            (self.seg_counts.len() as usize, size_of::<SegmentCount>()),
            (self.blend_spill.len() as usize, size_of::<u32>()),
            (self.active_tile_ids.len() as usize, size_of::<u32>()),
        ]
    }

    /// Create new buffer sizes with bump-allocated buffers scaled up to accommodate
    /// the actual usage reported by the GPU bump allocator.
    ///
    /// Each overflowed buffer is grown by 1.5x the actual usage (rounded up to a
    /// 256-element alignment boundary). This is less wasteful than the previous
    /// `next_power_of_two` which could double memory on every overflow.
    pub fn with_bump_estimates(&self, bump: &BumpAllocators) -> Self {
        fn grow(current: u32, needed: u32) -> u32 {
            if needed > current {
                // 1.5x headroom, rounded up to 256-element alignment.
                let grown = needed.saturating_add(needed / 2);
                (grown + 255) & !255
            } else {
                current
            }
        }
        // bin_data is shared: first `info.len()` elements are draw info (fixed),
        // remaining are binning data (bump-allocated). Only the binning portion grows.
        let bin_data_start = self.info.len();
        let current_binning = self.bin_data.len() - bin_data_start;
        let new_binning = grow(current_binning, bump.binning);
        // segments_needed: the coarse pass only allocates segments when it runs
        // (after flatten/path_count succeed). If it didn't run, bump.segments is 0.
        // Use seg_counts as a lower bound since each seg_count produces one segment.
        let segments_needed = bump.segments.max(bump.seg_counts);
        Self {
            lines: BufferSize::new(grow(self.lines.len(), bump.lines)),
            bin_data: BufferSize::new(bin_data_start + new_binning),
            tiles: BufferSize::new(grow(self.tiles.len(), bump.tile)),
            seg_counts: BufferSize::new(grow(self.seg_counts.len(), bump.seg_counts)),
            segments: BufferSize::new(grow(self.segments.len(), segments_needed)),
            blend_spill: BufferSize::new(grow(self.blend_spill.len(), bump.blend)),
            ptcl: BufferSize::new(grow(self.ptcl.len(), bump.ptcl)),
            ..*self
        }
    }
}

const fn align_up(len: u32, alignment: u32) -> u32 {
    len + (len.wrapping_neg() & (alignment - 1))
}
