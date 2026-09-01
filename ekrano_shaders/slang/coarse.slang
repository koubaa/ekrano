// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// The coarse rasterization stage.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint N_SLICE = 8;
static const uint LG_WG_SIZE = 8;
static const uint LG_N_TILE = 8;
static const uint BLEND_CLIP = (128 << 8) | 3;

groupshared Interlocked<uint> sh_bitmaps[N_SLICE][N_TILE];
groupshared uint sh_part_count[N_TILE];   // BinHeader.element_count inclusive prefix sum (partition phase)
groupshared uint sh_part_offsets[N_TILE]; // BinHeader.chunk_offset (partition phase)
groupshared uint sh_drawobj_ix[N_TILE];
groupshared uint sh_tile_bbox_xy[N_TILE]; // path.bbox.x | (path.bbox.y << 16) for Morton index (scatter/render phase)
groupshared uint sh_tile_width[N_TILE];   // tile x-span (scatter phase)
groupshared uint sh_tile_x0y0[N_TILE];
groupshared uint sh_tile_count[N_TILE];
groupshared uint sh_tile_base[N_TILE];

void alloc_cmd(inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump,
    Config config, uint size) {
    // Issue #46 suggested per-wave PTCL bump batching via WavePrefixSum/WaveActiveSum.
    // alloc_cmd runs under divergent control flow inside the bitmap-driven PTCL writer — inactive wave lanes break the assumptions for Wave intrinsics, so we keep one atomic per overflow event on each lane that reaches here.
    if (cmd_offset + size >= cmd_limit) {
        uint ptcl_dyn_start = config.width_in_tiles * config.height_in_tiles * PTCL_INITIAL_ALLOC;
        uint new_cmd = InterlockedAdd(bump[0].ptcl, PTCL_INCREMENT); new_cmd += ptcl_dyn_start;
        if (new_cmd + PTCL_INCREMENT > config.ptcl_size) {
            new_cmd = 0;
            InterlockedOr(bump[0].failed, STAGE_COARSE);
        }
        ptcl[cmd_offset] = CMD_JUMP;
        ptcl[cmd_offset + 1] = new_cmd;
        cmd_offset = new_cmd;
        cmd_limit = cmd_offset + (PTCL_INCREMENT - PTCL_HEADROOM);
    }
}

void write_path(uint tile_ix, uint draw_flags,
    inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump,
    StorageBuffer<TileRead> tiles, Config config) {
    uint n_segs = tiles[tile_ix].segment_count_or_ix;
    if (n_segs != 0) {
        uint seg_ix = InterlockedAdd(bump[0].segments, n_segs);
        tiles[tile_ix].segment_count_or_ix = ~seg_ix;
        alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 4);
        ptcl[cmd_offset] = CMD_FILL;
        uint even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0 ? 1u : 0u;
        ptcl[cmd_offset + 1] = (n_segs << 1) | even_odd;
        ptcl[cmd_offset + 2] = seg_ix;
        ptcl[cmd_offset + 3] = uint(tiles[tile_ix].backdrop);
        cmd_offset += 4;
    } else {
        alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 1);
        ptcl[cmd_offset] = CMD_SOLID;
        cmd_offset += 1;
    }
}

void write_color(uint rgba_color, inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 2);
    ptcl[cmd_offset] = CMD_COLOR;
    ptcl[cmd_offset + 1] = rgba_color;
    cmd_offset += 2;
}

void write_grad(uint ty, uint index, uint info_offset,
    inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 3);
    ptcl[cmd_offset] = ty;
    ptcl[cmd_offset + 1] = index;
    ptcl[cmd_offset + 2] = info_offset;
    cmd_offset += 3;
}

void write_image(uint cmd, uint info_offset, inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 2);
    ptcl[cmd_offset] = cmd;
    ptcl[cmd_offset + 1] = info_offset;
    cmd_offset += 2;
}

void write_begin_clip(inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 1);
    ptcl[cmd_offset] = CMD_BEGIN_CLIP;
    cmd_offset += 1;
}

void write_end_clip(CmdEndClip end_clip, inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 3);
    ptcl[cmd_offset] = CMD_END_CLIP;
    ptcl[cmd_offset + 1] = end_clip.blend;
    ptcl[cmd_offset + 2] = asuint(end_clip.alpha);
    cmd_offset += 3;
}

void write_end_clip_filter(CmdEndClip end_clip, uint layer_idx, inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 4);
    ptcl[cmd_offset] = CMD_END_CLIP_FILTER;
    ptcl[cmd_offset + 1] = end_clip.blend;
    ptcl[cmd_offset + 2] = asuint(end_clip.alpha);
    ptcl[cmd_offset + 3] = layer_idx;
    cmd_offset += 4;
}

void write_blurred_rounded_rect(uint rgba_color, uint info_offset,
    inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 3);
    ptcl[cmd_offset] = CMD_BLUR_RECT;
    ptcl[cmd_offset + 1] = info_offset;
    ptcl[cmd_offset + 2] = rgba_color;
    cmd_offset += 3;
}

void write_set_blend_mode(uint blend_packed, inout uint cmd_offset, inout uint cmd_limit,
    StorageBuffer<uint> ptcl, StorageBuffer<BumpAllocators> bump, Config config) {
    alloc_cmd(cmd_offset, cmd_limit, ptcl, bump, config, 2);
    ptcl[cmd_offset] = CMD_SET_BLEND_MODE;
    ptcl[cmd_offset + 1] = blend_packed;
    cmd_offset += 2;
}

// Slots: 0=config, 1=scene, 2=draw_monoids, 3=bin_headers, 4=info_bin_data, 5=paths, 6=tiles, 7=bump, 8=ptcl

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(BufRO<Config> config_buf, BufRO<uint> scene,
             BufRO<DrawMonoid> draw_monoids, BufRO<BinHeader> bin_headers,
             BufRO<uint> info_bin_data, BufRO<Path> paths,
             Scattered<TileRead> tiles, Scattered<BumpAllocators> bump, Scattered<uint> ptcl,
             GroupThreadId local_id, GroupId wg_id) {
    Config config = config_buf[0];

    if (local_id.x == 0) {
        uint failed = InterlockedLoad(bump[0].failed) & (STAGE_BINNING | STAGE_TILE_ALLOC | STAGE_FLATTEN);
        if (InterlockedLoad(bump[0].seg_counts) > config.seg_counts_size)
            failed |= STAGE_PATH_COUNT;
        sh_part_count[0] = failed;
    }
    GroupMemoryBarrierWithGroupSync();
    uint failed = sh_part_count[0];
    // Do not early-return before later workgroup barriers (lavapipe deadlock).
    if (failed != 0) {
        if (wg_id.x == 0 && local_id.x == 0)
            InterlockedOr(bump[0].failed, failed);
    }

    uint width_in_bins = (config.width_in_tiles + N_TILE_X - 1) / N_TILE_X;
    uint bin_ix = width_in_bins * wg_id.y + wg_id.x;
    // Skip when bin_ix exceeds bin_headers capacity (binning_wgs * 256). See vello #680.
    // `bin_ix` is uniform across the workgroup, so this return is WG-uniform.
    if (failed != 0 || bin_ix >= 256)
        return;
    uint n_partitions = (config.n_drawobj + N_TILE - 1) / N_TILE;

    uint bin_tile_x = N_TILE_X * wg_id.x;
    uint bin_tile_y = N_TILE_Y * wg_id.y;

    uint tile_x = local_id.x % N_TILE_X;
    uint tile_y = local_id.x / N_TILE_X;
    uint this_tile_ix = (bin_tile_y + tile_y) * config.width_in_tiles + bin_tile_x + tile_x;
    uint cmd_offset = this_tile_ix * PTCL_INITIAL_ALLOC;
    uint cmd_limit = cmd_offset + (PTCL_INITIAL_ALLOC - PTCL_HEADROOM);

    uint clip_zero_depth = 0;
    uint clip_depth = 0;
    uint partition_ix = 0;
    uint rd_ix = 0;
    uint wr_ix = 0;
    uint part_start_ix = 0;
    uint ready_ix = 0;
    uint render_blend_depth = 0;
    uint max_blend_depth = 0;

    uint blend_offset = cmd_offset;
    cmd_offset += 1;

    // Issue #46: bitmaps zero once before first iteration; thereafter cleared at loop bottom
    // (after bitmap reads) so we skip one barrier per iteration vs clearing at the top.
    for (uint i = 0; i < N_SLICE; i++)
        InterlockedStore(sh_bitmaps[i][local_id.x], 0);
    GroupMemoryBarrierWithGroupSync();

    while (true) {
        while (true) {
            if (ready_ix == wr_ix && partition_ix < n_partitions) {
                part_start_ix = ready_ix;
                uint count = 0;
                if (partition_ix + local_id.x < n_partitions) {
                    uint in_ix = (partition_ix + local_id.x) * N_TILE + bin_ix;
                    BinHeader bh = bin_headers[in_ix];
                    count = bh.element_count;
                sh_part_offsets[local_id.x] = bh.chunk_offset;
                }
                sh_part_count[local_id.x] = count;
                for (uint scan_i = 0; scan_i < LG_WG_SIZE; scan_i++) {
                    GroupMemoryBarrierWithGroupSync();
                    if (local_id.x >= (1u << scan_i)) {
                        count = sh_part_count[local_id.x - (1u << scan_i)] + count;
                    }
                    GroupMemoryBarrierWithGroupSync();
                    sh_part_count[local_id.x] = count;
                }
                GroupMemoryBarrierWithGroupSync();
                sh_part_count[local_id.x] = part_start_ix + count;
                GroupMemoryBarrierWithGroupSync();
                ready_ix = sh_part_count[WG_SIZE - 1];
                partition_ix += WG_SIZE;
            }
            uint ix = rd_ix + local_id.x;
            if (ix >= wr_ix && ix < ready_ix) {
                uint part_ix = workgroup_upper_bound<LG_WG_SIZE>(ix, sh_part_count);
                ix -= (part_ix > 0) ? sh_part_count[part_ix - 1] : part_start_ix;
                uint offset = config.bin_data_start + sh_part_offsets[part_ix];
                sh_drawobj_ix[local_id.x] = info_bin_data[offset + ix];
            }
            wr_ix = min(rd_ix + N_TILE, ready_ix);
            if (wr_ix - rd_ix >= N_TILE || (wr_ix >= ready_ix && partition_ix >= n_partitions))
                break;
            GroupMemoryBarrierWithGroupSync();
        }

        uint tag = DRAWTAG_NOP;
        uint drawobj_ix = 0;
        if (local_id.x + rd_ix < wr_ix) {
            drawobj_ix = sh_drawobj_ix[local_id.x];
            tag = scene[config.drawtag_base + drawobj_ix];
        }
        uint tile_count = 0;
        if (tag != DRAWTAG_NOP) {
            uint path_ix = draw_monoids[drawobj_ix].path_ix;
            Path path = paths[path_ix];
            int dx = int(path.bbox.x) - int(bin_tile_x);
            int dy = int(path.bbox.y) - int(bin_tile_y);
            int x0 = clamp(dx, 0, int(N_TILE_X));
            int y0 = clamp(dy, 0, int(N_TILE_Y));
            int x1 = clamp(int(path.bbox.z) - int(bin_tile_x), 0, int(N_TILE_X));
            int y1 = clamp(int(path.bbox.w) - int(bin_tile_y), 0, int(N_TILE_Y));
            sh_tile_width[local_id.x] = uint(x1 - x0);
            sh_tile_x0y0[local_id.x] = uint(x0) | (uint(y0) << 16);
            tile_count = uint(x1 - x0) * uint(y1 - y0);
            // Morton layout: sh_tile_base holds path.tiles (flat base offset);
            // sh_tile_bbox_xy holds path.bbox.xy packed so the inner
            // loop can convert bin-local tile_x/tile_y to path-relative coordinates.
            sh_tile_base[local_id.x] = path.tiles;
            sh_tile_bbox_xy[local_id.x] = path.bbox.x | (path.bbox.y << 16);
        }

        sh_tile_count[local_id.x] = tile_count;
        for (uint i = 0; i < LG_WG_SIZE; i++) {
            GroupMemoryBarrierWithGroupSync();
            if (local_id.x >= (1u << i)) {
                uint other = sh_tile_count[local_id.x - (1u << i)];
                tile_count = other + tile_count;
            }
            GroupMemoryBarrierWithGroupSync();
            sh_tile_count[local_id.x] = tile_count;
        }
        GroupMemoryBarrierWithGroupSync();
        uint total_tile_count = sh_tile_count[N_TILE - 1];

        for (uint ix = local_id.x; ix < total_tile_count; ix += N_TILE) {
            uint el_ix = workgroup_upper_bound<LG_N_TILE>(ix, sh_tile_count);
            drawobj_ix = sh_drawobj_ix[el_ix];
            tag = scene[config.drawtag_base + drawobj_ix];
            uint seq_ix = ix - ((el_ix > 0) ? sh_tile_count[el_ix - 1] : 0);
            uint width = sh_tile_width[el_ix];
            uint x0y0 = sh_tile_x0y0[el_ix];
            uint bbox_xy = sh_tile_bbox_xy[el_ix];
            uint x = (x0y0 & 0xffff) + seq_ix % width;
            uint y = (x0y0 >> 16) + seq_ix / width;
            // Convert bin-local (x, y) to path-relative coordinates for Morton index.
            uint x_path = uint(int(bin_tile_x) + int(x) - int(bbox_xy & 0xffffu));
            uint y_path = uint(int(bin_tile_y) + int(y) - int(bbox_xy >> 16u));
            uint tile_ix = sh_tile_base[el_ix] + morton_encode_2d(x_path, y_path);
            bool is_clip = (tag & 1) != 0;
            bool is_blend = false;
            if (is_clip) {
                uint scene_offset = draw_monoids[drawobj_ix].scene_offset;
                uint dd = config.drawdata_base + scene_offset;
                uint blend = scene[dd];
                is_blend = (blend != BLEND_CLIP);
            }
            uint di = draw_monoids[drawobj_ix].info_offset;
            uint draw_flags = info_bin_data[di];
            bool even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0;
            uint n_segs = tiles[tile_ix].segment_count_or_ix;
            int tile_backdrop = tiles[tile_ix].backdrop;
            int backdrop_val = even_odd ? (abs(tile_backdrop) & 1) : tile_backdrop;
            bool backdrop_clear = (backdrop_val == 0);
            bool include_tile = (tag == DRAWTAG_SET_BLEND_MODE) || (n_segs != 0) ||
                (backdrop_clear == is_clip) || is_blend;
            if (include_tile &&
                (bin_tile_x + x) < config.width_in_tiles &&
                (bin_tile_y + y) < config.height_in_tiles)
            {
                uint el_slice = el_ix / 32;
                uint el_mask = 1 << (el_ix & 31);
                InterlockedOr(sh_bitmaps[el_slice][y * N_TILE_X + x], el_mask);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        uint slice_ix = 0;
        uint bitmap = InterlockedLoad(sh_bitmaps[0][local_id.x]);
        while (true) {
            if (bitmap == 0) {
                slice_ix++;
                if (slice_ix == N_SLICE)
                    break;
                bitmap = InterlockedLoad(sh_bitmaps[slice_ix][local_id.x]);
                if (bitmap == 0)
                    continue;
            }
            uint el_ix = slice_ix * 32 + firstbitlow(bitmap);
            drawobj_ix = sh_drawobj_ix[el_ix];
            bitmap &= bitmap - 1;
            uint drawtag = scene[config.drawtag_base + drawobj_ix];
            DrawMonoid dm = draw_monoids[drawobj_ix];
            uint dd = config.drawdata_base + dm.scene_offset;
            uint di = dm.info_offset;
            uint draw_flags = info_bin_data[di];

            if (clip_zero_depth == 0) {
                // Convert this thread's bin-local tile coordinates to path-relative
                // for Morton index computation.
                uint bbox_xy = sh_tile_bbox_xy[el_ix];
                uint px = uint(int(bin_tile_x) + int(tile_x) - int(bbox_xy & 0xffffu));
                uint py = uint(int(bin_tile_y) + int(tile_y) - int(bbox_xy >> 16u));
                uint tile_ix = sh_tile_base[el_ix] + morton_encode_2d(px, py);
                if (drawtag == DRAWTAG_FILL_COLOR) {
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    write_color(scene[dd], cmd_offset, cmd_limit, ptcl, bump, config);
                } else if (drawtag == DRAWTAG_BLURRED_ROUNDED_RECT) {
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    write_blurred_rounded_rect(scene[dd], di + 1, cmd_offset, cmd_limit, ptcl, bump, config);
                } else if (drawtag == DRAWTAG_FILL_LIN_GRADIENT) {
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    write_grad(CMD_LIN_GRAD, scene[dd], di + 1, cmd_offset, cmd_limit, ptcl, bump, config);
                } else if (drawtag == DRAWTAG_FILL_RAD_GRADIENT) {
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    write_grad(CMD_RAD_GRAD, scene[dd], di + 1, cmd_offset, cmd_limit, ptcl, bump, config);
                } else if (drawtag == DRAWTAG_FILL_SWEEP_GRADIENT) {
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    write_grad(CMD_SWEEP_GRAD, scene[dd], di + 1, cmd_offset, cmd_limit, ptcl, bump, config);
                } else if (drawtag == DRAWTAG_FILL_IMAGE || drawtag == DRAWTAG_FILL_IMAGE_TINTED) {
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    uint cmd = drawtag == DRAWTAG_FILL_IMAGE_TINTED ? CMD_IMAGE_TINTED : CMD_IMAGE;
                    write_image(cmd, di + 1, cmd_offset, cmd_limit, ptcl, bump, config);
                } else if (drawtag == DRAWTAG_BEGIN_CLIP) {
                    bool even_odd = (draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0;
                    int tile_backdrop = tiles[tile_ix].backdrop;
                    int backdrop_val = even_odd ? (abs(tile_backdrop) & 1) : tile_backdrop;
                    bool backdrop_clear = (backdrop_val == 0);
                    if (tiles[tile_ix].segment_count_or_ix == 0 && backdrop_clear)
                        clip_zero_depth = clip_depth + 1;
                    else {
                        write_begin_clip(cmd_offset, cmd_limit, ptcl, bump, config);
                        render_blend_depth++;
                        max_blend_depth = max(max_blend_depth, render_blend_depth);
                    }
                    clip_depth++;
                } else if (drawtag == DRAWTAG_END_CLIP) {
                    clip_depth--;
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    CmdEndClip end_clip;
                    end_clip.blend = scene[dd];
                    end_clip.alpha = asfloat(scene[dd + 1]);
                    write_end_clip(end_clip, cmd_offset, cmd_limit, ptcl, bump, config);
                    render_blend_depth--;
                } else if (drawtag == DRAWTAG_END_CLIP_FILTER) {
                    clip_depth--;
                    write_path(tile_ix, draw_flags, cmd_offset, cmd_limit, ptcl, bump, tiles, config);
                    CmdEndClip end_clip;
                    end_clip.blend = scene[dd];
                    end_clip.alpha = asfloat(scene[dd + 1]);
                    // `layer_idx` lives in the matching `BEGIN_CLIP`'s scene data because
                    // `clip_leaf.slang` overwrites this `END_CLIP_FILTER`'s `scene_offset`
                    // with the `BEGIN_CLIP`'s (same override that lets `END_CLIP` read the
                    // clip's blend/alpha). `DrawTag::BEGIN_CLIP`'s scene slot reserves
                    // `scene[dd + 2]` for the filter layer index, backfilled in
                    // `encode_end_clip`. 0 means "no filter" (should not hit this branch).
                    uint layer_idx = scene[dd + 2];
                    // Pack "is nested inside another filter layer" into the top bit.
                    // Fine uses this to composite the layer result back into rgba so that
                    // the enclosing filter layer can capture it in its own snapshot.
                    if (render_blend_depth > 1u)
                        layer_idx |= 0x80000000u;
                    write_end_clip_filter(end_clip, layer_idx, cmd_offset, cmd_limit, ptcl, bump, config);
                    render_blend_depth--;
                } else if (drawtag == DRAWTAG_SET_BLEND_MODE) {
                    write_set_blend_mode(scene[dd], cmd_offset, cmd_limit, ptcl, bump, config);
                }
            } else {
                if (drawtag == DRAWTAG_BEGIN_CLIP)
                    clip_depth++;
                else if (drawtag == DRAWTAG_SET_BLEND_MODE) {
                } else if (drawtag == DRAWTAG_END_CLIP || drawtag == DRAWTAG_END_CLIP_FILTER) {
                    if (clip_depth == clip_zero_depth)
                        clip_zero_depth = 0;
                    clip_depth--;
                }
            }
        }

        rd_ix += N_TILE;
        if (rd_ix >= ready_ix && partition_ix >= n_partitions)
            break;
        GroupMemoryBarrierWithGroupSync();
        for (uint i = 0; i < N_SLICE; i++)
            InterlockedStore(sh_bitmaps[i][local_id.x], 0);
    }

    if (bin_tile_x + tile_x < config.width_in_tiles && bin_tile_y + tile_y < config.height_in_tiles) {
        ptcl[cmd_offset] = CMD_END;
        uint blend_ix = 0;
        if (max_blend_depth > BLEND_STACK_SPLIT) {
            uint scratch_size = (max_blend_depth - BLEND_STACK_SPLIT) * TILE_WIDTH * TILE_HEIGHT;
            blend_ix = InterlockedAdd(bump[0].blend, scratch_size);
            if (blend_ix + scratch_size > config.blend_size)
                InterlockedOr(bump[0].failed, STAGE_COARSE);
        }
        ptcl[blend_offset] = blend_ix;
    }
}
