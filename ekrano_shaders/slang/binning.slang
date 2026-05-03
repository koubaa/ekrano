// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// The binning stage.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint N_SLICE = 8;
static const uint N_SUBSLICE = 4;
static const float SX = 1.0 / float(N_TILE_X * TILE_WIDTH);
static const float SY = 1.0 / float(N_TILE_Y * TILE_HEIGHT);

groupshared uint sh_bitmaps[N_SLICE][N_TILE];
groupshared uint sh_count[N_SUBSLICE][N_TILE];
groupshared uint sh_chunk_offset[N_TILE];
groupshared uint sh_previous_failed;

// Slots: 0=config, 1=scene, 2=draw_monoids, 3=path_bbox_buf, 4=clip_bbox_buf, 5=intersected_bbox,
//       6=bump, 7=bin_data, 8=bin_header

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(BufRO<Config> config_buf, BufRO<uint> scene,
             BufRO<DrawMonoid> draw_monoids, BufRO<PathBbox> path_bbox_buf,
             BufRO<float4> clip_bbox_buf, Scattered<float4> intersected_bbox,
             Scattered<BumpAllocators> bump, Scattered<uint> bin_data, Scattered<BinHeader> bin_header,
             ThreadId global_id, GroupThreadId local_id) {
    Config config = config_buf[0];

    for (uint i = 0; i < N_SLICE; i++) {
        sh_bitmaps[i][local_id.x] = 0;
    }
    if (local_id.x == 0) {
        uint lines_val = bump[0].lines;
        sh_previous_failed = (lines_val > config.lines_size) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint failed = sh_previous_failed;
    if (failed != 0) {
        if (global_id.x == 0) {
            // Diagnostic: on STAGE_FLATTEN overflow, stash the GPU-observed
            // lines/lines_size into the unused binning/ptcl counters so the
            // CPU retry path can compare against its own values. If the CPU
            // thinks `config.lines_size` was 2M but the GPU reports 0, the
            // buffer write never reached the GPU (was a bindless-slot
            // aliasing bug). Uncomment to re-enable next time the flatten
            // retry cascade reappears.
            //   bump[0].binning = bump[0].lines;
            //   bump[0].ptcl = config.lines_size;
            InterlockedOr(bump[0].failed, STAGE_FLATTEN);
        }
        return;
    }

    uint element_ix = global_id.x;
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (element_ix < config.n_drawobj) {
        DrawMonoid draw_monoid = draw_monoids[element_ix];
        uint draw_tag = scene[config.drawtag_base + element_ix];
        float4 clip_bbox = Bbox2D.infinite().v;
        if (draw_monoid.clip_ix > 0) {
            uint clip_idx = min(draw_monoid.clip_ix - 1, config.n_clip - 1);
            clip_bbox = clip_bbox_buf[clip_idx];
        }
        float4 bbox;
        if (draw_tag == DRAWTAG_SET_BLEND_MODE) {
            // Per-draw blend state applies everywhere in the current clip (no path bbox).
            bbox = clip_bbox;
        } else {
            PathBbox path_bbox = path_bbox_buf[draw_monoid.path_ix];
            float4 pb = float4(float(path_bbox.x0), float(path_bbox.y0), float(path_bbox.x1), float(path_bbox.y1));
            bbox = bbox_intersect(clip_bbox, pb);
        }
        intersected_bbox[element_ix] = bbox;

        if (bbox.x < bbox.z && bbox.y < bbox.w) {
            x0 = int(floor(bbox.x * SX));
            y0 = int(floor(bbox.y * SY));
            x1 = int(ceil(bbox.z * SX));
            y1 = int(ceil(bbox.w * SY));
        }
    }
    int width_in_bins = int((config.width_in_tiles + N_TILE_X - 1) / N_TILE_X);
    int height_in_bins = int((config.height_in_tiles + N_TILE_Y - 1) / N_TILE_Y);
    x0 = clamp(x0, 0, width_in_bins);
    y0 = clamp(y0, 0, height_in_bins);
    x1 = clamp(x1, 0, width_in_bins);
    y1 = clamp(y1, 0, height_in_bins);
    if (x0 == x1)
        y1 = y0;

    int x = x0, y = y0;
    uint my_slice = local_id.x / 32;
    uint my_mask = 1 << (local_id.x & 31);
    while (y < y1) {
        int bin_ix = y * width_in_bins + x;
        if (bin_ix < int(N_TILE)) {
            InterlockedOr(sh_bitmaps[my_slice][bin_ix], my_mask);
        }
        x++;
        if (x == x1) {
            x = x0;
            y++;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    uint element_count = 0;
    for (uint i = 0; i < N_SUBSLICE; i++) {
        element_count += countbits(sh_bitmaps[i * 2][local_id.x]);
        uint element_count_lo = element_count;
        element_count += countbits(sh_bitmaps[i * 2 + 1][local_id.x]);
        uint element_count_hi = element_count;
        sh_count[i][local_id.x] = element_count_lo | (element_count_hi << 16);
    }
    uint chunk_offset; InterlockedAdd(bump[0].binning, element_count, chunk_offset);
    if (chunk_offset + element_count > config.binning_size) {
        chunk_offset = 0;
        InterlockedOr(bump[0].failed, STAGE_BINNING);
    }
    sh_chunk_offset[local_id.x] = chunk_offset;
    bin_header[global_id.x].element_count = element_count;
    bin_header[global_id.x].chunk_offset = chunk_offset;
    GroupMemoryBarrierWithGroupSync();

    x = x0;
    y = y0;
    while (y < y1) {
        int bin_ix = y * width_in_bins + x;
        if (bin_ix < int(N_TILE)) {
            uint out_mask = sh_bitmaps[my_slice][bin_ix];
            if ((out_mask & my_mask) != 0) {
                uint idx = countbits(out_mask & (my_mask - 1));
                if (my_slice > 0) {
                    uint count_ix = my_slice - 1;
                    uint count_packed = sh_count[count_ix / 2][bin_ix];
                    idx += (count_packed >> (16 * (count_ix & 1))) & 0xffff;
                }
                uint offset = config.bin_data_start + sh_chunk_offset[bin_ix];
                bin_data[offset + idx] = element_ix;
            }
        }
        x++;
        if (x == x1) {
            x = x0;
            y++;
        }
    }
}
