// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Prefix sum for dynamically allocated backdrops.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;
static const uint LG_WG_SIZE = 8;

groupshared uint sh_row_width[256];
groupshared uint sh_row_count[256];
groupshared uint sh_offset[256];
groupshared uint sh_failed;

// Slots: 0=config, 1=bump, 2=paths, 3=tiles

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(BufRO<Config> config_buf, Scattered<BumpAllocators> bump,
             BufRO<Path> paths, Scattered<Tile> tiles,
             ThreadId global_id, GroupThreadId local_id) {
    Config config = config_buf[0];

    // Broadcast bump.failed without an early `return`. Returning from only some
    // threads before GroupMemoryBarrierWithGroupSync deadlocks the workgroup on
    // lavapipe (observed: hillis-steele scan never completes; interior tiles keep
    // raw path_count backdrops and coarse skips solid fills).
    if (local_id.x == 0) {
        sh_failed = bump[0].failed;
    }
    GroupMemoryBarrierWithGroupSync();
    uint failed = sh_failed;

    uint drawobj_ix = global_id.x;
    uint row_count = 0;
    if (failed == 0 && drawobj_ix < config.n_drawobj) {
        Path path = paths[drawobj_ix];
        sh_row_width[local_id.x] = path.bbox.z - path.bbox.x;
        row_count = path.bbox.w - path.bbox.y;
        sh_offset[local_id.x] = path.tiles;
    } else {
        sh_row_width[local_id.x] = 0;
    }
    sh_row_count[local_id.x] = row_count;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x >= (1u << i)) {
            uint other = sh_row_count[local_id.x - (1u << i)];
            row_count = other + row_count;
        }
        GroupMemoryBarrierWithGroupSync();
        sh_row_count[local_id.x] = row_count;
    }
    GroupMemoryBarrierWithGroupSync();

    if (failed != 0) {
        return;
    }

    uint total_rows = sh_row_count[WG_SIZE - 1];
    for (uint row = local_id.x; row < total_rows; row += WG_SIZE) {
        uint el_ix = workgroup_upper_bound<LG_WG_SIZE>(row, sh_row_count);
        uint width = sh_row_width[el_ix];
        if (width > 0) {
            uint seq_ix = row - (el_ix > 0 ? sh_row_count[el_ix - 1] : 0);
            uint path_base = sh_offset[el_ix];
            uint row_y = seq_ix;  // row index within this path's bbox
            int sum = tiles[path_base + morton_encode_2d(0u, row_y)].backdrop;
            for (uint x = 1; x < width; x++) {
                uint tile_ix = path_base + morton_encode_2d(x, row_y);
                sum += tiles[tile_ix].backdrop;
                Tile t = tiles[tile_ix];
                t.backdrop = sum;
                tiles[tile_ix] = t;
            }
        }
    }
}
