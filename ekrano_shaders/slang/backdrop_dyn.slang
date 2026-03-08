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

// Slots: 0=config, 1=bump, 2=paths, 3=tiles

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    RWStructuredBuffer<BumpAllocators> bump = goldy_dyn_scattered<BumpAllocators>(1);
    RWStructuredBuffer<Path> paths = goldy_dyn_buf_ro<Path>(2);
    RWStructuredBuffer<Tile> tiles = goldy_dyn_scattered<Tile>(3);

    if (local_id.x == 0) {
        sh_row_count[0] = bump[0].failed;
    }
    GroupMemoryBarrierWithGroupSync();
    uint failed = sh_row_count[0];
    if (failed != 0)
        return;

    uint drawobj_ix = global_id.x;
    uint row_count = 0;
    if (drawobj_ix < config.n_drawobj) {
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
        if (local_id.x >= (1 << i)) {
            row_count += sh_row_count[local_id.x - (1 << i)];
        }
        GroupMemoryBarrierWithGroupSync();
        sh_row_count[local_id.x] = row_count;
    }
    GroupMemoryBarrierWithGroupSync();

    uint total_rows = sh_row_count[WG_SIZE - 1];
    for (uint row = local_id.x; row < total_rows; row += WG_SIZE) {
        uint el_ix = 0;
        for (uint i = 0; i < LG_WG_SIZE; i++) {
            uint probe = el_ix + ((WG_SIZE / 2) >> i);
            if (row >= sh_row_count[probe - 1])
                el_ix = probe;
        }
        uint width = sh_row_width[el_ix];
        if (width > 0) {
            uint seq_ix = row - (el_ix > 0 ? sh_row_count[el_ix - 1] : 0);
            uint tile_ix = sh_offset[el_ix] + seq_ix * width;
            int sum = tiles[tile_ix].backdrop;
            for (uint x = 1; x < width; x++) {
                tile_ix++;
                sum += tiles[tile_ix].backdrop;
                Tile t = tiles[tile_ix];
                t.backdrop = sum;
                tiles[tile_ix] = t;
            }
        }
    }
}
