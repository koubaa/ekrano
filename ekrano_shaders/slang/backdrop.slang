// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Prefix sum for backdrops (non-atomic version).
// Each workgroup computes the inclusive prefix sum of the backdrops in one row of tiles.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 64;
static const uint LG_WG_SIZE = 6;

groupshared int sh_backdrop[64];

// Slots: 0=config, 1=tiles

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    StorageBuffer<Tile> tiles = goldy_dyn_scattered<Tile>(1);

    uint width_in_tiles = config.width_in_tiles;
    uint ix = wg_id.x * width_in_tiles + local_id.x;
    int backdrop = 0;
    if (local_id.x < width_in_tiles) {
        backdrop = tiles[ix].backdrop;
    }
    sh_backdrop[local_id.x] = backdrop;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x >= (1 << i)) {
            backdrop += sh_backdrop[local_id.x - (1 << i)];
        }
        GroupMemoryBarrierWithGroupSync();
        sh_backdrop[local_id.x] = backdrop;
    }
    if (local_id.x < width_in_tiles) {
        tiles[ix].backdrop = backdrop;
    }
}
