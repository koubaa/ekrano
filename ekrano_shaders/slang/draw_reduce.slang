// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

import goldy_exp;
import ekrano_shared;

static const uint LG_WG_SIZE = 8;
static const uint WG_SIZE = 256;

// Slots: 0 = config, 1 = scene, 2 = reduced
groupshared DrawMonoid sh_scratch[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_broadcast<Config>(0);
    StorageBuffer<uint> scene = goldy_dyn_scattered<uint>(1);
    StorageBuffer<DrawMonoid> reduced = goldy_dyn_scattered<DrawMonoid>(2);

    uint num_blocks_total = (config.n_drawobj + (WG_SIZE - 1)) / WG_SIZE;
    uint n_blocks_base = num_blocks_total / WG_SIZE;
    uint remainder = num_blocks_total % WG_SIZE;
    uint first_block = n_blocks_base * wg_id.x + min(wg_id.x, remainder);
    uint n_blocks = n_blocks_base + (wg_id.x < remainder ? 1u : 0u);
    uint block_index = first_block * WG_SIZE + local_id.x;

    DrawMonoid agg = DrawMonoid.identity();
    for (uint i = 0; i < n_blocks; i++) {
        uint tag_word = read_draw_tag_from_scene(scene, config, block_index);
        agg = agg + DrawTagScanner.map(tag_word);
        block_index += WG_SIZE;
    }
    agg = workgroup_reduce(agg, local_id.x, sh_scratch);
    if (local_id.x == 0) {
        reduced[wg_id.x] = agg;
    }
}
