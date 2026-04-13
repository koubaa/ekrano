// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Second stage of reduction for pathtag monoid scan (large case).

import goldy_exp;
import ekrano_shared;

static const uint LG_WG_SIZE = 8;
static const uint WG_SIZE = 256;

groupshared TagMonoid sh_scratch[WG_SIZE];

// Slots: 0 = reduced_in, 1 = reduced
[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID) {
    ReadOnlyBuffer<TagMonoid> reduced_in = goldy_dyn_buf_ro<TagMonoid>(0);
    StorageBuffer<TagMonoid> reduced = goldy_dyn_scattered<TagMonoid>(1);

    uint ix = global_id.x;
    TagMonoid agg = reduced_in[ix];
    sh_scratch[local_id.x] = agg;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x + (1u << i) < WG_SIZE)
            agg = agg.combine(sh_scratch[local_id.x + (1u << i)]);
        GroupMemoryBarrierWithGroupSync();
        sh_scratch[local_id.x] = agg;
    }
    if (local_id.x == 0) {
        reduced[ix >> LG_WG_SIZE] = agg;
    }
}
