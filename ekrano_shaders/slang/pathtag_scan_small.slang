// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Pathtag scan - small (single-level) variant.

import goldy_exp;
import ekrano_shared;

static const uint LG_WG_SIZE = 8;
static const uint WG_SIZE = 256;

// Slots: 0 = config, 1 = scene, 2 = reduced, 3 = tag_monoids
groupshared TagMonoid sh_parent[256];
groupshared TagMonoid sh_monoid[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID, uint3 wg_id : SV_GroupID) {
    Config config = goldy_dyn_buf_ro<Config>(0)[0];
    ReadOnlyBuffer<uint> scene = goldy_dyn_buf_ro<uint>(1);
    ReadOnlyBuffer<TagMonoid> reduced = goldy_dyn_buf_ro<TagMonoid>(2);
    StorageBuffer<TagMonoid> tag_monoids = goldy_dyn_scattered<TagMonoid>(3);

    TagMonoid agg = tag_monoid_identity();
    if (local_id.x < wg_id.x) {
        agg = reduced[local_id.x];
    }
    sh_parent[local_id.x] = agg;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x + (1u << i) < WG_SIZE)
            agg = agg.combine(sh_parent[local_id.x + (1u << i)]);
        GroupMemoryBarrierWithGroupSync();
        sh_parent[local_id.x] = agg;
    }

    uint ix = global_id.x;
    uint tag_word = scene[config.pathtag_base + ix];
    TagMonoid agg_part = reduce_tag(tag_word);
    sh_monoid[local_id.x] = agg_part;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x >= (1u << i)) {
            TagMonoid other = sh_monoid[local_id.x - (1u << i)];
            agg_part = other.combine(agg_part);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_monoid[local_id.x] = agg_part;
    }
    GroupMemoryBarrierWithGroupSync();

    TagMonoid tm = sh_parent[0];
    if (local_id.x > 0) {
        tm = tm + sh_monoid[local_id.x - 1];
    }
    tag_monoids[ix] = tm;
}
