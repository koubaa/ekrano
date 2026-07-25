// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Pathtag scan1 - middle level for the large two-level monoid scan.
// Combines first-level `reduced` blocks using `reduced2` parent prefixes into
// `tag_monoids` (consumed as `reduced` by pathtag_scan_large).

import goldy_exp;
import ekrano_shared;

static const uint LG_WG_SIZE = 8;
static const uint WG_SIZE = 256;

// Slots: 0 = reduced, 1 = reduced2, 2 = tag_monoids
groupshared TagMonoid sh_parent[256];
groupshared TagMonoid sh_monoid[256];

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(BufRO<TagMonoid> reduced, BufRO<TagMonoid> reduced2, Scattered<TagMonoid> tag_monoids,
             ThreadId global_id, GroupThreadId local_id, GroupId wg_id) {

    TagMonoid agg = TagMonoid.identity();
    if (local_id.x < wg_id.x) {
        agg = reduced2[local_id.x];
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
    agg = reduced[ix];
    sh_monoid[local_id.x] = agg;
    for (uint i = 0; i < LG_WG_SIZE; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (local_id.x >= (1u << i)) {
            TagMonoid other = sh_monoid[local_id.x - (1u << i)];
            agg = other.combine(agg);
        }
        GroupMemoryBarrierWithGroupSync();
        sh_monoid[local_id.x] = agg;
    }
    GroupMemoryBarrierWithGroupSync();

    TagMonoid tm = sh_parent[0];
    if (local_id.x > 0) {
        tm = tm + sh_monoid[local_id.x - 1];
    }
    tag_monoids[ix] = tm;
}
