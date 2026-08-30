// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Set up dispatch size for path count stage. Leaf shader.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;

// Slots: 0 = bump (BumpAllocators), 1 = indirect (IndirectCount[N_INDIRECT_STAGES])

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<BumpAllocators> bump, Scattered<IndirectCount> indirect) {
    if (InterlockedLoad(bump[0].failed) != 0) {
        indirect[INDIRECT_STAGE_PATH_COUNT].count_x = 0;
    } else {
        uint lines = InterlockedLoad(bump[0].lines);
        indirect[INDIRECT_STAGE_PATH_COUNT].count_x = (lines + (WG_SIZE - 1)) / WG_SIZE;
    }
    indirect[INDIRECT_STAGE_PATH_COUNT].count_y = 1;
    indirect[INDIRECT_STAGE_PATH_COUNT].count_z = 1;
}
