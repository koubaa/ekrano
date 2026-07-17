// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Scheme-backend variant: set up dispatch size for the fine stage from the
// compacted active-tile count written by coarse.

import goldy_exp;
import ekrano_shared;

// Slots: 0 = bump (BumpAllocators), 1 = shape (DispatchShape — one element)

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<BumpAllocators> bump, Scattered<DispatchShape> shape) {
    if (bump[0].failed != 0) {
        shape[0].x = 0;
    } else {
        shape[0].x = bump[0].active_tiles;
    }
    shape[0].y = 1;
    shape[0].z = 1;
}
