// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Scheme-backend variant: set up dispatch size for the path count stage.
//
// Unlike the graph-backend path_count_setup.slang which writes into a shared
// IndirectCount[] array, this variant writes directly into a single
// DispatchShape element — the per-stage buffer acquired by the scheme renderer.

import goldy_exp;
import ekrano_shared;

static const uint WG_SIZE = 256;

// Slots: 0 = bump (BumpAllocators), 1 = shape (DispatchShape — one element)

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<BumpAllocators> bump, Scattered<DispatchShape> shape) {
    if (bump[0].failed != 0) {
        shape[0].x = 0;
    } else {
        uint lines = bump[0].lines;
        shape[0].x = (lines + (WG_SIZE - 1)) / WG_SIZE;
    }
    shape[0].y = 1;
    shape[0].z = 1;
}
