// Copyright 2022 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense
//
// Clear path bboxes to sentinel values. Leaf shader.

import goldy_exp;
import ekrano_shared;

// Slots: 0 = config (uniform), 1 = path_bboxes (storage rw)

[goldy_compute]
[numthreads(256, 1, 1)]
void cs_main(BufRO<Config> config_buf, Scattered<PathBbox> path_bboxes,
             ThreadId global_id) {
    Config config = config_buf[0];

    uint ix = global_id.x;
    if (ix < config.n_path) {
        path_bboxes[ix].x0 = 0x7fffffff;
        path_bboxes[ix].y0 = 0x7fffffff;
        path_bboxes[ix].x1 = -0x80000000;
        path_bboxes[ix].y1 = -0x80000000;
    }
}
