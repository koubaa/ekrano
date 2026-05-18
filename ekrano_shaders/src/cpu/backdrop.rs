// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

use ekrano_encoding::{BumpAllocators, ConfigUniform, Path, Tile};

use super::CpuBinding;
use super::util::morton_encode_2d;

fn backdrop_main(config: &ConfigUniform, _: &BumpAllocators, paths: &[Path], tiles: &mut [Tile]) {
    for drawobj_ix in 0..config.layout.n_draw_objects {
        let path = paths[drawobj_ix as usize];
        let width = path.bbox[2] - path.bbox[0];
        let height = path.bbox[3] - path.bbox[1];
        let base = path.tiles;
        for y in 0..height {
            let mut sum = 0;
            let tile_ix0 = (base + morton_encode_2d(0, y)) as usize;
            sum += tiles[tile_ix0].backdrop;
            for x in 1..width {
                let tile_ix = (base + morton_encode_2d(x, y)) as usize;
                sum += tiles[tile_ix].backdrop;
                tiles[tile_ix].backdrop = sum;
            }
        }
    }
}

pub fn backdrop(_n_wg: u32, resources: &[CpuBinding<'_>]) {
    let config = resources[0].as_typed();
    let bump = resources[1].as_typed();
    let paths = resources[2].as_slice();
    let mut tiles = resources[3].as_slice_mut();
    backdrop_main(&config, &bump, &paths, &mut tiles);
}
