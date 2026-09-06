// Copyright 2023 the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Slang shader sources for the Goldy backend.
//!
//! These are used when building with the `goldy` feature.

use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

macro_rules! include_slang {
    ($name:ident, $file:literal) => {
        pub const $name: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/slang/", $file));
    };
}

include_slang!(EKRANO_SHARED, "ekrano_shared.slang");
include_slang!(BBOX_CLEAR, "bbox_clear.slang");
include_slang!(PIPELINE_SETUP, "pipeline_setup.slang");
include_slang!(PATH_COUNT_SETUP, "path_count_setup.slang");
include_slang!(PATH_COUNT_SETUP_SCHEME, "path_count_setup_scheme.slang");
include_slang!(PATH_TILING_SETUP, "path_tiling_setup.slang");
include_slang!(PATH_TILING_SETUP_SCHEME, "path_tiling_setup_scheme.slang");
include_slang!(PATHTAG_REDUCE, "pathtag_reduce.slang");
include_slang!(PATHTAG_REDUCE2, "pathtag_reduce2.slang");
include_slang!(PATHTAG_SCAN1, "pathtag_scan1.slang");
include_slang!(PATHTAG_SCAN_SMALL, "pathtag_scan_small.slang");
include_slang!(PATHTAG_SCAN_LARGE, "pathtag_scan_large.slang");
include_slang!(DRAW_REDUCE, "draw_reduce.slang");
include_slang!(CLIP_REDUCE, "clip_reduce.slang");
include_slang!(CLIP_LEAF, "clip_leaf.slang");
include_slang!(DRAW_LEAF, "draw_leaf.slang");
include_slang!(BINNING, "binning.slang");
include_slang!(TILE_ALLOC, "tile_alloc.slang");
include_slang!(PATH_COUNT, "path_count.slang");
include_slang!(BACKDROP, "backdrop.slang");
include_slang!(BACKDROP_DYN, "backdrop_dyn.slang");
include_slang!(COARSE, "coarse.slang");
include_slang!(PATH_TILING, "path_tiling.slang");
include_slang!(FLATTEN, "flatten.slang");
include_slang!(FINE, "fine.slang");
include_slang!(FINE_CPU, "fine_cpu.slang");
include_slang!(FILTER_PASS, "filter_pass.slang");

/// Directory the Slang compiler searches for `import ekrano_shared`.
///
/// Entry shaders are compiled from memory, but that import still needs a file
/// on disk. Checkout builds use `ekrano_shaders/slang/`. Packaged binaries do
/// not have `CARGO_MANIFEST_DIR` on the target machine, so this also honors
/// `EKRANO_SLANG_DIR`, paths next to the executable, then a temp copy of the
/// embedded [`EKRANO_SHARED`] source.
pub fn slang_search_path() -> PathBuf {
    if let Some(p) = env_search_dir() {
        return p;
    }
    if let Some(p) = bundled_search_dir() {
        return p;
    }
    let cargo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("slang");
    if dir_has_shared(&cargo) {
        return cargo;
    }
    materialize_embedded_slang()
}

fn env_search_dir() -> Option<PathBuf> {
    let p = PathBuf::from(env::var_os("EKRANO_SLANG_DIR")?);
    dir_has_shared(&p).then_some(p)
}

fn bundled_search_dir() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    let dir = exe.parent()?;
    let candidates = [
        dir.join("slang"),
        dir.join("../Resources/ekrano/slang"),
        dir.join("../Resources/slang"),
    ];
    candidates.into_iter().find(|p| dir_has_shared(p))
}

fn dir_has_shared(dir: &Path) -> bool {
    dir.join("ekrano_shared.slang").is_file()
}

fn materialize_embedded_slang() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let stamp = {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            EKRANO_SHARED.hash(&mut h);
            h.finish()
        };
        let dir = env::temp_dir().join(format!("ekrano-slang-{}-{stamp:x}", env!("CARGO_PKG_VERSION")));
        let dest = dir.join("ekrano_shared.slang");
        if dest.is_file() {
            return dir;
        }
        if fs::create_dir_all(&dir).is_err() {
            return dir;
        }
        let tmp = dir.join("ekrano_shared.slang.tmp");
        if fs::write(&tmp, EKRANO_SHARED).is_ok() {
            let _ = fs::rename(&tmp, &dest);
        }
        dir
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slang_search_path_has_ekrano_shared() {
        let dir = slang_search_path();
        let shared = dir.join("ekrano_shared.slang");
        assert!(shared.is_file(), "missing {shared:?}");
        let body = fs::read_to_string(&shared).unwrap();
        assert!(body.contains("import goldy_exp"));
    }

    #[test]
    fn materialize_writes_shared_module() {
        let dir = materialize_embedded_slang();
        let body = fs::read_to_string(dir.join("ekrano_shared.slang")).unwrap();
        assert_eq!(body, EKRANO_SHARED);
    }
}
