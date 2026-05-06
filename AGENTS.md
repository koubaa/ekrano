# Ekrano Agent Notes

## Origin

Ekrano is a fork of [Vello](https://github.com/linebender/vello) (v0.7.0), a GPU compute-centric 2D renderer
by the Linebender project. The fork retains the full-GPU compute pipeline and removes the hybrid CPU/sparse-strips
backend, targeting Vulkan 1.4+/DX12/Metal 2+ exclusively via the Goldy GPU abstraction layer.

## Copyright Headers

**DO NOT remove or replace existing Vello copyright headers from source files.**

Original files carry:
```
// Copyright 20XX the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
```
(Shaders also carry `OR Unlicense`.)

When you **significantly modify** an existing file, add the Ekrano line **below** the Vello line:
```
// Copyright 20XX the Vello Authors
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
```

When you **create a new file**, use only the Ekrano header:
```
// Copyright 2026 the Ekrano Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
```

## Repository Structure

```
ekrano/                   # Git root (forked from Vello)
├── ekrano/               # Main crate (renderer, Scene, Renderer API)
├── ekrano_encoding/      # Scene encoding into GPU-friendly streams
├── ekrano_shaders/       # Slang compute shaders + optional CPU fallbacks
├── ekrano_tests/         # Snapshot / regression test suite
├── xtask/                # Kompari: diff snapshots/ vs current/
├── examples/
│   ├── headless/         # Offscreen rendering (primary use case)
│   ├── scenes/           # Shared scene library for examples and tests
│   └── simple/           # Minimal windowed example
├── doc/                  # Architecture notes
├── LICENSE-APACHE        # Must not be modified
└── LICENSE-MIT           # Must not be modified
```


## Test Commands

```bash
cargo test --workspace --all-features        # run all tests
cargo run -p headless -- -x 800 -y 600 -s 0  # render offscreen via GPU
EKRANO_CI_GPU_SUPPORT=yes cargo test -p ekrano_tests  # force GPU snapshot tests
EKRANO_TEST_UPDATE=all cargo test -p ekrano_tests     # update snapshot references
EKRANO_TEST_CREATE=all cargo test -p ekrano_tests     # create new snapshots
```
## Linting (do at the end of each agent interaction after code changes)

`cargo fmt`

## Cursor Cloud specific instructions

### Environment prerequisites

The following are installed by the update script on every session start:
- Rust stable (≥1.88 MSRV; CI uses 1.93+)
- `goldy` GPU abstraction cloned at `/goldy` (path deps resolve from subcrate dirs via `../../goldy`)
- Slang compiler binaries in `/goldy/slang/bin/`
- Vulkan/lavapipe (Mesa 25.x software Vulkan), `libstdc++-14-dev`

### Running services

Before running any GPU-related command, source the Vulkan environment:
```bash
source /tmp/ekrano-ci.env
```
This sets `VK_ICD_FILENAMES` to the lavapipe ICD and `GOLDY_BACKEND=vulkan`.

### Known lavapipe limitation

The full GPU compute pipeline (Slang shaders → SPIR-V → lavapipe) executes without errors but produces **visually empty frames** on lavapipe. The coarse/fine rasterization stages run (bump counts are non-zero in debug logs) but the fine rasterizer outputs blank pixels. Snapshot tests (`ekrano_tests`) will fail on pixel comparison. This is acceptable for development — the pipeline structure, encoding, and shader compilation are still fully testable. Use `EKRANO_SKIP_LFS_SNAPSHOTS=all` to skip snapshot comparisons if LFS files are unavailable.

### Quick reference

| Task | Command |
|------|---------|
| Build | `cargo build --workspace --all-features` |
| Lint | `cargo fmt` |
| Test (non-snapshot) | `cargo test --workspace --all-features --exclude ekrano_tests` |
| Test (all, expect snapshot failures) | `EKRANO_CI_GPU_SUPPORT=yes cargo test --workspace --all-features` |
| Headless render | `source /tmp/ekrano-ci.env && cargo run -p headless -- -x 800 -y 600 -s 0` |