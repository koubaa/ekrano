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

```bash
cargo check
cargo fmt -p ekrano -p ekrano_encoding -p ekrano_shaders -p ekrano_tests -p xtask -p headless -p scenes --check
taplo fmt --check --diff
bash .github/copyright.sh
cargo clippy --workspace --locked --all-features
```
