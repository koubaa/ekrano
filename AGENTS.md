# Ekrano Agent Notes

## Origin

Ekrano is a fork of [Vello](https://github.com/linebender/vello) (v0.7.0), a GPU compute-centric 2D renderer
by the Linebender project. The fork retains the full-GPU compute pipeline and removes the hybrid CPU/sparse-strips
backend, targeting Vulkan 1.4+/DX12/Metal 2+ exclusively via the Goldy GPU abstraction layer.

Vello is dual-licensed Apache-2.0 OR MIT. Ekrano inherits both licenses and adds its own contributions
under the same dual-license terms.

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
├── ekrano_shaders/       # 33 WGSL compute shaders + pipeline metadata
├── ekrano_tests/         # Snapshot / regression test suite
├── xtask/                # Snapshot diff tooling (kompari)
├── examples/
│   ├── headless/         # Offscreen rendering (primary use case)
│   ├── scenes/           # Shared scene library for examples and tests
│   └── simple/           # Minimal windowed example
├── doc/                  # Architecture notes
├── LICENSE-APACHE        # Must not be modified
└── LICENSE-MIT           # Must not be modified
```

## Development Phases

- **Phase 1** (complete): Build/test/run baseline
- **Phase 2** (complete): Strip to full-GPU pipeline; remove hybrid backend, mobile, WASM
- **Phase 2.4** (complete): Rebrand vello → ekrano
- **Phase 2.5** (todo): Make `render_to_texture` the primary API; remove surface/swapchain path
- **Phase 3a** (todo): Replace wgpu with Goldy (keep WGSL, compile via naga → SPIR-V → Goldy)
- **Phase 3b** (todo): Port WGSL shaders to Slang
- **Phase 3c** (todo): Exploit Goldy features (BDA, indirect dispatch, bounded memory)

## Test Commands

```bash
cargo test --workspace --all-features        # run all tests
cargo run -p headless -- -x 800 -y 600 -s 0  # render offscreen via GPU
EKRANO_CI_GPU_SUPPORT=yes cargo test -p ekrano_tests  # force GPU snapshot tests
EKRANO_TEST_UPDATE=all cargo test -p ekrano_tests     # update snapshot references
EKRANO_TEST_CREATE=all cargo test -p ekrano_tests     # create new snapshots
```
