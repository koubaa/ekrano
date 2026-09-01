# Architecture

This document should be updated semi-regularly. Feel free to open an issue if it hasn't been updated in more than a year.

## Goals

The major goal of Vello is to provide a high quality GPU accelerated renderer suitable for a range of 2D graphics applications, including rendering for GUI applications, creative tools, and scientific visualization.

Vello emerges from being a research project, which attempts to answer these hypotheses:

- To what extent is a compute-centered approach better than rasterization ([Direct2D])?
- To what extent do advanced GPU features (subgroups, descriptor arrays, device-scoped barriers) help?
- Can we improve quality and extend the imaging model in useful ways?

Another goal of the overall project is to explain how the renderer is built, and to advance the state of building applications on GPU compute shaders more generally.
Much of the progress on Vello is documented in blog entries.
See [blogs.md](blogs.md) for pointers to those.

Ideally, we'd like our documentation to be more structured; we may refactor it in the future (see [#488]).


## Roadmap

The [roadmap for 2023](roadmap_2023.md) is still largely applicable as historical Vello context.
The "Semi-stable encoding format" section can be considered implemented. The handwritten CPU fallback stages were removed; CPU debug of the same Slang kernels is tracked in Goldy ([koubaa/goldy#292](https://github.com/koubaa/goldy/issues/292)).

Our current priority is to fill in missing features and to fix rendering artifacts, so that Vello can reach feature parity with other 2D graphics engines.


## File structure

The repository is structured as such:

- `doc/` - Historical Vello vision documents; treat as background reading.
- `examples/` - Example crates (`headless`, `scenes`, etc.).
- `ekrano/` - Main renderer crate (Goldy backend).
- `ekrano_encoding/` - Scene → GPU-friendly buffer layouts.
- `ekrano_shaders/`
  - `slang/` - Slang compute sources; shared types live in `ekrano_shared.slang`. Compiled at runtime by Goldy.
- `ekrano_tests/` - Snapshot and regression tests.


## Shaders (Slang)

GPU stages are written in Slang and compiled per backend (SPIR-V / DXIL) through Goldy. The former WGSL tree and naga-driven build metadata have been removed; bind layouts are specified alongside dispatch in `ekrano/src/shaders.rs`.


## Path encoding

See [Path segment encoding](./pathseg.md) document.


## Intermediary layers

There are multiple layers between “draw in `Scene`” and GPU work:

- `Scene` builds an `Encoding` (paths, draws, transforms, etc.).
- That becomes a `Recording` of GPU commands (uploads, dispatches, copies).
- `GoldyRenderer` / `GoldyEngine` execute the recording on the Goldy device.


[direct2d]: https://docs.microsoft.com/en-us/windows/win32/direct2d/direct2d-portal
[#488]: https://github.com/linebender/vello/issues/488
[#467]: https://github.com/linebender/vello/issues/467
[#386]: https://github.com/linebender/vello/issues/386
