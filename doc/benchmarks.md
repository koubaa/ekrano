# Ekrano benchmarks (internal)

Formal public benchmarks are still TODO upstream; this file records **internal** numbers for
the Goldy-backed renderer on the Velato Tiger workload. Treat the current CUDA+DX12 numbers
as **state of the art** for that path until a structural API or driver change removes the
WDDM interop boundary (see `goldy/src/backend/cuda/WDDM_INTEROP.md`).

## Workload

| Item | Value |
|------|-------|
| Scene | Ghostscript Tiger (`refs/velato` default) |
| Renderer | Ekrano via `use_ekrano` + Goldy |
| Window | `with_winit_bin` |
| Vsync | off (`--no-vsync`) |
| Present mode | `Immediate` |
| Typical resolution | window-sized (Velato default ~800×600 class) |

## Summary (August 2026, NVIDIA consumer GPU, Windows)

| Backend | Median frame interval | FPS (from interval) | Notes |
|---------|----------------------:|--------------------:|-------|
| Native DX12 | **406 µs** | **~2,460** | Head-chasing tail; single graphics context |
| CUDA + DX12 | **742–757 µs** | **~1,320–1,330** | External-memory scratch + present copy |
| CUDA (standalone run) | — | **~1,334** | Same build, no nsys attached |

CUDA is ~1.85× slower end-to-end on this workload. The gap is **not** dominated by the fine
kernel alone (~175 µs); ~78% of the delta is GPU idle and cross-API scheduling (see breakdown).

## Per-frame GPU work (CUDA, node trace medians)

| Stage | Duration | Count / frame |
|-------|----------|---------------|
| Stream fine `cs_main` | ~174–177 µs | 1 |
| Graph `cs_main` nodes | ~196 µs total | ~15 |
| `goldy_apply_dispatch_shape` | ~21 µs total | ~14 |
| HtoA scene upload | ~1.2 µs | 1 |
| AtoA export to scratch | ~11 µs | 1 |
| DX12 present command list | ~10–11 µs | 1 |

Typical frame sequence on GPU:

```text
fine → AtoA → HtoD → HtoD → dx12 → memset → HtoA → graph kernels
```

## Scheduling gaps (CUDA, recurring medians)

| Transition | Median gap | Likely cause |
|------------|------------|--------------|
| HtoA → first graph updater | **~118 µs** | Graph launch / submission start |
| HtoD → DX12 workload | **~74 µs** | CUDA → DX12 context transition |
| DX12 workload → CUDA memset | **~78 µs** | DX12 → CUDA context transition |
| AtoA → HtoD | ~22 µs | Stream scheduling |
| memset → HtoA | ~25 µs | Stream scheduling |
| fine → AtoA | ~3 µs | — |

The two ~75 µs gaps around the ~11 µs DX12 copy sum to ~150 µs/frame — consistent with
WDDM context transitions on elevated traces.

## Parity accounting vs native DX12 (node trace)

| | CUDA | DX12 | Delta |
|---|--:|--:|--:|
| Frame interval | 742.3 µs | 406.1 µs | **336.2 µs** |
| Visible GPU work | 408.0 µs | 334.0 µs | 74.0 µs |
| GPU idle / gaps | 334.3 µs | 71.7 µs | **262.6 µs** |

Structural explanation (~336 µs total):

| Component | ~µs/frame | Notes |
|-----------|----------:|-------|
| CUDA↔DX12 WDDM transitions | ~150 | Two ~75 µs gaps around present copy |
| Graph / submission start | ~118 | HtoA → first graph node |
| CUDA indirect updaters | ~21 | vs DX12 `ExecuteIndirect` |
| Extra / slower visible GPU | ~74 | Codegen asymmetry (Config loads, Rgba8 pack) |
| Remainder | small | Queue ordering, fence propagation |

## Optimization experiments (no meaningful FPS win)

| Experiment | Result |
|------------|--------|
| DXGI-first acquire reorder | `fence_wait` ~instant; throughput unchanged |
| Remove present-stream CUDA-event bridge | ~14 µs median interval (~1.9%) |
| COPY hop for CUDA fence wait | 754.6 µs / 1325 FPS; copy still on DIRECT |
| Depth-3 scratch ring + ready/recycle fences | 757 µs / 1321 FPS; **0%** DX12 copy overlap with CUDA N+1 |

## Codegen notes (visible-work delta, not yet fully quantified)

Worth DXIL comparison before micro-optimizing shaders:

- Slang CUDA emits six full `Config` struct loads for six scalar field reads in `fine.slang`.
- CUDA `DirectSpatial<float4>` → `Rgba8Unorm` uses explicit pack/round in shader; DX12 uses
  typed UAV store with hardware UNORM conversion.
- DXIL build uses `SLANG_FLOATING_POINT_MODE_PRECISE`; CUDA path does not.
- CUDA adds ~14 `goldy_apply_dispatch_shape` updater kernels/frame; DX12 uses native indirect.

Nsight Compute on stream fine `cs_main`: latency-bound, high local-memory traffic — secondary
to interop for parity.
