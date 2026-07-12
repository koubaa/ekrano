# Ekrano tests

This crate holds integration tests for Ekrano.

- **Property tests** exercise rendering invariants (pixel colors, formats) via `render_then_debug_sync`.
- **Snapshot tests** compare GPU output (Goldy) to reference PNGs under `snapshots/`, using FLIP for a tolerant perceptual metric. Smoke references live in `snapshots/smoke/`; larger references use git LFS (`snapshots/*.png`).
- **Dual-backend coverage**: rendering tests are duplicated for the Scheme backend as explicit `scheme_<test>` cases that call the same shared body and compare against the same snapshot baselines as the Classic tests.

Failed snapshot runs write outputs under `current/` (mirroring the `snapshots/` layout). Use `cargo xtask snapshots report` (from the workspace root) with Kompari to diff reference vs. current.

## LFS

Install [git lfs](https://git-lfs.com/) and run `git lfs pull` before running LFS-backed snapshot tests locally. If LFS objects are missing, set `EKRANO_SKIP_LFS_SNAPSHOTS=all` to skip those tests (as documented in test failure messages).
