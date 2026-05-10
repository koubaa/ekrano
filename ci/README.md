# CI Reproduction

Reproduce the Ubuntu CI environment locally using Docker and
[lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html) (software Vulkan).
No GPU hardware or special Docker runtime is required.

## Quick start

```bash
# Build the image (from the repo root)
docker build -t ekrano-ci ci/

# Run all tests
docker run --rm -v "$(pwd)":/workspace/ekrano ekrano-ci

# Run a specific test
docker run --rm -v "$(pwd)":/workspace/ekrano ekrano-ci \
  "cargo nextest run --workspace --locked --all-features -E 'test(snapshot_splash)'"

# Interactive shell for debugging
docker run --rm -it -v "$(pwd)":/workspace/ekrano ekrano-ci bash
```

## How it works

`setup-ubuntu.sh` is the single source of truth for the Ubuntu/lavapipe
environment.  It is called by both:

- **GitHub Actions** (`ci.yml`) — the script detects `$GITHUB_ENV` and
  exports variables accordingly.
- **Dockerfile** — the script writes a sourceable env file instead.

When you need to change Mesa packages, Vulkan dependencies, or the lavapipe
ICD detection logic, edit `setup-ubuntu.sh` and both consumers stay in sync.

## Updating the Rust version

The Dockerfile accepts a build arg:

```bash
docker build --build-arg RUST_VERSION=1.94 -t ekrano-ci ci/
```

Keep this in sync with `RUST_STABLE_VER` in `.github/workflows/ci.yml` and with `channel` in `rust-toolchain.toml` at the repo root.
