# Copyright 2025 the Ekrano Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

# Reproduces the Ubuntu CI environment locally using lavapipe (software Vulkan).
# No GPU or special Docker runtime required.
#
# Build:
#   docker build -t ekrano-ci ci/
#
# Run tests (bind-mount your checkout):
#   docker run --rm -v "$(pwd)":/workspace/ekrano ekrano-ci

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Shared setup: Mesa/lavapipe + Vulkan deps
COPY setup-ubuntu.sh /tmp/setup-ubuntu.sh
ENV ENV_FILE=/etc/profile.d/ekrano-ci.sh
RUN bash /tmp/setup-ubuntu.sh && rm /tmp/setup-ubuntu.sh

# Build tooling
RUN apt-get install -y git git-lfs curl ca-certificates build-essential pkg-config unzip && \
    rm -rf /var/lib/apt/lists/*

# Rust (version kept in sync with RUST_STABLE_VER in .github/workflows/ci.yml)
ARG RUST_VERSION=1.93
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain "${RUST_VERSION}"
ENV PATH="/root/.cargo/bin:${PATH}"

# cargo-nextest
RUN cargo install cargo-nextest --locked

# Clone goldy as a sibling so path deps (../../goldy) resolve correctly,
# then download the vendored Slang compiler binaries (gitignored, fetched at CI time).
RUN git clone --depth 1 --branch main https://github.com/koubaa/goldy.git /workspace/goldy && \
    cd /workspace/goldy/slang && bash download.sh

# Source the lavapipe env vars for every shell
RUN echo '. /etc/profile.d/ekrano-ci.sh' >> /root/.bashrc

# Test-runner env vars (matching CI)
ENV EKRANO_CI_GPU_SUPPORT=yes
ENV EKRANO_CI_SKIP_SLOW=yes

WORKDIR /workspace/ekrano

ENTRYPOINT ["bash", "-lc"]
CMD ["cargo nextest run --workspace --locked --all-features --no-fail-fast"]
