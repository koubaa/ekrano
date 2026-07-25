#!/usr/bin/env bash
# Copyright 2026 the Ekrano Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

# Shared Ubuntu setup for CI and local Docker reproduction.
# Installs Mesa/lavapipe (software Vulkan), Vulkan tooling, and the loader,
# then locates the lavapipe ICD JSON and exports it.
#
# Headless CI only needs the lavapipe ICD + loader + vulkaninfo. Do not install
# X11/XCB -dev packages (leftover from windowed examples) and do not run a full
# `apt-get upgrade` — that upgrades unrelated runner packages (Chrome, PHP,
# Firefox snap, …) and dominates CI wall time.
#
# Usage:
#   In GitHub Actions:  bash ci/setup-ubuntu.sh
#   In Dockerfile:      RUN bash /tmp/setup-ubuntu.sh

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# --- Install packages ---------------------------------------------------

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

$SUDO apt-get update
# `apt-get install` upgrades already-installed packages to the newest candidate
# from the configured archives (noble-updates ships Mesa 25.x / Vulkan 1.4).
# Avoid `apt-get upgrade [pkgs]` — on Ubuntu runners that still performs a
# full system upgrade of every upgradable package.
$SUDO apt-get install -y --no-install-recommends \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers

# --- Locate lavapipe ICD ------------------------------------------------

LAVAPIPE_ICD=$(find /usr -name "lvp_icd*.json" 2>/dev/null | head -1)
if [ -z "$LAVAPIPE_ICD" ]; then
    for path in /usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
                /usr/share/vulkan/icd.d/lvp_icd.json; do
        if [ -f "$path" ]; then LAVAPIPE_ICD="$path"; break; fi
    done
fi

if [ -z "$LAVAPIPE_ICD" ]; then
    echo "WARNING: Could not locate lavapipe ICD JSON" >&2
fi

# --- Export environment --------------------------------------------------

if [ -n "${GITHUB_ENV:-}" ]; then
    # Running inside GitHub Actions
    echo "LAVAPIPE_ICD=$LAVAPIPE_ICD" >> "$GITHUB_ENV"
    echo "VK_ICD_FILENAMES=$LAVAPIPE_ICD" >> "$GITHUB_ENV"
    echo "VK_LAYER_PATH=" >> "$GITHUB_ENV"
    echo "GOLDY_BACKEND=vulkan" >> "$GITHUB_ENV"
else
    # Running in Docker or locally -- write to a sourceable env file
    ENV_FILE="${ENV_FILE:-/tmp/ekrano-ci.env}"
    cat > "$ENV_FILE" <<EOF
export LAVAPIPE_ICD="$LAVAPIPE_ICD"
export VK_ICD_FILENAMES="$LAVAPIPE_ICD"
export VK_LAYER_PATH=""
export GOLDY_BACKEND=vulkan
EOF
    echo "Wrote environment to $ENV_FILE (source it in your shell)"
fi

# --- Smoke-test Vulkan ---------------------------------------------------
# Skip under GitHub Actions: first lavapipe `vulkaninfo` is slow (~minutes of
# LLVM init), and ci.yml has a dedicated "Verify lavapipe" step. Keep the
# smoke test for Docker / local reproduction.

if [ -z "${GITHUB_ENV:-}" ]; then
    VK_ICD_FILENAMES="$LAVAPIPE_ICD" vulkaninfo --summary || echo "vulkaninfo failed, continuing anyway"
fi
