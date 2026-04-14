#!/usr/bin/env bash
# Copyright 2025 the Ekrano Authors
# SPDX-License-Identifier: Apache-2.0 OR MIT

# Shared Ubuntu setup for CI and local Docker reproduction.
# Installs Mesa/lavapipe (software Vulkan), Vulkan tooling, and build
# dependencies, then locates the lavapipe ICD JSON and exports it.
#
# Usage:
#   In GitHub Actions:  bash ci/setup-ubuntu.sh
#   In Dockerfile:      RUN bash /tmp/setup-ubuntu.sh

set -euo pipefail

# --- Install packages ---------------------------------------------------

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

$SUDO apt-get update
$SUDO apt-get install -y \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    libxcb-xfixes0-dev

# --- Install patched lavapipe from mesa-fork ----------------------------
# Workaround for https://gitlab.freedesktop.org/mesa/mesa/-/work_items/15227
# Remove this block and restore mesa-vulkan-drivers from apt once the
# upstream fix reaches the Ubuntu 24.04 packages.

MESA_FORK_URL="https://github.com/koubaa/mesa-fork/releases/download/v1/lavapipe-fix.tar.gz"
MESA_DIR="/opt/mesa-fork"
$SUDO mkdir -p "$MESA_DIR"
curl -sL "$MESA_FORK_URL" | $SUDO tar xz -C "$MESA_DIR"
LAVAPIPE_ICD="$MESA_DIR/lvp_icd.x86_64.json"

if [ ! -f "$LAVAPIPE_ICD" ]; then
    echo "WARNING: Could not locate lavapipe ICD JSON at $LAVAPIPE_ICD" >&2
fi

# --- Export environment --------------------------------------------------

if [ -n "${GITHUB_ENV:-}" ]; then
    # Running inside GitHub Actions
    echo "LAVAPIPE_ICD=$LAVAPIPE_ICD" >> "$GITHUB_ENV"
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

VK_ICD_FILENAMES="$LAVAPIPE_ICD" vulkaninfo --summary || echo "vulkaninfo failed, continuing anyway"
