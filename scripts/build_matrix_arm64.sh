#!/usr/bin/env bash
# ==============================================================================
# Pointcept wheel matrix: arm64
#
# Builds one wheelhouse per CUDA line (cu126, cu128, cu130) against torch 2.9.1
# and CPython 3.12, each with the widest device arch list that line's nvcc can
# emit. The agent picks a wheelhouse only when its `cuda_arch` tag names the
# GPU's capability exactly, so every generation is listed.
#
# Run it on a native aarch64 host such as a DGX Spark: nothing on this
# architecture has a prebuilt wheel, so every native package is compiled, and
# nvcc under QEMU is 10-30x slower.
#
# torch stays at 2.9.1 on purpose: spconv linked against 2.13.0 faults on
# sm_121 (merge_sort: cudaErrorIllegalAddress), and the agent prefers the
# highest torch, so a newer wheelhouse tagged 12.1 would be picked by a GB10.
#
# Orin (8.7) and Thor (11.0) are left out: the aarch64 torch the agent installs
# is an SBSA build, and flash-attn has no kernels for Thor.
#
# Quick start
#   ./scripts/build_matrix_arm64.sh                  # all three lines
#   ./scripts/build_matrix_arm64.sh --jobs 8         # extra build_wheels.sh args
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
BUILDER="${REPO_ROOT}/scripts/build_wheels.sh"
OUT="${REPO_ROOT}/wheelhouse-wide"

for spec in "cu126|8.0 8.6 8.9 9.0" \
            "cu128|8.0 8.6 8.9 9.0 10.0 12.0" \
            "cu130|8.0 8.6 8.9 9.0 10.0 10.3 12.0 12.1"; do
  bash "${BUILDER}" build --arch arm64 --accel "${spec%%|*}" \
    --torch 2.9.1 --python 3.12 --cuda-arch "${spec#*|}" \
    --out "${OUT}" --keep-going -y "$@"
done
