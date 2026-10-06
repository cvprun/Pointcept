#!/usr/bin/env bash
# ==============================================================================
# Pointcept wheel matrix: amd64
#
# Builds one wheelhouse per CUDA line (cu126, cu128, cu130) against torch 2.9.1
# and CPython 3.12, each with the widest device arch list that line's nvcc can
# emit. The agent picks a wheelhouse only when its `cuda_arch` tag names the
# GPU's capability exactly, so every generation is listed rather than relying
# on binary compatibility (an sm_80 cubin runs on 8.6, but a tag of 8.0 does
# not select it there).
#
# --force-source keeps the tag honest: the prebuilt spconv, cumm and PyG wheels
# carry whatever arch set upstream chose, not the one listed here.
#
# 7.5 (Turing) is left out until the agent stops enabling flash-attn on parts
# it has no kernels for.
#
# Quick start
#   ./scripts/build_matrix_amd64.sh                  # all three lines
#   ./scripts/build_matrix_amd64.sh --jobs 8         # extra build_wheels.sh args
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
BUILDER="${REPO_ROOT}/scripts/build_wheels.sh"
OUT="${REPO_ROOT}/wheelhouse"

for spec in "cu126|8.0 8.6 8.9 9.0" \
            "cu128|8.0 8.6 8.9 9.0 10.0 12.0" \
            "cu130|8.0 8.6 8.9 9.0 10.0 10.3 12.0"; do
  bash "${BUILDER}" build --arch amd64 --accel "${spec%%|*}" \
    --torch 2.9.1 --python 3.12 --cuda-arch "${spec#*|}" \
    --force-source --out "${OUT}" --keep-going -y "$@"
done

# Pointcept itself: one py3-none-any wheel, the same file in every directory.
bash "${REPO_ROOT}/scripts/build_pointcept_wheel.sh" --into "${OUT}" >/dev/null
