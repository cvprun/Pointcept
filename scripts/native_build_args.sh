#!/usr/bin/env bash
# ==============================================================================
# Pointcept native build argument estimator
#
# For a machine with no docker, no podman and no sudo: works out the arguments
# `build_wheels.sh build --native` needs on this host and prints the command.
# Nothing is built and nothing is installed.
#
# build_wheels.sh --native already reads the GPU and the toolkit CUDA_HOME
# points at. What it cannot do is choose: which of several toolkits in /usr/local,
# $HOME or a conda env to point CUDA_HOME at, which torch and python the wheels
# have to match when an env already has torch, which older g++ to hand nvcc when
# the system one is too new, and which packages to skip when a header the build
# needs cannot be installed without root. That is what is estimated here.
#
# Every candidate is checked with `build_wheels.sh matrix --native`, so the
# toolkit, arch and torch tables stay in one file: a candidate that script
# would drop is dropped here too.
#
# Quick start
#   ./scripts/native_build_args.sh                    # print the command
#   ./scripts/native_build_args.sh --run              # print it, then run it
#   ./scripts/native_build_args.sh -- --only pointops # extra build_wheels.sh args
#   ./scripts/native_build_args.sh --python-bin ~/venv/bin/python
#
# stdout carries only the command; the reasoning goes to stderr.
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
BUILDER="${REPO_ROOT}/scripts/build_wheels.sh"

if [[ -t 2 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""
fi

log()   { echo "${C_BLUE}[args ]${C_RESET} $*" >&2; }
info()  { echo "${C_GREEN}[ ok  ]${C_RESET} $*" >&2; }
warn()  { echo "${C_YELLOW}[warn ]${C_RESET} $*" >&2; }
error() { echo "${C_RED}[error]${C_RESET} $*" >&2; }
die()   { error "$*"; exit 1; }

usage() {
  cat <<EOF
${C_BOLD}Pointcept native build argument estimator${C_RESET}

Prints the build_wheels.sh --native command for this host (no container
engine, no root). Nothing is built unless --run is given.

${C_BOLD}USAGE${C_RESET}
  ${SCRIPT_NAME} [options] [-- build_wheels.sh options]

${C_BOLD}OPTIONS${C_RESET}
  --python-bin PATH  Interpreter whose torch the wheels must match
                     [\$VIRTUAL_ENV or \$CONDA_PREFIX python, when active]
  --cuda-home DIR    Consider only this toolkit
  --accel TOKEN      Force the accelerator (cu128, cpu, ...)
  --cuda-arch LIST   Force TORCH_CUDA_ARCH_LIST       [from nvidia-smi]
  --torch VER        Force the torch version           [from the env, if any]
  --python VER       Force the CPython version         [from the env, if any]
  --run              Run the command after printing it
  -h, --help         Show this message

  Anything after -- is appended to the build_wheels.sh command verbatim.
EOF
}

PYTHON_BIN=""; CUDA_HOME_ONLY=""
FORCE_ACCEL=""; FORCE_CUDA_ARCH=""; FORCE_TORCH=""; FORCE_PYTHON=""
RUN="0"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin) PYTHON_BIN="$2";      shift 2 ;;
    --cuda-home)  CUDA_HOME_ONLY="$2";  shift 2 ;;
    --accel)      FORCE_ACCEL="$2";     shift 2 ;;
    --cuda-arch)  FORCE_CUDA_ARCH="$2"; shift 2 ;;
    --torch)      FORCE_TORCH="$2";     shift 2 ;;
    --python)     FORCE_PYTHON="$2";    shift 2 ;;
    --run)        RUN="1";              shift   ;;
    -h|--help)    usage; exit 0 ;;
    --)           shift; EXTRA=("$@"); break ;;
    *) die "unknown option '$1' (try '${SCRIPT_NAME} --help')" ;;
  esac
done

[[ -x "${BUILDER}" || -r "${BUILDER}" ]] || die "cannot find ${BUILDER}"

# 12.8 -> cu128. The builder decides whether it knows the token.
accel_of_version() { local v="$1"; echo "cu${v//./}"; }

# `sort -V` needs the same number of components on both sides (see
# build_wheels.sh pad_version), and every version compared here is major.minor.
version_ge() {
  [[ "$1" == "$2" ]] && return 0
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]
}

# ------------------------------------------------------------------------------
# Host tools no container supplies. None of these can be fixed with apt here,
# so each hint is one that works without root.
# ------------------------------------------------------------------------------
check_tools() {
  local -a miss=()
  local t
  for t in gcc g++ git; do
    command -v "${t}" >/dev/null 2>&1 \
      || miss+=("${t}: conda install -c conda-forge compilers git (no root needed)")
  done
  if ! command -v uv >/dev/null 2>&1 && [[ ! -x "${HOME:-}/.local/bin/uv" ]] \
     && ! command -v curl >/dev/null 2>&1; then
    miss+=("uv (or curl to fetch it): https://docs.astral.sh/uv/getting-started/installation/")
  fi
  [[ ${#miss[@]} -eq 0 ]] && return 0
  error "this host lacks what build_wheels.sh --native cannot do without:"
  for t in "${miss[@]}"; do error "  ${t}"; done
  exit 1
}

# ------------------------------------------------------------------------------
# GPU and driver
# ------------------------------------------------------------------------------
GPU_CAPS=""; DRIVER_CUDA=""

probe_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  GPU_CAPS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | awk '!seen[$0]++' | paste -sd' ' || true)"
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' \
    | head -1 | grep -oE '[0-9]+\.[0-9]+' || true)"
}

# ------------------------------------------------------------------------------
# The env the wheels will be installed into. When it already has torch, its
# version and CUDA line bind the build: an extension compiled against another
# torch fails to import with an undefined symbol, and torch's cpp_extension
# refuses a toolkit whose CUDA major differs from its own.
# ------------------------------------------------------------------------------
ENV_PYTHON=""; ENV_TORCH=""; ENV_TORCH_CUDA=""

probe_env() {
  local py="${PYTHON_BIN}"
  if [[ -z "${py}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
      py="${VIRTUAL_ENV}/bin/python"
    elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
      py="${CONDA_PREFIX}/bin/python"
    fi
  fi
  [[ -n "${py}" ]] || return 0
  [[ -x "${py}" ]] || die "--python-bin ${py} is not executable"

  local out
  out="$("${py}" - <<'PY' 2>/dev/null || true
import sys
v = "%d.%d" % sys.version_info[:2]
try:
    import torch
    print(v, torch.__version__.split("+")[0], torch.version.cuda or "cpu")
except Exception:
    print(v, "-", "-")
PY
)"
  [[ -n "${out}" ]] || { warn "could not run ${py}; ignoring it"; return 0; }
  read -r ENV_PYTHON ENV_TORCH ENV_TORCH_CUDA <<< "${out}"
  [[ "${ENV_TORCH}" != "-" ]] || { ENV_TORCH=""; ENV_TORCH_CUDA=""; }
  log "env ${py}: python ${ENV_PYTHON}${ENV_TORCH:+, torch ${ENV_TORCH} (${ENV_TORCH_CUDA})}"
}

# ------------------------------------------------------------------------------
# Toolkits. Without root they end up anywhere: the runfile's --installpath,
# a conda env's cuda-toolkit, a module system under /opt.
# ------------------------------------------------------------------------------
declare -A TK_VERSION=()
TK_ORDER=()

add_toolkit() {
  local home="$1"
  [[ -n "${home}" && -x "${home}/bin/nvcc" ]] || return 0
  home="$(readlink -f "${home}")"
  [[ -z "${TK_VERSION[${home}]:-}" ]] || return 0
  local v; v="$("${home}/bin/nvcc" --version 2>/dev/null \
    | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
  [[ -n "${v}" ]] || return 0
  TK_VERSION["${home}"]="${v}"
  TK_ORDER+=("${home}")
}

find_toolkits() {
  if [[ -n "${CUDA_HOME_ONLY}" ]]; then
    add_toolkit "${CUDA_HOME_ONLY}"
    [[ ${#TK_ORDER[@]} -gt 0 ]] || die "--cuda-home ${CUDA_HOME_ONLY} has no working bin/nvcc"
    return 0
  fi
  local d nvcc
  add_toolkit "${CUDA_HOME:-}"
  add_toolkit "${CUDA_PATH:-}"
  if nvcc="$(command -v nvcc 2>/dev/null)"; then
    add_toolkit "$(dirname "$(dirname "$(readlink -f "${nvcc}")")")"
  fi
  add_toolkit "${CONDA_PREFIX:-}"
  shopt -s nullglob
  for d in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-* \
           /opt/nvidia/cuda* "${HOME:-/nonexistent}"/cuda* \
           "${HOME:-/nonexistent}"/.local/cuda* "${HOME:-/nonexistent}"/opt/cuda*; do
    add_toolkit "${d}"
  done
  shopt -u nullglob
}

# ------------------------------------------------------------------------------
# Host compiler. nvcc rejects a g++ newer than its release supports; without
# root the fix is an older g++ already on the box, handed over with -ccbin.
# Echoes the NVCC_PREPEND_FLAGS value to use ("" = none needed); fails when no
# compiler on this host works with that toolkit.
# ------------------------------------------------------------------------------
nvcc_compiles() {
  local home="$1" flags="$2" dir rc=0
  dir="$(mktemp -d)"
  echo '__global__ void probe() {}' > "${dir}/probe.cu"
  NVCC_PREPEND_FLAGS="${flags}" "${home}/bin/nvcc" -c "${dir}/probe.cu" -o "${dir}/probe.o" \
    >/dev/null 2>&1 || rc=$?
  rm -rf "${dir}"
  return ${rc}
}

ccbin_for() {
  local home="$1"
  # A caller's own NVCC_PREPEND_FLAGS is inherited by the build; keep it if it works.
  if nvcc_compiles "${home}" "${NVCC_PREPEND_FLAGS:-}"; then
    echo "${NVCC_PREPEND_FLAGS:-}"; return 0
  fi
  local cxx
  for cxx in $(compgen -c g++- 2>/dev/null | grep -E '^g\+\+-[0-9]+$' | sort -t- -k2 -nr | uniq); do
    if nvcc_compiles "${home}" "-ccbin ${cxx}"; then
      echo "-ccbin ${cxx}"; return 0
    fi
  done
  return 1
}

# ------------------------------------------------------------------------------
# Candidates
# ------------------------------------------------------------------------------

# torch to pin when the env does not dictate one and the builder's default
# (2.9.1) has no wheels on that index. Only the exceptions are listed.
torch_for_accel() {
  case "$1" in
    cu118) echo "2.7.1"  ;;
    cu121) echo "2.5.1"  ;;
    cu124) echo "2.6.0"  ;;
    cu129) echo "2.8.0"  ;;
    cu132) echo "2.12.0" ;;
    *)     echo ""       ;;
  esac
}

# Ask the builder itself. `matrix` runs the same validation `build` does and
# exits non-zero when every combination is dropped.
validate() {
  local kv="$1"; shift
  env ${kv:+"${kv}"} bash "${BUILDER}" matrix --native "$@" "${EXTRA[@]}"
}

# Toolkit homes in the order to try them: newest first, with a toolkit whose
# CUDA line matches the env's torch ahead of everything and a different major
# left out, since cpp_extension refuses that pairing outright.
ranked_toolkits() {
  local home v accel rank
  for home in "${TK_ORDER[@]}"; do
    v="${TK_VERSION[${home}]}"; accel="$(accel_of_version "${v}")"
    [[ -z "${FORCE_ACCEL}" || "${FORCE_ACCEL}" == "${accel}" ]] || continue
    rank=1
    if [[ -n "${ENV_TORCH_CUDA}" && "${ENV_TORCH_CUDA}" != "cpu" ]]; then
      [[ "${v%%.*}" == "${ENV_TORCH_CUDA%%.*}" ]] || continue
      [[ "${v}" == "${ENV_TORCH_CUDA}" ]] && rank=0
    fi
    # Wheels built against a toolkit newer than the driver's ceiling will not
    # load here; keep them only as a last resort.
    if [[ -n "${DRIVER_CUDA}" ]] && ! version_ge "${DRIVER_CUDA}" "${v}"; then rank=2; fi
    echo "${rank} ${v} ${home}"
  done | sort -k1,1n -k2,2Vr | awk '{print $3}'
}

# Result
OUT_ENV=(); OUT_ARGS=()

pick_cuda() {
  local caps="${FORCE_CUDA_ARCH:-${GPU_CAPS}}"
  if [[ -z "${caps}" ]]; then
    warn "no GPU compute capability could be read (nvidia-smi) and no --cuda-arch was given"
    return 1
  fi

  local home v accel torch python flags reason
  local -a args
  while IFS= read -r home; do
    [[ -n "${home}" ]] || continue
    v="${TK_VERSION[${home}]}"; accel="$(accel_of_version "${v}")"
    log "trying ${accel} (nvcc ${v} at ${home})"

    if ! flags="$(ccbin_for "${home}")"; then
      warn "  skip: nvcc ${v} cannot compile with g++ $(g++ -dumpfullversion 2>/dev/null), and no older g++-N on PATH works"
      continue
    fi
    [[ -z "${flags}" ]] || info "  nvcc needs an older host compiler: NVCC_PREPEND_FLAGS='${flags}'"

    torch="${FORCE_TORCH:-${ENV_TORCH:-$(torch_for_accel "${accel}")}}"
    python="${FORCE_PYTHON:-${ENV_PYTHON}}"
    args=(--accel "${accel}" --cuda-arch "${caps}")
    [[ -z "${torch}" ]]  || args+=(--torch "${torch}")
    [[ -z "${python}" ]] || args+=(--python "${python}")

    if ! reason="$(validate "CUDA_HOME=${home}" "${args[@]}" 2>&1 >/dev/null)"; then
      warn "  skip: build_wheels.sh rejects it:"
      grep -E 'skip:|error' <<< "${reason}" | sed 's/^/    /' >&2 || true
      continue
    fi

    if [[ -n "${DRIVER_CUDA}" ]] && ! version_ge "${DRIVER_CUDA}" "${v}"; then
      warn "  the driver runs CUDA ${DRIVER_CUDA} at most; these wheels build but will not load on this host"
    fi
    if [[ -n "${ENV_TORCH_CUDA}" && "${ENV_TORCH_CUDA}" != "cpu" && "${ENV_TORCH_CUDA}" != "${v}" ]]; then
      warn "  the env's torch is built for CUDA ${ENV_TORCH_CUDA}; the build compiles against torch ${torch} from the ${accel} index instead"
    fi

    OUT_ENV=("CUDA_HOME=${home}")
    [[ -z "${flags}" ]] || OUT_ENV+=("NVCC_PREPEND_FLAGS=${flags}")
    OUT_ARGS=("${args[@]}")
    return 0
  done < <(ranked_toolkits)
  return 1
}

pick_cpu() {
  local torch="${FORCE_TORCH:-${ENV_TORCH}}" python="${FORCE_PYTHON:-${ENV_PYTHON}}"
  OUT_ENV=()
  OUT_ARGS=(--accel cpu)
  [[ -z "${torch}" ]]  || OUT_ARGS+=(--torch "${torch}")
  [[ -z "${python}" ]] || OUT_ARGS+=(--python "${python}")
  validate "" "${OUT_ARGS[@]}" >/dev/null 2>&1 \
    || die "build_wheels.sh rejects even a cpu build: $(printf '%q ' "${OUT_ARGS[@]}")"
}

# ------------------------------------------------------------------------------
# sparsehash is header-only, and pointgroup_ops is the only package needing it.
# A copy under a user prefix only has to be put on CPATH; with none at all the
# package is skipped rather than failing the build an hour in.
# ------------------------------------------------------------------------------
pick_sparsehash() {
  [[ "${OUT_ARGS[1]}" != "cpu" ]] || return 0

  # build_wheels.sh keeps the last --skip it is given, so a --skip of the
  # caller's is extended rather than shadowed by a second one.
  local i skip_at="" only=""
  for (( i = 0; i + 1 < ${#EXTRA[@]}; i++ )); do
    case "${EXTRA[i]}" in
      --skip) skip_at="$(( i + 1 ))" ;;
      --only) only="${EXTRA[i + 1]}" ;;
    esac
  done
  [[ -z "${only}" || ",${only}," == *",pointgroup_ops,"* ]] || return 0
  [[ -z "${skip_at}" || ",${EXTRA[skip_at]}," != *",pointgroup_ops,"* ]] || return 0

  local probe='#include <google/dense_hash_map>'
  if g++ -x c++ -fsyntax-only - <<< "${probe}" >/dev/null 2>&1; then return 0; fi

  local dir
  for dir in "${CONDA_PREFIX:-}/include" "${HOME:-/nonexistent}/.local/include" \
             "${HOME:-/nonexistent}/include"; do
    [[ -f "${dir}/google/dense_hash_map" ]] || continue
    if CPATH="${dir}${CPATH:+:${CPATH}}" g++ -x c++ -fsyntax-only - <<< "${probe}" >/dev/null 2>&1; then
      info "sparsehash headers found in ${dir}"
      OUT_ENV+=("CPATH=${dir}${CPATH:+:${CPATH}}")
      return 0
    fi
  done

  warn "no sparsehash headers (pointgroup_ops): skipping that package"
  warn "  to build it without root: conda install -c conda-forge sparsehash, or"
  warn "  git clone https://github.com/sparsehash/sparsehash && cd sparsehash &&"
  warn "  ./configure --prefix=\$HOME/.local && make install   # then rerun this script"
  if [[ -n "${skip_at}" ]]; then
    EXTRA[skip_at]="${EXTRA[skip_at]},pointgroup_ops"
  else
    OUT_ARGS+=(--skip pointgroup_ops)
  fi
}

# printf %q is correct but escapes every comma and space; single quotes read
# better in a command meant to be copied. KEY=VALUE keeps its key bare.
shell_quote() {
  local w="$1" key=""
  if [[ "${w}" =~ ^([A-Za-z_][A-Za-z0-9_]*=)(.*)$ ]]; then
    key="${BASH_REMATCH[1]}"; w="${BASH_REMATCH[2]}"
  fi
  if [[ -n "${w}" && "${w}" =~ ^[A-Za-z0-9_./:,+@%=-]+$ ]]; then
    echo "${key}${w}"
  else
    echo "${key}'$(printf '%s' "${w}" | sed "s/'/'\\\\''/g")'"
  fi
}

main() {
  check_tools
  probe_gpu
  probe_env
  find_toolkits

  [[ -z "${GPU_CAPS}" ]] || log "GPU compute capability ${GPU_CAPS}${DRIVER_CUDA:+, driver runs CUDA ${DRIVER_CUDA}}"
  if [[ ${#TK_ORDER[@]} -gt 0 ]]; then
    local h; for h in "${TK_ORDER[@]}"; do log "toolkit nvcc ${TK_VERSION[${h}]} at ${h}"; done
  fi

  if [[ "${FORCE_ACCEL}" == "cpu" ]]; then
    pick_cpu
  elif [[ ${#TK_ORDER[@]} -gt 0 ]] && pick_cuda; then
    :
  elif [[ -n "${FORCE_ACCEL}" ]]; then
    die "no usable toolkit for --accel ${FORCE_ACCEL} on this host"
  elif [[ -n "${GPU_CAPS}" ]]; then
    error "this host has a GPU (compute ${GPU_CAPS}) but no CUDA toolkit here can build for it."
    error "Install one without root (the runfile leaves the driver alone), then rerun:"
    error "  sh cuda_<version>_linux.run --silent --toolkit --installpath=\$HOME/cuda-<version>"
    error "or build cpu wheels explicitly: ${SCRIPT_NAME} --accel cpu"
    exit 1
  else
    warn "no NVIDIA GPU and no usable CUDA toolkit: estimating a cpu build"
    pick_cpu
  fi

  pick_sparsehash

  local -a cmd=(build --native "${OUT_ARGS[@]}" "${EXTRA[@]}")
  local builder="${BUILDER}"
  [[ "${PWD}" != "${REPO_ROOT}" ]] || builder="./scripts/build_wheels.sh"

  # The builder prints its plan; show it so the estimate can be checked at a glance.
  env "${OUT_ENV[@]}" bash "${BUILDER}" matrix --native "${OUT_ARGS[@]}" "${EXTRA[@]}" >&2 || true

  local word line=""
  for word in "${OUT_ENV[@]}" "${builder}" "${cmd[@]}"; do
    line+="${line:+ }$(shell_quote "${word}")"
  done
  echo "${line}"

  if [[ "${RUN}" == "1" ]]; then
    exec env "${OUT_ENV[@]}" bash "${BUILDER}" "${cmd[@]}"
  fi
}

main
