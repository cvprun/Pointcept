#!/usr/bin/env bash
# ==============================================================================
# Pointcept universal wheel builder
#
# Builds every native dependency of Pointcept as a redistributable wheel, for an
# arbitrary matrix of (os x arch x accelerator x torch x python).
#
# The point of this script is arm64: spconv, cumm and the PyG companion
# packages (torch-scatter/sparse/cluster) publish x86_64 wheels only, and
# flash-attn publishes an sdist only. On aarch64 every one of them has to be
# compiled from source. This script decides per package whether a prebuilt
# wheel exists for the requested target and falls back to a source build when
# it does not, so the same command works on amd64 and arm64.
#
# The script is self-contained: it re-executes itself inside the build
# container (see `--in-container`), so there is exactly one file to maintain.
#
# Quick start
#   ./scripts/build_wheels.sh matrix --preset default     # show what would run
#   ./scripts/build_wheels.sh build --preset default      # build it
#   ./scripts/build_wheels.sh build --arch arm64 --accel cu128 --torch 2.9.1
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"

# ------------------------------------------------------------------------------
# Container engines
#
# Any Docker-compatible CLI works: docker, podman and nerdctl are probed in that
# order unless --engine says otherwise. They differ in three ways that matter
# here, all handled below:
#   * podman/nerdctl need fully qualified image names (no implicit docker.io)
#   * rootless engines already map container root onto the invoking user, so
#     chowning the output would hand it to an unusable subuid instead
#   * SELinux hosts need a relabel suffix on bind mounts
# ------------------------------------------------------------------------------
SUPPORTED_ENGINES=(docker podman nerdctl)
ENGINE=""           # resolved executable
ENGINE_KIND=""      # docker | podman | nerdctl
ENGINE_ROOTLESS="0"

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
DEFAULT_OS="ubuntu24.04"
DEFAULT_ARCH="amd64"
DEFAULT_ACCEL="cu128"
DEFAULT_TORCH="2.9.1"
DEFAULT_PYTHON="3.12"

# Every package this script knows how to produce, in dependency order.
ALL_PACKAGES=(
  pccm cumm spconv
  torch-scatter torch-sparse torch-cluster torch-geometric
  flash-attn ocnn swin3d
  pointops pointops2 pointgroup_ops pointseg pointrope
)

# Packages that are pure python (no compilation, arch independent).
PURE_PYTHON_PACKAGES=" pccm torch-geometric ocnn "

# Packages that live in this repository under libs/.
LOCAL_PACKAGES=" pointops pointops2 pointgroup_ops pointseg pointrope "

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_DIM=""; C_BOLD=""
fi

log()   { echo "${C_BLUE}[build]${C_RESET} $*" >&2; }
info()  { echo "${C_GREEN}[ ok  ]${C_RESET} $*" >&2; }
warn()  { echo "${C_YELLOW}[warn ]${C_RESET} $*" >&2; }
error() { echo "${C_RED}[error]${C_RESET} $*" >&2; }
debug() { [[ "${VERBOSE}" == "1" ]] && echo "${C_DIM}[debug]${C_RESET} $*" >&2 || true; }
die()   { error "$*"; exit 1; }

# ==============================================================================
# Target resolution helpers
#
# These translate a compact accelerator token (cu128, rocm6.4, cpu) into the
# base image, pip index and architecture flags needed for that target.
# ==============================================================================

# Every CUDA token this script knows, oldest first. Iterated when a capability
# has to be traced back to the toolkits that can emit it.
KNOWN_CUDA_ACCELS=(cu118 cu121 cu124 cu126 cu128 cu129 cu130 cu131 cu132)

# Map a `cuXYZ` token onto a concrete `nvidia/cuda` devel tag. Only tags that
# actually publish a linux/arm64 manifest are listed; keep this table in sync
# with `docker manifest inspect nvidia/cuda:<tag>-devel-<os>`.
cuda_full_version() {
  case "$1" in
    cu118) echo "11.8.0"  ;;
    cu121) echo "12.1.1"  ;;
    cu124) echo "12.4.1"  ;;
    cu126) echo "12.6.3"  ;;
    cu128) echo "12.8.1"  ;;
    cu129) echo "12.9.1"  ;;
    cu130) echo "13.0.3"  ;;
    cu131) echo "13.1.2"  ;;
    cu132) echo "13.2.1"  ;;
    *)     return 1       ;;
  esac
}

# PyTorch does not publish a wheel index for every CUDA minor release: the 13.x
# line goes cu130 -> cu132 with no cu131 at all. A toolkit without an index is
# still worth targeting -- CUDA guarantees minor version compatibility inside a
# major release, so nvcc 13.1 compiles extensions that load against the CUDA
# 13.0 runtime the torch wheel bundles. The toolkit token and the wheel index
# are therefore resolved separately, and this maps one onto the other.
torch_index_accel() {
  case "$1" in
    cu131) echo "cu130" ;;
    *)     echo "$1"    ;;
  esac
}

# The first torch release carrying wheels for each toolkit. An older torch has
# no entry in that index at all (cu130 starts at 2.9.0), so the pairing is
# dropped during validation rather than dying an hour later at install time.
cuda_min_torch() {
  case "$(torch_index_accel "$1")" in
    cu126) echo "2.6.0"  ;;
    cu128) echo "2.7.0"  ;;
    cu129) echo "2.8.0"  ;;
    cu130) echo "2.9.0"  ;;
    cu132) echo "2.12.0" ;;
    *)     echo ""       ;;
  esac
}

# `sort -V` only orders two versions correctly when they have the same number of
# components: it puts "2.9" before "2.9.0", which PEP 440 considers equal.
pad_version() {
  local v="$1" dots="${1//[^.]/}"
  while [[ "${#dots}" -lt 2 ]]; do v="${v}.0"; dots="${dots}."; done
  echo "${v}"
}

version_ge() {
  local a b; a="$(pad_version "$1")"; b="$(pad_version "$2")"
  [[ "${a}" == "${b}" ]] && return 0
  [[ "$(printf '%s\n%s\n' "${a}" "${b}" | sort -V | head -1)" == "${b}" ]]
}

# CUDA releases before 12.6 never shipped an ubuntu24.04 devel image.
cuda_supported_os() {
  case "$1" in
    cu118|cu121|cu124) echo "ubuntu22.04" ;;
    *)                 echo "ubuntu24.04 ubuntu22.04" ;;
  esac
}

accel_kind() {
  case "$1" in
    cu*)   echo "cuda" ;;
    rocm*) echo "rocm" ;;
    cpu)   echo "cpu"  ;;
    *)     return 1    ;;
  esac
}

base_image_for() {
  local accel="$1" os="$2"
  local kind; kind="$(accel_kind "${accel}")" || die "unknown accelerator: ${accel}"
  case "${kind}" in
    cuda)
      local full; full="$(cuda_full_version "${accel}")" \
        || die "unsupported CUDA token '${accel}' (known: ${KNOWN_CUDA_ACCELS[*]})"
      echo "nvidia/cuda:${full}-devel-${os}"
      ;;
    rocm)
      # rocm/dev-ubuntu-<ver>:<rocm>-complete carries the full HIP toolchain.
      local ver="${accel#rocm}"
      echo "rocm/dev-ubuntu-${os#ubuntu}:${ver}-complete"
      ;;
    cpu)
      echo "ubuntu:${os#ubuntu}"
      ;;
  esac
}

torch_index_url() {
  echo "https://download.pytorch.org/whl/$1"
}

# Every compute capability each toolkit's nvcc can emit code for, ascending.
# Read off `nvcc --list-gpu-arch` for each release rather than inferred, because
# neither end of the range moves predictably: Blackwell landed one family per
# minor release (12.8 brought sm_100/sm_120, 12.9 added sm_103/sm_121, sm_110
# only exists in 13.x), and CUDA 13 dropped Maxwell through Volta at the bottom.
# There are holes too -- sm_101 exists in 12.8 and 12.9 but not in 13.x -- so
# this is a membership test, not a floor-and-ceiling comparison.
#
# Asking nvcc for a target outside its list is a hard `nvcc fatal: Unsupported
# gpu architecture 'compute_NNN'` on the first CUDA source file, which is why
# both the defaults below and any --cuda-arch are checked against this.
cuda_supported_arches() {
  case "$1" in
    cu118) echo "3.5 3.7 5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0" ;;
    cu121|cu124|cu126)
           echo "5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0" ;;
    cu128) echo "5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 10.0 10.1 12.0" ;;
    cu129) echo "5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 10.0 10.1 10.3 12.0 12.1" ;;
    cu130|cu131|cu132)
           echo "7.5 8.0 8.6 8.7 8.8 8.9 9.0 10.0 10.3 11.0 12.0 12.1" ;;
    *)     return 1 ;;
  esac
}

cuda_arch_floor()   { cuda_supported_arches "$1" | awk '{print $1}';  }
cuda_arch_ceiling() { cuda_supported_arches "$1" | awk '{print $NF}'; }

# TORCH_CUDA_ARCH_LIST entries carry decoration a capability comparison must
# ignore: a trailing "+PTX", and the sm_90a / sm_100f family-variant suffixes.
normalize_arch() {
  local a="${1%%+*}"
  echo "${a%%[a-z]}"
}

cuda_supports_arch() {
  local list; list="$(cuda_supported_arches "$1")" || return 1
  [[ " ${list} " == *" $(normalize_arch "$2") "* ]]
}

# Capabilities read as sm_ names when the message is about devices rather than
# about the dotted form nvcc and TORCH_CUDA_ARCH_LIST take.
arch_names() {
  local cap
  local -a out=()
  for cap in "$@"; do out+=("sm_${cap//./}"); done
  echo "${out[*]}"
}

# Which known toolkits can target this capability -- what to name when the
# requested one cannot.
cuda_accels_supporting() {
  local cap="$1" accel
  local -a out=()
  for accel in "${KNOWN_CUDA_ACCELS[@]}"; do
    cuda_supports_arch "${accel}" "${cap}" && out+=("${accel}")
  done
  echo "${out[*]}"
}

# Default device-code targets. Wrong values here are the most common cause of
# "no kernel image is available for execution on the device" at runtime, so the
# lists are deliberately explicit per architecture rather than using `all`.
#
#   amd64 : datacenter + workstation NVIDIA parts
#   arm64 : Jetson Orin (8.7), GH200 (9.0), GB200 (10.0), Thor (11.0),
#           RTX Blackwell (12.0), DGX Spark / GB10 (12.1)
#
# Every list here has to be a subset of cuda_supported_arches for that toolkit,
# which is what splits the arm64 CUDA 12 entries: 12.6 stops at sm_90, 12.8
# reaches Blackwell but not GB10, and sm_121 needs 12.9 at the earliest. Only
# CUDA 13 has Thor (sm_110). Blackwell Ultra (8.8 / 10.3) is omitted from the
# defaults to keep build times sane; pass --cuda-arch when targeting a
# B300/GB300.
#
# Note that Blackwell splits into family-specific architectures: an sm_120 cubin
# does not load on an sm_121 device, so DGX Spark needs its own entry rather
# than riding on 12.0.
default_cuda_arch_list() {
  local accel="$1" arch="$2"
  if [[ "${arch}" == "arm64" ]]; then
    case "${accel}" in
      cu118|cu121|cu124) echo "7.2 8.7"                     ;;
      cu126)             echo "8.7 9.0"                     ;;
      cu128)             echo "8.7 9.0 10.0 12.0"           ;;
      cu129)             echo "8.7 9.0 10.0 12.0 12.1"      ;;
      cu130|cu131|cu132) echo "8.7 9.0 10.0 11.0 12.0 12.1" ;;
      *)                 echo "8.7 9.0 10.0 11.0 12.0 12.1" ;;
    esac
  else
    case "${accel}" in
      cu118)             echo "7.0 7.5 8.0 8.6"             ;;
      cu121|cu124)       echo "7.0 7.5 8.0 8.6 8.9 9.0"     ;;
      cu126)             echo "7.0 7.5 8.0 8.6 8.9 9.0"     ;;
      cu128|cu129)       echo "7.5 8.0 8.6 8.9 9.0 10.0 12.0" ;;
      cu130|cu131|cu132) echo "7.5 8.0 8.6 8.9 9.0 10.0 12.0" ;;
      *)                 echo "7.5 8.0 8.6 8.9 9.0 10.0 12.0" ;;
    esac
  fi
}

default_rocm_arch_list() {
  # MI200 (gfx90a), MI300 (gfx942), RDNA3 (gfx1100/1101/1102).
  echo "gfx90a;gfx942;gfx1100"
}

# ==============================================================================
# Argument parsing
# ==============================================================================
usage() {
  cat <<EOF
${C_BOLD}Pointcept universal wheel builder${C_RESET}

Builds native Pointcept dependencies as wheels across a device matrix. Packages
without a prebuilt wheel for the target (spconv/cumm/PyG on arm64, flash-attn
everywhere) are compiled from source automatically. Runs on docker, podman or
nerdctl, whichever is available.

${C_BOLD}USAGE${C_RESET}
  ${SCRIPT_NAME} <command> [options]

${C_BOLD}COMMANDS${C_RESET}
  build              Build wheels for every combination in the matrix
  matrix             Print the expanded matrix and exit (no build)
  image              Build a runnable Pointcept image per combination
  shell              Drop into an interactive container for one combination
  setup-qemu         Register QEMU binfmt handlers for cross-arch builds
  clean              Remove the output directory and build caches
  help               Show this message

${C_BOLD}MATRIX AXES${C_RESET} (comma separated; the cartesian product is built)
  --os        LIST   ubuntu24.04, ubuntu22.04            [${DEFAULT_OS}]
  --arch      LIST   amd64, arm64                        [${DEFAULT_ARCH}]
  --accel     LIST   cu118 cu121 cu124 cu126 cu128 cu129
                     cu130 cu131 cu132 rocm6.3 rocm6.4
                     cpu                                 [${DEFAULT_ACCEL}]
  --torch     LIST   PyTorch versions, e.g. 2.8.0,2.9.1   [${DEFAULT_TORCH}]
  --python    LIST   CPython versions, e.g. 3.11,3.12     [${DEFAULT_PYTHON}]
  --preset    NAME   default | full | arm64 | ci | local

${C_BOLD}PACKAGE SELECTION${C_RESET}
  --only      LIST   Build only these packages
  --skip      LIST   Skip these packages
  --list-packages    Print known package names and exit
  --force-source     Compile from source even if a prebuilt wheel exists
  --prefer-prebuilt  Reuse prebuilt wheels when available          [default]

${C_BOLD}BUILD TUNING${C_RESET}
  --cuda-arch LIST   Override TORCH_CUDA_ARCH_LIST, e.g. "8.9 9.0"
                     (also narrows flash-attn, which otherwise builds
                      sm_80/90/100/120 every time)
  --rocm-arch LIST   Override PYTORCH_ROCM_ARCH, e.g. "gfx90a;gfx942"
  --jobs      N      Parallel compile jobs (MAX_JOBS)      [nproc, capped at 16]
  --out       DIR    Output directory                              [wheelhouse]
  --engine    NAME   Container engine: docker | podman | nerdctl
                     [auto-detect, in that order; env PC_ENGINE]
  --remote-host URI  Build on a remote daemon, e.g. ssh://arm-box
                     (alias: --docker-host)
  --no-cache         Disable the shared build cache (downloads AND the venv;
                     every run then reprovisions torch from scratch)
  --keep-going       Continue to the next combination after a failure
  --dry-run          Print what would run without executing it
  -v, --verbose      Verbose logging (also streams compiler output)
  -h, --help         Show this message

${C_BOLD}PRESETS${C_RESET}
  default   amd64+arm64, cu128, torch ${DEFAULT_TORCH}, py ${DEFAULT_PYTHON}
  full      amd64+arm64 x cu126,cu128,cu130 x torch 2.8.0,2.9.1 x py 3.11,3.12
  arm64     arm64 only, cu126+cu128 (the source-build heavy path)
  ci        amd64, cu128, torch ${DEFAULT_TORCH}, py ${DEFAULT_PYTHON}, local libs only
  local     Host arch, cu128, local libs only (fastest sanity check)

${C_BOLD}EXAMPLES${C_RESET}
  # See the plan before spending an hour of nvcc time
  ${SCRIPT_NAME} matrix --preset full

  # arm64 wheels for a GH200 box, compiled from source where needed
  ${SCRIPT_NAME} build --arch arm64 --accel cu128 --cuda-arch "9.0"

  # Native arm64 build over SSH instead of QEMU (much faster)
  ${SCRIPT_NAME} build --arch arm64 --remote-host ssh://gh200-node

  # No docker on this box? podman works the same way
  ${SCRIPT_NAME} build --engine podman --preset default

  # Only the repository's own CUDA extensions
  ${SCRIPT_NAME} build --only pointops,pointgroup_ops,pointrope

  # One package per sitting on a slow box: same target, resumable, accumulating
  # into the same directory. The toolchain and torch are provisioned once.
  T="--arch arm64 --accel cu130 --torch 2.9.1 --cuda-arch 12.1"   # DGX Spark
  ${SCRIPT_NAME} build \${T} --only spconv        # pulls in pccm + cumm
  ${SCRIPT_NAME} build \${T} --only torch-sparse

${C_BOLD}NOTES${C_RESET}
  * Cross-arch builds need QEMU: run '${SCRIPT_NAME} setup-qemu' once. QEMU makes
    nvcc extremely slow; for a full arm64 matrix prefer --remote-host against a
    native aarch64 machine.
  * --remote-host runs entirely on the remote daemon: that host needs a checkout
    at this same path, and the wheels are written to its own <out> directory,
    so copy them back yourself (rsync/scp) when the build finishes.
  * Rootless engines (podman by default) cannot register binfmt handlers
    themselves; 'setup-qemu' prints the sudo / qemu-user-static alternatives.
    Output ownership is handled automatically either way.
  * Packages can be built one invocation at a time with --only, which is how a
    long arm64 matrix is made resumable. Every run is a separate container, so
    the toolchain, the venv and the torch install are kept in a cache volume and
    reused; the wheels accumulate in the same directory and manifest.txt is
    appended to rather than rewritten. 'clean' drops both.
  * Wheels land in <out>/linux-<arch>/<accel>/torch<ver>-cp<py>/ with a
    manifest.txt recording how each wheel was obtained.
  * 'build' checks the host driver and GPU against each CUDA target and warns
    when the wheels would not load here. No GPU is needed to build them, so the
    check never blocks; it is there for when you are building for this machine.
  * --preset local reads this machine's GPU and picks both the CUDA toolkit and
    the arch list from it, so it builds the smallest set of wheels that runs
    here. Any explicit --accel / --cuda-arch still wins.
  * A --cuda-arch (or default) target the chosen nvcc cannot emit drops that
    combination during validation: CUDA 12.6 stops at sm_90, 12.8 adds sm_100
    and sm_120, sm_121 (DGX Spark) needs 12.9+, and sm_110 (Thor) needs 13.x.
  * cumm and spconv accept a shorter arch list than nvcc does and reject the
    rest outright, so Thor, Blackwell Ultra and DGX Spark targets are built as
    the newest arch cumm knows plus PTX. Those kernels JIT on first load.
EOF
}

# Matrix axes
OS_LIST=""; ARCH_LIST=""; ACCEL_LIST=""; TORCH_LIST=""; PYTHON_LIST=""
PRESET=""
ONLY=""; SKIP=""
FORCE_SOURCE="0"
CUDA_ARCH_OVERRIDE=""; ROCM_ARCH_OVERRIDE=""
JOBS=""
OUT_DIR="${REPO_ROOT}/wheelhouse"
ENGINE_REQUESTED="${PC_ENGINE:-}"
DOCKER_HOST_OVERRIDE=""
USE_CACHE="1"
KEEP_GOING="0"
DRY_RUN="0"
VERBOSE="0"
IN_CONTAINER="0"
COMMAND=""

parse_args() {
  if [[ $# -eq 0 ]]; then usage; exit 0; fi

  case "${1:-}" in
    build|matrix|image|shell|setup-qemu|clean|help) COMMAND="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    --in-container) COMMAND="in-container"; IN_CONTAINER="1"; shift ;;
    -*) COMMAND="build" ;;
    *) die "unknown command '${1}' (try '${SCRIPT_NAME} help')" ;;
  esac

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --os)            OS_LIST="$2";              shift 2 ;;
      --arch)          ARCH_LIST="$2";            shift 2 ;;
      --accel|--cuda)  ACCEL_LIST="$2";           shift 2 ;;
      --torch)         TORCH_LIST="$2";           shift 2 ;;
      --python)        PYTHON_LIST="$2";          shift 2 ;;
      --preset)        PRESET="$2";               shift 2 ;;
      --only)          ONLY="$2";                 shift 2 ;;
      --skip)          SKIP="$2";                 shift 2 ;;
      --list-packages) printf '%s\n' "${ALL_PACKAGES[@]}"; exit 0 ;;
      --force-source)  FORCE_SOURCE="1";          shift   ;;
      --prefer-prebuilt) FORCE_SOURCE="0";        shift   ;;
      --cuda-arch)     CUDA_ARCH_OVERRIDE="$2";   shift 2 ;;
      --rocm-arch)     ROCM_ARCH_OVERRIDE="$2";   shift 2 ;;
      --jobs|-j)       JOBS="$2";                 shift 2 ;;
      --out)           OUT_DIR="$2";              shift 2 ;;
      --engine)        ENGINE_REQUESTED="$2";     shift 2 ;;
      # --docker-host is the historical spelling of --remote-host.
      --remote-host|--docker-host)
                       DOCKER_HOST_OVERRIDE="$2"; shift 2 ;;
      --no-cache)      USE_CACHE="0";             shift   ;;
      --keep-going)    KEEP_GOING="1";            shift   ;;
      --dry-run)       DRY_RUN="1";               shift   ;;
      -v|--verbose)    VERBOSE="1";               shift   ;;
      -h|--help)       usage; exit 0 ;;
      # Options consumed only by the in-container stage.
      --pkg-list)      PKG_LIST="$2";             shift 2 ;;
      *) die "unknown option '$1' (try '${SCRIPT_NAME} help')" ;;
    esac
  done
}

apply_preset() {
  case "${PRESET}" in
    "") ;;
    default)
      : "${ARCH_LIST:=amd64,arm64}"; : "${ACCEL_LIST:=cu128}"
      : "${TORCH_LIST:=${DEFAULT_TORCH}}"; : "${PYTHON_LIST:=${DEFAULT_PYTHON}}"
      ;;
    full)
      : "${ARCH_LIST:=amd64,arm64}"; : "${ACCEL_LIST:=cu126,cu128,cu130}"
      : "${TORCH_LIST:=2.8.0,2.9.1}"; : "${PYTHON_LIST:=3.11,3.12}"
      ;;
    # cu130 is not optional coverage here: Thor (sm_110) and DGX Spark (sm_121)
    # have no kernels at all in the CUDA 12 line, so without it the arm64 sweep
    # skips two shipping aarch64 parts.
    arm64)
      : "${ARCH_LIST:=arm64}"; : "${ACCEL_LIST:=cu126,cu128,cu130}"
      : "${TORCH_LIST:=${DEFAULT_TORCH}}"; : "${PYTHON_LIST:=${DEFAULT_PYTHON}}"
      ;;
    ci)
      : "${ARCH_LIST:=amd64}"; : "${ACCEL_LIST:=cu128}"
      : "${TORCH_LIST:=${DEFAULT_TORCH}}"; : "${PYTHON_LIST:=${DEFAULT_PYTHON}}"
      : "${ONLY:=pointops,pointops2,pointgroup_ops,pointseg,pointrope}"
      ;;
    # "local" means wheels that load on this machine, so the toolkit and the arch
    # list follow the GPU that is actually in it rather than a fixed default --
    # otherwise a host whose part the default cannot reach (a DGX Spark, say)
    # builds a full set of wheels it cannot run. Both fall back to the usual
    # defaults on a host with no NVIDIA driver.
    local)
      : "${ARCH_LIST:=$(host_arch)}"
      : "${TORCH_LIST:=${DEFAULT_TORCH}}"; : "${PYTHON_LIST:=${DEFAULT_PYTHON}}"
      : "${ACCEL_LIST:=$(accel_for_host_gpu cu128 "${TORCH_LIST%%,*}")}"
      : "${CUDA_ARCH_OVERRIDE:=$(host_gpu_caps || true)}"
      : "${ONLY:=pointops,pointops2,pointgroup_ops,pointseg,pointrope}"
      ;;
    *) die "unknown preset '${PRESET}' (default|full|arm64|ci|local)" ;;
  esac

  : "${OS_LIST:=${DEFAULT_OS}}"
  : "${ARCH_LIST:=${DEFAULT_ARCH}}"
  : "${ACCEL_LIST:=${DEFAULT_ACCEL}}"
  : "${TORCH_LIST:=${DEFAULT_TORCH}}"
  : "${PYTHON_LIST:=${DEFAULT_PYTHON}}"
}

host_arch() {
  case "$(uname -m)" in
    x86_64|amd64)   echo "amd64" ;;
    aarch64|arm64)  echo "arm64" ;;
    *) die "unsupported host architecture: $(uname -m)" ;;
  esac
}

split_list() { echo "$1" | tr ',' ' ' | tr -s ' '; }

# ==============================================================================
# Package selection
# ==============================================================================
selected_packages() {
  local -a chosen=()
  local pkg

  if [[ -n "${ONLY}" ]]; then
    for pkg in $(split_list "${ONLY}"); do
      [[ " ${ALL_PACKAGES[*]} " == *" ${pkg} "* ]] \
        || die "unknown package '${pkg}' (see --list-packages)"
    done
  fi

  for pkg in "${ALL_PACKAGES[@]}"; do
    if [[ -n "${ONLY}" ]]; then
      [[ " $(split_list "${ONLY}") " == *" ${pkg} "* ]] || continue
    fi
    if [[ -n "${SKIP}" ]]; then
      [[ " $(split_list "${SKIP}") " == *" ${pkg} "* ]] && continue
    fi
    chosen+=("${pkg}")
  done

  # spconv cannot be compiled without cumm, and cumm needs pccm. Pull the
  # dependencies in silently rather than failing halfway through the build.
  if [[ " ${chosen[*]} " == *" spconv "* ]]; then
    [[ " ${chosen[*]} " == *" cumm "* ]] || chosen=(cumm "${chosen[@]}")
  fi
  if [[ " ${chosen[*]} " == *" cumm "* ]]; then
    [[ " ${chosen[*]} " == *" pccm "* ]] || chosen=(pccm "${chosen[@]}")
  fi

  printf '%s\n' "${chosen[@]}"
}

# ==============================================================================
# Matrix expansion
#
# Emits one `os|arch|accel|torch|python` line per valid combination. Invalid
# combinations (a CUDA release with no image for that OS, ROCm on arm64, ...)
# are dropped here with a warning so `build` never starts work that cannot
# finish.
# ==============================================================================
expand_matrix() {
  local os arch accel torch python
  for os in $(split_list "${OS_LIST}"); do
    for arch in $(split_list "${ARCH_LIST}"); do
      for accel in $(split_list "${ACCEL_LIST}"); do
        for torch in $(split_list "${TORCH_LIST}"); do
          for python in $(split_list "${PYTHON_LIST}"); do
            validate_combo "${os}" "${arch}" "${accel}" "${torch}" "${python}" \
              && echo "${os}|${arch}|${accel}|${torch}|${python}"
          done
        done
      done
    done
  done
}

declare -A WARNED_INDEX_SUB=()
declare -A ARCH_LIST_VERDICT=()

# The effective TORCH_CUDA_ARCH_LIST has to be something this toolkit's nvcc
# accepts. Checking it here costs nothing; discovering it from nvcc costs the
# whole image pull, the torch install and the first CUDA source file. The
# verdict is cached per accel/arch because validate_combo runs once per point of
# the cartesian product and the answer does not depend on torch or python.
validate_cuda_arch_list() {
  local accel="$1" arch="$2"
  local key="${accel}|${arch}"
  [[ -n "${ARCH_LIST_VERDICT[${key}]:-}" ]] && return "${ARCH_LIST_VERDICT[${key}]}"

  # An accelerator with no entry in the table is not one to guess about.
  cuda_supported_arches "${accel}" >/dev/null || return 0

  local want source
  if [[ -n "${CUDA_ARCH_OVERRIDE}" ]]; then
    want="${CUDA_ARCH_OVERRIDE}"; source="--cuda-arch"
  else
    want="$(default_cuda_arch_list "${accel}" "${arch}")"; source="default ${arch} arch list"
  fi

  local a
  local -a bad=()
  for a in ${want}; do
    cuda_supports_arch "${accel}" "${a}" || bad+=("${a}")
  done

  if [[ ${#bad[@]} -gt 0 ]]; then
    ARCH_LIST_VERDICT["${key}"]=1
    local alt; alt="$(cuda_accels_supporting "${bad[0]}")"
    warn "skip: nvcc $(cuda_full_version "${accel}") cannot target $(arch_names "${bad[@]}"), asked for by the ${source} '${want}'"
    warn "  ${accel} spans $(arch_names "$(cuda_arch_floor "${accel}")") to $(arch_names "$(cuda_arch_ceiling "${accel}")"); $(arch_names "${bad[0]}") needs ${alt:-a toolkit this script does not know}"
    return 1
  fi

  ARCH_LIST_VERDICT["${key}"]=0
  return 0
}

validate_combo() {
  local os="$1" arch="$2" accel="$3" torch="$4" python="$5"
  local kind

  case "${arch}" in amd64|arm64) ;; *) warn "skip: unknown arch '${arch}'"; return 1 ;; esac

  kind="$(accel_kind "${accel}")" || { warn "skip: unknown accelerator '${accel}'"; return 1; }

  if [[ "${kind}" == "cuda" ]]; then
    cuda_full_version "${accel}" >/dev/null \
      || { warn "skip: no base image mapping for '${accel}'"; return 1; }
    if [[ " $(cuda_supported_os "${accel}") " != *" ${os} "* ]]; then
      warn "skip: ${accel} has no ${os} devel image (use $(cuda_supported_os "${accel}" | awk '{print $1}'))"
      return 1
    fi
    # Said once per accelerator, not once per combination: validate_combo runs
    # for every point of the cartesian product.
    local widx; widx="$(torch_index_accel "${accel}")"
    if [[ "${widx}" != "${accel}" && -z "${WARNED_INDEX_SUB[${accel}]:-}" ]]; then
      WARNED_INDEX_SUB["${accel}"]=1
      warn "${accel}: PyTorch publishes no ${accel} wheels; torch comes from ${widx} (nvcc stays ${accel})"
    fi
  fi

  # ROCm ships x86_64 only; there is no aarch64 ROCm userspace to build against.
  if [[ "${kind}" == "rocm" && "${arch}" == "arm64" ]]; then
    warn "skip: ROCm has no arm64 support (${accel}/${arch})"
    return 1
  fi

  # PyTorch stopped publishing aarch64 CUDA wheels for the older toolkits.
  if [[ "${kind}" == "cuda" && "${arch}" == "arm64" ]]; then
    case "${accel}" in
      cu118|cu121|cu124)
        warn "skip: PyTorch publishes no aarch64 wheels for ${accel} (use cu126+)"
        return 1
        ;;
    esac
  fi

  [[ "${torch}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] \
    || { warn "skip: malformed torch version '${torch}'"; return 1; }
  [[ "${python}" =~ ^3\.[0-9]+$ ]] \
    || { warn "skip: malformed python version '${python}'"; return 1; }

  if [[ "${kind}" == "cuda" ]]; then
    local min_torch; min_torch="$(cuda_min_torch "${accel}")"
    if [[ -n "${min_torch}" ]] && ! version_ge "${torch}" "${min_torch}"; then
      warn "skip: torch ${torch} predates $(torch_index_accel "${accel}") wheels (needs >= ${min_torch})"
      return 1
    fi
    validate_cuda_arch_list "${accel}" "${arch}" || return 1
  fi

  return 0
}

target_dir() {
  local arch="$1" accel="$2" torch="$3" python="$4"
  echo "linux-${arch}/${accel}/torch${torch}-cp${python//./}"
}

# ==============================================================================
# Host-side commands
# ==============================================================================
# Probe a candidate engine: it must exist and actually be able to talk to its
# backend. A docker CLI with a dead daemon should fall through to podman rather
# than fail the whole run.
engine_usable() {
  local bin="$1"
  command -v "${bin}" >/dev/null 2>&1 || return 1
  "${bin}" info >/dev/null 2>&1
}

detect_engine() {
  [[ -n "${ENGINE}" ]] && return 0

  local candidate
  if [[ -n "${ENGINE_REQUESTED}" ]]; then
    command -v "${ENGINE_REQUESTED}" >/dev/null 2>&1 \
      || die "requested engine '${ENGINE_REQUESTED}' is not on PATH"
    engine_usable "${ENGINE_REQUESTED}" \
      || die "'${ENGINE_REQUESTED}' is installed but not responding (daemon down? socket permissions?)"
    ENGINE="${ENGINE_REQUESTED}"
  else
    for candidate in "${SUPPORTED_ENGINES[@]}"; do
      if engine_usable "${candidate}"; then ENGINE="${candidate}"; break; fi
    done
    [[ -n "${ENGINE}" ]] || die "no usable container engine found (tried: ${SUPPORTED_ENGINES[*]}); install one or pass --engine"
  fi

  case "$(basename "${ENGINE}")" in
    docker*)  ENGINE_KIND="docker"  ;;
    podman*)  ENGINE_KIND="podman"  ;;
    nerdctl*) ENGINE_KIND="nerdctl" ;;
    *)        ENGINE_KIND="docker"  ;;   # assume docker-compatible CLI
  esac

  # Rootless engines remap container root to the invoking user already.
  case "${ENGINE_KIND}" in
    podman)
      [[ "$(${ENGINE} info --format '{{.Host.Security.Rootless}}' 2>/dev/null)" == "true" ]] \
        && ENGINE_ROOTLESS="1"
      ;;
    docker)
      ${ENGINE} info --format '{{println .SecurityOptions}}' 2>/dev/null | grep -q "name=rootless" \
        && ENGINE_ROOTLESS="1"
      ;;
  esac

  debug "engine=${ENGINE} kind=${ENGINE_KIND} rootless=${ENGINE_ROOTLESS}"
}

require_engine() {
  detect_engine
  local mode="rootful"; [[ "${ENGINE_ROOTLESS}" == "1" ]] && mode="rootless"
  log "engine: ${ENGINE} (${ENGINE_KIND}, ${mode})"
}

# docker silently prepends docker.io; podman and nerdctl do not and fail on
# unqualified names when no search registry is configured.
normalize_image() {
  local img="$1"
  [[ "${ENGINE_KIND}" == "docker" ]] && { echo "${img}"; return; }

  if [[ "${img}" != */* ]]; then
    echo "docker.io/library/${img}"
    return
  fi
  case "${img%%/*}" in
    *.*|*:*|localhost) echo "${img}" ;;          # already registry-qualified
    *)                 echo "docker.io/${img}" ;;
  esac
}

# SELinux blocks bind mounts unless the content is relabeled for the container.
mount_opts() {
  local base="$1"
  if [[ -d /sys/fs/selinux ]]; then
    [[ -n "${base}" ]] && echo "${base},z" || echo "z"
  else
    echo "${base}"
  fi
}

# Remote daemons are addressed by different variables per engine.
remote_env_var() {
  case "${ENGINE_KIND}" in
    podman) echo "CONTAINER_HOST" ;;
    *)      echo "DOCKER_HOST"    ;;
  esac
}

# Cross-architecture containers need binfmt_misc handlers registered on the
# host. This is a one-time, host-wide change, so it lives behind its own
# command instead of happening implicitly during a build.
cmd_setup_qemu() {
  require_engine
  local img; img="$(normalize_image "tonistiigi/binfmt")"

  log "registering QEMU binfmt handlers (host-wide, requires privileged container)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${ENGINE} run --privileged --rm ${img} --install all"
    return 0
  fi

  # binfmt_misc is global kernel state, so a rootless engine cannot write it
  # even with --privileged; the distro's qemu-user-static package is the way.
  if [[ "${ENGINE_ROOTLESS}" == "1" ]]; then
    warn "${ENGINE} is rootless and cannot register binfmt handlers itself"
    warn "use either of:"
    warn "  sudo ${ENGINE} run --privileged --rm ${img} --install all"
    warn "  sudo apt install qemu-user-static   # or dnf install qemu-user-static"
    return 1
  fi

  ${ENGINE} run --privileged --rm "${img}" --install all
  info "QEMU handlers installed; cross-arch builds are now possible"
  warn "QEMU-emulated nvcc is 10-30x slower than native; prefer --remote-host for a full arm64 matrix"
}

qemu_available_for() {
  local arch="$1"
  [[ "${arch}" == "$(host_arch)" ]] && return 0
  case "${arch}" in
    arm64) [[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]] ;;
    amd64) [[ -e /proc/sys/fs/binfmt_misc/qemu-x86_64  ]] ;;
    *) return 1 ;;
  esac
}

cmd_matrix() {
  local combos; combos="$(expand_matrix)"
  [[ -n "${combos}" ]] || die "matrix is empty after validation"

  local -a pkgs; mapfile -t pkgs < <(selected_packages)
  local n; n="$(echo "${combos}" | wc -l)"

  echo
  echo "${C_BOLD}Build matrix (${n} combination(s), ${#pkgs[@]} package(s) each)${C_RESET}"
  echo
  printf '%-14s %-7s %-9s %-9s %-7s %s\n' "OS" "ARCH" "ACCEL" "TORCH" "PYTHON" "BASE IMAGE"
  printf '%s\n' "$(printf '%.0s-' {1..92})"

  local os arch accel torch python
  while IFS='|' read -r os arch accel torch python; do
    [[ -n "${os}" ]] || continue
    printf '%-14s %-7s %-9s %-9s %-7s %s\n' \
      "${os}" "${arch}" "${accel}" "${torch}" "${python}" "$(base_image_for "${accel}" "${os}")"
  done <<< "${combos}"

  echo
  echo "${C_BOLD}Packages${C_RESET} (in build order)"
  local pkg
  for pkg in "${pkgs[@]}"; do
    local note=""
    [[ "${PURE_PYTHON_PACKAGES}" == *" ${pkg} "* ]] && note="pure python"
    [[ "${LOCAL_PACKAGES}"       == *" ${pkg} "* ]] && note="local (libs/${pkg})"
    printf '  %-16s %s\n' "${pkg}" "${C_DIM}${note}${C_RESET}"
  done

  echo
  echo "${C_BOLD}Output${C_RESET} ${OUT_DIR}/linux-<arch>/<accel>/torch<ver>-cp<py>/"
  echo

  # Warn about combinations that will need emulation.
  local arches; arches="$(echo "${combos}" | cut -d'|' -f2 | sort -u)"
  local a
  for a in ${arches}; do
    if ! qemu_available_for "${a}"; then
      warn "linux/${a} needs QEMU: run '${SCRIPT_NAME} setup-qemu' or pass --docker-host <native ${a} host>"
    fi
  done
}

cmd_clean() {
  # `--out` is user supplied and this rm is recursive, so refuse anything that
  # is not a plausible wheel output directory.
  [[ -n "${OUT_DIR}" && "${OUT_DIR}" != "/" ]] \
    || die "refusing to clean '${OUT_DIR}'"

  if [[ ! -d "${OUT_DIR}" ]]; then
    info "nothing to clean (${OUT_DIR} does not exist)"
  else
    local count; count="$(find "${OUT_DIR}" -name '*.whl' 2>/dev/null | wc -l)"
    log "removing ${OUT_DIR} (${count} wheel(s))"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "rm -rf ${OUT_DIR}"
      echo "<engine> volume rm pointcept-build-cache"
      return 0
    fi
    rm -rf "${OUT_DIR}"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    # Best effort: the cache volume only exists once a build has run.
    detect_engine 2>/dev/null \
      && "${ENGINE}" volume rm pointcept-build-cache >/dev/null 2>&1 || true
  fi
  info "cleaned"
}

# Assemble the `run` invocation shared by build/shell/image.
engine_run_args() {
  local os="$1" arch="$2" accel="$3" torch="$4" python="$5"
  local -a args=(run --rm --platform "linux/${arch}")

  args+=(-e "PC_OS=${os}" -e "PC_ARCH=${arch}" -e "PC_ACCEL=${accel}")
  args+=(-e "PC_TORCH=${torch}" -e "PC_PYTHON=${python}")
  args+=(-e "PC_FORCE_SOURCE=${FORCE_SOURCE}" -e "PC_VERBOSE=${VERBOSE}")
  args+=(-e "PC_JOBS=${JOBS:-$(default_jobs)}")
  args+=(-e "PC_TORCH_ACCEL=$(torch_index_accel "${accel}")")
  args+=(-e "PC_TORCH_INDEX=$(torch_index_url "$(torch_index_accel "${accel}")")")

  # Under a rootful engine the build runs as real root, so the wheels must be
  # handed back to the invoking user. Under a rootless engine container root is
  # ALREADY the invoking user, and chowning to their uid would instead map to an
  # unusable subuid (e.g. 100999) that the user cannot even read.
  if [[ "${ENGINE_ROOTLESS}" != "1" ]]; then
    args+=(-e "PC_HOST_UID=$(id -u)" -e "PC_HOST_GID=$(id -g)")
  fi

  local kind; kind="$(accel_kind "${accel}")"
  if [[ "${kind}" == "cuda" ]]; then
    args+=(-e "TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_OVERRIDE:-$(default_cuda_arch_list "${accel}" "${arch}")}")
  elif [[ "${kind}" == "rocm" ]]; then
    args+=(-e "PYTORCH_ROCM_ARCH=${ROCM_ARCH_OVERRIDE:-$(default_rocm_arch_list)}")
  fi

  # The repository is mounted read-only; libs/ are copied inside before build
  # so setuptools never writes build artifacts into the user's working tree.
  local ro rw; ro="$(mount_opts ro)"; rw="$(mount_opts "")"
  args+=(-v "${REPO_ROOT}:/src:${ro}")
  args+=(-v "${SCRIPT_PATH}:/builder.sh:${ro}")
  if [[ -n "${rw}" ]]; then
    args+=(-v "${OUT_DIR}/$(target_dir "${arch}" "${accel}" "${torch}" "${python}"):/out:${rw}")
  else
    args+=(-v "${OUT_DIR}/$(target_dir "${arch}" "${accel}" "${torch}" "${python}"):/out")
  fi

  # The volume holds far more than downloads: see container_init_cache. It is
  # what makes `--only <pkg>` runs cheap enough to build the matrix one package
  # at a time, since the venv and its torch install survive between them.
  if [[ "${USE_CACHE}" == "1" ]]; then
    args+=(-v "pointcept-build-cache:/cache")
    # The cache volume and the venv are on different filesystems, so uv cannot
    # hardlink between them; copy mode avoids a warning on every install.
    args+=(-e "UV_CACHE_DIR=/cache/uv" -e "PIP_CACHE_DIR=/cache/pip" -e "UV_LINK_MODE=copy")
  fi

  printf '%s\n' "${args[@]}"
}

default_jobs() {
  local n; n="$(nproc 2>/dev/null || echo 4)"
  # nvcc peaks around 2.5 GB per job; cap so a big machine does not OOM.
  [[ "${n}" -gt 16 ]] && n=16
  echo "${n}"
}

# ==============================================================================
# Host preflight
#
# The build never touches the host GPU: it runs in a container, nvcc cross
# compiles, and FORCE_CUDA makes the extensions emit device code with no device
# present. A host with no NVIDIA driver at all builds these wheels perfectly
# well. So none of this gates the build -- it exists for the common case where
# you are building for the machine you are sitting at, and answers whether the
# result would actually load on it.
# ==============================================================================

# Driver floor per CUDA major, from "CUDA Toolkit and Minimum Required Driver
# Version for CUDA Minor Version Compatibility" in the toolkit release notes.
cuda_driver_floor() {
  case "$1" in
    11) echo "450" ;;
    12) echo "525" ;;
    13) echo "580" ;;
    *)  echo ""    ;;
  esac
}

# One "driver,cap" line per GPU. compute_cap needs a reasonably recent
# nvidia-smi, so fall back to the driver alone when the field is rejected.
host_gpu_info() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null \
    || nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    || return 1
}

# The distinct compute capabilities in this machine, in nvidia-smi order. Fails
# rather than echoing nothing when there is no driver, or when nvidia-smi is too
# old for the compute_cap field: an empty arch list and an unknown one need
# different handling by the callers.
host_gpu_caps() {
  local info; info="$(host_gpu_info)" || return 1
  local line cap
  local -a caps=()
  while IFS= read -r line; do
    # Only the two-field form carries a capability; the driver-only fallback has
    # no comma, and `cut` would hand back the driver version as if it were one.
    [[ "${line}" == *,* ]] || continue
    cap="$(echo "${line}" | cut -d',' -f2 | tr -d ' ')"
    [[ "${cap}" =~ ^[0-9]+\.[0-9]+$ ]] || continue
    [[ " ${caps[*]} " == *" ${cap} "* ]] || caps+=("${cap}")
  done <<< "${info}"
  [[ ${#caps[@]} -gt 0 ]] || return 1
  echo "${caps[*]}"
}

accel_covers_caps() {
  local accel="$1" cap
  for cap in $2; do
    cuda_supports_arch "${accel}" "${cap}" || return 1
  done
}

# Which toolkit --preset local should build with. The preset's own default wins
# whenever it can emit code for every GPU here, so the common case keeps the
# toolkit it always used; only a part that default cannot reach moves the choice,
# and then to the newest toolkit that covers it, because the newest CUDA line is
# the one torch builds wheels for such a part against. Candidates are limited to
# tokens with a PyTorch index of their own and wheels for this torch version.
accel_for_host_gpu() {
  local fallback="$1" torch="$2"
  local caps; caps="$(host_gpu_caps)" || { echo "${fallback}"; return 0; }
  accel_covers_caps "${fallback}" "${caps}" && { echo "${fallback}"; return 0; }

  local i accel min_torch
  for (( i = ${#KNOWN_CUDA_ACCELS[@]} - 1; i >= 0; i-- )); do
    accel="${KNOWN_CUDA_ACCELS[i]}"
    [[ "$(torch_index_accel "${accel}")" == "${accel}" ]] || continue
    min_torch="$(cuda_min_torch "${accel}")"
    [[ -n "${min_torch}" ]] && version_ge "${torch}" "${min_torch}" || continue
    accel_covers_caps "${accel}" "${caps}" && { echo "${accel}"; return 0; }
  done

  # Nothing this script knows fits; the preflight warning below says why.
  echo "${fallback}"
}

declare -A PREFLIGHT_DONE=()

preflight_host_cuda() {
  local accel="$1" arch="$2"
  [[ "$(accel_kind "${accel}")" == "cuda" ]] || return 0

  # The answer only depends on the toolkit and the target architecture, so say
  # it once rather than once per torch/python combination.
  local key="${accel}|${arch}"
  [[ -n "${PREFLIGHT_DONE[${key}]:-}" ]] && return 0
  PREFLIGHT_DONE["${key}"]=1

  local info
  if ! info="$(host_gpu_info)" || [[ -z "${info}" ]]; then
    debug "no NVIDIA driver on this host; skipping the ${accel} runtime check"
    return 0
  fi

  local full major driver floor
  full="$(cuda_full_version "${accel}")"
  major="${full%%.*}"
  driver="$(echo "${info}" | head -1 | cut -d',' -f1 | tr -d ' ')"
  floor="$(cuda_driver_floor "${major}")"

  if [[ -n "${floor}" ]] && ! version_ge "${driver%%.*}" "${floor}"; then
    warn "host driver ${driver} predates the ${floor}+ that CUDA ${major}.x requires"
    warn "  the wheels still build, they just will not load on this host"
  else
    debug "host driver ${driver} satisfies CUDA ${major}.x (needs ${floor:-?}+)"
  fi

  # Comparing device code against the host GPU only means something when the
  # wheels are for the host's own CPU architecture.
  [[ "${arch}" == "$(host_arch)" ]] || return 0

  local caps; caps="$(host_gpu_caps)" || {
    debug "nvidia-smi reports no compute_cap; skipping the ${accel} arch check"
    return 0
  }

  local want; want="${CUDA_ARCH_OVERRIDE:-$(default_cuda_arch_list "${accel}" "${arch}")}"

  # Compare capabilities, not list entries: "12.0+PTX" covers an sm_120 device.
  local a cap
  local -a want_caps=() uncovered=()
  for a in ${want}; do want_caps+=("$(normalize_arch "${a}")"); done
  for cap in ${caps}; do
    [[ " ${want_caps[*]} " == *" ${cap} "* ]] || uncovered+=("${cap}")
  done

  [[ ${#uncovered[@]} -gt 0 ]] || return 0

  # Split the uncovered set three ways. Only capabilities this toolkit can emit
  # are a --cuda-arch away; the rest sit outside its range, and suggesting a flag
  # for those would send you into an `Unsupported gpu architecture` failure.
  local -a addable=() too_new=() too_old=()
  for cap in "${uncovered[@]}"; do
    if cuda_supports_arch "${accel}" "${cap}"; then
      addable+=("${cap}")
    elif version_ge "${cap}" "$(cuda_arch_ceiling "${accel}")"; then
      too_new+=("${cap}")
    else
      too_old+=("${cap}")
    fi
  done

  if [[ ${#addable[@]} -gt 0 ]]; then
    warn "this host's GPU ($(arch_names "${addable[@]}")) is not in the ${accel} arch list '${want}'"
    warn "  pass --cuda-arch \"${addable[*]}\" to get kernels that run here"
  fi
  if [[ ${#too_new[@]} -gt 0 ]]; then
    local alt; alt="$(cuda_accels_supporting "${too_new[0]}")"
    warn "this host's GPU ($(arch_names "${too_new[@]}")) is newer than CUDA ${full} can target, which tops out at $(arch_names "$(cuda_arch_ceiling "${accel}")")"
    warn "  --cuda-arch cannot reach it: build with ${alt:-a newer toolkit} instead"
  fi
  if [[ ${#too_old[@]} -gt 0 ]]; then
    warn "this host's GPU ($(arch_names "${too_old[@]}")) predates CUDA ${major}.x, which starts at $(arch_names "$(cuda_arch_floor "${accel}")")"
    warn "  build against an older toolkit for this machine"
  fi
}

cmd_build() {
  require_engine
  local combos; combos="$(expand_matrix)"
  [[ -n "${combos}" ]] || die "matrix is empty after validation"

  local -a pkgs; mapfile -t pkgs < <(selected_packages)
  [[ ${#pkgs[@]} -gt 0 ]] || die "no packages selected"
  local pkg_list; pkg_list="$(IFS=,; echo "${pkgs[*]}")"

  local total; total="$(echo "${combos}" | wc -l)"
  local index=0 failed=0
  local -a failures=()
  local -a produced=()

  log "building ${total} combination(s) x ${#pkgs[@]} package(s)"

  local os arch accel torch python
  while IFS='|' read -r os arch accel torch python; do
    [[ -n "${os}" ]] || continue
    index=$((index + 1))

    # The same string the wheels are written under, so the header names a path
    # that exists (cp312, not cp3.12).
    local tag; tag="$(target_dir "${arch}" "${accel}" "${torch}" "${python}")"
    echo
    echo "${C_BOLD}${C_BLUE}=== [${index}/${total}] ${tag} ===${C_RESET}" >&2

    preflight_host_cuda "${accel}" "${arch}"

    if ! qemu_available_for "${arch}" && [[ -z "${DOCKER_HOST_OVERRIDE}" ]]; then
      warn "linux/${arch} requires QEMU which is not registered"
      warn "run '${SCRIPT_NAME} setup-qemu', or build natively with --docker-host"
      failures+=("${tag} (no QEMU for linux/${arch})")
      failed=$((failed + 1))
      if [[ "${KEEP_GOING}" == "1" ]]; then
        continue
      fi
      return 1
    fi

    local base; base="$(normalize_image "$(base_image_for "${accel}" "${os}")")"
    local out_sub; out_sub="${OUT_DIR}/$(target_dir "${arch}" "${accel}" "${torch}" "${python}")"

    local -a dargs; mapfile -t dargs < <(engine_run_args "${os}" "${arch}" "${accel}" "${torch}" "${python}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "mkdir -p ${out_sub}"
      echo "${ENGINE} ${dargs[*]} ${base} bash /builder.sh --in-container --pkg-list ${pkg_list}"
      continue
    fi

    mkdir -p "${out_sub}"

    local -a engine_env=()
    if [[ -n "${DOCKER_HOST_OVERRIDE}" ]]; then
      engine_env=(env "$(remote_env_var)=${DOCKER_HOST_OVERRIDE}")
      # Bind mounts resolve on the daemon's filesystem, not this one: the remote
      # host needs the repository at this same path, and the wheels are written
      # to the remote's ${OUT_DIR}, to be collected from there afterwards.
      if [[ ${index} -eq 1 ]]; then
        warn "remote daemon ${DOCKER_HOST_OVERRIDE} (via $(remote_env_var)): mounts resolve remotely"
        warn "  the repo must exist at ${REPO_ROOT} on that host"
        warn "  wheels land in ${OUT_DIR} on that host (e.g. rsync them back)"
      fi
    fi

    if "${engine_env[@]}" "${ENGINE}" "${dargs[@]}" "${base}" \
        bash /builder.sh --in-container --pkg-list "${pkg_list}"; then
      info "${tag} done -> ${out_sub}"
      produced+=("${out_sub}")
    else
      error "${tag} failed"
      failures+=("${tag}")
      failed=$((failed + 1))
      [[ "${KEEP_GOING}" == "1" ]] || return 1
    fi
  done <<< "${combos}"

  echo
  if [[ "${DRY_RUN}" == "1" ]]; then
    info "dry run: ${total} combination(s) planned, nothing executed"
  elif [[ ${failed} -eq 0 ]]; then
    info "all ${total} combination(s) built -> ${OUT_DIR}"
    # List only what this run produced, not everything ever left in wheelhouse.
    local dir
    for dir in "${produced[@]}"; do
      find "${dir}" -name '*.whl' -printf "  ${dir#"${OUT_DIR}"/}/%f\n" 2>/dev/null | sort
    done
  else
    error "${failed}/${total} combination(s) failed:"
    printf '  - %s\n' "${failures[@]}" >&2
    return 1
  fi
}

cmd_shell() {
  require_engine
  local combos; combos="$(expand_matrix)"
  local first; first="$(echo "${combos}" | head -1)"
  [[ -n "${first}" ]] || die "matrix is empty after validation"

  local os arch accel torch python
  IFS='|' read -r os arch accel torch python <<< "${first}"
  local base; base="$(normalize_image "$(base_image_for "${accel}" "${os}")")"
  local out_sub; out_sub="${OUT_DIR}/$(target_dir "${arch}" "${accel}" "${torch}" "${python}")"
  mkdir -p "${out_sub}"

  local -a dargs; mapfile -t dargs < <(engine_run_args "${os}" "${arch}" "${accel}" "${torch}" "${python}")
  log "interactive shell for linux-${arch}/${accel}/torch${torch}-cp${python} (${base})"
  log "run 'bash /builder.sh --in-container --pkg-list <pkgs>' inside to build"
  "${ENGINE}" "${dargs[@]}" -it "${base}" bash
}

# `image` reuses the wheels produced by `build` to assemble a ready-to-run
# container, so the expensive compilation happens exactly once.
cmd_image() {
  require_engine
  local combos; combos="$(expand_matrix)"
  [[ -n "${combos}" ]] || die "matrix is empty after validation"

  local os arch accel torch python
  while IFS='|' read -r os arch accel torch python; do
    [[ -n "${os}" ]] || continue
    local sub; sub="$(target_dir "${arch}" "${accel}" "${torch}" "${python}")"
    local wheels="${OUT_DIR}/${sub}"

    if [[ ! -d "${wheels}" ]] || ! compgen -G "${wheels}/*.whl" >/dev/null; then
      warn "no wheels in ${wheels}; run '${SCRIPT_NAME} build' for this combination first"
      continue
    fi

    local base; base="$(normalize_image "$(base_image_for "${accel}" "${os}")")"
    local tag="pointcept/pointcept:torch${torch}-${accel}-py${python}-${arch}"
    local ctx; ctx="$(mktemp -d)"

    cp -r "${wheels}"/*.whl "${ctx}/"
    cat > "${ctx}/Dockerfile" <<EOF
FROM ${base}

ENV DEBIAN_FRONTEND=noninteractive \\
    PYTHONUNBUFFERED=1 \\
    PATH=/opt/venv/bin:\$PATH

RUN apt-get update \\
 && apt-get install -y --no-install-recommends \\
      ca-certificates curl git libgomp1 libopenblas0 tmux vim \\
 && rm -rf /var/lib/apt/lists/*

COPY *.whl /wheels/

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \\
 && /root/.local/bin/uv venv --python ${python} /opt/venv \\
 && /root/.local/bin/uv pip install --python /opt/venv/bin/python \\
      --index-url $(torch_index_url "$(torch_index_accel "${accel}")") \\
      torch==${torch} torchvision \\
 && /root/.local/bin/uv pip install --python /opt/venv/bin/python /wheels/*.whl \\
 && /root/.local/bin/uv pip install --python /opt/venv/bin/python \\
      h5py pyyaml tensorboard tensorboardx wandb yapf addict einops scipy \\
      plyfile termcolor timm ftfy regex tqdm matplotlib numpy peft \\
 && rm -rf /wheels

WORKDIR /workspace
EOF

    if [[ "${DRY_RUN}" == "1" ]]; then
      log "would build image ${tag} from ${ctx}"
      rm -rf "${ctx}"
      continue
    fi

    log "building image ${tag}"
    if "${ENGINE}" build --platform "linux/${arch}" -t "${tag}" "${ctx}"; then
      info "image ready: ${tag}"
    else
      error "image build failed: ${tag}"
    fi
    rm -rf "${ctx}"
  done <<< "${combos}"
}

# ==============================================================================
# In-container stage
#
# Everything below runs inside the target container, under the target
# architecture. It provisions a toolchain, installs the requested torch build,
# then produces one wheel per requested package into /out.
# ==============================================================================
PKG_LIST=""

c_log()  { echo "${C_BLUE}  ->${C_RESET} $*" >&2; }
c_ok()   { echo "${C_GREEN}  ok${C_RESET} $*" >&2; }
c_warn() { echo "${C_YELLOW}  !!${C_RESET} $*" >&2; }

MANIFEST="/out/manifest.txt"
record() { echo "$1" >> "${MANIFEST}"; }

# ------------------------------------------------------------------------------
# Persistent build environment
#
# Every `build` runs a fresh --rm container, so anything provisioned inside it is
# thrown away on exit. That is fine for a single sweep, but building the matrix
# one package at a time -- the only practical way to do it on a slow native arm64
# box, see --only -- would then pay for the apt install and a multi-gigabyte torch
# install on every single step.
#
# So the three preparation phases below write into the /cache volume the host
# mounts, and skip themselves outright when a previous run already filled it in:
#
#   /cache/apt     apt package lists and .debs
#   /cache/bin     the uv binary
#   /cache/python  uv-managed CPython builds -- a venv is a symlink farm into
#                  its interpreter, so that has to outlive the container too
#   /cache/venv/<key>   the build environment itself, torch and all
#
# The key spans every axis that changes what lands in the venv, so combinations
# of the matrix never share one. Under --no-cache nothing is mounted, CACHE_ROOT
# stays empty, and all three phases run in full against the container filesystem
# exactly as they did before.
# ------------------------------------------------------------------------------
CACHE_ROOT=""
VENV_DIR="/opt/venv"
PY=""

container_init_cache() {
  if [[ ! -d /cache ]]; then
    c_log "no cache volume mounted; provisioning the build environment from scratch"
    return 0
  fi
  CACHE_ROOT=/cache
  VENV_DIR="/cache/venv/${PC_OS}-${PC_ARCH}-${PC_ACCEL}-torch${PC_TORCH}-cp${PC_PYTHON//./}"
  # apt insists the partial/ subdirectories exist before it will use either path.
  mkdir -p /cache/apt/archives/partial /cache/apt/lists/partial /cache/bin /cache/python

  # uv keeps its managed interpreters outside UV_CACHE_DIR, and a venv is a
  # symlink farm into whichever one it was created from. `--python 3.12` gets a
  # uv download rather than the base image's 3.12, so without this the venv would
  # survive in the volume pointing at an interpreter the next container lacks.
  export UV_PYTHON_INSTALL_DIR=/cache/python
}

container_prepare_system() {
  c_log "installing system toolchain"
  export DEBIAN_FRONTEND=noninteractive
  local quiet="-qq"; [[ "${PC_VERBOSE}" == "1" ]] && quiet=""

  # nvidia/cuda images ship apt lists pinned to a repo that occasionally rotates
  # signing keys; dropping the extra sources keeps `apt-get update` reliable and
  # we never install CUDA packages from apt here anyway.
  rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list 2>/dev/null || true

  local redirect="/dev/null"; [[ "${PC_VERBOSE}" == "1" ]] && redirect="/dev/stderr"

  # The toolchain itself cannot be cached -- it installs into the container
  # filesystem -- but its inputs can. With the lists in the volume `update` is a
  # few InRelease checks instead of a 34 MB download, and with the archives there
  # `install` is a pure dpkg unpack.
  local -a apt_opts=()
  if [[ -n "${CACHE_ROOT}" ]]; then
    apt_opts=(-o "Dir::Cache::Archives=${CACHE_ROOT}/apt/archives"
              -o "Dir::State::Lists=${CACHE_ROOT}/apt/lists")
  fi

  apt-get "${apt_opts[@]}" update ${quiet} > "${redirect}"
  apt-get "${apt_opts[@]}" install -y --no-install-recommends ${quiet} \
    build-essential cmake ninja-build git curl ca-certificates \
    libopenblas-dev libsparsehash-dev pkg-config \
    > "${redirect}"
  c_ok "system toolchain ready"
}

container_prepare_python() {
  local uv_dir="/root/.local/bin"
  [[ -n "${CACHE_ROOT}" ]] && uv_dir="${CACHE_ROOT}/bin"
  export PATH="${uv_dir}:/root/.local/bin:${PATH}"

  if ! command -v uv >/dev/null 2>&1; then
    c_log "installing uv into ${uv_dir}"
    # The installer reads these from its own environment, and it is the piped
    # `sh` that runs it -- a `VAR=x curl ... | sh` prefix would set them on curl
    # and silently land uv back in ~/.local/bin, outside the cache volume.
    export UV_INSTALL_DIR="${uv_dir}" INSTALLER_NO_MODIFY_PATH=1
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 \
      || die "failed to install uv"
    hash -r
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH (${uv_dir})"
  fi

  PY="${VENV_DIR}/bin/python"
  if [[ -x "${PY}" ]] && "${PY}" -c 'import sys' >/dev/null 2>&1; then
    c_ok "reusing build venv at ${VENV_DIR}"
  else
    c_log "provisioning CPython ${PC_PYTHON} via uv"
    # A half-written venv from an interrupted run would fail the check above and
    # confuse `uv venv`, so start from nothing rather than trying to repair it.
    rm -rf "${VENV_DIR}"
    mkdir -p "$(dirname "${VENV_DIR}")"
    # A venv keeps the interpreter independent of whatever python the base image
    # happens to ship, which is what makes the python axis of the matrix work.
    # --seed is required: uv creates bare venvs, but `pip download` and
    # `pip wheel` are the front-ends used below and they must exist in the venv.
    uv venv --seed --python "${PC_PYTHON}" "${VENV_DIR}" >/dev/null 2>&1 \
      || die "uv could not provision CPython ${PC_PYTHON}"
  fi

  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
  c_ok "python $(${PY} -V 2>&1 | awk '{print $2}') at ${PY}"
}

container_install_torch() {
  # A reused venv already carries torch and its several gigabytes of CUDA wheels.
  # The local version suffix is what separates two builds of the same release
  # (2.9.1+cu130 vs 2.9.1+cu128), so it has to be part of the comparison.
  local have; have="$(${PY} -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  if [[ "${have}" == "${PC_TORCH}" || "${have}" == "${PC_TORCH}+"* ]]; then
    c_ok "torch ${have} already provisioned"
  else
    c_log "installing torch==${PC_TORCH} from ${PC_TORCH_INDEX}"
    uv pip install --python "${PY}" \
      --index-url "${PC_TORCH_INDEX}" \
      "torch==${PC_TORCH}" \
      || die "torch ${PC_TORCH} is not available for ${PC_ACCEL}/${PC_ARCH} on CPython ${PC_PYTHON}"
  fi

  # setup.py of every native package imports torch, so the build front-end needs
  # these regardless of build isolation. Cheap to re-run: uv audits an already
  # satisfied set in milliseconds.
  uv pip install --python "${PY}" setuptools wheel ninja packaging numpy >/dev/null

  local torch_ver cuda_ver
  torch_ver="$(${PY} -c 'import torch; print(torch.__version__)')"
  cuda_ver="$(${PY} -c 'import torch; print(torch.version.cuda or torch.version.hip or "cpu")')"
  c_ok "torch ${torch_ver} (device runtime: ${cuda_ver})"
  record "# torch=${torch_ver} accel=${PC_ACCEL} arch=${PC_ARCH} python=${PC_PYTHON}"
  if [[ -n "${PC_TORCH_ACCEL:-}" && "${PC_TORCH_ACCEL}" != "${PC_ACCEL}" ]]; then
    c_warn "torch was built for ${PC_TORCH_ACCEL} (runtime ${cuda_ver}) but nvcc here is ${PC_ACCEL}"
    record "# NOTE: compiled with the ${PC_ACCEL} toolkit against a ${PC_TORCH_ACCEL} torch"
  fi
}

# Try to fetch a prebuilt wheel for the current interpreter/platform. Returns 0
# and drops the wheel into /out when one exists, 1 when the package has to be
# compiled. This is the check that makes amd64 fast and arm64 correct.
try_prebuilt() {
  local spec="$1"; shift
  local -a extra_index=("$@")
  [[ "${PC_FORCE_SOURCE}" == "1" ]] && return 1

  local tmp; tmp="$(mktemp -d)"
  # `uv` has no `pip download`, so this uses the seeded pip. Running under the
  # target architecture means pip resolves wheels for exactly this platform,
  # which is what turns "does a prebuilt exist for arm64?" into a real answer.
  local -a args=("${PY}" -m pip download --no-deps --only-binary=:all: -d "${tmp}" "${spec}")
  local idx
  for idx in "${extra_index[@]}"; do args+=(--find-links "${idx}"); done

  local rc=0
  set +e
  "${args[@]}" >/dev/null 2>&1
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]] && compgen -G "${tmp}/*.whl" >/dev/null; then
    local name; name="$(basename "$(ls "${tmp}"/*.whl | head -1)")"
    cp "${tmp}"/*.whl /out/
    rm -rf "${tmp}"
    c_ok "prebuilt  ${name}"
    record "prebuilt  ${name}"
    return 0
  fi
  rm -rf "${tmp}"
  debug "no prebuilt wheel for ${spec} on ${PC_ARCH}"
  return 1
}

# Build a wheel from a source tree or a VCS/sdist spec into /out.
build_wheel() {
  local label="$1" src="$2"; shift 2
  local -a env_pairs=("$@")

  c_log "compiling ${label} (this is the slow path)"

  # pip builds into a staging directory, not straight into /out. /out is not
  # empty on a second run -- `--only spconv` rebuilds cumm by design, and any
  # package can be rerun -- and pointing --wheel-dir at it makes "did this build
  # produce anything?" unanswerable: a rebuilt wheel overwrites its predecessor
  # under the same name, and pip skips the work entirely when the file it was
  # asked for is already sitting there ("File was already downloaded ..."). A
  # directory that starts empty has neither problem: whatever lands in it is
  # this build's output, and it is moved into /out afterwards.
  local stage; stage="$(mktemp -d)"

  local -a cmd=(env "MAX_JOBS=${PC_JOBS}")
  [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]] && cmd+=("TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
  [[ -n "${PYTORCH_ROCM_ARCH:-}"    ]] && cmd+=("PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}")
  local kv
  for kv in "${env_pairs[@]}"; do cmd+=("${kv}"); done

  # --no-build-isolation: these setup.py files import the *installed* torch to
  # discover the CUDA/HIP toolchain, which an isolated env would not have.
  cmd+=("${PY}" -m pip wheel --no-deps --no-build-isolation --wheel-dir "${stage}" "${src}")

  local logfile="/tmp/${label//\//_}.log"
  local rc=0
  # errexit is explicitly suspended around the compile: a failing package must
  # be recorded and reported, not abort the whole matrix mid-flight.
  set +e
  if [[ "${PC_VERBOSE}" == "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "${logfile}"
    rc="${PIPESTATUS[0]}"
  else
    "${cmd[@]}" > "${logfile}" 2>&1
    rc=$?
  fi
  set -e

  if [[ ${rc} -ne 0 ]]; then
    rm -rf "${stage}"
    c_warn "${label} failed; last 40 lines of ${logfile}:"
    tail -40 "${logfile}" >&2
    return 1
  fi

  if ! compgen -G "${stage}/*.whl" >/dev/null; then
    rm -rf "${stage}"
    c_warn "${label}: pip reported success but produced no wheel"
    return 1
  fi

  local name; name="$(basename "$(ls -t "${stage}"/*.whl | head -1)")"
  mv -f "${stage}"/*.whl /out/
  rm -rf "${stage}"

  c_ok "compiled  ${name}"
  record "compiled  ${name}"
  return 0
}

# Wheels land in /out; they are not installed. That is right for the output of a
# build, but two packages here are also build *inputs*: cumm's setup.py imports
# pccm at module scope and spconv's imports both. --no-build-isolation means pip
# provisions nothing, so setup.py sees only what is already in the venv -- and a
# missing one surfaces as a bare `ModuleNotFoundError` out of a pyproject hook,
# thousands of lines into a log, with no hint that a sibling package is the fix.
#
# The wheel this run just produced is preferred over anything on an index: it was
# built for this exact target, and for cumm that is the whole point -- a released
# cumm carries a different (or no) CUDA build and would have spconv generate the
# wrong kernels. So `fallback` is only passed for packages where an index copy is
# genuinely interchangeable.
install_build_dep() {
  local pkg="$1" dist="$2" fallback="${3-}"

  # `pkg` is what --only calls it; `dist` is what its setup.py names the
  # distribution, and the two diverge exactly where it matters: CUMM_CUDA_VERSION
  # renames cumm to cumm-cu130, which PEP 427 then escapes into the filename
  # cumm_cu130-0.8.2-....whl. Globbing the package name finds nothing at all.
  local esc="${dist//[-._]/_}"

  local whl=""
  compgen -G "/out/${esc}-*.whl" >/dev/null \
    && whl="$(ls -t /out/"${esc}"-*.whl | head -1)"

  # No --no-deps: pccm and cumm are unimportable without theirs (ccimport,
  # pybind11, fire), and those are pure python, so resolving them from the index
  # is harmless. --reinstall-package pins the name to *this* wheel even when a
  # same-version copy is already in a venv carried over from an earlier step.
  if [[ -n "${whl}" ]] \
     && uv pip install --python "${PY}" --reinstall-package "${dist}" "${whl}" >/dev/null 2>&1; then
    c_ok "build dep ${dist} <- $(basename "${whl}")"
    return 0
  fi

  if [[ -n "${fallback}" ]] \
     && uv pip install --python "${PY}" "${fallback}" >/dev/null 2>&1; then
    c_ok "build dep ${dist} <- ${fallback} (index)"
    return 0
  fi

  c_warn "${dist} is required to build this package but could not be installed"
  [[ -z "${whl}" ]] && c_warn "  no ${esc}-*.whl in /out; build it first (--only ${pkg})"
  return 1
}

pyg_find_links() {
  # PyG publishes per-(torch, cuda) wheel indices, x86_64/win only. The index is
  # keyed by the torch build's CUDA version, which is not always the toolkit in
  # this container (see torch_index_accel).
  local t="${PC_TORCH}" a="${PC_TORCH_ACCEL:-${PC_ACCEL}}"
  [[ "${a}" == cu* ]] || { echo ""; return; }
  echo "https://data.pyg.org/whl/torch-${t}+${a}.html"
}

pkg_pccm() {
  try_prebuilt "pccm" && return 0
  build_wheel "pccm" "pccm"
}

# spconv 2.x and its cumm backend are CUDA-only; there is no HIP path.
spconv_family_supported() {
  case "${PC_ACCEL}" in
    rocm*)
      c_warn "$1 skipped: spconv/cumm have no ROCm backend (target is ${PC_ACCEL})"
      record "skipped   $1 (ROCm target)"
      return 1
      ;;
  esac
  return 0
}

# CUMM_CUDA_VERSION only names the wheel (cumm-cu128) and selects a CPU-only
# build when empty; cumm strips the dots itself, so "128" and "12.8" agree.
cumm_cuda_version() {
  case "${PC_ACCEL}" in
    cu*) echo "${PC_ACCEL#cu}" ;;
    *)   echo ""               ;;   # empty => CPU-only build
  esac
}

# cumm keeps its own table of acceptable targets (supported_arches in
# cumm/common.py) and raises `Unknown CUDA arch (X) or GPU not supported` from
# setup.py for anything outside it, so what nvcc can compile is not the last word
# here. The table below is that list, and it knows nothing about Thor (11.0),
# Blackwell Ultra (8.8 / 10.3) or GB10 (12.1) -- the parts this script targets
# for arm64 CUDA 13.
cumm_supported_arches() {
  echo "3.5 3.7 5.0 5.2 5.3 6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 10.0 12.0"
}

# Map one capability onto the closest thing cumm accepts: itself when the table
# has it, and otherwise the newest entry below it, asked for as PTX so the driver
# JIT compiles it for the real device on first load. A cubin cannot cross a
# Blackwell family boundary, but PTX can, which is what makes sm_121 reachable
# through 12.0+PTX. Fails only when every entry is newer than the target.
cumm_arch_for() {
  local want; want="$(normalize_arch "$1")"
  local a best=""
  for a in $(cumm_supported_arches); do
    [[ "${a}" == "${want}" ]] && { echo "${a}"; return 0; }
    version_ge "${want}" "${a}" && best="${a}"
  done
  [[ -n "${best}" ]] || return 1
  echo "${best}+PTX"
}

# cumm/spconv take a semicolon-separated arch list, unlike torch's space form.
cumm_arch_list() {
  local -A want_ptx=()
  local -a bases=()
  local a mapped base
  for a in ${TORCH_CUDA_ARCH_LIST:-}; do
    if ! mapped="$(cumm_arch_for "${a}")"; then
      c_warn "cumm/spconv have no arch at or below ${a}; dropped from their build"
      continue
    fi
    base="${mapped%+PTX}"
    [[ " ${bases[*]} " == *" ${base} "* ]] || bases+=("${base}")
    # PTX whenever the mapping had to move, or the caller asked for it. Emitting
    # both "12.0" and "12.0+PTX" would hand nvcc the same gencode twice.
    [[ "${mapped}" == *+PTX || "${a}" == *+PTX ]] && want_ptx["${base}"]=1
  done

  local -a out=()
  for base in "${bases[@]}"; do
    if [[ -n "${want_ptx[${base}]:-}" ]]; then out+=("${base}+PTX"); else out+=("${base}"); fi
  done
  (IFS=';'; echo "${out[*]}")
}

# The list to build cumm and spconv with, or a skip when nothing in this target
# survives the mapping. Announced like flash-attn's, because a source build that
# quietly compiles for a different arch than asked for is worth seeing.
cumm_arch_list_or_skip() {
  local name="$1" archs
  archs="$(cumm_arch_list)"
  if [[ "${PC_ACCEL}" == cu* && -z "${archs}" ]]; then
    c_warn "${name} skipped: nothing in '${TORCH_CUDA_ARCH_LIST:-}' maps onto an arch cumm accepts"
    record "skipped   ${name} (no cumm-compatible arch in ${TORCH_CUDA_ARCH_LIST:-n/a})"
    return 1
  fi
  [[ -n "${archs}" ]] && c_log "cumm arch set: ${archs} (from ${TORCH_CUDA_ARCH_LIST})"
  echo "${archs}"
}

pkg_cumm() {
  spconv_family_supported "cumm" || return 0

  # cumm publishes cumm-cuXYZ wheels for x86_64 only, and only up to cu126.
  # Everywhere else (all of arm64, and cu128+) it has to be compiled.
  local variant="cumm"
  [[ "${PC_ACCEL}" == cu* ]] && variant="cumm-${PC_ACCEL}"

  if [[ "${PC_ARCH}" == "amd64" ]] && try_prebuilt "${variant}"; then return 0; fi

  local archs; archs="$(cumm_arch_list_or_skip cumm)" || return 0

  # pccm is imported by cumm's setup.py, so it has to be in the venv, not just
  # in /out. The index copy is fine here: pccm is pure python and target neutral.
  install_build_dep pccm pccm pccm || return 1

  # cumm has to build against its own source tree, and its setup.py reaches for
  # that tree with `sys.path.append` rather than an insert -- so an already
  # installed cumm sits ahead of it and wins the import. pccm then reads the
  # pccm classes out of site-packages, cannot express those paths relative to
  # the namespace root the extension was handed, and falls back to their fully
  # qualified names (pccm/core/__init__.py, extract_module_id_of_class). Every
  # binding lands a level too deep: cumm.core_cc.cumm.tensorview_bind instead of
  # cumm.core_cc.tensorview_bind. That wheel compiles, installs and imports
  # without a word, and only spconv's setup.py finds out. The venv outlives an
  # --only run and pkg_spconv leaves cumm installed in it, so this would spoil
  # every rebuild after the first. Hand the build a venv with no cumm in it.
  local d
  for d in cumm "${variant}"; do
    uv pip uninstall --python "${PY}" "${d}" >/dev/null 2>&1 || true
  done

  build_wheel "cumm" "git+https://github.com/FindDefinition/cumm.git" \
    "CUMM_DISABLE_JIT=1" \
    "CUMM_CUDA_ARCH_LIST=${archs}" \
    "CUMM_CUDA_VERSION=$(cumm_cuda_version)"
}

# spconv's setup.py caps cumm at `<0.8.0`. That ceiling was written on
# 2024-12-15 -- the date of spconv's last commit -- while cumm 0.8.0 landed
# three months later, so it guards against a minor that did not exist yet
# rather than one that was tried and rejected. cumm 0.8.0 is a pure version
# bump: its diff against 0.7.13 is CHANGELOG.md and version.txt and nothing
# else. It exists so *published* spconv wheels would not pair with cumm
# prebuilts from a CI whose gcc had changed -- the two extensions hand pybind11
# objects across the module boundary, which is ABI-sensitive. Here one
# toolchain compiles both in one container, so that pairing holds by
# construction. And the ceiling has to lift regardless: only cumm 0.8.x knows
# the Blackwell arches (10.0, 12.0) that a cu128+ target needs.
SPCONV_REPO="${PC_SPCONV_REPO:-https://github.com/traveller59/spconv.git}"

# Clone spconv and raise that ceiling to the next minor above the cumm this
# build will link against. Echoes the path to the patched tree.
spconv_patched_source() {
  local cumm_dist="$1"
  local work="/tmp/spconv-src"

  rm -rf "${work}"
  git clone --quiet --depth 1 "${SPCONV_REPO}" "${work}" || {
    c_warn "spconv: could not clone ${SPCONV_REPO}"
    return 1
  }

  # Read the version off the installed distribution rather than by importing
  # cumm: `cumm-cu130` is the name the pin is written against, and metadata
  # needs no CUDA runtime to answer. install_build_dep gave cumm no index
  # fallback, so this is the wheel pkg_cumm just built for this target.
  local ver ceiling
  ver="$(${PY} -c "from importlib.metadata import version; print(version('${cumm_dist}'))" 2>/dev/null)" || {
    c_warn "spconv: ${cumm_dist} reports no version; cannot size the pin"
    return 1
  }
  ceiling="$(awk -F. '{print $1"."($2+1)".0"}' <<< "${ver}")"

  # Both branches of setup.py's `if cuda_ver:` carry the same specifier -- one
  # for cumm-cuXYZ, one for plain cumm -- so this is a global substitution.
  sed -i -E "s/(cumm[^\"]*>=[0-9.]+, *<)[0-9]+\.[0-9]+\.[0-9]+/\1${ceiling}/g" \
    "${work}/setup.py"

  # A silent miss would rebuild the exact wheel that cannot be installed, and
  # nothing would say so until pip ran against the finished wheelhouse.
  grep -q "cumm[^\"]*<${ceiling}" "${work}/setup.py" || {
    c_warn "spconv: no cumm pin matched in setup.py; upstream changed its shape"
    return 1
  }

  c_ok "spconv: cumm ceiling raised to <${ceiling} (built against ${cumm_dist} ${ver})"
  record "# spconv: cumm pin relaxed to <${ceiling} (built against ${cumm_dist} ${ver})"
  echo "${work}"
}

pkg_spconv() {
  spconv_family_supported "spconv" || return 0

  local variant="spconv"
  [[ "${PC_ACCEL}" == cu* ]] && variant="spconv-${PC_ACCEL}"

  if [[ "${PC_ARCH}" == "amd64" ]] && try_prebuilt "${variant}"; then return 0; fi

  local archs; archs="$(cumm_arch_list_or_skip spconv)" || return 0

  # setup.py imports both. cumm gets no index fallback: only the wheel built
  # above knows this target's CUDA version and arch list -- and it carries the
  # same accelerator suffix pkg_cumm compiled it under.
  local cumm_dist="cumm"
  [[ "${PC_ACCEL}" == cu* ]] && cumm_dist="cumm-${PC_ACCEL}"

  install_build_dep pccm pccm pccm || return 1
  install_build_dep cumm "${cumm_dist}" || return 1

  # The one thing spconv needs from cumm that a broken build still installs
  # cleanly (see pkg_cumm): the binding submodule. Asking here costs a python
  # startup and turns a traceback thousands of lines into a compile log into a
  # sentence naming the package that has to be rebuilt.
  if ! "${PY}" -c 'from cumm.core_cc import tensorview_bind' >/dev/null 2>&1; then
    c_warn "${cumm_dist} has no core_cc.tensorview_bind; its bindings were generated"
    c_warn "  against an installed cumm rather than its own source. Rebuild it with"
    c_warn "  --only cumm, which now clears the venv first."
    return 1
  fi

  # A patched checkout rather than the git+ URL: what pip resolves later is the
  # metadata baked into the wheel, and upstream's ceiling makes that wheel
  # uninstallable beside the cumm it was just compiled against.
  c_log "spconv source: ${SPCONV_REPO}"
  record "# spconv source: ${SPCONV_REPO}"
  local src; src="$(spconv_patched_source "${cumm_dist}")" || return 1

  build_wheel "spconv" "${src}" \
    "SPCONV_DISABLE_JIT=1" \
    "CUMM_CUDA_ARCH_LIST=${archs}" \
    "CUMM_CUDA_VERSION=$(cumm_cuda_version)"
}

pkg_pyg_ext() {
  local name="$1" repo="$2"
  local links; links="$(pyg_find_links)"

  if [[ -n "${links}" ]] && try_prebuilt "${name}" "${links}"; then return 0; fi
  if try_prebuilt "${name}"; then return 0; fi

  # FORCE_CUDA makes the extension compile device code even though no GPU is
  # visible inside the build container.
  local -a env_pairs=()
  case "${PC_ACCEL}" in
    cu*)   env_pairs+=("FORCE_CUDA=1") ;;
    rocm*) env_pairs+=("FORCE_ONLY_CUDA=0" "FORCE_CUDA=1") ;;
    cpu)   env_pairs+=("FORCE_ONLY_CPU=1") ;;
  esac
  build_wheel "${name}" "git+https://github.com/${repo}.git" "${env_pairs[@]}"
}

pkg_torch_scatter() { pkg_pyg_ext "torch-scatter" "rusty1s/pytorch_scatter"; }
pkg_torch_sparse()  { pkg_pyg_ext "torch-sparse"  "rusty1s/pytorch_sparse";  }
pkg_torch_cluster() { pkg_pyg_ext "torch-cluster" "rusty1s/pytorch_cluster"; }

pkg_torch_geometric() {
  try_prebuilt "torch-geometric" && return 0
  build_wheel "torch-geometric" "torch-geometric"
}

# flash-attn 2.x ships kernels for exactly four targets -- sm_80, sm_90, sm_100
# and sm_120 -- and picks them from its own FLASH_ATTN_CUDA_ARCHS variable, not
# from TORCH_CUDA_ARCH_LIST. Left alone it compiles all four no matter what
# --cuda-arch says, which is the single most expensive mistake in this script.
#
# The mapping below is not a filter: an sm_80 cubin also runs on sm_86/87/89,
# because cubins stay binary compatible across a major compute revision. The
# Blackwell entries are the exception, which is why 10.3 and 12.1 map to
# nothing -- upstream torch keeps them as separate targets (see named_arches in
# torch/utils/cpp_extension.py), so an sm_100/sm_120 cubin does not cover them.
flash_attn_arch_for() {
  case "$1" in
    8.0|8.6|8.7|8.9) echo "80"  ;;
    9.0)             echo "90"  ;;
    10.0)            echo "100" ;;
    12.0)            echo "120" ;;
    *)               echo ""    ;;
  esac
}

flash_attn_archs() {
  local -a out=()
  local a code
  for a in ${TORCH_CUDA_ARCH_LIST:-}; do
    code="$(flash_attn_arch_for "${a%%+*}")"
    [[ -n "${code}" ]] || continue
    [[ " ${out[*]} " == *" ${code} "* ]] || out+=("${code}")
  done
  (IFS=';'; echo "${out[*]}")
}

pkg_flash_attn() {
  if [[ "${PC_ACCEL}" != cu* ]]; then
    c_warn "flash-attn skipped: requires CUDA (target is ${PC_ACCEL})"
    record "skipped   flash-attn (non-CUDA target)"
    return 0
  fi

  local archs; archs="$(flash_attn_archs)"
  if [[ -z "${archs}" ]]; then
    c_warn "flash-attn skipped: no target in '${TORCH_CUDA_ARCH_LIST:-}' has flash-attn 2.x kernels"
    c_warn "  it supports sm_80/90/100/120 only; Turing, Orin-only and Thor builds have nothing to compile"
    record "skipped   flash-attn (no supported arch in ${TORCH_CUDA_ARCH_LIST:-n/a})"
    return 0
  fi
  c_log "flash-attn arch set: ${archs} (from ${TORCH_CUDA_ARCH_LIST})"

  # PyPI carries an sdist only, so this always compiles. It is by far the
  # longest step; MAX_JOBS and the arch set above are what keep it bounded.
  build_wheel "flash-attn" "git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3.post1" \
    "FLASH_ATTENTION_FORCE_BUILD=TRUE" \
    "FLASH_ATTN_CUDA_ARCHS=${archs}"
}

pkg_ocnn() {
  try_prebuilt "ocnn" && return 0
  build_wheel "ocnn" "git+https://github.com/octree-nn/ocnn-pytorch.git"
}

# microsoft/Swin3D is a single commit from June 2023 and does not compile against
# a current PyTorch: its AT_DISPATCH_* calls pass tensor.type(), whose
# at::DeprecatedTypeProperties overload has been removed, and it includes
# <THC/THCAtomics.cuh> from the THC library that PyTorch deleted (what survives
# is a shim header marked for removal). The fork below carries those fixes plus
# the equivalent updates on the python side. Point this at your own fork or back
# at upstream by editing the URL.
SWIN3D_REPO="${PC_SWIN3D_REPO:-https://github.com/cvprun/Swin3D.git}"

pkg_swin3d() {
  if [[ "${PC_ACCEL}" != cu* ]]; then
    c_warn "swin3d skipped: requires CUDA (target is ${PC_ACCEL})"
    record "skipped   swin3d (non-CUDA target)"
    return 0
  fi
  c_log "swin3d source: ${SWIN3D_REPO}"
  record "# swin3d source: ${SWIN3D_REPO}"
  build_wheel "swin3d" "git+${SWIN3D_REPO}"
}

# Local extensions under libs/. The tree is copied out of the read-only mount
# first so setuptools can drop build/ and *.egg-info next to the sources.
pkg_local() {
  local name="$1"
  local src="/src/libs/${name}"
  [[ -d "${src}" ]] || { c_warn "${name}: ${src} not found in the repository"; return 1; }

  # pointseg is a CppExtension and builds without a device toolchain; the rest
  # need CUDA or HIP.
  if [[ "${name}" != "pointseg" && "${PC_ACCEL}" == "cpu" ]]; then
    c_warn "${name} skipped: needs CUDA/ROCm (target is cpu)"
    record "skipped   ${name} (cpu target)"
    return 0
  fi

  local work="/tmp/libs/${name}"
  mkdir -p "$(dirname "${work}")"
  rm -rf "${work}"
  cp -r "${src}" "${work}"

  # pointgroup_ops includes <google/dense_hash_map>, which libsparsehash-dev
  # installs into /usr/include -- already on the default search path.
  build_wheel "${name}" "${work}"
}

run_in_container() {
  : "${PC_OS:?}"; : "${PC_ARCH:?}"; : "${PC_ACCEL:?}"
  : "${PC_TORCH:?}"; : "${PC_PYTHON:?}"
  : "${PC_JOBS:=4}"; : "${PC_VERBOSE:=0}"; : "${PC_FORCE_SOURCE:=0}"
  [[ -n "${PKG_LIST}" ]] || die "--pkg-list is required in container mode"

  # The host passes verbosity through the environment, not argv, so wire it
  # into the shared VERBOSE that debug() reads.
  VERBOSE="${PC_VERBOSE}"

  mkdir -p /out
  # Appended, never truncated: --only makes "one package per invocation" a
  # supported workflow, and each of those is a separate container writing to the
  # same directory. Truncating would leave the manifest describing the last
  # package built rather than the wheels actually sitting next to it.
  record ""
  record "# $(date -u '+%Y-%m-%d %H:%M:%SZ') build_wheels.sh linux/${PC_ARCH} ${PC_ACCEL} torch${PC_TORCH} cp${PC_PYTHON}"
  record "# cuda arch list: ${TORCH_CUDA_ARCH_LIST:-n/a}  rocm arch: ${PYTORCH_ROCM_ARCH:-n/a}"
  record "# packages: ${PKG_LIST}"

  echo "${C_BOLD}target${C_RESET} linux/${PC_ARCH} ${PC_ACCEL} torch${PC_TORCH} cp${PC_PYTHON} jobs=${PC_JOBS}" >&2
  [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]] && echo "${C_BOLD}archs ${C_RESET} ${TORCH_CUDA_ARCH_LIST}" >&2

  container_init_cache
  container_prepare_system
  container_prepare_python
  container_install_torch

  local failed=0 pkg
  for pkg in $(split_list "${PKG_LIST}"); do
    echo >&2
    echo "${C_BOLD}[${pkg}]${C_RESET}" >&2
    local rc=0
    case "${pkg}" in
      pccm)            pkg_pccm            || rc=$? ;;
      cumm)            pkg_cumm            || rc=$? ;;
      spconv)          pkg_spconv          || rc=$? ;;
      torch-scatter)   pkg_torch_scatter   || rc=$? ;;
      torch-sparse)    pkg_torch_sparse    || rc=$? ;;
      torch-cluster)   pkg_torch_cluster   || rc=$? ;;
      torch-geometric) pkg_torch_geometric || rc=$? ;;
      flash-attn)      pkg_flash_attn      || rc=$? ;;
      ocnn)            pkg_ocnn            || rc=$? ;;
      swin3d)          pkg_swin3d          || rc=$? ;;
      pointops|pointops2|pointgroup_ops|pointseg|pointrope)
                       pkg_local "${pkg}"  || rc=$? ;;
      *) c_warn "unknown package '${pkg}'"; rc=1 ;;
    esac
    if [[ ${rc} -ne 0 ]]; then
      record "FAILED    ${pkg}"
      failed=$((failed + 1))
    fi
  done

  echo >&2
  if [[ -n "${PC_HOST_UID:-}" ]]; then
    chown -R "${PC_HOST_UID}:${PC_HOST_GID:-${PC_HOST_UID}}" /out 2>/dev/null || true
  fi
  if [[ ${failed} -gt 0 ]]; then
    error "${failed} package(s) failed; see ${MANIFEST}"
    return 1
  fi
  info "all packages built into /out"
}

# ==============================================================================
# Entry point
# ==============================================================================
main() {
  parse_args "$@"

  if [[ "${IN_CONTAINER}" == "1" ]]; then
    run_in_container
    return $?
  fi

  apply_preset

  case "${COMMAND}" in
    build)      cmd_build      ;;
    matrix)     cmd_matrix     ;;
    image)      cmd_image      ;;
    shell)      cmd_shell      ;;
    setup-qemu) cmd_setup_qemu ;;
    clean)      cmd_clean      ;;
    help)       usage          ;;
    *)          die "no command given" ;;
  esac
}

main "$@"
