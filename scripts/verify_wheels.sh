#!/usr/bin/env bash
# ==============================================================================
# Pointcept wheelhouse verifier
#
# Checks that a directory of wheels built by build_wheels.sh actually runs on
# the machine it is meant for. Building a wheel proves that nvcc accepted the
# source; it proves nothing about whether the kernels load and execute on this
# GPU, and that is exactly where an arm64 / CUDA 13 build goes wrong:
#
#   * a wheel compiled against a different torch imports with an undefined
#     symbol, not with a helpful message;
#   * cumm and spconv reject arch lists newer than their own table, so
#     build_wheels.sh compiles them as the newest arch they know plus PTX. On
#     sm_121 (DGX Spark) or sm_110 (Thor) the driver JIT-compiles that PTX on
#     the first kernel launch -- and only then can it fail;
#   * flash-attn has no kernels for those parts at all and is skipped, so the
#     models that want it must still run through their fallback path.
#
# So the stages below go all the way to a real forward pass rather than
# stopping at `import`. Every stage keeps going after a failure and the summary
# at the end lists what broke; the exit status is non-zero if anything did.
#
# Quick start
#   ./scripts/verify_wheels.sh                       # newest wheelhouse for this host
#   ./scripts/verify_wheels.sh wheelhouse/linux-arm64/cu130/torch2.9.1-cp312
#   ./scripts/verify_wheels.sh --in-place            # check the active env instead
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"

# ------------------------------------------------------------------------------
# The wheel-name -> import-name table.
#
# Almost none of these match: the distribution carries the CUDA suffix
# (spconv_cu130), the import does not (spconv); Swin3D is capitalised only on
# import; pointops2's extension module is a top-level `pointops2_cuda` reached
# through `pointops2.pointops`. Getting this wrong turns a broken wheel into a
# passing test, so the table is explicit rather than derived.
#
# Format: <wheel dist prefix>|<import name>|<required?>
# ------------------------------------------------------------------------------
MODULE_TABLE=(
  "pccm|pccm|yes"
  "cumm|cumm|yes"
  "spconv|spconv.pytorch|yes"
  "torch_scatter|torch_scatter|yes"
  "torch_sparse|torch_sparse|yes"
  "torch_cluster|torch_cluster|yes"
  "torch_geometric|torch_geometric|yes"
  "ocnn|ocnn|yes"
  "swin3d|Swin3D.sparse_dl.knn|yes"
  "pointops|pointops|yes"
  "pointops2|pointops2.pointops|yes"
  "pointgroup_ops|pointgroup_ops|yes"
  "pointseg|pointseg|yes"
  "pointrope|pointrope|yes"
  "flash_attn|flash_attn|no"
)

# Python packages the model stage needs that are not wheels under test.
#
# `import pointcept.models` runs every model family, not just the one the stage
# builds: models/__init__.py imports them all, and models/modules.py reaches
# into pointcept.engines.hooks on the way. The list below is that whole
# module-level closure -- peft (default.py), transformers and timm (concerto,
# utonia), scipy (sgiformer), wandb (engines/hooks) -- and all of it is pure
# python. A miss here surfaces as a ModuleNotFoundError in the model stage,
# which says nothing at all about the wheels under test.
#
# torchvision belongs to the same closure (utonia) and is deliberately absent:
# it is compiled against one specific libtorch, so it is installed next to torch
# from the same index in stage_install rather than resolved from PyPI here.
RUNTIME_DEPS=(addict einops timm numpy packaging peft scipy transformers wandb)

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_DIM=""; C_BOLD=""
fi

c_log()   { echo "${C_BLUE}==>${C_RESET} $*"; }
c_ok()    { echo "${C_GREEN}  ok${C_RESET}   $*"; }
c_fail()  { echo "${C_RED}  FAIL${C_RESET} $*"; }
c_skip()  { echo "${C_DIM}  skip${C_RESET} $*"; }
c_warn()  { echo "${C_YELLOW}warn:${C_RESET} $*" >&2; }
c_die()   { echo "${C_RED}error:${C_RESET} $*" >&2; exit 1; }

FAILED_STAGES=()
stage_result() {
  local name="$1" rc="$2"
  if [[ "${rc}" -eq 0 ]]; then
    c_ok "stage '${name}' passed"
  else
    c_fail "stage '${name}' failed"
    FAILED_STAGES+=("${name}")
  fi
}

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
WHEELHOUSE=""
VENV_DIR=""
KEEP_VENV="0"
IN_PLACE="0"
INSTALL_TORCH="1"
TORCH_INDEX=""
QUICK="0"
PYTHON_BIN="python3"

usage() {
  cat <<EOF
${C_BOLD}USAGE${C_RESET}
  ${SCRIPT_NAME} [OPTIONS] [WHEELHOUSE_DIR]

${C_BOLD}ARGUMENTS${C_RESET}
  WHEELHOUSE_DIR     Directory holding the wheels and manifest.txt. Defaults to
                     the newest wheelhouse/linux-<hostarch>/*/* in this repo.

${C_BOLD}OPTIONS${C_RESET}
  --in-place         Verify the currently active environment; install nothing.
                     Use this after installing the wheels yourself.
  --venv DIR         Create the throwaway venv here (default: a temp dir).
  --keep             Keep the venv after the run (it prints the path).
  --no-torch         Do not install torch; the venv is expected to have it.
                     torchvision is installed alongside torch, so the model
                     stage expects that one to be present already too.
  --torch-index URL  pip index for torch (default: derived from the accel in
                     the wheelhouse path, e.g. cu130 -> .../whl/cu130).
  --quick            Stop after the import stage; skip kernels and the model.
  --python BIN       Interpreter used to create the venv (default: python3).
  -h, --help         Show this message.

${C_BOLD}STAGES${C_RESET}
  env       interpreter, torch, CUDA runtime, driver and GPU capability
  tags      wheel filename tags against this interpreter and platform
  install   pip install torch + every wheel into a fresh venv
  import    import each module by the name Pointcept actually uses
  kernel    run one real op per native package (forces cubin/PTX to load)
  model     build PTv3 and run a forward pass on synthetic points

${C_BOLD}NOTES${C_RESET}
  * flash-attn is optional: on sm_121 / sm_110 it has no kernels and
    build_wheels.sh skips it, which the model stage compensates for by
    disabling flash attention. Its absence is not a failure.
  * The kernel stage is where a PTX-only spconv build shows itself. The first
    launch can take tens of seconds while the driver JITs; that is normal, a
    'no kernel image is available for execution' is not.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-place)    IN_PLACE="1"; shift ;;
    --venv)        VENV_DIR="${2:?--venv needs a path}"; shift 2 ;;
    --keep)        KEEP_VENV="1"; shift ;;
    --no-torch)    INSTALL_TORCH="0"; shift ;;
    --torch-index) TORCH_INDEX="${2:?--torch-index needs a URL}"; shift 2 ;;
    --quick)       QUICK="1"; shift ;;
    --python)      PYTHON_BIN="${2:?--python needs a binary}"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    -*)            c_die "unknown option: $1 (see --help)" ;;
    *)             WHEELHOUSE="$1"; shift ;;
  esac
done

# ------------------------------------------------------------------------------
# Locate the wheelhouse
#
# build_wheels.sh writes to wheelhouse/linux-<arch>/<accel>/torch<ver>-cp<py>/,
# so with no argument the newest directory for this host's architecture is the
# one that was just built.
# ------------------------------------------------------------------------------
host_arch() {
  case "$(uname -m)" in
    x86_64)          echo "amd64" ;;
    aarch64|arm64)   echo "arm64" ;;
    *)               echo "$(uname -m)" ;;
  esac
}

if [[ "${IN_PLACE}" == "0" || -n "${WHEELHOUSE}" ]]; then
  if [[ -z "${WHEELHOUSE}" ]]; then
    root="${REPO_ROOT}/wheelhouse/linux-$(host_arch)"
    [[ -d "${root}" ]] || c_die "no wheelhouse for this host: ${root}"
    WHEELHOUSE="$(find "${root}" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' \
      | sort -rn | head -1 | cut -d' ' -f2-)"
    [[ -n "${WHEELHOUSE}" ]] || c_die "no wheel directory under ${root}"
    c_log "auto-selected wheelhouse: ${WHEELHOUSE}"
  fi
  WHEELHOUSE="$(readlink -f "${WHEELHOUSE}")"
  [[ -d "${WHEELHOUSE}" ]] || c_die "not a directory: ${WHEELHOUSE}"
  shopt -s nullglob
  WHEELS=("${WHEELHOUSE}"/*.whl)
  shopt -u nullglob
  [[ ${#WHEELS[@]} -gt 0 ]] || c_die "no wheels in ${WHEELHOUSE}"
fi

# The accel token is the second-to-last path component (…/cu130/torch2.9.1-cp312).
ACCEL=""
if [[ -n "${WHEELHOUSE}" ]]; then
  ACCEL="$(basename "$(dirname "${WHEELHOUSE}")")"
fi

# ------------------------------------------------------------------------------
# Stage: env
#
# Everything downstream is meaningless if the interpreter or torch does not
# match what the wheels were built against, so this runs first and prints the
# numbers rather than asserting them -- a mismatch is usually the answer.
# ------------------------------------------------------------------------------
stage_env() {
  c_log "stage env: interpreter, torch and device"
  "${PY}" - <<'PYEOF'
import platform
import sys

print(f"  python       {sys.version.split()[0]}  ({platform.machine()})")
try:
    import torch
except ImportError:
    print("  torch        NOT INSTALLED")
    raise SystemExit(1)

print(f"  torch        {torch.__version__}")
print(f"  torch.cuda   {torch.version.cuda or torch.version.hip or 'cpu-only'}")
print(f"  available    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"  device       {torch.cuda.get_device_name(0)}  sm_{major}{minor}")
    # The arch list baked into torch itself is a good proxy for whether this
    # build even intends to serve the device in front of it.
    print(f"  torch archs  {' '.join(torch.cuda.get_arch_list())}")
else:
    print("  device       none visible -- kernel and model stages will be skipped")
PYEOF
}

# ------------------------------------------------------------------------------
# Stage: tags
#
# A wheel filename encodes the interpreter ABI and platform it was built for.
# pip refuses a mismatched one with "not a supported wheel on this platform",
# which is easy to misread as a broken wheel rather than a wrong target.
# ------------------------------------------------------------------------------
stage_tags() {
  c_log "stage tags: wheel filenames against this interpreter"
  "${PY}" - "${WHEELS[@]}" <<'PYEOF'
import sys
from pathlib import Path

try:
    from packaging.tags import sys_tags
    supported = {str(t) for t in sys_tags()}
except ImportError:  # packaging ships with pip, but not always exposed
    supported = None

bad = 0
for raw in sys.argv[1:]:
    name = Path(raw).name
    parts = name[: -len(".whl")].split("-")
    # {dist}-{version}(-{build})?-{python}-{abi}-{platform}.whl
    tags = "-".join(parts[-3:])
    if supported is None:
        print(f"  ?    {name}")
        continue
    combos = {
        f"{py}-{abi}-{plat}"
        for py in parts[-3].split(".")
        for abi in parts[-2].split(".")
        for plat in parts[-1].split(".")
    }
    if combos & supported:
        print(f"  ok   {name}")
    else:
        bad += 1
        print(f"  FAIL {name}  (tags {tags} unsupported here)")

if supported is None:
    print("  note: packaging.tags unavailable, tags not checked")
raise SystemExit(1 if bad else 0)
PYEOF
}

# ------------------------------------------------------------------------------
# Stage: install
#
# torch goes in first and on its own: every native wheel here was linked
# against a specific libtorch, and pip resolving a different one after the fact
# is what produces undefined-symbol imports later.
# ------------------------------------------------------------------------------
stage_install() {
  c_log "stage install: torch + ${#WHEELS[@]} wheels into ${VENV_DIR}"

  if [[ "${INSTALL_TORCH}" == "1" ]]; then
    local index="${TORCH_INDEX}"
    if [[ -z "${index}" && "${ACCEL}" == cu* ]]; then
      index="https://download.pytorch.org/whl/${ACCEL}"
    fi
    # The torch version is in the wheelhouse path (torch2.9.1-cp312).
    local want; want="$(basename "${WHEELHOUSE}")"
    want="${want#torch}"; want="${want%%-*}"
    # torchvision rides along in this call rather than sitting in RUNTIME_DEPS:
    # it pins the torch it was compiled against, so resolving it afterwards
    # either fails or quietly installs a second libtorch over the one every
    # wheel here was linked to. Asked for beside a pinned torch, pip picks the
    # build that matches.
    c_log "  torch==${want} + torchvision from ${index:-PyPI}"
    if [[ -n "${index}" ]]; then
      "${PY}" -m pip install --quiet "torch==${want}" torchvision --index-url "${index}" || return 1
    else
      "${PY}" -m pip install --quiet "torch==${want}" torchvision || return 1
    fi
  fi

  # Every native wheel below was linked against the torch that is installed now,
  # so anything that swaps it out invalidates the rest of the run.
  local before after
  before="$("${PY}" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"

  # One pip call for every wheel: they depend on each other (spconv needs cumm
  # needs pccm) and resolving them together keeps pip from reaching for PyPI.
  "${PY}" -m pip install --quiet "${WHEELS[@]}" || return 1
  "${PY}" -m pip install --quiet "${RUNTIME_DEPS[@]}" || return 1

  # peft and transformers carry their own torch specifiers, and pip may satisfy
  # one by replacing torch with a PyPI build that has no CUDA at all -- after
  # which every later stage reports "no CUDA device visible" and none of it is
  # about the wheels. Name it here instead.
  after="$("${PY}" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  if [[ -n "${before}" && "${before}" != "${after}" ]]; then
    c_fail "  torch was replaced during install: ${before} -> ${after:-none}"
    c_fail "  a runtime dep pulled its own torch; these wheels are linked to ${before}"
    return 1
  fi
}

# ------------------------------------------------------------------------------
# Stage: import
#
# The names come from MODULE_TABLE, which is what Pointcept imports -- not what
# the wheel is called. An undefined symbol surfaces here.
# ------------------------------------------------------------------------------
stage_import() {
  c_log "stage import: module by module"
  local rc=0 entry dist mod required out
  local -a absent=()
  for entry in "${MODULE_TABLE[@]}"; do
    IFS='|' read -r dist mod required <<<"${entry}"
    # A wheel that is not in this directory was never built for this target --
    # flash-attn on sm_121, or a partial `build --only` run. Importing it would
    # test whatever else is installed on the machine, so skip and say so.
    if [[ -n "${WHEELHOUSE:-}" ]]; then
      shopt -s nullglob
      local present=("${WHEELHOUSE}/${dist}"-*.whl "${WHEELHOUSE}/${dist}"_*.whl)
      shopt -u nullglob
      if [[ ${#present[@]} -eq 0 ]]; then
        c_skip "${mod} (no ${dist} wheel in this wheelhouse)"
        if [[ "${required}" == "yes" ]]; then
          absent+=("${dist}")
        fi
        continue
      fi
    fi
    # torch first, always. Most of these are packages whose __init__ imports
    # torch itself, but pointrope is a bare CUDAExtension -- the .so *is* the
    # top-level module -- and it resolves libc10 through the RTLD_GLOBAL handles
    # torch opens on import. Without that line it fails with a missing
    # libc10.so, which reads like a broken wheel and is not one.
    if out="$("${PY}" -c "import torch; import ${mod}; print(getattr(${mod}, '__version__', ''))" 2>&1)"; then
      c_ok "${mod} ${out}"
    elif [[ "${required}" == "no" ]]; then
      c_skip "${mod} (optional, not importable)"
    else
      c_fail "${mod}"
      echo "${out}" | sed 's/^/       /' | tail -5
      rc=1
    fi
  done
  if [[ ${#absent[@]} -gt 0 ]]; then
    c_warn "not built for this target: ${absent[*]}"
    c_warn "  intentional after 'build --only', otherwise check manifest.txt for a skip"
  fi
  return "${rc}"
}

# ------------------------------------------------------------------------------
# Stage: kernel
#
# One real op per native package, on the GPU. This is the stage that catches a
# PTX-only build failing to JIT for the device, which no import can tell you.
# ------------------------------------------------------------------------------
stage_kernel() {
  c_log "stage kernel: one op per native package"
  "${PY}" - <<'PYEOF'
import traceback

import torch

if not torch.cuda.is_available():
    print("  no CUDA device visible; nothing to launch")
    raise SystemExit(0)

dev = torch.device("cuda")
failures = []


def check(name, fn):
    try:
        fn()
        torch.cuda.synchronize()
    except Exception:
        failures.append(name)
        print(f"  FAIL {name}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"       {line}")
    else:
        print(f"  ok   {name}")


N = 4096
xyz = torch.rand(N, 3, device=dev).contiguous()
offset = torch.tensor([N], dtype=torch.int32, device=dev)
feat = torch.rand(N, 16, device=dev)


def t_pointops():
    import pointops

    idx, dist = pointops.knn_query(8, xyz, offset)
    assert idx.shape == (N, 8), idx.shape
    fps = pointops.farthest_point_sampling(xyz, offset, torch.tensor([256], dtype=torch.int32, device=dev))
    assert fps.numel() == 256, fps.shape


def t_pointops2():
    import pointops2.pointops as p2

    idx, _ = p2.knnquery(8, xyz, xyz, offset, offset)
    assert idx.shape == (N, 8), idx.shape


def t_pointgroup_ops():
    import pointgroup_ops

    # The installed package is the autograd wrapper, not the raw extension:
    # (coords, batch_idxs, batch_offsets, radius, meanActive).
    batch_idxs = torch.zeros(N, dtype=torch.int32, device=dev)
    batch_offsets = torch.tensor([0, N], dtype=torch.int32, device=dev)
    idx, start_len = pointgroup_ops.ballquery_batch_p(
        xyz, batch_idxs, batch_offsets, 0.1, 32
    )
    assert start_len.shape == (N, 2), start_len.shape


def t_pointrope():
    import pointrope

    # tokens are (B, N, H, C) and the kernel splits C into 3 axes x 2 halves,
    # so C has to be a multiple of 6 (see Q = D / 6 in kernels.cu).
    tokens = torch.rand(1, 64, 4, 24, device=dev).contiguous()
    # positions are read as `pos.data_ptr<int64_t>()`, so a float tensor is not
    # converted, it is refused. They are voxel indices: Pointcept passes
    # point.grid_coord straight through (litept_v1.py). Ramped rather than
    # random so the "did anything rotate?" check below cannot draw all zeros.
    positions = (
        torch.arange(64, dtype=torch.int64, device=dev).view(1, 64, 1).repeat(1, 1, 3).contiguous()
    )
    before = tokens.clone()
    pointrope.pointrope(tokens, positions, 100.0, 1.0)
    # The op rotates in place and returns nothing, so shape proves nothing here.
    assert torch.isfinite(tokens).all(), "pointrope produced non-finite tokens"
    assert not torch.equal(tokens, before), "pointrope left the tokens untouched"


def t_pointseg():
    import pointseg

    # A CppExtension, so this one runs on the CPU. Big enough that the
    # segMinVerts=20 merge pass has something to do.
    g = torch.arange(8, dtype=torch.float32)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    verts = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.zeros(64)], dim=1)
    quads = [
        [r * 8 + c, r * 8 + c + 1, (r + 1) * 8 + c, (r + 1) * 8 + c + 1]
        for r in range(7)
        for c in range(7)
    ]
    faces = torch.tensor(
        [[q[0], q[1], q[2]] for q in quads] + [[q[1], q[3], q[2]] for q in quads],
        dtype=torch.int64,
    )
    seg = pointseg.segment_mesh(verts, faces)
    assert seg.numel() == verts.shape[0], seg.shape


def t_spconv():
    import spconv.pytorch as spconv

    # The PTX JIT happens here on the first launch, not on import.
    coords = torch.randint(0, 32, (1024, 3), dtype=torch.int32, device=dev)
    batch = torch.zeros(1024, 1, dtype=torch.int32, device=dev)
    indices = torch.cat([batch, coords], dim=1).contiguous()
    x = spconv.SparseConvTensor(
        features=torch.rand(1024, 16, device=dev),
        indices=indices,
        spatial_shape=[32, 32, 32],
        batch_size=1,
    )
    conv = spconv.SubMConv3d(16, 32, kernel_size=3, padding=1, bias=False, indice_key="s0").to(dev)
    out = conv(x)
    assert out.features.shape[1] == 32, out.features.shape


def t_torch_scatter():
    import torch_scatter

    index = torch.randint(0, 128, (N,), device=dev)
    out, _ = torch_scatter.scatter_max(feat, index, dim=0, dim_size=128)
    assert out.shape == (128, 16), out.shape


def t_torch_cluster():
    import torch_cluster

    idx = torch_cluster.fps(xyz, ratio=0.25, random_start=False)
    assert idx.numel() > 0


def t_torch_sparse():
    import torch_sparse

    row = torch.randint(0, 64, (256,), device=dev)
    col = torch.randint(0, 64, (256,), device=dev)
    val = torch.rand(256, device=dev)
    idx, val = torch_sparse.coalesce(torch.stack([row, col]), val, 64, 64)
    assert idx.shape[0] == 2, idx.shape


def t_torch_geometric():
    from torch_geometric.nn.pool import voxel_grid

    # torch-geometric 2.8 moved grid_cluster from torch-cluster to pyg-lib, and
    # pyg-lib publishes no aarch64 wheel; build_wheels.sh holds torch-geometric
    # below 2.8 for that reason. An ImportError naming pyg-lib here means the
    # wheelhouse predates that pin.
    cluster = voxel_grid(xyz, size=0.1, batch=torch.zeros(N, dtype=torch.long, device=dev))
    assert cluster.numel() == N


def t_swin3d():
    from Swin3D.sparse_dl.knn import KNN

    idx, _ = KNN.apply(8, xyz, xyz, offset, offset)
    assert idx.shape == (N, 8), idx.shape


for name, fn in [
    ("pointops", t_pointops),
    ("pointops2", t_pointops2),
    ("pointgroup_ops", t_pointgroup_ops),
    ("pointrope", t_pointrope),
    ("pointseg", t_pointseg),
    ("spconv", t_spconv),
    ("torch_scatter", t_torch_scatter),
    ("torch_cluster", t_torch_cluster),
    ("torch_sparse", t_torch_sparse),
    ("torch_geometric", t_torch_geometric),
    ("Swin3D", t_swin3d),
]:
    check(name, fn)

# ocnn and torch_geometric ship as py3-none-any: there is no compiled artifact
# of their own to launch, so importing them (the stage above) is the whole
# check. torch_geometric still appears here because voxel_grid calls into
# torch_cluster, which is compiled.

raise SystemExit(1 if failures else 0)
PYEOF
}

# ------------------------------------------------------------------------------
# Stage: model
#
# The packages above are exercised one at a time; a real model exercises them
# together, and that is where a serialization/spconv/torch_scatter mismatch
# shows up. PTv3 is the default backbone and touches all three.
# ------------------------------------------------------------------------------
stage_model() {
  c_log "stage model: PTv3 forward on synthetic points"
  ( cd "${REPO_ROOT}" && "${PY}" - <<'PYEOF'
import torch

if not torch.cuda.is_available():
    print("  no CUDA device visible; skipping")
    raise SystemExit(0)

try:
    import flash_attn  # noqa: F401
    has_flash = True
except ImportError:
    has_flash = False
print(f"  flash_attn   {'present' if has_flash else 'absent (fallback path)'}")

# The m1 module by name, not the package. All three PTv3 variants define a class
# called PointTransformerV3 and __init__.py star-imports them m1 -> m2 -> m3, so
# the bare name is whichever came last -- today m3 (utonia), whose Point3DRoPE
# asserts head_dim % 3 == 0 while PTv3's default head_dim is 16 at every stage.
# The registry keeps the three apart (PT-v3m1/m2/m3); a direct import does not.
# m1 is the backbone this stage means: the one the wheels below it have to carry.
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)

dev = torch.device("cuda")
N = 20000

model = PointTransformerV3(
    in_channels=6,
    enable_flash=has_flash,
    # Without flash-attn the fallback needs the padded path, which upcasts.
    upcast_attention=not has_flash,
    upcast_softmax=not has_flash,
).to(dev).eval()

data = dict(
    coord=torch.rand(N, 3, device=dev) * 5.0,
    feat=torch.rand(N, 6, device=dev),
    grid_size=0.02,
    offset=torch.tensor([N], device=dev),
)

with torch.no_grad():
    point = model(data)

feat = point.feat if hasattr(point, "feat") else point
print(f"  output       {tuple(feat.shape)}  dtype={feat.dtype}")
assert torch.isfinite(feat).all(), "forward produced non-finite values"
print("  forward pass produced finite features")
PYEOF
  )
}

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
CLEANUP_VENV=""
cleanup() {
  if [[ -n "${CLEANUP_VENV}" && "${KEEP_VENV}" == "0" ]]; then
    rm -rf "${CLEANUP_VENV}"
  fi
}
trap cleanup EXIT

if [[ "${IN_PLACE}" == "1" ]]; then
  PY="$(command -v python3 || true)"
  [[ -n "${PY}" ]] || c_die "no python3 on PATH"
  c_log "verifying the active environment: ${PY}"
else
  command -v "${PYTHON_BIN}" >/dev/null || c_die "interpreter not found: ${PYTHON_BIN}"
  if [[ -z "${VENV_DIR}" ]]; then
    VENV_DIR="$(mktemp -d -t pointcept-verify-XXXXXX)"
    CLEANUP_VENV="${VENV_DIR}"
  fi
  c_log "creating venv: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  PY="${VENV_DIR}/bin/python"
  # Bootstrapped here rather than in stage_install so the tag check -- which
  # needs packaging.tags -- can run first and explain an install failure
  # instead of following it.
  "${PY}" -m pip install --upgrade --quiet pip setuptools wheel packaging
fi

echo
echo "${C_BOLD}Pointcept wheelhouse verification${C_RESET}"
if [[ -n "${WHEELHOUSE}" ]]; then
  echo "  wheelhouse: ${WHEELHOUSE}"
  # The manifest records how each wheel was obtained and, just as usefully,
  # what was skipped for this target -- read it alongside the stage results.
  if [[ -f "${WHEELHOUSE}/manifest.txt" ]]; then
    echo "  manifest:"
    sed 's/^/    /' "${WHEELHOUSE}/manifest.txt"
  else
    c_warn "no manifest.txt in ${WHEELHOUSE}"
  fi
fi
echo

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
# Tags first: an unsupported tag is why pip would refuse the wheel a moment
# later, and saying so up front beats a bare "no matching distribution".
if [[ -n "${WHEELHOUSE}" ]]; then
  stage_tags && rc=0 || rc=1; stage_result tags "${rc}"
fi

if [[ "${IN_PLACE}" == "0" ]]; then
  stage_install && rc=0 || rc=1
  stage_result install "${rc}"
  # Nothing downstream can pass if the install did not, so stop here.
  if [[ "${rc}" -ne 0 ]]; then
    echo
    c_fail "install failed; later stages skipped"
    exit 1
  fi
fi

stage_env && rc=0 || rc=1; stage_result env "${rc}"
stage_import && rc=0 || rc=1; stage_result import "${rc}"

if [[ "${QUICK}" == "0" ]]; then
  stage_kernel && rc=0 || rc=1; stage_result kernel "${rc}"
  stage_model  && rc=0 || rc=1; stage_result model  "${rc}"
else
  c_skip "kernel and model stages (--quick)"
fi

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
report_venv() {
  if [[ "${KEEP_VENV}" == "1" && -n "${VENV_DIR}" ]]; then
    echo "venv kept at ${VENV_DIR}"
  fi
}

echo
if [[ ${#FAILED_STAGES[@]} -eq 0 ]]; then
  echo "${C_GREEN}${C_BOLD}All stages passed.${C_RESET}"
  report_venv
  exit 0
fi
echo "${C_RED}${C_BOLD}Failed stages: ${FAILED_STAGES[*]}${C_RESET}"
report_venv
exit 1
