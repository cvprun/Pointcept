#!/usr/bin/env bash
# ==============================================================================
# Pointcept wheelhouse -> mlops pipeline verifier
#
# verify_wheels.sh proves that the wheels import and that a model runs a forward
# pass. That is necessary and not sufficient: the mlops platform does not call
# Pointcept the way the verifier does. It generates a config, registers its own
# hooks and trainer through the HOOKS/TRAINERS registries, converts .ply files
# into a DefaultDataset layout, trains, evaluates, writes a checkpoint, logs an
# MLflow pyfunc model, and later loads that model back to predict. Every one of
# those steps has broken on a wheelhouse that passed verify_wheels.sh -- an
# evaluation path that runs kernels training never reaches, a pyfunc that
# rebuilds the network from a saved config, a serving import that pulls a
# different torch.
#
# So this script runs the real entrypoints, unmodified, out of the mlops
# checkout:
#
#   pointcept_semseg.py   the training job the platform's Kubernetes Job runs
#   pointcept_pyfunc.py   the model the serving container loads
#
# It reuses verify_wheels.sh's venv rather than building a second one: those
# wheels are linked against one torch, and a separate environment would either
# duplicate a multi-gigabyte install or quietly resolve a different one.
#
# Sample data is a real public scan -- Open3D's `fragment.ply`, an indoor RGB
# point cloud -- sliced into scenes. Its per-point labels are derived from the
# surface normals it ships with (horizontal / vertical / other), because no
# labelled point cloud can be fetched unattended: SemanticKITTI, S3DIS,
# Toronto3D and Semantic3D all sit behind a registration form, a licence click
# or tens of gigabytes. Derived labels are honest for this purpose -- the point
# is whether the pipeline runs end to end and the loss moves, not what the mIoU
# is -- and they are a real geometric signal rather than noise, so a run that
# learns nothing is a finding rather than the expected outcome.
#
# Quick start
#   ./scripts/verify_mlops.sh                        # newest wheelhouse, ../mlops
#   ./scripts/verify_mlops.sh --mlops ~/src/mlops
#   ./scripts/verify_mlops.sh --spconv-native        # exercise the Native path
#   ./scripts/verify_mlops.sh --skip-wheels --keep   # iterate on an existing venv
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_DIM=$'\033[2m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_DIM=""
fi

c_log()   { echo "${C_BLUE}==>${C_RESET} $*"; }
c_ok()    { echo "${C_GREEN}  ok${C_RESET}   $*"; }
c_fail()  { echo "${C_RED}  FAIL${C_RESET} $*"; }
c_skip()  { echo "${C_DIM}  skip${C_RESET} $*"; }
c_warn()  { echo "${C_YELLOW}warn:${C_RESET} $*" >&2; }
c_die()   { echo "${C_RED}error:${C_RESET} $*" >&2; exit 1; }

STAGE_NAMES=(); STAGE_CODES=()
stage_result() {
  STAGE_NAMES+=("$1"); STAGE_CODES+=("$2")
  if [[ "$2" == "0" ]]; then c_ok "stage $1"; else c_fail "stage $1"; fi
}

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
MLOPS_DIR=""
WHEELHOUSE=""
VENV_DIR=""
KEEP_VENV="0"
SKIP_WHEELS="0"
EPOCHS="2"
GRID_SIZE="0.02"
BATCH_SIZE="2"
SPCONV_NATIVE="0"
SCENES="8"
CACHE_DIR="${POINTCEPT_VERIFY_CACHE:-${REPO_ROOT}/.verify-cache}"

usage() {
  cat <<EOF
${SCRIPT_NAME} -- run the mlops training and inference entrypoints on a wheelhouse

Usage: ${SCRIPT_NAME} [options] [wheelhouse]

  --mlops PATH       mlops checkout (default: ${REPO_ROOT}/../mlops)
  --venv PATH        virtualenv to build in (default: a temporary one)
  --keep             keep the venv and the work directory
  --skip-wheels      do not run verify_wheels.sh first
  --epochs N         training epochs (default: ${EPOCHS})
  --grid-size F      voxel size, in the units of the sample data (default: ${GRID_SIZE})
  --batch-size N     scenes per step (default: ${BATCH_SIZE})
  --scenes N         scenes to slice the sample cloud into (default: ${SCENES})
  --spconv-native    submit with spconv_native=true (pins ConvAlgo.Native)
  -h, --help         this text

The wheelhouse argument is passed straight to verify_wheels.sh; with none, both
scripts pick the newest build for this host.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mlops)          MLOPS_DIR="${2:?--mlops needs a path}"; shift 2 ;;
    --venv)           VENV_DIR="${2:?--venv needs a path}"; shift 2 ;;
    --keep)           KEEP_VENV="1"; shift ;;
    --skip-wheels)    SKIP_WHEELS="1"; shift ;;
    --epochs)         EPOCHS="${2:?}"; shift 2 ;;
    --grid-size)      GRID_SIZE="${2:?}"; shift 2 ;;
    --batch-size)     BATCH_SIZE="${2:?}"; shift 2 ;;
    --scenes)         SCENES="${2:?}"; shift 2 ;;
    --spconv-native)  SPCONV_NATIVE="1"; shift ;;
    -h|--help)        usage; exit 0 ;;
    -*)               c_die "unknown option: $1" ;;
    *)                WHEELHOUSE="$1"; shift ;;
  esac
done

[[ -n "${MLOPS_DIR}" ]] || MLOPS_DIR="${REPO_ROOT}/../mlops"
MLOPS_DIR="$(readlink -f "${MLOPS_DIR}" 2>/dev/null || echo "${MLOPS_DIR}")"

#: The public sample. A single file, no registration, stable release URL.
SAMPLE_URL="https://github.com/isl-org/open3d_downloads/releases/download/20220201-data/fragment.ply"
SAMPLE_NAME="fragment.ply"

#: What the trainer needs beyond the wheels and torch. These are the same pure
#: Python dependencies the platform's runtime image installs; several are
#: unguarded imports inside `import pointcept`, so a missing one fails at import
#: rather than at use.
TRAINER_DEPS=(
  addict einops h5py plyfile scipy tensorboardX termcolor timm yapf
  wandb mlflow pyyaml pillow pandas
)

#: open3d is handled apart from the list above because it is the one dependency
#: whose availability depends on the architecture, and `pointcept.datasets`
#: imports it unguarded through loaders this platform never uses
#: (semantic_kitti, nuscenes). That import runs from the package __init__, so
#: without it `import pointcept` fails outright. Upstream publishes no aarch64
#: wheel for CPython 3.12 and no sdist, which is why the runtime image drops a
#: shim in its place on arm64 -- imports succeed, touching an attribute fails
#: with the reason. Same trick here, and also as the fallback when the install
#: simply does not work: a verifier that stops over a loader the platform does
#: not use would be reporting on the wrong thing.
SHIM_DIR=""

UV_BIN="$(command -v uv || true)"
PY=""

# ------------------------------------------------------------------------------
# Stage: env
# ------------------------------------------------------------------------------
stage_env() {
  c_log "stage env"
  local rc=0

  [[ -d "${MLOPS_DIR}" ]] || { c_fail "  no mlops checkout at ${MLOPS_DIR}"; return 1; }
  RUNTIME_SRC="${MLOPS_DIR}/src/geo_mlops/resources/training_runtime"
  for f in pointcept_semseg.py pointcept_pyfunc.py annotations.py; do
    if [[ ! -f "${RUNTIME_SRC}/${f}" ]]; then
      c_fail "  ${RUNTIME_SRC}/${f} is missing"; rc=1
    fi
  done
  [[ "${rc}" == "0" ]] && c_ok "  mlops entrypoints at ${RUNTIME_SRC}"

  if [[ -z "${UV_BIN}" ]]; then
    c_warn "  uv not found; falling back to python -m venv / pip"
  else
    c_ok "  uv $(${UV_BIN} --version 2>/dev/null | awk '{print $2}')"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    c_ok "  gpu $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
  else
    c_warn "  no nvidia-smi; spconv and pointops are CUDA-only and will fail"
  fi
  return "${rc}"
}

# ------------------------------------------------------------------------------
# Stage: wheels
#
# Delegated rather than reimplemented, and pointed at the venv this script goes
# on to use: whatever verify_wheels.sh installs is exactly what the training run
# must run against.
# ------------------------------------------------------------------------------
stage_wheels() {
  if [[ "${SKIP_WHEELS}" == "1" ]]; then
    c_skip "stage wheels (--skip-wheels)"
    return 0
  fi
  c_log "stage wheels: verify_wheels.sh --venv ${VENV_DIR} --keep"
  local args=(--venv "${VENV_DIR}" --keep)
  [[ -n "${WHEELHOUSE}" ]] && args+=("${WHEELHOUSE}")
  "${REPO_ROOT}/scripts/verify_wheels.sh" "${args[@]}"
}

# ------------------------------------------------------------------------------
# Stage: deps
# ------------------------------------------------------------------------------
py_install() {
  if [[ -n "${UV_BIN}" ]]; then
    "${UV_BIN}" pip install --python "${PY}" "$@"
  else
    "${PY}" -m pip install "$@"
  fi
}

stage_deps() {
  c_log "stage deps: ${#TRAINER_DEPS[@]} pure-Python packages"
  local before after
  before="$("${PY}" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  [[ -n "${before}" ]] || { c_fail "  no torch in ${VENV_DIR}; run without --skip-wheels"; return 1; }

  py_install --quiet "${TRAINER_DEPS[@]}" || return 1

  if py_install --quiet open3d >/dev/null 2>&1; then
    c_ok "  open3d $("${PY}" -c 'import open3d; print(open3d.__version__)' 2>/dev/null || echo installed)"
  else
    SHIM_DIR="${WORK_DIR}/shims"
    mkdir -p "${SHIM_DIR}"
    cat > "${SHIM_DIR}/open3d.py" <<'SHIM'
"""Stand-in for open3d where upstream publishes no wheel.

Mirrors what the platform's runtime image installs on arm64: the import has to
succeed because pointcept.datasets does it unguarded, and anything that
actually reaches into the module deserves to be told why it is not there.
"""


def __getattr__(name):
    raise ImportError(
        "open3d is not installed in this environment: upstream ships no "
        "aarch64 wheel for CPython 3.12. The dataset loader that reached "
        f"open3d.{name} needs it -- this pipeline only trains through "
        "DefaultDataset."
    )
SHIM
    c_warn "  open3d unavailable; using the runtime image's import shim"
  fi

  # Same guard verify_wheels.sh uses at install time, for the same reason: timm
  # and mlflow carry torch specifiers, and a resolver that satisfies one by
  # swapping torch invalidates every native wheel under test.
  after="$("${PY}" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
  if [[ "${before}" != "${after}" ]]; then
    c_fail "  torch was replaced during install: ${before} -> ${after:-none}"
    return 1
  fi
  c_ok "  torch still ${after}"
}

# ------------------------------------------------------------------------------
# Stage: data
# ------------------------------------------------------------------------------
stage_data() {
  c_log "stage data: ${SCENES} scenes from ${SAMPLE_NAME}"
  mkdir -p "${CACHE_DIR}" "${DATA_DIR}"
  local sample="${CACHE_DIR}/${SAMPLE_NAME}"
  if [[ -s "${sample}" ]]; then
    c_ok "  cached $(du -h "${sample}" | cut -f1) ${sample}"
  else
    c_log "  downloading ${SAMPLE_URL}"
    curl -fsSL --max-time 300 -o "${sample}.part" "${SAMPLE_URL}" || {
      c_fail "  download failed"; rm -f "${sample}.part"; return 1; }
    mv "${sample}.part" "${sample}"
    c_ok "  downloaded $(du -h "${sample}" | cut -f1)"
  fi

  "${PY}" - "${sample}" "${DATA_DIR}" "${SCENES}" <<'PYEOF'
import sys
import numpy as np
from plyfile import PlyData, PlyElement

sample, out_dir, scenes = sys.argv[1], sys.argv[2], int(sys.argv[3])
vertex = PlyData.read(sample)["vertex"].data
coord = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
color = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
normal = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)

# Labels from the surface normals the scan ships with. The up axis is the one
# the normals cluster on rather than a guess: an indoor scan's largest surfaces
# are its floor and ceiling, so whichever component is most often near +/-1 is
# vertical. Deriving it keeps the split meaningful if the sample is ever swapped.
up = int(np.argmax([(np.abs(normal[:, i]) > 0.85).sum() for i in range(3)]))
vertical = np.abs(normal[:, up])
label = np.full(len(coord), 2, dtype=np.int32)   # 2: everything else
label[vertical > 0.85] = 0                       # 0: horizontal surfaces
label[vertical < 0.30] = 1                       # 1: vertical surfaces

# Slice along the longest axis so each scene is a different part of the room
# rather than a subsample of the same one -- a train/val split over subsamples
# would score a model that had already seen every surface.
axis = int(np.argmax(coord.max(0) - coord.min(0)))
edges = np.quantile(coord[:, axis], np.linspace(0.0, 1.0, scenes + 1))
edges[0] -= 1.0
edges[-1] += 1.0

counts = []
for i in range(scenes):
    keep = (coord[:, axis] > edges[i]) & (coord[:, axis] <= edges[i + 1])
    n = int(keep.sum())
    counts.append(n)
    if n == 0:
        raise SystemExit(f"scene {i} is empty; use fewer --scenes")
    data = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "i4"),
    ])
    for j, name in enumerate(("x", "y", "z")):
        data[name] = coord[keep, j]
    for j, name in enumerate(("red", "green", "blue")):
        data[name] = color[keep, j]
    data["label"] = label[keep]
    element = PlyElement.describe(data, "vertex")
    PlyData([element], text=False).write(f"{out_dir}/scene_{i:02d}.ply")

hist = np.bincount(label, minlength=3)
print(f"   up axis {'xyz'[up]}, slice axis {'xyz'[axis]}")
print(f"   points/scene min {min(counts)} max {max(counts)}")
print(f"   labels: horizontal {hist[0]} vertical {hist[1]} other {hist[2]}")
if (hist == 0).any():
    raise SystemExit("a class is empty; the label rule does not fit this cloud")
PYEOF
}

# ------------------------------------------------------------------------------
# Stage: train
#
# The platform's entrypoint, run the way its Job runs it: everything it needs
# arrives through GEO_* and MLFLOW_* rather than through arguments.
# ------------------------------------------------------------------------------
stage_train() {
  c_log "stage train: ${EPOCHS} epoch(s), grid_size ${GRID_SIZE}, spconv_native ${SPCONV_NATIVE}"

  local native="false"
  [[ "${SPCONV_NATIVE}" == "1" ]] && native="true"
  "${PY}" - "${WORK_DIR}/experiment.json" "${EPOCHS}" "${BATCH_SIZE}" \
            "${GRID_SIZE}" "${native}" <<'PYEOF'
import json, sys
path, epochs, batch, grid, native = sys.argv[1:6]
json.dump({
    "name": "verify-mlops",
    "experiment_id": "verify-mlops",
    "task": "semantic_segmentation",
    "framework": "pointcept",
    "model": "pt-v3",
    # The label values in the generated .ply, in the order that *is* the class
    # index. -1 is deliberately absent: it means "unlabelled" to the trainer.
    "classes": [0, 1, 2],
    "split": {"method": "random", "train_percent": 75, "seed": 0},
    "hyperparameters": {
        "epochs": int(epochs), "batch_size": int(batch), "num_workers": 2,
        "grid_size": float(grid), "spconv_native": native == "true",
    },
}, open(path, "w"))
PYEOF

  # A run to log into. The trainer writes metrics and the model into whatever
  # MLFLOW_RUN_ID names; without one it skips model logging entirely and the
  # inference stage would have nothing to load.
  local run_id
  run_id="$("${PY}" - "${MLFLOW_URI}" "${MLFLOW_ARTIFACTS}" <<'PYEOF'
import sys, mlflow

# sqlite rather than a bare directory: MLflow 3 put the filesystem tracking
# backend into maintenance mode and raises rather than opening one, so a
# ./mlruns path fails before the trainer is ever started. The artifact root
# stays a plain directory -- only the tracking store moved.
uri, artifacts = sys.argv[1], sys.argv[2]
mlflow.set_tracking_uri(uri)
existing = mlflow.get_experiment_by_name("verify-mlops")
experiment_id = (
    existing.experiment_id
    if existing is not None
    else mlflow.create_experiment("verify-mlops", artifact_location=artifacts)
)
with mlflow.start_run(experiment_id=experiment_id) as run:
    print(run.info.run_id)
PYEOF
)" || { c_fail "  could not create an MLflow run"; return 1; }
  echo "${run_id}" > "${WORK_DIR}/run_id"
  c_ok "  mlflow run ${run_id}"

  GEO_WORK_DIR="${WORK_DIR}" \
  GEO_DATA_DIR="${DATA_DIR}" \
  GEO_CONFIG="${WORK_DIR}/experiment.json" \
  MLFLOW_TRACKING_URI="${MLFLOW_URI}" \
  MLFLOW_RUN_ID="${run_id}" \
  PYTHONPATH="${REPO_ROOT}:${APP_DIR}${SHIM_DIR:+:${SHIM_DIR}}" \
    "${PY}" "${APP_DIR}/pointcept_semseg.py" 2>&1 | tee "${WORK_DIR}/train.log"

  local rc="${PIPESTATUS[0]}"
  [[ "${rc}" == "0" ]] || { c_fail "  trainer exited ${rc}"; return 1; }
  grep -q "training finished" "${WORK_DIR}/train.log" || {
    c_fail "  trainer did not report 'training finished'"; return 1; }

  # An evaluation that never ran is the failure this whole script exists for:
  # it is the first place kernels run outside the training path.
  grep -q "Val result" "${WORK_DIR}/train.log" || {
    c_fail "  no evaluation result in the log"; return 1; }
  c_ok "  $(grep 'Val result' "${WORK_DIR}/train.log" | tail -1 | sed 's/.*Val result: //')"
}

# ------------------------------------------------------------------------------
# Stage: infer
#
# Loads what the trainer logged, through MLflow, exactly as the serving
# container does -- pointcept_pyfunc.py rebuilds the network from the saved
# config and checkpoint, so this is the first time that path runs at all.
# ------------------------------------------------------------------------------
stage_infer() {
  c_log "stage infer: load the logged model and predict"
  local uri
  uri="$(grep -o '\-> logged model .*' "${WORK_DIR}/train.log" | tail -1 | sed 's/-> logged model //')"
  [[ -n "${uri}" ]] || { c_fail "  the trainer logged no model"; return 1; }
  c_ok "  ${uri}"

  PYTHONPATH="${REPO_ROOT}:${APP_DIR}${SHIM_DIR:+:${SHIM_DIR}}" \
  MLFLOW_TRACKING_URI="${MLFLOW_URI}" \
    "${PY}" - "${uri}" "${DATA_DIR}/scene_00.ply" "${GRID_SIZE}" <<'PYEOF'
import base64, sys
import pandas as pd
import mlflow

uri, scene, grid = sys.argv[1], sys.argv[2], float(sys.argv[3])
model = mlflow.pyfunc.load_model(uri)
payload = base64.b64encode(open(scene, "rb").read()).decode("ascii")
frame = pd.DataFrame({"pointcloud_b64": [payload]})
out = model.predict(frame, params={"grid_size": grid, "include_ply": True})

# pointcept_pyfunc returns {"predictions": [...]}, one entry per input row.
# MLflow hands a PythonModel's return value back untouched for a local
# load_model, but a served model answers with the same dict as JSON, so accept
# either that or a frame rather than pinning one call path.
if isinstance(out, dict):
    records = out["predictions"]
elif isinstance(out, list):
    records = out
else:
    records = out.to_dict("records")
assert records, f"no predictions came back: {out!r}"
record = records[0]
assert record["task"] == "semantic_segmentation", record
assert abs(record["grid_size"] - grid) < 1e-9, record["grid_size"]
assert record["point_count"] > 0, record
classes = record.get("classes") or []
assert classes, "the model assigned no classes"
total = sum(c["points"] for c in classes)
assert total == record["point_count"], f"{total} != {record['point_count']}"
assert record.get("ply_b64"), "include_ply was true but no ply came back"
print(f"   {record['point_count']} points ->", ", ".join(
    f"{c['name']}:{c['ratio']:.3f}" for c in classes))
PYEOF
}

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
CLEANUP=()
cleanup() {
  if [[ "${KEEP_VENV}" == "0" ]]; then
    for path in "${CLEANUP[@]}"; do [[ -n "${path}" ]] && rm -rf "${path}"; done
  fi
}
trap cleanup EXIT

if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="$(mktemp -d -t pointcept-mlops-venv-XXXXXX)"
  rmdir "${VENV_DIR}"
  CLEANUP+=("${VENV_DIR}")
fi
PY="${VENV_DIR}/bin/python"

WORK_DIR="$(mktemp -d -t pointcept-mlops-work-XXXXXX)"
CLEANUP+=("${WORK_DIR}")
DATA_DIR="${WORK_DIR}/dataset"
APP_DIR="${WORK_DIR}/app"
MLFLOW_URI="sqlite:///${WORK_DIR}/mlflow.db"
MLFLOW_ARTIFACTS="${WORK_DIR}/artifacts"

c_log "${SCRIPT_NAME}: ${REPO_ROOT} against ${MLOPS_DIR}"

# Unlike verify_wheels.sh, which checks independent things and keeps going,
# each stage here feeds the next: after a failed install every later report is
# a restatement of it. So the run stops at the first break and the summary
# shows what never ran.
stage_env && rc=0 || rc=1; stage_result env "${rc}"
if [[ "${rc}" == "0" ]]; then
  stage_wheels && rc=0 || rc=1; stage_result wheels "${rc}"
fi
if [[ "${rc}" == "0" ]]; then
  # Copied rather than imported in place: the trainer resolves its sibling
  # modules next to __file__ (annotations.py, and the pyfunc it hands MLflow),
  # and MLflow copies that pyfunc into the artifact by path.
  mkdir -p "${APP_DIR}"
  cp "${RUNTIME_SRC}"/pointcept_semseg.py \
     "${RUNTIME_SRC}"/pointcept_pyfunc.py \
     "${RUNTIME_SRC}"/annotations.py "${APP_DIR}/"
  stage_deps && rc=0 || rc=1; stage_result deps "${rc}"
fi
if [[ "${rc}" == "0" ]]; then stage_data  && rc=0 || rc=1; stage_result data  "${rc}"; fi
if [[ "${rc}" == "0" ]]; then stage_train && rc=0 || rc=1; stage_result train "${rc}"; fi
if [[ "${rc}" == "0" ]]; then stage_infer && rc=0 || rc=1; stage_result infer "${rc}"; fi

echo
c_log "summary"
failed=0
for i in "${!STAGE_NAMES[@]}"; do
  if [[ "${STAGE_CODES[$i]}" == "0" ]]; then
    c_ok "${STAGE_NAMES[$i]}"
  else
    c_fail "${STAGE_NAMES[$i]}"; failed=1
  fi
done
if [[ "${KEEP_VENV}" == "1" ]]; then
  echo "venv kept at ${VENV_DIR}"
  echo "work kept at ${WORK_DIR}"
fi
exit "${failed}"
