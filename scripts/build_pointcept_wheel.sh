#!/usr/bin/env bash
# ==============================================================================
# Pointcept source wheel
#
# Packages Pointcept's python source as pointcept-<version>-py3-none-any.whl,
# so a consumer installs it like every other wheel in the wheelhouse instead of
# unpacking a source tree and putting it on PYTHONPATH.
#
# The wheel is pure python. pointcept/ holds no compiled code, loads none from
# its own directory and builds none at import time: every kernel it calls comes
# from a separate wheel -- libs/ (pointops, pointgroup_ops, ...) and the
# third-party builds (spconv, torch-scatter, flash-attn, ...) build_wheels.sh
# produces per target. So one file serves every (arch x accel x torch x python)
# directory, and its tag says so.
#
# What goes in is what a consumer of the source tree reads:
#
#   pointcept/   the package, every file of it
#   configs/     as pointcept/configs/. Configs are addressed by path rather
#                than imported, and `_base_` resolves against the config's own
#                directory, so the tree goes in whole and unchanged, next to
#                the code: what a checkout calls configs/scannet/x.py is
#                <dirname(pointcept.__file__)>/configs/scannet/x.py here
#   LICENSE      into the .dist-info
#
# libs/ ships as its own wheels; tools/ and scripts/ are entry points for a
# checkout and stay out.
#
# "Every file" is meant literally. pointcept/datasets/preprocessing/ has no
# __init__.py, yet the ScanNet200 configs import their class list from it, so
# discovering packages by __init__.py would drop it and leave a wheel that
# imports cleanly and fails on the first config read. The wheel is assembled
# by directory instead, and checked against the commit's file list afterwards.
#
# The wheel declares no dependencies. torch has to come from the index of the
# accelerator the kernels were built for, before anything else is installed;
# the kernels are distributions whose names carry that accelerator
# (spconv-cu130); and the pure python closure is installed by each consumer next
# to the wheelhouse (requirements-pypi.txt, RUNTIME_DEPS in verify_wheels.sh). A
# Requires-Dist here would let a resolver pull a torch of its own while
# installing the wheelhouse.
#
# Built from a commit, not the working tree, through `git archive`. The version
# names that commit:
#
#   1.7.0.post12+g92f4d68      12 commits past v1.7.0, at 92f4d68
#
# The base is the newest plain v* tag the commit contains, this repository's or
# else upstream's (the fork carries none of its own), or --version; .postN
# orders builds and the local label tells two of them apart. Every commit gets its own filename,
# which matters because release_wheels.sh skips an asset whose name and size a
# release already holds, and installers cache by filename. The build is
# reproducible -- a pinned backend, stamped with the commit time -- so one
# commit built twice gives the same bytes.
#
# Quick start
#   ./scripts/build_pointcept_wheel.sh                      # dist/pointcept-*.whl from HEAD
#   ./scripts/build_pointcept_wheel.sh --into wheelhouse    # and into every wheelhouse directory
#   ./scripts/build_pointcept_wheel.sh --ref v1.7.0 --out /tmp/w
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"

# Upstream's tags, for a history that has no v* tag of its own (see
# release_wheels.sh, which picks its release version the same way).
UPSTREAM_URL="https://github.com/Pointcept/Pointcept.git"

# The backend writes its own name and version into the wheel, so it is pinned:
# otherwise the same commit built next month is a different file of the same
# name.
BUILD_BACKEND="hatchling==1.32.4"

VERSION_RE='^v[0-9]+(\.[0-9]+)*$'

if [[ -t 2 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""
fi

log()  { echo "${C_BLUE}[pointcept]${C_RESET} $*" >&2; }
info() { echo "${C_GREEN}[   ok    ]${C_RESET} $*" >&2; }
warn() { echo "${C_YELLOW}[  warn   ]${C_RESET} $*" >&2; }
die()  { echo "${C_RED}[  error  ]${C_RESET} $*" >&2; exit 1; }

usage() {
  cat <<EOF
${C_BOLD}USAGE${C_RESET}
  ${SCRIPT_NAME} [OPTIONS]

${C_BOLD}OPTIONS${C_RESET}
  --ref REF          Commit to package                  [HEAD]
  --version VER      Base version, e.g. v1.7.0          [newest v* tag in REF, else upstream's]
  --out DIR          Where the wheel is written         [dist]
  --into ROOT        Also place it in every linux-<arch>/<accel>/torch<ver>-cp<py>/
                     directory under ROOT, replacing any other pointcept wheel there
  -h, --help         Show this message

The wheel's path is printed on stdout; everything else goes to stderr.
EOF
}

REF="HEAD"
VERSION=""
OUT="${REPO_ROOT}/dist"
INTO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)      REF="${2:?--ref needs a commit}"; shift 2 ;;
    --version)  VERSION="${2:?--version needs a value}"; shift 2 ;;
    --out)      OUT="${2:?--out needs a directory}"; shift 2 ;;
    --into)     INTO="${2:?--into needs a directory}"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)          die "unknown argument: $1 (see --help)" ;;
  esac
done

git_() { git -C "${REPO_ROOT}" "$@"; }

git_ rev-parse --git-dir >/dev/null 2>&1 || die "${REPO_ROOT} is not a git checkout"
COMMIT="$(git_ rev-parse --verify -q "${REF}^{commit}")" || die "not a commit: ${REF}"
SHORT="$(git_ rev-parse --short=7 "${COMMIT}")"

if [[ -n "${VERSION}" ]]; then
  [[ "${VERSION}" == v* ]] || VERSION="v${VERSION}"
  [[ "${VERSION}" =~ ${VERSION_RE} ]] || die "not a plain version: ${VERSION} (expected e.g. v1.7.0)"
fi
if [[ -n "${INTO}" ]]; then
  [[ -d "${INTO}" ]] || die "no such directory: ${INTO}"
  INTO="$(readlink -f "${INTO}")"
fi

# ------------------------------------------------------------------------------
# Version
# ------------------------------------------------------------------------------
# Prints "<tag> <commit>" for the version the build counts from: VERSION when
# given, else the newest plain v* tag COMMIT contains. Release tags
# (v1.7.0-linux-...) are not versions and never match.
resolve_base() {
  local name sha ref
  while read -r name; do
    [[ -z "${VERSION}" || "${name}" == "${VERSION}" ]] || continue
    echo "${name} $(git_ rev-parse "${name}^{commit}")"; return 0
  done < <(git_ tag --merged "${COMMIT}" --sort=-v:refname | grep -E "${VERSION_RE}" || true)

  # ls-remote needs no token, unlike the API. An annotated tag is listed twice
  # and its peeled line (name^{}) is the one naming the commit; it sorts after
  # the other, so the last line seen for a name wins.
  local peel='^{}'
  local -A commits=()
  while read -r sha ref; do
    name="${ref#refs/tags/}"; name="${name%"${peel}"}"
    [[ "${name}" =~ ${VERSION_RE} ]] && commits["${name}"]="${sha}"
  done < <(git ls-remote --tags "${UPSTREAM_URL}" 2>/dev/null || true)
  while read -r name; do
    [[ -n "${name}" ]] || continue
    [[ -z "${VERSION}" || "${name}" == "${VERSION}" ]] || continue
    sha="${commits[${name}]}"
    git_ merge-base --is-ancestor "${sha}" "${COMMIT}" 2>/dev/null || continue
    echo "${name} ${sha}"; return 0
  done < <(printf '%s\n' "${!commits[@]}" | sort -rV)
  return 1
}

if BASE="$(resolve_base)"; then
  read -r BASE_TAG BASE_COMMIT <<<"${BASE}"
  DISTANCE="$(git_ rev-list --count "${BASE_COMMIT}..${COMMIT}")"
elif [[ -n "${VERSION}" ]]; then
  # A version not tagged yet (release_wheels.sh --version) is still a version;
  # there is just nothing to count commits from.
  warn "${VERSION} is not a tag ${SHORT} contains, here or upstream; no .postN"
  BASE_TAG="${VERSION}"; DISTANCE="0"
else
  die "cannot tell the Pointcept version of ${SHORT}; pass --version (e.g. v1.7.0)"
fi
WHEEL_VERSION="${BASE_TAG#v}"
[[ "${DISTANCE}" == "0" ]] || WHEEL_VERSION+=".post${DISTANCE}"
WHEEL_VERSION+="+g${SHORT}"
WHEEL_NAME="pointcept-${WHEEL_VERSION}-py3-none-any.whl"

log "${C_BOLD}${WHEEL_NAME}${C_RESET} from ${COMMIT}"
if [[ "${COMMIT}" == "$(git_ rev-parse HEAD)" \
      && -n "$(git_ status --porcelain -- pointcept configs LICENSE)" ]]; then
  warn "uncommitted changes under pointcept/ or configs/ are not in the wheel; it is built from ${SHORT}"
fi

# https://github.com/<owner>/<repo> of origin, for the metadata; nothing when
# origin is not on GitHub.
github_url() {
  local url; url="$(git_ remote get-url origin 2>/dev/null)" || return 0
  url="${url%.git}"
  case "${url}" in
    https://github.com/*)    echo "${url}" ;;
    git@github.com:*)        echo "https://github.com/${url#git@github.com:}" ;;
    ssh://git@github.com/*)  echo "https://github.com/${url#ssh://git@github.com/}" ;;
  esac
}

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
STAGE="$(mktemp -d -t pointcept-wheel-XXXXXX)"
trap 'rm -rf "${STAGE}"' EXIT
SRC="${STAGE}/src"
mkdir -p "${SRC}"

git_ archive --format=tar "${COMMIT}" pointcept configs LICENSE | tar -x -C "${SRC}"
[[ ! -e "${SRC}/pointcept/configs" ]] \
  || die "pointcept/configs already exists at ${SHORT}; configs/ has nowhere to go"
mv "${SRC}/configs" "${SRC}/pointcept/configs"

URL="$(github_url)"
URLS=""
[[ -z "${URL}" ]] || URLS=$'\n[project.urls]\nSource = "'"${URL}/tree/${COMMIT}"$'"\n'
cat > "${SRC}/pyproject.toml" <<EOF
[build-system]
requires = ["${BUILD_BACKEND}"]
build-backend = "hatchling.build"

[project]
name = "pointcept"
version = "${WHEEL_VERSION}"
description = "Pointcept point cloud perception codebase: python source and configs"
license = "MIT"
license-files = ["LICENSE"]
# environment.yml's interpreter. It also keeps hatchling from tagging the wheel
# py2.py3, which it does whenever nothing rules Python 2 out.
requires-python = ">=3.10"
${URLS}
[tool.hatch.build.targets.wheel]
packages = ["pointcept"]
# hatchling 1.32 writes 2.5 by default, newer than some installers still in use
# read. 2.4 is the oldest that carries License-Expression.
core-metadata-version = "2.4"
EOF

# hatchling zips every file with this timestamp rather than the time of the
# build, which is what makes the result a function of the commit alone.
export SOURCE_DATE_EPOCH
SOURCE_DATE_EPOCH="$(git_ log -1 --format=%ct "${COMMIT}")"

UV_BIN="$(command -v uv || true)"
[[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]] && UV_BIN="${HOME}/.local/bin/uv"
if [[ -n "${UV_BIN}" ]]; then
  "${UV_BIN}" build --wheel --out-dir "${STAGE}/dist" "${SRC}" >&2 \
    || die "uv build failed"
elif python3 -m pip --version >/dev/null 2>&1; then
  python3 -m pip wheel --no-deps --wheel-dir "${STAGE}/dist" "${SRC}" >&2 \
    || die "pip wheel failed"
else
  die "neither uv nor pip is available to build with"
fi
BUILT="${STAGE}/dist/${WHEEL_NAME}"
if [[ ! -f "${BUILT}" ]]; then
  shopt -s nullglob; got=("${STAGE}"/dist/*); shopt -u nullglob
  die "the build did not produce ${WHEEL_NAME}: ${got[*]##*/}"
fi

# ------------------------------------------------------------------------------
# Check
#
# The wheel has to hold exactly the commit's files -- one missing is a config
# that fails to load on some other machine, one extra is a stray build product
# -- and nothing that would make it something other than pure python.
# ------------------------------------------------------------------------------
git_ ls-tree -r -z --name-only "${COMMIT}" -- pointcept configs > "${STAGE}/expected"
python3 - "${BUILT}" "${STAGE}/expected" <<'PYEOF' || die "the wheel does not match ${SHORT}"
import sys
import zipfile

wheel, expected_list = sys.argv[1], sys.argv[2]
with open(expected_list, "rb") as f:
    names = [n.decode("utf-8") for n in f.read().split(b"\0") if n]
expected = {"pointcept/" + n if n.startswith("configs/") else n for n in names}

with zipfile.ZipFile(wheel) as z:
    entries = [n for n in z.namelist() if not n.endswith("/")]
    info = next(n for n in entries if n.endswith(".dist-info/WHEEL"))
    meta = z.read(info).decode("utf-8").splitlines()

files = {n for n in entries if not n.split("/", 1)[0].endswith(".dist-info")}
bad = 0
for label, diff in (("missing", expected - files), ("unexpected", files - expected)):
    for name in sorted(diff):
        print(f"  {label}: {name}", file=sys.stderr)
        bad += 1
compiled = sorted(n for n in files if n.endswith((".so", ".pyd", ".dylib", ".dll", ".pyc")))
for name in compiled:
    print(f"  compiled: {name}", file=sys.stderr)
    bad += 1
for line in ("Root-Is-Purelib: true", "Tag: py3-none-any"):
    if line not in meta:
        print(f"  WHEEL lacks '{line}'", file=sys.stderr)
        bad += 1
if not bad:
    print(f"  {len(files)} files, py3-none-any", file=sys.stderr)
raise SystemExit(1 if bad else 0)
PYEOF

mkdir -p "${OUT}"
OUT="$(readlink -f "${OUT}")"
cp "${BUILT}" "${OUT}/${WHEEL_NAME}"
info "${OUT}/${WHEEL_NAME} ($(du -h "${BUILT}" | cut -f1))"

# ------------------------------------------------------------------------------
# Into the wheelhouse
#
# Any other pointcept wheel in a directory is an older build: installing both
# would leave the choice to the resolver, and release_wheels.sh mirrors a
# directory into its release, so a stale one would be published too.
# ------------------------------------------------------------------------------
if [[ -n "${INTO}" ]]; then
  mapfile -t DIRS < <(find "${INTO}" -mindepth 3 -maxdepth 3 -type d \
                         -path "${INTO}/linux-*/*/torch*-cp*" | sort)
  [[ ${#DIRS[@]} -gt 0 ]] || die "no linux-<arch>/<accel>/torch<ver>-cp<py> directories under ${INTO}"
  for dir in "${DIRS[@]}"; do
    rel="${dir#"${INTO}"/}"
    if cmp -s "${BUILT}" "${dir}/${WHEEL_NAME}"; then
      log "  unchanged ${rel}"
      continue
    fi
    for old in "${dir}"/pointcept-*.whl; do
      [[ -e "${old}" && "$(basename "${old}")" != "${WHEEL_NAME}" ]] || continue
      rm -f "${old}"
      log "  removed $(basename "${old}") from ${rel}"
    done
    cp "${BUILT}" "${dir}/${WHEEL_NAME}"
    # manifest.txt says how each wheel in the directory was obtained.
    [[ ! -f "${dir}/manifest.txt" ]] \
      || echo "packaged  ${WHEEL_NAME}  (${SCRIPT_NAME}, ${SHORT})" >> "${dir}/manifest.txt"
    info "  ${rel}"
  done
fi

echo "${OUT}/${WHEEL_NAME}"
