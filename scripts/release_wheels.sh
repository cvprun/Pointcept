#!/usr/bin/env bash
# ==============================================================================
# Pointcept wheel release
#
# Publishes each wheelhouse directory (linux-<arch>/<accel>/torch<ver>-cp<py>)
# as its own GitHub Release, tagged with the Pointcept version it was built
# from:
#
#   v1.7.0-linux-amd64-cu128-torch2.9.1-cp312
#
# One release per directory rather than one per version, because the wheels
# of different CUDA lines share filenames (pointops-1.0-cp312-...x86_64.whl is
# in cu126 and cu128 alike) and a release cannot hold two assets of one name.
# pip reads a release directly:
#
#   pip install --find-links https://github.com/<repo>/releases/expanded_assets/<tag> ...
#
# With --build the host's matrix (build_matrix_<arch>.sh) runs first. Nothing
# is verified: the free GitHub runners have no GPU, so this runs on the build
# host and uploads what it built.
#
# Re-running is safe: an existing release has its assets replaced, and assets
# no longer in the directory are deleted, so a release always mirrors its
# directory.
#
# Quick start
#   ./scripts/release_wheels.sh --dry-run            # show what would go up
#   ./scripts/release_wheels.sh path/to/root         # upload another root
#   ./scripts/release_wheels.sh --build              # build this host's matrix, upload it
#   ./scripts/release_wheels.sh --build -- --jobs 8  # args after -- go to the matrix
#
# Author: Pointcept contributors
# ==============================================================================

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"

# The fork carries none of upstream's tags, so the version is looked up there
# when the local history has no v* tag of its own.
UPSTREAM_REPO="Pointcept/Pointcept"

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_RED=$'\033[31m'; C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'
else
  C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""
fi

log()  { echo "${C_BLUE}[release]${C_RESET} $*" >&2; }
info() { echo "${C_GREEN}[  ok   ]${C_RESET} $*" >&2; }
warn() { echo "${C_YELLOW}[ warn  ]${C_RESET} $*" >&2; }
die()  { echo "${C_RED}[ error ]${C_RESET} $*" >&2; exit 1; }

usage() {
  cat <<EOF
${C_BOLD}USAGE${C_RESET}
  ${SCRIPT_NAME} [OPTIONS] [WHEELHOUSE_ROOT] [-- MATRIX_ARGS...]

${C_BOLD}ARGUMENTS${C_RESET}
  WHEELHOUSE_ROOT    Directory holding linux-<arch>/<accel>/torch<ver>-cp<py>/
                     [wheelhouse, the output of build_matrix_*.sh]
  MATRIX_ARGS        Passed to build_matrix_<arch>.sh with --build

${C_BOLD}OPTIONS${C_RESET}
  --build            Run build_matrix_<host arch>.sh first
  --version VER      Pointcept version for the tags   [nearest upstream v* tag]
  --repo OWNER/NAME  Repository to publish to         [the gh default: origin]
  --dry-run          Print the releases and assets; change nothing
  -h, --help         Show this message
EOF
}

BUILD="0"
VERSION=""
REPO=""
DRY_RUN="0"
ROOT=""
MATRIX_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)    BUILD="1"; shift ;;
    --version)  VERSION="${2:?--version needs a value}"; shift 2 ;;
    --repo)     REPO="${2:?--repo needs OWNER/NAME}"; shift 2 ;;
    --dry-run)  DRY_RUN="1"; shift ;;
    -h|--help)  usage; exit 0 ;;
    --)         shift; MATRIX_ARGS=("$@"); break ;;
    -*)         die "unknown option: $1 (see --help)" ;;
    *)          ROOT="$1"; shift ;;
  esac
done

ROOT="${ROOT:-${REPO_ROOT}/wheelhouse}"

command -v gh >/dev/null || die "gh (GitHub CLI) is required"
gh auth status >/dev/null 2>&1 || die "gh is not logged in: run 'gh auth login'"
GH_REPO_ARGS=()
[[ -z "${REPO}" ]] || GH_REPO_ARGS=(--repo "${REPO}")
REPO="${REPO:-$(cd "${REPO_ROOT}" && gh repo view --json nameWithOwner -q .nameWithOwner)}"

# ------------------------------------------------------------------------------
# Version and target commit
# ------------------------------------------------------------------------------
# The newest plain version tag (v1.7.0) in this history, else the newest
# upstream one whose commit this history contains -- the release the fork was
# branched from. Only a bare version counts: the release tags this script
# creates start with one too, and taking one of those for the version would
# nest every tag inside the last.
VERSION_RE='^v[0-9]+(\.[0-9]+)*$'
detect_version() {
  local v
  v="$(git -C "${REPO_ROOT}" tag --merged HEAD --sort=-v:refname | grep -E "${VERSION_RE}" | head -1 || true)"
  if [[ -n "${v}" ]]; then
    echo "${v}"; return 0
  fi
  local name sha
  while read -r name sha; do
    if git -C "${REPO_ROOT}" merge-base --is-ancestor "${sha}" HEAD 2>/dev/null; then
      echo "${name}"; return 0
    fi
  done < <(gh api "repos/${UPSTREAM_REPO}/tags" --paginate \
             --jq '.[] | .name + " " + .commit.sha' | grep -E "${VERSION_RE%$} ")
  return 1
}

[[ -n "${VERSION}" ]] || VERSION="$(detect_version)" \
  || die "cannot tell the Pointcept version; pass --version (e.g. v1.7.0)"
[[ "${VERSION}" == v* ]] || VERSION="v${VERSION}"
[[ "${VERSION}" =~ ${VERSION_RE} ]] || die "not a plain version: ${VERSION} (expected e.g. v1.7.0)"

# A tag has to point at a commit GitHub already has. HEAD may be ahead of the
# remote, so the tag goes on the newest commit both share.
git -C "${REPO_ROOT}" fetch --quiet origin 2>/dev/null || warn "git fetch origin failed; using the last known remote state"
UPSTREAM_REF="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref '@{upstream}' 2>/dev/null || echo origin/main)"
TARGET="$(git -C "${REPO_ROOT}" merge-base HEAD "${UPSTREAM_REF}")" \
  || die "HEAD shares no commit with ${UPSTREAM_REF}"
if [[ "${TARGET}" != "$(git -C "${REPO_ROOT}" rev-parse HEAD)" ]]; then
  warn "HEAD is not on ${UPSTREAM_REF}; tags will point at ${TARGET:0:7}, the newest pushed commit"
fi

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
if [[ "${BUILD}" == "1" ]]; then
  case "$(uname -m)" in
    x86_64)        host_arch="amd64" ;;
    aarch64|arm64) host_arch="arm64" ;;
    *)             die "no build matrix for $(uname -m)" ;;
  esac
  [[ "$(readlink -f "${ROOT}")" == "${REPO_ROOT}/wheelhouse" ]] \
    || die "--build writes to wheelhouse; do not pass another WHEELHOUSE_ROOT with it"
  log "building the ${host_arch} matrix"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "would run build_matrix_${host_arch}.sh ${MATRIX_ARGS[*]}"
  else
    # --keep-going inside means a failed combination does not stop the rest;
    # its directory then lacks wheels, and the manifest check below catches it.
    bash "${REPO_ROOT}/scripts/build_matrix_${host_arch}.sh" ${MATRIX_ARGS[@]+"${MATRIX_ARGS[@]}"} \
      || warn "the matrix reported failures; affected directories are skipped below"
  fi
fi

# ------------------------------------------------------------------------------
# Release
# ------------------------------------------------------------------------------
[[ -d "${ROOT}" ]] || die "no wheelhouse at ${ROOT}"
ROOT="$(readlink -f "${ROOT}")"

mapfile -t DIRS < <(find "${ROOT}" -mindepth 3 -maxdepth 3 -type d -path "${ROOT}/linux-*/*/torch*-cp*" | sort)
[[ ${#DIRS[@]} -gt 0 ]] || die "no linux-<arch>/<accel>/torch<ver>-cp<py> directories under ${ROOT}"

# The manifest is appended to on every run; only the last run's lines describe
# the wheels in the directory now.
last_run() { awk '/^# [0-9-]+ [0-9:]+Z build_wheels\.sh /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$1"; }

write_notes() {
  local dir="$1" tag="$2" arch="$3" accel="$4" tp="$5" out="$6"
  local run archs=""
  local torch="${tp%-cp*}" py="${tp#*-cp}"
  torch="${torch#torch}"; py="${py:0:1}.${py:1}"
  run="$(last_run "${dir}/manifest.txt" 2>/dev/null || true)"
  archs="$(sed -n 's/^# cuda arch list: \([^ ]*\( [^ ]*\)*\)  rocm arch:.*/\1/p' <<<"${run}" | tail -1)"
  {
    echo "Prebuilt native dependencies for **Pointcept ${VERSION}**."
    echo
    echo "| | |"
    echo "|---|---|"
    echo "| Platform | linux/${arch} (manylinux_2_28) |"
    echo "| Accelerator | ${accel} |"
    echo "| PyTorch / CPython | ${torch} / ${py} |"
    [[ -z "${archs}" ]] || echo "| CUDA archs | ${archs} |"
    echo "| Source | ${REPO}@${TARGET:0:7} |"
    echo
    echo "These wheels are built but not verified on a GPU."
    echo
    echo '```bash'
    echo "pip install --find-links https://github.com/${REPO}/releases/expanded_assets/${tag} \\"
    echo "  spconv-${accel} torch-scatter torch-sparse torch-cluster pointops pointops2 \\"
    echo "  pointgroup_ops pointseg pointrope swin3d flash-attn"
    echo "pip install -r https://github.com/${REPO}/releases/download/${tag}/requirements-pypi.txt"
    echo '```'
    echo
    echo "Install torch==${torch} from download.pytorch.org/whl/${accel} first; every wheel here is"
    echo "linked against it."
  } > "${out}"
}

upload_with_retry() {
  local tag="$1" file="$2" n
  for n in 1 2 3; do
    gh release upload "${GH_REPO_ARGS[@]}" "${tag}" "${file}" --clobber >/dev/null && return 0
    warn "upload of $(basename "${file}") failed (attempt ${n}/3)"
    sleep $((n * 10))
  done
  return 1
}

publish_dir() {
  local dir="$1"
  local rel="${dir#"${ROOT}"/}"
  local arch accel tp
  IFS=/ read -r arch accel tp <<<"${rel}"
  arch="${arch#linux-}"
  local tag="${VERSION}-linux-${arch}-${accel}-${tp}"
  local title="Pointcept ${VERSION} wheels: linux-${arch} ${accel} ${tp/-/ }"

  echo >&2
  log "${C_BOLD}${tag}${C_RESET}"

  # A directory built before the PyPI split has no requirements-pypi.txt; its
  # consumers need the list all the same.
  if [[ ! -f "${dir}/requirements-pypi.txt" && "${DRY_RUN}" == "0" ]]; then
    bash "${REPO_ROOT}/scripts/build_wheels.sh" --list-pypi > "${dir}/requirements-pypi.txt"
    log "  wrote requirements-pypi.txt"
  fi

  shopt -s nullglob
  local -a files=("${dir}"/*.whl)
  [[ -f "${dir}/manifest.txt" ]]          && files+=("${dir}/manifest.txt")
  [[ -f "${dir}/requirements-pypi.txt" || "${DRY_RUN}" == "1" ]] && files+=("${dir}/requirements-pypi.txt")
  shopt -u nullglob

  log "  ${#files[@]} assets"
  if ! compgen -G "${dir}/*.whl" >/dev/null; then
    warn "  no wheels; skipped"; return 1
  fi
  if [[ -f "${dir}/manifest.txt" ]] && last_run "${dir}/manifest.txt" | grep -q '^FAILED'; then
    warn "  its last build has failures ($(last_run "${dir}/manifest.txt" | awk '/^FAILED/{print $2}' | xargs)); skipped"
    return 1
  fi

  local notes; notes="$(mktemp)"
  write_notes "${dir}" "${tag}" "${arch}" "${accel}" "${tp}" "${notes}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    local f; for f in "${files[@]}"; do echo "    $(basename "${f}")" >&2; done
    rm -f "${notes}"; return 0
  fi

  # name -> size of what the release already holds; an asset of the same name
  # and size is taken as unchanged and not sent again.
  local -A remote=()
  local name size
  if gh release view "${GH_REPO_ARGS[@]}" "${tag}" >/dev/null 2>&1; then
    while read -r name size; do
      [[ -n "${name}" ]] && remote["${name}"]="${size}"
    done < <(gh release view "${GH_REPO_ARGS[@]}" "${tag}" --json assets -q '.assets[] | .name + " " + (.size | tostring)')
    gh release edit "${GH_REPO_ARGS[@]}" "${tag}" --title "${title}" --notes-file "${notes}" >/dev/null
    # Mirror the directory: drop what is no longer in it.
    local -A keep=(); local f
    for f in "${files[@]}"; do keep["$(basename "${f}")"]=1; done
    for name in "${!remote[@]}"; do
      [[ -z "${keep[${name}]:-}" ]] || continue
      gh release delete-asset "${GH_REPO_ARGS[@]}" "${tag}" "${name}" -y >/dev/null
      log "  removed stale ${name}"
    done
  else
    # --latest=false: a wheel drop is not a Pointcept release and must not take
    # the repository's "Latest" badge.
    gh release create "${GH_REPO_ARGS[@]}" "${tag}" --target "${TARGET}" \
      --title "${title}" --notes-file "${notes}" --latest=false >/dev/null
    log "  created release"
  fi
  rm -f "${notes}"

  local rc=0
  for f in "${files[@]}"; do
    if [[ "${remote[$(basename "${f}")]:-}" == "$(stat -c %s "${f}")" ]]; then
      log "  unchanged $(basename "${f}")"
      continue
    fi
    if upload_with_retry "${tag}" "${f}"; then
      info "  $(basename "${f}")"
    else
      warn "  giving up on $(basename "${f}")"; rc=1
    fi
  done
  [[ ${rc} -eq 0 ]] && info "https://github.com/${REPO}/releases/tag/${tag}"
  return "${rc}"
}

log "repo ${REPO}  version ${VERSION}  target ${TARGET:0:7}  root ${ROOT}"
[[ "${DRY_RUN}" == "0" ]] || log "dry run: nothing is created or uploaded"

failed=()
for dir in "${DIRS[@]}"; do
  publish_dir "${dir}" || failed+=("${dir#"${ROOT}"/}")
done

echo >&2
if [[ ${#failed[@]} -gt 0 ]]; then
  warn "not published: ${failed[*]}"
  exit 1
fi
info "published ${#DIRS[@]} release(s)"
