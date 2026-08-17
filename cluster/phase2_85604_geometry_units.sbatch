#!/bin/bash
# Release audit for frozen 85604 geometry, SI units, and ensemble semantics.
# CPU-only and development-shot-only.
#SBATCH --job-name=p2_geomunit
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_geometry_units_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_geometry_units"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RESULT="${OUTDIR}/geometry_units.json"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly AUDITOR="${PAPER0_ROOT}/paper0/tools/audit_85604_geometry_units.py"
readonly GEOMETRY_CODE="${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
readonly TRANSPORT_CODE="${PAPER0_ROOT}/src/tcv_diagnostics/transport.py"
readonly METRICS_CODE="${PAPER0_ROOT}/src/tcv_diagnostics/metrics.py"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly GRID="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly HYPNOTOAD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hypnotoad-e4a1dff"
readonly HYPNOTOAD_MESH="${HYPNOTOAD_ROOT}/hypnotoad/core/mesh.py"
readonly BOUT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly BOUT_CONSTANTS="${BOUT_ROOT}/include/bout/constants.hxx"
readonly HERMES_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: got ${PAPER0_COMMIT}, expected ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 worktree is dirty; refusing provenance-ambiguous execution" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

check_clean_revision() {
  local repository="$1"
  local expected_revision="$2"
  local actual_revision
  actual_revision="$(git -C "${repository}" rev-parse HEAD)"
  if [[ "${actual_revision}" != "${expected_revision}" ]]; then
    echo "Revision mismatch for ${repository}: got ${actual_revision}, expected ${expected_revision}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${repository}" status --porcelain --untracked-files=all)" ]]; then
    echo "Source checkout is dirty: ${repository}" >&2
    git -C "${repository}" status --short >&2
    exit 2
  fi
}

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: got ${actual}, expected ${expected}" >&2
    exit 2
  fi
  echo "verified ${actual}  ${path}"
}

echo "Phase 2 geometry/unit audit started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

check_clean_revision "${HYPNOTOAD_ROOT}" "e4a1dff39b80e30aaa05eb6903a8dc72cf4ed832"
check_clean_revision "${BOUT_ROOT}" "7d28d67c3f12c24ec281c0982e870f5369c65a6f"
check_clean_revision "${HERMES_ROOT}" "920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${MANIFEST}"
check_sha256 "ef26a2f88a73826259be95fbe0f24cadece4dbf04a28e0eb116159d7afd7c478" "${AUDITOR}"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${GEOMETRY_CODE}"
check_sha256 "b788ca25d2aaa7991120e84be4235250f6331bb35710b3c76fd8a35f28584518" "${TRANSPORT_CODE}"
check_sha256 "0a7540a3e1b9698dea2293e729fd1f956986924cb3be6cc15a546add98521b64" "${METRICS_CODE}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"
check_sha256 "3c4a3d8f5b94ab728650726fbf010af70f63ae6452a83e024460d34ab99336e3" "${HYPNOTOAD_MESH}"
check_sha256 "4a89ceb00a66799668b1b73d3598e2995d9e171680be0d5ce0d20fe6b33e63b2" "${BOUT_CONSTANTS}"
check_sha256 "458eeecbd6da1afb882d0de2b652271fc2c2ca142c39a636a52f3adc5c16ef3f" "${HERMES_ROOT}/src/div_ops.cxx"
check_sha256 "3c766083078ec17d737a7ac595868adf1706e0596a9e614bb3ac73f071c1834d" "${HERMES_ROOT}/src/evolve_density.cxx"

{
  echo "${PYTHON} ${AUDITOR} --manifest ${MANIFEST} --geometry ${GRID} --hypnotoad-mesh-source ${HYPNOTOAD_MESH} --bout-constants ${BOUT_CONSTANTS} --hermes-source-root ${HERMES_ROOT} --output ${RESULT} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID}"
} > "${OUTDIR}/commands.sh"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import netCDF4, numpy; print("netCDF4=" + netCDF4.__version__); print("numpy=" + numpy.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

PYTHONPATH="${PAPER0_ROOT}/src" "${PYTHON}" \
  "${AUDITOR}" \
  --manifest "${MANIFEST}" \
  --geometry "${GRID}" \
  --hypnotoad-mesh-source "${HYPNOTOAD_MESH}" \
  --bout-constants "${BOUT_CONSTANTS}" \
  --hermes-source-root "${HERMES_ROOT}" \
  --output "${RESULT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
sha256sum \
  "${OUTDIR}/commands.sh" \
  "${OUTDIR}/environment.txt" \
  "${RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 geometry/unit audit finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
