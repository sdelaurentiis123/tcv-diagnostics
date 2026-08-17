#!/bin/bash
# Audit the saved radial potential-boundary state on 85604 only.
# CPU-only; no model, training, diagnostics, or held-out data are in scope.
#SBATCH --job-name=p2_phi_bnd
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_phi_boundary_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RAW_ROOT="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_phi_boundary_state"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_phi_boundary_state.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_PHI_BOUNDARY_STATE_PROTOCOL.md"
readonly AUDITOR="${PAPER0_ROOT}/paper0/tools/audit_85604_phi_boundary_state.py"
readonly CORE="${PAPER0_ROOT}/src/tcv_diagnostics/phi_boundary.py"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly RESULT="${OUTDIR}/phi_boundary_state_audit.json"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

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

echo "Phase 2 85604 phi-boundary-state audit started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi
if [[ "$(git -C "${HERMES_SOURCE}" rev-parse HEAD)" != "${EXPECTED_HERMES_COMMIT}" ]]; then
  echo "Hermes source revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${HERMES_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
  echo "Hermes tracked source is dirty" >&2
  exit 2
fi

check_sha256 "34462a32af60d8f683da49435803e12ffcb5e8bfcfc07a472b3866c6bad8ffb3" "${MANIFEST}"
check_sha256 "1a1f5f099d380997413792132dfe493b647e081d67ef9eaeabf28eed5e0fd9b6" "${PROTOCOL}"
check_sha256 "c2264c7acc5765078192336f4de9a5542bbc23d174580b16d798ca5b95a82cc1" "${AUDITOR}"
check_sha256 "c78bc34b82e225a32cb0c0764d17304c1c8e21cae58f0987d8a82c9d55869f99" "${CORE}"
check_sha256 "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a" "${HERMES_SOURCE}/src/vorticity.cxx"
check_sha256 "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded" "${HERMES_SOURCE}/include/vorticity.hxx"
check_sha256 "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad" "${RAW_ROOT}/BOUT.inp"
check_sha256 "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57" "${RAW_ROOT}/BOUT.settings"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${RAW_ROOT}/tcv_85604_adjusted.nc"

module purge
module load gcc/13.3.0
module load hdf5/1.12.3
module load netcdf-c/4.9.2

{
  echo "${PYTHON} ${AUDITOR} --manifest ${MANIFEST} --raw-root ${RAW_ROOT} --output ${RESULT} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID}"
} > "${OUTDIR}/commands.sh"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import netCDF4, numpy; print("netCDF4=" + netCDF4.__version__); print("numpy=" + numpy.__version__)'
  module list
} > "${OUTDIR}/environment.txt" 2>&1

"${PYTHON}" \
  "${AUDITOR}" \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --output "${RESULT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
sha256sum \
  "${OUTDIR}/commands.sh" \
  "${OUTDIR}/environment.txt" \
  "${RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 85604 phi-boundary-state audit finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
