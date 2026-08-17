#!/bin/bash
# Compare the Paper 0 shifted-xy face candidate with the executed Hermes source.
# CPU-only; reads 85604 geometry but no plasma frame or blind-test data.
#SBATCH --job-name=p2_hermes_xy_face
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_hermes_xy_face_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly GRID="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_xy_face"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${OUTDIR}/build"
readonly RUN_DIR="${OUTDIR}/run"
readonly ORACLE_SOURCE="${PAPER0_ROOT}/paper0/oracles/hermes_xy_face"
readonly COMPARATOR="${PAPER0_ROOT}/paper0/tools/compare_hermes_xy_face_oracle.py"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}" "${RUN_DIR}"
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

echo "Phase 2 compiled Hermes shifted-xy face oracle started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "SLURM job: ${SLURM_JOB_ID}"
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
check_sha256 "458eeecbd6da1afb882d0de2b652271fc2c2ca142c39a636a52f3adc5c16ef3f" "${HERMES_SOURCE}/src/div_ops.cxx"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"
check_sha256 "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333" "${BOUT_INSTALL}/lib64/libbout++.so.5.2.0"
check_sha256 "b5d86b8adedf11f3fab2783cbc75fd7b1064e87d91c7df596437fd7d26dd11e6" "${BOUT_INSTALL}/bin/bout-config"

module purge
module load gcc/13.3.0
module load openmpi/4.1.8
module load cmake/3.31.6
module load ninja/1.12.1
module load hdf5/1.12.3
module load netcdf-c/4.9.2
module load fftw/3.3.10

cp "${ORACLE_SOURCE}/BOUT.inp" "${RUN_DIR}/BOUT.inp"
{
  echo "cmake -S ${ORACLE_SOURCE} -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${BOUT_INSTALL}"
  echo "cmake --build ${BUILD_DIR} --parallel ${SLURM_CPUS_PER_TASK}"
  echo "srun --ntasks=4 ${BUILD_DIR}/hermes_xy_face_oracle -d ${RUN_DIR}"
  echo "PYTHONPATH=${PAPER0_ROOT}/src python ${COMPARATOR} --bout-output ${RUN_DIR}/BOUT.dmp.[0-3].nc --grid ${GRID} --output ${OUTDIR}/oracle_comparison.json --arrays ${OUTDIR}/comparison_arrays.npz --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --atol 5e-10 --rtol 5e-10"
} > "${OUTDIR}/commands.sh"

cmake \
  -S "${ORACLE_SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS=1
srun --ntasks=4 "${BUILD_DIR}/hermes_xy_face_oracle" -d "${RUN_DIR}"
mapfile -t BOUT_OUTPUTS < <(find "${RUN_DIR}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V)
if [[ "${#BOUT_OUTPUTS[@]}" -ne 4 ]]; then
  echo "Expected four BOUT++ rank outputs; found ${#BOUT_OUTPUTS[@]}" >&2
  exit 2
fi

set +e
PYTHONPATH="${PAPER0_ROOT}/src" \
  /mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python \
  "${COMPARATOR}" \
  --bout-output "${BOUT_OUTPUTS[@]}" \
  --grid "${GRID}" \
  --output "${OUTDIR}/oracle_comparison.json" \
  --arrays "${OUTDIR}/comparison_arrays.npz" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --atol 5e-10 \
  --rtol 5e-10
readonly COMPARISON_STATUS=$?
set -e

HASH_ARTIFACTS=(
  "${OUTDIR}/commands.sh"
  "${RUN_DIR}/BOUT.inp"
  "${BUILD_DIR}/hermes_xy_face_oracle"
  "${BOUT_OUTPUTS[@]}"
)
if [[ -f "${OUTDIR}/oracle_comparison.json" ]]; then
  HASH_ARTIFACTS+=("${OUTDIR}/oracle_comparison.json")
fi
if [[ -f "${OUTDIR}/comparison_arrays.npz" ]]; then
  HASH_ARTIFACTS+=("${OUTDIR}/comparison_arrays.npz")
fi
sha256sum "${HASH_ARTIFACTS[@]}" > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 compiled Hermes shifted-xy face oracle finished at $(date --iso-8601=seconds)"
echo "Comparison exit status: ${COMPARISON_STATUS}"
echo "Result directory: ${OUTDIR}"
exit "${COMPARISON_STATUS}"
