#!/bin/bash
# Compare the Paper 0 NumPy shifted-DDY candidate with exact compiled BOUT++.
# CPU-only; reads the 85604 geometry but no plasma-state frame or blind-test data.
#SBATCH --job-name=p2_shifted_ddy
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_shifted_ddy_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890722/install"
readonly GRID="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_shifted_ddy"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${OUTDIR}/build"
readonly RUN_DIR="${OUTDIR}/run"
readonly ORACLE_SOURCE="${PAPER0_ROOT}/paper0/oracles/bout_shifted_ddy"
readonly COMPARATOR="${PAPER0_ROOT}/paper0/tools/compare_shifted_ddy_oracle.py"
readonly BOUT_OUTPUT="${RUN_DIR}/BOUT.dmp.0.nc"

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

echo "Phase 2 compiled shifted-DDY oracle started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"
check_sha256 "fa11368c90e5c392b20290910d5856b39c122154cbcf3b814a1e98ed65f9a6ce" "${BOUT_INSTALL}/lib64/libbout++.so.5.2.0"
check_sha256 "aa20fd55d7da46f31e87c8bc09b8a725c4ba3a94abe73215d4e06f4c55c6a480" "${BOUT_INSTALL}/bin/bout-config"

module purge
module load gcc/13.3.0
module load openmpi/4.1.8
module load cmake/3.31.6
module load ninja/1.12.1
module load hdf5/1.14.5
module load netcdf-c/4.9.2
module load fftw/3.3.10

cp "${ORACLE_SOURCE}/BOUT.inp" "${RUN_DIR}/BOUT.inp"

{
  echo "cmake -S ${ORACLE_SOURCE} -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${BOUT_INSTALL}"
  echo "cmake --build ${BUILD_DIR} --parallel ${SLURM_CPUS_PER_TASK}"
  echo "srun --ntasks=1 ${BUILD_DIR}/shifted_ddy_oracle -d ${RUN_DIR}"
  echo "PYTHONPATH=${PAPER0_ROOT}/src python ${COMPARATOR} --bout-output ${BOUT_OUTPUT} --grid ${GRID} --output ${OUTDIR}/oracle_comparison.json --arrays ${OUTDIR}/comparison_arrays.npz --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --atol 5e-10 --rtol 5e-10"
} > "${OUTDIR}/commands.sh"

cmake \
  -S "${ORACLE_SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS=1
srun --ntasks=1 "${BUILD_DIR}/shifted_ddy_oracle" -d "${RUN_DIR}"
if [[ ! -f "${BOUT_OUTPUT}" ]]; then
  echo "Expected BOUT++ output is missing: ${BOUT_OUTPUT}" >&2
  exit 2
fi

set +e
PYTHONPATH="${PAPER0_ROOT}/src" \
  /mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python \
  "${COMPARATOR}" \
  --bout-output "${BOUT_OUTPUT}" \
  --grid "${GRID}" \
  --output "${OUTDIR}/oracle_comparison.json" \
  --arrays "${OUTDIR}/comparison_arrays.npz" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --atol 5e-10 \
  --rtol 5e-10
readonly COMPARISON_STATUS=$?
set -e

sha256sum \
  "${OUTDIR}/commands.sh" \
  "${RUN_DIR}/BOUT.inp" \
  "${BUILD_DIR}/shifted_ddy_oracle" \
  "${BOUT_OUTPUT}" \
  "${OUTDIR}/oracle_comparison.json" \
  "${OUTDIR}/comparison_arrays.npz" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 compiled shifted-DDY oracle finished at $(date --iso-8601=seconds)"
echo "Comparison exit status: ${COMPARISON_STATUS}"
echo "Result directory: ${OUTDIR}"
exit "${COMPARISON_STATUS}"
