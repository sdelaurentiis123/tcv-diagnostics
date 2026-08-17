#!/bin/bash
# Compare selected native-81 85604 states with the executed Hermes source.
# CPU-only; the held-out run is outside this launcher's scope.
#SBATCH --job-name=p2_native_flow
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_native_flow_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly RAW_ROOT="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604"
readonly GRID="${RAW_ROOT}/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_native_frames"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${OUTDIR}/build"
readonly RUN_DIR="${OUTDIR}/run"
readonly ORACLE_SOURCE="${PAPER0_ROOT}/paper0/oracles/hermes_native_frames"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_native_frame_oracle.json"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_native_85604_frames.py"
readonly COMPARATOR="${PAPER0_ROOT}/paper0/tools/compare_hermes_native_frame_oracle.py"
readonly CANONICAL="${OUTDIR}/canonical_frames.nc"
readonly EXTRACTION_RECORD="${OUTDIR}/extraction_record.json"
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

echo "Phase 2 native-frame Hermes oracle started at $(date --iso-8601=seconds)"
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
check_sha256 "3c766083078ec17d737a7ac595868adf1706e0596a9e614bb3ac73f071c1834d" "${HERMES_SOURCE}/src/evolve_density.cxx"
check_sha256 "d3abc5e32cdad3ea9c42faf432dcaed465070ee5792cb62e550b9baaad9953e6" "${HERMES_SOURCE}/src/evolve_pressure.cxx"
check_sha256 "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad" "${RAW_ROOT}/BOUT.inp"
check_sha256 "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57" "${RAW_ROOT}/BOUT.settings"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"
check_sha256 "f99c540e00b1f8f99ae0712a3b161e851b2b72199cff4ad609a7026b27b0ee5e" "${MANIFEST}"
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
  echo "python ${EXTRACTOR} --manifest ${MANIFEST} --raw-root ${RAW_ROOT} --output ${CANONICAL} --record ${EXTRACTION_RECORD}"
  echo "cmake -S ${ORACLE_SOURCE} -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${BOUT_INSTALL}"
  echo "cmake --build ${BUILD_DIR} --parallel ${SLURM_CPUS_PER_TASK}"
  echo "srun --ntasks=4 ${BUILD_DIR}/hermes_native_frames_oracle -d ${RUN_DIR} paper0:input_file=${CANONICAL}"
  echo "python ${COMPARATOR} --canonical ${CANONICAL} --extraction-record ${EXTRACTION_RECORD} --bout-output ${RUN_DIR}/BOUT.dmp.[0-3].nc --grid ${GRID} --output ${OUTDIR}/oracle_comparison.json --arrays ${OUTDIR}/comparison_arrays.npz --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --atol 5e-10 --rtol 5e-10"
} > "${OUTDIR}/commands.sh"

/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python \
  "${EXTRACTOR}" \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --output "${CANONICAL}" \
  --record "${EXTRACTION_RECORD}"

cmake \
  -S "${ORACLE_SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}" \
  -DnetCDF_DIR="$(nc-config --prefix)/lib64/cmake/netCDF"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1
srun --ntasks=4 \
  "${BUILD_DIR}/hermes_native_frames_oracle" \
  -d "${RUN_DIR}" \
  "paper0:input_file=${CANONICAL}"

mapfile -t BOUT_OUTPUTS < <(find "${RUN_DIR}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V)
if [[ "${#BOUT_OUTPUTS[@]}" -ne 4 ]]; then
  echo "Expected four BOUT++ rank outputs; found ${#BOUT_OUTPUTS[@]}" >&2
  exit 2
fi

set +e
PYTHONPATH="${PAPER0_ROOT}/src" \
  /mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python \
  "${COMPARATOR}" \
  --canonical "${CANONICAL}" \
  --extraction-record "${EXTRACTION_RECORD}" \
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
  "${CANONICAL}"
  "${EXTRACTION_RECORD}"
  "${RUN_DIR}/BOUT.inp"
  "${BUILD_DIR}/hermes_native_frames_oracle"
  "${BOUT_OUTPUTS[@]}"
)
if [[ -f "${OUTDIR}/oracle_comparison.json" ]]; then
  HASH_ARTIFACTS+=("${OUTDIR}/oracle_comparison.json")
fi
if [[ -f "${OUTDIR}/comparison_arrays.npz" ]]; then
  HASH_ARTIFACTS+=("${OUTDIR}/comparison_arrays.npz")
fi
sha256sum "${HASH_ARTIFACTS[@]}" > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 native-frame Hermes oracle finished at $(date --iso-8601=seconds)"
echo "Comparison exit status: ${COMPARISON_STATUS}"
echo "Result directory: ${OUTDIR}"
exit "${COMPARISON_STATUS}"
