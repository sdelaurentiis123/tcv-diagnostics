#!/bin/bash
# Reconstruct selected 85604 potential states and compare two radial boundaries.
# CPU-only; no learned model is trained or evaluated.
#SBATCH --job-name=p2_phi_elliptic
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_phi_elliptic_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly RAW_ROOT="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604"
readonly GRID="${RAW_ROOT}/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_elliptic"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${OUTDIR}/build"
readonly RUN_DIR="${OUTDIR}/run"
readonly ORACLE_SOURCE="${PAPER0_ROOT}/paper0/oracles/potential_elliptic"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_potential_elliptic_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_POTENTIAL_ELLIPTIC_PROTOCOL.md"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_potential_elliptic_85604_frames.py"
readonly COMPARATOR="${PAPER0_ROOT}/paper0/tools/compare_potential_elliptic_oracle.py"
readonly CANONICAL="${OUTDIR}/canonical_potential_inputs.nc"
readonly EXTRACTION_RECORD="${OUTDIR}/extraction_record.json"
readonly COMPARISON="${OUTDIR}/potential_elliptic_comparison.json"
readonly COMPARISON_ARRAYS="${OUTDIR}/comparison_arrays.npz"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"

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

echo "Phase 2 paired potential elliptic oracle started at $(date --iso-8601=seconds)"
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
if [[ "$(git -C "${BOUT_SOURCE}" rev-parse HEAD)" != "${EXPECTED_BOUT_COMMIT}" ]]; then
  echo "BOUT++ source revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${HERMES_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
  echo "Hermes tracked source is dirty" >&2
  exit 2
fi
if [[ -n "$(git -C "${BOUT_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
  echo "BOUT++ tracked source is dirty" >&2
  exit 2
fi

check_sha256 "29feca35c73d8eeac24cf1b6da0e93473d1fec33a79b014a4ea91875f742af48" "${MANIFEST}"
check_sha256 "c208b4d8a654f1a33c557630461a00918ea62d41996acd9270c454ada170fdba" "${PROTOCOL}"
check_sha256 "a65f3a43c2c3b38990a958506ebf1448ae423315817c60cb40f384e1e8bf8a65" "${EXTRACTOR}"
check_sha256 "1e9aa801e5f89cd1735f55314218d12a7f6fdfc79caa7a8b232fe2d336ced298" "${COMPARATOR}"
check_sha256 "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2" "${ORACLE_SOURCE}/BOUT.inp"
check_sha256 "49424c894a183337c7da0a74a5537493e000be2e9aac9ca86e38a10fd2ac8a83" "${ORACLE_SOURCE}/CMakeLists.txt"
check_sha256 "48c754d2d793621aad54ff7ff28ce0ceb8da4546e3764cd00ab6d2ced422403e" "${ORACLE_SOURCE}/potential_elliptic_oracle.cxx"
check_sha256 "5e484fa85b8a06ec8bf8d6f867ad3ed8715dca567e51e3b47ad22f96b5750a4c" "${PAPER0_ROOT}/paper0/tools/extract_native_85604_frames.py"
check_sha256 "c2264c7acc5765078192336f4de9a5542bbc23d174580b16d798ca5b95a82cc1" "${PAPER0_ROOT}/paper0/tools/audit_85604_phi_boundary_state.py"
check_sha256 "c78bc34b82e225a32cb0c0764d17304c1c8e21cae58f0987d8a82c9d55869f99" "${PAPER0_ROOT}/src/tcv_diagnostics/phi_boundary.py"
check_sha256 "92dc8946db969c37e79074b9865522c6698b6abe8f391581721524110724da10" "${PAPER0_ROOT}/paper0/tools/compare_hermes_radial_flow_oracle.py"
check_sha256 "2e89dc9aec65129bcd02c88ae68beb2f3a7d51b58178ee3244856ee96361b972" "${PAPER0_ROOT}/paper0/tools/compare_shifted_ddy_oracle.py"
check_sha256 "36e2e80668ccbc121c31e25595a43bc18b02d903d5122f74f11e64e8a8a52678" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "09e99af7d87f7a0b23433d79bb1f6198ffb49cfd48ee2cf6e1c252beeae5c85a" "${PAPER0_ROOT}/src/tcv_diagnostics/transport.py"
check_sha256 "f99c540e00b1f8f99ae0712a3b161e851b2b72199cff4ad609a7026b27b0ee5e" "${PAPER0_ROOT}/paper0/manifests/phase2_native_frame_oracle.json"
check_sha256 "79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc" "${PAPER0_ROOT}/paper0/results/phase2_phi_boundary_state_6891890.json"

check_sha256 "843ea048baa90b4a55ead6aa6ed546c6746e81c2718a8ade3ed977381364f65a" "${HERMES_SOURCE}/hermes-3.cxx"
check_sha256 "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a" "${HERMES_SOURCE}/src/vorticity.cxx"
check_sha256 "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded" "${HERMES_SOURCE}/include/vorticity.hxx"
check_sha256 "858752dd39c55caf7afd00fa84583b5e1c7944763f6497f6ffaa2530ba0e1fc3" "${BOUT_SOURCE}/include/bout/invert_laplace.hxx"
check_sha256 "47c2cbe674c61e510cc96835823ba8a7b55f2a9b6c2a5c1f93d730c8dba5d157" "${BOUT_SOURCE}/src/invert/laplace/invert_laplace.cxx"
check_sha256 "568faa3c9bcf17732779868f28aa219f5e8beea2ad4f98c490b233ce9552d38a" "${BOUT_SOURCE}/src/invert/laplace/impls/cyclic/cyclic_laplace.cxx"
check_sha256 "1309f7eebf5fe076663224b1a66a08b1a3ae6ce7526d0f0b595d6537db5bf296" "${BOUT_SOURCE}/src/invert/laplace/impls/cyclic/cyclic_laplace.hxx"

check_sha256 "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad" "${RAW_ROOT}/BOUT.inp"
check_sha256 "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57" "${RAW_ROOT}/BOUT.settings"
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
module list 2> "${OUTDIR}/modules.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"

cp "${ORACLE_SOURCE}/BOUT.inp" "${RUN_DIR}/BOUT.inp"
{
  echo "python ${EXTRACTOR} --manifest ${MANIFEST} --raw-root ${RAW_ROOT} --output ${CANONICAL} --record ${EXTRACTION_RECORD}"
  echo "cmake -S ${ORACLE_SOURCE} -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${BOUT_INSTALL}"
  echo "cmake --build ${BUILD_DIR} --parallel ${SLURM_CPUS_PER_TASK}"
  echo "srun --ntasks=4 ${BUILD_DIR}/potential_elliptic_oracle -d ${RUN_DIR} paper0:input_file=${CANONICAL}"
  echo "python ${COMPARATOR} --manifest ${MANIFEST} --canonical ${CANONICAL} --extraction-record ${EXTRACTION_RECORD} --bout-output ${RUN_DIR}/BOUT.dmp.[0-3].nc --grid ${GRID} --output ${COMPARISON} --arrays ${COMPARISON_ARRAYS} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --atol 5e-10 --rtol 5e-10"
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
  "${BUILD_DIR}/potential_elliptic_oracle" \
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
  --manifest "${MANIFEST}" \
  --canonical "${CANONICAL}" \
  --extraction-record "${EXTRACTION_RECORD}" \
  --bout-output "${BOUT_OUTPUTS[@]}" \
  --grid "${GRID}" \
  --output "${COMPARISON}" \
  --arrays "${COMPARISON_ARRAYS}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --atol 5e-10 \
  --rtol 5e-10
readonly COMPARISON_STATUS=$?
set -e

HASH_ARTIFACTS=(
  "${OUTDIR}/commands.sh"
  "${OUTDIR}/modules.txt"
  "${OUTDIR}/slurm_job.txt"
  "${CANONICAL}"
  "${EXTRACTION_RECORD}"
  "${RUN_DIR}/BOUT.inp"
  "${BUILD_DIR}/potential_elliptic_oracle"
  "${BOUT_OUTPUTS[@]}"
)
if [[ -f "${COMPARISON}" ]]; then
  HASH_ARTIFACTS+=("${COMPARISON}")
fi
if [[ -f "${COMPARISON_ARRAYS}" ]]; then
  HASH_ARTIFACTS+=("${COMPARISON_ARRAYS}")
fi
sha256sum "${HASH_ARTIFACTS[@]}" > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 paired potential elliptic oracle finished at $(date --iso-8601=seconds)"
echo "Comparison exit status: ${COMPARISON_STATUS}"
echo "Result directory: ${OUTDIR}"
exit "${COMPARISON_STATUS}"
