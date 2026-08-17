#!/bin/bash
# All-624-frame source-matched potential/vorticity closure on 85604.
# CPU-only; raw extraction occurs once and eight 78-frame replays are serial.
#SBATCH --job-name=p2_vort_all
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_vort_all_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RAW_ROOT="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604"
readonly GRID="${RAW_ROOT}/tcv_85604_adjusted.nc"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_vorticity_all_frame"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly CANONICAL_DIR="${OUTDIR}/canonical"
readonly SHARD_BASE="${OUTDIR}/shards"
readonly BUILD_DIR="${OUTDIR}/build"
readonly ORACLE_SOURCE="${PAPER0_ROOT}/paper0/oracles/potential_vorticity_all_frame"
readonly DRIVER="${ORACLE_SOURCE}/potential_vorticity_all_frame_oracle.cxx"
readonly BOUT_INPUT="${PAPER0_ROOT}/paper0/oracles/potential_elliptic/BOUT.inp"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_potential_vorticity_all_frame_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_POTENTIAL_VORTICITY_ALL_FRAME_PROTOCOL.md"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_potential_vorticity_all_frame_85604.py"
readonly COMPARATOR="${PAPER0_ROOT}/paper0/tools/compare_potential_vorticity_all_frame_shard.py"
readonly MERGER="${PAPER0_ROOT}/paper0/tools/merge_potential_vorticity_all_frame_shards.py"
readonly EXTRACTION_RECORD="${OUTDIR}/extraction_record.json"
readonly FINAL_RESULT="${OUTDIR}/potential_vorticity_all_frame_comparison.json"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}" "${SHARD_BASE}"
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

echo "Phase 2 all-frame potential/vorticity audit started at $(date --iso-8601=seconds)"
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

check_sha256 "a8adb03a5ea6633b114fdb2442739c928f2cb846da9c4a6de43b6355900ef333" "${MANIFEST}"
check_sha256 "b52816365f78755c3433c56af6b71931805a839d9958bdd711e91c4b0b378723" "${PROTOCOL}"
check_sha256 "5840704805b04c8586b1835064a5ae34b62ad22a95257722e6a416db5e2c191a" "${EXTRACTOR}"
check_sha256 "79daf7925cb6a8b7d8751eee51f3fa9f5e6139289700654999d659a8bce6d254" "${DRIVER}"
check_sha256 "b2ac21ea37793e24417320b7fef8c143f0db6a1ff80f65a0e6663328f019169e" "${ORACLE_SOURCE}/CMakeLists.txt"
check_sha256 "ffd5efbb59abb0eef3bd9a187431f85d5a9c9700fd976627c1d9b89c8b64e967" "${COMPARATOR}"
check_sha256 "3bedce322efbf22e4942afc84101422f807b179e82c9fa739f1c6800ab4e6fa6" "${MERGER}"
check_sha256 "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2" "${BOUT_INPUT}"

check_sha256 "42b3aa3d56ff6f4dbfda9b2cf7317c1f3c0080c3a229f331ff75d58a95406871" "${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_forward_6892764.json"
check_sha256 "ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a" "${PAPER0_ROOT}/paper0/results/phase2_potential_elliptic_runtime_pressure_6892641.json"
check_sha256 "79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc" "${PAPER0_ROOT}/paper0/results/phase2_phi_boundary_state_6891890.json"
check_sha256 "9511571507bcd0d46b632843c3220646b8c1b9b53c909845262946fe6f1e4c51" "${PAPER0_ROOT}/paper0/results/phase2_pressure_closure_6891583.json"
check_sha256 "565a4e27e87d4f5a3e647daf77486020ac627f43ffb5cd30a8daf74b7199cf20" "${PAPER0_ROOT}/paper0/results/phase2_state_completeness_6891855.json"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "c78bc34b82e225a32cb0c0764d17304c1c8e21cae58f0987d8a82c9d55869f99" "${PAPER0_ROOT}/src/tcv_diagnostics/phi_boundary.py"

check_sha256 "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad" "${RAW_ROOT}/BOUT.inp"
check_sha256 "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57" "${RAW_ROOT}/BOUT.settings"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"
check_sha256 "d3abc5e32cdad3ea9c42faf432dcaed465070ee5792cb62e550b9baaad9953e6" "${HERMES_SOURCE}/src/evolve_pressure.cxx"
check_sha256 "5e3b8055ae068481319a90329a3bdf3605b82b9024ee8273b210fd2be915bd85" "${HERMES_SOURCE}/include/hermes_utils.hxx"
check_sha256 "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a" "${HERMES_SOURCE}/src/vorticity.cxx"
check_sha256 "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded" "${HERMES_SOURCE}/include/vorticity.hxx"
check_sha256 "858752dd39c55caf7afd00fa84583b5e1c7944763f6497f6ffaa2530ba0e1fc3" "${BOUT_SOURCE}/include/bout/invert_laplace.hxx"
check_sha256 "ca8167cab0cf0ab71a9c135240d383394455bfc469a33858ccd90673da85c5b1" "${BOUT_SOURCE}/include/bout/fft.hxx"
check_sha256 "c7da57b5dcbda9f0e93cae7b847d4b5f43a21426ee8c938ff2b137fbee55310e" "${BOUT_SOURCE}/src/invert/fft_fftw.cxx"
check_sha256 "47c2cbe674c61e510cc96835823ba8a7b55f2a9b6c2a5c1f93d730c8dba5d157" "${BOUT_SOURCE}/src/invert/laplace/invert_laplace.cxx"
check_sha256 "568faa3c9bcf17732779868f28aa219f5e8beea2ad4f98c490b233ce9552d38a" "${BOUT_SOURCE}/src/invert/laplace/impls/cyclic/cyclic_laplace.cxx"
check_sha256 "1309f7eebf5fe076663224b1a66a08b1a3ae6ce7526d0f0b595d6537db5bf296" "${BOUT_SOURCE}/src/invert/laplace/impls/cyclic/cyclic_laplace.hxx"
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

{
  echo "${PYTHON} ${EXTRACTOR} --manifest ${MANIFEST} --raw-root ${RAW_ROOT} --output-dir ${CANONICAL_DIR} --record ${EXTRACTION_RECORD} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID}"
  echo "cmake -S ${ORACLE_SOURCE} -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${BOUT_INSTALL}"
  echo "cmake --build ${BUILD_DIR} --parallel ${SLURM_CPUS_PER_TASK}"
  for ((shard = 0; shard < 8; shard++)); do
    start=$((shard * 78))
    echo "srun --ntasks=4 ${BUILD_DIR}/potential_vorticity_all_frame_oracle -d ${SHARD_BASE}/shard_${shard}/run paper0:input_file=${CANONICAL_DIR}/canonical_shard_${shard}.nc paper0:shard_start=${start}"
    echo "${PYTHON} ${COMPARATOR} --manifest ${MANIFEST} --protocol ${PROTOCOL} --canonical ${CANONICAL_DIR}/canonical_shard_${shard}.nc --extraction-record ${EXTRACTION_RECORD} --shard-index ${shard} --bout-output ${SHARD_BASE}/shard_${shard}/run/BOUT.dmp.[0-3].nc --grid ${GRID} --output ${SHARD_BASE}/shard_${shard}/comparison.json --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --atol 5e-10 --rtol 5e-10"
  done
  echo "${PYTHON} ${MERGER} --manifest ${MANIFEST} --protocol ${PROTOCOL} --extraction-record ${EXTRACTION_RECORD} --shard-result ${SHARD_BASE}/shard_[0-7]/comparison.json --output ${FINAL_RESULT} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --artifact-root ${OUTDIR}"
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
  "${EXTRACTOR}" \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW_ROOT}" \
  --output-dir "${CANONICAL_DIR}" \
  --record "${EXTRACTION_RECORD}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}"

cmake \
  -S "${ORACLE_SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}" \
  -DnetCDF_DIR="$(nc-config --prefix)/lib64/cmake/netCDF"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS=1
declare -a SHARD_RESULTS=()
declare -a SHARD_STATUSES=()
declare -a SHARD_HASH_RECORDS=()
for ((shard = 0; shard < 8; shard++)); do
  start=$((shard * 78))
  shard_dir="${SHARD_BASE}/shard_${shard}"
  run_dir="${shard_dir}/run"
  comparison="${shard_dir}/comparison.json"
  if [[ -e "${shard_dir}" ]]; then
    echo "Refusing to overwrite existing shard directory: ${shard_dir}" >&2
    exit 2
  fi
  mkdir -p "${run_dir}"
  cp "${BOUT_INPUT}" "${run_dir}/BOUT.inp"

  srun --ntasks=4 \
    "${BUILD_DIR}/potential_vorticity_all_frame_oracle" \
    -d "${run_dir}" \
    "paper0:input_file=${CANONICAL_DIR}/canonical_shard_${shard}.nc" \
    "paper0:shard_start=${start}"

  mapfile -t bout_outputs < <(
    find "${run_dir}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V
  )
  if [[ "${#bout_outputs[@]}" -ne 4 ]]; then
    echo "Expected four BOUT++ outputs for shard ${shard}; found ${#bout_outputs[@]}" >&2
    exit 2
  fi

  set +e
  PYTHONPATH="${PAPER0_ROOT}/src" \
    "${PYTHON}" \
    "${COMPARATOR}" \
    --manifest "${MANIFEST}" \
    --protocol "${PROTOCOL}" \
    --canonical "${CANONICAL_DIR}/canonical_shard_${shard}.nc" \
    --extraction-record "${EXTRACTION_RECORD}" \
    --shard-index "${shard}" \
    --bout-output "${bout_outputs[@]}" \
    --grid "${GRID}" \
    --output "${comparison}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}" \
    --atol 5e-10 \
    --rtol 5e-10
  comparison_status=$?
  set -e
  if [[ ! -f "${comparison}" ]]; then
    echo "Shard ${shard} comparator produced no result artifact" >&2
    exit 2
  fi
  SHARD_RESULTS+=("${comparison}")
  SHARD_STATUSES+=("${comparison_status}")
  sha256sum \
    "${run_dir}/BOUT.inp" \
    "${bout_outputs[@]}" \
    "${comparison}" \
    > "${shard_dir}/artifact_sha256.txt"
  SHARD_HASH_RECORDS+=("${shard_dir}/artifact_sha256.txt")
done

set +e
"${PYTHON}" \
  "${MERGER}" \
  --manifest "${MANIFEST}" \
  --protocol "${PROTOCOL}" \
  --extraction-record "${EXTRACTION_RECORD}" \
  --shard-result "${SHARD_RESULTS[@]}" \
  --output "${FINAL_RESULT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --artifact-root "${OUTDIR}"
readonly MERGE_STATUS=$?
set -e
if [[ ! -f "${FINAL_RESULT}" ]]; then
  echo "All-frame merger produced no result artifact" >&2
  exit 2
fi

sha256sum \
  "${OUTDIR}/commands.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/modules.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${EXTRACTION_RECORD}" \
  "${BUILD_DIR}/potential_vorticity_all_frame_oracle" \
  "${SHARD_RESULTS[@]}" \
  "${SHARD_HASH_RECORDS[@]}" \
  "${FINAL_RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 all-frame potential/vorticity audit finished at $(date --iso-8601=seconds)"
echo "Shard comparator statuses: ${SHARD_STATUSES[*]}"
echo "Merge exit status: ${MERGE_STATUS}"
echo "Result directory: ${OUTDIR}"
exit "${MERGE_STATUS}"
