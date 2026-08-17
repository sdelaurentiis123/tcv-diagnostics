#!/bin/bash
# All-frame zero-interior-seed replay for the matched E6B elliptic operator.
# CPU-only; verifies that exact phi reconstruction does not consume truth phi.
#SBATCH --job-name=p2_e6b_replay
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_e6b_replay_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly SOURCE="${PAPER0_ROOT}/paper0/oracles/matched_e6b_elliptic"
readonly BOUT_INPUT="${PAPER0_ROOT}/paper0/oracles/potential_elliptic/BOUT.inp"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_matched_e6b_phi.py"
readonly SUMMARIZER="${PAPER0_ROOT}/paper0/tools/summarize_matched_e6b_truth_replay.py"
readonly NATIVE_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly NATIVE_DIR="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_vorticity_all_frame/job_6893033/canonical"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_truth_replay"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${OUTDIR}/build"
readonly SHARD_ROOT="${OUTDIR}/shards"
readonly FINAL_RESULT="${OUTDIR}/truth_replay_summary.json"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"
readonly NATIVE_RESULT_SHA256="cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"

canonical_hashes=(
  618a5b63c43165b174e2c6be5bf172ec14e76ab0e41752dae85f82c51178de25
  e43754b8f998026404766ac8b4a9dba6af56cd65f6a0587c65e6313af5867f57
  9d4e4f3819b8a53e4d0b78e88a78b7256441e555dd72cecbe09cf9151b1209dc
  694230f6197adf1932986296c4500f83fba1bb0f5dc25caf7075ce3868108a58
  ae747722fc7c9f06093abfe316ce323a01ac7fb213c38d9f1b9884865f3da877
  d4a05e6e78a6c2575f619b77955cea66fa34b2c9c71f779b08548eab14c37972
  3454f20940370d64d8a5d7c92360be49281975f8b32ca633180834f51ab91c8f
  d8fa85092b2010230d357cf0e9ee08d705b36b9ffce887f6c5045b115066b471
)

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}" "${SHARD_ROOT}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: ${PAPER0_COMMIT} != ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is dirty" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: ${actual} != ${expected}" >&2
    exit 2
  fi
}

source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This replay requires Rocky 9" >&2
  exit 2
fi
if [[ "$(git -C "${BOUT_SOURCE}" rev-parse HEAD)" != "${EXPECTED_BOUT_COMMIT}" ]]; then
  echo "BOUT++ source revision differs" >&2
  exit 2
fi
if [[ "$(git -C "${HERMES_SOURCE}" rev-parse HEAD)" != "${EXPECTED_HERMES_COMMIT}" ]]; then
  echo "Hermes source revision differs" >&2
  exit 2
fi

check_sha256 "7c13a60c26238acbe0db0ed6eb0e3c18f8b18f570bb7f22913642bae9bce22e5" "${SOURCE}/CMakeLists.txt"
check_sha256 "d3d1addaa421ddc005cd4ecfedfe303370034c43de6137ef648e3b116ceb8ab4" "${SOURCE}/matched_e6b_elliptic_oracle.cxx"
check_sha256 "04ad0f91c657c9ef4488873003d8bef528502d2285f45c797876d8c2cab9f004" "${EXTRACTOR}"
check_sha256 "5c6667e7db7c0af60ce25fe98c35d8500c45d40bb1fc06ad2ceae1c0f7d98915" "${SUMMARIZER}"
check_sha256 "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2" "${BOUT_INPUT}"
check_sha256 "${NATIVE_RESULT_SHA256}" "${NATIVE_RESULT}"
check_sha256 "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333" "${BOUT_INSTALL}/lib64/libbout++.so.5.2.0"
check_sha256 "b5d86b8adedf11f3fab2783cbc75fd7b1064e87d91c7df596437fd7d26dd11e6" "${BOUT_INSTALL}/bin/bout-config"
for ((shard = 0; shard < 8; shard++)); do
  check_sha256 "${canonical_hashes[shard]}" "${NATIVE_DIR}/canonical_shard_${shard}.nc"
done

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
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export OMP_NUM_THREADS=1

cmake \
  -S "${SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}" \
  -DnetCDF_DIR="$(nc-config --prefix)/lib64/cmake/netCDF"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"
readonly EXECUTABLE="${BUILD_DIR}/matched_e6b_elliptic_oracle"

declare -a RESULTS=()
declare -a STATUSES=()
for ((shard = 0; shard < 8; shard++)); do
  shard_dir="${SHARD_ROOT}/shard_${shard}"
  run_dir="${shard_dir}/run"
  input="${NATIVE_DIR}/canonical_shard_${shard}.nc"
  phi="${shard_dir}/derived_phi.h5"
  result="${shard_dir}/result.json"
  mkdir -p "${run_dir}"
  cp "${BOUT_INPUT}" "${run_dir}/BOUT.inp"
  srun --ntasks=4 \
    "${EXECUTABLE}" \
    -d "${run_dir}" \
    "paper0:input_file=${input}" \
    "paper0:truth_layout=true"
  mapfile -t outputs < <(
    find "${run_dir}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V
  )
  if [[ "${#outputs[@]}" -ne 4 ]]; then
    echo "Expected four BOUT outputs for truth shard ${shard}" >&2
    exit 2
  fi
  set +e
  "${PYTHON}" \
    "${EXTRACTOR}" \
    --input "${input}" \
    --bout-output "${outputs[@]}" \
    --output-phi "${phi}" \
    --output-result "${result}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:shard${shard}" \
    --truth-layout \
    --atol 5e-10 \
    --rtol 5e-10
  status=$?
  set -e
  if [[ ! -f "${result}" ]]; then
    echo "Truth replay shard ${shard} produced no result" >&2
    exit 2
  fi
  RESULTS+=("${result}")
  STATUSES+=("${status}")
  sha256sum "${run_dir}/BOUT.inp" "${outputs[@]}" "${phi}" "${result}" \
    > "${shard_dir}/artifact_sha256.txt"
done

"${PYTHON}" \
  "${SUMMARIZER}" \
  --result "${RESULTS[@]}" \
  --native-truth-result "${NATIVE_RESULT}" \
  --native-truth-result-sha256 "${NATIVE_RESULT_SHA256}" \
  --output "${FINAL_RESULT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}"

sha256sum \
  "${OUTDIR}/modules.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${EXECUTABLE}" \
  "${RESULTS[@]}" \
  "${FINAL_RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Matched E6B truth replay completed"
echo "Per-shard scientific statuses: ${STATUSES[*]}"
echo "Summary: ${FINAL_RESULT}"
