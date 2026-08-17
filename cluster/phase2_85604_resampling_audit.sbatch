#!/bin/bash
# Evaluate native-81 round trips and direct-88 transport on all 85604 frames.
# CPU-only, development-shot-only, and prospectively frozen before execution.
#SBATCH --job-name=p2_resample
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=17
#SBATCH --mem=68G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_resampling_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_resampling"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly PARTIAL_DIR="${OUTDIR}/partials"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_resampling_sensitivity.json"
readonly AUDITOR="${PAPER0_ROOT}/paper0/tools/audit_85604_resampling.py"
readonly MERGER="${PAPER0_ROOT}/paper0/tools/merge_85604_resampling_shards.py"
readonly RESAMPLING="${PAPER0_ROOT}/src/tcv_diagnostics/resampling.py"
readonly TRANSPORT="${PAPER0_ROOT}/src/tcv_diagnostics/transport.py"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly LEGACY_CONVERTER="/mnt/home/sdelaurentiis/tcv-gaot-3d/src/data/build_c5_fast.py"
readonly NATIVE_TRAIN="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/train/TCV_85604_train.hdf5"
readonly NATIVE_VALID="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/valid/TCV_85604_valid.hdf5"
readonly LEGACY_TRAIN="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5"
readonly LEGACY_VALID="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5"
readonly RAW_SELECTED="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_native_frames/job_6891379/canonical_frames.nc"
readonly GRID="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly RESULT="${OUTDIR}/resampling_sensitivity.json"
readonly SHARD_COUNT=17

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}" "${PARTIAL_DIR}"
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

echo "Phase 2 native/resampled audit started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

check_sha256 "875164259879621c3273a043b7fc529e8bef4875cf51b8c4d23764b95b00ce91" "${MANIFEST}"
check_sha256 "8b36c516ad3002268261c169cc95b07a5f1e349386ea522c4e05b6066824f229" "${AUDITOR}"
check_sha256 "f56ac1b366cc8a278032068cdc718ed604c9b79f01e920c3cb6d1b46e35a4c63" "${MERGER}"
check_sha256 "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e" "${RESAMPLING}"
check_sha256 "e61f007bb6268fdcd754bc975e1b0cb04133d471d86eda9d3bab01927fe8401e" "${TRANSPORT}"
check_sha256 "f446c0d6f068de0c4d190c77292467efa7312731343239082cf7a5920aa595a3" "${LEGACY_CONVERTER}"
check_sha256 "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777" "${NATIVE_TRAIN}"
check_sha256 "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba" "${NATIVE_VALID}"
check_sha256 "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817" "${LEGACY_TRAIN}"
check_sha256 "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43" "${LEGACY_VALID}"
check_sha256 "61dee30a92bb1a3eefcece80faa42d143271bfe200f4024b9747aeb06747bc21" "${RAW_SELECTED}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"

{
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    echo "srun --exclusive --exact --ntasks=1 --cpus-per-task=1 --mem=4G ${PYTHON} ${AUDITOR} --manifest ${MANIFEST} --output ${PARTIAL_DIR}/shard_${shard}.json --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --shard-index ${shard} --shard-count ${SHARD_COUNT}"
  done
  printf '%s' "${PYTHON} ${MERGER} --manifest ${MANIFEST} --partial-output"
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    printf ' %s' "${PARTIAL_DIR}/shard_${shard}.json"
  done
  printf ' --output %s --paper0-commit %s --slurm-job-id %s --artifact-root %s\n' \
    "${RESULT}" "${PAPER0_COMMIT}" "${SLURM_JOB_ID}" "${OUTDIR}"
} > "${OUTDIR}/commands.sh"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py, netCDF4, numpy, scipy; print("h5py=" + h5py.__version__); print("netCDF4=" + netCDF4.__version__); print("numpy=" + numpy.__version__); print("scipy=" + scipy.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

declare -a SHARD_PIDS=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  srun \
    --exclusive \
    --exact \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=4G \
    "${PYTHON}" \
    "${AUDITOR}" \
    --manifest "${MANIFEST}" \
    --output "${PARTIAL_DIR}/shard_${shard}.json" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}" \
    --shard-index "${shard}" \
    --shard-count "${SHARD_COUNT}" &
  SHARD_PIDS+=("$!")
done

SHARD_STATUS=0
for pid in "${SHARD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    SHARD_STATUS=1
  fi
done
if [[ "${SHARD_STATUS}" -ne 0 ]]; then
  echo "At least one frame shard failed; refusing incomplete merge" >&2
  exit 2
fi

mapfile -t PARTIAL_OUTPUTS < <(
  find "${PARTIAL_DIR}" -maxdepth 1 -name 'shard_*.json' -type f | sort -V
)
if [[ "${#PARTIAL_OUTPUTS[@]}" -ne "${SHARD_COUNT}" ]]; then
  echo "Expected ${SHARD_COUNT} shard outputs; found ${#PARTIAL_OUTPUTS[@]}" >&2
  exit 2
fi
for partial in "${PARTIAL_OUTPUTS[@]}"; do
  "${PYTHON}" -m json.tool "${partial}" >/dev/null
done

"${PYTHON}" \
  "${MERGER}" \
  --manifest "${MANIFEST}" \
  --partial-output "${PARTIAL_OUTPUTS[@]}" \
  --output "${RESULT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --artifact-root "${OUTDIR}"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
sha256sum \
  "${OUTDIR}/commands.sh" \
  "${OUTDIR}/environment.txt" \
  "${PARTIAL_OUTPUTS[@]}" \
  "${RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 native/resampled audit finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
