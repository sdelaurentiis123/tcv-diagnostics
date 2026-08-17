#!/bin/bash
# Stream the frozen 85604 evolved-state inventory and momentum closure.
# CPU-only; no model, training, diagnostics, or held-out data are in scope.
#SBATCH --job-name=p2_state
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_state_completeness_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RAW_ROOT="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_state_completeness"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_state_completeness.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_STATE_COMPLETENESS_PROTOCOL.md"
readonly AUDITOR="${PAPER0_ROOT}/paper0/tools/audit_85604_state_completeness.py"
readonly MERGER="${PAPER0_ROOT}/paper0/tools/merge_85604_state_completeness_shards.py"
readonly CORE="${PAPER0_ROOT}/src/tcv_diagnostics/state_completeness.py"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly RESULT="${OUTDIR}/state_completeness_audit.json"
readonly PARTIAL_DIR="${OUTDIR}/partials"
readonly SHARD_COUNT=16
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

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

echo "Phase 2 85604 state-completeness audit started at $(date --iso-8601=seconds)"
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

check_sha256 "8ce098ddc4f75a6c81d2d74c6e4190884c2f4eb1273436e522fff0326e86bdc5" "${MANIFEST}"
check_sha256 "d134f46ce517457299fece0e978d8ed032283529b69dd2facf3a7634cda9da22" "${PROTOCOL}"
check_sha256 "200c2fd96a788eec16e9e3c495e310f8c0af36376656fbd20ea5207f4b46030b" "${AUDITOR}"
check_sha256 "d049f33ab42e97229ca6689bf482223686278d3c84a5effaa06135446952f6b1" "${MERGER}"
check_sha256 "579197834948fbd60d17fb10feca71c35a83fe433cd310e691928fb4a8aa5aeb" "${CORE}"
check_sha256 "83d56e8335aeb3c7595f26b1ecbbed7a9ad016bb7bc4df3280a7b752ba6061ae" "${HERMES_SOURCE}/src/evolve_momentum.cxx"
check_sha256 "5e3b8055ae068481319a90329a3bdf3605b82b9024ee8273b210fd2be915bd85" "${HERMES_SOURCE}/include/hermes_utils.hxx"
check_sha256 "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad" "${RAW_ROOT}/BOUT.inp"
check_sha256 "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57" "${RAW_ROOT}/BOUT.settings"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${RAW_ROOT}/tcv_85604_adjusted.nc"

module purge
module load gcc/13.3.0
module load hdf5/1.12.3
module load netcdf-c/4.9.2

{
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    echo "srun --exclusive --exact --ntasks=1 --cpus-per-task=1 --mem=4G ${PYTHON} ${AUDITOR} --manifest ${MANIFEST} --raw-root ${RAW_ROOT} --output ${PARTIAL_DIR}/shard_${shard}.json --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --shard-index ${shard} --shard-count ${SHARD_COUNT}"
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
  "${PYTHON}" -c 'import netCDF4, numpy; print("netCDF4=" + netCDF4.__version__); print("numpy=" + numpy.__version__)'
  module list
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
    --raw-root "${RAW_ROOT}" \
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
  echo "At least one rank shard failed; refusing incomplete merge" >&2
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

echo "Phase 2 85604 state-completeness audit finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
