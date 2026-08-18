#!/bin/bash
# Freeze all three completed B2 training seeds before any evaluator smoke.
# CPU-only provenance gate; no forecast truth, physics metric, or held-out run.
#SBATCH --job-name=p3_b2_freeze
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_freeze_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B2_TRAINING_JOB_ID:?submit with B2_TRAINING_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_full/job_${B2_TRAINING_JOB_ID}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_training_freeze"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly MATRIX="${OUTDIR}/training_matrix.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/freeze_b2_training_matrix.py"
readonly TRAINING_COMMIT="46e2ca07e15c7114aace18202b26a9756489a3f0"
declare -ar SEEDS=(1701 1702 1703)

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B2 training freeze: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
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
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "B2 training freeze requires Rocky Linux 9" >&2
  exit 2
fi
check_sha256 "3e5158aca0f8284227018344445ce1eb6a2b80340e2126d3d71d6c41bd5f6206" "${ENTRYPOINT}"
check_sha256 "196bc86c489231c529d51aed6b87f068db9af654ff61701b2ec93f5cca6a51d2" "${PAPER0_ROOT}/paper0/tools/evaluate_b2_checkpoint.py"

declare -a result_args=()
for task in 0 1 2; do
  seed="${SEEDS[task]}"
  output="${TRAINING_ROOT}/task_${task}/b2_ldm_h2_seed_${seed}"
  result="${output}/result.json"
  if [[ ! -f "${result}" ]]; then
    echo "Missing complete B2 training result: ${result}" >&2
    exit 2
  fi
  sha256sum -c "${output}/artifact_sha256.txt"
  result_sha="$(sha256sum "${result}" | awk '{print $1}')"
  result_args+=(--training-result "${result}")
  result_args+=(--training-result-sha256 "${result_sha}")
done

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_job_id=${B2_TRAINING_JOB_ID}"
  source /etc/os-release
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_b2_evaluation_entrypoints.py \
  tests/test_phase3_b2_evaluation_launchers.py \
  | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  "${result_args[@]}"
  --training-commit "${TRAINING_COMMIT}"
  --training-job-id "${B2_TRAINING_JOB_ID}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --output "${MATRIX}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"
"${PYTHON}" -m json.tool "${MATRIX}" >/dev/null

sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${MATRIX}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Frozen three-seed B2 training matrix completed"
echo "The bounded evaluator smoke remains required"
