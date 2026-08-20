#!/bin/bash
# Data-free reduction of the complete four-arm, three-seed ECRD score matrix.
# Submitted with afterok on the full 85604 scientific-evaluation array.
#SBATCH --job-name=p0_ecrd_gate
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_evaluation_finalize_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${ECRD_EVALUATION_FREEZE_JOB_ID:?submit with ECRD_EVALUATION_FREEZE_JOB_ID set}"
: "${ECRD_EVALUATION_ARRAY_JOB_ID:?submit with ECRD_EVALUATION_ARRAY_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly EVALUATION_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_scientific_evaluation_full/job_${ECRD_EVALUATION_ARRAY_JOB_ID}"
readonly FREEZE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_evaluation_freeze/job_${ECRD_EVALUATION_FREEZE_JOB_ID}/freeze"
readonly EVALUATION_MANIFEST="${FREEZE_ROOT}/ecrd_scientific_evaluation_85604.json"
readonly EVENT_THRESHOLD_RESULT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_6897493/thresholds/result.json"
readonly REDUCER="${PAPER0_ROOT}/paper0/tools/summarize_ecrd_model_ladder.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_model_ladder_final"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/reduction"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD model-ladder reduction directory" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch" >&2
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
  echo "verified ${actual}  ${path}"
}

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "ECRD model-ladder reduction requires Rocky Linux 9" >&2
  exit 2
fi
check_sha256 "d9a1049cacc8e5d7657d8d5a231413a9f01b3abee66615b664723e21a853a591" "${REDUCER}"
check_sha256 "de1744e97f583557b46dcb713ace4c1efe6d84002b3f2583b58b63164c8c671d" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_acceptance.py"
sha256sum -c "${FREEZE_ROOT}/artifact_sha256.txt"
readonly EVALUATION_MANIFEST_SHA="$(sha256sum "${EVALUATION_MANIFEST}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${ECRD_EVALUATION_ARRAY_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/evaluation_sacct.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "evaluation_freeze_job_id=${ECRD_EVALUATION_FREEZE_JOB_ID}"
  echo "evaluation_array_job_id=${ECRD_EVALUATION_ARRAY_JOB_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import numpy,pytest,scipy; print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running complete clean Rocky 9 suite before the data-free ECRD gate"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

TASK_SPECS=(
  "0:B5:b5:1701"
  "1:B5:b5:1702"
  "2:B5:b5:1703"
  "3:B5-Context:b5_context:1701"
  "4:B5-Context:b5_context:1702"
  "5:B5-Context:b5_context:1703"
  "6:ECRD:ecrd:1701"
  "7:ECRD:ecrd:1702"
  "8:ECRD:ecrd:1703"
  "9:ECRD-History:ecrd_history:1701"
  "10:ECRD-History:ecrd_history:1702"
  "11:ECRD-History:ecrd_history:1703"
)
SCORE_ARGS=()
for specification in "${TASK_SPECS[@]}"; do
  IFS=: read -r task arm safe_arm seed <<< "${specification}"
  task_root="${EVALUATION_ROOT}/task_${task}_${safe_arm}_seed_${seed}/evaluation"
  sha256sum -c "${task_root}/artifact_sha256.txt"
  score_path="${task_root}/score.json"
  score_sha="$(sha256sum "${score_path}" | awk '{print $1}')"
  SCORE_ARGS+=(--score "${arm}:${seed}:${score_path}:${score_sha}")
done

COMMAND=(
  "${PYTHON}" -u "${REDUCER}"
  "${SCORE_ARGS[@]}"
  --event-threshold-result "${EVENT_THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA}"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/acceptance.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/evaluation_sacct.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/job_artifact_sha256.txt"

echo "ECRD 85604 model-ladder gate completed"
