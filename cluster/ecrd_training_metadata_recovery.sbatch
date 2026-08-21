#!/bin/bash
# Metadata-only recovery for completed ECRD full-training task 6913340_8.
# No training data, GPU, forecast generation, or scientific scoring is used.
#SBATCH --job-name=p0_ecrd_recover
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_training_metadata_recovery_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly SOURCE_ARRAY_JOB_ID="6913340"
readonly SOURCE_TASK_ID="8"
readonly SOURCE_MODEL_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_full_training/job_6913340/task_8_ecrd_history_seed_1701/model"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/recover_ecrd_training_metadata.py"
readonly ENTRYPOINT_SHA="ef1eb0ca88af330d6fce12b1187258c2b80f1e10e63bf51e758c08540c2f1fd1"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_training_metadata_recovery"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/recovery"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD metadata-recovery directory" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

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
  echo "verified ${actual}  ${path}"
}

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "ECRD metadata recovery requires Rocky Linux 9" >&2
  exit 2
fi
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  echo "Metadata recovery was incorrectly allocated a GPU" >&2
  exit 2
fi
check_sha256 "${ENTRYPOINT_SHA}" "${ENTRYPOINT}"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${SOURCE_ARRAY_JOB_ID}_${SOURCE_TASK_ID}" \
  --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P \
  > "${JOB_ROOT}/source_task_sacct.txt"
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
  echo "source_slurm_job_id=${SOURCE_ARRAY_JOB_ID}_${SOURCE_TASK_ID}"
  echo "scope=metadata_only_recovery_for_completed_ECRD_training_85604"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import numpy,pytest,torch,wandb; print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("torch="+torch.__version__); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

"${PYTHON}" - <<'PY' > "${JOB_ROOT}/wandb_preflight.json"
import json
import wandb

api = wandb.Api(timeout=30)
run = api.run(
    "sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/"
    "p0ecrdfull-6913340-8-s1701"
)
record = {
    "api_key_present": bool(api.api_key),
    "entity": str(api.viewer.entity),
    "username": str(api.viewer.username),
    "run_id": str(run.id),
    "run_state": str(run.state),
}
required = {
    "api_key_present": True,
    "entity": "sdelaurentiis123-columbia-university",
    "run_id": "p0ecrdfull-6913340-8-s1701",
    "run_state": "finished",
}
for name, expected in required.items():
    if record.get(name) != expected:
        raise RuntimeError(f"W&B recovery preflight field {name!r} differs")
print(json.dumps(record, indent=2, sort_keys=True))
PY

echo "Running the complete clean Rocky 9 suite before metadata recovery"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --model-root "${SOURCE_MODEL_ROOT}"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${SOURCE_MODEL_ROOT}/wandb.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
sha256sum -c "${SOURCE_MODEL_ROOT}/artifact_sha256.txt"
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/source_task_sacct.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${SOURCE_MODEL_ROOT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/job_artifact_sha256.txt"

echo "ECRD terminal metadata recovery completed; no training was rerun"
