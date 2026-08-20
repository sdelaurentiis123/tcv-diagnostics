#!/bin/bash
# Read-only checksum finalization for completed ECRD full-training array 6913340.
# This does not score forecasts or access held-out data.
#SBATCH --job-name=p0_ecrd_fullfin
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_full_training_finalize_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARRAY_JOB_ID="6913340"
readonly TRAINING_COMMIT="d822ee2147a98713f1b2ecdfd0f5a4077eded062"
readonly ARRAY_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_full_training/job_${ARRAY_JOB_ID}"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/ecrd_full_training_85604.json"
readonly MANIFEST_SHA="94100e13a20aa23249aa39f0d910f160ad7aadf60ab42b7e4975969dd3abe6e1"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/summarize_ecrd_full_training.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_full_training_finalization"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/finalization"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD training-finalization directory" >&2
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
  echo "ECRD training finalization requires Rocky Linux 9" >&2
  exit 2
fi
check_sha256 "${MANIFEST_SHA}" "${MANIFEST}"
check_sha256 "0a3d2ef6ea45c133c2907bb855ae755edf429278b7f497698c9e470095e81b8d" "${ENTRYPOINT}"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${ARRAY_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/array_sacct.txt"
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
  echo "training_commit=${TRAINING_COMMIT}"
  echo "scope=read_only_ECRD_full_training_finalization_85604"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import numpy,pytest,torch,wandb; print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("torch="+torch.__version__); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running the complete clean Rocky 9 suite before training finalization"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --array-root "${ARRAY_ROOT}"
  --array-job-id "${ARRAY_JOB_ID}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${MANIFEST_SHA}"
  --training-commit "${TRAINING_COMMIT}"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
export PAPER0_ECRD_FINAL_RESULT="${RUN_OUTPUT}/result.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_ECRD_FINAL_RESULT"]).read_text())
required = {
    "status": "all_eleven_new_training_tasks_verified",
    "training_commit": "d822ee2147a98713f1b2ecdfd0f5a4077eded062",
    "array_job_id": "6913340",
    "development_run": "85604",
    "total_ladder_runs": 12,
    "all_training_artifact_indices_verified": True,
    "all_wandb_runs_finished": True,
    "paired_training_order_verified": True,
    "paired_validation_seed_bank_verified": True,
    "scientific_result": False,
    "physics_metric_evaluated": False,
    "guard_frames_read": False,
    "held_out_85606_read": False,
    "scientific_forecast_generated": False,
    "assimilation_performed": False,
}
for name, expected in required.items():
    if result.get(name) != expected:
        raise RuntimeError(f"ECRD training finalization field {name!r} differs")
if len(result.get("new_runs", ())) != 11:
    raise RuntimeError("ECRD training finalization run count differs")
print(json.dumps({
    "status": "verified",
    "new_training_runs": 11,
    "total_ladder_runs": 12,
    "held_out_85606_read": False,
}, sort_keys=True))
PY

sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/array_sacct.txt" \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/job_artifact_sha256.txt"

echo "ECRD full-training array finalization completed"
