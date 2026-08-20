#!/bin/bash
# Diagnostic-only framewise comparison of the bounded CPU-smoke and H100 parents.
# The H100 artifact remains the sole scientific/full-training parent.
#SBATCH --job-name=p0_ecrd_parent_cmp
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_parent_execution_comparison_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/compare_ecrd_parent_executions.py"
readonly CPU_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent_cpu_smoke/job_6912481/parent"
readonly H100_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent/job_6913234/parent"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_parent_execution_comparison"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/comparison"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD parent-comparison directory" >&2
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
  echo "ECRD parent comparison requires Rocky Linux 9" >&2
  exit 2
fi

check_sha256 "930f3b1f8e759afa0e30b0a57b51472505dd683920deb49076e6c81bca2ecb82" "${ENTRYPOINT}"
check_sha256 "d238d055c3f1da9e3096a81cac67176f90365c99dfb423a1a0629f85b61f9532" "${CPU_ROOT}/sym_h1_train.h5"
check_sha256 "9acaf190507d2cc0216a4f137e7e63717370e52484e99351d6ded903e30cc2d3" "${CPU_ROOT}/sym_h1_validation.h5"
check_sha256 "21725a5ae29832068676840aabea37a2e59eac861e8b9243bf45414de6811cbd" "${H100_ROOT}/sym_h1_train.h5"
check_sha256 "fd23a42a6603e9150868a7eb0dc5f76580f0044bf1bd0b9e8e01ef04b378fd8d" "${H100_ROOT}/sym_h1_validation.h5"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=diagnostic_only_CPU_smoke_to_H100_parent_comparison_85604"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running the complete clean Rocky 9 suite before parent comparison"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --cpu-train "${CPU_ROOT}/sym_h1_train.h5"
  --cpu-train-sha256 "d238d055c3f1da9e3096a81cac67176f90365c99dfb423a1a0629f85b61f9532"
  --cpu-validation "${CPU_ROOT}/sym_h1_validation.h5"
  --cpu-validation-sha256 "9acaf190507d2cc0216a4f137e7e63717370e52484e99351d6ded903e30cc2d3"
  --h100-train "${H100_ROOT}/sym_h1_train.h5"
  --h100-train-sha256 "21725a5ae29832068676840aabea37a2e59eac861e8b9243bf45414de6811cbd"
  --h100-validation "${H100_ROOT}/sym_h1_validation.h5"
  --h100-validation-sha256 "fd23a42a6603e9150868a7eb0dc5f76580f0044bf1bd0b9e8e01ef04b378fd8d"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
export PAPER0_ECRD_COMPARISON_RESULT="${RUN_OUTPUT}/result.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_ECRD_COMPARISON_RESULT"]).read_text())
required = {
    "status": "comparison_completed_H100_remains_only_scientific_parent",
    "development_run": "85604",
    "comparison_completed_for_CPU_smoke_amendment": True,
    "CPU_parent_promoted": False,
    "full_training_parent_authority": "scientific_H100_parent_only",
    "scientific_result": False,
    "physics_metric_evaluated": False,
    "target_truth_read": False,
    "guard_frames_read": False,
    "held_out_85606_read": False,
    "training_performed": False,
}
for name, expected in required.items():
    if result.get(name) != expected:
        raise RuntimeError(f"ECRD parent comparison field {name!r} differs")
for split in ("train", "validation"):
    guard = result["comparisons"][split]["engineering_consistency_guard"]
    if not guard["passed"] or not guard["diagnostic_only"]:
        raise RuntimeError(f"ECRD {split} execution-consistency guard failed")
    if guard["may_promote_CPU_parent"]:
        raise RuntimeError(f"ECRD {split} guard improperly promotes CPU")
print(json.dumps({
    "status": "verified",
    "H100_remains_only_scientific_parent": True,
    "held_out_85606_read": False,
}, sort_keys=True))
PY

sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/job_artifact_sha256.txt"

echo "ECRD CPU/H100 parent comparison completed; H100 remains authoritative"
