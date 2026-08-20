#!/bin/bash
# Fail-closed mechanical finalizer for one completed four-arm ECRD smoke array.
# Submit with --dependency=afterok:<array-job-id> and ECRD_SMOKE_ARRAY_JOB_ID set.
#SBATCH --job-name=p0_ecrd_smkfin
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_engineering_smoke_finalize_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${ECRD_SMOKE_ARRAY_JOB_ID:?submit with ECRD_SMOKE_ARRAY_JOB_ID set}"

if [[ ! "${ECRD_SMOKE_ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ECRD_SMOKE_ARRAY_JOB_ID must be one numeric Slurm array ID" >&2
  exit 2
fi

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_engineering_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${ECRD_SMOKE_ARRAY_JOB_ID}"
readonly FINALIZER_ROOT="${JOB_ROOT}/finalizer_job_${SLURM_JOB_ID}"
readonly SUMMARY_OUTPUT="${JOB_ROOT}/result.json"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/ecrd_engineering_smoke_85604.json"
readonly MANIFEST_SHA="6204e7bfef02c449f464ea7647b8d6fc14c0bba22135c0b56ea47d77f530ac45"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/summarize_ecrd_engineering_smoke.py"

if [[ ! -d "${JOB_ROOT}" || -e "${FINALIZER_ROOT}" || -e "${SUMMARY_OUTPUT}" ]]; then
  echo "ECRD smoke job root is absent or finalizer output would be overwritten" >&2
  exit 2
fi
mkdir "${FINALIZER_ROOT}"
exec > >(tee -a "${FINALIZER_ROOT}/job.log") 2>&1

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
  echo "The ECRD smoke finalizer requires Rocky Linux 9" >&2
  exit 2
fi
check_sha256 "${MANIFEST_SHA}" "${MANIFEST}"
check_sha256 "b6307b419a2cccc41fa945376ee69d6c71a16f5d73d9c198f6eb1d2015dede43" "${ENTRYPOINT}"
check_sha256 "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4" "${PAPER0_ROOT}/paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
check_sha256 "b366f067217f8fe675ddd5ab0741ecb4b1c19325c6af7cd27d97b21a3e51f0e2" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_training.py"
check_sha256 "38d1d69a930fa27923de1cebf98553f70bf797a1e8ac250a7c4aef9b4961421f" "${PAPER0_ROOT}/src/tcv_diagnostics/models/ecrd.py"
check_sha256 "b0d2211ebab290a99c77c0295535669097cf05105b56922d94d4e61a933d6432" "${PAPER0_ROOT}/src/tcv_diagnostics/models/field_residual_edm.py"

scontrol show job -dd "${SLURM_JOB_ID}" > "${FINALIZER_ROOT}/slurm_job.txt"
sacct -j "${ECRD_SMOKE_ARRAY_JOB_ID}" --format=JobID,JobName%32,Partition,State,Elapsed,ExitCode,NodeList -n -P > "${FINALIZER_ROOT}/array_sacct.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "smoke_array_job_id=${ECRD_SMOKE_ARRAY_JOB_ID}"
  echo "scope=bounded_non_scientific_ECRD_engineering_smoke_finalizer_85604"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import numpy,torch; print("numpy="+numpy.__version__); print("torch="+torch.__version__)'
} > "${FINALIZER_ROOT}/environment.txt" 2>&1

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${MANIFEST_SHA}"
  --run "B5=${JOB_ROOT}/task_0_b5_seed_1701/model"
  --run "B5-Context=${JOB_ROOT}/task_1_b5_context_seed_1701/model"
  --run "ECRD=${JOB_ROOT}/task_2_ecrd_seed_1701/model"
  --run "ECRD-History=${JOB_ROOT}/task_3_ecrd_history_seed_1701/model"
  --output "${SUMMARY_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --smoke-paper0-commit "fe4ba35ff0134729371209d89d42ac895792026b"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${FINALIZER_ROOT}/command.sh"
printf '\n' >> "${FINALIZER_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${FINALIZER_ROOT}/stdout.jsonl"
"${PYTHON}" -m json.tool "${SUMMARY_OUTPUT}" >/dev/null

export PAPER0_ECRD_SMOKE_SUMMARY="${SUMMARY_OUTPUT}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["PAPER0_ECRD_SMOKE_SUMMARY"]).read_text())
required = {
    "status": "all_four_arms_passed_mechanical_smoke",
    "scope": "bounded_non_scientific_ECRD_engineering_smoke_85604",
    "development_run": "85604",
    "all_four_arms_passed": True,
    "scientific_result": False,
    "physics_metric_evaluated": False,
    "full_training_authorized": False,
    "guard_frames_read": False,
    "held_out_85606_read": False,
    "assimilation_performed": False,
    "diagnostic_ranking_performed": False,
    "steering_performed": False,
}
for name, expected in required.items():
    if summary.get(name) != expected:
        raise RuntimeError(f"ECRD smoke summary field {name!r} differs")
if list(summary.get("arms", {})) != ["B5", "B5-Context", "ECRD", "ECRD-History"]:
    raise RuntimeError("ECRD smoke summary arm order differs")
print(json.dumps({
    "status": "all_four_mechanical_smokes_verified",
    "scientific_result": False,
    "full_training_authorized": False,
    "held_out_85606_read": False,
}, sort_keys=True))
PY

sha256sum \
  "${SUMMARY_OUTPUT}" \
  "${FINALIZER_ROOT}/array_sacct.txt" \
  "${FINALIZER_ROOT}/command.sh" \
  "${FINALIZER_ROOT}/environment.txt" \
  "${FINALIZER_ROOT}/slurm_job.txt" \
  > "${FINALIZER_ROOT}/artifact_sha256.txt"

echo "All four ECRD engineering smokes passed at ${JOB_ROOT}"
echo "This finalizer does not authorize full training or scientific evaluation."
