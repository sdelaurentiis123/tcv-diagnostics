#!/bin/bash
# CPU-only final reduction of all three B2 probabilistic evaluation seeds.
# Must run after the complete evaluation array; never opens 85606.
#SBATCH --job-name=p3_b2_gate
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_gate_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B2_TRAINING_FREEZE_JOB_ID:?submit with B2_TRAINING_FREEZE_JOB_ID set}"
: "${B2_COMPARATOR_JOB_ID:?submit with B2_COMPARATOR_JOB_ID set}"
: "${B2_THRESHOLD_JOB_ID:?submit with B2_THRESHOLD_JOB_ID set}"
: "${B2_SMOKE_JOB_ID:?submit with B2_SMOKE_JOB_ID set}"
: "${B2_EVALUATION_JOB_ID:?submit with B2_EVALUATION_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_MATRIX_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_training_freeze/job_${B2_TRAINING_FREEZE_JOB_ID}"
readonly TRAINING_MATRIX="${TRAINING_MATRIX_ROOT}/training_matrix.json"
readonly COMPARATOR_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_deterministic_comparators/job_${B2_COMPARATOR_JOB_ID}"
readonly COMPARATOR_MATRIX="${COMPARATOR_ROOT}/comparators/comparators.json"
readonly THRESHOLD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_${B2_THRESHOLD_JOB_ID}"
readonly THRESHOLD_RESULT="${THRESHOLD_ROOT}/thresholds/result.json"
readonly SMOKE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_evaluator_smoke/job_${B2_SMOKE_JOB_ID}"
readonly SMOKE_RESULT="${SMOKE_ROOT}/b2_ldm_h2_seed_1701/result.json"
readonly EVALUATION_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_evaluation_full/job_${B2_EVALUATION_JOB_ID}"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_full_evaluation_85604.json"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B2_FULL_EVALUATION_PROTOCOL.md"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/finalize_b2_evaluation.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_evaluation_matrix"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
declare -ar SEEDS=(1701 1702 1703)

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B2 final evaluation matrix: ${OUTDIR}" >&2
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
  echo "B2 acceptance finalization requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "7df86c9fa4a10f98fd0f0c7315989bc83794777051d5062459ee2d1fb2668708" "${ENTRYPOINT}"
check_sha256 "6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate.py"
check_sha256 "a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c" "${EVALUATION_MANIFEST}"
check_sha256 "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490" "${EVALUATION_PROTOCOL}"

sha256sum -c "${TRAINING_MATRIX_ROOT}/artifact_sha256.txt"
sha256sum -c "${COMPARATOR_ROOT}/artifact_sha256.txt"
sha256sum -c "${COMPARATOR_ROOT}/comparators/artifact_sha256.txt"
sha256sum -c "${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt"
sha256sum -c "${SMOKE_ROOT}/artifact_sha256.txt"
readonly TRAINING_MATRIX_SHA="$(sha256sum "${TRAINING_MATRIX}" | awk '{print $1}')"
readonly COMPARATOR_MATRIX_SHA="$(sha256sum "${COMPARATOR_MATRIX}" | awk '{print $1}')"
readonly THRESHOLD_RESULT_SHA="$(sha256sum "${THRESHOLD_RESULT}" | awk '{print $1}')"
readonly SMOKE_RESULT_SHA="$(sha256sum "${SMOKE_RESULT}" | awk '{print $1}')"

declare -a evaluation_args=()
for task in 0 1 2; do
  seed="${SEEDS[task]}"
  task_root="${EVALUATION_ROOT}/task_${task}"
  run_root="${task_root}/b2_ldm_h2_seed_${seed}"
  result="${run_root}/result.json"
  if [[ ! -f "${result}" ]]; then
    echo "Missing B2 evaluation result: ${result}" >&2
    exit 2
  fi
  sha256sum -c "${task_root}/artifact_sha256.txt"
  sha256sum -c "${run_root}/artifact_sha256.txt"
  result_sha="$(sha256sum "${result}" | awk '{print $1}')"
  evaluation_args+=(--evaluation-result "${result}")
  evaluation_args+=(--evaluation-result-sha256 "${result_sha}")
done

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_freeze_job_id=${B2_TRAINING_FREEZE_JOB_ID}"
  echo "comparator_job_id=${B2_COMPARATOR_JOB_ID}"
  echo "threshold_job_id=${B2_THRESHOLD_JOB_ID}"
  echo "smoke_job_id=${B2_SMOKE_JOB_ID}"
  echo "evaluation_job_id=${B2_EVALUATION_JOB_ID}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_b2_acceptance_gate.py \
  tests/test_phase3_b2_finalizer.py \
  tests/test_phase3_b2_evaluation_launchers.py \
  | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --training-matrix "${TRAINING_MATRIX}"
  --training-matrix-sha256 "${TRAINING_MATRIX_SHA}"
  --comparator-matrix "${COMPARATOR_MATRIX}"
  --comparator-matrix-sha256 "${COMPARATOR_MATRIX_SHA}"
  --event-threshold-result "${THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "${THRESHOLD_RESULT_SHA}"
  --smoke-result "${SMOKE_RESULT}"
  --smoke-result-sha256 "${SMOKE_RESULT_SHA}"
  "${evaluation_args[@]}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c"
  --evaluation-protocol "${EVALUATION_PROTOCOL}"
  --evaluation-protocol-sha256 "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490"
  --paper0-commit "${PAPER0_COMMIT}"
  --evaluation-job-id "${B2_EVALUATION_JOB_ID}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --output-directory "${OUTDIR}/matrix"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

export PAPER0_B2_FINAL_MATRIX="${OUTDIR}/matrix/final_matrix.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_B2_FINAL_MATRIX"]).read_text())
if result["scope"] != "phase3_B2_LDM_H2_full_probabilistic_evaluation_matrix_85604":
    raise RuntimeError("B2 final matrix scope differs")
if result["held_out_85606_read"] is not False:
    raise RuntimeError("B2 final matrix opened 85606")
if result["O3_launch_allowed"] is not False:
    raise RuntimeError("B2 finalizer launched O3 without a new protocol")
if result["assimilation_allowed"] is not False:
    raise RuntimeError("B2 finalizer authorized assimilation")
if result["diagnostic_ranking_allowed"] is not False:
    raise RuntimeError("B2 finalizer authorized diagnostic ranking")
if len(result["architecture_acceptance"]["per_seed"]) != 3:
    raise RuntimeError("B2 final matrix seed count differs")
PY

sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/stdout.jsonl" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/matrix/final_matrix.json" \
  "${OUTDIR}/matrix/artifact_sha256.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "B2 frozen three-seed acceptance matrix completed"
echo "No O3, 85606, assimilation, or diagnostic ranking was launched"
