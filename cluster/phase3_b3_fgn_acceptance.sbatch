#!/bin/bash
# CPU-only frozen one-seed B3 acceptance reduction.
# Performs no training, inference, truth scoring, 85606 access, or downstream launch.
#SBATCH --job-name=p3_b3_gate
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b3_gate_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B3_TRAINING_JOB_ID:?submit with B3_TRAINING_JOB_ID set}"
: "${B3_EVALUATION_JOB_ID:?submit with B3_EVALUATION_JOB_ID set}"
: "${B3_COMPARATOR_JOB_ID:?submit with B3_COMPARATOR_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_full/job_${B3_TRAINING_JOB_ID}"
readonly TRAINING_OUTPUT="${TRAINING_ROOT}/b3_fgn_h1_seed_1701"
readonly EVALUATION_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_evaluation_full/job_${B3_EVALUATION_JOB_ID}"
readonly EVALUATION_OUTPUT="${EVALUATION_ROOT}/b3_fgn_h1_seed_1701"
readonly COMPARATOR_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_h1_comparator/job_${B3_COMPARATOR_JOB_ID}"
readonly EVALUATION_RESULT="${EVALUATION_OUTPUT}/result.json"
readonly COMPARATOR_RESULT="${COMPARATOR_ROOT}/comparators/comparators.json"
readonly TRAINING_WANDB="${TRAINING_OUTPUT}/wandb.json"
readonly EVALUATION_WANDB="${EVALUATION_OUTPUT}/wandb.json"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b3_full_evaluation_85604.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/finalize_b3_fgn_one_seed.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_acceptance"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B3 one-seed gate: ${OUTDIR}" >&2
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
  echo "B3 one-seed gate requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "36640742fea989cc5e94604fc6a2970cf431802ca32f176ca06302158a454737" "${ENTRYPOINT}"
check_sha256 "611229c0333feed3b42870760f0f4f9c0776e95d5ee58028a1885e4405c9c655" "${PAPER0_ROOT}/src/tcv_diagnostics/fgn_acceptance_gate.py"
check_sha256 "6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate.py"
check_sha256 "899bf42d7d709badd2631e5b94db11e6c6603635d2ce5f0cf341e747216f86d3" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate_event_eligibility.py"
check_sha256 "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2" "${EVALUATION_MANIFEST}"

sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${TRAINING_ROOT}/artifact_sha256.txt"
sha256sum -c "${EVALUATION_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${EVALUATION_ROOT}/artifact_sha256.txt"
sha256sum -c "${COMPARATOR_ROOT}/comparators/artifact_sha256.txt"
sha256sum -c "${COMPARATOR_ROOT}/artifact_sha256.txt"

readonly EVALUATION_RESULT_SHA="$(sha256sum "${EVALUATION_RESULT}" | awk '{print $1}')"
readonly COMPARATOR_RESULT_SHA="$(sha256sum "${COMPARATOR_RESULT}" | awk '{print $1}')"
readonly TRAINING_WANDB_SHA="$(sha256sum "${TRAINING_WANDB}" | awk '{print $1}')"
readonly EVALUATION_WANDB_SHA="$(sha256sum "${EVALUATION_WANDB}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_job_id=${B3_TRAINING_JOB_ID}"
  echo "evaluation_job_id=${B3_EVALUATION_JOB_ID}"
  echo "comparator_job_id=${B3_COMPARATOR_JOB_ID}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  echo "training_performed=false"
  echo "inference_performed=false"
  echo "truth_scoring_performed=false"
  echo "held_out_85606_read=false"
  "${PYTHON}" --version
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_b2_acceptance_gate.py \
  tests/test_phase3_b3_acceptance_gate.py \
  tests/test_phase3_b3_acceptance_finalizer.py \
  tests/test_phase3_b3_h1_comparator.py \
  | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --evaluation-result "${EVALUATION_RESULT}"
  --evaluation-result-sha256 "${EVALUATION_RESULT_SHA}"
  --comparator-result "${COMPARATOR_RESULT}"
  --comparator-result-sha256 "${COMPARATOR_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
  --training-wandb "${TRAINING_WANDB}"
  --training-wandb-sha256 "${TRAINING_WANDB_SHA}"
  --evaluation-wandb "${EVALUATION_WANDB}"
  --evaluation-wandb-sha256 "${EVALUATION_WANDB_SHA}"
  --gate-execution-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --output-directory "${OUTDIR}/gate"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

export PAPER0_B3_GATE_RESULT="${OUTDIR}/gate/final_gate.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_B3_GATE_RESULT"]).read_text())
if result["scope"] != "phase3_B3_FGN_H1_seed1701_scientific_gate_85604":
    raise RuntimeError("B3 gate result scope differs")
if result["held_out_85606_read"] is not False:
    raise RuntimeError("B3 gate opened 85606")
for flag in (
    "raw_forecast_changed",
    "raw_score_changed",
    "metrics_recomputed",
    "training_performed",
    "inference_performed",
    "truth_scoring_performed",
    "seed1702_1703_training_authorized",
    "O3_protocol_may_be_written",
    "O3_launch_allowed",
    "held_out_85606_access_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result[flag] is not False:
        raise RuntimeError(f"B3 gate scope flag {flag} differs")
acceptance = result["acceptance"]
if set(acceptance["families"]) != {"field", "spectral", "transport"}:
    raise RuntimeError("B3 gate scientific-family set differs")
if acceptance["event_block_policy"] != "truth_event_count_eligible_v1":
    raise RuntimeError("B3 gate event-block policy differs")
if result["seed1702_1703_replication_protocol_may_be_written"] != acceptance["passes_complete_one_seed_gate"]:
    raise RuntimeError("B3 gate replication-protocol disposition differs")
PY

sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/stdout.jsonl" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/gate/final_gate.json" \
  "${OUTDIR}/gate/artifact_sha256.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Frozen B3 seed-1701 gate-only reduction completed"
echo "No training, inference, truth scoring, O3, 85606, assimilation, or ranking occurred"
