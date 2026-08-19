#!/bin/bash
# CPU-only retry of the pure B4 H-det/H-prob reduction over immutable job 6901015.
# Performs no training, inference, truth scoring, forecast mutation, or 85606 access.
# This exists because job 6901015 completed evaluation and then encountered the
# documented relative_L2_max -> relative_l2_max adapter-name defect.
#SBATCH --job-name=p3_b4_gate
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b4_gate_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly EVALUATION_JOB_ID="6901015"
readonly EVALUATION_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_evaluation_full/job_${EVALUATION_JOB_ID}"
readonly EVALUATION_OUTPUT="${EVALUATION_ROOT}/b4_pde_refiner_h1_seed_1701"
readonly EVALUATION_RESULT="${EVALUATION_OUTPUT}/result.json"
readonly EVALUATION_WANDB="${EVALUATION_OUTPUT}/wandb.json"
readonly TRAINING_OUTPUT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_full/job_6899682/b4_pde_refiner_h1_seed_1701"
readonly TRAINING_WANDB="${TRAINING_OUTPUT}/wandb.json"
readonly COMPARATOR_OUTPUT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_h1_comparator/job_6899063/comparators"
readonly COMPARATOR_RESULT="${COMPARATOR_OUTPUT}/comparators.json"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b4_full_evaluation_85604.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/finalize_b4_pde_refiner_one_seed.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_acceptance_retry"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B4 acceptance retry: ${OUTDIR}" >&2
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
  echo "verified ${actual}  ${path}"
}

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "B4 acceptance retry requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "0988f71aa0749044e51ded92b9ea594563232df746415dccbbc6031443ca7e92" "${EVALUATION_RESULT}"
check_sha256 "b678542a19d33abc8a212888f3834f687b784bdfb60eb491bd1dd7ebd7572b21" "${EVALUATION_WANDB}"
check_sha256 "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229" "${EVALUATION_MANIFEST}"
check_sha256 "777ecf7f5d601d68cf1dbfc4b02e1ede384ab682bda3758ea7306177aa98c439" "${ENTRYPOINT}"
check_sha256 "8725d48b712ac1b3c8e2928de05ddf545ed329035f41897b3d727ca28192e94f" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_acceptance_gate.py"
check_sha256 "6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate.py"
check_sha256 "899bf42d7d709badd2631e5b94db11e6c6603635d2ce5f0cf341e747216f86d3" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate_event_eligibility.py"

sha256sum -c "${EVALUATION_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${COMPARATOR_OUTPUT}/artifact_sha256.txt"

readonly EVALUATION_RESULT_SHA="$(sha256sum "${EVALUATION_RESULT}" | awk '{print $1}')"
readonly EVALUATION_WANDB_SHA="$(sha256sum "${EVALUATION_WANDB}" | awk '{print $1}')"
readonly TRAINING_WANDB_SHA="$(sha256sum "${TRAINING_WANDB}" | awk '{print $1}')"
readonly COMPARATOR_RESULT_SHA="$(sha256sum "${COMPARATOR_RESULT}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "immutable_evaluation_job_id=${EVALUATION_JOB_ID}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  echo "raw_forecasts_changed=false"
  echo "raw_scores_changed=false"
  echo "metrics_recomputed=false"
  echo "training_performed=false"
  echo "inference_performed=false"
  echo "truth_scoring_performed=false"
  echo "held_out_85606_read=false"
  "${PYTHON}" --version
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --evaluation-result "${EVALUATION_RESULT}"
  --evaluation-result-sha256 "${EVALUATION_RESULT_SHA}"
  --comparator-result "${COMPARATOR_RESULT}"
  --comparator-result-sha256 "${COMPARATOR_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
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

export PAPER0_B4_GATE_RESULT="${OUTDIR}/gate/final_gate.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_B4_GATE_RESULT"]).read_text())
if result["scope"] != "phase3_B4_PDE_Refiner_H1_seed1701_scientific_gate_85604":
    raise RuntimeError("B4 gate result scope differs")
if result["held_out_85606_read"] is not False:
    raise RuntimeError("B4 gate retry opened 85606")
for flag in (
    "raw_forecasts_changed",
    "raw_scores_changed",
    "metrics_recomputed",
    "training_performed",
    "inference_performed",
    "truth_scoring_performed",
    "seed1702_1703_training_authorized",
    "O3_launch_allowed",
    "held_out_85606_access_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result[flag] is not False:
        raise RuntimeError(f"B4 gate retry scope flag {flag} differs")
if result["inputs"]["evaluation"]["sha256"] != (
    "0988f71aa0749044e51ded92b9ea594563232df746415dccbbc6031443ca7e92"
):
    raise RuntimeError("B4 gate retry did not reduce immutable job 6901015")
if set(result["summary"]) != {"H_det", "H_prob", "stagewise_repair_passes"}:
    raise RuntimeError("B4 gate retry summary schema differs")
PY

cp -a "${OUTDIR}/gate/final_gate.json" "${OUTDIR}/result.json"
sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/stdout.jsonl" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/result.json" \
  "${OUTDIR}/gate/final_gate.json" \
  "${OUTDIR}/gate/artifact_sha256.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "B4 gate-only retry over immutable job 6901015 completed"
echo "No training, inference, truth scoring, forecast mutation, O3, 85606, assimilation, or ranking occurred"
