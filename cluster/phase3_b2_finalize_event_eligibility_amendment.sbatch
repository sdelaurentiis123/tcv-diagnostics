#!/bin/bash
# CPU-only A016 reduction of the immutable three-seed B2 score records.
# It performs no training, inference, truth scoring, or 85606 access.
#SBATCH --job-name=p3_b2_A016
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_A016_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ORIGINAL_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_evaluation_matrix/job_6897564"
readonly ORIGINAL_MATRIX="${ORIGINAL_ROOT}/matrix/final_matrix.json"
readonly ORIGINAL_MATRIX_SHA="cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6"
readonly AMENDMENT_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_event_eligibility_amendment_85604.json"
readonly AMENDMENT_MANIFEST_SHA="a664410597ae71b5e5579e225b1d87901da188c5a3cbfa4836dba3084fb476c2"
readonly AMENDMENT_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B2_EVENT_ELIGIBILITY_AMENDMENT.md"
readonly AMENDMENT_PROTOCOL_SHA="ddfff77f48ae11117ccc8f0bd0f27043421ee700046e975c8263087e0142e2a9"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/finalize_b2_event_eligibility_amendment.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_event_eligibility_amendment"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite A016 result: ${OUTDIR}" >&2
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
  echo "A016 finalization requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "${ORIGINAL_MATRIX_SHA}" "${ORIGINAL_MATRIX}"
check_sha256 "${AMENDMENT_MANIFEST_SHA}" "${AMENDMENT_MANIFEST}"
check_sha256 "${AMENDMENT_PROTOCOL_SHA}" "${AMENDMENT_PROTOCOL}"
check_sha256 "2cf9ff6211b9bdd871ffd5155262dfb711cf69e5774160a05320cf7861dbc97b" "${ENTRYPOINT}"
check_sha256 "899bf42d7d709badd2631e5b94db11e6c6603635d2ce5f0cf341e747216f86d3" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate_event_eligibility.py"
check_sha256 "6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance_gate.py"
sha256sum -c "${ORIGINAL_ROOT}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "original_matrix=${ORIGINAL_MATRIX}"
  echo "original_matrix_sha256=${ORIGINAL_MATRIX_SHA}"
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
  tests/test_phase3_b2_event_eligibility_amendment_protocol.py \
  tests/test_phase3_b2_event_eligibility_finalizer.py \
  tests/test_phase3_b2_finalizer.py \
  | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --original-matrix "${ORIGINAL_MATRIX}"
  --original-matrix-sha256 "${ORIGINAL_MATRIX_SHA}"
  --amendment-manifest "${AMENDMENT_MANIFEST}"
  --amendment-manifest-sha256 "${AMENDMENT_MANIFEST_SHA}"
  --amendment-protocol "${AMENDMENT_PROTOCOL}"
  --amendment-protocol-sha256 "${AMENDMENT_PROTOCOL_SHA}"
  --gate-execution-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --output-directory "${OUTDIR}/matrix"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

export PAPER0_A016_MATRIX="${OUTDIR}/matrix/final_matrix.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_A016_MATRIX"]).read_text())
if result["scope"] != "phase3_B2_LDM_H2_A016_event_eligible_gate_matrix_85604":
    raise RuntimeError("A016 result scope differs")
if result["held_out_85606_read"] is not False:
    raise RuntimeError("A016 opened 85606")
if result["raw_forecasts_changed"] is not False:
    raise RuntimeError("A016 changed forecasts")
if result["raw_scores_changed"] is not False:
    raise RuntimeError("A016 changed scores")
if result["metrics_recomputed"] is not False:
    raise RuntimeError("A016 recomputed metrics")
if result["O3_launch_allowed"] is not False:
    raise RuntimeError("A016 launched O3")
if result["assimilation_allowed"] is not False:
    raise RuntimeError("A016 authorized assimilation")
if result["diagnostic_ranking_allowed"] is not False:
    raise RuntimeError("A016 authorized diagnostic ranking")
if len(result["architecture_acceptance"]["per_seed"]) != 3:
    raise RuntimeError("A016 seed count differs")
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

echo "A016 three-seed gate-only reduction completed"
echo "No training, inference, truth scoring, O3, 85606, assimilation, or ranking occurred"
