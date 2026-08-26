#!/bin/bash
# Freeze the causal C5P/E6B common-view physics comparison before scoring.
# This job reads and verifies manifests/results only; it performs no inference,
# training, checkpoint selection, or physics evaluation.
#SBATCH --job-name=p0_old_state_pfreeze
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_state_physics_freeze_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_GENERATION_MANIFEST:?submit with generation manifest set}"
: "${PAPER0_GENERATION_MANIFEST_SHA256:?submit with generation manifest SHA-256 set}"
: "${PAPER0_C5P_GENERATION_RESULT:?submit with C5P generation result set}"
: "${PAPER0_C5P_GENERATION_RESULT_SHA256:?submit with C5P generation result SHA-256 set}"
: "${PAPER0_E6B_GENERATION_RESULT:?submit with E6B generation result set}"
: "${PAPER0_E6B_GENERATION_RESULT_SHA256:?submit with E6B generation result SHA-256 set}"
: "${PAPER0_EXACT_PHI_RESULT:?submit with exact-phi result set}"
: "${PAPER0_EXACT_PHI_RESULT_SHA256:?submit with exact-phi result SHA-256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly FREEZER="${PAPER0_ROOT}/paper0/tools/freeze_matched_state_physics_scoring.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_physics_freeze"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly MANIFEST="${JOB_ROOT}/matched_state_physics_manifest.json"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing physics freeze" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous freeze" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" || ! -f "${FREEZER}" ]]; then
  echo "Audited Python or freezer is unavailable" >&2
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
check_sha256 "${PAPER0_GENERATION_MANIFEST_SHA256}" "${PAPER0_GENERATION_MANIFEST}"
check_sha256 "${PAPER0_C5P_GENERATION_RESULT_SHA256}" "${PAPER0_C5P_GENERATION_RESULT}"
check_sha256 "${PAPER0_E6B_GENERATION_RESULT_SHA256}" "${PAPER0_E6B_GENERATION_RESULT}"
check_sha256 "${PAPER0_EXACT_PHI_RESULT_SHA256}" "${PAPER0_EXACT_PHI_RESULT}"

source /etc/os-release
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "development_run=85604"
  echo "held_out_85606_read=false"
  echo "new_nersc_data_read=false"
  "${PYTHON}" --version
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=2

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_matched_state_physics_freeze.py \
  tests/test_matched_state_exact_phi_assembly.py \
  tests/test_matched_state_bounded_freeze.py \
  | tee "${JOB_ROOT}/test_output.txt"

"${PYTHON}" "${FREEZER}" \
  --generation-manifest "${PAPER0_GENERATION_MANIFEST}" \
  --generation-manifest-sha256 "${PAPER0_GENERATION_MANIFEST_SHA256}" \
  --c5p-generation-result "${PAPER0_C5P_GENERATION_RESULT}" \
  --c5p-generation-result-sha256 "${PAPER0_C5P_GENERATION_RESULT_SHA256}" \
  --e6b-generation-result "${PAPER0_E6B_GENERATION_RESULT}" \
  --e6b-generation-result-sha256 "${PAPER0_E6B_GENERATION_RESULT_SHA256}" \
  --exact-phi-result "${PAPER0_EXACT_PHI_RESULT}" \
  --exact-phi-result-sha256 "${PAPER0_EXACT_PHI_RESULT_SHA256}" \
  --paper0-root "${PAPER0_ROOT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --output "${MANIFEST}"

jq -e '.status == "frozen_after_causal_exact_phi_before_paired_physics" and .development_run == "85604" and .held_out_85606_access_allowed == false and .new_nersc_data_access_allowed == false and .training_allowed == false and .checkpoint_selection_allowed == false and .physics_derived_training_loss_allowed == false and .assimilation_allowed == false and .diagnostic_ranking_allowed == false and .steering_allowed == false' \
  "${MANIFEST}" >/dev/null

sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${MANIFEST}" \
  > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,development_run,evaluation,decision}' "${MANIFEST}"
echo "Matched state-view physics freeze completed at $(date --iso-8601=seconds)"
echo "Manifest: ${MANIFEST}"
