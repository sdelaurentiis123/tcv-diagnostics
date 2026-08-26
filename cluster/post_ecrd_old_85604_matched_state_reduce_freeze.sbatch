#!/bin/bash
# Reduce the completed matched training pair and, only on a two-arm transition
# pass, freeze the bounded-generation manifest.  This job performs no training.
#SBATCH --job-name=p0_old_state_reduce
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_state_reduce_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_C5P_TRAINING_RESULT:?submit with C5P result set}"
: "${PAPER0_C5P_TRAINING_RESULT_SHA256:?submit with C5P result SHA-256 set}"
: "${PAPER0_E6B_TRAINING_RESULT:?submit with E6B result set}"
: "${PAPER0_E6B_TRAINING_RESULT_SHA256:?submit with E6B result SHA-256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_COMMIT="6bb5e8ed75540287dd917f1b130aa4102051a9d7"
readonly TRAINING_MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_matched_state_multilead.json"
readonly TRAINING_MANIFEST_SHA256="7749dcd6231d0d5857a9da8bfc05f592c6c4918a5cad60693d1eecf847465be8"
readonly REDUCER="${PAPER0_ROOT}/paper0/tools/reduce_matched_state_multilead.py"
readonly FREEZER="${PAPER0_ROOT}/paper0/tools/freeze_matched_state_bounded_generation.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_reduce_freeze"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly REDUCTION="${JOB_ROOT}/reduction.json"
readonly GENERATION_MANIFEST="${JOB_ROOT}/bounded_generation_manifest.json"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing matched reduction" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous reduction" >&2
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
check_sha256 "${TRAINING_MANIFEST_SHA256}" "${TRAINING_MANIFEST}"
check_sha256 "${PAPER0_C5P_TRAINING_RESULT_SHA256}" "${PAPER0_C5P_TRAINING_RESULT}"
check_sha256 "${PAPER0_E6B_TRAINING_RESULT_SHA256}" "${PAPER0_E6B_TRAINING_RESULT}"

source /etc/os-release
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_commit=${TRAINING_COMMIT}"
  echo "training_manifest_sha256=${TRAINING_MANIFEST_SHA256}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import numpy; print("numpy="+numpy.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=2

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_matched_state_multilead_reduction.py \
  tests/test_matched_state_bounded_freeze.py \
  tests/test_matched_state_multilead.py \
  | tee "${JOB_ROOT}/test_output.txt"

"${PYTHON}" "${REDUCER}" \
  --manifest "${TRAINING_MANIFEST}" \
  --manifest-sha256 "${TRAINING_MANIFEST_SHA256}" \
  --c5p-result "${PAPER0_C5P_TRAINING_RESULT}" \
  --c5p-result-sha256 "${PAPER0_C5P_TRAINING_RESULT_SHA256}" \
  --e6b-result "${PAPER0_E6B_TRAINING_RESULT}" \
  --e6b-result-sha256 "${PAPER0_E6B_TRAINING_RESULT_SHA256}" \
  --output "${REDUCTION}" \
  --paper0-root "${PAPER0_ROOT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --training-commit "${TRAINING_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${REDUCTION}" >/dev/null
if jq -e '.paired_physics_evaluation_authorized == true' "${REDUCTION}" >/dev/null; then
  readonly REDUCTION_SHA="$(sha256sum "${REDUCTION}" | awk '{print $1}')"
  "${PYTHON}" "${FREEZER}" \
    --paired-reduction "${REDUCTION}" \
    --paired-reduction-sha256 "${REDUCTION_SHA}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --output "${GENERATION_MANIFEST}"
  jq -e '.status == "frozen_after_paired_transition_reduction_before_inference" and .held_out_85606_access_allowed == false and .new_nersc_data_access_allowed == false and .training_allowed == false and .checkpoint_selection_allowed == false' \
    "${GENERATION_MANIFEST}" >/dev/null
  echo "Paired transition passed; bounded generation is frozen and authorized"
else
  jq -e '.decision == "stop_before_paired_physics_and_record_transition_failure" and .paired_physics_evaluation_authorized == false' \
    "${REDUCTION}" >/dev/null
  echo "Paired transition failed scientifically; no generation manifest created"
fi

find "${JOB_ROOT}" -maxdepth 1 -type f ! -name artifact_sha256.txt -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,transition_comparison,paired_physics_evaluation_authorized,decision}' \
  "${REDUCTION}"
echo "Matched state-view reduction completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
