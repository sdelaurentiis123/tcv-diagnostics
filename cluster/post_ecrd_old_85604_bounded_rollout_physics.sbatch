#!/bin/bash
# CPU-only physics scoring of a completed frozen old-85604 bounded rollout.
# Reads saved forecast states; performs no inference, training, or 85606 access.
#SBATCH --job-name=p0_bound_phys
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_bounded_rollout_physics_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_BOUNDED_MANIFEST:?submit with PAPER0_BOUNDED_MANIFEST set}"
: "${PAPER0_BOUNDED_MANIFEST_SHA256:?submit with PAPER0_BOUNDED_MANIFEST_SHA256 set}"
: "${PAPER0_BOUNDED_STATE_RESULT:?submit with PAPER0_BOUNDED_STATE_RESULT set}"
: "${PAPER0_BOUNDED_STATE_RESULT_SHA256:?submit with PAPER0_BOUNDED_STATE_RESULT_SHA256 set}"
: "${PAPER0_BOUNDED_FORECAST:?submit with PAPER0_BOUNDED_FORECAST set}"
: "${PAPER0_BOUNDED_FORECAST_SHA256:?submit with PAPER0_BOUNDED_FORECAST_SHA256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/score_codec_free_bounded_rollout_physics.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout_physics"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-bounded-rollout-physics"
readonly RUN_ID="p0oldboundphys-j${SLURM_JOB_ID}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing bounded-rollout physics output" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous scoring" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Audited Python environment is unavailable" >&2
  exit 2
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}" >&2
    exit 2
  fi
}

check_sha256 "${PAPER0_BOUNDED_MANIFEST_SHA256}" "${PAPER0_BOUNDED_MANIFEST}"
check_sha256 "${PAPER0_BOUNDED_STATE_RESULT_SHA256}" "${PAPER0_BOUNDED_STATE_RESULT}"
check_sha256 "${PAPER0_BOUNDED_FORECAST_SHA256}" "${PAPER0_BOUNDED_FORECAST}"

source /etc/os-release
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=post_ecrd_old_85604_bounded_rollout_physics"
  echo "manifest=${PAPER0_BOUNDED_MANIFEST}"
  echo "manifest_sha256=${PAPER0_BOUNDED_MANIFEST_SHA256}"
  echo "state_result=${PAPER0_BOUNDED_STATE_RESULT}"
  echo "state_result_sha256=${PAPER0_BOUNDED_STATE_RESULT_SHA256}"
  echo "forecast=${PAPER0_BOUNDED_FORECAST}"
  echo "forecast_sha256=${PAPER0_BOUNDED_FORECAST_SHA256}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_bounded_rollout.py \
  tests/test_bounded_rollout_physics.py \
  tests/test_matched_codec_metrics.py \
  tests/test_phase2_codec_transport.py \
  tests/test_matched_o1_transport.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${PAPER0_BOUNDED_MANIFEST}"
  --manifest-sha256 "${PAPER0_BOUNDED_MANIFEST_SHA256}"
  --state-result "${PAPER0_BOUNDED_STATE_RESULT}"
  --state-result-sha256 "${PAPER0_BOUNDED_STATE_RESULT_SHA256}"
  --forecast "${PAPER0_BOUNDED_FORECAST}"
  --forecast-sha256 "${PAPER0_BOUNDED_FORECAST_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-bounded-rollout-physics-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/physics_metrics.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '.status == "bounded_rollout_physics_scored" and .training_performed == false and .physics_derived_loss_used == false' \
  "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' \
  "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{status,physics_metrics,example_fields,forecast_paper0_commit}' \
  "${RUN_OUTPUT}/result.json"
echo "Old-85604 bounded rollout physics scoring completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
