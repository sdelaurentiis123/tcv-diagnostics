#!/bin/bash
# CPU truth scoring of one already closed and SHA-256-verified forecast.
# Submit with afterok on the matching generation job.
#SBATCH --job-name=p0_old_pgl_score
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_score_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256:?submit with the frozen manifest SHA-256}"
: "${PGL_GENERATION_JOB_ID:?submit with the successful generation job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/score_persistent_global_local_physics.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_persistent_global_local_physics_evaluation"
readonly GENERATION_ROOT="${RESULT_BASE}/job_${PGL_GENERATION_JOB_ID}/generation"
readonly GENERATION_RESULT="${GENERATION_ROOT}/result.json"
readonly FORECAST="${GENERATION_ROOT}/forecast_M32_four_frame.h5"
readonly JOB_ROOT="${RESULT_BASE}/score_job_${SLURM_JOB_ID}_generation_${PGL_GENERATION_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/scoring"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-persistent-global-local-physics"
readonly RUN_ID="p0oldpglscore-j${SLURM_JOB_ID}-g${PGL_GENERATION_JOB_ID}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite persistent physics score" >&2
  exit 2
fi
if [[ ! -f "${GENERATION_RESULT}" || ! -f "${FORECAST}" ]]; then
  echo "Closed generation artifacts are unavailable" >&2
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
  echo "Paper 0 checkout is dirty" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256}" ]]; then
  echo "Persistent evaluation manifest SHA-256 differs" >&2
  exit 2
fi
readonly GENERATION_RESULT_SHA="$(sha256sum "${GENERATION_RESULT}" | awk '{print $1}')"
readonly FORECAST_SHA="$(sha256sum "${FORECAST}" | awk '{print $1}')"

source /etc/os-release
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${PGL_GENERATION_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/generation_sacct.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=truth_separated_persistent_global_local_physics_scoring"
  echo "development_run=85604"
  echo "generation_job_id=${PGL_GENERATION_JOB_ID}"
  echo "generation_result_sha256=${GENERATION_RESULT_SHA}"
  echo "forecast_sha256=${FORECAST_SHA}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,scipy,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_persistent_global_local_sparse_metrics.py \
  tests/test_persistent_global_local_gates.py \
  tests/test_persistent_global_local_forecast.py \
  tests/test_ecrd_scoring.py \
  tests/test_phase3_b2_transport_metrics.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${ACTUAL_MANIFEST_SHA}"
  --generation-result "${GENERATION_RESULT}"
  --generation-result-sha256 "${GENERATION_RESULT_SHA}"
  --forecast "${FORECAST}"
  --forecast-sha256 "${FORECAST_SHA}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-persistent-global-local-physics-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/score.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '(.status == "completed_passed" or .status == "completed_failed") and .target_truth_used_during_generation == false and .physics_derived_training_loss_used == false and .held_out_85606_read == false and .new_nersc_data_read == false' "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum "${JOB_ROOT}/command.sh" "${JOB_ROOT}/environment.txt" "${JOB_ROOT}/generation_sacct.txt" "${JOB_ROOT}/slurm_job.txt" "${JOB_ROOT}/test_output.txt" "${RUN_OUTPUT}/artifact_sha256.txt" > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{status,gate:{passed:.gate.passed,family_pass:.gate.family_pass},confirmation_seeds_authorized}' "${RUN_OUTPUT}/result.json"
echo "Persistent physics score completed: ${JOB_ROOT}"
