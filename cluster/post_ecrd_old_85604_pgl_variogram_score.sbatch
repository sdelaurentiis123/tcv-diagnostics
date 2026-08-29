#!/bin/bash
# CPU truth-separated physics scoring for four closed PGL variogram forecasts.
#SBATCH --job-name=p0_pgl_var_score
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_variogram_score_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256:?set frozen manifest SHA-256}"
: "${PAPER0_PGL_VARIOGRAM_SCREEN_JOB_ID:?set completed screen job ID}"
: "${PAPER0_PGL_VARIOGRAM_GENERATION_JOB_ID:?set completed generation job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/score_pgl_variogram_physics.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_evaluation"
readonly SCREEN_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_screen/job_${PAPER0_PGL_VARIOGRAM_SCREEN_JOB_ID}"
readonly GENERATION_ROOT="${RESULT_BASE}/generation_job_${PAPER0_PGL_VARIOGRAM_GENERATION_JOB_ID}"
readonly ARMS=(A B C D)
readonly ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
readonly TRAINING_RESULT="${SCREEN_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${ARM}/run/result.json"
readonly CHECKPOINT="${SCREEN_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${ARM}/run/training/selected.pt"
readonly GENERATION_RESULT="${GENERATION_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${ARM}/generation/result.json"
readonly FORECAST="${GENERATION_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${ARM}/generation/forecast_M32_four_frame.h5"
readonly JOB_ROOT="${RESULT_BASE}/score_job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${ARM}"
readonly RUN_OUTPUT="${JOB_ROOT}/scoring"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-pgl-variogram-M32"
readonly RUN_ID="p0oldpglvarscore-j${SLURM_ARRAY_JOB_ID}-a${ARM,,}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite variogram score output" >&2
  exit 2
fi
for path in "${TRAINING_RESULT}" "${CHECKPOINT}" "${GENERATION_RESULT}" "${FORECAST}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required closed variogram artifact is unavailable: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is dirty" >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly CHECKPOINT_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
readonly GENERATION_RESULT_SHA="$(sha256sum "${GENERATION_RESULT}" | awk '{print $1}')"
readonly FORECAST_SHA="$(sha256sum "${FORECAST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256}" ]]; then
  echo "Evaluation manifest SHA-256 differs" >&2
  exit 2
fi
jq -e --arg arm "${ARM}" --arg forecast "${FORECAST_SHA}" '.status == "truth_free_forecast_completed_and_hash_closed" and .arm == $arm and .forecast.sha256 == $forecast' "${GENERATION_RESULT}" >/dev/null

source /etc/os-release
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${PAPER0_PGL_VARIOGRAM_GENERATION_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/generation_sacct.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=old_85604_pgl_variogram_truth_separated_physics_scoring"
  echo "arm=${ARM}"
  echo "forecast_sha256=${FORECAST_SHA}"
  "${PYTHON}" --version
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
  tests/test_pgl_variogram_evaluation.py \
  tests/test_persistent_global_local_sparse_metrics.py \
  tests/test_persistent_global_local_gates.py \
  tests/test_ecrd_scoring.py \
  tests/test_phase3_b2_transport_metrics.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --arm "${ARM}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${ACTUAL_MANIFEST_SHA}"
  --training-result "${TRAINING_RESULT}"
  --training-result-sha256 "${TRAINING_RESULT_SHA}"
  --checkpoint "${CHECKPOINT}"
  --checkpoint-sha256 "${CHECKPOINT_SHA}"
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
  --wandb-run-name "old-85604-pgl-variogram-physics-arm-${ARM}-j${SLURM_ARRAY_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/score.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e --arg arm "${ARM}" '(.status == "completed_passed" or .status == "completed_failed") and .arm == $arm and .target_truth_used_during_generation == false and .held_out_85606_read == false and .new_nersc_data_read == false' "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum "${JOB_ROOT}/command.sh" "${JOB_ROOT}/environment.txt" "${JOB_ROOT}/generation_sacct.txt" "${JOB_ROOT}/slurm_job.txt" "${JOB_ROOT}/test_output.txt" "${RUN_OUTPUT}/artifact_sha256.txt" > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,arm,gate:{passed:.gate.passed,family_pass:.gate.family_pass},confirmation_seeds_authorized}' "${RUN_OUTPUT}/result.json"

