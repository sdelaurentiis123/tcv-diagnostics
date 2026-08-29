#!/bin/bash
# M32 truth-free four-frame forecasts for the four fixed-final variogram arms.
#SBATCH --job-name=p0_pgl_var_fcst
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_variogram_forecast_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256:?set frozen manifest SHA-256}"
: "${PAPER0_PGL_VARIOGRAM_SCREEN_JOB_ID:?set completed screen array job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/generate_pgl_variogram_forecast.py"
readonly SCREEN_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_screen/job_${PAPER0_PGL_VARIOGRAM_SCREEN_JOB_ID}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_evaluation"
readonly ARMS=(A B C D)
readonly ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
readonly TRAINING_RESULT="${SCREEN_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${ARM}/run/result.json"
readonly CHECKPOINT="${SCREEN_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${ARM}/run/training/selected.pt"
readonly JOB_ROOT="${RESULT_BASE}/generation_job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${ARM}"
readonly RUN_OUTPUT="${JOB_ROOT}/generation"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-pgl-variogram-M32"
readonly RUN_ID="p0oldpglvarfcst-j${SLURM_ARRAY_JOB_ID}-a${ARM,,}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite variogram forecast output" >&2
  exit 2
fi
if [[ ! -f "${TRAINING_RESULT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Completed variogram training artifacts are unavailable" >&2
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
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256}" ]]; then
  echo "Evaluation manifest SHA-256 differs" >&2
  exit 2
fi
jq -e --arg arm "${ARM}" '.status == "screen_training_completed" and .mode == "screen" and .arm == $arm and .completed_optimizer_updates == 214 and .mean_frozen_bitwise == true and .checkpoint_selection_performed == false' "${TRAINING_RESULT}" >/dev/null
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly CHECKPOINT_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "$(jq -r '.selected_checkpoint.sha256' "${TRAINING_RESULT}")" != "${CHECKPOINT_SHA}" ]]; then
  echo "Variogram checkpoint hash differs from training result" >&2
  exit 2
fi

source /etc/os-release
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=old_85604_pgl_variogram_truth_free_M32_generation"
  echo "arm=${ARM}"
  echo "training_result_sha256=${TRAINING_RESULT_SHA}"
  echo "checkpoint_sha256=${CHECKPOINT_SHA}"
  "${PYTHON}" --version
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_pgl_variogram_evaluation.py \
  tests/test_persistent_global_local_forecast.py \
  tests/test_persistent_global_local_gates.py \
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
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-pgl-variogram-forecast-arm-${ARM}-j${SLURM_ARRAY_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e --arg arm "${ARM}" '.status == "truth_free_forecast_completed_and_hash_closed" and .arm == $arm and .start_count == 36 and .ensemble_members == 32 and .future_frames == 4 and .target_truth_read == false and .physics_diagnostics_scored == false and .held_out_85606_read == false and .new_nersc_data_read == false' "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum "${JOB_ROOT}/command.sh" "${JOB_ROOT}/environment.txt" "${JOB_ROOT}/slurm_job.txt" "${JOB_ROOT}/test_output.txt" "${RUN_OUTPUT}/artifact_sha256.txt" > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,arm,forecast,timing,peak_cuda_memory_GiB}' "${RUN_OUTPUT}/result.json"

