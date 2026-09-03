#!/bin/bash
# M32 truth-free forecasts for both arms at 0.5, 1, and 2 epochs.
#SBATCH --job-name=p0_pgl_hier_fcst
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%3
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_hierarchical_forecast_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256:?set frozen manifest SHA-256}"
: "${PAPER0_PGL_HIERARCHICAL_SCREEN_JOB_ID:?set completed screen array job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/generate_pgl_hierarchical_forecast.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_hierarchical_transport"
readonly SCREEN_ROOT="${RESULT_BASE}/screen_job_${PAPER0_PGL_HIERARCHICAL_SCREEN_JOB_ID}"
readonly EVALUATION_BASE="${RESULT_BASE}/evaluation"
readonly ARMS=(CONTROL TRANSPORT)
readonly UPDATES=(107 214 428)
readonly TRAIN_TASK=$((SLURM_ARRAY_TASK_ID / 3))
readonly CHECKPOINT_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
readonly ARM="${ARMS[${TRAIN_TASK}]}"
readonly UPDATE="${UPDATES[${CHECKPOINT_INDEX}]}"
readonly UPDATE_PADDED="$(printf '%04d' "${UPDATE}")"
readonly TRAINING_RESULT="${SCREEN_ROOT}/task_${TRAIN_TASK}_${ARM}/run/result.json"
readonly CHECKPOINT="${SCREEN_ROOT}/task_${TRAIN_TASK}_${ARM}/run/training/checkpoints/fixed_update_${UPDATE_PADDED}.pt"
readonly JOB_ROOT="${EVALUATION_BASE}/generation_job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${ARM}_u${UPDATE_PADDED}"
readonly RUN_OUTPUT="${JOB_ROOT}/generation"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-pgl-hierarchical-M32"
readonly RUN_ID="p0oldpglhierfcst-j${SLURM_ARRAY_JOB_ID}-t${SLURM_ARRAY_TASK_ID}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite hierarchical forecast output" >&2
  exit 2
fi
for path in "${TRAINING_RESULT}" "${CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing fixed training artifact: ${path}" >&2; exit 2; }
done
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]] || [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is not the locked clean commit" >&2
  exit 2
fi
readonly MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly CHECKPOINT_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${MANIFEST_SHA}" != "${PAPER0_PGL_EVALUATION_MANIFEST_SHA256}" ]]; then
  echo "Evaluation manifest SHA-256 differs" >&2
  exit 2
fi
jq -e --arg arm "${ARM}" --argjson update "${UPDATE}" --arg sha "${CHECKPOINT_SHA}" '.status == "screen_training_completed" and .arm == $arm and .completed_optimizer_updates == 428 and any(.checkpoints[]; .optimizer_update == $update and .sha256 == $sha) and .held_out_85606_read == false and .new_nersc_data_read == false' "${TRAINING_RESULT}" >/dev/null

source /etc/os-release
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "arm=${ARM}"
  echo "optimizer_update=${UPDATE}"
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
  tests/test_pgl_hierarchical_evaluation.py \
  tests/test_persistent_global_local_forecast.py \
  tests/test_persistent_global_local_gates.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --arm "${ARM}"
  --optimizer-update "${UPDATE}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${MANIFEST_SHA}"
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
  --wandb-run-name "old-85604-pgl-hierarchical-${ARM,,}-u${UPDATE}-j${SLURM_ARRAY_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

jq -e --arg arm "${ARM}" --argjson update "${UPDATE}" '.status == "truth_free_forecast_completed_and_hash_closed" and .arm == $arm and .optimizer_update == $update and .start_count == 36 and .ensemble_members == 32 and .future_frames == 4 and .target_truth_read == false and .physics_diagnostics_scored == false and .held_out_85606_read == false and .new_nersc_data_read == false' "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum "${JOB_ROOT}/command.sh" "${JOB_ROOT}/environment.txt" "${JOB_ROOT}/slurm_job.txt" "${JOB_ROOT}/test_output.txt" "${RUN_OUTPUT}/artifact_sha256.txt" > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,arm,optimizer_update,forecast,timing,peak_cuda_memory_GiB}' "${RUN_OUTPUT}/result.json"
