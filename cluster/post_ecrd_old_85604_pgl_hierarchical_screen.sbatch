#!/bin/bash
# Fixed two-epoch matched CONTROL versus TRANSPORT hierarchy screen.
#SBATCH --job-name=p0_pgl_hier_train
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_hierarchical_screen_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256:?set evidence-manifest SHA-256}"
: "${PAPER0_PGL_HIERARCHICAL_PREFLIGHT_JOB_ID:?set passed preflight job ID}"
: "${PAPER0_PGL_HIERARCHICAL_PREFLIGHT_SHA256:?set preflight SHA-256}"
: "${PAPER0_PGL_HIERARCHICAL_SMOKE_JOB_ID:?set passed smoke array job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly EVIDENCE_MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly PREFLIGHT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_hierarchical_transport/preflight_job_${PAPER0_PGL_HIERARCHICAL_PREFLIGHT_JOB_ID}/run"
readonly PREFLIGHT="${PREFLIGHT_ROOT}/preflight.json"
readonly SMOKE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_hierarchical_transport/smoke_job_${PAPER0_PGL_HIERARCHICAL_SMOKE_JOB_ID}"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_pgl_hierarchical_transport.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_hierarchical_transport"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-pgl-hierarchical-screen"
readonly ARMS=(CONTROL TRANSPORT)
readonly ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
readonly JOB_ROOT="${RESULT_BASE}/screen_job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${ARM}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly RUN_ID="p0oldpglhierscreen-j${SLURM_ARRAY_JOB_ID}-a${ARM,,}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite hierarchical screen output" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]] || [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is not the locked clean commit" >&2
  exit 2
fi
if [[ "$(sha256sum "${EVIDENCE_MANIFEST}" | awk '{print $1}')" != "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}" ]] || [[ "$(sha256sum "${PREFLIGHT}" | awk '{print $1}')" != "${PAPER0_PGL_HIERARCHICAL_PREFLIGHT_SHA256}" ]]; then
  echo "Hierarchical screen input SHA-256 differs" >&2
  exit 2
fi

git -C "${PAPER0_ROOT}" status --short --branch > "${JOB_ROOT}/git_status.txt"
scontrol show job "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
env | sort > "${JOB_ROOT}/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
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
  tests/test_pgl_hierarchical_transport.py \
  tests/test_pgl_hierarchical_training.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --mode screen
  --arm "${ARM}"
  --artifact-root "${ARTIFACT_ROOT}"
  --evidence-manifest "${EVIDENCE_MANIFEST}"
  --evidence-manifest-sha256 "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}"
  --preflight "${PREFLIGHT}"
  --preflight-sha256 "${PAPER0_PGL_HIERARCHICAL_PREFLIGHT_SHA256}"
  --smoke-root "${SMOKE_ROOT}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-pgl-hierarchical-screen-${ARM,,}-j${SLURM_ARRAY_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

jq -e --arg arm "${ARM}" '.status == "screen_training_completed" and .mode == "screen" and .arm == $arm and .completed_optimizer_updates == 428 and (.checkpoints|map(.optimizer_update)) == [107,214,428] and .checkpoint_selection_performed == false and .full_sampler_compute_control_executed == true and .held_out_85606_read == false and .new_nersc_data_read == false and ((($arm == "TRANSPORT") and .physics_derived_training_loss_used == true) or (($arm == "CONTROL") and .physics_derived_training_loss_used == false))' "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum "${JOB_ROOT}/command.sh" "${JOB_ROOT}/environment.txt" "${JOB_ROOT}/slurm_job.txt" "${JOB_ROOT}/test_output.txt" "${RUN_OUTPUT}/result.json" "${RUN_OUTPUT}/wandb.json" > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,arm,completed_optimizer_updates,checkpoints,peak_cuda_memory_GiB,wall_seconds,physics_derived_training_loss_used}' "${RUN_OUTPUT}/result.json"
echo "Hierarchical two-epoch screen completed: ${JOB_ROOT}"
