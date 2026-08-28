#!/bin/bash
# Exact one-epoch, 214-update bounded PGL variogram A-D screen.
# Submit only after every task in the production-size smoke array passes.
#SBATCH --job-name=p0_pgl_var_screen
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_variogram_screen_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256:?set evidence-manifest SHA-256}"
: "${PAPER0_PGL_VARIOGRAM_PREFLIGHT_JOB_ID:?set passed preflight job ID}"
: "${PAPER0_PGL_VARIOGRAM_PREFLIGHT_SHA256:?set preflight.json SHA-256}"
: "${PAPER0_PGL_VARIOGRAM_SMOKE_JOB_ID:?set the all-passed smoke array job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly EVIDENCE_MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly PREFLIGHT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_preflight/job_${PAPER0_PGL_VARIOGRAM_PREFLIGHT_JOB_ID}/run"
readonly PREFLIGHT="${PREFLIGHT_ROOT}/preflight.json"
readonly SMOKE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_smoke/job_${PAPER0_PGL_VARIOGRAM_SMOKE_JOB_ID}"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_pgl_variogram_arm.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_screen"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-pgl-variogram-screen"
readonly ARMS=(A B C D)
readonly ARM="${ARMS[${SLURM_ARRAY_TASK_ID}]}"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${ARM}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly RUN_ID="p0oldpglvarscreen-j${SLURM_ARRAY_JOB_ID}-a${ARM,,}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite variogram screen output" >&2
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
if [[ "$(sha256sum "${EVIDENCE_MANIFEST}" | awk '{print $1}')" != "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}" ]]; then
  echo "Evidence-manifest SHA-256 differs" >&2
  exit 2
fi
if [[ "$(sha256sum "${PREFLIGHT}" | awk '{print $1}')" != "${PAPER0_PGL_VARIOGRAM_PREFLIGHT_SHA256}" ]]; then
  echo "Preflight SHA-256 differs" >&2
  exit 2
fi
for index in 0 1 2 3; do
  smoke_arm="${ARMS[${index}]}"
  smoke_result="${SMOKE_ROOT}/task_${index}_${smoke_arm}/run/result.json"
  if [[ ! -f "${smoke_result}" ]]; then
    echo "Missing variogram smoke result for arm ${smoke_arm}" >&2
    exit 2
  fi
  jq -e --arg arm "${smoke_arm}" '.status == "smoke_passed" and .arm == $arm and .completed_optimizer_updates == 1 and .mean_frozen_bitwise == true' "${smoke_result}" >/dev/null
done

source /etc/os-release
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=old_85604_pgl_variogram_one_epoch_screen"
  echo "development_run=85604"
  echo "arm=${ARM}"
  echo "preflight_sha256=${PAPER0_PGL_VARIOGRAM_PREFLIGHT_SHA256}"
  echo "smoke_job_id=${PAPER0_PGL_VARIOGRAM_SMOKE_JOB_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
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
  tests/test_pgl_variogram.py \
  tests/test_pgl_torch_transport.py \
  tests/test_pgl_variogram_training.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --mode screen
  --arm "${ARM}"
  --artifact-root "${ARTIFACT_ROOT}"
  --evidence-manifest "${EVIDENCE_MANIFEST}"
  --evidence-manifest-sha256 "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}"
  --preflight "${PREFLIGHT}"
  --preflight-sha256 "${PAPER0_PGL_VARIOGRAM_PREFLIGHT_SHA256}"
  --smoke-root "${SMOKE_ROOT}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-pgl-variogram-screen-arm-${ARM}-j${SLURM_ARRAY_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e --arg arm "${ARM}" '.status == "screen_training_completed" and .mode == "screen" and .arm == $arm and .completed_epochs == 1 and .completed_optimizer_updates == 214 and .mean_frozen_bitwise == true and .checkpoint_selection_performed == false and .full_sampler_compute_control_executed == true and .sampler_members == 4 and .sampler_steps == 18 and .network_evaluations_per_member == 35 and .held_out_85606_read == false and .held_out_run_read == false and .new_nersc_data_read == false and .new_segment_read == false and (((($arm == "C") or ($arm == "D")) and .physics_derived_training_loss_used == true) or ((($arm == "A") or ($arm == "B")) and .physics_derived_training_loss_used == false))' "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,mode,arm,completed_optimizer_updates,mean_frozen_bitwise,checkpoint_selection_performed,peak_cuda_memory_GiB,wall_seconds,physics_derived_training_loss_used}' "${RUN_OUTPUT}/result.json"
echo "Variogram one-epoch screen completed: ${JOB_ROOT}"
