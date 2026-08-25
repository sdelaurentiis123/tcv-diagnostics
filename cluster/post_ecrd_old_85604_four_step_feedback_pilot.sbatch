#!/bin/bash
# One-seed, four-step predicted-feedback fine-tuning on old 85604 only.
#SBATCH --job-name=p0_old_push4
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_four_step_feedback_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_FEEDBACK_MANIFEST:?submit with PAPER0_FEEDBACK_MANIFEST set}"
: "${PAPER0_FEEDBACK_MANIFEST_SHA256:?submit with PAPER0_FEEDBACK_MANIFEST_SHA256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_codec_free_four_step_feedback.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_four_step_feedback_pilot"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-four-step-feedback-pilot"
readonly SEED=1702
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}_seed${SEED}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly RUN_ID="p0oldpush4-j${SLURM_JOB_ID}-s${SEED}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing four-step-feedback output" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous training" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Audited Python environment is unavailable" >&2
  exit 2
fi
if [[ ! -f "${PAPER0_FEEDBACK_MANIFEST}" ]]; then
  echo "Four-step-feedback manifest is unavailable" >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${PAPER0_FEEDBACK_MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_FEEDBACK_MANIFEST_SHA256}" ]]; then
  echo "Four-step-feedback manifest SHA-256 differs" >&2
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
  echo "scope=post_ecrd_old_85604_four_step_feedback_pilot"
  echo "family=c5p"
  echo "seed=${SEED}"
  echo "manifest=${PAPER0_FEEDBACK_MANIFEST}"
  echo "manifest_sha256=${PAPER0_FEEDBACK_MANIFEST_SHA256}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_four_step_feedback_pilot.py \
  tests/test_codec_free_operator.py \
  tests/test_codec_free_stage2_multilead.py \
  tests/test_phase2_model_training_data.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${PAPER0_FEEDBACK_MANIFEST}"
  --manifest-sha256 "${PAPER0_FEEDBACK_MANIFEST_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --seed "${SEED}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-four-step-feedback-s${SEED}-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '.status == "completed" and .training_performed == true and .physics_derived_loss_used == false' \
  "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' \
  "${RUN_OUTPUT}/wandb.json" >/dev/null

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/parent_state_evaluation.json" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{status,best_checkpoint,mechanical_gate,state_gate,state_pilot_passed,physics_evaluation_authorized,peak_cuda_memory_GiB,wall_seconds}' \
  "${RUN_OUTPUT}/result.json"
echo "Old-85604 four-step feedback pilot completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
