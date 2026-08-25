#!/bin/bash
# Seed-1702/1703 confirmation of old-85604 C5P multi-lead fine-tuning.
#SBATCH --job-name=p0_old_ml_scale
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=1-2%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_multilead_scale_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_SCALING_MANIFEST:?submit with PAPER0_SCALING_MANIFEST set}"
: "${PAPER0_SCALING_MANIFEST_SHA256:?submit with PAPER0_SCALING_MANIFEST_SHA256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_codec_free_stage2_multilead.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_stage2_multilead_scaling"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-stage2-multilead-scaling"
readonly SEEDS=(1702 1703)
readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
readonly SEED="${SEEDS[$((TASK_INDEX - 1))]}"
readonly ARRAY_ROOT="${RESULT_BASE}/array_${SLURM_ARRAY_JOB_ID}"
readonly JOB_ROOT="${ARRAY_ROOT}/task_${TASK_INDEX}_seed_${SEED}_job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly RUN_ID="p0oldmultileadscale-s${SEED}-j${SLURM_JOB_ID}"

if [[ "${TASK_INDEX}" -lt 1 || "${TASK_INDEX}" -gt 2 ]]; then
  echo "Array task leaves frozen seed set" >&2
  exit 2
fi
if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing multi-lead-scaling output" >&2
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
if [[ ! -f "${PAPER0_SCALING_MANIFEST}" ]]; then
  echo "Multi-lead-scaling manifest is unavailable" >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${PAPER0_SCALING_MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_SCALING_MANIFEST_SHA256}" ]]; then
  echo "Multi-lead-scaling manifest SHA-256 differs" >&2
  exit 2
fi
if [[ "$(jq -r '.paper0_commit_at_freeze' "${PAPER0_SCALING_MANIFEST}")" != "${PAPER0_COMMIT}" ]]; then
  echo "Multi-lead-scaling manifest was frozen at another commit" >&2
  exit 2
fi
if [[ "$(jq -r --arg seed "${SEED}" '.parents[$seed].seed' "${PAPER0_SCALING_MANIFEST}")" != "${SEED}" ]]; then
  echo "Array seed is absent from frozen scaling manifest" >&2
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
  echo "scope=post_ecrd_old_85604_stage2_multilead_scaling"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${TASK_INDEX}"
  echo "family=c5p"
  echo "seed=${SEED}"
  echo "manifest=${PAPER0_SCALING_MANIFEST}"
  echo "manifest_sha256=${PAPER0_SCALING_MANIFEST_SHA256}"
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
  tests/test_codec_free_stage2_multilead.py \
  tests/test_codec_free_stage2_multilead_freeze.py \
  tests/test_codec_free_stage2_scaling_freeze.py \
  tests/test_codec_free_stage2_scaling_launcher.py \
  tests/test_codec_free_operator.py \
  tests/test_phase2_model_training_data.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${PAPER0_SCALING_MANIFEST}"
  --manifest-sha256 "${PAPER0_SCALING_MANIFEST_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --seed "${SEED}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-multilead-scale-c5p-s${SEED}-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '.status == "passed" and .training_gate.passed == true' \
  "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' \
  "${RUN_OUTPUT}/wandb.json" >/dev/null

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/parent_multilead_evaluation.json" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{seed,best_checkpoint,parent_selection_metric,parent_improvement_fraction,lead1_shared_mse,longer_lead_improved_count,screen_gates,seed_confirmation_passed}' \
  "${RUN_OUTPUT}/result.json"
echo "Old-85604 C5P multi-lead seed confirmation completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
