#!/bin/bash
# Causal bounded generation for the frozen C5P/E6B state-view pair.
#SBATCH --job-name=p0_old_state_gen
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_state_generation_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_MATCHED_GENERATION_MANIFEST:?submit with frozen manifest path set}"
: "${PAPER0_MATCHED_GENERATION_MANIFEST_SHA256:?submit with manifest SHA-256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/generate_matched_state_bounded_forecasts.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_bounded_generation"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-matched-state-bounded-generation"
readonly FAMILIES=(c5p e6b)

if [[ "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -ge "${#FAMILIES[@]}" ]]; then
  echo "Array task leaves the frozen state-view pair" >&2
  exit 2
fi
readonly FAMILY="${FAMILIES[${SLURM_ARRAY_TASK_ID}]}"
readonly JOB_ROOT="${RESULT_BASE}/array_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${FAMILY}_job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/generation"
readonly RUN_ID="p0oldstategen-${FAMILY}-a${SLURM_ARRAY_JOB_ID}-t${SLURM_ARRAY_TASK_ID}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing matched-state generation" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous inference" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" || ! -f "${PAPER0_MATCHED_GENERATION_MANIFEST}" || ! -f "${ENTRYPOINT}" ]]; then
  echo "Audited Python, manifest, or entrypoint is unavailable" >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${PAPER0_MATCHED_GENERATION_MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_MATCHED_GENERATION_MANIFEST_SHA256}" ]]; then
  echo "Matched-state generation manifest SHA-256 differs" >&2
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
  echo "scope=post_ecrd_old_85604_matched_state_bounded_generation"
  echo "development_run=85604"
  echo "family=${FAMILY}"
  echo "manifest=${PAPER0_MATCHED_GENERATION_MANIFEST}"
  echo "manifest_sha256=${PAPER0_MATCHED_GENERATION_MANIFEST_SHA256}"
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
  tests/test_state_view_rollout.py \
  tests/test_matched_state_bounded_generation.py \
  tests/test_matched_state_bounded_freeze.py \
  tests/test_codec_free_operator.py \
  tests/test_phase2_model_training_data.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${PAPER0_MATCHED_GENERATION_MANIFEST}"
  --manifest-sha256 "${PAPER0_MATCHED_GENERATION_MANIFEST_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --family "${FAMILY}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-matched-state-generation-${FAMILY}-a${SLURM_ARRAY_JOB_ID}-t${SLURM_ARRAY_TASK_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '.status == "completed" and .training_performed == false and .checkpoint_selection_performed == false and .target_truth_used_during_generation == false and .held_out_85606_read == false and .new_nersc_data_read == false' \
  "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' \
  "${RUN_OUTPUT}/wandb.json" >/dev/null

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{status,family,model,consumed_current_frames,unread_terminal_truth_frames,timing,peak_cuda_memory_GiB}' \
  "${RUN_OUTPUT}/result.json"
echo "Old-85604 matched-state ${FAMILY} generation completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
