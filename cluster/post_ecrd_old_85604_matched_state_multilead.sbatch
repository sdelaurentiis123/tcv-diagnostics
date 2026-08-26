#!/bin/bash
# Matched C5P-versus-E6B old-85604 multi-lead pilot.
#SBATCH --job-name=p0_old_state_ml
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_state_multilead_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_MATCHED_STATE_MANIFEST_SHA256:?submit with manifest SHA-256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_matched_state_multilead.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_matched_state_multilead.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_multilead"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-matched-state-multilead"
readonly SEED=1701
readonly FAMILIES=(c5p e6b)

if [[ "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -ge "${#FAMILIES[@]}" ]]; then
  echo "Array task leaves the frozen state-view pair" >&2
  exit 2
fi
readonly FAMILY="${FAMILIES[${SLURM_ARRAY_TASK_ID}]}"
readonly JOB_ROOT="${RESULT_BASE}/array_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${FAMILY}_job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly RUN_ID="p0oldstateml-${FAMILY}-a${SLURM_ARRAY_JOB_ID}-t${SLURM_ARRAY_TASK_ID}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing matched-state output" >&2
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
if [[ ! -x "${PYTHON}" || ! -f "${MANIFEST}" || ! -f "${ENTRYPOINT}" ]]; then
  echo "Audited Python, manifest, or entrypoint is unavailable" >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_MATCHED_STATE_MANIFEST_SHA256}" ]]; then
  echo "Matched-state manifest SHA-256 differs" >&2
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
  echo "scope=post_ecrd_old_85604_matched_state_multilead_pilot"
  echo "development_run=85604"
  echo "family=${FAMILY}"
  echo "seed=${SEED}"
  echo "manifest=${MANIFEST}"
  echo "manifest_sha256=${PAPER0_MATCHED_STATE_MANIFEST_SHA256}"
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
  tests/test_matched_state_multilead.py \
  tests/test_codec_free_stage2_multilead.py \
  tests/test_codec_free_operator.py \
  tests/test_phase2_model_training_data.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${PAPER0_MATCHED_STATE_MANIFEST_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --family "${FAMILY}"
  --seed "${SEED}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-matched-state-${FAMILY}-s${SEED}-a${SLURM_ARRAY_JOB_ID}-t${SLURM_ARRAY_TASK_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '.status == "passed" and .training_gate.passed == true and .optimizer_updates == 6396 and .held_out_85606_read == false and .new_nersc_data_read == false and .physics_derived_loss_used == false' \
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

jq -c '{family,best_checkpoint,parent_selection_metric,parent_improvement_fraction,training_gate,transition_gate,paired_physics_evaluation_eligible,peak_cuda_memory_GiB,wall_seconds}' \
  "${RUN_OUTPUT}/result.json"
echo "Old-85604 matched-state ${FAMILY} arm completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
