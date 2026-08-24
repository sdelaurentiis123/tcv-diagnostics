#!/bin/bash
# Sequential inference-only scoring of six frozen old-85604 Stage-1 arms.
# One job and one GPU deliberately minimize the scheduler footprint.
#SBATCH --job-name=p0_old_state1_blocks
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_stage1_blocks_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_FROZEN_MANIFEST:?submit with PAPER0_FROZEN_MANIFEST set}"
: "${PAPER0_FROZEN_MANIFEST_SHA256:?submit with PAPER0_FROZEN_MANIFEST_SHA256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MATRIX_RUNNER="${PAPER0_ROOT}/paper0/tools/run_codec_free_stage1_block_matrix.py"
readonly REDUCER="${PAPER0_ROOT}/paper0/tools/summarize_codec_free_stage1_full.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_stage1_block_evaluation"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly MATRIX_OUTPUT="${JOB_ROOT}/matrix"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing block-matrix output" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous evaluation" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Audited Python environment is unavailable" >&2
  exit 2
fi
if [[ ! -f "${PAPER0_FROZEN_MANIFEST}" ]]; then
  echo "Frozen input manifest is unavailable" >&2
  exit 2
fi
readonly ACTUAL_FROZEN_SHA="$(sha256sum "${PAPER0_FROZEN_MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_FROZEN_SHA}" != "${PAPER0_FROZEN_MANIFEST_SHA256}" ]]; then
  echo "Frozen input manifest SHA-256 differs" >&2
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
  echo "scope=post_ecrd_old_85604_stage1_block_evaluation"
  echo "frozen_manifest=${PAPER0_FROZEN_MANIFEST}"
  echo "frozen_manifest_sha256=${PAPER0_FROZEN_MANIFEST_SHA256}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda))'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_codec_free_stage1_block_evaluation.py \
  tests/test_codec_free_stage1_block_matrix.py \
  tests/test_codec_free_stage1_matrix_freeze.py \
  tests/test_codec_free_stage1_reducer.py \
  | tee "${JOB_ROOT}/test_output.txt"

MATRIX_COMMAND=(
  "${PYTHON}" -u "${MATRIX_RUNNER}"
  --artifact-root "${ARTIFACT_ROOT}"
  --frozen-manifest "${PAPER0_FROZEN_MANIFEST}"
  --frozen-manifest-sha256 "${PAPER0_FROZEN_MANIFEST_SHA256}"
  --output-dir "${MATRIX_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
)
printf '%q ' "${MATRIX_COMMAND[@]}" > "${JOB_ROOT}/command_matrix.sh"
printf '\n' >> "${JOB_ROOT}/command_matrix.sh"
"${MATRIX_COMMAND[@]}" | tee "${JOB_ROOT}/matrix_stdout.jsonl"

readonly COMPLETE_MATRIX="${MATRIX_OUTPUT}/complete_matrix.json"
readonly COMPLETE_MATRIX_SHA="$(sha256sum "${COMPLETE_MATRIX}" | awk '{print $1}')"
REDUCE_COMMAND=(
  "${PYTHON}" -u "${REDUCER}"
  --manifest "${COMPLETE_MATRIX}"
  --manifest-sha256 "${COMPLETE_MATRIX_SHA}"
  --output "${JOB_ROOT}/summary.json"
)
printf '%q ' "${REDUCE_COMMAND[@]}" > "${JOB_ROOT}/command_reduce.sh"
printf '\n' >> "${JOB_ROOT}/command_reduce.sh"
"${REDUCE_COMMAND[@]}" | tee "${JOB_ROOT}/reduce_stdout.jsonl"

"${PYTHON}" -m json.tool "${COMPLETE_MATRIX}" >/dev/null
"${PYTHON}" -m json.tool "${JOB_ROOT}/summary.json" >/dev/null
jq -c '{decision,advance_e6b_as_primary_state,comparison,decision_gates}' \
  "${JOB_ROOT}/summary.json"

sha256sum \
  "${JOB_ROOT}/command_matrix.sh" \
  "${JOB_ROOT}/command_reduce.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/matrix_stdout.jsonl" \
  "${JOB_ROOT}/reduce_stdout.jsonl" \
  "${JOB_ROOT}/summary.json" \
  "${MATRIX_OUTPUT}/commands.json" \
  "${MATRIX_OUTPUT}/complete_matrix.json" \
  "${MATRIX_OUTPUT}"/evaluations/*.json \
  "${MATRIX_OUTPUT}"/logs/*.log \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Old-85604 Stage-1 block matrix completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
