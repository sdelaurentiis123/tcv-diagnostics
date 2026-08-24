#!/bin/bash
# Bounded C5P/E6B codec-free operator smoke on one available Rusty GPU.
# Non-scientific: two optimizer steps per state view and one validation pair.
#SBATCH --job-name=p0_stateop_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/stateop_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_state_data_scaling_85604.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/smoke_codec_free_operator.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_state_operator_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-codec-free-state-operator"
readonly RUN_ID="p0stateop-${SLURM_JOB_ID}-s1701"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing state-operator smoke output" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: ${PAPER0_COMMIT} != ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous smoke" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Audited Python environment is unavailable: ${PYTHON}" >&2
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
  echo "scope=bounded_non_scientific_85604_codec_free_operator_smoke"
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

echo "Running the focused codec-free operator test set"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_codec_free_operator.py \
  tests/test_phase2_model_training_data.py \
  tests/test_ecrd_model.py \
  | tee "${JOB_ROOT}/test_output.txt"

readonly MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${MANIFEST_SHA}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --seed 1701
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "stateop-smoke-c5p-vs-e6b-s1701-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
export PAPER0_STATEOP_RESULT="${RUN_OUTPUT}/result.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["PAPER0_STATEOP_RESULT"]).read_text())
if result.get("status") != "passed":
    raise RuntimeError("codec-free operator smoke did not pass")
if result.get("optimizer_steps") != 4:
    raise RuntimeError("codec-free operator smoke step count differs")
if result.get("held_out_85606_read") is not False:
    raise RuntimeError("codec-free operator smoke held-out scope differs")
for family in ("c5p", "e6b"):
    if result["families"][family]["all_mechanical_gates_passed"] is not True:
        raise RuntimeError(f"{family} mechanical smoke gates failed")
print(json.dumps({
    "status": "passed",
    "scientific_result": False,
    "full_training_authorized": False,
    "held_out_85606_read": False,
}, sort_keys=True))
PY

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Codec-free operator smoke completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
