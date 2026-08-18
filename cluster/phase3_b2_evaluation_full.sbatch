#!/bin/bash
# Truth-separated M32 B2 evaluation for three frozen full-training seeds.
# One Rocky 9 H100/H200 per seed; no 85606, O3, assimilation, or ranking.
#SBATCH --job-name=p3_b2_eval
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_eval_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B2_TRAINING_JOB_ID:?submit with B2_TRAINING_JOB_ID set}"
: "${B2_THRESHOLD_JOB_ID:?submit with B2_THRESHOLD_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b2_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_evaluation_full"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_ARRAY_JOB_ID}"
readonly TASK_ROOT="${JOB_ROOT}/task_${SLURM_ARRAY_TASK_ID}"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_full/job_${B2_TRAINING_JOB_ID}"
readonly THRESHOLD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_${B2_THRESHOLD_JOB_ID}"
readonly THRESHOLD_RESULT="${THRESHOLD_ROOT}/thresholds/result.json"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_full_evaluation_85604.json"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B2_FULL_EVALUATION_PROTOCOL.md"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_b2_checkpoint.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b2_evaluation_wandb.py"
readonly TRAINING_COMMIT="46e2ca07e15c7114aace18202b26a9756489a3f0"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b2-ldm-h2-probabilistic-evaluation-85604"

declare -ar SEEDS=(1701 1702 1703)
readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
if [[ ! "${TASK_INDEX}" =~ ^[0-2]$ ]]; then
  echo "Array task ${TASK_INDEX} is outside the frozen 0-2 matrix" >&2
  exit 2
fi
readonly SEED="${SEEDS[TASK_INDEX]}"
readonly TRAINING_OUTPUT="${TRAINING_ROOT}/task_${TASK_INDEX}/b2_ldm_h2_seed_${SEED}"
readonly TRAINING_RESULT="${TRAINING_OUTPUT}/result.json"
readonly RUN_OUTPUT="${TASK_ROOT}/b2_ldm_h2_seed_${SEED}"

if [[ -e "${TASK_ROOT}" ]]; then
  echo "Refusing to overwrite existing evaluation task: ${TASK_ROOT}" >&2
  exit 2
fi
if [[ -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to reuse node-local evaluation data: ${NODE_LOCAL_ROOT}" >&2
  exit 2
fi
mkdir -p "${TASK_ROOT}"
exec > >(tee -a "${TASK_ROOT}/task.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: ${PAPER0_COMMIT} != ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is dirty" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: ${actual} != ${expected}" >&2
    exit 2
  fi
  echo "verified ${actual}  ${path}"
}

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "Full B2 evaluation requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated H100 or H200 is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100|H200"; then
  echo "The allocated accelerator is neither H100 nor H200" >&2
  exit 2
fi

check_sha256 "a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c" "${EVALUATION_MANIFEST}"
check_sha256 "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490" "${EVALUATION_PROTOCOL}"
check_sha256 "9f37f15d8e5d6c26554cc9afd015a614f1f6e3f56f39fb26cbbcd54fe8769981" "${EVALUATOR}"
check_sha256 "8f88bd4304fcc863cf774b75d4e1eccabb6b8afe18889398cdf3c4c455e502c4" "${WANDB_WRAPPER}"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "a1af83a4a92a020bf1f44ef2c40367cd7544eb3dfc9ff31b7a95991e09574d06" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_forecast.py"
check_sha256 "0beb9404f1818e34490475d3fd28dab6eb21b53d7996fa0a46dde245be0819ad" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_scoring.py"
check_sha256 "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_metrics.py"
check_sha256 "9bc110067a7124de015202e7eac40030fccda655322f8386dc1d1adca20b9112" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_scoring.py"
check_sha256 "382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_spectral_metrics.py"
check_sha256 "b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_transport_metrics.py"
check_sha256 "edef6fbbe7b40348fa450c7428d796f4b5ebc3d9b2070e135c7bb3f58a2b6650" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_probabilistic_metrics.py"
check_sha256 "1eba2a8e0c54928264c2c84d5fda08ef6080cf4d73017d40193e3c23e2d2854a" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_o1_transport.py"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "f3b815fe3141f61a91bec06be3ede42b67ab876b4a6efd90f1e63fddb8117ede" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_context_data.py"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

if [[ ! -f "${TRAINING_RESULT}" ]]; then
  echo "Frozen B2 training result is unavailable: ${TRAINING_RESULT}" >&2
  exit 2
fi
if [[ ! -f "${THRESHOLD_RESULT}" ]]; then
  echo "Frozen B2 event thresholds are unavailable: ${THRESHOLD_RESULT}" >&2
  exit 2
fi
sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly THRESHOLD_RESULT_SHA="$(sha256sum "${THRESHOLD_RESULT}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${TASK_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_commit=${TRAINING_COMMIT}"
  echo "training_job_id=${B2_TRAINING_JOB_ID}"
  echo "threshold_job_id=${B2_THRESHOLD_JOB_ID}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${TASK_INDEX}"
  echo "seed=${SEED}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import azula,h5py,numpy,scipy,torch,wandb; import importlib.metadata; print("azula="+importlib.metadata.version("azula")); print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${TASK_ROOT}/environment.txt" 2>&1

echo "Running targeted B2 evaluator tests on Rocky 9"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_b2_evaluation_entrypoints.py \
  tests/test_phase3_b2_evaluation_wandb.py \
  tests/test_phase3_b2_forecast.py \
  tests/test_phase3_b2_scoring.py \
  tests/test_phase3_b2_field_scoring.py \
  tests/test_phase3_b2_spectral_metrics.py \
  tests/test_phase3_b2_transport_metrics.py \
  | tee "${TASK_ROOT}/test_output.txt"

echo "Staging the immutable 85604 model dataset"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_STAGED}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_STAGED}/normalization.json"
verified_shards=0
while read -r expected source_path; do
  case "${source_path}" in
    "${MODEL_DATA_SOURCE}"/shards/*.h5)
      relative="${source_path#${MODEL_DATA_SOURCE}/}"
      check_sha256 "${expected}" "${MODEL_DATA_STAGED}/${relative}"
      verified_shards=$((verified_shards + 1))
      ;;
  esac
done < "${MODEL_DATA_SOURCE}/artifact_sha256.txt"
if [[ "${verified_shards}" -ne 8 ]]; then
  echo "Expected eight verified dataset shards, got ${verified_shards}" >&2
  exit 2
fi

readonly EVALUATION_MANIFEST_SHA="a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c"
readonly EVALUATION_PROTOCOL_SHA="8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490"
readonly EVALUATOR_SHA="9f37f15d8e5d6c26554cc9afd015a614f1f6e3f56f39fb26cbbcd54fe8769981"
readonly RUN_ID="p0b2eval-${SLURM_ARRAY_JOB_ID}-${TASK_INDEX}-s${SEED}"

COMMAND=(
  "${PYTHON}" -u "${WANDB_WRAPPER}"
  --evaluator "${EVALUATOR}"
  --evaluator-sha256 "${EVALUATOR_SHA}"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --seed "${SEED}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "b2-eval-s${SEED}-j${SLURM_ARRAY_JOB_ID}-t${TASK_INDEX}"
  --
  --seed "${SEED}"
  --training-result "${TRAINING_RESULT}"
  --training-result-sha256 "${TRAINING_RESULT_SHA}"
  --training-commit "${TRAINING_COMMIT}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --event-threshold-result "${THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "${THRESHOLD_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA}"
  --evaluation-protocol "${EVALUATION_PROTOCOL}"
  --evaluation-protocol-sha256 "${EVALUATION_PROTOCOL_SHA}"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}"
  --member-batch-size 4
)
printf '%q ' "${COMMAND[@]}" > "${TASK_ROOT}/command.sh"
printf '\n' >> "${TASK_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${TASK_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null

export PAPER0_B2_EVAL_OUTPUT="${RUN_OUTPUT}"
export PAPER0_B2_EVAL_TASK_ROOT="${TASK_ROOT}"
export PAPER0_B2_EVAL_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B2_EVAL_ARRAY_JOB="${SLURM_ARRAY_JOB_ID}"
export PAPER0_B2_EVAL_TASK="${TASK_INDEX}"
export PAPER0_B2_EVAL_SEED="${SEED}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

output = Path(os.environ["PAPER0_B2_EVAL_OUTPUT"])
task = Path(os.environ["PAPER0_B2_EVAL_TASK_ROOT"])
result = json.loads((output / "result.json").read_text())
score = json.loads((output / "score.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())
seed = int(os.environ["PAPER0_B2_EVAL_SEED"])
if result["status"] != "completed_pending_frozen_acceptance_gate":
    raise RuntimeError("B2 evaluation status differs")
if result["seed"] != seed or result["held_out_85606_read"] is not False:
    raise RuntimeError("B2 evaluation identity or scope differs")
if score["scope"] != "B2_truth_separated_probabilistic_scoring_85604":
    raise RuntimeError("B2 scientific score scope differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("B2 evaluation lacks a finished online W&B run")
if result["forecast"]["bytes"] < 14_000_000_000:
    raise RuntimeError("B2 canonical M32 forecast artifact is unexpectedly small")

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

summary = {
    "schema_version": 1,
    "scope": "phase3_B2_LDM_H2_probabilistic_evaluation_task",
    "status": "completed_pending_frozen_acceptance_gate",
    "paper0_commit": os.environ["PAPER0_B2_EVAL_COMMIT"],
    "array_job_id": os.environ["PAPER0_B2_EVAL_ARRAY_JOB"],
    "array_task_id": int(os.environ["PAPER0_B2_EVAL_TASK"]),
    "seed": seed,
    "development_run": "85604",
    "held_out_85606_read": False,
    "target_frames": [498, 624],
    "ensemble_members": 32,
    "result": {"path": str(output / "result.json"), "sha256": sha(output / "result.json")},
    "score": result["score"],
    "forecast": result["forecast"],
    "wandb_url": tracking["run_url"],
    "probabilistic_scientific_gate_evaluated": False,
    "O3_launch_allowed": False,
    "assimilation_allowed": False,
    "diagnostic_ranking_allowed": False,
}
temporary = task / ".task_summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(task / "task_summary.json")
PY

sha256sum \
  "${TASK_ROOT}/command.sh" \
  "${TASK_ROOT}/environment.txt" \
  "${TASK_ROOT}/slurm_job.txt" \
  "${TASK_ROOT}/task_summary.json" \
  "${TASK_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${TASK_ROOT}/artifact_sha256.txt"

echo "Completed B2 M32 evaluation seed ${SEED}"
echo "The frozen cross-seed acceptance gate has not yet been evaluated"
