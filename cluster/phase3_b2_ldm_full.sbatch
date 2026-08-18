#!/bin/bash
# Three frozen B2-LDM-H2 seeds on independent Rocky 9 H100/H200 workers.
# Training remains data-only; the separately committed evaluator decides B2.
#SBATCH --job-name=p3_b2_full
#SBATCH --partition=gpuxl
#SBATCH --qos=gen
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_full_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b2_full_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_full"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_ARRAY_JOB_ID}"
readonly TASK_ROOT="${JOB_ROOT}/task_${SLURM_ARRAY_TASK_ID}"
readonly FULL_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_full_evaluation_85604.json"
readonly FULL_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B2_FULL_EVALUATION_PROTOCOL.md"
readonly IMPLEMENTATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_ldm_85604.json"
readonly SMOKE_RESULT="${PAPER0_ROOT}/paper0/results/phase3_b2_ldm_gpu_smoke_6896402.json"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_b2_ldm_full.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b2-ldm-h2-full-85604"

declare -ar SEEDS=(1701 1702 1703)
declare -ar CODEC_PATHS=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_2_c5p_seed_1702/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_4_c5p_seed_1703/selected.pt"
)
declare -ar CODEC_SHAS=(
  "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
  "c4f180171510489bf022ea003201494ecee991cc3b2f200e97854ec4ffd86836"
  "3d029572a24f69058889de43c42cefa29633e7167db441db4f41fa1e0c1b3f5d"
)

readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID}"
if [[ ! "${TASK_INDEX}" =~ ^[0-2]$ ]]; then
  echo "Array task ${TASK_INDEX} is outside the frozen 0-2 matrix" >&2
  exit 2
fi
readonly SEED="${SEEDS[TASK_INDEX]}"
readonly CODEC_PATH="${CODEC_PATHS[TASK_INDEX]}"
readonly CODEC_SHA="${CODEC_SHAS[TASK_INDEX]}"
readonly RUN_OUTPUT="${TASK_ROOT}/b2_ldm_h2_seed_${SEED}"

if [[ -e "${TASK_ROOT}" ]]; then
  echo "Refusing to overwrite existing task directory: ${TASK_ROOT}" >&2
  exit 2
fi
if [[ -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to reuse existing node-local directory: ${NODE_LOCAL_ROOT}" >&2
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
  echo "Full B2 training requires Rocky Linux 9" >&2
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

check_sha256 "a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c" "${FULL_MANIFEST}"
check_sha256 "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490" "${FULL_PROTOCOL}"
check_sha256 "75d6f812fc7d203d5c92c588515f4a59d4b3b0cb8688fdd7514b6d09b842f9c3" "${IMPLEMENTATION_MANIFEST}"
check_sha256 "fa2b29665b4b39b60c9ce24c1e8b067ebc6165322d40bb8de169bf9492ae5360" "${SMOKE_RESULT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "9bfa4c86287f533e6baa488be3f0ac6727a318a2e5a83f5c0b1be11f9dfe3bbe" "${PAPER0_ROOT}/src/tcv_diagnostics/models/modulated_vit.py"
check_sha256 "447427a5dd3273689b52d012ebe73fbeeaac15d143a4fda00a06ee065b1e1d4f" "${PAPER0_ROOT}/src/tcv_diagnostics/models/latent_diffusion.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "84cdba2fa2926abe4a60f4be397844941e504e893e10f7cda38692ab4e28fdc7" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_training.py"
check_sha256 "ae5dccabcd475ce118710595a1224f14c26935c931e000388e2ae7e259b6b8dd" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_wandb_tracking.py"
check_sha256 "7e13c8acc68d4516db15e3d02b14eab3911f4a0e9809f6724214b4df3f43fe1e" "${ENTRYPOINT}"
check_sha256 "4492111f3089d871ad77ffc06729b16ebb8822f6b737cd6422e5dc18086b2751" "${PAPER0_ROOT}/pyproject.toml"
check_sha256 "${CODEC_SHA}" "${CODEC_PATH}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${TASK_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
  echo "array_task_id=${TASK_INDEX}"
  echo "seed=${SEED}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import azula,h5py,numpy,pytest,scipy,torch,wandb; import importlib.metadata; print("azula="+importlib.metadata.version("azula")); print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${TASK_ROOT}/environment.txt" 2>&1

export PAPER0_WANDB_ENTITY="${WANDB_ENTITY}"
"${PYTHON}" - <<'PY' | tee "${TASK_ROOT}/wandb_preflight.json"
import json
import os
import wandb

api = wandb.Api(timeout=30)
viewer = api.viewer
entity = str(getattr(viewer, "entity", ""))
if not api.api_key:
    raise RuntimeError("W&B API key is absent")
if entity != os.environ["PAPER0_WANDB_ENTITY"]:
    raise RuntimeError(f"W&B entity {entity!r} differs")
print(json.dumps({
    "authenticated": True,
    "entity": entity,
    "username": str(getattr(viewer, "username", "")),
    "wandb_version": wandb.__version__,
}, sort_keys=True))
PY

echo "Running the complete Rocky 9 suite before B2 seed ${SEED}"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${TASK_ROOT}/test_output.txt"

echo "Staging verified 85604 model data to ${MODEL_DATA_STAGED}"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_STAGED}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_STAGED}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_STAGED}/artifact_sha256.txt"
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

readonly RUN_ID="p0b2full-${SLURM_ARRAY_JOB_ID}-${TASK_INDEX}-s${SEED}"
"${PYTHON}" -u "${ENTRYPOINT}" \
  --mode full \
  --seed "${SEED}" \
  --artifact-root "${MODEL_DATA_STAGED}" \
  --codec-checkpoint "${CODEC_PATH}" \
  --codec-sha256 "${CODEC_SHA}" \
  --output "${RUN_OUTPUT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_ARRAY_JOB_ID}_${TASK_INDEX}" \
  --full-manifest "${FULL_MANIFEST}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-id "${RUN_ID}" \
  --wandb-run-name "b2-ldm-h2-full-s${SEED}-j${SLURM_ARRAY_JOB_ID}-t${TASK_INDEX}"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum \
  "${RUN_OUTPUT}/config.json" \
  "${RUN_OUTPUT}/latent_normalization.json" \
  "${RUN_OUTPUT}/history.jsonl" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/selected.pt" \
  "${RUN_OUTPUT}/final_training_state.pt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

export PAPER0_B2_TASK_ROOT="${TASK_ROOT}"
export PAPER0_B2_RUN_OUTPUT="${RUN_OUTPUT}"
export PAPER0_B2_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B2_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
export PAPER0_B2_ARRAY_TASK_ID="${TASK_INDEX}"
export PAPER0_B2_SEED="${SEED}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

task_root = Path(os.environ["PAPER0_B2_TASK_ROOT"])
output = Path(os.environ["PAPER0_B2_RUN_OUTPUT"])
result = json.loads((output / "result.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())
normalization = json.loads((output / "latent_normalization.json").read_text())
seed = int(os.environ["PAPER0_B2_SEED"])
if result["scope"] != "B2_LDM_H2_full_training_85604":
    raise RuntimeError("full B2 task scope differs")
if result["config"]["seed"] != seed or result["config"]["arm"] != "B2-LDM-H2":
    raise RuntimeError("full B2 task identity differs")
if result["completed_epochs"] != 200 or result["completed_optimizer_steps"] != 5400:
    raise RuntimeError("full B2 task did not complete the frozen budget")
if result["config"]["train_targets"] != [2, 432]:
    raise RuntimeError("full B2 training interval differs")
if result["config"]["validation_targets"] != [498, 624]:
    raise RuntimeError("full B2 validation interval differs")
if normalization["fit_frames"] != [0, 432] or normalization["scientific_authority"] is not True:
    raise RuntimeError("full B2 latent normalization differs")
if result["checkpoint_reload_bitwise_exact"] is not True:
    raise RuntimeError("full B2 selected-checkpoint reload identity failed")
for flag in (
    "held_out_85606_read",
    "physics_derived_loss_used",
    "target_truth_used_as_model_input",
    "scientific_result",
    "probabilistic_scientific_gate_evaluated",
    "O3_launch_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result[flag] is not False:
        raise RuntimeError(f"full B2 task scope flag {flag} differs")
if result["full_B2_training_authorized"] is not True:
    raise RuntimeError("full B2 task lacks its exact authorization")
if result["training_complete_is_scientific_acceptance"] is not False:
    raise RuntimeError("full B2 training was mislabeled as scientific acceptance")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("full B2 task lacks a finished online W&B run")
if tracking["epochs_logged"] != 200:
    raise RuntimeError("full B2 task did not log all epochs")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "scope": "phase3_B2_LDM_H2_full_training_task",
    "paper0_commit": os.environ["PAPER0_B2_COMMIT"],
    "array_job_id": os.environ["PAPER0_B2_ARRAY_JOB_ID"],
    "array_task_id": int(os.environ["PAPER0_B2_ARRAY_TASK_ID"]),
    "seed": seed,
    "development_run": "85604",
    "held_out_85606_read": False,
    "completed_epochs": result["completed_epochs"],
    "completed_optimizer_steps": result["completed_optimizer_steps"],
    "selected_epoch": result["selected_epoch"],
    "selected_validation": result["selected_validation"],
    "selected_checkpoint": result["selected_checkpoint"],
    "final_training_state": result["final_training_state"],
    "parameter_count": result["parameter_count"],
    "peak_cuda_bytes": result["peak_cuda_bytes"],
    "wall_seconds": result["wall_seconds"],
    "wandb_url": tracking["run_url"],
    "artifact_index": {
        "path": str(output / "artifact_sha256.txt"),
        "sha256": sha256(output / "artifact_sha256.txt"),
    },
    "training_complete": True,
    "training_summary_is_scientific_acceptance": False,
    "probabilistic_scientific_gate_evaluated": False,
    "O3_launch_allowed": False,
}
temporary = task_root / ".task_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(task_root / "task_summary.json")
PY

sha256sum \
  "${TASK_ROOT}/environment.txt" \
  "${TASK_ROOT}/slurm_job.txt" \
  "${TASK_ROOT}/task_summary.json" \
  "${TASK_ROOT}/test_output.txt" \
  "${TASK_ROOT}/wandb_preflight.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${TASK_ROOT}/artifact_sha256.txt"

echo "Completed full B2 seed ${SEED} in array task ${TASK_INDEX}"
echo "B2 remains scientifically undecided until the frozen probabilistic evaluator runs"
