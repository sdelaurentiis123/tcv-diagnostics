#!/bin/bash
# Frozen seed-1701 full B4-PDE-Refiner-H1 training on 85604 only.
# Training completion is not H-det or H-prob acceptance.
#SBATCH --job-name=p3_b4_full
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b4_full_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b4_full_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_full"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/b4_pde_refiner_h1_seed_1701"
readonly FULL_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b4_full_evaluation_85604.json"
readonly FULL_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md"
readonly IMPLEMENTATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b4_pde_refiner_85604.json"
readonly IMPLEMENTATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B4_PDE_REFINER_PROTOCOL.md"
readonly B4_SMOKE_RESULT="${PAPER0_ROOT}/paper0/results/phase3_b4_pde_refiner_gpu_smoke_6899469.json"
readonly PARENT_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
readonly PARENT_SHA="5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
readonly C5P_CODEC="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly C5P_CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
readonly LATENT_NORMALIZATION="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/latent_normalization.json"
readonly LATENT_NORMALIZATION_SHA="afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_b4_pde_refiner_full.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b4-pde-refiner-h1-full"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing result directory: ${JOB_ROOT}" >&2
  exit 2
fi
if [[ -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to reuse existing node-local directory: ${NODE_LOCAL_ROOT}" >&2
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
  echo "Full B4 training requires Rocky Linux 9" >&2
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

check_sha256 "ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1" "${FULL_PROTOCOL}"
check_sha256 "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229" "${FULL_MANIFEST}"
check_sha256 "29b71b2209df816c0847ea1b9170283f94a579fab6e9637ee0d33ccd55cd701f" "${IMPLEMENTATION_PROTOCOL}"
check_sha256 "29848b5ec080cbd1dbcb1932594842d493e18d3b0b87d97429476a2b1526f7c5" "${IMPLEMENTATION_MANIFEST}"
check_sha256 "fd2b5465f612eb8da4943f6284e317145eff64b25346895137981ce3e3993eef" "${B4_SMOKE_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb" "${PAPER0_ROOT}/src/tcv_diagnostics/models/o2.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "d1f0c4be211b548f84bc1a5b5003699419938b679418aa25f159c83b35c8a0f8" "${PAPER0_ROOT}/src/tcv_diagnostics/models/pde_refiner.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "3dd6e02259280538e24a2f0eedf8a309cac55dc6dffae64dd38d4ab34b58dcc1" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_training.py"
check_sha256 "962cd0b11088f26c3f08b18e1dad7ebc66339c1340635bcb0d4df9ec7aca429b" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_full_training.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "78e57680c9b55b34a695e4f497d2190453b34c600ded05f738092491e6664236" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_full_wandb_tracking.py"
check_sha256 "6b6a0cb7936ed4c4f5197354288a1da078951cc75e6921a7325b3b805f78d026" "${ENTRYPOINT}"
check_sha256 "4492111f3089d871ad77ffc06729b16ebb8822f6b737cd6422e5dc18086b2751" "${PAPER0_ROOT}/pyproject.toml"
check_sha256 "${PARENT_SHA}" "${PARENT_CHECKPOINT}"
check_sha256 "${C5P_CODEC_SHA}" "${C5P_CODEC}"
check_sha256 "${LATENT_NORMALIZATION_SHA}" "${LATENT_NORMALIZATION}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PAPER0_WANDB_ENTITY="${WANDB_ENTITY}"
"${PYTHON}" - <<'PY' | tee "${JOB_ROOT}/wandb_preflight.json"
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

echo "Running the complete clean Rocky 9 suite before full B4 training"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

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

readonly RUN_ID="p0b4full-${SLURM_JOB_ID}-s1701"
"${PYTHON}" -u "${ENTRYPOINT}" \
  --mode full \
  --seed 1701 \
  --artifact-root "${MODEL_DATA_STAGED}" \
  --codec-checkpoint "${C5P_CODEC}" \
  --codec-sha256 "${C5P_CODEC_SHA}" \
  --parent-checkpoint "${PARENT_CHECKPOINT}" \
  --parent-sha256 "${PARENT_SHA}" \
  --latent-normalization "${LATENT_NORMALIZATION}" \
  --latent-normalization-sha256 "${LATENT_NORMALIZATION_SHA}" \
  --output "${RUN_OUTPUT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --full-manifest "${FULL_MANIFEST}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-id "${RUN_ID}" \
  --wandb-run-name "b4-pde-refiner-h1-full-s1701-j${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum \
  "${RUN_OUTPUT}/config.json" \
  "${RUN_OUTPUT}/latent_normalization.json" \
  "${RUN_OUTPUT}/training_levels.npy" \
  "${RUN_OUTPUT}/validation_seed_bank.npy" \
  "${RUN_OUTPUT}/history.jsonl" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/selected.pt" \
  "${RUN_OUTPUT}/final_training_state.pt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

export PAPER0_B4_FULL_JOB_ROOT="${JOB_ROOT}"
export PAPER0_B4_FULL_RUN_OUTPUT="${RUN_OUTPUT}"
export PAPER0_B4_FULL_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B4_FULL_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import write_strict_json_atomic

root = Path(os.environ["PAPER0_B4_FULL_JOB_ROOT"])
output = Path(os.environ["PAPER0_B4_FULL_RUN_OUTPUT"])
result = json.loads((output / "result.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())

if result["scope"] != "B4_PDE_Refiner_H1_seed1701_full_training_85604":
    raise RuntimeError("full B4 scope differs")
config = result["config"]
if config["seed"] != 1701 or config["mode"] != "full":
    raise RuntimeError("full B4 identity differs")
if config["train_targets"] != [2, 432] or config["validation_targets"] != [498, 624]:
    raise RuntimeError("full B4 target intervals differ")
if config["validation_members"] != 2:
    raise RuntimeError("full B4 selection member count differs")
if result["completed_epochs"] != 100 or result["completed_optimizer_steps"] != 2700:
    raise RuntimeError("full B4 did not complete the frozen budget")
if result["EMA_updates"] != 2700:
    raise RuntimeError("full B4 EMA update count differs")
if result["validation_candidates_evaluated"] != 20:
    raise RuntimeError("full B4 validation-candidate count differs")
if result["validation_completed_epochs"] != list(range(5, 101, 5)):
    raise RuntimeError("full B4 validation schedule differs")
if result["selected_completed_epoch"] not in range(5, 101, 5):
    raise RuntimeError("full B4 selected a non-candidate epoch")
if result["selected_optimizer_step"] != result["selected_completed_epoch"] * 27:
    raise RuntimeError("full B4 selected optimizer step differs")
if result["preoptimization_parent_identity"]["bitwise_exact"] is not True:
    raise RuntimeError("full B4 parent identity failed")
if result["deterministic_parent_load_audit"]["passed"] is not True:
    raise RuntimeError("full B4 parent load audit failed")
if result["checkpoint_reload_bitwise_exact"] is not True:
    raise RuntimeError("full B4 selected-checkpoint reload failed")
if result["codec_bitwise_unchanged"] is not True:
    raise RuntimeError("full B4 frozen codec changed")
if result["parent_parameter_gradient_seen"] is not True:
    raise RuntimeError("full B4 parent parameters received no gradient")
if result["refinement_parameter_gradient_seen"] is not True:
    raise RuntimeError("full B4 refinement parameters received no gradient")
if result["training_level_counts"] != [10831, 10680, 10722, 10767]:
    raise RuntimeError("full B4 training-level counts differ")
if result["all_four_training_levels_exercised"] is not True:
    raise RuntimeError("full B4 omitted a training level")
if result["training_levels"]["shape"] != [100, 430]:
    raise RuntimeError("full B4 level-matrix shape differs")
if result["training_levels"]["raw_C_order_sha256"] != "ac370fa17291d8bd4c36ac4d451f78e63250c19ad77cf70a3f8403465e339ff6":
    raise RuntimeError("full B4 level-matrix bytes differ")
if result["validation_seed_bank"]["shape"] != [126, 2, 3]:
    raise RuntimeError("full B4 selection seed-bank shape differs")
if result["validation_seed_bank"]["sha256"] != "127936e25054925f4b114d5b174cbe876847555ffd0963ca54ce0e6c72f29884":
    raise RuntimeError("full B4 selection seed-bank bytes differ")
if result["network_calls_per_unamortized_member"] != 4:
    raise RuntimeError("full B4 network-call count differs")
if result["training_dtype"] != "float32" or result["validation_dtype"] != "float32":
    raise RuntimeError("full B4 did not run in float32")
if result["torch_float32_matmul_precision"] != "highest":
    raise RuntimeError("full B4 float32 matmul precision differs")
if result["cuda_matmul_allow_tf32"] is not False or result["cudnn_allow_tf32"] is not False:
    raise RuntimeError("full B4 unexpectedly used TF32")
if result["full_B4_training_authorized"] is not True:
    raise RuntimeError("full B4 result lacks full-training authorization")
if result["training_complete_is_scientific_acceptance"] is not False:
    raise RuntimeError("full B4 training was mislabeled as scientific acceptance")
for flag in (
    "held_out_85606_read",
    "physics_derived_loss_used",
    "target_truth_used_as_model_input",
    "absolute_time_used_as_model_input",
    "scientific_result",
    "scientific_B4_evaluation_performed",
    "H_det_evaluated",
    "H_prob_evaluated",
    "O3_launch_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result[flag] is not False:
        raise RuntimeError(f"full B4 scope flag {flag} differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("full B4 lacks a finished online W&B run")
if tracking["epochs_logged"] != 100:
    raise RuntimeError("full B4 did not log all 100 epochs")

history = [
    json.loads(line)
    for line in (output / "history.jsonl").read_text().splitlines()
    if line.strip()
]
if len(history) != 100:
    raise RuntimeError("full B4 history does not contain 100 epochs")
validation_epochs = [item["completed_epoch"] for item in history if item["validation_performed"]]
if validation_epochs != list(range(5, 101, 5)):
    raise RuntimeError("full B4 history validation cadence differs")
if history[-1]["learning_rate"] != 1e-6:
    raise RuntimeError("full B4 final learning rate differs")

summary = {
    "schema_version": 1,
    "scope": "phase3_B4_PDE_Refiner_H1_seed1701_full_training",
    "status": "training_completed",
    "scientific_acceptance": False,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_B4_FULL_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_B4_FULL_JOB_ID"],
    "completed_epochs": result["completed_epochs"],
    "completed_optimizer_steps": result["completed_optimizer_steps"],
    "validation_candidates_evaluated": result["validation_candidates_evaluated"],
    "selected_epoch": result["selected_epoch"],
    "selected_completed_epoch": result["selected_completed_epoch"],
    "selected_validation": result["selected_validation"],
    "selected_checkpoint": result["selected_checkpoint"],
    "final_training_state": result["final_training_state"],
    "validation_seed_bank": result["validation_seed_bank"],
    "training_levels": result["training_levels"],
    "parameter_count": result["parameter_count"],
    "peak_cuda_bytes": result["peak_cuda_bytes"],
    "wall_seconds": result["wall_seconds"],
    "wandb": tracking,
    "run_result": {
        "path": str(output / "result.json"),
        "sha256": sha256_path(output / "result.json"),
    },
    "artifact_index": {
        "path": str(output / "artifact_sha256.txt"),
        "sha256": sha256_path(output / "artifact_sha256.txt"),
    },
    "H_det_evaluated": False,
    "H_prob_evaluated": False,
    "O3_launch_allowed": False,
    "assimilation_allowed": False,
    "diagnostic_ranking_allowed": False,
}
write_strict_json_atomic(root / "result.json", summary)
PY

sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}/result.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"

sha256sum "${JOB_ROOT}/result.json" "${RUN_OUTPUT}/artifact_sha256.txt"
echo "Completed full B4 seed 1701 training at ${JOB_ROOT}"
echo "B4 remains scientifically undecided until H-det and H-prob evaluation"
