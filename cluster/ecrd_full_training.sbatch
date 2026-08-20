#!/bin/bash
# Matched full ECRD model-development training on 85604 only.
# Eleven new tasks complement the hash-verified historical B5 seed-1701 run.
# No scientific forecasts, physics metrics, assimilation, or held-out access.
#SBATCH --job-name=p0_ecrd_full
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --array=0-10%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_full_training_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_TRAIN_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/h1_training_forecast.h5"
readonly H1_TRAIN_SHA="d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
readonly H1_VALIDATION_SOURCE="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly H1_VALIDATION_SHA="a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
readonly SYM_PARENT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent/job_6913234/parent"
readonly SYM_H1_TRAIN_SOURCE="${SYM_PARENT_SOURCE}/sym_h1_train.h5"
readonly SYM_H1_TRAIN_SHA="21725a5ae29832068676840aabea37a2e59eac861e8b9243bf45414de6811cbd"
readonly SYM_H1_VALIDATION_SOURCE="${SYM_PARENT_SOURCE}/sym_h1_validation.h5"
readonly SYM_H1_VALIDATION_SHA="fd23a42a6603e9150868a7eb0dc5f76580f0044bf1bd0b9e8e01ef04b378fd8d"
readonly SYM_PARENT_RESULT_SOURCE="${SYM_PARENT_SOURCE}/result.json"
readonly SYM_PARENT_RESULT_SHA="e52abfae2616ad14ea468dc4e7ff18f56a43d15775a7cb373e6b52f75539c1af"
readonly SYM_PARENT_ARTIFACT_INDEX="${SYM_PARENT_SOURCE}/artifact_sha256.txt"
readonly SYM_PARENT_ARTIFACT_INDEX_SHA="740047ad1d3cf7a4ecef992ea082ef1a79f9ea8dedc2ab3d33a8551347f6cf76"
readonly SMOKE_RESULT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_engineering_smoke/job_6912495/result.json"
readonly SMOKE_RESULT_SHA="e135c78a925c09137f6609f38ebb40e8de218dd99e9b6a393f34dc26fd8768ad"
readonly SMOKE_INDEX="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_engineering_smoke/job_6912495/finalizer_job_6913232/artifact_sha256.txt"
readonly SMOKE_INDEX_SHA="a939e8c42e3ecbc7341de80d8ac04c5ee54a4bdf4f07dcdc530ea64c486d7ece"
readonly PARENT_COMPARISON="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_parent_execution_comparison/job_6913271/comparison/result.json"
readonly PARENT_COMPARISON_SHA="b4fb353bb4070309c8d59234951f738fe4bf0eee13d59f8ad4cba366b3bed908"
readonly HISTORICAL_B5_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_field_residual_edm_full/job_6901531/b5_joint_field_residual_edm_seed_1701"
readonly HISTORICAL_B5_RESULT="${HISTORICAL_B5_ROOT}/result.json"
readonly HISTORICAL_B5_RESULT_SHA="31d17363261cbe3e007bdeb0862760cfb731b2490da7d76b37f474f8fdb93068"
readonly HISTORICAL_B5_CHECKPOINT="${HISTORICAL_B5_ROOT}/selected.pt"
readonly HISTORICAL_B5_CHECKPOINT_SHA="255904ef362c4d3f0fdb873131cd0b30bc02ea384e76e244d50698bd50df0c72"
readonly HISTORICAL_B5_INDEX="${HISTORICAL_B5_ROOT}/artifact_sha256.txt"
readonly HISTORICAL_B5_INDEX_SHA="19cacc363d7c886e754c0181ee999a24d469b11db60076ae0d71e0c2236901a0"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/ecrd_full_training_85604.json"
readonly MANIFEST_SHA="94100e13a20aa23249aa39f0d910f160ad7aadf60ab42b7e4975969dd3abe6e1"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_ecrd.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_full_training"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_ARRAY_JOB_ID}"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_ecrd_full_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly H1_TRAIN_STAGED="${NODE_LOCAL_ROOT}/h1_training_forecast.h5"
readonly H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/h1_validation_forecast.h5"
readonly SYM_H1_TRAIN_STAGED="${NODE_LOCAL_ROOT}/sym_h1_train.h5"
readonly SYM_H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/sym_h1_validation.h5"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="ecrd-model-development"

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly ARM="B5"; readonly SAFE_ARM="b5"; readonly SEED="1702"; readonly PARAMS="11604709" ;;
  1) readonly ARM="B5"; readonly SAFE_ARM="b5"; readonly SEED="1703"; readonly PARAMS="11604709" ;;
  2) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context"; readonly SEED="1701"; readonly PARAMS="11350909" ;;
  3) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context"; readonly SEED="1702"; readonly PARAMS="11350909" ;;
  4) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context"; readonly SEED="1703"; readonly PARAMS="11350909" ;;
  5) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd"; readonly SEED="1701"; readonly PARAMS="11455746" ;;
  6) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd"; readonly SEED="1702"; readonly PARAMS="11455746" ;;
  7) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd"; readonly SEED="1703"; readonly PARAMS="11455746" ;;
  8) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history"; readonly SEED="1701"; readonly PARAMS="11462766" ;;
  9) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history"; readonly SEED="1702"; readonly PARAMS="11462766" ;;
  10) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history"; readonly SEED="1703"; readonly PARAMS="11462766" ;;
  *) echo "Unsupported ECRD full-training task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

readonly TASK_ROOT="${JOB_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${SAFE_ARM}_seed_${SEED}"
readonly RUN_OUTPUT="${TASK_ROOT}/model"

if [[ -e "${TASK_ROOT}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD full-training task or node-local directory" >&2
  exit 2
fi
mkdir -p "${TASK_ROOT}"
exec > >(tee -a "${TASK_ROOT}/job.log") 2>&1

cleanup() {
  if [[ "${NODE_LOCAL_ROOT}" == "/tmp/${USER}/tcv_diagnostics_ecrd_full_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" ]]; then
    rm -rf "${NODE_LOCAL_ROOT}"
  fi
}
trap cleanup EXIT

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
  echo "ECRD full training requires Rocky Linux 9" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated H100 is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100"; then
  echo "The allocated accelerator is not an H100" >&2
  exit 2
fi

check_sha256 "${MANIFEST_SHA}" "${MANIFEST}"
check_sha256 "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4" "${PROTOCOL}"
check_sha256 "4e74409a3e4fd7beb7c4fa3c31e536e1739d6683cbc76e62fb5d05872fe85d59" "${ENTRYPOINT}"
check_sha256 "b366f067217f8fe675ddd5ab0741ecb4b1c19325c6af7cd27d97b21a3e51f0e2" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_training.py"
check_sha256 "1c03bb2f3cc0249a58c8a14f238348692798970a6424188833acfbf10b35f679" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_data.py"
check_sha256 "38d1d69a930fa27923de1cebf98553f70bf797a1e8ac250a7c4aef9b4961421f" "${PAPER0_ROOT}/src/tcv_diagnostics/models/ecrd.py"
check_sha256 "a42e3c9ed2ff652573288bbb11e610bae5dca3c66d14c98a89caa80fe9cd5ae2" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_wandb_tracking.py"
check_sha256 "b0d2211ebab290a99c77c0295535669097cf05105b56922d94d4e61a933d6432" "${PAPER0_ROOT}/src/tcv_diagnostics/models/field_residual_edm.py"
check_sha256 "4e28c00ab6c166256f899b5384fa140dae8a6bf64da73fb56e5183d50118ec53" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_forecast.py"
check_sha256 "7847e57452020b7fb23f8cfec4c4e7f00f0b6585df51fbac53aa62bd2fe0223e" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_forecast.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "4492111f3089d871ad77ffc06729b16ebb8822f6b737cd6422e5dc18086b2751" "${PAPER0_ROOT}/pyproject.toml"
check_sha256 "${H1_TRAIN_SHA}" "${H1_TRAIN_SOURCE}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_SOURCE}"
check_sha256 "${SYM_H1_TRAIN_SHA}" "${SYM_H1_TRAIN_SOURCE}"
check_sha256 "${SYM_H1_VALIDATION_SHA}" "${SYM_H1_VALIDATION_SOURCE}"
check_sha256 "${SYM_PARENT_RESULT_SHA}" "${SYM_PARENT_RESULT_SOURCE}"
check_sha256 "${SYM_PARENT_ARTIFACT_INDEX_SHA}" "${SYM_PARENT_ARTIFACT_INDEX}"
check_sha256 "${SMOKE_RESULT_SHA}" "${SMOKE_RESULT}"
check_sha256 "${SMOKE_INDEX_SHA}" "${SMOKE_INDEX}"
check_sha256 "${PARENT_COMPARISON_SHA}" "${PARENT_COMPARISON}"
check_sha256 "${HISTORICAL_B5_RESULT_SHA}" "${HISTORICAL_B5_RESULT}"
check_sha256 "${HISTORICAL_B5_CHECKPOINT_SHA}" "${HISTORICAL_B5_CHECKPOINT}"
check_sha256 "${HISTORICAL_B5_INDEX_SHA}" "${HISTORICAL_B5_INDEX}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" > "${TASK_ROOT}/slurm_job.txt"
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
  echo "scope=ECRD_matched_full_training_85604"
  echo "arm=${ARM}"
  echo "seed=${SEED}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
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

echo "Running the complete clean Rocky 9 suite before ${ARM} seed ${SEED}"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${TASK_ROOT}/test_output.txt"

echo "Staging and re-verifying immutable 85604-only inputs"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
cp -a "${H1_TRAIN_SOURCE}" "${H1_TRAIN_STAGED}"
cp -a "${H1_VALIDATION_SOURCE}" "${H1_VALIDATION_STAGED}"
cp -a "${SYM_H1_TRAIN_SOURCE}" "${SYM_H1_TRAIN_STAGED}"
cp -a "${SYM_H1_VALIDATION_SOURCE}" "${SYM_H1_VALIDATION_STAGED}"
check_sha256 "${H1_TRAIN_SHA}" "${H1_TRAIN_STAGED}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_STAGED}"
check_sha256 "${SYM_H1_TRAIN_SHA}" "${SYM_H1_TRAIN_STAGED}"
check_sha256 "${SYM_H1_VALIDATION_SHA}" "${SYM_H1_VALIDATION_STAGED}"
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

readonly RUN_ID="p0ecrdfull-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-s${SEED}"
COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --mode full
  --arm "${ARM}"
  --seed "${SEED}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --h1-training-parent "${H1_TRAIN_STAGED}"
  --h1-training-parent-sha256 "${H1_TRAIN_SHA}"
  --h1-validation-parent "${H1_VALIDATION_STAGED}"
  --h1-validation-parent-sha256 "${H1_VALIDATION_SHA}"
  --sym-h1-training-parent "${SYM_H1_TRAIN_STAGED}"
  --sym-h1-training-parent-sha256 "${SYM_H1_TRAIN_SHA}"
  --sym-h1-validation-parent "${SYM_H1_VALIDATION_STAGED}"
  --sym-h1-validation-parent-sha256 "${SYM_H1_VALIDATION_SHA}"
  --manifest "${MANIFEST}"
  --manifest-sha256 "${MANIFEST_SHA}"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "ecrd-full-${SAFE_ARM}-s${SEED}-j${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${TASK_ROOT}/command.sh"
printf '\n' >> "${TASK_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${TASK_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
export PAPER0_ECRD_FULL_OUTPUT="${RUN_OUTPUT}"
export PAPER0_ECRD_FULL_ARM="${ARM}"
export PAPER0_ECRD_FULL_SEED="${SEED}"
export PAPER0_ECRD_FULL_PARAMS="${PARAMS}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_ECRD_FULL_OUTPUT"])
arm = os.environ["PAPER0_ECRD_FULL_ARM"]
seed = int(os.environ["PAPER0_ECRD_FULL_SEED"])
parameters = int(os.environ["PAPER0_ECRD_FULL_PARAMS"])
result = json.loads((root / "result.json").read_text())
tracking = json.loads((root / "wandb.json").read_text())
required = {
    "status": "training_completed_checkpoint_selected",
    "mode": "full",
    "arm": arm,
    "seed": seed,
    "development_run": "85604",
    "parameter_count": parameters,
    "completed_epochs": 100,
    "completed_optimizer_steps": 10800,
    "target_presentations": 43000,
    "candidate_count": 20,
    "checkpoint_reload_bitwise_exact": True,
    "physics_derived_loss_used": False,
    "physics_metric_used_for_checkpoint_selection": False,
    "target_truth_used_as_condition": False,
    "guard_frames_read": False,
    "held_out_85606_read": False,
    "scientific_forecast_generated": False,
    "assimilation_performed": False,
    "diagnostic_ranking_performed": False,
    "steering_performed": False,
}
for name, expected in required.items():
    if result.get(name) != expected:
        raise RuntimeError(f"{arm} seed {seed} result field {name!r} differs")
epochs = [item["completed_epoch"] for item in result["artifacts"]["candidate_checkpoints"]]
if epochs != list(range(5, 101, 5)):
    raise RuntimeError(f"{arm} seed {seed} candidate epochs differ")
if tracking.get("mode") != "online" or tracking.get("remote_state_after_finish") != "finished":
    raise RuntimeError(f"{arm} seed {seed} lacks a finished online W&B run")
print(json.dumps({
    "status": "full_training_task_passed",
    "arm": arm,
    "seed": seed,
    "held_out_85606_read": False,
    "scientific_forecast_generated": False,
}, sort_keys=True))
PY

CANDIDATES=("${RUN_OUTPUT}"/candidates/ema_epoch_*.pt)
if [[ "${#CANDIDATES[@]}" -ne 20 ]]; then
  echo "Expected twenty ECRD candidate checkpoints" >&2
  exit 2
fi
sha256sum \
  "${RUN_OUTPUT}/config.json" \
  "${RUN_OUTPUT}/training_order.npy" \
  "${RUN_OUTPUT}/validation_seed_bank.npy" \
  "${RUN_OUTPUT}/history.jsonl" \
  "${CANDIDATES[@]}" \
  "${RUN_OUTPUT}/selected.pt" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

check_sha256 "${H1_TRAIN_SHA}" "${H1_TRAIN_SOURCE}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_SOURCE}"
check_sha256 "${SYM_H1_TRAIN_SHA}" "${SYM_H1_TRAIN_SOURCE}"
check_sha256 "${SYM_H1_VALIDATION_SHA}" "${SYM_H1_VALIDATION_SOURCE}"
check_sha256 "${SMOKE_RESULT_SHA}" "${SMOKE_RESULT}"
check_sha256 "${PARENT_COMPARISON_SHA}" "${PARENT_COMPARISON}"
sha256sum \
  "${TASK_ROOT}/command.sh" \
  "${TASK_ROOT}/environment.txt" \
  "${TASK_ROOT}/slurm_job.txt" \
  "${TASK_ROOT}/test_output.txt" \
  "${TASK_ROOT}/wandb_preflight.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${TASK_ROOT}/artifact_sha256.txt"

echo "${ARM} seed ${SEED} ECRD full training completed at ${TASK_ROOT}"
echo "Scientific forecast generation and 85606 access remain unauthorized."
