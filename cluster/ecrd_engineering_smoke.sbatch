#!/bin/bash
# Four-arm bounded ECRD engineering smoke on Rocky 9 H100.
# Non-scientific: four 85604 train targets, two updates, four validation targets.
#SBATCH --job-name=p0_ecrd_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%1
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_engineering_smoke_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_TRAIN_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/h1_training_forecast.h5"
readonly H1_TRAIN_SHA="d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
readonly H1_VALIDATION_SOURCE="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly H1_VALIDATION_SHA="a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
readonly SYM_PARENT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent_cpu_smoke/job_6912481/parent"
readonly SYM_H1_TRAIN_SOURCE="${SYM_PARENT_SOURCE}/sym_h1_train.h5"
readonly SYM_H1_TRAIN_SHA="d238d055c3f1da9e3096a81cac67176f90365c99dfb423a1a0629f85b61f9532"
readonly SYM_H1_VALIDATION_SOURCE="${SYM_PARENT_SOURCE}/sym_h1_validation.h5"
readonly SYM_H1_VALIDATION_SHA="9acaf190507d2cc0216a4f137e7e63717370e52484e99351d6ded903e30cc2d3"
readonly SYM_PARENT_RESULT_SOURCE="${SYM_PARENT_SOURCE}/result.json"
readonly SYM_PARENT_RESULT_SHA="ba8710e1e0813652fbbddc5c0ce7de20ddcdc39a3d74cf478ed344d8f3bb4037"
readonly SYM_PARENT_ARTIFACT_INDEX="${SYM_PARENT_SOURCE}/artifact_sha256.txt"
readonly SYM_PARENT_ARTIFACT_INDEX_SHA="7a81650059e9ca3d2157843e0072477713cfe9f9c74744b89aae86a9fc159643"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_ecrd_smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly H1_TRAIN_STAGED="${NODE_LOCAL_ROOT}/h1_training_forecast.h5"
readonly H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/h1_validation_forecast.h5"
readonly SYM_H1_TRAIN_STAGED="${NODE_LOCAL_ROOT}/sym_h1_train.h5"
readonly SYM_H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/sym_h1_validation.h5"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_engineering_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_ARRAY_JOB_ID}"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/ecrd_engineering_smoke_85604.json"
readonly MANIFEST_SHA="6204e7bfef02c449f464ea7647b8d6fc14c0bba22135c0b56ea47d77f530ac45"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_ecrd.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="ecrd-engineering-smoke-85604"

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly ARM="B5"; readonly SAFE_ARM="b5" ;;
  1) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context" ;;
  2) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd" ;;
  3) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history" ;;
  *) echo "Unsupported ECRD smoke task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

readonly TASK_ROOT="${JOB_ROOT}/task_${SLURM_ARRAY_TASK_ID}_${SAFE_ARM}_seed_1701"
readonly RUN_OUTPUT="${TASK_ROOT}/model"

if [[ -e "${TASK_ROOT}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD smoke task or node-local directory" >&2
  exit 2
fi
mkdir -p "${TASK_ROOT}"
exec > >(tee -a "${TASK_ROOT}/job.log") 2>&1

cleanup() {
  if [[ "${NODE_LOCAL_ROOT}" == "/tmp/${USER}/tcv_diagnostics_ecrd_smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" ]]; then
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
  echo "The ECRD engineering smoke requires Rocky Linux 9" >&2
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
  echo "scope=bounded_non_scientific_ECRD_engineering_smoke_85604"
  echo "arm=${ARM}"
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

echo "Running the complete clean Rocky 9 suite before ${ARM} smoke"
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

readonly RUN_ID="p0ecrdsmk-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --mode smoke
  --arm "${ARM}"
  --seed 1701
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
  --wandb-run-name "ecrd-smoke-${SAFE_ARM}-s1701-j${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${TASK_ROOT}/command.sh"
printf '\n' >> "${TASK_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${TASK_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/smoke_probe.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
export PAPER0_ECRD_SMOKE_OUTPUT="${RUN_OUTPUT}"
export PAPER0_ECRD_SMOKE_ARM="${ARM}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_ECRD_SMOKE_OUTPUT"])
arm = os.environ["PAPER0_ECRD_SMOKE_ARM"]
result = json.loads((root / "result.json").read_text())
probe = json.loads((root / "smoke_probe.json").read_text())
tracking = json.loads((root / "wandb.json").read_text())
required = {
    "status": "training_completed_checkpoint_selected",
    "mode": "smoke",
    "arm": arm,
    "seed": 1701,
    "completed_epochs": 1,
    "completed_optimizer_steps": 2,
    "target_presentations": 4,
    "checkpoint_reload_bitwise_exact": True,
    "held_out_85606_read": False,
    "scientific_forecast_generated": False,
    "assimilation_performed": False,
    "diagnostic_ranking_performed": False,
    "steering_performed": False,
}
for name, expected in required.items():
    if result.get(name) != expected:
        raise RuntimeError(f"{arm} smoke result field {name!r} differs")
if probe.get("all_mechanical_gates_passed") is not True:
    raise RuntimeError(f"{arm} failed its full-volume mechanical gates")
if tracking.get("mode") != "online" or tracking.get("remote_state_after_finish") != "finished":
    raise RuntimeError(f"{arm} lacks a finished online W&B run")
print(json.dumps({
    "status": "task_passed",
    "arm": arm,
    "scientific_result": False,
    "full_training_authorized": False,
    "held_out_85606_read": False,
}, sort_keys=True))
PY

sha256sum \
  "${RUN_OUTPUT}/config.json" \
  "${RUN_OUTPUT}/training_order.npy" \
  "${RUN_OUTPUT}/validation_seed_bank.npy" \
  "${RUN_OUTPUT}/history.jsonl" \
  "${RUN_OUTPUT}/candidates/ema_epoch_001.pt" \
  "${RUN_OUTPUT}/selected.pt" \
  "${RUN_OUTPUT}/smoke_probe.json" \
  "${RUN_OUTPUT}/smoke_probe_standardized_fields.npz" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

check_sha256 "${H1_TRAIN_SHA}" "${H1_TRAIN_SOURCE}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_SOURCE}"
check_sha256 "${SYM_H1_TRAIN_SHA}" "${SYM_H1_TRAIN_SOURCE}"
check_sha256 "${SYM_H1_VALIDATION_SHA}" "${SYM_H1_VALIDATION_SOURCE}"
check_sha256 "${H1_TRAIN_SHA}" "${H1_TRAIN_STAGED}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_STAGED}"
check_sha256 "${SYM_H1_TRAIN_SHA}" "${SYM_H1_TRAIN_STAGED}"
check_sha256 "${SYM_H1_VALIDATION_SHA}" "${SYM_H1_VALIDATION_STAGED}"
sha256sum \
  "${TASK_ROOT}/command.sh" \
  "${TASK_ROOT}/environment.txt" \
  "${TASK_ROOT}/slurm_job.txt" \
  "${TASK_ROOT}/test_output.txt" \
  "${TASK_ROOT}/wandb_preflight.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${TASK_ROOT}/artifact_sha256.txt"

echo "${ARM} bounded ECRD engineering smoke passed at ${TASK_ROOT}"
echo "This is a mechanical result only; full training and scientific evaluation remain unauthorized."
