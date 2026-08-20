#!/bin/bash
# Truth-free four-phase H1 parent generation on CPU for bounded ECRD smoke.
# This artifact is explicitly unauthorized for full training or science.
#SBATCH --job-name=p0_ecrd_cpu_parent
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_sym_h1_parent_cpu_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
readonly H1_SHA="5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
readonly H1_TRAINING_COMMIT="9035bc3ce9d2351cd17586f4429af8116d43a47e"
readonly CODEC_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_ecrd_cpu_parent_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent_cpu_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/parent"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/ecrd_sym_h1_parent_cpu_smoke_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
readonly AMENDMENT="${PAPER0_ROOT}/paper0/protocol/ECRD_PARENT_CPU_SMOKE_AMENDMENT_2026-08-20.md"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/build_ecrd_sym_h1_parent.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="ecrd-sym-h1-parent-cpu-smoke-85604"

if [[ -e "${JOB_ROOT}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD CPU parent or node-local directory" >&2
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
  echo "ECRD CPU parent generation requires Rocky Linux 9" >&2
  exit 2
fi

check_sha256 "6bdc16136cdcd532ca3b9faf41e7d58df6aaca629b354335fd40875d27f884a9" "${MANIFEST}"
check_sha256 "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4" "${PROTOCOL}"
check_sha256 "4165cdc2f035c62eb7e6ade88a6380a8060e012e8c40c0b243ba446396276cbc" "${AMENDMENT}"
check_sha256 "${H1_SHA}" "${H1_CHECKPOINT}"
check_sha256 "${CODEC_SHA}" "${CODEC_CHECKPOINT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=truth_free_four_phase_H1_CPU_parent_for_bounded_smoke_85604"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("torch_threads="+str(torch.get_num_threads())); print("wandb="+wandb.__version__)'
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

echo "Running the complete clean Rocky 9 suite before CPU parent generation"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

echo "Staging and verifying immutable 85604 model data"
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
  echo "Expected eight verified model-data shards, got ${verified_shards}" >&2
  exit 2
fi

readonly RUN_ID="p0ecrdparentsmoke-${SLURM_JOB_ID}"
COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --h1-checkpoint "${H1_CHECKPOINT}"
  --h1-checkpoint-sha256 "${H1_SHA}"
  --codec-checkpoint "${CODEC_CHECKPOINT}"
  --codec-checkpoint-sha256 "${CODEC_SHA}"
  --h1-training-commit "${H1_TRAINING_COMMIT}"
  --manifest "${MANIFEST}"
  --execution-device cpu-smoke
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "ecrd-sym-h1-parent-cpu-smoke-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum \
  "${RUN_OUTPUT}/sym_h1_train.h5" \
  "${RUN_OUTPUT}/sym_h1_validation.h5" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

export PAPER0_ECRD_PARENT_ROOT="${RUN_OUTPUT}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import h5py

root = Path(os.environ["PAPER0_ECRD_PARENT_ROOT"])
result = json.loads((root / "result.json").read_text())
tracking = json.loads((root / "wandb.json").read_text())
required = {
    "status": "completed",
    "development_run": "85604",
    "execution_device": "cpu-smoke",
    "artifact_authority": "bounded_non_scientific_engineering_smoke_only",
    "full_training_authorized": False,
    "target_truth_read": False,
    "guard_frames_read": False,
    "held_out_85606_read": False,
    "training_performed": False,
    "physics_metric_evaluated": False,
    "assimilation_performed": False,
}
for name, expected in required.items():
    if result.get(name) != expected:
        raise RuntimeError(f"ECRD CPU parent result field {name!r} differs")
if result["splits"]["train"]["target_count"] != 430:
    raise RuntimeError("ECRD CPU parent training count differs")
if result["splits"]["validation"]["target_count"] != 126:
    raise RuntimeError("ECRD CPU parent validation count differs")
if tracking.get("remote_state_after_finish") != "finished":
    raise RuntimeError("ECRD CPU parent lacks finished online W&B")
for filename, split, count in (
    ("sym_h1_train.h5", "train", 430),
    ("sym_h1_validation.h5", "validation", 126),
):
    with h5py.File(root / filename, "r") as handle:
        if not bool(handle.attrs["completed"]):
            raise RuntimeError(f"{filename} is incomplete")
        if str(handle.attrs["split"]) != split:
            raise RuntimeError(f"{filename} split differs")
        if bool(handle.attrs["target_truth_read"]):
            raise RuntimeError(f"{filename} read target truth")
        if handle["standardized_parent_mean"].shape != (count, 5, 64, 32, 88):
            raise RuntimeError(f"{filename} shape differs")
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        if metadata.get("artifact_authority") != "bounded_non_scientific_engineering_smoke_only":
            raise RuntimeError(f"{filename} authority differs")
print(json.dumps({
    "status": "verified",
    "scope": "bounded_non_scientific_ECRD_smoke_parent_only",
    "full_training_authorized": False,
    "held_out_85606_read": False,
}, sort_keys=True))
PY

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Bounded-smoke CPU parent generation finished at $(date --iso-8601=seconds)"
echo "Result directory: ${JOB_ROOT}"
echo "This parent remains unauthorized for full training or scientific evaluation."
