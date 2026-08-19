#!/bin/bash
# One frozen full-field B5 residual-EDM mechanical smoke on Rocky 9 H100.
# Non-scientific: eight 85604 training targets, 64 optimizer steps, no validation.
#SBATCH --job-name=p3_b5_edmsmk
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b5_edm_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_FORECAST_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/h1_training_forecast.h5"
readonly H1_FORECAST_SHA="d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
readonly RESIDUAL_AUDIT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/residual_audit.json"
readonly RESIDUAL_AUDIT_SHA="d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b5_edm_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly H1_FORECAST_STAGED="${NODE_LOCAL_ROOT}/h1_training_forecast.h5"
readonly RESIDUAL_AUDIT_STAGED="${NODE_LOCAL_ROOT}/residual_audit.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_field_residual_edm_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/b5_joint_field_residual_edm_seed_1701"
readonly B5_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b5_field_residual_edm_smoke_85604.json"
readonly B5_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B5_FIELD_RESIDUAL_EDM_SMOKE_PROTOCOL.md"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_b5_field_residual_edm_smoke.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b5-joint-field-residual-edm-smoke"

if [[ -e "${JOB_ROOT}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite a B5 smoke or node-local directory" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

cleanup() {
  if [[ "${NODE_LOCAL_ROOT}" == "/tmp/${USER}/tcv_diagnostics_b5_edm_smoke_${SLURM_JOB_ID}" ]]; then
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
  echo "The frozen B5 residual-EDM smoke requires Rocky Linux 9" >&2
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

check_sha256 "2189c501071d245fbb87c0ddbb679f00fee73f8637e764b23d391421e8551283" "${B5_MANIFEST}"
check_sha256 "e50a2a3f150f08f24c444f3b877fd887877cfe57a444c89d5c86b25054afde75" "${B5_PROTOCOL}"
check_sha256 "9a3d391179d8128cc7f16ac7eab634cd5a40deacc43c3a0e80f2851c449c216e" "${ENTRYPOINT}"
check_sha256 "b0d2211ebab290a99c77c0295535669097cf05105b56922d94d4e61a933d6432" "${PAPER0_ROOT}/src/tcv_diagnostics/models/field_residual_edm.py"
check_sha256 "fa45cccced6da2651fda2b0862305d3ae2916083b868f941ce7f17961e66ec26" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_training.py"
check_sha256 "fbadd7cf6d171f32ce51203302a5128a6cf0d45d7e5d4ea1d18d00edce51a27e" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_wandb_tracking.py"
check_sha256 "4e28c00ab6c166256f899b5384fa140dae8a6bf64da73fb56e5183d50118ec53" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_forecast.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "4492111f3089d871ad77ffc06729b16ebb8822f6b737cd6422e5dc18086b2751" "${PAPER0_ROOT}/pyproject.toml"
check_sha256 "${H1_FORECAST_SHA}" "${H1_FORECAST_SOURCE}"
check_sha256 "${RESIDUAL_AUDIT_SHA}" "${RESIDUAL_AUDIT_SOURCE}"
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
  echo "scope=bounded_non_scientific_B5_joint_field_residual_EDM_smoke_85604"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
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

echo "Running the complete clean Rocky 9 suite before the B5 GPU smoke"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

echo "Staging and verifying the immutable 85604 inputs"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
cp -a "${H1_FORECAST_SOURCE}" "${H1_FORECAST_STAGED}"
cp -a "${RESIDUAL_AUDIT_SOURCE}" "${RESIDUAL_AUDIT_STAGED}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_STAGED}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_STAGED}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_STAGED}/artifact_sha256.txt"
check_sha256 "${H1_FORECAST_SHA}" "${H1_FORECAST_STAGED}"
check_sha256 "${RESIDUAL_AUDIT_SHA}" "${RESIDUAL_AUDIT_STAGED}"
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

readonly RUN_ID="p0b5edmsmk-${SLURM_JOB_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --mode smoke
  --seed 1701
  --artifact-root "${MODEL_DATA_STAGED}"
  --h1-training-forecast "${H1_FORECAST_STAGED}"
  --h1-training-forecast-sha256 "${H1_FORECAST_SHA}"
  --residual-audit "${RESIDUAL_AUDIT_STAGED}"
  --residual-audit-sha256 "${RESIDUAL_AUDIT_SHA}"
  --b5-manifest "${B5_MANIFEST}"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "b5-joint-field-residual-edm-smoke-s1701-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum \
  "${RUN_OUTPUT}/config.json" \
  "${RUN_OUTPUT}/history.jsonl" \
  "${RUN_OUTPUT}/smoke_checkpoint.pt" \
  "${RUN_OUTPUT}/sampler_probe.npz" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

export PAPER0_B5_SMOKE_JOB_ROOT="${JOB_ROOT}"
export PAPER0_B5_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B5_SMOKE_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import write_strict_json_atomic

root = Path(os.environ["PAPER0_B5_SMOKE_JOB_ROOT"])
output = root / "b5_joint_field_residual_edm_seed_1701"
result = json.loads((output / "result.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())

if result.get("status") != "passed" or result.get("all_mechanical_gates_passed") is not True:
    raise RuntimeError("B5 residual-EDM smoke did not pass every mechanical gate")
if result.get("scope") != "bounded_non_scientific_B5_joint_field_residual_EDM_smoke_85604":
    raise RuntimeError("B5 smoke scope differs")
if result.get("parameter_count") != 11604709:
    raise RuntimeError("B5 smoke parameter count differs")
if result.get("completed_optimizer_steps") != 64:
    raise RuntimeError("B5 smoke optimizer-step count differs")
if result.get("checkpoint_reload_bitwise_exact") is not True:
    raise RuntimeError("B5 smoke checkpoint reload differs")
if result.get("toroidal_equivariance", {}).get("passed") is not True:
    raise RuntimeError("B5 smoke toroidal equivariance failed")
if result.get("peak_cuda_GiB", 100.0) >= 75.0:
    raise RuntimeError("B5 smoke exceeded its H100 memory gate")
probe = result.get("sampler_probe", {})
if probe.get("canonical_field_shape") != [1, 2, 1, 5, 64, 32, 88]:
    raise RuntimeError("B5 sampler canonical axes differ")
if probe.get("finite") is not True or probe.get("nonzero_member_diversity") is not True:
    raise RuntimeError("B5 sampler output is invalid or collapsed")
if probe.get("network_evaluations_per_member") != 35:
    raise RuntimeError("B5 sampler network-evaluation count differs")
for flag in (
    "scientific_result",
    "physics_derived_loss_used",
    "target_truth_used_as_condition",
    "absolute_time_used_as_condition",
    "guard_frames_read",
    "validation_frames_read",
    "held_out_85606_read",
    "full_B5_training_authorized",
    "scientific_B5_evaluation_authorized",
    "O3_launch_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result.get(flag) is not False:
        raise RuntimeError(f"B5 smoke closed-scope flag differs: {flag}")
if tracking.get("mode") != "online" or tracking.get("remote_state_after_finish") != "finished":
    raise RuntimeError("B5 smoke lacks a finished online W&B run")
if tracking.get("steps_logged") != 64:
    raise RuntimeError("B5 W&B optimizer-step count differs")
if sum(1 for _ in (output / "history.jsonl").open()) != 64:
    raise RuntimeError("B5 local history does not contain exactly 64 records")

summary = {
    "schema_version": 1,
    "scope": "phase3_B5_joint_field_residual_EDM_bounded_GPU_smoke",
    "status": "passed",
    "scientific_result": False,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_B5_SMOKE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_B5_SMOKE_JOB_ID"],
    "run_result": {
        "path": str(output / "result.json"),
        "sha256": sha256_path(output / "result.json"),
    },
    "artifact_index": {
        "path": str(output / "artifact_sha256.txt"),
        "sha256": sha256_path(output / "artifact_sha256.txt"),
    },
    "parameter_count": result["parameter_count"],
    "completed_optimizer_steps": result["completed_optimizer_steps"],
    "initial_fixed_probe": result["initial_fixed_probe"],
    "final_fixed_probe": result["final_fixed_probe"],
    "fixed_probe_relative_change": result["fixed_probe_relative_change"],
    "toroidal_equivariance": result["toroidal_equivariance"],
    "sampler_probe": result["sampler_probe"],
    "peak_cuda_GiB": result["peak_cuda_GiB"],
    "wandb": tracking,
    "frozen_H1_inputs_reverified_after_smoke": True,
    "full_B5_training_authorized": False,
    "validation_read_allowed": False,
    "next_permitted_action": "write_separate_B5_full_training_and_validation_protocol_only",
}
write_strict_json_atomic(root / "result.json", summary)
PY

check_sha256 "${H1_FORECAST_SHA}" "${H1_FORECAST_SOURCE}"
check_sha256 "${RESIDUAL_AUDIT_SHA}" "${RESIDUAL_AUDIT_SOURCE}"
check_sha256 "${H1_FORECAST_SHA}" "${H1_FORECAST_STAGED}"
check_sha256 "${RESIDUAL_AUDIT_SHA}" "${RESIDUAL_AUDIT_STAGED}"
sha256sum \
  "${JOB_ROOT}/result.json" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"
echo "B5 bounded residual-EDM smoke passed at ${JOB_ROOT}"
echo "No scientific checkpoint was selected; validation, 85606, O3, assimilation, and ranking remain closed"
