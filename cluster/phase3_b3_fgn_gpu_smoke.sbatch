#!/bin/bash
# Rocky 9 one-GPU implementation gate for the frozen B3-FGN-H1 retrofit.
# Non-scientific: seed 1701, 16 train targets, 4 validation targets, 2 epochs.
#SBATCH --job-name=p3_b3_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b3_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b3_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_gpu_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/b3_fgn_h1_seed_1701"
readonly B3_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b3_fgn_85604.json"
readonly B3_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B3_FGN_PROTOCOL.md"
readonly B2_RESULT="${PAPER0_ROOT}/paper0/results/phase3_b2_event_eligibility_amendment_6898348.json"
readonly O2_RESULT="${PAPER0_ROOT}/paper0/results/phase2_o2_evaluation_full_6896117.json"
readonly O2_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_c5p_o2_full_runs_85604.json"
readonly PARENT_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
readonly PARENT_SHA="5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
readonly C5P_CODEC="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly C5P_CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
readonly LATENT_NORMALIZATION="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/latent_normalization.json"
readonly LATENT_NORMALIZATION_SHA="afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_b3_fgn.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b3-fgn-h1-smoke"

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
  echo "This B3 smoke requires Rocky Linux 9" >&2
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

check_sha256 "c31d24d843050c5708440018217e48ca66f5c3a3f4ee0ddb110a3287a73292d1" "${B3_PROTOCOL}"
check_sha256 "8789c1a922bbeb9817144344563107f6e72a7a7c549436ae698155d93daba900" "${B3_MANIFEST}"
check_sha256 "15bcfc0b4c9cec2a858848e1d8fbc0fdf2da6f8a3bc6fd001e5c233abae7b397" "${B2_RESULT}"
check_sha256 "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb" "${O2_RESULT}"
check_sha256 "1f49699396a8529a1ee42fd5b0d746c75c98a750c3e376f3720c23dfae4ed203" "${O2_MANIFEST}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb" "${PAPER0_ROOT}/src/tcv_diagnostics/models/o2.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "69c5e140e6f456e2442ae40ec5be7c04c5d228baf6ec95e119de5a3d1c86e79d" "${PAPER0_ROOT}/src/tcv_diagnostics/models/functional_noise.py"
check_sha256 "cb4951b3d04c3153a6192951a520d7042c927d1f0574978d503596d6c14e36c9" "${PAPER0_ROOT}/src/tcv_diagnostics/models/FGN_LICENSE.txt"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "e301a99d3e5d63795c9d2b27d0d562afb133d186e2fd5d958feeb7b2e5e97f77" "${PAPER0_ROOT}/src/tcv_diagnostics/fgn_training.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "ead75d9763ed8ce2d76e53f8b18a163c09cd8f921e979d72062c8b4d5d336d15" "${PAPER0_ROOT}/src/tcv_diagnostics/fgn_wandb_tracking.py"
check_sha256 "2bd4573b2588a199b164b49997f2ce6a9a355c813b6e3d66b9da39768be110fe" "${ENTRYPOINT}"
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

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
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

echo "Running the complete Rocky 9 suite before the B3 GPU smoke"
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

"${PYTHON}" -u "${ENTRYPOINT}" \
  --mode smoke \
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
  --b3-manifest "${B3_MANIFEST}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-id "p0b3smoke-${SLURM_JOB_ID}-s1701" \
  --wandb-run-name "b3-fgn-h1-smoke-s1701-j${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
sha256sum \
  "${RUN_OUTPUT}/config.json" \
  "${RUN_OUTPUT}/latent_normalization.json" \
  "${RUN_OUTPUT}/validation_noise.npy" \
  "${RUN_OUTPUT}/history.jsonl" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/selected.pt" \
  "${RUN_OUTPUT}/final_training_state.pt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${RUN_OUTPUT}/artifact_sha256.txt"

export PAPER0_B3_SMOKE_JOB_ROOT="${JOB_ROOT}"
export PAPER0_B3_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B3_SMOKE_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import write_strict_json_atomic

root = Path(os.environ["PAPER0_B3_SMOKE_JOB_ROOT"])
output = root / "b3_fgn_h1_seed_1701"
result = json.loads((output / "result.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())

if result["scope"] != "bounded_non_scientific_B3_FGN_H1_GPU_smoke":
    raise RuntimeError("B3 smoke scope differs")
config = result["config"]
if config["seed"] != 1701 or config["mode"] != "smoke":
    raise RuntimeError("B3 smoke identity differs")
if config["train_targets"] != [2, 18] or config["validation_targets"] != [498, 502]:
    raise RuntimeError("B3 smoke target bounds differ")
if config["ensemble_members"] != 2:
    raise RuntimeError("B3 smoke ensemble count differs")
if result["completed_epochs"] != 2 or result["completed_optimizer_steps"] != 2:
    raise RuntimeError("B3 smoke optimization budget differs")
if result["preoptimization_parent_identity"]["bitwise_exact"] is not True:
    raise RuntimeError("B3 noise-disabled parent identity failed")
if result["deterministic_parent_load_audit"]["passed"] is not True:
    raise RuntimeError("B3 deterministic parent load audit failed")
if result["checkpoint_reload_bitwise_exact"] is not True:
    raise RuntimeError("B3 selected-checkpoint reload identity failed")
if result["codec_bitwise_unchanged"] is not True:
    raise RuntimeError("B3 frozen codec changed")
if result["common_parameter_gradient_seen"] is not True:
    raise RuntimeError("B3 common parameters received no gradient")
if result["new_parameter_gradient_seen"] is not True:
    raise RuntimeError("B3 new parameters received no gradient")

probe = result["member_probe"]
if probe["finite"] is not True:
    raise RuntimeError("B3 probe is non-finite")
if probe["nonzero_latent_diversity"] is not True or probe["nonzero_field_diversity"] is not True:
    raise RuntimeError("B3 members collapsed")
if probe["canonical_forecast_shape"] != [1, 2, 1, 5, 64, 32, 88]:
    raise RuntimeError("B3 canonical forecast axes differ")
if result["validation_noise_bank"]["shape"] != [126, 2, 32]:
    raise RuntimeError("B3 validation-noise bank shape differs")

for flag in (
    "held_out_85606_read",
    "physics_derived_loss_used",
    "target_truth_used_as_model_input",
    "absolute_time_used_as_model_input",
    "scientific_result",
    "full_B3_training_authorized",
    "probabilistic_scientific_gate_evaluated",
    "O3_launch_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result[flag] is not False:
        raise RuntimeError(f"B3 smoke scope flag {flag} differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("B3 smoke lacks a finished online W&B run")
if tracking["epochs_logged"] != 2:
    raise RuntimeError("B3 W&B epoch count differs")

summary = {
    "schema_version": 1,
    "scope": "phase3_B3_FGN_H1_bounded_GPU_smoke",
    "status": "passed",
    "scientific_result": False,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_B3_SMOKE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_B3_SMOKE_JOB_ID"],
    "run_result": {
        "path": str(output / "result.json"),
        "sha256": sha256_path(output / "result.json"),
    },
    "artifact_index": {
        "path": str(output / "artifact_sha256.txt"),
        "sha256": sha256_path(output / "artifact_sha256.txt"),
    },
    "selected_checkpoint": result["selected_checkpoint"],
    "preoptimization_parent_identity": result["preoptimization_parent_identity"],
    "checkpoint_reload_bitwise_exact": result["checkpoint_reload_bitwise_exact"],
    "codec_bitwise_unchanged": result["codec_bitwise_unchanged"],
    "member_probe": result["member_probe"],
    "wandb": tracking,
    "full_B3_training_authorized": False,
}
write_strict_json_atomic(root / "result.json", summary)
PY

sha256sum "${JOB_ROOT}/result.json" "${RUN_OUTPUT}/artifact_sha256.txt"
echo "B3 bounded smoke passed at ${JOB_ROOT}"
