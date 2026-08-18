#!/bin/bash
# Rocky 9 one-GPU implementation gate for the frozen B2-LDM-H2 arm.
# This is non-scientific: seed 1701, 16 train targets, and exactly two epochs.
#SBATCH --job-name=p3_b2_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b2_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_gpu_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/b2_ldm_h2_seed_1701"
readonly B2_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_ldm_85604.json"
readonly B2_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B2_LDM_PROTOCOL.md"
readonly O2_RESULT="${PAPER0_ROOT}/paper0/results/phase2_o2_evaluation_full_6896117.json"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly C5P_CODEC="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly C5P_CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b2-ldm-h2-smoke"

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
  echo "This B2 smoke requires Rocky Linux 9" >&2
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

check_sha256 "22b20a1050b68aaada21efc3840b322da16bfa0ab4a870e03a5bfffc97ecfe44" "${B2_PROTOCOL}"
check_sha256 "75d6f812fc7d203d5c92c588515f4a59d4b3b0cb8688fdd7514b6d09b842f9c3" "${B2_MANIFEST}"
check_sha256 "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb" "${O2_RESULT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "8eb062349ab78d8f163c46bb3539e7fd3791cc095ac3b8ed17ae9c000a5be512" "${PAPER0_ROOT}/src/tcv_diagnostics/models/__init__.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "9bfa4c86287f533e6baa488be3f0ac6727a318a2e5a83f5c0b1be11f9dfe3bbe" "${PAPER0_ROOT}/src/tcv_diagnostics/models/modulated_vit.py"
check_sha256 "447427a5dd3273689b52d012ebe73fbeeaac15d143a4fda00a06ee065b1e1d4f" "${PAPER0_ROOT}/src/tcv_diagnostics/models/latent_diffusion.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "e6e1743332be263c0d1984b32864a8d328198de9f766c9dbc75cf12949cbfb1a" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_training.py"
check_sha256 "c3f87a79ee71dd90f9c3b852c479aed611ee06f9ddf2b1bdb47fcca2477eaaba" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_wandb_tracking.py"
check_sha256 "5b905350aeab1972c48810036b4cb7d114063b7eed5ddc2aa7e560aa9bd0b175" "${PAPER0_ROOT}/paper0/tools/train_b2_ldm.py"
check_sha256 "e13ec80db2310c152e9e7f5fbeee8f2d9d617b519a026d3e76736a01efab6267" "${PAPER0_ROOT}/paper0/PROVENANCE.md"
check_sha256 "4492111f3089d871ad77ffc06729b16ebb8822f6b737cd6422e5dc18086b2751" "${PAPER0_ROOT}/pyproject.toml"
check_sha256 "${C5P_CODEC_SHA}" "${C5P_CODEC}"
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
  "${PYTHON}" -c 'import azula,h5py,numpy,pytest,scipy,torch,wandb; import importlib.metadata; print("azula="+importlib.metadata.version("azula")); print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
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

echo "Running the complete Rocky 9 CPU suite before the B2 GPU smoke"
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

readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_b2_ldm.py"
"${PYTHON}" -u "${ENTRYPOINT}" \
  --mode smoke \
  --seed 1701 \
  --artifact-root "${MODEL_DATA_STAGED}" \
  --codec-checkpoint "${C5P_CODEC}" \
  --codec-sha256 "${C5P_CODEC_SHA}" \
  --output "${RUN_OUTPUT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --b2-manifest "${B2_MANIFEST}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-id "p0b2smoke-${SLURM_JOB_ID}-s1701" \
  --wandb-run-name "b2-ldm-h2-smoke-s1701-j${SLURM_JOB_ID}"

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

export PAPER0_B2_SMOKE_JOB_ROOT="${JOB_ROOT}"
export PAPER0_B2_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B2_SMOKE_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_B2_SMOKE_JOB_ROOT"])
output = root / "b2_ldm_h2_seed_1701"
result = json.loads((output / "result.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())
normalization = json.loads((output / "latent_normalization.json").read_text())
if result["scope"] != "bounded_non_scientific_B2_LDM_H2_GPU_smoke":
    raise RuntimeError("B2 smoke scope differs")
if result["config"]["seed"] != 1701 or result["config"]["arm"] != "B2-LDM-H2":
    raise RuntimeError("B2 smoke identity differs")
if result["completed_epochs"] != 2 or result["completed_optimizer_steps"] != 2:
    raise RuntimeError("B2 smoke budget differs")
if len(range(*result["config"]["train_targets"])) > 16:
    raise RuntimeError("B2 smoke read too many training targets")
if normalization["scientific_authority"] is not False:
    raise RuntimeError("B2 smoke normalization claims scientific authority")
if result["checkpoint_reload_bitwise_exact"] is not True:
    raise RuntimeError("B2 selected-checkpoint reload identity failed")
probe = result["sampler_probe"]
if probe["finite"] is not True or probe["nonzero_latent_diversity"] is not True:
    raise RuntimeError("B2 sampler gate failed")
if probe["ensemble_size"] != 2 or probe["canonical_forecast_shape"] != [1,2,1,5,64,32,88]:
    raise RuntimeError("B2 sampler axes or member count differ")
for flag in (
    "held_out_85606_read",
    "physics_derived_loss_used",
    "target_truth_used_as_model_input",
    "scientific_result",
    "full_B2_training_authorized",
    "probabilistic_scientific_gate_evaluated",
    "O3_launch_allowed",
    "assimilation_allowed",
    "diagnostic_ranking_allowed",
):
    if result[flag] is not False:
        raise RuntimeError(f"B2 smoke scope flag {flag} differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("B2 smoke lacks a finished online W&B run")
if tracking["epochs_logged"] != 2:
    raise RuntimeError("B2 smoke did not log both epochs")
payload = {
    "schema_version": 1,
    "scope": "phase3_B2_LDM_H2_Rocky9_GPU_smoke",
    "paper0_commit": os.environ["PAPER0_B2_SMOKE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_B2_SMOKE_JOB_ID"],
    "development_run": "85604",
    "held_out_85606_read": False,
    "rocky9_cpu_suite_passed": True,
    "one_gpu_smoke_passed": True,
    "allowed_accelerators": ["NVIDIA H100", "NVIDIA H200"],
    "seed": 1701,
    "training_targets": 16,
    "epochs": 2,
    "optimizer_steps": 2,
    "selected_epoch": result["selected_epoch"],
    "selected_validation": result["selected_validation"],
    "sampler_probe": result["sampler_probe"],
    "selected_checkpoint_sha256": result["selected_checkpoint"]["sha256"],
    "wandb_url": tracking["run_url"],
    "scientific_result": False,
    "full_B2_training_authorized": False,
    "probabilistic_evaluation_protocol_still_required": True,
    "O3_launch_allowed": False,
}
temporary = root / ".smoke_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "smoke_summary.json")
PY

sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}/smoke_summary.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Phase 3 B2-LDM-H2 Rocky 9 GPU smoke completed"
echo "This bounded smoke is not a scientific B2 result and does not authorize full training"
