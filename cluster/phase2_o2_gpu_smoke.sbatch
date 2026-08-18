#!/bin/bash
# Rocky 9 one-GPU implementation gate for both frozen C5P O2 history arms.
# This bounded run is non-scientific: <=16 train targets and exactly two epochs.
#SBATCH --job-name=p2_o2_smoke
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
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o2_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_o2_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_gpu_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly CONTINUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_c5p_o2_continuation_85604.json"
readonly CONTINUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md"
readonly R2_RESULT="${PAPER0_ROOT}/paper0/results/phase2_matched_o1_finalize_r2_6894863.json"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly C5P_CODEC="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly C5P_CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="o2-c5p-l10-smoke"

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
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This O2 smoke requires Rocky 9" >&2
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

check_sha256 "7ad5cd207b2b2c69fb82a2b22f65d55e81fad941a508f977966485611ba078d6" "${CONTINUATION_MANIFEST}"
check_sha256 "e969311c0b8012242d44815201f135bd012aa6a9b9c6dc7ca4c6c9043df753eb" "${CONTINUATION_PROTOCOL}"
check_sha256 "ad1cf868d9532869b918ba5892e3926bb06f9961653ac718857b42ac0a182f89" "${R2_RESULT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "8eb062349ab78d8f163c46bb3539e7fd3791cc095ac3b8ed17ae9c000a5be512" "${PAPER0_ROOT}/src/tcv_diagnostics/models/__init__.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb" "${PAPER0_ROOT}/src/tcv_diagnostics/models/o2.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "07c8116afec7a0842b229ecc431a43467788447bb3bfa0ea2efcf51a78908c2e" "${PAPER0_ROOT}/paper0/tools/train_o2.py"
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

echo "Running the complete Rocky 9 CPU suite before GPU smoke"
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

readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_o2.py"
run_smoke() {
  local arm="$1"
  local slug="$2"
  local output="${JOB_ROOT}/${slug}_seed_1701"
  "${PYTHON}" -u "${ENTRYPOINT}" \
    --mode smoke \
    --arm "${arm}" \
    --seed 1701 \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --codec-checkpoint "${C5P_CODEC}" \
    --codec-sha256 "${C5P_CODEC_SHA}" \
    --output "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:${slug}" \
    --continuation-manifest "${CONTINUATION_MANIFEST}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-id "p0o2smoke-${SLURM_JOB_ID}-${slug}" \
    --wandb-run-name "o2-smoke-${slug}-s1701-j${SLURM_JOB_ID}"
  "${PYTHON}" -m json.tool "${output}/result.json" >/dev/null
  "${PYTHON}" -m json.tool "${output}/wandb.json" >/dev/null
  sha256sum \
    "${output}/config.json" \
    "${output}/latent_normalization.json" \
    "${output}/history.jsonl" \
    "${output}/result.json" \
    "${output}/selected.pt" \
    "${output}/final_training_state.pt" \
    "${output}/wandb.json" \
    > "${output}/artifact_sha256.txt"
}

run_smoke "C5P-H1" "c5p_h1"
run_smoke "C5P-H2" "c5p_h2"

export PAPER0_O2_SMOKE_JOB_ROOT="${JOB_ROOT}"
export PAPER0_O2_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_O2_SMOKE_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_O2_SMOKE_JOB_ROOT"])
runs = []
for arm, slug in (("C5P-H1", "c5p_h1"), ("C5P-H2", "c5p_h2")):
    output = root / f"{slug}_seed_1701"
    result = json.loads((output / "result.json").read_text())
    tracking = json.loads((output / "wandb.json").read_text())
    normalization = json.loads((output / "latent_normalization.json").read_text())
    if result["scope"] != "O2_teacher_forced_one_step_smoke":
        raise RuntimeError(f"{arm} smoke scope differs")
    if result["config"]["arm"] != arm or int(result["config"]["seed"]) != 1701:
        raise RuntimeError(f"{arm} smoke identity differs")
    if result["completed_epochs"] != 2 or result["completed_optimizer_steps"] != 2:
        raise RuntimeError(f"{arm} smoke budget differs")
    if len(range(*result["config"]["train_targets"])) > 16:
        raise RuntimeError(f"{arm} smoke read too many training targets")
    if normalization["scientific_authority"] is not False:
        raise RuntimeError(f"{arm} smoke normalization claims authority")
    if result["checkpoint_reload_bitwise_exact"] is not True:
        raise RuntimeError(f"{arm} reload identity failed")
    if result["held_out_85606_read"] is not False:
        raise RuntimeError(f"{arm} touched held-out data")
    if result["physics_derived_loss_used"] is not False:
        raise RuntimeError(f"{arm} used a physics-derived loss")
    if result["target_truth_used_as_model_input"] is not False:
        raise RuntimeError(f"{arm} used target truth as model input")
    if result["O3_launch_allowed"] is not False:
        raise RuntimeError(f"{arm} smoke authorized O3")
    if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
        raise RuntimeError(f"{arm} lacks a finished online W&B run")
    if tracking["epochs_logged"] != 2:
        raise RuntimeError(f"{arm} did not log both epochs")
    runs.append({
        "arm": arm,
        "seed": 1701,
        "output": str(output),
        "selected_epoch": result["selected_epoch"],
        "selected_validation_equal_channel_mae": result["selected_validation_equal_channel_mae"],
        "selected_checkpoint_sha256": result["selected_checkpoint"]["sha256"],
        "wandb_url": tracking["run_url"],
    })
payload = {
    "schema_version": 1,
    "scope": "phase2_C5P_O2_Rocky9_GPU_smoke",
    "paper0_commit": os.environ["PAPER0_O2_SMOKE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_O2_SMOKE_JOB_ID"],
    "development_run": "85604",
    "held_out_85606_read": False,
    "rocky9_cpu_suite_passed": True,
    "one_gpu_smoke_passed": True,
    "allowed_accelerators": ["NVIDIA H100", "NVIDIA H200"],
    "runs": runs,
    "scientific_result": False,
    "full_O2_training_allowed_after_compaction": True,
    "O2_scientific_gate_evaluated": False,
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
  "${JOB_ROOT}"/*/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Phase 2 C5P O2 Rocky 9 GPU smoke completed"
echo "This bounded smoke is not a scientific O2 result"
