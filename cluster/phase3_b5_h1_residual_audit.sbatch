#!/bin/bash
# One frozen 85604 training-residual audit. This job does not train B5.
# Submit from rusty9 only after the implementation commit and local suite pass.
#SBATCH --job-name=p3_b5_resid
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b5_h1_residual_audit_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b5_residual_audit_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${OUTDIR}/audit"
readonly CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
readonly CODEC_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly LATENT_NORMALIZATION="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/latent_normalization.json"
readonly TRAINING_COMMIT="9035bc3ce9d2351cd17586f4429af8116d43a47e"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly AUDIT_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b5_residual_audit_85604.json"
readonly AUDIT_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B5_RESIDUAL_AUDIT_PROTOCOL.md"
readonly DECORRELATION_RESULT="${PAPER0_ROOT}/paper0/results/phase1_85604_profile_6890606.json"
readonly B4_GATE="${PAPER0_ROOT}/paper0/results/phase3_b4_pde_refiner_one_seed_gate_6901285.json"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/audit_b5_h1_training_residual.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b5_residual_audit_wandb.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b5-h1-residual-audit"

if [[ -e "${OUTDIR}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite a B5 audit or node-local directory" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

cleanup() {
  rm -rf "${NODE_LOCAL_ROOT}"
}
trap cleanup EXIT

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch" >&2
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
  echo "B5 residual audit requires Rocky Linux 9" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated accelerator is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100|H200"; then
  echo "B5 residual audit requires H100 or H200" >&2
  exit 2
fi

check_sha256 "a97ede5bce9e42ac7f94634da26d4824245d3ecdbb2589b9347df5a8149d6cc7" "${AUDIT_MANIFEST}"
check_sha256 "15f6ae16bc68c5dd8c3077812a79b0b1f38efcb11a5ccc89822838d7a81b5f84" "${AUDIT_PROTOCOL}"
check_sha256 "3125a98851b6e9a0873846233f10683e5542d2319839717cdda208828e34637d" "${EVALUATOR}"
check_sha256 "cc522643c14e01a92dcb037e8baf5a17600d48619baec126224706800a78e28e" "${WANDB_WRAPPER}"
check_sha256 "18175d4ae60237a48f161de7e278e0d54756c92280ba56df577f6bb175dbaeb0" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_audit.py"
check_sha256 "4e28c00ab6c166256f899b5384fa140dae8a6bf64da73fb56e5183d50118ec53" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_forecast.py"
check_sha256 "7847e57452020b7fb23f8cfec4c4e7f00f0b6585df51fbac53aa62bd2fe0223e" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_forecast.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb" "${PAPER0_ROOT}/src/tcv_diagnostics/models/o2.py"
check_sha256 "1eba2a8e0c54928264c2c84d5fda08ef6080cf4d73017d40193e3c23e2d2854a" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_o1_transport.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_metrics.py"
check_sha256 "9ef0868a21ebbee883f154f13fe4068d50d47474017cf775ba3d5c3e51b7fc15" "${DECORRELATION_RESULT}"
check_sha256 "4c07a7f4886c14ca2e53d6e322fe309e5efde1f76ab2ed779a3acd14d110f6be" "${B4_GATE}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e" "${CHECKPOINT}"
check_sha256 "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d" "${CODEC_CHECKPOINT}"
check_sha256 "afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738" "${LATENT_NORMALIZATION}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export MPLCONFIGDIR="${NODE_LOCAL_ROOT}/matplotlib"
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
mkdir -p "${MPLCONFIGDIR}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_commit=${TRAINING_COMMIT}"
  echo "scope=85604_training_residual_architecture_sizing_only"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,matplotlib,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("matplotlib="+matplotlib.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

echo "Running the complete Rocky 9 suite before the B5 residual audit"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

echo "Staging and verifying the immutable 85604 model dataset"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
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
  echo "Expected eight verified model-data shards" >&2
  exit 2
fi

readonly RUN_ID="p0b5resid-${SLURM_JOB_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${WANDB_WRAPPER}"
  --evaluator "${EVALUATOR}"
  --evaluator-sha256 "3125a98851b6e9a0873846233f10683e5542d2319839717cdda208828e34637d"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "b5-h1-training-residual-audit-j${SLURM_JOB_ID}"
  --
  --checkpoint "${CHECKPOINT}"
  --checkpoint-sha256 "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
  --codec-checkpoint "${CODEC_CHECKPOINT}"
  --codec-checkpoint-sha256 "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
  --latent-normalization "${LATENT_NORMALIZATION}"
  --latent-normalization-sha256 "afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738"
  --training-commit "${TRAINING_COMMIT}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --audit-manifest "${AUDIT_MANIFEST}"
  --audit-manifest-sha256 "a97ede5bce9e42ac7f94634da26d4824245d3ecdbb2589b9347df5a8149d6cc7"
  --audit-protocol "${AUDIT_PROTOCOL}"
  --audit-protocol-sha256 "15f6ae16bc68c5dd8c3077812a79b0b1f38efcb11a5ccc89822838d7a81b5f84"
  --decorrelation-result "${DECORRELATION_RESULT}"
  --decorrelation-result-sha256 "9ef0868a21ebbee883f154f13fe4068d50d47474017cf775ba3d5c3e51b7fc15"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

"${PYTHON}" - "${RUN_OUTPUT}" "${PAPER0_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
commit = sys.argv[2]
result = json.loads((output / "result.json").read_text())
audit = json.loads((output / "residual_audit.json").read_text())
wandb = json.loads((output / "wandb.json").read_text())
if result.get("paper0_commit") != commit:
    raise RuntimeError("B5 result commit differs")
if result.get("status") != "completed_architecture_sizing_audit_only":
    raise RuntimeError("B5 residual audit did not complete")
for key in ("held_out_85606_read", "guard_frames_read", "validation_frames_read", "training_performed", "B5_training_authorized", "O3_launch_allowed", "assimilation_allowed", "diagnostic_ranking_allowed"):
    if result.get(key) is not False:
        raise RuntimeError(f"B5 closed-scope flag differs: {key}")
if result.get("forecast_closed_and_hashed_before_truth_read") is not True:
    raise RuntimeError("B5 truth-separation gate differs")
if audit.get("canonical_shape") != [430, 5, 64, 32, 88]:
    raise RuntimeError("B5 residual-audit canonical shape differs")
if audit.get("toroidal_support", {}).get("mode_mapping") != "n=5k":
    raise RuntimeError("B5 toroidal mode mapping differs")
if audit.get("scientific_boundaries", {}).get("architecture_selected") is not False:
    raise RuntimeError("B5 audit unexpectedly selected an architecture")
if len(result.get("figures", [])) != 5:
    raise RuntimeError("B5 audit figure inventory differs")
if wandb.get("mode") != "online" or wandb.get("remote_state_after_finish") != "finished":
    raise RuntimeError("B5 audit W&B record differs")
PY

cp -a "${RUN_OUTPUT}/result.json" "${OUTDIR}/result.json"
sha256sum \
  "${OUTDIR}/result.json" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${OUTDIR}/artifact_sha256.txt"
echo "Completed frozen B5 H1 training-residual audit at ${OUTDIR}"
echo "No B5 model was trained; validation, 85606, O3, assimilation, and ranking remain closed"
