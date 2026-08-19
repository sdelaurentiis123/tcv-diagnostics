#!/bin/bash
# Four-target M32 preflight of the frozen B5 residual-EDM evaluator.
# Submit only after the exact seed-1701 full training job completes.
#SBATCH --job-name=p3_b5_esmk
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b5_eval_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B5_TRAINING_JOB_ID:?submit with B5_TRAINING_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_VALIDATION_SOURCE="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly H1_VALIDATION_SHA="a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b5_eval_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/h1_validation_forecast.h5"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_field_residual_edm_full/job_${B5_TRAINING_JOB_ID}"
readonly TRAINING_OUTPUT="${TRAINING_ROOT}/b5_joint_field_residual_edm_seed_1701"
readonly TRAINING_RESULT="${TRAINING_OUTPUT}/result.json"
readonly THRESHOLD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_6897493"
readonly THRESHOLD_RESULT="${THRESHOLD_ROOT}/thresholds/result.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluator_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/b5_joint_field_residual_edm_seed_1701"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b5_full_training_evaluation_85604.json"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_b5_residual_edm_checkpoint.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b5_residual_edm_evaluation_wandb.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b5-residual-edm-evaluator-smoke-85604"

if [[ -e "${JOB_ROOT}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite a B5 evaluator-smoke or node-local directory" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

cleanup() {
  if [[ "${NODE_LOCAL_ROOT}" == "/tmp/${USER}/tcv_diagnostics_b5_eval_smoke_${SLURM_JOB_ID}" ]]; then
    rm -rf "${NODE_LOCAL_ROOT}"
  fi
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
  echo "B5 evaluator smoke requires Rocky Linux 9" >&2
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

check_sha256 "61f1fa565e2bcff008cbe72909daa97362dabe96d160a9beee4a3d5aa87d1334" "${EVALUATION_MANIFEST}"
check_sha256 "faab336bf3ae1a49008eff0e6604d48d9c475aa83732184668c4c2e444c928b9" "${EVALUATION_PROTOCOL}"
check_sha256 "ed6b40434b76a6326173f0128f369def2b05a5d7c05112b09b4a86eba13eabea" "${EVALUATOR}"
check_sha256 "107e56efc77ea52645312f100cf42ecf3b2fc9c82d8af57774353e7deeef4efe" "${WANDB_WRAPPER}"
check_sha256 "553ba769612d4d54530e2daef573bc3723d22ab7fb1dc95f73dbd46ff211a21b" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_forecast.py"
check_sha256 "e5871d3e444f337e1efe8c5e7030f1e5cefa82610076cbd66bbe08005b24f658" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_scoring.py"
check_sha256 "eeafebf0fb4705ccf004072eaaac43a9b89331afa2c693b822f3766c9b5e955d" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_full_training.py"
check_sha256 "b0d2211ebab290a99c77c0295535669097cf05105b56922d94d4e61a933d6432" "${PAPER0_ROOT}/src/tcv_diagnostics/models/field_residual_edm.py"
check_sha256 "2dfdf6f7b620302826971c9fec4ed8233f46fa1950c8461ed9d79194411178fe" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_scoring.py"
check_sha256 "edef6fbbe7b40348fa450c7428d796f4b5ebc3d9b2070e135c7bb3f58a2b6650" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_probabilistic_metrics.py"
check_sha256 "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_metrics.py"
check_sha256 "382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_spectral_metrics.py"
check_sha256 "b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_transport_metrics.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" "${THRESHOLD_RESULT}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_SOURCE}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${TRAINING_ROOT}/artifact_sha256.txt"
sha256sum -c "${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly TRAINING_COMMIT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["paper0_commit"])' "${TRAINING_RESULT}")"
readonly THRESHOLD_RESULT_SHA="$(sha256sum "${THRESHOLD_RESULT}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_commit=${TRAINING_COMMIT}"
  echo "training_job_id=${B5_TRAINING_JOB_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running complete Rocky 9 suite before B5 evaluator smoke"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

echo "Staging verified 85604 model data and frozen H1 mean"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
cp -a "${H1_VALIDATION_SOURCE}" "${H1_VALIDATION_STAGED}"
check_sha256 "${H1_VALIDATION_SHA}" "${H1_VALIDATION_STAGED}"
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

readonly EVALUATOR_SHA="ed6b40434b76a6326173f0128f369def2b05a5d7c05112b09b4a86eba13eabea"
readonly RUN_ID="p0b5evalsmoke-${SLURM_JOB_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${WANDB_WRAPPER}"
  --evaluator "${EVALUATOR}"
  --evaluator-sha256 "${EVALUATOR_SHA}"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --seed 1701
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "b5-residual-edm-evaluator-smoke-s1701-j${SLURM_JOB_ID}"
  --
  --mode smoke
  --seed 1701
  --training-result "${TRAINING_RESULT}"
  --training-result-sha256 "${TRAINING_RESULT_SHA}"
  --training-commit "${TRAINING_COMMIT}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --h1-validation-forecast "${H1_VALIDATION_STAGED}"
  --h1-validation-forecast-sha256 "${H1_VALIDATION_SHA}"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --event-threshold-result "${THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "${THRESHOLD_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "61f1fa565e2bcff008cbe72909daa97362dabe96d160a9beee4a3d5aa87d1334"
  --evaluation-protocol "${EVALUATION_PROTOCOL}"
  --evaluation-protocol-sha256 "faab336bf3ae1a49008eff0e6604d48d9c475aa83732184668c4c2e444c928b9"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --member-batch-size 8
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

export PAPER0_B5_SMOKE_OUTPUT="${RUN_OUTPUT}"
export PAPER0_B5_SMOKE_ROOT="${JOB_ROOT}"
export PAPER0_B5_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B5_SMOKE_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import write_strict_json_atomic

output = Path(os.environ["PAPER0_B5_SMOKE_OUTPUT"])
root = Path(os.environ["PAPER0_B5_SMOKE_ROOT"])
result = json.loads((output / "result.json").read_text())
generation = json.loads((output / "generation.json").read_text())
score = json.loads((output / "score.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())
if result["scope"] != "bounded_non_scientific_B5_residual_EDM_evaluator_smoke_85604":
    raise RuntimeError("B5 evaluator smoke scope differs")
if result["status"] != "bounded_evaluator_smoke_completed":
    raise RuntimeError("B5 evaluator smoke did not complete")
if result["target_frames"] != [498, 502] or result["ensemble_members"] != 32:
    raise RuntimeError("B5 evaluator smoke size differs")
if generation["scope"] != "bounded_non_scientific_B5_residual_EDM_M32_forecast_smoke_85604":
    raise RuntimeError("B5 smoke generation scope differs")
if score["scope"] != "bounded_non_scientific_B5_residual_EDM_evaluator_smoke_scoring_85604":
    raise RuntimeError("B5 evaluator smoke scoring scope differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("B5 evaluator smoke W&B record differs")
if result["full_evaluation_preconditions_passed"] is not True:
    raise RuntimeError("B5 evaluator smoke preconditions did not pass")
if result["truth_opened_only_after_forecast_hash"] is not True:
    raise RuntimeError("B5 evaluator smoke truth separation differs")
if result["scientific_acceptance_evaluated"] is not False or result["O3_launch_allowed"] is not False:
    raise RuntimeError("B5 evaluator smoke prejudged or authorized the gate")
if result["scientific_sampler_seed_bank"]["sha256"] != "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14":
    raise RuntimeError("B5 scientific sampler seed bank differs")
if generation["inference"]["network_evaluations_per_member"] != 35:
    raise RuntimeError("B5 evaluator sampler NFE differs")

summary = {
    "schema_version": 1,
    "scope": "phase3_B5_residual_EDM_bounded_evaluator_smoke",
    "status": "passed_bounded_evaluator_smoke",
    "scientific_result": False,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_B5_SMOKE_COMMIT"],
    "training_commit": result["training_commit"],
    "slurm_job_id": os.environ["PAPER0_B5_SMOKE_JOB_ID"],
    "target_frames": [498, 502],
    "ensemble_members": 32,
    "truth_opened_only_after_forecast_hash": True,
    "sampler_steps": 18,
    "network_evaluations_per_member": 35,
    "full_evaluation_preconditions_passed": True,
    "scientific_acceptance_evaluated": False,
    "O3_launch_allowed": False,
    "run_result": {
        "path": str(output / "result.json"),
        "sha256": sha256_path(output / "result.json"),
    },
    "forecast": result["forecast"],
    "score": result["score"],
    "scientific_sampler_seed_bank": result["scientific_sampler_seed_bank"],
    "wandb": tracking,
}
write_strict_json_atomic(root / "result.json", summary)
PY

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/stdout.jsonl" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/result.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

sha256sum "${JOB_ROOT}/result.json" "${RUN_OUTPUT}/artifact_sha256.txt"
echo "Bounded four-target B5 residual-EDM evaluator smoke completed"
echo "Full 85604 evaluation may run; the gate, O3, and 85606 remain closed"
