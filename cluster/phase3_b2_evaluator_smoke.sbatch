#!/bin/bash
# Four-target, M32, non-scientific preflight of the exact B2 evaluator.
# Runs only after the three-seed training matrix and thresholds are frozen.
#SBATCH --job-name=p3_b2_esmk
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_eval_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B2_TRAINING_JOB_ID:?submit with B2_TRAINING_JOB_ID set}"
: "${B2_TRAINING_FREEZE_JOB_ID:?submit with B2_TRAINING_FREEZE_JOB_ID set}"
: "${B2_THRESHOLD_JOB_ID:?submit with B2_THRESHOLD_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b2_eval_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly TRAINING_OUTPUT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_full/job_${B2_TRAINING_JOB_ID}/task_0/b2_ldm_h2_seed_1701"
readonly TRAINING_RESULT="${TRAINING_OUTPUT}/result.json"
readonly MATRIX_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_training_freeze/job_${B2_TRAINING_FREEZE_JOB_ID}"
readonly TRAINING_MATRIX="${MATRIX_ROOT}/training_matrix.json"
readonly THRESHOLD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_${B2_THRESHOLD_JOB_ID}"
readonly THRESHOLD_RESULT="${THRESHOLD_ROOT}/thresholds/result.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_evaluator_smoke"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${OUTDIR}/b2_ldm_h2_seed_1701"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b2_full_evaluation_85604.json"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B2_FULL_EVALUATION_PROTOCOL.md"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_b2_checkpoint.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b2_evaluation_wandb.py"
readonly TRAINING_COMMIT="46e2ca07e15c7114aace18202b26a9756489a3f0"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b2-ldm-h2-evaluator-smoke-85604"

if [[ -e "${OUTDIR}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite a B2 evaluator smoke directory" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch" >&2
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
}

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "B2 evaluator smoke requires Rocky Linux 9" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated accelerator is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100|H200"; then
  echo "The evaluator smoke requires H100 or H200" >&2
  exit 2
fi

check_sha256 "a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c" "${EVALUATION_MANIFEST}"
check_sha256 "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490" "${EVALUATION_PROTOCOL}"
check_sha256 "196bc86c489231c529d51aed6b87f068db9af654ff61701b2ec93f5cca6a51d2" "${EVALUATOR}"
check_sha256 "8f88bd4304fcc863cf774b75d4e1eccabb6b8afe18889398cdf3c4c455e502c4" "${WANDB_WRAPPER}"
check_sha256 "b8f4efe71cc12c0593c61886a670312ffb88a71a32f0977ba96a404955db88b8" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_forecast.py"
check_sha256 "002a0027e0d18c4bf99584bfda7f4e399960b02ae6947240f5a6c7513cb2ff90" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_scoring.py"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${MATRIX_ROOT}/artifact_sha256.txt"
sha256sum -c "${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly TRAINING_MATRIX_SHA="$(sha256sum "${TRAINING_MATRIX}" | awk '{print $1}')"
readonly THRESHOLD_RESULT_SHA="$(sha256sum "${THRESHOLD_RESULT}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_job_id=${B2_TRAINING_JOB_ID}"
  echo "training_freeze_job_id=${B2_TRAINING_FREEZE_JOB_ID}"
  echo "threshold_job_id=${B2_THRESHOLD_JOB_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import azula,h5py,numpy,scipy,torch,wandb; print("torch="+torch.__version__); print("wandb="+wandb.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_b2_evaluation_entrypoints.py \
  tests/test_phase3_b2_evaluation_wandb.py \
  tests/test_phase3_b2_forecast.py \
  tests/test_phase3_b2_scoring.py \
  tests/test_phase3_b2_spectral_metrics.py \
  tests/test_phase3_b2_transport_metrics.py \
  | tee "${OUTDIR}/test_output.txt"

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
  echo "Expected eight model-data shards" >&2
  exit 2
fi

readonly EVALUATOR_SHA="196bc86c489231c529d51aed6b87f068db9af654ff61701b2ec93f5cca6a51d2"
readonly RUN_ID="p0b2evalsmoke-${SLURM_JOB_ID}-s1701"
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
  --wandb-run-name "b2-evaluator-smoke-s1701-j${SLURM_JOB_ID}"
  --
  --mode smoke
  --seed 1701
  --training-matrix "${TRAINING_MATRIX}"
  --training-matrix-sha256 "${TRAINING_MATRIX_SHA}"
  --training-result "${TRAINING_RESULT}"
  --training-result-sha256 "${TRAINING_RESULT_SHA}"
  --training-commit "${TRAINING_COMMIT}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --event-threshold-result "${THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "${THRESHOLD_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c"
  --evaluation-protocol "${EVALUATION_PROTOCOL}"
  --evaluation-protocol-sha256 "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --member-batch-size 4
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

export PAPER0_B2_SMOKE_OUTPUT="${RUN_OUTPUT}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["PAPER0_B2_SMOKE_OUTPUT"])
result = json.loads((output / "result.json").read_text())
score = json.loads((output / "score.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())
if result["scope"] != "bounded_non_scientific_B2_evaluator_smoke_85604":
    raise RuntimeError("B2 evaluator smoke scope differs")
if result["status"] != "bounded_evaluator_smoke_completed":
    raise RuntimeError("B2 evaluator smoke did not complete")
if result["target_frames"] != [498, 502] or result["ensemble_members"] != 32:
    raise RuntimeError("B2 evaluator smoke size differs")
if score["scope"] != "bounded_non_scientific_B2_evaluator_smoke_scoring_85604":
    raise RuntimeError("B2 evaluator smoke scoring scope differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("B2 evaluator smoke W&B record differs")
if result["full_probabilistic_evaluation_preconditions_passed"] is not True:
    raise RuntimeError("B2 evaluator smoke preconditions did not pass")
if result["O3_launch_allowed"] is not False:
    raise RuntimeError("B2 evaluator smoke authorized O3")
PY

sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Bounded four-target B2 evaluator smoke completed"
echo "Full evaluation may run; O3 and 85606 remain closed"
