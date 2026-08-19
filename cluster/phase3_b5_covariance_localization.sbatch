#!/bin/bash
# One frozen read-only covariance localization of the completed B5 M32 forecast.
# CPU only: no checkpoint load, inference, training, retuning, or 85606 access.
# Online W&B mirrors approved compact metrics and metadata; Ceph stays authoritative.
#SBATCH --job-name=p3_b5_covloc
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b5_covariance_localization_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly B5_FORECAST="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/forecast_M32.h5"
readonly B5_SEED_BANK="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/scientific_sampler_seeds_M32.npy"
readonly H1_VALIDATION_FORECAST="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly B5_SCORE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/score.json"
readonly B5_GATE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_acceptance/job_6901661/gate/final_gate.json"
readonly TRAINING_AUDIT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/residual_audit.json"
readonly TRAINING_RAW="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/raw_accumulators.npz"
readonly H1_TRAINING_FORECAST="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/h1_training_forecast.h5"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly LOCALIZATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b5_covariance_localization_85604.json"
readonly LOCALIZATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B5_COVARIANCE_LOCALIZATION_PROTOCOL.md"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/localize_b5_covariance.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b5_covariance_localization_wandb.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_covariance_localization"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${OUTDIR}/localization"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b5-covariance-localization"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B5 covariance localization: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

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
  echo "B5 covariance localization requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "ca60f75de26e6f6af087d1bb5a0af6ead516c6be9445ea226e6281d18c92a7a0" "${LOCALIZATION_MANIFEST}"
check_sha256 "fa319b8608e3dfc5248f3cfb05e070531409f05be76ca0347c6136a7765f7e8d" "${LOCALIZATION_PROTOCOL}"
check_sha256 "cbb33665577d202dc9cef02fe92193d39c1369e9d7b31dec57a460ed383f58a4" "${EVALUATOR}"
check_sha256 "f63c00545bdb7bd7cdb413aa27be2497c58c987ee1e4d79ed2590408ac69963a" "${WANDB_WRAPPER}"
check_sha256 "f521de9a886f360b9b494c4506895dd32134fc82faf0c753ecbd9dabc730eba1" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_covariance_localization.py"
check_sha256 "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_metrics.py"
check_sha256 "553ba769612d4d54530e2daef573bc3723d22ab7fb1dc95f73dbd46ff211a21b" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_forecast.py"
check_sha256 "eeafebf0fb4705ccf004072eaaac43a9b89331afa2c693b822f3766c9b5e955d" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_edm_full_training.py"
check_sha256 "4e28c00ab6c166256f899b5384fa140dae8a6bf64da73fb56e5183d50118ec53" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_forecast.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "1eba2a8e0c54928264c2c84d5fda08ef6080cf4d73017d40193e3c23e2d2854a" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_o1_transport.py"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "7847e57452020b7fb23f8cfec4c4e7f00f0b6585df51fbac53aa62bd2fe0223e" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_forecast.py"
check_sha256 "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e" "${PAPER0_ROOT}/src/tcv_diagnostics/resampling.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export MPLCONFIGDIR="${OUTDIR}/runtime/matplotlib"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
mkdir -p "${MPLCONFIGDIR}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=read_only_B5_covariance_localization_85604"
  echo "mode_mapping=n=5k"
  echo "training_performed=false"
  echo "model_inference_performed=false"
  echo "held_out_85606_read=false"
  echo "raw_artifacts_uploaded=false"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,matplotlib,numpy,pytest,scipy,wandb; print("h5py="+h5py.__version__); print("matplotlib="+matplotlib.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("wandb="+wandb.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

echo "Running the complete Rocky 9 suite before B5 covariance localization"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

readonly RUN_ID="p0b5covloc-${SLURM_JOB_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${WANDB_WRAPPER}"
  --evaluator "${EVALUATOR}"
  --evaluator-sha256 "cbb33665577d202dc9cef02fe92193d39c1369e9d7b31dec57a460ed383f58a4"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "b5-covariance-localization-j${SLURM_JOB_ID}"
  --
  --localization-manifest "${LOCALIZATION_MANIFEST}"
  --localization-manifest-sha256 "ca60f75de26e6f6af087d1bb5a0af6ead516c6be9445ea226e6281d18c92a7a0"
  --localization-protocol "${LOCALIZATION_PROTOCOL}"
  --localization-protocol-sha256 "fa319b8608e3dfc5248f3cfb05e070531409f05be76ca0347c6136a7765f7e8d"
  --artifact-root "${MODEL_DATA}"
  --b5-forecast "${B5_FORECAST}"
  --b5-forecast-sha256 "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
  --b5-seed-bank "${B5_SEED_BANK}"
  --b5-seed-bank-sha256 "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
  --h1-forecast "${H1_VALIDATION_FORECAST}"
  --h1-forecast-sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
  --b5-score "${B5_SCORE}"
  --b5-score-sha256 "c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6"
  --b5-gate "${B5_GATE}"
  --b5-gate-sha256 "a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b"
  --training-audit "${TRAINING_AUDIT}"
  --training-audit-sha256 "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
  --training-raw "${TRAINING_RAW}"
  --training-raw-sha256 "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b"
  --h1-training-forecast "${H1_TRAINING_FORECAST}"
  --h1-training-forecast-sha256 "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

"${PYTHON}" - "${RUN_OUTPUT}" "${PAPER0_COMMIT}" "${SLURM_JOB_ID}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
commit = sys.argv[2]
job_id = sys.argv[3]
result = json.loads((output / "result.json").read_text())
localization = json.loads((output / "covariance_localization.json").read_text())
wandb = json.loads((output / "wandb.json").read_text())
for record in (result, localization):
    if record.get("paper0_commit") != commit or str(record.get("slurm_job_id")) != job_id:
        raise RuntimeError("B5 localization result identity differs")
if result.get("status") != "completed_without_retraining_or_downstream_opening":
    raise RuntimeError("B5 covariance localization did not complete")
for key in (
    "checkpoint_loaded",
    "model_inference_performed",
    "model_training_performed",
    "forecast_mutated",
    "additional_seed_trained",
    "O3_launched",
    "assimilation_performed",
    "diagnostic_ranking_performed",
    "held_out_85606_read",
):
    if result.get(key) is not False:
        raise RuntimeError(f"B5 localization closed-scope flag differs: {key}")
if localization.get("variogram_scores", {}).get("used_as_training_loss") is not False:
    raise RuntimeError("B5 localization variogram scope differs")
if len(result.get("tables", {})) != 6 or len(result.get("figures", [])) != 12:
    raise RuntimeError("B5 localization output inventory differs")
if wandb.get("mode") != "online" or wandb.get("remote_state_after_finish") != "finished":
    raise RuntimeError("B5 localization W&B record differs")
for key in (
    "checkpoints_uploaded",
    "forecasts_uploaded",
    "simulation_fields_uploaded",
    "raw_accumulators_uploaded",
    "figures_uploaded",
    "tables_uploaded",
):
    if wandb.get(key) is not False:
        raise RuntimeError(f"B5 localization upload boundary differs: {key}")
PY

cp -a "${RUN_OUTPUT}/result.json" "${OUTDIR}/result.json"
sha256sum \
  "${OUTDIR}/result.json" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/covariance_localization.json" \
  "${RUN_OUTPUT}/raw_accumulators.npz" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Completed frozen B5 covariance localization at ${OUTDIR}"
echo "No training, inference, retuning, 85606, O3, assimilation, or ranking occurred"
