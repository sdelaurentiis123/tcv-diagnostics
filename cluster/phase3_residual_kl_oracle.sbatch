#!/bin/bash
# One frozen 85604 residual-KL representation oracle and static covariance baseline.
# CPU only: no checkpoint, model inference, training, 85606, O3/O4/O5, or assimilation.
# W&B receives compact monitoring only; all scientific arrays stay on Ceph.
#SBATCH --job-name=p3_reskl
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_residual_kl_oracle_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_TRAINING_FORECAST="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/h1_training_forecast.h5"
readonly H1_VALIDATION_FORECAST="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly TRAINING_AUDIT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/residual_audit.json"
readonly TRAINING_RAW="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/job_6901393/audit/raw_accumulators.npz"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly B5_LOCALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase3_b5_covariance_localization_6901914.json"
readonly ORACLE_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_residual_kl_oracle_85604.json"
readonly ORACLE_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_RESIDUAL_KL_ORACLE_PROTOCOL.md"
readonly DECISION_MEMO="${PAPER0_ROOT}/paper0/PHASE3_POST_LOCALIZATION_DECISION.md"
readonly BUILDER="${PAPER0_ROOT}/paper0/tools/build_residual_kl_pretruth.py"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_residual_kl_oracle.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_residual_kl_oracle_wandb.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_residual_kl_oracle"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly PRETRUTH_OUTPUT="${OUTDIR}/pretruth"
readonly EVALUATION_OUTPUT="${OUTDIR}/evaluation"
readonly NODE_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
readonly SCRATCH_ROOT="${NODE_SCRATCH}/paper0_residual_kl_${SLURM_JOB_ID}"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="residual-kl-oracle"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite residual-KL output: ${OUTDIR}" >&2
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
  echo "Residual-KL oracle requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${NODE_SCRATCH}" || ! -w "${NODE_SCRATCH}" ]]; then
  echo "Writable node scratch is required: ${NODE_SCRATCH}" >&2
  exit 2
fi

check_sha256 "742ed3bbdafca1949baba67af19840e83f9e8c28fb9efd93cdee53beef7969bb" "${DECISION_MEMO}"
check_sha256 "3e1006e52793e612a0daaf21c67e9da2298fc83bfa542af1be4ba376a6acaff7" "${ORACLE_PROTOCOL}"
check_sha256 "9255699ed902b314cdc27b9d252d1df2fcff794866299ca6dc8708d9671bf575" "${ORACLE_MANIFEST}"
check_sha256 "325c3df054fd963df45353690525e1a1c7f50b7b8a6b2126bcf99c6218df1d01" "${BUILDER}"
check_sha256 "d4aadb22fd05db86192b2093cdfee338937ed1db7f34f07b187c14624073c755" "${EVALUATOR}"
check_sha256 "17d07516c39cdb73abb53de4c65eb69d66b04f7a9a194bd2a32b4603a25c6742" "${WANDB_WRAPPER}"
check_sha256 "c1ea4007281ccba484f8e0850add3ee5f5c37cabf070716524c967723cba93cd" "${PAPER0_ROOT}/src/tcv_diagnostics/residual_kl_oracle.py"
check_sha256 "a11f5566af52dd52382743d2f2b3717e95278c894a06bdb3c674cc0d29764f35" "${PAPER0_ROOT}/src/tcv_diagnostics/residual_kl_metrics.py"
check_sha256 "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_metrics.py"
check_sha256 "9bc110067a7124de015202e7eac40030fccda655322f8386dc1d1adca20b9112" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_scoring.py"
check_sha256 "b8f4efe71cc12c0593c61886a670312ffb88a71a32f0977ba96a404955db88b8" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_forecast.py"
check_sha256 "382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_spectral_metrics.py"
check_sha256 "f521de9a886f360b9b494c4506895dd32134fc82faf0c753ecbd9dabc730eba1" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_covariance_localization.py"
check_sha256 "18175d4ae60237a48f161de7e278e0d54756c92280ba56df577f6bb175dbaeb0" "${PAPER0_ROOT}/src/tcv_diagnostics/b5_residual_audit.py"
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
check_sha256 "edef6fbbe7b40348fa450c7428d796f4b5ebc3d9b2070e135c7bb3f58a2b6650" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_probabilistic_metrics.py"
check_sha256 "0a7540a3e1b9698dea2293e729fd1f956986924cb3be6cc15a546add98521b64" "${PAPER0_ROOT}/src/tcv_diagnostics/metrics.py"

check_sha256 "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18" "${H1_TRAINING_FORECAST}"
check_sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed" "${H1_VALIDATION_FORECAST}"
check_sha256 "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67" "${TRAINING_AUDIT}"
check_sha256 "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b" "${TRAINING_RAW}"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2" "${B5_LOCALIZATION_RESULT}"
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
export XDG_CACHE_HOME="${OUTDIR}/runtime/cache"
export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=residual_KL_oracle_and_static_covariance_85604"
  echo "os=Rocky_Linux_9"
  echo "mode_mapping=n=5k"
  echo "training_performed=false"
  echo "model_inference_performed=false"
  echo "held_out_85606_read=false"
  echo "raw_artifacts_uploaded=false"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,matplotlib,numpy,pytest,scipy,wandb; print("h5py="+h5py.__version__); print("matplotlib="+matplotlib.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("wandb="+wandb.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

echo "Running the complete Rocky 9 suite before opening residual-KL scientific data"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

readonly RUN_ID="p0reskl-${SLURM_JOB_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${WANDB_WRAPPER}"
  --builder "${BUILDER}"
  --builder-sha256 "325c3df054fd963df45353690525e1a1c7f50b7b8a6b2126bcf99c6218df1d01"
  --evaluator "${EVALUATOR}"
  --evaluator-sha256 "d4aadb22fd05db86192b2093cdfee338937ed1db7f34f07b187c14624073c755"
  --oracle-manifest "${ORACLE_MANIFEST}"
  --oracle-manifest-sha256 "9255699ed902b314cdc27b9d252d1df2fcff794866299ca6dc8708d9671bf575"
  --oracle-protocol "${ORACLE_PROTOCOL}"
  --oracle-protocol-sha256 "3e1006e52793e612a0daaf21c67e9da2298fc83bfa542af1be4ba376a6acaff7"
  --decision-memo "${DECISION_MEMO}"
  --decision-memo-sha256 "742ed3bbdafca1949baba67af19840e83f9e8c28fb9efd93cdee53beef7969bb"
  --artifact-root "${MODEL_DATA}"
  --h1-training-forecast "${H1_TRAINING_FORECAST}"
  --h1-training-forecast-sha256 "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
  --h1-validation-forecast "${H1_VALIDATION_FORECAST}"
  --h1-validation-forecast-sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
  --training-audit "${TRAINING_AUDIT}"
  --training-audit-sha256 "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
  --training-raw "${TRAINING_RAW}"
  --training-raw-sha256 "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --b5-localization-result "${B5_LOCALIZATION_RESULT}"
  --b5-localization-result-sha256 "331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2"
  --pretruth-output "${PRETRUTH_OUTPUT}"
  --evaluation-output "${EVALUATION_OUTPUT}"
  --scratch-root "${SCRATCH_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "residual-kl-oracle-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

"${PYTHON}" - "${PRETRUTH_OUTPUT}" "${EVALUATION_OUTPUT}" "${PAPER0_COMMIT}" "${SLURM_JOB_ID}" <<'PY'
import json
import sys
from pathlib import Path

pretruth_dir = Path(sys.argv[1])
evaluation_dir = Path(sys.argv[2])
commit = sys.argv[3]
job_id = sys.argv[4]
pretruth = json.loads((pretruth_dir / "pretruth_result.json").read_text())
result = json.loads((evaluation_dir / "result.json").read_text())
scientific = json.loads((evaluation_dir / "residual_kl_oracle.json").read_text())
wandb = json.loads((evaluation_dir / "wandb.json").read_text())
for record in (pretruth, result, scientific):
    if record.get("paper0_commit") != commit or str(record.get("slurm_job_id")) != job_id:
        raise RuntimeError("residual-KL result identity differs")
if pretruth.get("status") != "completed_and_closed_before_validation_truth":
    raise RuntimeError("residual-KL pretruth did not close")
if result.get("status") != "completed_without_model_training_or_downstream_opening":
    raise RuntimeError("residual-KL evaluation did not complete")
if result.get("primary_outcome") not in {
    "K1_compact_representation_static_covariance_useful",
    "K2_compact_representation_conditional_coefficients_required",
    "K3_only_moderate_or_high_rank_adequate",
    "K4_training_residual_span_does_not_transfer",
    "inconsistent_diagnostic_requires_review",
}:
    raise RuntimeError("residual-KL scientific outcome differs")
for key in (
    "checkpoint_loaded",
    "model_inference_performed",
    "optimizer_or_trainable_parameter_created",
    "model_training_performed",
    "physics_metric_used_as_training_loss",
    "O3_launched",
    "assimilation_performed",
    "diagnostic_ranking_performed",
):
    if result.get(key) is not False:
        raise RuntimeError(f"residual-KL closed-scope flag differs: {key}")
if wandb.get("mode") != "online" or wandb.get("remote_state_after_finish") != "finished":
    raise RuntimeError("residual-KL W&B record differs")
for key in (
    "checkpoints_uploaded",
    "forecasts_uploaded",
    "simulation_fields_uploaded",
    "basis_arrays_uploaded",
    "raw_accumulators_uploaded",
    "figures_uploaded",
    "tables_uploaded",
):
    if wandb.get(key) is not False:
        raise RuntimeError(f"residual-KL upload boundary differs: {key}")
PY

cp -a "${EVALUATION_OUTPUT}/result.json" "${OUTDIR}/result.json"
sha256sum \
  "${OUTDIR}/result.json" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${PRETRUTH_OUTPUT}/artifact_sha256.txt" \
  "${PRETRUTH_OUTPUT}/pretruth_result.json" \
  "${PRETRUTH_OUTPUT}/pretruth_closure.json" \
  "${PRETRUTH_OUTPUT}/training_kl_summary.json" \
  "${PRETRUTH_OUTPUT}/training_kl_basis.h5" \
  "${PRETRUTH_OUTPUT}/static_seed_bank.npy" \
  "${PRETRUTH_OUTPUT}/compressed_forecast_manifest.json" \
  "${EVALUATION_OUTPUT}/artifact_sha256.txt" \
  "${EVALUATION_OUTPUT}/result.json" \
  "${EVALUATION_OUTPUT}/residual_kl_oracle.json" \
  "${EVALUATION_OUTPUT}/raw_sufficient_statistics.npz" \
  "${EVALUATION_OUTPUT}/wandb.json" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Completed frozen residual-KL oracle at ${OUTDIR}"
echo "No checkpoint, inference, training, 85606, O3/O4/O5, assimilation, or ranking occurred"
