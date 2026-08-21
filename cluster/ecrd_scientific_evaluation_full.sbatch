#!/bin/bash
# Matched four-arm, three-seed M32 one-step evaluation on 85604 validation only.
# The historical B5 seed-1701 forecast is reused without mutation.
#SBATCH --job-name=p0_ecrd_eval
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-11%2
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_evaluation_%A_%a.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${ECRD_EVALUATION_FREEZE_JOB_ID:?submit with ECRD_EVALUATION_FREEZE_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_VALIDATION_SOURCE="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly SYM_H1_VALIDATION_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent/job_6913234/parent/sym_h1_validation.h5"
readonly SEED_BANK="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/scientific_sampler_seeds_M32.npy"
readonly HISTORICAL_B5_FORECAST="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/forecast_M32.h5"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVENT_THRESHOLD_RESULT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_6897493/thresholds/result.json"
readonly EVALUATION_FREEZE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_evaluation_freeze/job_${ECRD_EVALUATION_FREEZE_JOB_ID}/freeze"
readonly EVALUATION_MANIFEST="${EVALUATION_FREEZE_ROOT}/ecrd_scientific_evaluation_85604.json"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_ecrd_checkpoint.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_scientific_evaluation_full"

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly ARM="B5"; readonly SAFE_ARM="b5"; readonly MODEL_SEED="1701" ;;
  1) readonly ARM="B5"; readonly SAFE_ARM="b5"; readonly MODEL_SEED="1702" ;;
  2) readonly ARM="B5"; readonly SAFE_ARM="b5"; readonly MODEL_SEED="1703" ;;
  3) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context"; readonly MODEL_SEED="1701" ;;
  4) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context"; readonly MODEL_SEED="1702" ;;
  5) readonly ARM="B5-Context"; readonly SAFE_ARM="b5_context"; readonly MODEL_SEED="1703" ;;
  6) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd"; readonly MODEL_SEED="1701" ;;
  7) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd"; readonly MODEL_SEED="1702" ;;
  8) readonly ARM="ECRD"; readonly SAFE_ARM="ecrd"; readonly MODEL_SEED="1703" ;;
  9) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history"; readonly MODEL_SEED="1701" ;;
  10) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history"; readonly MODEL_SEED="1702" ;;
  11) readonly ARM="ECRD-History"; readonly SAFE_ARM="ecrd_history"; readonly MODEL_SEED="1703" ;;
  *) echo "Unknown ECRD evaluation task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}_${SAFE_ARM}_seed_${MODEL_SEED}"
readonly RUN_OUTPUT="${JOB_ROOT}/evaluation"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_ecrd_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/model_data"
readonly H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/h1_validation.h5"
readonly SYM_H1_VALIDATION_STAGED="${NODE_LOCAL_ROOT}/sym_h1_validation.h5"

if [[ -e "${JOB_ROOT}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD evaluation or node-local directory" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

cleanup() {
  if [[ "${NODE_LOCAL_ROOT}" == "/tmp/${USER}/tcv_ecrd_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" ]]; then
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
  echo "ECRD scientific evaluation requires Rocky Linux 9" >&2
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

check_sha256 "54e7f17787f97e505bef472b0895ca5764649e35ce3f6620213b7008d1714061" "${EVALUATOR}"
check_sha256 "d413cbd79d6c9a11dbfb43801abec7ec2ea9c2231a8759c859880fd2bd44f082" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_forecast.py"
check_sha256 "f941ab30bd16a9a9be362b10f6b10905e212240617b4e7fc9415216803faca5b" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_scoring.py"
check_sha256 "de1744e97f583557b46dcb713ace4c1efe6d84002b3f2583b58b63164c8c671d" "${PAPER0_ROOT}/src/tcv_diagnostics/ecrd_acceptance.py"
sha256sum -c "${EVALUATION_FREEZE_ROOT}/artifact_sha256.txt"
readonly EVALUATION_MANIFEST_SHA="$(sha256sum "${EVALUATION_MANIFEST}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${ECRD_EVALUATION_FREEZE_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/evaluation_freeze_sacct.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
export CUBLAS_WORKSPACE_CONFIG=:4096:8

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "arm=${ARM}"
  echo "model_seed=${MODEL_SEED}"
  echo "evaluation_freeze_job_id=${ECRD_EVALUATION_FREEZE_JOB_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda))'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running complete clean Rocky 9 suite before ECRD scientific evaluation"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

echo "Staging verified 85604 model data and deterministic parents"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
cp -a "${H1_VALIDATION_SOURCE}" "${H1_VALIDATION_STAGED}"
cp -a "${SYM_H1_VALIDATION_SOURCE}" "${SYM_H1_VALIDATION_STAGED}"
check_sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed" "${H1_VALIDATION_STAGED}"
check_sha256 "fd23a42a6603e9150868a7eb0dc5f76580f0044bf1bd0b9e8e01ef04b378fd8d" "${SYM_H1_VALIDATION_STAGED}"
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
  echo "Expected eight verified model-data shards" >&2
  exit 2
fi

COMMON_ARGS=(
  --arm "${ARM}"
  --seed "${MODEL_SEED}"
  --manifest "${EVALUATION_MANIFEST}"
  --manifest-sha256 "${EVALUATION_MANIFEST_SHA}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --h1-validation-parent "${H1_VALIDATION_STAGED}"
  --h1-validation-parent-sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
  --sym-h1-validation-parent "${SYM_H1_VALIDATION_STAGED}"
  --sym-h1-validation-parent-sha256 "fd23a42a6603e9150868a7eb0dc5f76580f0044bf1bd0b9e8e01ef04b378fd8d"
  --seed-bank "${SEED_BANK}"
  --seed-bank-sha256 "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --event-threshold-result "${EVENT_THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb"
  --output "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  --member-batch-size 8
)

if [[ "${ARM}" == "B5" && "${MODEL_SEED}" == "1701" ]]; then
  COMMAND=(
    "${PYTHON}" -u "${EVALUATOR}"
    "${COMMON_ARGS[@]}"
    --historical-b5-forecast "${HISTORICAL_B5_FORECAST}"
    --historical-b5-forecast-sha256 "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
  )
else
  mapfile -t TRAINING_FIELDS < <(
    "${PYTHON}" - "${EVALUATION_MANIFEST}" "${ARM}" "${MODEL_SEED}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
run = manifest["runs"][sys.argv[2]][sys.argv[3]]
print(run["training_result_path"])
print(run["training_result_sha256"])
PY
  )
  if [[ "${#TRAINING_FIELDS[@]}" -ne 2 ]]; then
    echo "ECRD training-result lookup differs" >&2
    exit 2
  fi
  COMMAND=(
    "${PYTHON}" -u "${EVALUATOR}"
    "${COMMON_ARGS[@]}"
    --training-result "${TRAINING_FIELDS[0]}"
    --training-result-sha256 "${TRAINING_FIELDS[1]}"
  )
fi

printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/score.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/evaluation_freeze_sacct.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/job_artifact_sha256.txt"

echo "ECRD ${ARM} seed ${MODEL_SEED} scientific evaluation completed"
