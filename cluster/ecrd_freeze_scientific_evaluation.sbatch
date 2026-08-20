#!/bin/bash
# Metadata-only post-training freeze for the 85604 ECRD scientific evaluation.
# Submitted with afterok on the checksum-only training finalizer.
#SBATCH --job-name=p0_ecrd_evalfrz
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_eval_freeze_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_FINALIZER_JOB_ID="6913439"
readonly FINALIZER_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_full_training_finalization/job_${TRAINING_FINALIZER_JOB_ID}/finalization"
readonly FINALIZER_RESULT="${FINALIZER_ROOT}/result.json"
readonly TRAINING_MANIFEST="${PAPER0_ROOT}/paper0/manifests/ecrd_full_training_85604.json"
readonly MODEL_DATA="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_VALIDATION="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly SYM_H1_VALIDATION="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_sym_h1_parent/job_6913234/parent/sym_h1_validation.h5"
readonly SEED_BANK="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/scientific_sampler_seeds_M32.npy"
readonly HISTORICAL_B5_FORECAST="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/forecast_M32.h5"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVENT_THRESHOLD_RESULT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_6897493/thresholds/result.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/freeze_ecrd_scientific_evaluation.py"
readonly ENTRYPOINT_SHA="d266f288c4816f57c266710509812a97cf7dc92c0e9a5ef64d2a1adec6de382c"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_evaluation_freeze"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/freeze"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite an ECRD evaluation-freeze directory" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

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
  echo "ECRD evaluation freeze requires Rocky Linux 9" >&2
  exit 2
fi
check_sha256 "${ENTRYPOINT_SHA}" "${ENTRYPOINT}"
check_sha256 "94100e13a20aa23249aa39f0d910f160ad7aadf60ab42b7e4975969dd3abe6e1" "${TRAINING_MANIFEST}"
sha256sum -c "${FINALIZER_ROOT}/artifact_sha256.txt"
readonly FINALIZER_RESULT_SHA="$(sha256sum "${FINALIZER_RESULT}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${TRAINING_FINALIZER_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/training_finalizer_sacct.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_finalizer_job_id=${TRAINING_FINALIZER_JOB_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import numpy,pytest,torch; print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("torch="+torch.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running complete clean Rocky 9 suite before freezing ECRD evaluation"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --training-finalization-result "${FINALIZER_RESULT}"
  --training-finalization-result-sha256 "${FINALIZER_RESULT_SHA}"
  --training-manifest "${TRAINING_MANIFEST}"
  --training-manifest-sha256 "94100e13a20aa23249aa39f0d910f160ad7aadf60ab42b7e4975969dd3abe6e1"
  --artifact-root "${MODEL_DATA}"
  --h1-validation-parent "${H1_VALIDATION}"
  --h1-validation-parent-sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
  --sym-h1-validation-parent "${SYM_H1_VALIDATION}"
  --sym-h1-validation-parent-sha256 "fd23a42a6603e9150868a7eb0dc5f76580f0044bf1bd0b9e8e01ef04b378fd8d"
  --scientific-seed-bank "${SEED_BANK}"
  --scientific-seed-bank-sha256 "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
  --historical-b5-forecast "${HISTORICAL_B5_FORECAST}"
  --historical-b5-forecast-sha256 "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
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
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/ecrd_scientific_evaluation_85604.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
sha256sum -c "${RUN_OUTPUT}/artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/training_finalizer_sacct.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/job_artifact_sha256.txt"

echo "ECRD 85604 scientific-evaluation manifest frozen"
