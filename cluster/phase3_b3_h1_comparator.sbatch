#!/bin/bash
# Freeze the exact gauge-consistent C5P-H1 parent comparator for B3.
# CPU-only: consumes immutable 85604 artifacts and never opens B3 forecasts or 85606.
#SBATCH --job-name=p3_b3_h1comp
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b3_h1_comparator_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b3_full_evaluation_85604.json"
readonly B2_COMPARATORS="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_deterministic_comparators/job_6897029/comparators/comparators.json"
readonly H1_FORECAST="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
readonly H1_SCORE="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/score.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/build_b3_h1_comparator.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_h1_comparator"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B3 H1 comparator: ${OUTDIR}" >&2
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
}

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "B3 H1 comparator requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "d3b4b1feec8a804b75427087c3f290c6320ed5166c6b533b94da4eab07aeedf9" "${ENTRYPOINT}"
check_sha256 "6a147401cb691746b1bf56ce147a2f72a02b60581a7d4a0fd34a54897fa3bf74" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance.py"
check_sha256 "9bc110067a7124de015202e7eac40030fccda655322f8386dc1d1adca20b9112" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_scoring.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "7847e57452020b7fb23f8cfec4c4e7f00f0b6585df51fbac53aa62bd2fe0223e" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_forecast.py"
check_sha256 "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2" "${EVALUATION_MANIFEST}"
check_sha256 "2e96359cf2213d62ea81c7bec33e30551ade6fd081ca88a1aa088f65d84de72e" "${B2_COMPARATORS}"
check_sha256 "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed" "${H1_FORECAST}"
check_sha256 "ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593" "${H1_SCORE}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_ROOT}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_ROOT}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_ROOT}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_b3_h1_comparator.py \
  tests/test_phase3_b3_full_evaluation_protocol.py \
  | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
  --b2-comparators "${B2_COMPARATORS}"
  --b2-comparators-sha256 "2e96359cf2213d62ea81c7bec33e30551ade6fd081ca88a1aa088f65d84de72e"
  --artifact-root "${MODEL_DATA_ROOT}"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --output-directory "${OUTDIR}/comparators"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

export PAPER0_B3_H1_COMPARATOR_RESULT="${OUTDIR}/comparators/comparators.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["PAPER0_B3_H1_COMPARATOR_RESULT"])
result = json.loads(path.read_text())
if result["scope"] != "phase3_B3_frozen_matched_H1_comparators_85604":
    raise RuntimeError("B3 H1 comparator scope differs")
if result["status"] != "completed_before_B3_scientific_acceptance":
    raise RuntimeError("B3 H1 comparator did not complete")
if result["arm"] != "C5P-H1" or result["seed"] != 1701:
    raise RuntimeError("B3 H1 parent identity differs")
if result["held_out_85606_read"] is not False:
    raise RuntimeError("B3 H1 comparator opened 85606")
if result["B3_forecasts_or_scores_read"] is not False:
    raise RuntimeError("B3 H1 comparator inspected B3 results")
if result["scientific_acceptance_evaluated"] is not False:
    raise RuntimeError("B3 H1 comparator evaluated acceptance")
if result["field"]["context_frames"] != 1:
    raise RuntimeError("B3 deterministic comparator is not matched H1")
if result["field"]["potential_policy"] != "subtract_full_spatial_mean_separately_per_forecast_and_truth_target":
    raise RuntimeError("B3 H1 phi comparator policy differs")
if result["best_uncompressed"]["name"] != "training_only_toroidal_spectral_AR1":
    raise RuntimeError("B3 best uncompressed comparator differs")
if result["O3_launch_allowed"] is not False:
    raise RuntimeError("B3 H1 comparator unexpectedly authorized O3")
PY

sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/stdout.jsonl" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/comparators/comparators.json" \
  "${OUTDIR}/comparators/artifact_sha256.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Frozen gauge-consistent B3 H1 comparator completed"
