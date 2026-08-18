#!/bin/bash
# Freeze gauge-consistent paired C5P-H2 comparators before B2 acceptance.
# CPU-only: consumes immutable 85604 H2 forecasts and never opens 85606.
#SBATCH --job-name=p3_b2_comp
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_comparators_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly O2_FINAL_MATRIX="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/final_matrix.json"
readonly O2_AUDIT="${PAPER0_ROOT}/paper0/results/phase2_o2_evaluation_full_6896117.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/build_b2_deterministic_comparators.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_deterministic_comparators"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite B2 deterministic comparators: ${OUTDIR}" >&2
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
  echo "B2 deterministic comparator freeze requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "ad046ba02607027becaeaa7f30dca64702c115288ed004f4bc6ec2747d66f53c" "${ENTRYPOINT}"
check_sha256 "6a147401cb691746b1bf56ce147a2f72a02b60581a7d4a0fd34a54897fa3bf74" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_acceptance.py"
check_sha256 "a60299ec59400910198b6ffd64aa99412c7658b9b773e6a7908cdc03a0b73821" "${O2_FINAL_MATRIX}"
check_sha256 "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb" "${O2_AUDIT}"
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
  tests/test_phase3_b2_acceptance.py \
  tests/test_phase3_b2_evaluation_entrypoints.py \
  | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --o2-final-matrix "${O2_FINAL_MATRIX}"
  --o2-final-matrix-sha256 "a60299ec59400910198b6ffd64aa99412c7658b9b773e6a7908cdc03a0b73821"
  --o2-audit-result "${O2_AUDIT}"
  --o2-audit-result-sha256 "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb"
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

export PAPER0_B2_COMPARATOR_RESULT="${OUTDIR}/comparators/comparators.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["PAPER0_B2_COMPARATOR_RESULT"])
result = json.loads(path.read_text())
if result["scope"] != "phase3_B2_frozen_paired_deterministic_comparators_85604":
    raise RuntimeError("B2 deterministic comparator scope differs")
if result["status"] != "completed_before_B2_scientific_acceptance":
    raise RuntimeError("B2 deterministic comparator did not complete")
if result["seeds"] != [1701, 1702, 1703] or result["seed_count"] != 3:
    raise RuntimeError("B2 deterministic comparator seed matrix differs")
if result["held_out_85606_read"] is not False:
    raise RuntimeError("B2 deterministic comparator opened 85606")
if result["B2_forecasts_or_scores_read"] is not False:
    raise RuntimeError("B2 deterministic comparator inspected B2 results")
if result["scientific_acceptance_evaluated"] is not False:
    raise RuntimeError("B2 deterministic comparator evaluated acceptance")
if result["best_uncompressed"]["name"] != "training_only_toroidal_spectral_AR1":
    raise RuntimeError("B2 best uncompressed comparator differs")
for run in result["runs"]:
    if run["field"]["potential_policy"] != "subtract_full_spatial_mean_separately_per_forecast_and_truth_target":
        raise RuntimeError("B2 deterministic phi comparator policy differs")
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

echo "Frozen gauge-consistent B2 deterministic comparators completed"
