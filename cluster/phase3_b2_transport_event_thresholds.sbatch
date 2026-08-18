#!/bin/bash
# Frozen training-only transport event thresholds for B2 evaluation.
# CPU-only, 85604-only, and deliberately independent of B2 forecast values.
#SBATCH --job-name=p3_b2_threshold
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b2_threshold_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_MATERIALITY="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_evaluation_full/job_6896117/references/training_materiality.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/build_b2_transport_event_thresholds.py"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing threshold directory: ${OUTDIR}" >&2
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
  echo "B2 thresholds require Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

check_sha256 "2c6fc52d58c38e2be84ab1c69ba10314db94e579532fb878ecf97dd953464da1" "${ENTRYPOINT}"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "2dfdf6f7b620302826971c9fec4ed8233f46fa1950c8461ed9d79194411178fe" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_scoring.py"
check_sha256 "b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_transport_metrics.py"
check_sha256 "1eba2a8e0c54928264c2c84d5fda08ef6080cf4d73017d40193e3c23e2d2854a" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_o1_transport.py"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e" "${PAPER0_ROOT}/src/tcv_diagnostics/resampling.py"
check_sha256 "f76d27df3fdebcd75114401d3747dc6380584ad870cd5caaffa5ca4ce40c662f" "${TRAINING_MATERIALITY}"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "os_id=${ID}"
  echo "os_version=${VERSION_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,scipy; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

echo "Running the complete Rocky 9 suite before freezing event thresholds"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --training-materiality "${TRAINING_MATERIALITY}"
  --training-materiality-sha256 "f76d27df3fdebcd75114401d3747dc6380584ad870cd5caaffa5ca4ce40c662f"
  --output-directory "${OUTDIR}/thresholds"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

"${PYTHON}" -m json.tool "${OUTDIR}/thresholds/result.json" >/dev/null
sha256sum -c "${OUTDIR}/thresholds/artifact_sha256.txt"
sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/thresholds/artifact_sha256.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Frozen B2 training-only event thresholds completed"
echo "Validation truth and held-out 85606 were not read"
