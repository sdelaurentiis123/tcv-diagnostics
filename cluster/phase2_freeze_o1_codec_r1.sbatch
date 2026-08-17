#!/bin/bash
# Freeze all six completed R1 training artifacts before physics evaluation.
# CPU-only; this job reads hashes and metadata but performs no model inference.
#SBATCH --job-name=p2_o1_freeze
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o1_freeze_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_COMMIT="a5adc57499239702a8b28b661357b3dcfcbaa167"
readonly TRAINING_JOB_ID="6893802"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_r1/job_${TRAINING_JOB_ID}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_r1_freeze"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RESULT="${OUTDIR}/frozen_training_matrix.json"
readonly FREEZER="${PAPER0_ROOT}/paper0/tools/freeze_o1_codec_training_matrix.py"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
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
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This artifact freeze requires Rocky 9" >&2
  exit 2
fi
check_sha256 "d326a5b4e77cbca199245b9352833db62612d65406abc13fe47a02d9f237d70b" "${FREEZER}"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"

"${PYTHON}" \
  "${FREEZER}" \
  --job-root "${TRAINING_ROOT}" \
  --training-commit "${TRAINING_COMMIT}" \
  --training-slurm-job-id "${TRAINING_JOB_ID}" \
  --output "${RESULT}" \
  --audit-commit "${PAPER0_COMMIT}" \
  --audit-slurm-job-id "${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
sha256sum "${OUTDIR}/slurm_job.txt" "${RESULT}" > "${OUTDIR}/artifact_sha256.txt"
echo "Frozen R1 training matrix: ${RESULT}"
