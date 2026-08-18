#!/bin/bash
# Freeze all six completed C5P O2 artifacts before scientific evaluation.
# CPU-only: hashes and metadata are read, but no model inference is performed.
#SBATCH --job-name=p2_o2_freeze
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o2_freeze_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_COMMIT="9035bc3ce9d2351cd17586f4429af8116d43a47e"
readonly TRAINING_JOB_ID="6894980"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_full/job_${TRAINING_JOB_ID}"
readonly FULL_RUN_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_c5p_o2_full_runs_85604.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_training_freeze"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RESULT="${OUTDIR}/frozen_training_matrix.json"
readonly FREEZER="${PAPER0_ROOT}/paper0/tools/freeze_o2_training_matrix.py"

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
  echo "verified ${actual}  ${path}"
}

source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "The O2 artifact freeze requires Rocky 9" >&2
  exit 2
fi
check_sha256 "b223f21869d702344d4095949c067dcea8961a15f1831dac69dd808db52185b7" "${FREEZER}"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb" "${PAPER0_ROOT}/src/tcv_diagnostics/models/o2.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "1f49699396a8529a1ee42fd5b0d746c75c98a750c3e376f3720c23dfae4ed203" "${FULL_RUN_MANIFEST}"
check_sha256 "6d539b540a61d35df34d368499f7468f6931342abf512097f5dbd4774f8597df" "${TRAINING_ROOT}/training_summary.json"
check_sha256 "c0336239c94d204328e17a2b05822f210937a5e935e8857d0375664d94e2690a" "${TRAINING_ROOT}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"

"${PYTHON}" \
  "${FREEZER}" \
  --job-root "${TRAINING_ROOT}" \
  --training-commit "${TRAINING_COMMIT}" \
  --training-slurm-job-id "${TRAINING_JOB_ID}" \
  --full-run-manifest "${FULL_RUN_MANIFEST}" \
  --output "${RESULT}" \
  --audit-commit "${PAPER0_COMMIT}" \
  --audit-slurm-job-id "${SLURM_JOB_ID}"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
sha256sum "${OUTDIR}/slurm_job.txt" "${RESULT}" > "${OUTDIR}/artifact_sha256.txt"
echo "Frozen O2 training matrix: ${RESULT}"
echo "O2 remains scientifically undecided pending separate evaluation"
