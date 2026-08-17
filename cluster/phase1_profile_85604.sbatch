#!/bin/bash
# Phase 1: immutable chronology, stationarity, normalization, and decorrelation.
# This is a CPU-only, read-only data profile of shot 85604.
#SBATCH --job-name=p0_phase1_profile
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase1_profile_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase1_85604_sources.json"
readonly RESULT_BASE="${PAPER0_RESULT_BASE:-/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase1_85604_profile}"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: got ${PAPER0_COMMIT}, expected ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 worktree is dirty; refusing provenance-ambiguous profile" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Required audited Python environment is unavailable: ${PYTHON}" >&2
  exit 2
fi

echo "Phase 1 profile started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" - <<'PY' > "${OUTDIR}/python_environment.txt"
import h5py
import numpy
import platform
import sys

print(f"python_executable={sys.executable}")
print(f"python_version={platform.python_version()}")
print(f"numpy={numpy.__version__}")
print(f"h5py={h5py.__version__}")
PY

COMMAND=(
  "${PYTHON}" -u "${PAPER0_ROOT}/paper0/tools/profile_85604_protocol.py"
  --manifest "${MANIFEST}"
  --expected-commit "${PAPER0_COMMIT}"
  --chunk-frames 8
  --output "${OUTDIR}/phase1_85604_profile.json"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"

"${COMMAND[@]}"

sha256sum \
  "${OUTDIR}/phase1_85604_profile.json" \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/python_environment.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 1 profile completed at $(date --iso-8601=seconds)"
echo "Results: ${OUTDIR}"

