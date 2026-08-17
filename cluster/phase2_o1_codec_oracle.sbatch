#!/bin/bash
# Phase 2 O1: deterministic f8/z44 codec reconstruction on 85604 only.
# No dynamics, sensors, filters, training, or sequestered-run access.
#SBATCH --job-name=p2_o1_codec
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o1_codec_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly LEGACY_ROOT="/mnt/home/sdelaurentiis/tcv-gaot-3d"
readonly VENV="${LEGACY_ROOT}/.venv-lola"
readonly TRAIN_H5="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5"
readonly VALID_H5="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5"
readonly F8_RUN="/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64"
readonly Z44_RUN="/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/z44c6604191_tcv_c5_dcae_3d_tcv_f8z2c64"
readonly RESULT_BASE="${PAPER0_RESULT_BASE:-/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec}"
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
  echo "Paper 0 worktree is dirty; refusing provenance-ambiguous run" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: got ${actual}, expected ${expected}" >&2
    exit 2
  fi
  echo "verified ${actual}  ${path}"
}

echo "Phase 2 O1 codec oracle started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

check_sha256 "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817" "${TRAIN_H5}"
check_sha256 "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43" "${VALID_H5}"
check_sha256 "66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776" "${F8_RUN}/config.yaml"
check_sha256 "9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e" "${F8_RUN}/state.pth"
check_sha256 "5d868c1cfc5a17ce26c2f6ce86ced50d7b55525c6967c5b599b1074058b67284" "${Z44_RUN}/config.yaml"
check_sha256 "095d25f9b6e867103d4cfb946cc9ea8a172a5a6db5b28e5726428c4c57e4979d" "${Z44_RUN}/state.pth"
check_sha256 "5c20d880799301a636ff8de67d34d39221d9c5a7e9e0bc2123ae84ee43fc5c83" "${LEGACY_ROOT}/external/lola/lola/autoencoder.py"
check_sha256 "def0e35e3a97a31627415186130b5b8ac6bf69611dc50caacafb73398a706bc5" "${LEGACY_ROOT}/external/lola/lola/data.py"
check_sha256 "a56aef6d1d04c86af91238eeb51d93d345dce5a74481dde5ae3dc244f842691f" "${LEGACY_ROOT}/external/lola/lola/emulation.py"
check_sha256 "281f8541aa09822147f8769e9a11fb63497aa54783dd9a806b173e76c5fbaede" "${LEGACY_ROOT}/external/lola/lola/nn/dcae.py"
check_sha256 "ad6aab36d52ea7aba2a0c45006a33413304c0d9ceb9abffd52a497a24adf616f" "${LEGACY_ROOT}/external/lola/lola/nn/layers.py"

readonly LOLA_PACKAGE_SHA256="$({
  find "${LEGACY_ROOT}/external/lola/lola" -type f -name '*.py' -print0 \
    | sort -z \
    | xargs -0 sha256sum
} | sha256sum | awk '{print $1}')"
if [[ "${LOLA_PACKAGE_SHA256}" != "3fb6e6be7649e86fc0626f5d847adf13649e213c82b543c714ae258332bfdf7d" ]]; then
  echo "LOLA package composite hash mismatch: ${LOLA_PACKAGE_SHA256}" >&2
  exit 2
fi
echo "verified ${LOLA_PACKAGE_SHA256}  ${LEGACY_ROOT}/external/lola/lola/**/*.py"

# Match the audited legacy runtime while forcing both source trees explicitly.
# shellcheck disable=SC1091
source "${LEGACY_ROOT}/env.sh" >/dev/null 2>&1
deactivate 2>/dev/null || true
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${LEGACY_ROOT}/external/lola${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY' > "${OUTDIR}/python_environment.txt"
import h5py
import lola
import numpy
from pathlib import Path
import sys
import torch

print(f"python={sys.executable}")
print(f"lola={lola.__file__}")
print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
print(f"h5py={h5py.__version__}")
expected = Path("/mnt/home/sdelaurentiis/tcv-gaot-3d/external/lola").resolve()
actual = Path(lola.__file__).resolve()
if expected not in actual.parents:
    raise RuntimeError(f"wrong LOLA import: {actual}")
PY

cd "${PAPER0_ROOT}"
python -m unittest discover -s tests -v

COMMAND=(
  python -u "${PAPER0_ROOT}/paper0/tools/evaluate_codec_oracle.py"
  --train-h5 "${TRAIN_H5}"
  --valid-h5 "${VALID_H5}"
  --f8-run "${F8_RUN}"
  --z44-run "${Z44_RUN}"
  --chunk-frames 4
  --device cuda
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --command-file "${OUTDIR}/command.sh"
  --output "${OUTDIR}/o1_codec_metrics.json"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"

"${COMMAND[@]}"
python -m json.tool "${OUTDIR}/o1_codec_metrics.json" >/dev/null
sha256sum "${OUTDIR}/o1_codec_metrics.json" > "${OUTDIR}/result_sha256.txt"

echo "Phase 2 O1 codec oracle completed at $(date --iso-8601=seconds)"
echo "Results: ${OUTDIR}"
