#!/bin/bash
# Phase 2 O1 extension: geometry-aware f8/z44 codec transport on 85604 only.
# Deterministic evaluation only; no training, dynamics, sensors, or blind test.
#SBATCH --job-name=p2_o1_xport
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o1_codec_transport_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly LEGACY_ROOT="/mnt/home/sdelaurentiis/tcv-gaot-3d"
readonly VENV="${LEGACY_ROOT}/.venv-lola"
readonly NATIVE_TRAIN="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/train/TCV_85604_train.hdf5"
readonly NATIVE_VALID="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/valid/TCV_85604_valid.hdf5"
readonly LEGACY_TRAIN="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5"
readonly LEGACY_VALID="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5"
readonly GRID="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly F8_RUN="/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64"
readonly Z44_RUN="/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/z44c6604191_tcv_c5_dcae_3d_tcv_f8z2c64"
readonly PRIOR_O1="${PAPER0_ROOT}/paper0/results/phase2_o1_codec_6890650.json"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_o1_transport_85604.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_O1_TRANSPORT_PROTOCOL.md"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_codec_transport_oracle.py"
readonly CORE="${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
readonly RESULT_BASE="${PAPER0_RESULT_BASE:-/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_transport}"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RESULT="${OUTDIR}/o1_codec_transport.json"

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
  echo "Paper 0 worktree is dirty; refusing provenance-ambiguous execution" >&2
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

echo "Phase 2 O1 codec-transport oracle started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

check_sha256 "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777" "${NATIVE_TRAIN}"
check_sha256 "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba" "${NATIVE_VALID}"
check_sha256 "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817" "${LEGACY_TRAIN}"
check_sha256 "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43" "${LEGACY_VALID}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"
check_sha256 "66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776" "${F8_RUN}/config.yaml"
check_sha256 "9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e" "${F8_RUN}/state.pth"
check_sha256 "5d868c1cfc5a17ce26c2f6ce86ced50d7b55525c6967c5b599b1074058b67284" "${Z44_RUN}/config.yaml"
check_sha256 "095d25f9b6e867103d4cfb946cc9ea8a172a5a6db5b28e5726428c4c57e4979d" "${Z44_RUN}/state.pth"
check_sha256 "957464ad2510ac47bdf882d419db3061fa60d4455d6bc5242bc05baa9a410bf6" "${PRIOR_O1}"

check_sha256 "1191f439f8b58fdb38d570a5ac6b54abed12c06b8455b634c5063d787fe2b357" "${MANIFEST}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "23ddb09a1af8cea58a8dd0d7c54d7728d25f4f72fee7327e865b865ae8896b31" "${PROTOCOL}"
check_sha256 "0f67375309e8c9d12118ab0a8acc5430f26ae1ee2e633f06f96e0140af3f076c" "${EVALUATOR}"
check_sha256 "36e2e80668ccbc121c31e25595a43bc18b02d903d5122f74f11e64e8a8a52678" "${CORE}"
check_sha256 "09e99af7d87f7a0b23433d79bb1f6198ffb49cfd48ee2cf6e1c252beeae5c85a" "${PAPER0_ROOT}/src/tcv_diagnostics/transport.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e" "${PAPER0_ROOT}/src/tcv_diagnostics/resampling.py"
check_sha256 "13f68fbc907eee5f100b7bcb1b7ddb513053eecb4b35684e04af21d859812bb0" "${PAPER0_ROOT}/src/tcv_diagnostics/well.py"
check_sha256 "4338c734e73c1e94ece6b6269f0cbfe2b077d6346d0058243a9734d7844cd28b" "${PAPER0_ROOT}/src/tcv_diagnostics/data_protocol.py"

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
from omegaconf import OmegaConf
from pathlib import Path
import scipy
import sys
import torch

print(f"python={sys.executable}")
print(f"lola={lola.__file__}")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"h5py={h5py.__version__}")
print(f"omegaconf={OmegaConf.__module__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cudnn={torch.backends.cudnn.version()}")
expected = Path("/mnt/home/sdelaurentiis/tcv-gaot-3d/external/lola").resolve()
actual = Path(lola.__file__).resolve()
if expected not in actual.parents:
    raise RuntimeError(f"wrong LOLA import: {actual}")
PY

cd "${PAPER0_ROOT}"
python -m unittest discover -s tests -v

COMMAND=(
  python -u "${EVALUATOR}"
  --manifest "${MANIFEST}"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --native-train-h5 "${NATIVE_TRAIN}"
  --native-valid-h5 "${NATIVE_VALID}"
  --legacy-train-h5 "${LEGACY_TRAIN}"
  --legacy-valid-h5 "${LEGACY_VALID}"
  --geometry "${GRID}"
  --f8-run "${F8_RUN}"
  --z44-run "${Z44_RUN}"
  --prior-o1 "${PRIOR_O1}"
  --chunk-frames 4
  --device cuda
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --command-file "${OUTDIR}/command.sh"
  --output "${RESULT}"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"

"${COMMAND[@]}"
python -m json.tool "${RESULT}" >/dev/null
sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/python_environment.txt" \
  "${RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 O1 codec-transport oracle completed at $(date --iso-8601=seconds)"
echo "Results: ${OUTDIR}"
