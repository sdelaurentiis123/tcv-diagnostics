#!/bin/bash
# Phase 0 audit-only reproduction: one legacy 85604-valid free/ETKF rollout.
# The direct-state `iter` layout is an oracle/control, not a physical diagnostic.
#SBATCH --job-name=p0_legacy_repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-80gb|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase0_repro_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly LEGACY_ROOT="/mnt/home/sdelaurentiis/tcv-gaot-3d"
readonly VENV="${LEGACY_ROOT}/.venv-lola"
readonly RUN="/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/dm/idllqvqe_tcv_c5_f8c64_vit_small"
readonly AE="/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64"
readonly RAW_BOUT="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/BOUT.dmp.0.nc"
readonly TRAIN_H5="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5"
readonly VALID_H5="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5"
readonly GRID="/mnt/home/sdelaurentiis/ceph/lola_tcv/grid_85604.npz"
readonly RESULT_BASE="${PAPER0_RESULT_BASE:-/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase0_legacy_valid}"
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

echo "Phase 0 legacy reproduction started at $(date --iso-8601=seconds)"
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

check_sha256 "f5ced68fc90bad8fdf073ef7be508002b69aba68e2084d603a6d8f04698c9d38" "${RAW_BOUT}"
check_sha256 "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817" "${TRAIN_H5}"
check_sha256 "18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43" "${VALID_H5}"
check_sha256 "0bdf6ec083c32c9ef5e75ad4bcd6ac44b8dea4280a2b8a6d17c9ea36bdc249a0" "${GRID}"
check_sha256 "5d8e14807b62e81e928628c8413c2563d00884bf339f50e9b221bd758c02e759" "${RUN}/state_best.pth"
check_sha256 "b7ce08657d142cf9d7139798dff0aec847697dd4b198e41d79b6e03ea85a35a5" "${RUN}/config.yaml"
check_sha256 "9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e" "${AE}/state.pth"
check_sha256 "66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776" "${AE}/config.yaml"
check_sha256 "3b70e4eadf3ba1c8ca0d5463a037ebe6fa78da6725ed5dca4331a96aea5df518" "${LEGACY_ROOT}/lola_ext/experiments/da_anchored_rollout.py"
check_sha256 "1009005e60ab99c20941a52b743ce274572318c9031122978e157703bbfb10ad" "${LEGACY_ROOT}/lola_ext/experiments/etkf.py"
check_sha256 "7620a83b0299747418e4d83dceab3d52033f78a36aca25b56000b212f607f669" "${LEGACY_ROOT}/lola_ext/experiments/diagnostic_layouts.py"
check_sha256 "a913e1d4404de20afff0903107c944d21f6b6de3d2c6ed42008086d42f148d21" "${LEGACY_ROOT}/lola_ext/experiments/visualize_probe_data.py"

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

# Match the legacy runtime setup, then force imports to the hashed source tree.
# shellcheck disable=SC1091
source "${LEGACY_ROOT}/env.sh" >/dev/null 2>&1
deactivate 2>/dev/null || true
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${LEGACY_ROOT}/external/lola${PYTHONPATH:+:${PYTHONPATH}}"
python - <<'PY' > "${OUTDIR}/python_environment.txt"
import h5py
import lola
from pathlib import Path
import sys
import torch

print(f"python={sys.executable}")
print(f"lola={lola.__file__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"h5py={h5py.__version__}")
expected = Path("/mnt/home/sdelaurentiis/tcv-gaot-3d/external/lola").resolve()
actual = Path(lola.__file__).resolve()
if expected not in actual.parents:
    raise RuntimeError(f"wrong LOLA import: {actual}")
PY

python "${PAPER0_ROOT}/paper0/tools/audit_85604_data.py" \
  --raw-bout "${RAW_BOUT}" \
  --train-h5 "${TRAIN_H5}" \
  --valid-h5 "${VALID_H5}" \
  --expected-zperiod 5 \
  --output "${OUTDIR}/85604_data_audit.json"

COMMAND=(
  python -u "${LEGACY_ROOT}/lola_ext/experiments/da_anchored_rollout.py"
  "run=${RUN}"
  "target=state_best"
  "grid=${GRID}"
  "split=valid"
  "index=0"
  "start=24"
  "horizon=48"
  "context=1"
  "samples=64"
  "steps=16"
  "algorithm=ab"
  "update=etkf"
  "analysis_mode=filter"
  "layout=iter"
  "assim_every=4"
  "assim_start=0"
  "obs_std=0.05"
  "observation_noise_policy=frame_keyed"
  "inflation=1.0"
  "inflate=off"
  "init_temp=1.0"
  "churn=1.0"
  "seed=0"
  "index_coordinates=state"
  "ixsep=16"
  "edge_x0=16"
  "source_commit=predecessor-b367d8b-src3b70e4e"
  "outdir=${OUTDIR}/da"
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"

cd "${LEGACY_ROOT}/lola_ext/experiments"
"${COMMAND[@]}"

python "${PAPER0_ROOT}/paper0/tools/summarize_da_reproduction.py" \
  --summary "${OUTDIR}/da/da_summary.json" \
  --command-file "${OUTDIR}/command.sh" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --numerical-atol 0.001 \
  --output "${OUTDIR}/reproduction_summary.json"

echo "Phase 0 legacy reproduction completed at $(date --iso-8601=seconds)"
echo "Results: ${OUTDIR}"
