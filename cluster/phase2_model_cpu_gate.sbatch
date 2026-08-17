#!/bin/bash
# Rocky 9 CPU-only implementation gate for the frozen deterministic codec.
# This reads no simulation data, trains no model, and requests no GPU.
#SBATCH --job-name=p2_model_cpu
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_model_cpu_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_cpu_gate"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_o1_o2_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly LAYERS="${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
readonly DCAE="${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
readonly LICENSE="${PAPER0_ROOT}/src/tcv_diagnostics/models/LOLA_LICENSE.txt"
readonly TEST="${PAPER0_ROOT}/tests/test_phase2_dcae.py"

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

echo "Phase 2 deterministic-model CPU gate started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

check_sha256 "6cbbf991e311565ab2cc2d3ae0eb4f1d2572f4c23fb804723c990d5b2401c562" "${MANIFEST}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${LAYERS}"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${DCAE}"
check_sha256 "6a483108d787c61c7e2306216ecaa2e15f80a0a2e7fb44cb70d200bbbab63605" "${LICENSE}"
check_sha256 "d41b174a874da8ffef31f872db3d713d8c0613fa43ece74cf2de2374cf71bc3a" "${TEST}"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda))'
} > "${OUTDIR}/environment.txt" 2>&1

printf '%q ' "${PYTHON}" -m pytest -p no:cacheprovider -q > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

export PAPER0_RESULT="${OUTDIR}/result.json"
export PAPER0_JOB_ID="${SLURM_JOB_ID}"
export PAPER0_HOST="$(hostname --fqdn)"
export PAPER0_FINISHED_AT="$(date --iso-8601=seconds)"
export PAPER0_TEST_OUTPUT="${OUTDIR}/test_output.txt"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

test_output = Path(os.environ["PAPER0_TEST_OUTPUT"])
payload = {
    "schema_version": 1,
    "status": "passed",
    "scope": "deterministic_codec_cpu_implementation_gate",
    "paper0_commit": os.environ["PAPER0_EXPECTED_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_JOB_ID"],
    "host": os.environ["PAPER0_HOST"],
    "finished_at": os.environ["PAPER0_FINISHED_AT"],
    "rocky_major": 9,
    "gpu_requested": False,
    "simulation_data_read": False,
    "training_performed": False,
    "held_out_85606_read": False,
    "test_output_sha256": hashlib.sha256(test_output.read_bytes()).hexdigest(),
}
Path(os.environ["PAPER0_RESULT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
PY

"${PYTHON}" -m json.tool "${OUTDIR}/result.json" >/dev/null
sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/result.json" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 deterministic-model CPU gate finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
echo "No simulation data were read and no training was performed."
