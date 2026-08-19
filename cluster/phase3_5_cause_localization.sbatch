#!/bin/bash
# Frozen 85604-only diagnostic cause localization after residual-KL K4.
# One H100 is used for Gram matrices and frozen-checkpoint inference only.
# No production training, guard read, assimilation, ranking, steering, or 85606 access.
#SBATCH --job-name=p3_5_causes
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_5_cause_localization_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly RUNNER="${PAPER0_ROOT}/paper0/tools/run_phase3_5_cause_localization.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_5_cause_localization"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/analysis"
readonly NODE_SCRATCH="${SLURM_TMPDIR:-/tmp/${USER}_phase3_5_${SLURM_JOB_ID}}"
readonly SCRATCH_ROOT="${NODE_SCRATCH}/cause_localization"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite Phase 3.5 output: ${JOB_ROOT}" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Phase 3.5 commit mismatch: ${PAPER0_COMMIT} != ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Phase 3.5 checkout is dirty" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "Phase 3.5 requires Rocky Linux 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python is unavailable: ${PYTHON}" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Phase 3.5 requires exactly one visible GPU" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100"; then
  echo "Phase 3.5 requires an H100" >&2
  exit 2
fi
if [[ ! -d "$(dirname "${NODE_SCRATCH}")" || ! -w "$(dirname "${NODE_SCRATCH}")" ]]; then
  echo "Writable node-local scratch parent is unavailable" >&2
  exit 2
fi

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${JOB_ROOT}/runtime/matplotlib"
export XDG_CACHE_HOME="${JOB_ROOT}/runtime/cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${SCRATCH_ROOT}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=85604_phase3_5_diagnostic_cause_localization"
  echo "os=Rocky_Linux_9"
  echo "production_model_training=false"
  echo "guard_frames_read=false"
  echo "held_out_85606_read=false"
  echo "assimilation=false"
  echo "diagnostic_ranking=false"
  echo "steering=false"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,matplotlib,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("matplotlib="+matplotlib.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

echo "Running Phase 3.5 scope and known-answer tests before scientific arrays"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_phase3_5_protocol.py \
  tests/test_phase3_5_analysis.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${RUNNER}"
  --output-directory "${RUN_OUTPUT}"
  --scratch-directory "${SCRATCH_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "sdelaurentiis123-columbia-university"
  --wandb-project "tcv-diagnostics-paper0"
  --wandb-group "phase3-5-cause-localization"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" - "${RUN_OUTPUT}" "${PAPER0_COMMIT}" "${SLURM_JOB_ID}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
commit = sys.argv[2]
job_id = sys.argv[3]
required = (
    "stationarity_summary.csv",
    "effective_sample_size.csv",
    "block_transfer_matrix.csv",
    "translation_diagnostics.csv",
    "equivariance_audit.csv",
    "representation_transfer.csv",
    "context_dependence.csv",
    "PHASE3_5_DECISION_MEMO.md",
    "run_manifest.json",
    "figure_captions.md",
    "sufficient_statistics.npz",
)
for name in required:
    path = output / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing Phase 3.5 output: {path}")
for name in required[:7]:
    with (output / name).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        first_data_row = next(reader, None)
        if not header or first_data_row is None:
            raise RuntimeError(f"empty Phase 3.5 CSV: {name}")
manifest = json.loads((output / "run_manifest.json").read_text())
if manifest.get("status") != "completed_phase3_5_without_architecture_training_or_downstream_opening":
    raise RuntimeError("Phase 3.5 manifest status differs")
if manifest.get("paper0_commit") != commit or str(manifest.get("slurm_job_id")) != job_id:
    raise RuntimeError("Phase 3.5 result identity differs")
for key in (
    "held_out_85606_read",
    "guard_frames_read",
    "production_model_training_performed",
    "assimilation_performed",
    "diagnostic_ranking_performed",
):
    if manifest.get(key) is not False:
        raise RuntimeError(f"Phase 3.5 closed-scope flag differs: {key}")
memo_lines = (output / "PHASE3_5_DECISION_MEMO.md").read_text().splitlines()
expected = "A single, fixed, condition-independent, global linear residual distribution learned from adjacent 85604 training frames does not describe later 85604 residuals well."
if not memo_lines or memo_lines[0] != expected:
    raise RuntimeError("Phase 3.5 memo opening differs")
recommendations = [line for line in memo_lines if line.startswith("Recommended next action: ")]
if len(recommendations) != 1 or memo_lines[-1] != recommendations[0]:
    raise RuntimeError("Phase 3.5 memo must end with exactly one recommendation")
figures = list((output / "figures").glob("*.png"))
if len(figures) < 7 or len(list((output / "figures").glob("*.svg"))) != len(figures):
    raise RuntimeError("Phase 3.5 publication figure inventory differs")
PY

find "${RUN_OUTPUT}" -type f -print0 | sort -z | xargs -0 sha256sum > "${JOB_ROOT}/scientific_artifact_sha256.txt"
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/scientific_artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Completed Paper 0 Phase 3.5 at ${RUN_OUTPUT}"
echo "The recommended next action is recorded but not automatically authorized"
