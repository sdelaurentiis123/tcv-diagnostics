#!/bin/bash
# Paired production-size C5P/E6B learnability pilot on old 85604 data.
#SBATCH --job-name=p0_old_state1
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_stage1_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_stage1_pilot.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_codec_free_stage1_pilot.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_stage1_pilot"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-codec-free-stage1-pilot"
readonly SEED=1701

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing Stage-1 pilot output" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: ${PAPER0_COMMIT} != ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous training" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Audited Python environment is unavailable: ${PYTHON}" >&2
  exit 2
fi

source /etc/os-release
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=post_ecrd_old_85604_stage1_codec_free_pilot"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_codec_free_operator.py \
  tests/test_codec_free_stage1_pilot.py \
  tests/test_phase2_model_training_data.py \
  | tee "${JOB_ROOT}/test_output.txt"

readonly MANIFEST_SHA="$(sha256sum "${MANIFEST}" | awk '{print $1}')"
for FAMILY in c5p e6b; do
  RUN_OUTPUT="${JOB_ROOT}/${FAMILY}"
  RUN_ID="p0oldcf-${FAMILY}-${SLURM_JOB_ID}-s${SEED}"
  COMMAND=(
    "${PYTHON}" -u "${ENTRYPOINT}"
    --artifact-root "${ARTIFACT_ROOT}"
    --manifest "${MANIFEST}"
    --manifest-sha256 "${MANIFEST_SHA}"
    --output "${RUN_OUTPUT}"
    --paper0-root "${PAPER0_ROOT}"
    --paper0-commit "${PAPER0_COMMIT}"
    --slurm-job-id "${SLURM_JOB_ID}"
    --family "${FAMILY}"
    --seed "${SEED}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-run-id "${RUN_ID}"
    --wandb-run-name "old-85604-stage1-${FAMILY}-s${SEED}-j${SLURM_JOB_ID}"
  )
  printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command_${FAMILY}.sh"
  printf '\n' >> "${JOB_ROOT}/command_${FAMILY}.sh"
  "${COMMAND[@]}" | tee "${JOB_ROOT}/stdout_${FAMILY}.jsonl"
  "${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
  "${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
done

export PAPER0_STAGE1_JOB_ROOT="${JOB_ROOT}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_STAGE1_JOB_ROOT"])
summary = {}
for family in ("c5p", "e6b"):
    result = json.loads((root / family / "result.json").read_text())
    if result.get("held_out_85606_read") is not False:
        raise RuntimeError(f"{family} held-out flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise RuntimeError(f"{family} physics-loss flag differs")
    summary[family] = {
        "status": result["status"],
        "pilot_gate_passed": result["pilot_gate"]["passed"],
        "best_shared_field_derivative_mse": result["best_checkpoint"][
            "selection_metric"
        ],
    }
(root / "paired_summary.json").write_text(
    json.dumps(
        {
            "scope": "post_ecrd_old_85604_stage1_paired_pilot",
            "held_out_85606_read": False,
            "families": summary,
            "both_pilot_gates_passed": all(
                record["pilot_gate_passed"] for record in summary.values()
            ),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(json.dumps(summary, sort_keys=True))
PY

sha256sum \
  "${JOB_ROOT}/command_c5p.sh" \
  "${JOB_ROOT}/command_e6b.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/paired_summary.json" \
  "${JOB_ROOT}/c5p/result.json" \
  "${JOB_ROOT}/c5p/wandb.json" \
  "${JOB_ROOT}/e6b/result.json" \
  "${JOB_ROOT}/e6b/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Old-85604 Stage-1 paired pilot completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
