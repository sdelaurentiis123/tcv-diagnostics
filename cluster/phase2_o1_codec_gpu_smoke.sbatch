#!/bin/bash
# Bounded, non-scientific GPU smoke for both frozen O1 codec state families.
# Two epochs and at most 16 frames per split; never a model-selection result.
#SBATCH --job-name=p2_codec_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o1_codec_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_gpu_smoke"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_o1_o2_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly LAYERS="${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
readonly DCAE="${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
readonly DATA_READER="${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
readonly TRAINING="${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_codec.py"

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

echo "Phase 2 O1 codec GPU smoke started at $(date --iso-8601=seconds)"
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

check_sha256 "6cbbf991e311565ab2cc2d3ae0eb4f1d2572f4c23fb804723c990d5b2401c562" "${MANIFEST}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${LAYERS}"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${DCAE}"
check_sha256 "e4cf03475e37e808acc21beead82fa7c0849b2857f6bac16f93538611a261ed1" "${DATA_READER}"
check_sha256 "727408e524a1c1b9fbe644327e6fa5ec096e6b7eea4373bc4ee7c6e4d5546f7a" "${TRAINING}"
check_sha256 "ddea107b6d6ed79557e05aa55cb63069441e36fa9510e10ae63673ec23c1607d" "${ENTRYPOINT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_ROOT}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_ROOT}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_ROOT}/artifact_sha256.txt"

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
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version()))'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
} > "${OUTDIR}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

{
  for family in c5p e6b; do
    printf '%q ' \
      "${PYTHON}" -u "${ENTRYPOINT}" \
      --mode smoke \
      --codec dcae_l20 \
      --family "${family}" \
      --seed 1701 \
      --artifact-root "${MODEL_DATA_ROOT}" \
      --output "${OUTDIR}/${family}" \
      --paper0-commit "${PAPER0_COMMIT}" \
      --slurm-job-id "${SLURM_JOB_ID}"
    printf '\n'
  done
} > "${OUTDIR}/commands.sh"

for family in c5p e6b; do
  "${PYTHON}" -u "${ENTRYPOINT}" \
    --mode smoke \
    --codec dcae_l20 \
    --family "${family}" \
    --seed 1701 \
    --artifact-root "${MODEL_DATA_ROOT}" \
    --output "${OUTDIR}/${family}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}"
  "${PYTHON}" -m json.tool "${OUTDIR}/${family}/result.json" >/dev/null
done

export PAPER0_SMOKE_OUTDIR="${OUTDIR}"
export PAPER0_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_SMOKE_JOB="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_SMOKE_OUTDIR"])
runs = {}
for family in ("c5p", "e6b"):
    path = root / family / "result.json"
    result = json.loads(path.read_text())
    config = result["config"]
    if result["scope"] != "O1_codec_smoke":
        raise ValueError("a smoke child has the wrong scope")
    if config["epochs"] > 2:
        raise ValueError("smoke epoch bound was exceeded")
    if config["train_stop"] - config["train_start"] > 16:
        raise ValueError("smoke training-frame bound was exceeded")
    if config["validation_stop"] - config["validation_start"] > 16:
        raise ValueError("smoke validation-frame bound was exceeded")
    if result["held_out_85606_read"] is not False:
        raise ValueError("held-out status differs")
    runs[family] = {
        "result_path": str(path),
        "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selected_validation_equal_channel_mae": result[
            "selected_validation_equal_channel_mae"
        ],
        "checkpoint_reload_bitwise_exact": result[
            "checkpoint_reload_bitwise_exact"
        ],
        "parameter_count": result["parameter_count"],
        "peak_cuda_bytes": result["peak_cuda_bytes"],
        "wall_seconds": result["wall_seconds"],
    }
payload = {
    "schema_version": 1,
    "scope": "bounded_non_scientific_O1_codec_gpu_smoke",
    "paper0_commit": os.environ["PAPER0_SMOKE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_SMOKE_JOB"],
    "rocky_major": 9,
    "codec": "dcae_l20",
    "seed": 1701,
    "runs": runs,
    "training_result_accepted": False,
    "O1_scientific_gate_evaluated": False,
    "held_out_85606_read": False,
}
(root / "smoke_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
)
PY

"${PYTHON}" -m json.tool "${OUTDIR}/smoke_summary.json" >/dev/null
sha256sum \
  "${OUTDIR}/commands.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/c5p/config.json" \
  "${OUTDIR}/c5p/history.jsonl" \
  "${OUTDIR}/c5p/selected.pt" \
  "${OUTDIR}/c5p/final_training_state.pt" \
  "${OUTDIR}/c5p/result.json" \
  "${OUTDIR}/e6b/config.json" \
  "${OUTDIR}/e6b/history.jsonl" \
  "${OUTDIR}/e6b/selected.pt" \
  "${OUTDIR}/e6b/final_training_state.pt" \
  "${OUTDIR}/e6b/result.json" \
  "${OUTDIR}/smoke_summary.json" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 O1 codec GPU smoke finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
echo "This bounded run is an implementation test, not an accepted training result."
