#!/bin/bash
# Full R1 O1 codec training: six logical runs on one nonpreemptible H100 node.
# gpuxl requires four GPUs, so runs 0..3 execute first and runs 4..5 second.
# This job trains codecs only; it cannot accept R1 or advance to O2.
#SBATCH --job-name=p2_o1_r1_codec
#SBATCH --partition=gpuxl
#SBATCH --qos=gpuxl
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o1_r1_codec_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"
: "${SLURM_TMPDIR:?Slurm node-local storage is required}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly MODEL_DATA_STAGED="${SLURM_TMPDIR}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_r1"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_o1_codec_r1_runs.json"
readonly MATCHED_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_o1_o2_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly LAYERS="${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
readonly DCAE="${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
readonly DATA_READER="${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
readonly TRAINING="${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
readonly TRACKING="${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_codec.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="o1-dcae-l20-r1"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing result directory: ${JOB_ROOT}" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

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

echo "Phase 2 O1 full R1 codec job started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 4 ]]; then
  echo "Exactly four allocated H100 GPUs are required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -vq "H100"; then
  echo "A non-H100 accelerator was allocated" >&2
  exit 2
fi

check_sha256 "724c7c1f5da11acbd64d4eff5a99fcc2d6a24ef1dda9ebe75243fb20a254ef05" "${RUN_MANIFEST}"
check_sha256 "6cbbf991e311565ab2cc2d3ae0eb4f1d2572f4c23fb804723c990d5b2401c562" "${MATCHED_MANIFEST}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${LAYERS}"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${DCAE}"
check_sha256 "e4cf03475e37e808acc21beead82fa7c0849b2857f6bac16f93538611a261ed1" "${DATA_READER}"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${TRAINING}"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${TRACKING}"
check_sha256 "d902eb470826d86369fe2791f7762d4ff2b198717d66c85d18549d820a3b2c5b" "${ENTRYPOINT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export TOKENIZERS_PARALLELISM=false

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PAPER0_WANDB_ENTITY="${WANDB_ENTITY}"
"${PYTHON}" - <<'PY' | tee "${JOB_ROOT}/wandb_preflight.json"
import json
import os
import wandb

api = wandb.Api(timeout=30)
viewer = api.viewer
entity = str(getattr(viewer, "entity", ""))
if not api.api_key:
    raise RuntimeError("W&B API key is absent")
if entity != os.environ["PAPER0_WANDB_ENTITY"]:
    raise RuntimeError(f"W&B entity {entity!r} differs")
print(json.dumps({
    "authenticated": True,
    "entity": entity,
    "username": str(getattr(viewer, "username", "")),
    "wandb_version": wandb.__version__,
}, sort_keys=True))
PY

echo "Staging verified 85604 model data to ${MODEL_DATA_STAGED}"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_STAGED}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_STAGED}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_STAGED}/artifact_sha256.txt"
verified_shards=0
while read -r expected source_path; do
  case "${source_path}" in
    "${MODEL_DATA_SOURCE}"/shards/*.h5)
      relative="${source_path#${MODEL_DATA_SOURCE}/}"
      check_sha256 "${expected}" "${MODEL_DATA_STAGED}/${relative}"
      verified_shards=$((verified_shards + 1))
      ;;
  esac
done < "${MODEL_DATA_SOURCE}/artifact_sha256.txt"
if [[ "${verified_shards}" -ne 8 ]]; then
  echo "Expected eight verified dataset shards, got ${verified_shards}" >&2
  exit 2
fi

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

families=(c5p e6b c5p e6b c5p e6b)
seeds=(1701 1701 1702 1702 1703 1703)
declare -a pids

task_output() {
  local run_index="$1"
  echo "${JOB_ROOT}/task_${run_index}_${families[run_index]}_seed_${seeds[run_index]}"
}

write_command() {
  local run_index="$1"
  local gpu_index="$2"
  local family="${families[run_index]}"
  local seed="${seeds[run_index]}"
  local output
  output="$(task_output "${run_index}")"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_index}"
  printf '%q ' \
    "${PYTHON}" -u "${ENTRYPOINT}" \
    --mode full \
    --codec dcae_l20 \
    --family "${family}" \
    --seed "${seed}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --output "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}" \
    --run-manifest "${RUN_MANIFEST}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-id "p0o1r1-${SLURM_JOB_ID}-${run_index}" \
    --wandb-run-name "o1-r1-${family}-s${seed}-j${SLURM_JOB_ID}-t${run_index}"
  printf '\n'
}

{
  for run_index in 0 1 2 3; do
    write_command "${run_index}" "${run_index}"
  done
  write_command 4 0
  write_command 5 1
} > "${JOB_ROOT}/commands.sh"

run_task() {
  local run_index="$1"
  local gpu_index="$2"
  local family="${families[run_index]}"
  local seed="${seeds[run_index]}"
  local output
  output="$(task_output "${run_index}")"
  export CUDA_VISIBLE_DEVICES="${gpu_index}"
  export OMP_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  echo "Starting logical run ${run_index}: family=${family}, seed=${seed}, GPU=${gpu_index}"
  "${PYTHON}" -u "${ENTRYPOINT}" \
    --mode full \
    --codec dcae_l20 \
    --family "${family}" \
    --seed "${seed}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --output "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}" \
    --run-manifest "${RUN_MANIFEST}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-id "p0o1r1-${SLURM_JOB_ID}-${run_index}" \
    --wandb-run-name "o1-r1-${family}-s${seed}-j${SLURM_JOB_ID}-t${run_index}"
  "${PYTHON}" -m json.tool "${output}/result.json" >/dev/null
  "${PYTHON}" -m json.tool "${output}/wandb.json" >/dev/null
  sha256sum \
    "${output}/config.json" \
    "${output}/history.jsonl" \
    "${output}/result.json" \
    "${output}/selected.pt" \
    "${output}/final_training_state.pt" \
    "${output}/wandb.json" \
    > "${output}/artifact_sha256.txt"
  echo "Completed logical run ${run_index}: family=${family}, seed=${seed}"
}

launch_task() {
  local run_index="$1"
  local gpu_index="$2"
  local task_log="${JOB_ROOT}/task_${run_index}_${families[run_index]}_seed_${seeds[run_index]}.log"
  run_task "${run_index}" "${gpu_index}" > >(tee -a "${task_log}") 2>&1 &
  pids[run_index]=$!
}

wait_wave() {
  local failed=0
  local run_index
  for run_index in "$@"; do
    if ! wait "${pids[run_index]}"; then
      echo "Logical run ${run_index} failed" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one logical run failed; later waves will not launch" >&2
    exit 1
  fi
}

echo "Launching wave 0: runs 0, 1, 2, 3"
for run_index in 0 1 2 3; do
  launch_task "${run_index}" "${run_index}"
done
wait_wave 0 1 2 3

echo "Launching wave 1: runs 4, 5"
launch_task 4 0
launch_task 5 1
wait_wave 4 5

export PAPER0_R1_JOB_ROOT="${JOB_ROOT}"
export PAPER0_R1_COMMIT="${PAPER0_COMMIT}"
export PAPER0_R1_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_R1_JOB_ROOT"])
families = ("c5p", "e6b", "c5p", "e6b", "c5p", "e6b")
seeds = (1701, 1701, 1702, 1702, 1703, 1703)
runs = []
for index, (family, seed) in enumerate(zip(families, seeds)):
    output = root / f"task_{index}_{family}_seed_{seed}"
    result = json.loads((output / "result.json").read_text())
    tracking = json.loads((output / "wandb.json").read_text())
    if result["scope"] != "O1_codec_full":
        raise ValueError(f"run {index} has the wrong scope")
    if result["completed_epochs"] != 200:
        raise ValueError(f"run {index} did not complete 200 epochs")
    if result["held_out_85606_read"] is not False:
        raise ValueError(f"run {index} touched the held-out run")
    if result["physics_derived_loss_used"] is not False:
        raise ValueError(f"run {index} used a physics-derived loss")
    if not result["checkpoint_reload_bitwise_exact"]:
        raise ValueError(f"run {index} failed checkpoint reload identity")
    if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
        raise ValueError(f"run {index} lacks a finished online W&B record")
    if tracking["epochs_logged"] != 200:
        raise ValueError(f"run {index} did not mirror every epoch to W&B")
    runs.append({
        "run_index": index,
        "family": family,
        "seed": seed,
        "output": str(output),
        "selected_epoch": result["selected_epoch"],
        "selected_validation_equal_channel_mae": result[
            "selected_validation_equal_channel_mae"
        ],
        "checkpoint_sha256": result["selected_checkpoint"]["sha256"],
        "wandb_url": tracking["run_url"],
    })
payload = {
    "schema_version": 1,
    "scope": "phase2_O1_R1_full_codec_training",
    "paper0_commit": os.environ["PAPER0_R1_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_R1_JOB_ID"],
    "development_run": "85604",
    "held_out_85606_read": False,
    "completed_logical_runs": 6,
    "runs": runs,
    "training_result_accepted": False,
    "O1_scientific_evaluation_completed": False,
    "next_required_action": "complete separately committed O1 evaluation",
}
temporary = root / ".training_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "training_summary.json")
PY

sha256sum \
  "${JOB_ROOT}/commands.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/training_summary.json" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}"/task_*/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Phase 2 O1 full R1 codec job completed at $(date --iso-8601=seconds)"
echo "Training is complete; R1 remains undecided until the separate O1 evaluation."
