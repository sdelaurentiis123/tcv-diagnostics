#!/bin/bash
# Six frozen C5P O2 one-step training runs on one nonpreemptible H100 node.
# Training remains data-only; a separate committed evaluator decides O2.
#SBATCH --job-name=p2_o2_full
#SBATCH --partition=gpuxl
#SBATCH --qos=gen
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o2_full_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_o2_full_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_full"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly FULL_RUN_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_c5p_o2_full_runs_85604.json"
readonly CONTINUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_c5p_o2_continuation_85604.json"
readonly CONTINUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md"
readonly SMOKE_RESULT="${PAPER0_ROOT}/paper0/results/phase2_o2_gpu_smoke_6894971.json"
readonly R2_RESULT="${PAPER0_ROOT}/paper0/results/phase2_matched_o1_finalize_r2_6894863.json"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/train_o2.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="o2-c5p-l10-full"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing result directory: ${JOB_ROOT}" >&2
  exit 2
fi
if [[ -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to reuse existing node-local directory: ${NODE_LOCAL_ROOT}" >&2
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
  echo "Full O2 training requires Rocky 9" >&2
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

check_sha256 "fd7d601e592eb05fb391fea03dbe6aad4b4d1dd8b4a2ecad4156802a64a19aac" "${FULL_RUN_MANIFEST}"
check_sha256 "7ad5cd207b2b2c69fb82a2b22f65d55e81fad941a508f977966485611ba078d6" "${CONTINUATION_MANIFEST}"
check_sha256 "e969311c0b8012242d44815201f135bd012aa6a9b9c6dc7ca4c6c9043df753eb" "${CONTINUATION_PROTOCOL}"
check_sha256 "a812a5642c5d3d9dedc25711af6d92994bcebf5432d9dd7a99911fc1f5e53451" "${SMOKE_RESULT}"
check_sha256 "ad1cf868d9532869b918ba5892e3926bb06f9961653ac718857b42ac0a182f89" "${R2_RESULT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "8eb062349ab78d8f163c46bb3539e7fd3791cc095ac3b8ed17ae9c000a5be512" "${PAPER0_ROOT}/src/tcv_diagnostics/models/__init__.py"
check_sha256 "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853" "${PAPER0_ROOT}/src/tcv_diagnostics/models/vit.py"
check_sha256 "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb" "${PAPER0_ROOT}/src/tcv_diagnostics/models/o2.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "755bf1ef51b80eb178d4ed535b8da9131c37350aee4bc905655036ebbd49667a" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training_data.py"
check_sha256 "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_training.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "74c514cc09c77c7d3b44da71cd450116772d576c7949d11c4a7393642b2fc849" "${PAPER0_ROOT}/src/tcv_diagnostics/wandb_tracking.py"
check_sha256 "07c8116afec7a0842b229ecc431a43467788447bb3bfa0ea2efcf51a78908c2e" "${ENTRYPOINT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

arms=("C5P-H1" "C5P-H1" "C5P-H1" "C5P-H2" "C5P-H2" "C5P-H2")
slugs=("c5p_h1" "c5p_h1" "c5p_h1" "c5p_h2" "c5p_h2" "c5p_h2")
seeds=(1701 1702 1703 1701 1702 1703)
contexts=(1 1 1 2 2 2)
codec_paths=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_2_c5p_seed_1702/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_4_c5p_seed_1703/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_2_c5p_seed_1702/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_4_c5p_seed_1703/selected.pt"
)
codec_shas=(
  "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
  "c4f180171510489bf022ea003201494ecee991cc3b2f200e97854ec4ffd86836"
  "3d029572a24f69058889de43c42cefa29633e7167db441db4f41fa1e0c1b3f5d"
  "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
  "c4f180171510489bf022ea003201494ecee991cc3b2f200e97854ec4ffd86836"
  "3d029572a24f69058889de43c42cefa29633e7167db441db4f41fa1e0c1b3f5d"
)

for run_index in 0 1 2 3 4 5; do
  check_sha256 "${codec_shas[run_index]}" "${codec_paths[run_index]}"
done

export PAPER0_FULL_RUN_MANIFEST="${FULL_RUN_MANIFEST}"
export PAPER0_SMOKE_RESULT="${SMOKE_RESULT}"
"${PYTHON}" - <<'PY' > "${JOB_ROOT}/authorization.json"
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["PAPER0_FULL_RUN_MANIFEST"]).read_text())
smoke = json.loads(Path(os.environ["PAPER0_SMOKE_RESULT"]).read_text())
expected = [
    (0, "C5P-H1", 1, 1701),
    (1, "C5P-H1", 1, 1702),
    (2, "C5P-H1", 1, 1703),
    (3, "C5P-H2", 2, 1701),
    (4, "C5P-H2", 2, 1702),
    (5, "C5P-H2", 2, 1703),
]
if manifest["status"] != "frozen_after_passing_Rocky9_GPU_smoke_before_full_O2_training":
    raise RuntimeError("full O2 run manifest status differs")
if manifest["development_run"] != "85604" or manifest["held_out_85606_access_allowed"] is not False:
    raise RuntimeError("full O2 run manifest data scope differs")
observed = [
    (int(task["run_index"]), task["arm"], int(task["context_frames"]), int(task["seed"]))
    for task in manifest["tasks"]
]
if observed != expected:
    raise RuntimeError("full O2 task matrix differs")
if manifest["model"]["physics_derived_loss_allowed"] is not False:
    raise RuntimeError("full O2 manifest permits a physics-derived loss")
if manifest["training"]["total_optimizer_steps"] != 5400:
    raise RuntimeError("full O2 optimizer budget differs")
if not (
    smoke["status"] == "passed"
    and smoke["scientific_result"] is False
    and smoke["full_O2_training_allowed"] is True
    and smoke["held_out_85606_read"] is False
):
    raise RuntimeError("tracked GPU smoke does not authorize full O2 training")
print(json.dumps({
    "development_run": "85604",
    "held_out_85606_read": False,
    "gpu_smoke_job_id": smoke["slurm"]["job_id"],
    "logical_runs": len(observed),
    "full_O2_training_authorized": True,
    "O2_scientific_gate_evaluated": False,
    "O3_launch_allowed": False,
}, sort_keys=True))
PY

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
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

echo "Running the complete Rocky 9 suite before full O2 training"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

declare -a pids

task_output() {
  local run_index="$1"
  echo "${JOB_ROOT}/task_${run_index}_${slugs[run_index]}_seed_${seeds[run_index]}"
}

write_command() {
  local run_index="$1"
  local gpu_index="$2"
  local output
  output="$(task_output "${run_index}")"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_index}"
  printf '%q ' \
    "${PYTHON}" -u "${ENTRYPOINT}" \
    --mode full \
    --arm "${arms[run_index]}" \
    --seed "${seeds[run_index]}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --codec-checkpoint "${codec_paths[run_index]}" \
    --codec-sha256 "${codec_shas[run_index]}" \
    --output "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}" \
    --continuation-manifest "${CONTINUATION_MANIFEST}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-id "p0o2full-${SLURM_JOB_ID}-${run_index}" \
    --wandb-run-name "o2-full-${slugs[run_index]}-s${seeds[run_index]}-j${SLURM_JOB_ID}-t${run_index}"
  printf '\n'
}

{
  write_command 0 0
  write_command 3 1
  write_command 1 2
  write_command 4 3
  write_command 2 0
  write_command 5 1
} > "${JOB_ROOT}/commands.sh"

run_task() {
  local run_index="$1"
  local gpu_index="$2"
  local output
  output="$(task_output "${run_index}")"
  export CUDA_VISIBLE_DEVICES="${gpu_index}"
  export OMP_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  echo "Starting O2 run ${run_index}: arm=${arms[run_index]}, seed=${seeds[run_index]}, GPU=${gpu_index}"
  "${PYTHON}" -u "${ENTRYPOINT}" \
    --mode full \
    --arm "${arms[run_index]}" \
    --seed "${seeds[run_index]}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --codec-checkpoint "${codec_paths[run_index]}" \
    --codec-sha256 "${codec_shas[run_index]}" \
    --output "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}" \
    --continuation-manifest "${CONTINUATION_MANIFEST}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-run-id "p0o2full-${SLURM_JOB_ID}-${run_index}" \
    --wandb-run-name "o2-full-${slugs[run_index]}-s${seeds[run_index]}-j${SLURM_JOB_ID}-t${run_index}"
  "${PYTHON}" -m json.tool "${output}/result.json" >/dev/null
  "${PYTHON}" -m json.tool "${output}/wandb.json" >/dev/null
  sha256sum \
    "${output}/config.json" \
    "${output}/latent_normalization.json" \
    "${output}/history.jsonl" \
    "${output}/result.json" \
    "${output}/selected.pt" \
    "${output}/final_training_state.pt" \
    "${output}/wandb.json" \
    > "${output}/artifact_sha256.txt"
  echo "Completed O2 run ${run_index}: arm=${arms[run_index]}, seed=${seeds[run_index]}"
}

launch_task() {
  local run_index="$1"
  local gpu_index="$2"
  local task_log="${JOB_ROOT}/task_${run_index}_${slugs[run_index]}_seed_${seeds[run_index]}.log"
  run_task "${run_index}" "${gpu_index}" > >(tee -a "${task_log}") 2>&1 &
  pids[run_index]=$!
}

wait_wave() {
  local failed=0
  local run_index
  for run_index in "$@"; do
    if ! wait "${pids[run_index]}"; then
      echo "O2 run ${run_index} failed" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one O2 run failed; later waves will not launch" >&2
    exit 1
  fi
}

echo "Launching O2 wave 0: runs 0, 3, 1, 4"
launch_task 0 0
launch_task 3 1
launch_task 1 2
launch_task 4 3
wait_wave 0 3 1 4

echo "Launching O2 wave 1: runs 2, 5"
launch_task 2 0
launch_task 5 1
wait_wave 2 5

export PAPER0_O2_JOB_ROOT="${JOB_ROOT}"
export PAPER0_O2_COMMIT="${PAPER0_COMMIT}"
export PAPER0_O2_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_O2_JOB_ROOT"])
arms = ("C5P-H1", "C5P-H1", "C5P-H1", "C5P-H2", "C5P-H2", "C5P-H2")
slugs = ("c5p_h1", "c5p_h1", "c5p_h1", "c5p_h2", "c5p_h2", "c5p_h2")
seeds = (1701, 1702, 1703, 1701, 1702, 1703)
contexts = (1, 1, 1, 2, 2, 2)
runs = []
for index, (arm, slug, seed, context) in enumerate(zip(arms, slugs, seeds, contexts)):
    output = root / f"task_{index}_{slug}_seed_{seed}"
    result = json.loads((output / "result.json").read_text())
    tracking = json.loads((output / "wandb.json").read_text())
    latent = json.loads((output / "latent_normalization.json").read_text())
    config = result["config"]
    if result["scope"] != "O2_teacher_forced_one_step_full":
        raise RuntimeError(f"O2 run {index} scope differs")
    if config["arm"] != arm or int(config["seed"]) != seed or int(config["context_frames"]) != context:
        raise RuntimeError(f"O2 run {index} identity differs")
    if result["completed_epochs"] != 200 or result["completed_optimizer_steps"] != 5400:
        raise RuntimeError(f"O2 run {index} did not complete the frozen budget")
    if config["train_targets"] != [2, 432] or config["validation_targets"] != [498, 624]:
        raise RuntimeError(f"O2 run {index} frame ranges differ")
    if latent["fit_frames"] != [0, 432] or latent["scientific_authority"] is not True:
        raise RuntimeError(f"O2 run {index} latent normalization differs")
    if result["held_out_85606_read"] is not False:
        raise RuntimeError(f"O2 run {index} touched held-out data")
    if result["physics_derived_loss_used"] is not False:
        raise RuntimeError(f"O2 run {index} used a physics-derived loss")
    if result["target_truth_used_as_model_input"] is not False:
        raise RuntimeError(f"O2 run {index} used target truth as model input")
    if result["checkpoint_reload_bitwise_exact"] is not True:
        raise RuntimeError(f"O2 run {index} failed checkpoint reload identity")
    if result["O2_scientific_gate_evaluated"] is not False or result["O3_launch_allowed"] is not False:
        raise RuntimeError(f"O2 run {index} prematurely advanced the protocol")
    if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
        raise RuntimeError(f"O2 run {index} lacks a finished online W&B record")
    if tracking["epochs_logged"] != 200:
        raise RuntimeError(f"O2 run {index} did not mirror all epochs to W&B")
    runs.append({
        "run_index": index,
        "arm": arm,
        "context_frames": context,
        "seed": seed,
        "output": str(output),
        "codec_checkpoint_sha256": result["codec_checkpoint"]["sha256"],
        "selected_epoch": result["selected_epoch"],
        "selected_validation_equal_channel_mae": result["selected_validation_equal_channel_mae"],
        "selected_checkpoint_sha256": result["selected_checkpoint"]["sha256"],
        "parameter_count": result["parameter_count"],
        "peak_cuda_bytes": result["peak_cuda_bytes"],
        "wall_seconds": result["wall_seconds"],
        "wandb_url": tracking["run_url"],
    })
payload = {
    "schema_version": 1,
    "scope": "phase2_C5P_O2_full_training",
    "paper0_commit": os.environ["PAPER0_O2_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_O2_JOB_ID"],
    "development_run": "85604",
    "held_out_85606_read": False,
    "completed_logical_runs": len(runs),
    "runs": runs,
    "training_complete": True,
    "training_summary_is_scientific_acceptance": False,
    "O2_scientific_gate_evaluated": False,
    "O3_launch_allowed": False,
    "next_required_action": "freeze selected checkpoints and run separately committed O2 evaluation",
}
temporary = root / ".training_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "training_summary.json")
PY

sha256sum \
  "${JOB_ROOT}/authorization.json" \
  "${JOB_ROOT}/commands.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/training_summary.json" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}"/task_*/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Phase 2 C5P O2 full training completed"
echo "O2 remains scientifically undecided until separate frozen evaluation"
