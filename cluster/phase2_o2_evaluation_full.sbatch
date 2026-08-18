#!/bin/bash
# Full frozen 85604 O2 scientific evaluation for six selected C5P checkpoints.
# References are built once; six seed gates are evaluated independently.
# This job records the matrix decision but never launches O3 or opens 85606.
#SBATCH --job-name=p2_o2_eval_full
#SBATCH --partition=gpuxl
#SBATCH --qos=gen
#SBATCH --gres=gpu:h200:4
#SBATCH --constraint=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o2_eval_full_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_o2_eval_full_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_evaluation_full"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly REFERENCES_ROOT="${JOB_ROOT}/references"
readonly FINAL_MATRIX="${JOB_ROOT}/final_matrix.json"

readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_o2_scientific_evaluation_85604.json"
readonly EVALUATION_MANIFEST_SHA="c9056310d032b1e7c4d63c5d188a6fe1bbf2bf45a13c7520ba8f409b6ea8c29f"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_O2_SCIENTIFIC_EVALUATION_PROTOCOL.md"
readonly EVALUATION_PROTOCOL_SHA="61200fa6224a035f6f5b129eaa27e849ad3b55bb4c3901c751c3a3fbf7b65aa4"
readonly TRAINING_FREEZE="${PAPER0_ROOT}/paper0/results/phase2_o2_training_freeze_6895637.json"
readonly TRAINING_FREEZE_SHA="dd8951e39e60d1631866ebe7af7c4d529ad543daf211233369b8fec9936ee837"
readonly SMOKE_RESULT="${PAPER0_ROOT}/paper0/results/phase2_o2_evaluation_smoke_6895931.json"
readonly SMOKE_RESULT_SHA="dc53d9561d0ef0f00cbb41eb14f510fd4a19cf5427acd17be2947baff4211273"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly NATIVE_TRUTH_RESULT_SHA="cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY_MANIFEST_SHA="76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly GEOMETRY_SHA="0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
readonly TRAINING_COMMIT="9035bc3ce9d2351cd17586f4429af8116d43a47e"

readonly BUILD_REFERENCES="${PAPER0_ROOT}/paper0/tools/build_o2_references.py"
readonly EVALUATE_CHECKPOINT="${PAPER0_ROOT}/paper0/tools/evaluate_o2_checkpoint.py"
readonly FINALIZE_MATRIX="${PAPER0_ROOT}/paper0/tools/finalize_o2_evaluation.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="o2-scientific-evaluation-full"

arms=("C5P-H1" "C5P-H1" "C5P-H1" "C5P-H2" "C5P-H2" "C5P-H2")
slugs=("c5p_h1" "c5p_h1" "c5p_h1" "c5p_h2" "c5p_h2" "c5p_h2")
seeds=(1701 1702 1703 1701 1702 1703)
checkpoints=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_1_c5p_h1_seed_1702/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_2_c5p_h1_seed_1703/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_3_c5p_h2_seed_1701/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_4_c5p_h2_seed_1702/selected.pt"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_5_c5p_h2_seed_1703/selected.pt"
)
checkpoint_shas=(
  "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
  "d15c74717fad6a3ccf5b5af895e3eefb7271667f4bbde2164514a61a526bc0e8"
  "a718b2135c7019d05541bd5ffb029ce9408df8225603cffc957c42d2ce5abae3"
  "3b971b2081901469e1f98adbe27b5cdbf3281d08a99ee28e0d8d8b1577722a84"
  "5edc3e002730eb78232967255cfab66ee860b8b3858eed007f7061341b5c36eb"
  "a70bd271117f1b0afb21258e4c5d7d4eb4919dc4a528509ccbf6ac2464622d85"
)
codecs=(
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

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing result directory: ${JOB_ROOT}" >&2
  exit 2
fi
if [[ -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to reuse node-local directory: ${NODE_LOCAL_ROOT}" >&2
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
  echo "The full O2 evaluation requires Rocky 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python environment is unavailable: ${PYTHON}" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 4 ]]; then
  echo "Exactly four allocated H200 GPUs are required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -vq "H200"; then
  echo "A non-H200 accelerator was allocated" >&2
  exit 2
fi

check_sha256 "${EVALUATION_MANIFEST_SHA}" "${EVALUATION_MANIFEST}"
check_sha256 "${EVALUATION_PROTOCOL_SHA}" "${EVALUATION_PROTOCOL}"
check_sha256 "${TRAINING_FREEZE_SHA}" "${TRAINING_FREEZE}"
check_sha256 "${SMOKE_RESULT_SHA}" "${SMOKE_RESULT}"
check_sha256 "${NATIVE_TRUTH_RESULT_SHA}" "${NATIVE_TRUTH_RESULT}"
check_sha256 "${GEOMETRY_MANIFEST_SHA}" "${GEOMETRY_MANIFEST}"
check_sha256 "${GEOMETRY_SHA}" "${GEOMETRY}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"
check_sha256 "45182fa3729d5edbc7494233457f746c5c40cc532b3906b1b38da407f2c01abf" "${BUILD_REFERENCES}"
check_sha256 "f2a59d19ed5eebc42575a4e52878d9cdf06ef2061aec8ac4f4c4fbf17f1bbb55" "${EVALUATE_CHECKPOINT}"
check_sha256 "d17626b7cdd9aafb7158d9264bdf7f0f076de9385773d3dca55d8006210cb93f" "${FINALIZE_MATRIX}"
check_sha256 "859054d7dcdc96b80871ce17a77e501fa5dec53a7158017944fceba71977d8f9" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_evaluation.py"
check_sha256 "7847e57452020b7fb23f8cfec4c4e7f00f0b6585df51fbac53aa62bd2fe0223e" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_forecast.py"
check_sha256 "d5e39d4a055db843cd974316bf01206aaa52f16dd769b505bca5c9944b31fc27" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_scoring.py"
check_sha256 "1a22153942401de69546a625e305edf1e27554e6369f983c04175dcd521bf4d0" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_matrix.py"

for run_index in 0 1 2 3 4 5; do
  check_sha256 "${checkpoint_shas[run_index]}" "${checkpoints[run_index]}"
  check_sha256 "${codec_shas[run_index]}" "${codecs[run_index]}"
done

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export WANDB_DIR="${JOB_ROOT}/wandb_runtime"
mkdir -p "${WANDB_DIR}"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "rocky_version=${VERSION_ID}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("pytest="+pytest.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version())); print("wandb="+wandb.__version__)'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PAPER0_WANDB_ENTITY="${WANDB_ENTITY}"
"${PYTHON}" - <<'PY' > "${JOB_ROOT}/wandb_preflight.json"
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

echo "Running complete Rocky 9 CPU suite before full O2 evaluation"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

echo "Staging and verifying the immutable 85604 model dataset"
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

echo "Building the frozen full references once"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
"${PYTHON}" -u "${BUILD_REFERENCES}" \
  --mode full \
  --artifact-root "${MODEL_DATA_STAGED}" \
  --native-truth-result "${NATIVE_TRUTH_RESULT}" \
  --native-truth-result-sha256 "${NATIVE_TRUTH_RESULT_SHA}" \
  --geometry-manifest "${GEOMETRY_MANIFEST}" \
  --geometry-manifest-sha256 "${GEOMETRY_MANIFEST_SHA}" \
  --geometry "${GEOMETRY}" \
  --evaluation-manifest "${EVALUATION_MANIFEST}" \
  --evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA}" \
  --output-directory "${REFERENCES_ROOT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}:references" \
  | tee "${JOB_ROOT}/references_stdout.jsonl"

readonly REFERENCES_RESULT="${REFERENCES_ROOT}/result.json"
readonly REFERENCES_RESULT_SHA="$(sha256sum "${REFERENCES_RESULT}" | awk '{print $1}')"
sha256sum -c "${REFERENCES_ROOT}/artifact_sha256.txt"

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
    "${PYTHON}" -u "${EVALUATE_CHECKPOINT}" \
    --mode full \
    --arm "${arms[run_index]}" \
    --seed "${seeds[run_index]}" \
    --checkpoint "${checkpoints[run_index]}" \
    --checkpoint-sha256 "${checkpoint_shas[run_index]}" \
    --codec-checkpoint "${codecs[run_index]}" \
    --codec-checkpoint-sha256 "${codec_shas[run_index]}" \
    --training-commit "${TRAINING_COMMIT}" \
    --training-freeze-result "${TRAINING_FREEZE}" \
    --training-freeze-result-sha256 "${TRAINING_FREEZE_SHA}" \
    --references-result "${REFERENCES_RESULT}" \
    --references-result-sha256 "${REFERENCES_RESULT_SHA}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --native-truth-result "${NATIVE_TRUTH_RESULT}" \
    --native-truth-result-sha256 "${NATIVE_TRUTH_RESULT_SHA}" \
    --geometry-manifest "${GEOMETRY_MANIFEST}" \
    --geometry-manifest-sha256 "${GEOMETRY_MANIFEST_SHA}" \
    --geometry "${GEOMETRY}" \
    --evaluation-manifest "${EVALUATION_MANIFEST}" \
    --evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA}" \
    --output-directory "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}"
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
  export OMP_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  echo "Starting full O2 evaluation ${run_index}: arm=${arms[run_index]}, seed=${seeds[run_index]}, GPU=${gpu_index}"
  "${PYTHON}" -u "${EVALUATE_CHECKPOINT}" \
    --mode full \
    --arm "${arms[run_index]}" \
    --seed "${seeds[run_index]}" \
    --checkpoint "${checkpoints[run_index]}" \
    --checkpoint-sha256 "${checkpoint_shas[run_index]}" \
    --codec-checkpoint "${codecs[run_index]}" \
    --codec-checkpoint-sha256 "${codec_shas[run_index]}" \
    --training-commit "${TRAINING_COMMIT}" \
    --training-freeze-result "${TRAINING_FREEZE}" \
    --training-freeze-result-sha256 "${TRAINING_FREEZE_SHA}" \
    --references-result "${REFERENCES_RESULT}" \
    --references-result-sha256 "${REFERENCES_RESULT_SHA}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --native-truth-result "${NATIVE_TRUTH_RESULT}" \
    --native-truth-result-sha256 "${NATIVE_TRUTH_RESULT_SHA}" \
    --geometry-manifest "${GEOMETRY_MANIFEST}" \
    --geometry-manifest-sha256 "${GEOMETRY_MANIFEST_SHA}" \
    --geometry "${GEOMETRY}" \
    --evaluation-manifest "${EVALUATION_MANIFEST}" \
    --evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA}" \
    --output-directory "${output}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}"
  "${PYTHON}" -m json.tool "${output}/result.json" >/dev/null
  sha256sum -c "${output}/artifact_sha256.txt"
  echo "Completed full O2 evaluation ${run_index}: arm=${arms[run_index]}, seed=${seeds[run_index]}"
}

declare -a pids

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
      echo "O2 evaluation ${run_index} failed" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one O2 evaluation failed; finalization is prohibited" >&2
    exit 1
  fi
}

echo "Launching full O2 evaluation wave 0: runs 0, 3, 1, 4"
launch_task 0 0
launch_task 3 1
launch_task 1 2
launch_task 4 3
wait_wave 0 3 1 4

echo "Launching full O2 evaluation wave 1: runs 2, 5"
launch_task 2 0
launch_task 5 1
wait_wave 2 5

declare -a result_args
for run_index in 0 1 2 3 4 5; do
  output="$(task_output "${run_index}")"
  result_path="${output}/result.json"
  result_sha="$(sha256sum "${result_path}" | awk '{print $1}')"
  result_args+=(--result "${result_path}" --result-sha256 "${result_sha}")
done

"${PYTHON}" -u "${FINALIZE_MATRIX}" \
  "${result_args[@]}" \
  --training-freeze-result "${TRAINING_FREEZE}" \
  --training-freeze-result-sha256 "${TRAINING_FREEZE_SHA}" \
  --references-result "${REFERENCES_RESULT}" \
  --references-result-sha256 "${REFERENCES_RESULT_SHA}" \
  --evaluation-manifest "${EVALUATION_MANIFEST}" \
  --evaluation-manifest-sha256 "${EVALUATION_MANIFEST_SHA}" \
  --output "${FINAL_MATRIX}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}:finalize" \
  | tee "${JOB_ROOT}/finalize_stdout.jsonl"

export PAPER0_FULL_JOB_ROOT="${JOB_ROOT}"
export PAPER0_FULL_JOB_ID="${SLURM_JOB_ID}"
export PAPER0_FULL_COMMIT="${PAPER0_COMMIT}"
export PAPER0_FULL_WANDB_ENTITY="${WANDB_ENTITY}"
export PAPER0_FULL_WANDB_PROJECT="${WANDB_PROJECT}"
export PAPER0_FULL_WANDB_GROUP="${WANDB_GROUP}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_FULL_JOB_ROOT"])
matrix = json.loads((root / "final_matrix.json").read_text())
if matrix["scope"] != "phase2_C5P_O2_complete_scientific_matrix":
    raise RuntimeError("full O2 matrix scope differs")
if matrix["run_count"] != 6 or matrix["O2_scientific_evaluation_completed"] is not True:
    raise RuntimeError("full O2 matrix is incomplete")
if matrix["O3_launch_allowed"] is not False:
    raise RuntimeError("full O2 evaluation directly authorized O3")
if matrix["held_out_85606_read"] is not False or matrix["held_out_85606_access_allowed"] is not False:
    raise RuntimeError("full O2 evaluation escaped 85604 scope")
expected = [
    (0, "C5P-H1", 1701),
    (1, "C5P-H1", 1702),
    (2, "C5P-H1", 1703),
    (3, "C5P-H2", 1701),
    (4, "C5P-H2", 1702),
    (5, "C5P-H2", 1703),
]
observed = [
    (int(run["training_run_index"]), run["arm"], int(run["seed"]))
    for run in matrix["runs"]
]
if observed != expected:
    raise RuntimeError("full O2 matrix identity or order differs")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "schema_version": 1,
    "scope": "phase2_C5P_O2_full_85604_scientific_evaluation_execution",
    "status": "completed",
    "scientific_authority": True,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_FULL_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_FULL_JOB_ID"],
    "target_frames": [498, 624],
    "target_count": 126,
    "run_count": 6,
    "accepted_arms": matrix["O2_accepted_arms"],
    "decision": matrix["decision"],
    "final_matrix": {
        "path": str((root / "final_matrix.json").resolve()),
        "sha256": sha(root / "final_matrix.json"),
    },
    "references_result": matrix["references_result"],
    "O2_scientific_evaluation_completed": True,
    "new_O3_protocol_may_be_frozen": matrix["new_O3_protocol_may_be_frozen"],
    "O3_launch_allowed": False,
    "stochastic_model_authorized": False,
    "held_out_85606_access_allowed": False,
    "wandb_online_requested": True,
}
temporary = root / ".evaluation_summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "evaluation_summary.json")
PY

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import wandb

root = Path(os.environ["PAPER0_FULL_JOB_ROOT"])
matrix = json.loads((root / "final_matrix.json").read_text())
run_id = f"p0-o2-eval-full-{os.environ['PAPER0_FULL_JOB_ID']}"
run = wandb.init(
    entity=os.environ["PAPER0_FULL_WANDB_ENTITY"],
    project=os.environ["PAPER0_FULL_WANDB_PROJECT"],
    group=os.environ["PAPER0_FULL_WANDB_GROUP"],
    job_type="scientific-evaluation",
    id=run_id,
    name=f"O2-full-evaluation-j{os.environ['PAPER0_FULL_JOB_ID']}",
    resume="never",
    config={
        "paper0_commit": os.environ["PAPER0_FULL_COMMIT"],
        "development_run": "85604",
        "target_start": 498,
        "target_stop": 624,
        "target_count": 126,
        "seed_count": 6,
        "arm_acceptance_requires_all_three_seeds": True,
        "scientific_authority": True,
    },
)
for item in matrix["runs"]:
    result = json.loads(Path(item["result"]["path"]).read_text())
    score = json.loads(Path(item["score"]["path"]).read_text())
    prefix = f"{item['arm'].lower().replace('-', '_')}/seed_{item['seed']}"
    field = score["field_spectral_cross"]["overall"]
    run.log({
        f"{prefix}/aggregate_rmse_standardized": field[
            "aggregate_equal_channel_rmse_standardized"
        ],
        f"{prefix}/aggregate_mae_standardized": field[
            "aggregate_equal_channel_mae_standardized"
        ],
        f"{prefix}/seed_gate_pass": int(result["O2_seed_accepted"]),
    })
run.summary["accepted_arm_count"] = matrix["decision"]["accepted_arm_count"]
run.summary["accepted_arms"] = ",".join(matrix["O2_accepted_arms"])
run.summary["disposition"] = matrix["decision"]["disposition"]
run.summary["O3_launch_allowed"] = False
artifact = wandb.Artifact(
    f"o2-full-scientific-evaluation-j{os.environ['PAPER0_FULL_JOB_ID']}",
    type="scientific-evaluation",
)
artifact.add_file(str(root / "final_matrix.json"))
artifact.add_file(str(root / "evaluation_summary.json"))
artifact.add_file(str(root / "references/result.json"), name="references_result.json")
for item in matrix["runs"]:
    artifact.add_file(
        item["result"]["path"],
        name=f"{item['training_run_index']}_{item['arm']}_seed_{item['seed']}_result.json",
    )
run.log_artifact(artifact)
run.finish()
remote = wandb.Api(timeout=30).run(
    f"{os.environ['PAPER0_FULL_WANDB_ENTITY']}/{os.environ['PAPER0_FULL_WANDB_PROJECT']}/{run_id}"
)
if str(remote.state) != "finished":
    raise RuntimeError(f"W&B full evaluation state {remote.state!r} is not finished")
record = {
    "mode": "online",
    "entity": os.environ["PAPER0_FULL_WANDB_ENTITY"],
    "project": os.environ["PAPER0_FULL_WANDB_PROJECT"],
    "group": os.environ["PAPER0_FULL_WANDB_GROUP"],
    "run_id": run_id,
    "run_url": f"https://wandb.ai/{os.environ['PAPER0_FULL_WANDB_ENTITY']}/{os.environ['PAPER0_FULL_WANDB_PROJECT']}/runs/{run_id}",
    "remote_state_after_finish": str(remote.state),
    "scientific_authority": True,
}
temporary = root / ".wandb.json.tmp"
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "wandb.json")
PY

sha256sum \
  "${JOB_ROOT}/commands.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}/wandb.json" \
  "${JOB_ROOT}/evaluation_summary.json" \
  "${JOB_ROOT}/final_matrix.json" \
  "${REFERENCES_ROOT}/artifact_sha256.txt" \
  "${JOB_ROOT}"/task_*/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Phase 2 full 85604 O2 scientific evaluation completed"
echo "The matrix decision is recorded; O3 remains unlaunched and 85606 remains closed"
