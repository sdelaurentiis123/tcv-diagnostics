#!/bin/bash
# Reconstruct the frozen R1 train/validation frames for all six selected codecs.
# This GPU stage writes candidates and pre-transport metrics; it cannot accept R1.
# gpuxl allocates four H100s, so runs 0..3 execute first and runs 4..5 second.
#SBATCH --job-name=p2_o1_recon
#SBATCH --partition=gpuxl
#SBATCH --qos=gen
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_matched_o1_recon_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_r1/job_6893802"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_o1_recon_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly FREEZE_RESULT="${PAPER0_ROOT}/paper0/results/phase2_o1_codec_r1_freeze_6894300.json"
readonly TRUTH_REPLAY="${PAPER0_ROOT}/paper0/results/phase2_matched_e6b_truth_replay_6894325.json"
readonly RECON_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_o1_r1_reconstruction.json"
readonly MATCHED_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_o1_o2_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly DATA_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_6893525.json"
readonly NORMALIZATION_RESULT="${PAPER0_ROOT}/paper0/results/phase2_model_dataset_normalization_6893525.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/evaluate_matched_codec_reconstruction.py"

readonly -a families=(c5p e6b c5p e6b c5p e6b)
readonly -a seeds=(1701 1701 1702 1702 1703 1703)
readonly -a checkpoints=(
  "${TRAINING_ROOT}/task_0_c5p_seed_1701/selected.pt"
  "${TRAINING_ROOT}/task_1_e6b_seed_1701/selected.pt"
  "${TRAINING_ROOT}/task_2_c5p_seed_1702/selected.pt"
  "${TRAINING_ROOT}/task_3_e6b_seed_1702/selected.pt"
  "${TRAINING_ROOT}/task_4_c5p_seed_1703/selected.pt"
  "${TRAINING_ROOT}/task_5_e6b_seed_1703/selected.pt"
)
readonly -a checkpoint_hashes=(
  646cd939a73fd6d2417d67f61309b400ac4ab2b3982241d8922769cc2f854f3f
  6ff6464a7ed516ea1c0a9b2c6b26cc044fca23ad22eca0ef149687a81a78dd26
  77b191ca4ddc8a46c726b5d7c2f66726aa926fc2c781f138ce0095eba0581072
  360fd67dfb2d793d6dde7546252e69bf567a15d18983e8d61387097b8fc5cae6
  878b79715c1f08a3572341981914777124ef6966d3899653b05208e10827409e
  a8b0dbca56476a056a1d0e704edfc84ec30d68ed3c53970bc9a64121099ed2be
)
readonly -a training_results=(
  "${TRAINING_ROOT}/task_0_c5p_seed_1701/result.json"
  "${TRAINING_ROOT}/task_1_e6b_seed_1701/result.json"
  "${TRAINING_ROOT}/task_2_c5p_seed_1702/result.json"
  "${TRAINING_ROOT}/task_3_e6b_seed_1702/result.json"
  "${TRAINING_ROOT}/task_4_c5p_seed_1703/result.json"
  "${TRAINING_ROOT}/task_5_e6b_seed_1703/result.json"
)
readonly -a training_result_hashes=(
  3925bd3210a8baa4e5764dea97c6187a57461a43f0f2875d3fe1c68d10c99bf8
  5021b8c3e41e7bec390ac31636ddcc36d4fb8a4ce9d2b1466ba107d2ee5a42a6
  370c5054daa061dfbbaad638f6f622d6f093e00a5520906f8993e3ebf1197a94
  7bb8b6b69fe014ce2b73526f7c9b5072ed0e59afb2645c129ef20db3befddd1c
  456ac0e9559ed9f37a25c39b2d6ca4bb891e82d9122c40d1c8c06ec42ce532cb
  4c7936c6e5141f5a960193860ba7318d344eb0c82e3d7f9345f7a5097a731e88
)

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
  echo "This reconstruction requires Rocky 9" >&2
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

check_sha256 "7589069f6447c10c44debe3eaa3ccdeffe0d553a512280c1cf0b440ae6aefb57" "${FREEZE_RESULT}"
check_sha256 "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27" "${TRUTH_REPLAY}"
check_sha256 "949c875baa7c69e51ff839a0598f50db3eb7e96bfb708e537c7ae4d0dd06cd4c" "${RECON_MANIFEST}"
check_sha256 "6cbbf991e311565ab2cc2d3ae0eb4f1d2572f4c23fb804723c990d5b2401c562" "${MATCHED_MANIFEST}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${DATA_RESULT}"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${NORMALIZATION_RESULT}"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "e060674d9afa0d8fd4d709bc927ef2c144fb40d284f7e672fc37636f820d23bf" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_codec_evaluation.py"
check_sha256 "4d6847b1f7dcae3043df07d405d9dad07cc86c96b1f63c6fb8ccdbcf745a37ed" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_codec_metrics.py"
check_sha256 "a94dc2a64c565454776f8a38d1c3f7fad9ca54c4c4fc01c00640c3c2bc91cc53" "${ENTRYPOINT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"
for ((run_index = 0; run_index < 6; run_index++)); do
  check_sha256 "${checkpoint_hashes[run_index]}" "${checkpoints[run_index]}"
  check_sha256 "${training_result_hashes[run_index]}" "${training_results[run_index]}"
done

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export TOKENIZERS_PARALLELISM=false

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,scipy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cudnn="+str(torch.backends.cudnn.version()))'
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
} > "${JOB_ROOT}/environment.txt" 2>&1

export PAPER0_FREEZE_RESULT="${FREEZE_RESULT}"
export PAPER0_TRUTH_REPLAY="${TRUTH_REPLAY}"
export PAPER0_RECON_MANIFEST="${RECON_MANIFEST}"
"${PYTHON}" - <<'PY' > "${JOB_ROOT}/preflight.json"
import json
import os
from pathlib import Path

freeze = json.loads(Path(os.environ["PAPER0_FREEZE_RESULT"]).read_text())
truth = json.loads(Path(os.environ["PAPER0_TRUTH_REPLAY"]).read_text())
manifest = json.loads(Path(os.environ["PAPER0_RECON_MANIFEST"]).read_text())
expected = [(0, "c5p", 1701), (1, "e6b", 1701), (2, "c5p", 1702),
            (3, "e6b", 1702), (4, "c5p", 1703), (5, "e6b", 1703)]
observed = [(int(run["run_index"]), run["family"], int(run["seed"]))
            for run in freeze["runs"]]
if observed != expected or int(freeze["run_count"]) != 6:
    raise RuntimeError("frozen training matrix run order differs")
if not freeze["checkpoint_choice_frozen_before_physics_metrics"]:
    raise RuntimeError("checkpoint choices were not frozen")
if truth["status"] != "pass" or not truth["all_frames_passed"]:
    raise RuntimeError("exact E6B phi prerequisite did not pass")
if truth["coverage"] != [0, 624] or int(truth["frame_count"]) != 624:
    raise RuntimeError("exact E6B phi prerequisite coverage differs")
if manifest["held_out_85606_access_allowed"] is not False:
    raise RuntimeError("reconstruction manifest permits held-out access")
if manifest["evaluation"]["chunk_frames"] != 4:
    raise RuntimeError("reconstruction chunk size differs")
print(json.dumps({
    "development_run": "85604",
    "held_out_85606_read": False,
    "frozen_run_order": observed,
    "exact_e6b_phi_truth_replay": "pass",
    "chunk_frames": 4,
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

declare -a pids

task_output() {
  local run_index="$1"
  echo "${JOB_ROOT}/task_${run_index}_${families[run_index]}_seed_${seeds[run_index]}"
}

write_command() {
  local run_index="$1"
  local gpu_index="$2"
  local output
  output="$(task_output "${run_index}")"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_index}"
  printf '%q ' \
    "${PYTHON}" -u "${ENTRYPOINT}" \
    --codec dcae_l20 \
    --family "${families[run_index]}" \
    --seed "${seeds[run_index]}" \
    --checkpoint "${checkpoints[run_index]}" \
    --checkpoint-sha256 "${checkpoint_hashes[run_index]}" \
    --training-result "${training_results[run_index]}" \
    --training-result-sha256 "${training_result_hashes[run_index]}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --output "${output}" \
    --evaluation-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}" \
    --chunk-frames 4 \
    --device cuda:0
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
  local output
  output="$(task_output "${run_index}")"
  export CUDA_VISIBLE_DEVICES="${gpu_index}"
  export OMP_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  echo "Starting reconstruction ${run_index}: family=${families[run_index]}, seed=${seeds[run_index]}, GPU=${gpu_index}"
  "${PYTHON}" -u "${ENTRYPOINT}" \
    --codec dcae_l20 \
    --family "${families[run_index]}" \
    --seed "${seeds[run_index]}" \
    --checkpoint "${checkpoints[run_index]}" \
    --checkpoint-sha256 "${checkpoint_hashes[run_index]}" \
    --training-result "${training_results[run_index]}" \
    --training-result-sha256 "${training_result_hashes[run_index]}" \
    --artifact-root "${MODEL_DATA_STAGED}" \
    --output "${output}" \
    --evaluation-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:gpu${gpu_index}" \
    --chunk-frames 4 \
    --device cuda:0
  "${PYTHON}" -m json.tool "${output}/result.json" >/dev/null
  sha256sum \
    "${output}/result.json" \
    "${output}/train_candidate_native81.h5" \
    "${output}/validation_candidate_native81.h5" \
    > "${output}/artifact_sha256.txt"
  echo "Completed reconstruction ${run_index}: family=${families[run_index]}, seed=${seeds[run_index]}"
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
      echo "Reconstruction ${run_index} failed" >&2
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "At least one reconstruction failed; later waves will not launch" >&2
    exit 1
  fi
}

echo "Launching reconstruction wave 0: runs 0, 1, 2, 3"
for run_index in 0 1 2 3; do
  launch_task "${run_index}" "${run_index}"
done
wait_wave 0 1 2 3

echo "Launching reconstruction wave 1: runs 4, 5"
launch_task 4 0
launch_task 5 1
wait_wave 4 5

export PAPER0_RECON_JOB_ROOT="${JOB_ROOT}"
export PAPER0_RECON_COMMIT="${PAPER0_COMMIT}"
export PAPER0_RECON_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_RECON_JOB_ROOT"])
families = ("c5p", "e6b", "c5p", "e6b", "c5p", "e6b")
seeds = (1701, 1701, 1702, 1702, 1703, 1703)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

runs = []
for index, (family, seed) in enumerate(zip(families, seeds)):
    output = root / f"task_{index}_{family}_seed_{seed}"
    path = output / "result.json"
    result = json.loads(path.read_text())
    if result["scope"] != "phase2_matched_o1_codec_reconstruction":
        raise RuntimeError(f"run {index} scope differs")
    if result["status"] != "completed_pending_exact_elliptic_and_transport":
        raise RuntimeError(f"run {index} status differs")
    if result["held_out_85606_read"] is not False:
        raise RuntimeError(f"run {index} touched held-out data")
    if result["identity"]["family"] != family or result["identity"]["seed"] != seed:
        raise RuntimeError(f"run {index} identity differs")
    if result["training"]["frame_count"] != 432:
        raise RuntimeError(f"run {index} training coverage differs")
    if result["validation"]["frame_count"] != 128:
        raise RuntimeError(f"run {index} validation coverage differs")
    if result["downstream"]["complete_O1_decision_allowed"] is not False:
        raise RuntimeError(f"run {index} prematurely permits an O1 decision")
    runs.append({
        "run_index": index,
        "family": family,
        "seed": seed,
        "result": {"path": str(path), "sha256": sha256(path)},
        "native_view_pretransport_status": result["native_view_pretransport_gate"]["status"],
        "training_candidate": result["training"]["candidate_native81"],
        "validation_candidate": result["validation"]["candidate_native81"],
        "complete_O1_decision_allowed": False,
    })
payload = {
    "schema_version": 1,
    "scope": "phase2_matched_o1_r1_reconstruction_matrix",
    "status": "completed_pending_exact_e6b_phi_and_authoritative_transport",
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_RECON_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_RECON_JOB_ID"],
    "run_count": len(runs),
    "runs": runs,
    "R1_accepted": False,
    "complete_O1_decision_allowed": False,
}
temporary = root / ".reconstruction_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "reconstruction_summary.json")
PY

sha256sum \
  "${JOB_ROOT}/commands.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/preflight.json" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/reconstruction_summary.json" \
  "${JOB_ROOT}"/task_*/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Matched O1 R1 reconstruction matrix completed"
echo "No complete O1 or R1 decision is allowed before exact E6B phi and transport evaluation"
