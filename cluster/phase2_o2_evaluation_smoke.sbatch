#!/bin/bash
# Rocky 9 implementation smoke for the frozen 85604 O2 scientific evaluator.
# This runs all references and one selected checkpoint on four targets only.
# It has no scientific authority and cannot open O3 or held-out evaluation.
#SBATCH --job-name=p2_o2_eval_smoke
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_o2_eval_smoke_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_o2_eval_smoke_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_evaluation_smoke"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly REFERENCES_ROOT="${JOB_ROOT}/references"
readonly CANDIDATE_ROOT="${JOB_ROOT}/c5p_h1_seed_1701"

readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_o2_scientific_evaluation_85604.json"
readonly EVALUATION_MANIFEST_SHA="c9056310d032b1e7c4d63c5d188a6fe1bbf2bf45a13c7520ba8f409b6ea8c29f"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_O2_SCIENTIFIC_EVALUATION_PROTOCOL.md"
readonly EVALUATION_PROTOCOL_SHA="61200fa6224a035f6f5b129eaa27e849ad3b55bb4c3901c751c3a3fbf7b65aa4"
readonly TRAINING_FREEZE="${PAPER0_ROOT}/paper0/results/phase2_o2_training_freeze_6895637.json"
readonly TRAINING_FREEZE_SHA="dd8951e39e60d1631866ebe7af7c4d529ad543daf211233369b8fec9936ee837"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly NATIVE_TRUTH_RESULT_SHA="cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY_MANIFEST_SHA="76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly GEOMETRY_SHA="0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"

readonly TRAINING_COMMIT="9035bc3ce9d2351cd17586f4429af8116d43a47e"
readonly CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
readonly CHECKPOINT_SHA="5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
readonly CODEC="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"

readonly BUILD_REFERENCES="${PAPER0_ROOT}/paper0/tools/build_o2_references.py"
readonly EVALUATE_CHECKPOINT="${PAPER0_ROOT}/paper0/tools/evaluate_o2_checkpoint.py"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="o2-scientific-evaluation-smoke"

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
  echo "The O2 evaluation smoke requires Rocky 9" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Pinned Python environment is unavailable: ${PYTHON}" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated H100 or H200 is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100|H200"; then
  echo "The allocated accelerator is neither H100 nor H200" >&2
  exit 2
fi

check_sha256 "${EVALUATION_MANIFEST_SHA}" "${EVALUATION_MANIFEST}"
check_sha256 "${EVALUATION_PROTOCOL_SHA}" "${EVALUATION_PROTOCOL}"
check_sha256 "${TRAINING_FREEZE_SHA}" "${TRAINING_FREEZE}"
check_sha256 "${NATIVE_TRUTH_RESULT_SHA}" "${NATIVE_TRUTH_RESULT}"
check_sha256 "${GEOMETRY_MANIFEST_SHA}" "${GEOMETRY_MANIFEST}"
check_sha256 "${GEOMETRY_SHA}" "${GEOMETRY}"
check_sha256 "${CHECKPOINT_SHA}" "${CHECKPOINT}"
check_sha256 "${CODEC_SHA}" "${CODEC}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"
check_sha256 "45182fa3729d5edbc7494233457f746c5c40cc532b3906b1b38da407f2c01abf" "${BUILD_REFERENCES}"
check_sha256 "f2a59d19ed5eebc42575a4e52878d9cdf06ef2061aec8ac4f4c4fbf17f1bbb55" "${EVALUATE_CHECKPOINT}"
check_sha256 "859054d7dcdc96b80871ce17a77e501fa5dec53a7158017944fceba71977d8f9" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_evaluation.py"
check_sha256 "7847e57452020b7fb23f8cfec4c4e7f00f0b6585df51fbac53aa62bd2fe0223e" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_forecast.py"
check_sha256 "d5e39d4a055db843cd974316bf01206aaa52f16dd769b505bca5c9944b31fc27" "${PAPER0_ROOT}/src/tcv_diagnostics/o2_scoring.py"

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
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

echo "Running complete Rocky 9 CPU suite before the non-scientific smoke"
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

"${PYTHON}" -u "${BUILD_REFERENCES}" \
  --mode smoke \
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

"${PYTHON}" -u "${EVALUATE_CHECKPOINT}" \
  --mode smoke \
  --arm C5P-H1 \
  --seed 1701 \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sha256 "${CHECKPOINT_SHA}" \
  --codec-checkpoint "${CODEC}" \
  --codec-checkpoint-sha256 "${CODEC_SHA}" \
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
  --output-directory "${CANDIDATE_ROOT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}:C5P-H1:1701" \
  | tee "${JOB_ROOT}/candidate_stdout.jsonl"

export PAPER0_SMOKE_JOB_ROOT="${JOB_ROOT}"
export PAPER0_SMOKE_JOB_ID="${SLURM_JOB_ID}"
export PAPER0_SMOKE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_SMOKE_WANDB_ENTITY="${WANDB_ENTITY}"
export PAPER0_SMOKE_WANDB_PROJECT="${WANDB_PROJECT}"
export PAPER0_SMOKE_WANDB_GROUP="${WANDB_GROUP}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_SMOKE_JOB_ROOT"])
references = json.loads((root / "references/result.json").read_text())
candidate = json.loads((root / "c5p_h1_seed_1701/result.json").read_text())
candidate_score = json.loads(Path(candidate["score"]["path"]).read_text())
if references["mode"] != "smoke" or references["scientific_authority"] is not False:
    raise RuntimeError("reference smoke incorrectly claims scientific authority")
if references["target_frames"] != [498, 502] or set(references["references"]) != {
    "persistence", "spectral_ar1", "linear_extrapolation"
}:
    raise RuntimeError("reference smoke target or inventory differs")
if candidate["status"] != "smoke_passed" or candidate["scientific_authority"] is not False:
    raise RuntimeError("candidate smoke contract differs")
if candidate["gate"] is not None or candidate["O2_seed_accepted"] is not False:
    raise RuntimeError("smoke evaluated or passed the scientific O2 gate")
if candidate_score["target_frames"] != [498, 502] or candidate_score["target_count"] != 4:
    raise RuntimeError("candidate smoke target interval differs")
if candidate_score["target_truth_used_during_forecast_generation"] is not False:
    raise RuntimeError("candidate generation read future truth")
if candidate["held_out_85606_read"] is not False or candidate["O3_launch_allowed"] is not False:
    raise RuntimeError("candidate smoke escaped authorized scope")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "schema_version": 1,
    "scope": "phase2_C5P_O2_scientific_evaluator_Rocky9_smoke",
    "status": "passed",
    "scientific_authority": False,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_SMOKE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_SMOKE_JOB_ID"],
    "target_frames": [498, 502],
    "reference_names": sorted(references["references"]),
    "candidate": {"arm": candidate["arm"], "seed": candidate["seed"]},
    "truth_separated_forecast_generation_passed": True,
    "native_transport_scoring_passed": True,
    "complete_cpu_suite_passed": True,
    "wandb_online_requested": True,
    "O2_scientific_gate_evaluated": False,
    "O2_seed_accepted": False,
    "O3_launch_allowed": False,
    "full_evaluation_may_be_launched_after_result_audit": True,
    "references_result": {
        "path": str((root / "references/result.json").resolve()),
        "sha256": sha(root / "references/result.json"),
    },
    "candidate_result": {
        "path": str((root / "c5p_h1_seed_1701/result.json").resolve()),
        "sha256": sha(root / "c5p_h1_seed_1701/result.json"),
    },
    "candidate_aggregate_rmse_standardized_descriptive_only": candidate_score[
        "field_spectral_cross"
    ]["overall"]["aggregate_equal_channel_rmse_standardized"],
}
temporary = root / ".smoke_summary.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "smoke_summary.json")
PY

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import wandb

root = Path(os.environ["PAPER0_SMOKE_JOB_ROOT"])
summary = json.loads((root / "smoke_summary.json").read_text())
run_id = f"p0-o2-eval-smoke-{os.environ['PAPER0_SMOKE_JOB_ID']}"
run = wandb.init(
    entity=os.environ["PAPER0_SMOKE_WANDB_ENTITY"],
    project=os.environ["PAPER0_SMOKE_WANDB_PROJECT"],
    group=os.environ["PAPER0_SMOKE_WANDB_GROUP"],
    job_type="evaluation-smoke",
    id=run_id,
    name=f"O2-eval-smoke-j{os.environ['PAPER0_SMOKE_JOB_ID']}",
    resume="never",
    config={
        "paper0_commit": os.environ["PAPER0_SMOKE_COMMIT"],
        "development_run": "85604",
        "mode": "smoke",
        "target_start": 498,
        "target_stop": 502,
        "arm": "C5P-H1",
        "seed": 1701,
        "scientific_authority": False,
    },
)
run.log({
    "smoke/candidate_aggregate_rmse_standardized": summary[
        "candidate_aggregate_rmse_standardized_descriptive_only"
    ],
    "smoke/truth_separation_passed": 1,
    "smoke/native_transport_scoring_passed": 1,
})
artifact = wandb.Artifact(f"o2-eval-smoke-j{os.environ['PAPER0_SMOKE_JOB_ID']}", type="evaluation-smoke")
artifact.add_file(str(root / "smoke_summary.json"))
artifact.add_file(str(root / "references/result.json"), name="references_result.json")
artifact.add_file(str(root / "c5p_h1_seed_1701/result.json"), name="candidate_result.json")
run.log_artifact(artifact)
run.finish()
remote = wandb.Api(timeout=30).run(
    f"{os.environ['PAPER0_SMOKE_WANDB_ENTITY']}/{os.environ['PAPER0_SMOKE_WANDB_PROJECT']}/{run_id}"
)
if str(remote.state) != "finished":
    raise RuntimeError(f"W&B smoke run state {remote.state!r} is not finished")
record = {
    "mode": "online",
    "entity": os.environ["PAPER0_SMOKE_WANDB_ENTITY"],
    "project": os.environ["PAPER0_SMOKE_WANDB_PROJECT"],
    "group": os.environ["PAPER0_SMOKE_WANDB_GROUP"],
    "run_id": run_id,
    "run_url": f"https://wandb.ai/{os.environ['PAPER0_SMOKE_WANDB_ENTITY']}/{os.environ['PAPER0_SMOKE_WANDB_PROJECT']}/runs/{run_id}",
    "remote_state_after_finish": str(remote.state),
    "scientific_authority": False,
}
temporary = root / ".wandb.json.tmp"
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "wandb.json")
PY

sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${JOB_ROOT}/wandb_preflight.json" \
  "${JOB_ROOT}/wandb.json" \
  "${JOB_ROOT}/smoke_summary.json" \
  "${REFERENCES_ROOT}/artifact_sha256.txt" \
  "${CANDIDATE_ROOT}/artifact_sha256.txt" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Phase 2 O2 scientific-evaluator smoke passed"
echo "This four-target smoke has no scientific authority and does not evaluate the O2 gate"
