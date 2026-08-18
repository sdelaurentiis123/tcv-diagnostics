#!/bin/bash
# Full 126-target M32 B3-FGN-H1 evaluation on 85604 validation only.
# This scores the model but does not apply or prejudge the frozen acceptance gate.
#SBATCH --job-name=p3_b3_eval
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b3_eval_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B3_TRAINING_JOB_ID:?submit with B3_TRAINING_JOB_ID set}"
: "${B3_SMOKE_JOB_ID:?submit with B3_SMOKE_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b3_eval_full_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_full/job_${B3_TRAINING_JOB_ID}"
readonly TRAINING_OUTPUT="${TRAINING_ROOT}/b3_fgn_h1_seed_1701"
readonly TRAINING_RESULT="${TRAINING_OUTPUT}/result.json"
readonly SMOKE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_evaluator_smoke/job_${B3_SMOKE_JOB_ID}"
readonly SMOKE_OUTPUT="${SMOKE_ROOT}/b3_fgn_h1_seed_1701"
readonly SMOKE_RESULT="${SMOKE_OUTPUT}/result.json"
readonly THRESHOLD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_6897493"
readonly THRESHOLD_RESULT="${THRESHOLD_ROOT}/thresholds/result.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_fgn_evaluation_full"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${OUTDIR}/b3_fgn_h1_seed_1701"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b3_full_evaluation_85604.json"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B3_FULL_EVALUATION_PROTOCOL.md"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_b3_fgn_checkpoint.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b3_fgn_evaluation_wandb.py"
readonly TRAINING_COMMIT="a2a17cf3fc30fd504bc3eee3274e78623bf15e2b"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b3-fgn-h1-full-evaluation-85604"

if [[ -e "${OUTDIR}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite a full B3 evaluation directory" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch" >&2
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
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "Full B3 evaluation requires Rocky Linux 9" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated accelerator is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100|H200"; then
  echo "Full B3 evaluation requires H100 or H200" >&2
  exit 2
fi

check_sha256 "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2" "${EVALUATION_MANIFEST}"
check_sha256 "db717c5605ad9653d2b051ec13254b43bf230f514cb173d295e95d3c68af8030" "${EVALUATION_PROTOCOL}"
check_sha256 "114545c375bbb985d20ca8bd014720981e6495686f45cc78b17458df16c18673" "${EVALUATOR}"
check_sha256 "8b187290f9ae9124cb957bfc0ba2e0a11bb35b453ad50474bd5ec86203760550" "${WANDB_WRAPPER}"
check_sha256 "264309a309dd8625b4acb64624f102ee05071bc3ccf785d1e291e9483c0f9cce" "${PAPER0_ROOT}/src/tcv_diagnostics/fgn_forecast.py"
check_sha256 "16656a0057b2e19f52cdb6265546a924511882cf2eee64522e266acee27d4953" "${PAPER0_ROOT}/src/tcv_diagnostics/fgn_scoring.py"
check_sha256 "1a8ec4227f5de595f27e9d2679ca3b830e598516d56e19d01910cc1c0413fef6" "${PAPER0_ROOT}/src/tcv_diagnostics/fgn_training.py"
check_sha256 "69c5e140e6f456e2442ae40ec5be7c04c5d228baf6ec95e119de5a3d1c86e79d" "${PAPER0_ROOT}/src/tcv_diagnostics/models/functional_noise.py"
check_sha256 "2dfdf6f7b620302826971c9fec4ed8233f46fa1950c8461ed9d79194411178fe" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_scoring.py"
check_sha256 "edef6fbbe7b40348fa450c7428d796f4b5ebc3d9b2070e135c7bb3f58a2b6650" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_probabilistic_metrics.py"
check_sha256 "c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_field_metrics.py"
check_sha256 "382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_spectral_metrics.py"
check_sha256 "b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e" "${PAPER0_ROOT}/src/tcv_diagnostics/b2_transport_metrics.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "14c977ee0ce5ebac0ec3ed05682b71f7d2a517448ed8d563974def62498f1fcb" "${THRESHOLD_RESULT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${TRAINING_ROOT}/artifact_sha256.txt"
sha256sum -c "${SMOKE_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${SMOKE_ROOT}/artifact_sha256.txt"
sha256sum -c "${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly SMOKE_RESULT_SHA="$(sha256sum "${SMOKE_RESULT}" | awk '{print $1}')"
readonly THRESHOLD_RESULT_SHA="$(sha256sum "${THRESHOLD_RESULT}" | awk '{print $1}')"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export MKL_NUM_THREADS=12
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_commit=${TRAINING_COMMIT}"
  echo "training_job_id=${B3_TRAINING_JOB_ID}"
  echo "evaluator_smoke_job_id=${B3_SMOKE_JOB_ID}"
  echo "threshold_job_id=6897493"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

echo "Running complete Rocky 9 suite before full B3 evaluation"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${OUTDIR}/test_output.txt"

echo "Staging verified 85604 model data"
mkdir -p "${MODEL_DATA_STAGED}/shards"
cp -a "${MODEL_DATA_SOURCE}/model_dataset_manifest.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/normalization.json" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/artifact_sha256.txt" "${MODEL_DATA_STAGED}/"
cp -a "${MODEL_DATA_SOURCE}/shards/." "${MODEL_DATA_STAGED}/shards/"
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
  echo "Expected eight verified model-data shards" >&2
  exit 2
fi

readonly EVALUATOR_SHA="114545c375bbb985d20ca8bd014720981e6495686f45cc78b17458df16c18673"
readonly RUN_ID="p0b3eval-${SLURM_JOB_ID}-s1701"
COMMAND=(
  "${PYTHON}" -u "${WANDB_WRAPPER}"
  --evaluator "${EVALUATOR}"
  --evaluator-sha256 "${EVALUATOR_SHA}"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --seed 1701
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "b3-fgn-full-evaluation-s1701-j${SLURM_JOB_ID}"
  --
  --mode full
  --seed 1701
  --training-result "${TRAINING_RESULT}"
  --training-result-sha256 "${TRAINING_RESULT_SHA}"
  --training-commit "${TRAINING_COMMIT}"
  --artifact-root "${MODEL_DATA_STAGED}"
  --native-truth-result "${NATIVE_TRUTH_RESULT}"
  --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
  --geometry-manifest "${GEOMETRY_MANIFEST}"
  --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
  --geometry "${GEOMETRY}"
  --geometry-sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
  --event-threshold-result "${THRESHOLD_RESULT}"
  --event-threshold-result-sha256 "${THRESHOLD_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
  --evaluation-protocol "${EVALUATION_PROTOCOL}"
  --evaluation-protocol-sha256 "db717c5605ad9653d2b051ec13254b43bf230f514cb173d295e95d3c68af8030"
  --smoke-result "${SMOKE_RESULT}"
  --smoke-result-sha256 "${SMOKE_RESULT_SHA}"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --member-batch-size 8
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

export PAPER0_B3_FULL_EVAL_OUTPUT="${RUN_OUTPUT}"
export PAPER0_B3_FULL_EVAL_ROOT="${OUTDIR}"
export PAPER0_B3_FULL_EVAL_COMMIT="${PAPER0_COMMIT}"
export PAPER0_B3_FULL_EVAL_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import write_strict_json_atomic

output = Path(os.environ["PAPER0_B3_FULL_EVAL_OUTPUT"])
root = Path(os.environ["PAPER0_B3_FULL_EVAL_ROOT"])
result = json.loads((output / "result.json").read_text())
generation = json.loads((output / "generation.json").read_text())
score = json.loads((output / "score.json").read_text())
tracking = json.loads((output / "wandb.json").read_text())
if result["scope"] != "B3_FGN_H1_full_probabilistic_evaluation_85604":
    raise RuntimeError("full B3 evaluation scope differs")
if result["status"] != "completed_pending_frozen_acceptance_gate":
    raise RuntimeError("full B3 evaluation status differs")
if result["target_frames"] != [498, 624] or result["target_count"] != 126:
    raise RuntimeError("full B3 evaluation target interval differs")
if result["ensemble_members"] != 32 or result["scientific_authority"] is not True:
    raise RuntimeError("full B3 evaluation ensemble/scientific scope differs")
if score["scope"] != "B3_FGN_H1_truth_separated_probabilistic_scoring_85604":
    raise RuntimeError("full B3 scoring scope differs")
if tracking["mode"] != "online" or tracking["remote_state_after_finish"] != "finished":
    raise RuntimeError("full B3 evaluation W&B record differs")
if result["full_probabilistic_evaluation_preconditions_passed"] is not True:
    raise RuntimeError("full B3 evaluation preconditions differ")
if result["truth_opened_only_after_forecast_hash"] is not True:
    raise RuntimeError("full B3 truth separation differs")
if result["probabilistic_scientific_gate_evaluated"] is not False:
    raise RuntimeError("full B3 evaluator prejudged the gate")
if result["O3_launch_allowed"] is not False:
    raise RuntimeError("full B3 evaluator authorized O3")
if result["scientific_noise"]["sha256"] != "1449777a61d40af49ccb3bd6bed5edcba0fd8afe24d113e6175218c04865aa9c":
    raise RuntimeError("full B3 scientific-noise bank differs")
if generation["inference"]["network_evaluations_per_member"] != 1:
    raise RuntimeError("full B3 evaluation was not direct single-pass sampling")

summary = {
    "schema_version": 1,
    "scope": "phase3_B3_FGN_H1_full_probabilistic_evaluation",
    "status": "completed_pending_frozen_acceptance_gate",
    "scientific_result": True,
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_B3_FULL_EVAL_COMMIT"],
    "training_commit": result["training_commit"],
    "slurm_job_id": os.environ["PAPER0_B3_FULL_EVAL_JOB_ID"],
    "target_frames": [498, 624],
    "target_count": 126,
    "ensemble_members": 32,
    "truth_opened_only_after_forecast_hash": True,
    "direct_network_evaluations_per_member": 1,
    "probabilistic_scientific_gate_evaluated": False,
    "O3_launch_allowed": False,
    "run_result": {
        "path": str(output / "result.json"),
        "sha256": sha256_path(output / "result.json"),
    },
    "forecast": result["forecast"],
    "score": result["score"],
    "scientific_noise": result["scientific_noise"],
    "wandb": tracking,
}
write_strict_json_atomic(root / "result.json", summary)
PY

sha256sum \
  "${OUTDIR}/command.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${OUTDIR}/stdout.jsonl" \
  "${OUTDIR}/test_output.txt" \
  "${OUTDIR}/result.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${OUTDIR}/artifact_sha256.txt"

sha256sum "${OUTDIR}/result.json" "${RUN_OUTPUT}/artifact_sha256.txt"
echo "Full B3 FGN M32 evaluation completed on 85604 validation"
echo "The frozen one-seed acceptance gate remains unevaluated; O3 and 85606 remain closed"
