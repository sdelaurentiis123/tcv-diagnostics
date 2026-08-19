#!/bin/bash
# Full frozen 85604 B4 M32/M4 evaluation followed by independent H-det/H-prob reduction.
# Submit only after jobs 6899682 (training) and 6900991 (evaluator smoke) pass.
#SBATCH --job-name=p3_b4_eval
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase3_b4_eval_full_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${B4_TRAINING_JOB_ID:?submit with B4_TRAINING_JOB_ID set}"
: "${B4_SMOKE_JOB_ID:?submit with B4_SMOKE_JOB_ID set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly NODE_LOCAL_ROOT="/tmp/${USER}/tcv_diagnostics_b4_eval_full_${SLURM_JOB_ID}"
readonly MODEL_DATA_STAGED="${NODE_LOCAL_ROOT}/phase2_model_dataset_85604"
readonly TRAINING_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_full/job_${B4_TRAINING_JOB_ID}"
readonly TRAINING_OUTPUT="${TRAINING_ROOT}/b4_pde_refiner_h1_seed_1701"
readonly TRAINING_RESULT="${TRAINING_OUTPUT}/result.json"
readonly TRAINING_WANDB="${TRAINING_OUTPUT}/wandb.json"
readonly SMOKE_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_evaluator_smoke/job_${B4_SMOKE_JOB_ID}"
readonly SMOKE_OUTPUT="${SMOKE_ROOT}/b4_pde_refiner_h1_seed_1701"
readonly SMOKE_RESULT="${SMOKE_OUTPUT}/result.json"
readonly THRESHOLD_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_transport_event_thresholds/job_6897493"
readonly THRESHOLD_RESULT="${THRESHOLD_ROOT}/thresholds/result.json"
readonly COMPARATOR_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b3_h1_comparator/job_6899063"
readonly COMPARATOR_RESULT="${COMPARATOR_ROOT}/comparators/comparators.json"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b4_pde_refiner_evaluation_full"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${OUTDIR}/b4_pde_refiner_h1_seed_1701"
readonly GATE_OUTPUT="${OUTDIR}/gate"
readonly EVALUATION_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase3_b4_full_evaluation_85604.json"
readonly EVALUATION_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly EVALUATOR="${PAPER0_ROOT}/paper0/tools/evaluate_b4_pde_refiner_checkpoint.py"
readonly WANDB_WRAPPER="${PAPER0_ROOT}/paper0/tools/run_b4_pde_refiner_evaluation_wandb.py"
readonly FINALIZER="${PAPER0_ROOT}/paper0/tools/finalize_b4_pde_refiner_one_seed.py"
readonly TRAINING_COMMIT="0350b063a1f7e5c6a00b76bf4d6aeaec71d511ef"
readonly PASSING_SMOKE_COMMIT="029f6d9d425fd9bbac11aebf82466588a97ac658"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="b4-pde-refiner-h1-full-evaluation-85604"

if [[ -e "${OUTDIR}" || -e "${NODE_LOCAL_ROOT}" ]]; then
  echo "Refusing to overwrite a B4 full-evaluation directory" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

cleanup() {
  rm -rf "${NODE_LOCAL_ROOT}"
}
trap cleanup EXIT

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
  echo "B4 full evaluation requires Rocky Linux 9" >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]]; then
  echo "Exactly one allocated accelerator is required" >&2
  exit 2
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -Evq "H100|H200"; then
  echo "The B4 full evaluation requires H100 or H200" >&2
  exit 2
fi

check_sha256 "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229" "${EVALUATION_MANIFEST}"
check_sha256 "ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1" "${EVALUATION_PROTOCOL}"
check_sha256 "ad0c1d1d2fd22c48b0a877e58a35c9df0a1ed1aac2b54cf5f2dde1a696e32f1a" "${EVALUATOR}"
check_sha256 "11e57da24ad92ef48cec74dfd4fb34e23915cb3c154369d9abef4587efad8199" "${WANDB_WRAPPER}"
check_sha256 "777ecf7f5d601d68cf1dbfc4b02e1ede384ab682bda3758ea7306177aa98c439" "${FINALIZER}"
check_sha256 "8725d48b712ac1b3c8e2928de05ddf545ed329035f41897b3d727ca28192e94f" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_acceptance_gate.py"
check_sha256 "7691a31fd57305ed27eed5e9be97c93e1934184b385dfabfbd85f92950b67155" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_forecast.py"
check_sha256 "e00eabcaf4170f3aaf0d1da488a62b3b1e3042786fdfbd00085da58ddb39d6bf" "${PAPER0_ROOT}/src/tcv_diagnostics/pde_refiner_scoring.py"
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
check_sha256 "2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2" "${COMPARATOR_RESULT}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_SOURCE}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_SOURCE}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_SOURCE}/artifact_sha256.txt"

sha256sum -c "${TRAINING_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${TRAINING_ROOT}/artifact_sha256.txt"
sha256sum -c "${SMOKE_OUTPUT}/artifact_sha256.txt"
sha256sum -c "${SMOKE_ROOT}/artifact_sha256.txt"
sha256sum -c "${THRESHOLD_ROOT}/thresholds/artifact_sha256.txt"
sha256sum -c "${COMPARATOR_ROOT}/comparators/artifact_sha256.txt"
readonly TRAINING_RESULT_SHA="$(sha256sum "${TRAINING_RESULT}" | awk '{print $1}')"
readonly TRAINING_WANDB_SHA="$(sha256sum "${TRAINING_WANDB}" | awk '{print $1}')"
readonly SMOKE_RESULT_SHA="$(sha256sum "${SMOKE_RESULT}" | awk '{print $1}')"
readonly THRESHOLD_RESULT_SHA="$(sha256sum "${THRESHOLD_RESULT}" | awk '{print $1}')"
readonly COMPARATOR_RESULT_SHA="$(sha256sum "${COMPARATOR_RESULT}" | awk '{print $1}')"

"${PYTHON}" - "${SMOKE_RESULT}" "${PASSING_SMOKE_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
if record.get("paper0_commit") != sys.argv[2]:
    raise RuntimeError("passing B4 evaluator smoke commit differs")
if record.get("status") != "bounded_evaluator_smoke_completed":
    raise RuntimeError("B4 evaluator smoke did not pass")
if record.get("held_out_85606_read") is not False:
    raise RuntimeError("B4 evaluator smoke opened 85606")
PY

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "training_commit=${TRAINING_COMMIT}"
  echo "training_job_id=${B4_TRAINING_JOB_ID}"
  echo "smoke_job_id=${B4_SMOKE_JOB_ID}"
  echo "comparator_job_id=6899063"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,pytest,scipy,torch,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("wandb="+wandb.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

echo "Running complete Rocky 9 suite before full B4 evaluation"
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

readonly EVALUATOR_SHA="ad0c1d1d2fd22c48b0a877e58a35c9df0a1ed1aac2b54cf5f2dde1a696e32f1a"
readonly RUN_ID="p0b4evalfull-${SLURM_JOB_ID}-s1701"
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
  --wandb-run-name "b4-pde-refiner-full-evaluation-s1701-j${SLURM_JOB_ID}"
  --
  --mode full
  --seed 1701
  --training-result "${TRAINING_RESULT}"
  --training-result-sha256 "${TRAINING_RESULT_SHA}"
  --training-commit "${TRAINING_COMMIT}"
  --smoke-result "${SMOKE_RESULT}"
  --smoke-result-sha256 "${SMOKE_RESULT_SHA}"
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
  --evaluation-manifest-sha256 "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
  --evaluation-protocol "${EVALUATION_PROTOCOL}"
  --evaluation-protocol-sha256 "ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1"
  --output-directory "${RUN_OUTPUT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --member-batch-size 8
)
printf '%q ' "${COMMAND[@]}" > "${OUTDIR}/command.sh"
printf '\n' >> "${OUTDIR}/command.sh"
"${COMMAND[@]}" | tee "${OUTDIR}/stdout.jsonl"

readonly EVALUATION_RESULT="${RUN_OUTPUT}/result.json"
readonly EVALUATION_WANDB="${RUN_OUTPUT}/wandb.json"
readonly EVALUATION_RESULT_SHA="$(sha256sum "${EVALUATION_RESULT}" | awk '{print $1}')"
readonly EVALUATION_WANDB_SHA="$(sha256sum "${EVALUATION_WANDB}" | awk '{print $1}')"

GATE_COMMAND=(
  "${PYTHON}" -u "${FINALIZER}"
  --evaluation-result "${EVALUATION_RESULT}"
  --evaluation-result-sha256 "${EVALUATION_RESULT_SHA}"
  --comparator-result "${COMPARATOR_RESULT}"
  --comparator-result-sha256 "${COMPARATOR_RESULT_SHA}"
  --evaluation-manifest "${EVALUATION_MANIFEST}"
  --evaluation-manifest-sha256 "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
  --training-wandb "${TRAINING_WANDB}"
  --training-wandb-sha256 "${TRAINING_WANDB_SHA}"
  --evaluation-wandb "${EVALUATION_WANDB}"
  --evaluation-wandb-sha256 "${EVALUATION_WANDB_SHA}"
  --gate-execution-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --output-directory "${GATE_OUTPUT}"
)
printf '%q ' "${GATE_COMMAND[@]}" > "${OUTDIR}/gate_command.sh"
printf '\n' >> "${OUTDIR}/gate_command.sh"
"${GATE_COMMAND[@]}" | tee "${OUTDIR}/gate_stdout.jsonl"

"${PYTHON}" - "${GATE_OUTPUT}/final_gate.json" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
if record.get("scientific_authority") is not True:
    raise RuntimeError("B4 final gate lacks scientific authority")
if record.get("held_out_85606_read") is not False:
    raise RuntimeError("B4 final gate opened 85606")
if record.get("O3_launch_allowed") is not False:
    raise RuntimeError("B4 final gate unexpectedly authorized O3 execution")
if record.get("assimilation_allowed") is not False:
    raise RuntimeError("B4 final gate unexpectedly authorized assimilation")
PY

cp -a "${GATE_OUTPUT}/final_gate.json" "${OUTDIR}/result.json"
sha256sum \
  "${OUTDIR}/result.json" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${GATE_OUTPUT}/artifact_sha256.txt" \
  > "${OUTDIR}/artifact_sha256.txt"
echo "Completed full frozen B4 H-det/H-prob evaluation at ${OUTDIR}"
echo "No B4 outcome authorizes O3 execution, assimilation, diagnostic ranking, or 85606 access"
