#!/bin/bash
# Read-only score/data calibration for the bounded old-85604 PGL variogram screen.
# It does not perform an optimizer update and never accesses another run.
#SBATCH --job-name=p0_pgl_var_pre
#SBATCH --partition=gpupreempt
#SBATCH --qos=gpupreempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_variogram_preflight_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256:?set evidence-manifest SHA-256}"
: "${PAPER0_PGL_VARIOGRAM_BASE_PROTOCOL_SHA256:?set base-protocol SHA-256}"
: "${PAPER0_PGL_VARIOGRAM_AMENDMENT_SHA256:?set amendment SHA-256}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly EVIDENCE_MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly BASE_PROTOCOL="${PAPER0_ROOT}/paper0/protocol/POST_ECRD_OLD_85604_PGL_BOUNDED_VARIOGRAM_FINETUNE_2026-08-28.md"
readonly AMENDMENT="${PAPER0_ROOT}/paper0/protocol/POST_ECRD_OLD_85604_PGL_BOUNDED_VARIOGRAM_FINETUNE_AMENDMENT_2026-08-28.md"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/preflight_pgl_variogram_screen.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_preflight"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite variogram preflight output" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

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
for specification in \
  "${EVIDENCE_MANIFEST}:${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}" \
  "${BASE_PROTOCOL}:${PAPER0_PGL_VARIOGRAM_BASE_PROTOCOL_SHA256}" \
  "${AMENDMENT}:${PAPER0_PGL_VARIOGRAM_AMENDMENT_SHA256}"; do
  path="${specification%%:*}"
  expected="${specification##*:}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Locked variogram input SHA-256 differs: ${path}" >&2
    exit 2
  fi
done

source /etc/os-release
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=old_85604_pgl_variogram_preflight"
  echo "development_run=85604"
  echo "training_performed=false"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,scipy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda))'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_pgl_variogram.py \
  tests/test_pgl_torch_transport.py \
  tests/test_pgl_variogram_training.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --evidence-manifest "${EVIDENCE_MANIFEST}"
  --evidence-manifest-sha256 "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}"
  --base-protocol "${BASE_PROTOCOL}"
  --base-protocol-sha256 "${PAPER0_PGL_VARIOGRAM_BASE_PROTOCOL_SHA256}"
  --amendment "${AMENDMENT}"
  --amendment-sha256 "${PAPER0_PGL_VARIOGRAM_AMENDMENT_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/preflight.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/control_magnitudes.json" >/dev/null
jq -e '.status == "passed" and .development_run == "85604" and .known_answer_gates.passed == true and .sampler_regression_gate.passed == true and .transport_equivalence_gate.passed == true and .screen_training_authorized == false and .future_truth_used_by_sampler == false and .held_out_85606_read == false and .held_out_run_read == false and .new_nersc_data_read == false and .new_segment_read == false' "${RUN_OUTPUT}/preflight.json" >/dev/null
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/preflight.json" \
  "${RUN_OUTPUT}/control_magnitudes.json" \
  "${RUN_OUTPUT}/pair_banks.npz" \
  "${RUN_OUTPUT}/native_transport_truth_train_0_432.npz" \
  > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,sampler_regression_gate,transport_equivalence_gate,pair_banks,native_transport_truth,control_magnitudes,peak_cuda_memory_GiB,wall_seconds}' "${RUN_OUTPUT}/preflight.json"
echo "Variogram preflight completed: ${JOB_ROOT}"
