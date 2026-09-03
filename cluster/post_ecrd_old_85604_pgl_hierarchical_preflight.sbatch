#!/bin/bash
# Hash-close hierarchical transport controls and the 25% gradient multiplier.
# Submit from rusty9 so the automatically applied reservation matches the OS.
#SBATCH --job-name=p0_pgl_hier_pre
#SBATCH --partition=gpu
#SBATCH --qos=gen
#SBATCH --gres=gpu:h100_pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_hierarchical_preflight_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256:?set evidence-manifest SHA-256}"
: "${PAPER0_PGL_HIERARCHICAL_PROTOCOL_SHA256:?set hierarchical protocol SHA-256}"
: "${PAPER0_PGL_PRIOR_PREFLIGHT_SHA256:?set prior preflight SHA-256}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ARTIFACT_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly EVIDENCE_MANIFEST="${PAPER0_ROOT}/paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/POST_ECRD_OLD_85604_PGL_HIERARCHICAL_TRANSPORT_TRAINING_2026-09-02.md"
readonly PRIOR_PREFLIGHT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_variogram_preflight/job_6958597/run/preflight.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/preflight_pgl_hierarchical_transport.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_hierarchical_transport"
readonly JOB_ROOT="${RESULT_BASE}/preflight_job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite hierarchical preflight output" >&2
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
if [[ "$(sha256sum "${EVIDENCE_MANIFEST}" | awk '{print $1}')" != "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}" ]]; then
  echo "Evidence-manifest SHA-256 differs" >&2
  exit 2
fi
if [[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" != "${PAPER0_PGL_HIERARCHICAL_PROTOCOL_SHA256}" ]]; then
  echo "Hierarchical protocol SHA-256 differs" >&2
  exit 2
fi
if [[ "$(sha256sum "${PRIOR_PREFLIGHT}" | awk '{print $1}')" != "${PAPER0_PGL_PRIOR_PREFLIGHT_SHA256}" ]]; then
  echo "Prior preflight SHA-256 differs" >&2
  exit 2
fi

git -C "${PAPER0_ROOT}" status --short --branch > "${JOB_ROOT}/git_status.txt"
git -C "${PAPER0_ROOT}" log -1 --format=fuller > "${JOB_ROOT}/git_commit.txt"
scontrol show job "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
env | sort > "${JOB_ROOT}/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_pgl_hierarchical_transport.py \
  tests/test_pgl_hierarchical_training.py \
  tests/test_pgl_variogram.py \
  tests/test_pgl_torch_transport.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${ARTIFACT_ROOT}"
  --evidence-manifest "${EVIDENCE_MANIFEST}"
  --evidence-manifest-sha256 "${PAPER0_PGL_EVIDENCE_MANIFEST_SHA256}"
  --protocol "${PROTOCOL}"
  --protocol-sha256 "${PAPER0_PGL_HIERARCHICAL_PROTOCOL_SHA256}"
  --prior-preflight "${PRIOR_PREFLIGHT}"
  --prior-preflight-sha256 "${PAPER0_PGL_PRIOR_PREFLIGHT_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/preflight.json" >/dev/null
jq -e '.status == "passed" and .known_answer_gates.passed == true and .gradient_calibration.observed_ratio >= 0.2499 and .gradient_calibration.observed_ratio <= 0.2501 and .screen_training_authorized == false and .held_out_85606_read == false and .new_nersc_data_read == false' "${RUN_OUTPUT}/preflight.json" >/dev/null
sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/git_commit.txt" \
  "${JOB_ROOT}/git_status.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/control_magnitudes.json" \
  "${RUN_OUTPUT}/gradient_calibration.json" \
  "${RUN_OUTPUT}/preflight.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,gradient_calibration,peak_cuda_memory_GiB,wall_seconds}' "${RUN_OUTPUT}/preflight.json"
echo "Hierarchical transport preflight completed: ${JOB_ROOT}"
