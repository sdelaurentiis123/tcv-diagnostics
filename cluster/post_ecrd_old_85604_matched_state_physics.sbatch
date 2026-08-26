#!/bin/bash
# Evaluation-only common-view physics scoring for the frozen C5P/E6B pair.
#SBATCH --job-name=p0_old_state_phys
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_state_physics_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_MATCHED_PHYSICS_MANIFEST:?submit with frozen physics manifest set}"
: "${PAPER0_MATCHED_PHYSICS_MANIFEST_SHA256:?submit with manifest SHA-256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/score_matched_state_bounded_physics.py"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_bounded_physics"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/scoring"
readonly WANDB_ENTITY="sdelaurentiis123-columbia-university"
readonly WANDB_PROJECT="tcv-diagnostics-paper0"
readonly WANDB_GROUP="post-ecrd-old-85604-matched-state-bounded-physics"
readonly RUN_ID="p0oldstatephys-j${SLURM_JOB_ID}"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing paired physics result" >&2
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
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous scoring" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" || ! -f "${ENTRYPOINT}" || ! -f "${PAPER0_MATCHED_PHYSICS_MANIFEST}" ]]; then
  echo "Audited Python, entrypoint, or frozen manifest is unavailable" >&2
  exit 2
fi
readonly ACTUAL_MANIFEST_SHA="$(sha256sum "${PAPER0_MATCHED_PHYSICS_MANIFEST}" | awk '{print $1}')"
if [[ "${ACTUAL_MANIFEST_SHA}" != "${PAPER0_MATCHED_PHYSICS_MANIFEST_SHA256}" ]]; then
  echo "Paired physics manifest SHA-256 differs" >&2
  exit 2
fi

source /etc/os-release
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=post_ecrd_old_85604_matched_state_bounded_physics"
  echo "development_run=85604"
  echo "manifest=${PAPER0_MATCHED_PHYSICS_MANIFEST}"
  echo "manifest_sha256=${PAPER0_MATCHED_PHYSICS_MANIFEST_SHA256}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,wandb; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("wandb="+wandb.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export WANDB_MODE=online
export WANDB_REQUIRE_SERVICE=true
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_matched_state_bounded_physics.py \
  tests/test_matched_state_physics_freeze.py \
  tests/test_matched_state_exact_phi_assembly.py \
  tests/test_matched_codec_evaluation.py \
  tests/test_bounded_rollout_physics.py \
  tests/test_phase2_transport_partial.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --manifest "${PAPER0_MATCHED_PHYSICS_MANIFEST}"
  --manifest-sha256 "${PAPER0_MATCHED_PHYSICS_MANIFEST_SHA256}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-run-id "${RUN_ID}"
  --wandb-run-name "old-85604-matched-state-bounded-physics-j${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RUN_OUTPUT}/result.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/physics_metrics.json" >/dev/null
"${PYTHON}" -m json.tool "${RUN_OUTPUT}/wandb.json" >/dev/null
jq -e '.status == "paired_bounded_physics_scored" and .training_performed == false and .checkpoint_selection_performed == false and .physics_derived_loss_used == false and .held_out_85606_read == false and .new_nersc_data_read == false and .assimilation_performed == false and .diagnostic_ranking_performed == false and .steering_performed == false' \
  "${RUN_OUTPUT}/result.json" >/dev/null
jq -e '.required == true and .mode == "online" and .remote_state_after_finish == "finished"' \
  "${RUN_OUTPUT}/wandb.json" >/dev/null

sha256sum \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${RUN_OUTPUT}/artifact_sha256.txt" \
  "${RUN_OUTPUT}/result.json" \
  "${RUN_OUTPUT}/physics_metrics.json" \
  "${RUN_OUTPUT}/primary_metrics.csv" \
  "${RUN_OUTPUT}/separatrix_transport_series.npz" \
  "${RUN_OUTPUT}/wandb.json" \
  > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{status,decision}' "${RUN_OUTPUT}/result.json"
echo "Paired old-85604 state-view physics scoring completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
