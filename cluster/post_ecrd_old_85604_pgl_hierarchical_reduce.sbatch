#!/bin/bash
# Reduce six completed M32 score records using the frozen two-epoch rule.
#SBATCH --job-name=p0_pgl_hier_reduce
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_pgl_hierarchical_reduce_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?set PAPER0_EXPECTED_COMMIT}"
: "${PAPER0_PGL_HIERARCHICAL_SCORE_JOB_ID:?set completed score array job ID}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_pgl_hierarchical_transport/evaluation"
readonly SCORE_ROOT="${RESULT_BASE}/score_job_${PAPER0_PGL_HIERARCHICAL_SCORE_JOB_ID}"
readonly JOB_ROOT="${RESULT_BASE}/reduction_job_${SLURM_JOB_ID}"
readonly RUN_OUTPUT="${JOB_ROOT}/run"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/reduce_pgl_hierarchical_screen.py"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite hierarchical reduction output" >&2
  exit 2
fi
mkdir -p "${JOB_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]] || [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is not the locked clean commit" >&2
  exit 2
fi
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
sacct -j "${PAPER0_PGL_HIERARCHICAL_SCORE_JOB_ID}" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList -n -P > "${JOB_ROOT}/score_sacct.txt"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_pgl_hierarchical_decision.py \
  tests/test_pgl_hierarchical_reduction.py \
  | tee "${JOB_ROOT}/test_output.txt"

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --score-root "${SCORE_ROOT}"
  --output "${RUN_OUTPUT}"
  --paper0-root "${PAPER0_ROOT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

jq -e '.status == "completed" and .checkpoint_selection_performed == false and .held_out_85606_read == false and .new_nersc_data_read == false' "${RUN_OUTPUT}/decision.json" >/dev/null
sha256sum "${JOB_ROOT}/command.sh" "${JOB_ROOT}/score_sacct.txt" "${JOB_ROOT}/slurm_job.txt" "${JOB_ROOT}/test_output.txt" "${RUN_OUTPUT}/decision.json" "${RUN_OUTPUT}/checkpoint_metrics.csv" "${RUN_OUTPUT}/hierarchy_metrics.csv" "${RUN_OUTPUT}/variogram_curves.csv" "${RUN_OUTPUT}/DECISION_READOUT.md" "${RUN_OUTPUT}/manifest.json" > "${JOB_ROOT}/artifact_sha256.txt"
jq -c '{status,longer_4_to_8_epoch_extension_authorized,transport_arm_passed_production_gate,next_action,epoch_two_matched_difference}' "${RUN_OUTPUT}/decision.json"
