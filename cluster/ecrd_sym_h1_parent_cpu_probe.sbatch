#!/bin/bash
# Non-scientific one-context CPU timing probe for the ECRD H1 parent.
# It reads no future truth, writes no forecast tensor, and trains no model.
#SBATCH --job-name=p0_ecrd_cpu_probe
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/ecrd_parent_cpu_probe_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly MODEL_DATA_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly H1_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
readonly CODEC_CHECKPOINT="/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_parent_cpu_probe"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly RESULT="${JOB_ROOT}/result.json"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/probe_ecrd_sym_h1_parent_cpu.py"
readonly NOTE="${PAPER0_ROOT}/paper0/protocol/ECRD_PARENT_CPU_TIMING_PROBE_2026-08-20.md"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite CPU timing probe output" >&2
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
  exit 2
fi

source /etc/os-release
if [[ "${ID}" != "rocky" || "${VERSION_ID%%.*}" != "9" ]]; then
  echo "ECRD CPU timing probe requires Rocky Linux 9" >&2
  exit 2
fi

scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export WANDB_MODE=disabled

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "scope=non_scientific_ECRD_four_phase_H1_CPU_timing_probe"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("torch_threads="+str(torch.get_num_threads()))'
} > "${JOB_ROOT}/environment.txt" 2>&1

COMMAND=(
  "${PYTHON}" -u "${ENTRYPOINT}"
  --artifact-root "${MODEL_DATA_SOURCE}"
  --h1-checkpoint "${H1_CHECKPOINT}"
  --codec-checkpoint "${CODEC_CHECKPOINT}"
  --output "${RESULT}"
  --paper0-commit "${PAPER0_COMMIT}"
  --slurm-job-id "${SLURM_JOB_ID}"
)
printf '%q ' "${COMMAND[@]}" > "${JOB_ROOT}/command.sh"
printf '\n' >> "${JOB_ROOT}/command.sh"
"${COMMAND[@]}" | tee "${JOB_ROOT}/stdout.jsonl"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
sha256sum \
  "${NOTE}" \
  "${ENTRYPOINT}" \
  "${JOB_ROOT}/command.sh" \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/stdout.jsonl" \
  "${RESULT}" \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "CPU timing probe complete; no forecast tensor or scientific metric was written."
