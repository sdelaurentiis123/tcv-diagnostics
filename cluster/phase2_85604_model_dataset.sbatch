#!/bin/bash
# Build one verified shared 85604 model dataset for matched O1/O2 experiments.
# CPU-only, Rocky-9-only, no training, and no held-out access.
#SBATCH --job-name=p2_model_data
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_model_dataset_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly SHARD_DIR="${OUTDIR}/shards"
readonly PARTIAL_DIR="${OUTDIR}/partials"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_model_dataset_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MODEL_DATASET_PROTOCOL.md"
readonly LIBRARY="${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"
readonly RESAMPLING="${PAPER0_ROOT}/src/tcv_diagnostics/resampling.py"
readonly BUILDER="${PAPER0_ROOT}/paper0/tools/build_85604_model_dataset_shard.py"
readonly MERGER="${PAPER0_ROOT}/paper0/tools/merge_85604_model_dataset_shards.py"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly NATIVE_TRAIN="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/train/TCV_85604_train.hdf5"
readonly NATIVE_VALID="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/valid/TCV_85604_valid.hdf5"
readonly LEGACY_TRAIN="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5"
readonly LEGACY_VALID="/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5"
readonly EXTRACTION_RECORD="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_vorticity_all_frame/job_6893033/extraction_record.json"
readonly SOURCE_PREFLIGHT="${OUTDIR}/source_sha256.txt"
readonly RESULT="${OUTDIR}/model_dataset_manifest.json"
readonly NORMALIZATION="${OUTDIR}/normalization.json"
readonly SHARD_COUNT=8

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${SHARD_DIR}" "${PARTIAL_DIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: got ${PAPER0_COMMIT}, expected ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 worktree is dirty; refusing provenance-ambiguous execution" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

check_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: got ${actual}, expected ${expected}" >&2
    exit 2
  fi
  echo "verified ${actual}  ${path}"
}

record_source_sha256() {
  local expected="$1"
  local path="$2"
  local record="$3"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: got ${actual}, expected ${expected}" >&2
    exit 2
  fi
  echo "${actual}  ${path}" >> "${record}"
  echo "verified source ${actual}  ${path}"
}

echo "Phase 2 model-dataset build started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "Slurm job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

check_sha256 "f60bcb4109c55b2927017017e0a59fc43173fbb8f648525c651f72656c910c97" "${MANIFEST}"
check_sha256 "3d6ffe8d0805e42b46c141a1af2b2fbf5a002cec4ad9c94a980614df0046575d" "${PROTOCOL}"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${LIBRARY}"
check_sha256 "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e" "${RESAMPLING}"
check_sha256 "3020c5262cc7dab1e14bafbb200a19229159469ceeff2e62def7483f0e9e8d13" "${BUILDER}"
check_sha256 "f590e6108db5f5f8ac53ddc62bbd96b8576303d675e861c3d38ee8b47235ebc2" "${MERGER}"

: > "${SOURCE_PREFLIGHT}"
record_source_sha256 "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777" "${NATIVE_TRAIN}" "${SOURCE_PREFLIGHT}"
record_source_sha256 "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba" "${NATIVE_VALID}" "${SOURCE_PREFLIGHT}"
record_source_sha256 "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817" "${LEGACY_TRAIN}" "${SOURCE_PREFLIGHT}"
record_source_sha256 "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43" "${LEGACY_VALID}" "${SOURCE_PREFLIGHT}"
record_source_sha256 "2922c0faadf23a68b43189890f6c0de69dd3cc03442b827c1b8f7e95b8d61e63" "${EXTRACTION_RECORD}" "${SOURCE_PREFLIGHT}"

scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"
{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py, netCDF4, numpy, scipy; print("h5py=" + h5py.__version__); print("netCDF4=" + netCDF4.__version__); print("numpy=" + numpy.__version__); print("scipy=" + scipy.__version__)'
} > "${OUTDIR}/environment.txt" 2>&1

{
  for ((shard = 0; shard < SHARD_COUNT; shard++)); do
    echo "srun --exclusive --exact --ntasks=1 --cpus-per-task=2 --mem=5G ${PYTHON} ${BUILDER} --manifest ${MANIFEST} --protocol ${PROTOCOL} --output-root ${OUTDIR} --source-preflight ${SOURCE_PREFLIGHT} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --shard-index ${shard}"
  done
  echo "${PYTHON} ${MERGER} --manifest ${MANIFEST} --protocol ${PROTOCOL} --source-preflight ${SOURCE_PREFLIGHT} --partial-output ${PARTIAL_DIR}/partial_[0-7].json --output ${RESULT} --normalization-output ${NORMALIZATION} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --artifact-root ${OUTDIR}"
} > "${OUTDIR}/commands.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
declare -a SHARD_PIDS=()
for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  srun \
    --exclusive \
    --exact \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=5G \
    "${PYTHON}" \
    "${BUILDER}" \
    --manifest "${MANIFEST}" \
    --protocol "${PROTOCOL}" \
    --output-root "${OUTDIR}" \
    --source-preflight "${SOURCE_PREFLIGHT}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}" \
    --shard-index "${shard}" &
  SHARD_PIDS+=("$!")
done

SHARD_STATUS=0
for pid in "${SHARD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    SHARD_STATUS=1
  fi
done
if [[ "${SHARD_STATUS}" -ne 0 ]]; then
  echo "At least one model-dataset shard failed; refusing incomplete merge" >&2
  exit 2
fi

mapfile -t PARTIAL_OUTPUTS < <(
  find "${PARTIAL_DIR}" -maxdepth 1 -name 'partial_*.json' -type f | sort -V
)
mapfile -t SHARD_OUTPUTS < <(
  find "${SHARD_DIR}" -maxdepth 1 -name 'model_dataset_shard_*.h5' -type f | sort -V
)
if [[ "${#PARTIAL_OUTPUTS[@]}" -ne "${SHARD_COUNT}" ]]; then
  echo "Expected ${SHARD_COUNT} partial outputs; found ${#PARTIAL_OUTPUTS[@]}" >&2
  exit 2
fi
if [[ "${#SHARD_OUTPUTS[@]}" -ne "${SHARD_COUNT}" ]]; then
  echo "Expected ${SHARD_COUNT} HDF5 shards; found ${#SHARD_OUTPUTS[@]}" >&2
  exit 2
fi
for partial in "${PARTIAL_OUTPUTS[@]}"; do
  "${PYTHON}" -m json.tool "${partial}" >/dev/null
done

PYTHONPATH="${PAPER0_ROOT}/src" \
  "${PYTHON}" \
  "${MERGER}" \
  --manifest "${MANIFEST}" \
  --protocol "${PROTOCOL}" \
  --source-preflight "${SOURCE_PREFLIGHT}" \
  --partial-output "${PARTIAL_OUTPUTS[@]}" \
  --output "${RESULT}" \
  --normalization-output "${NORMALIZATION}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --artifact-root "${OUTDIR}"

"${PYTHON}" -m json.tool "${RESULT}" >/dev/null
"${PYTHON}" -m json.tool "${NORMALIZATION}" >/dev/null
sha256sum \
  "${OUTDIR}/commands.sh" \
  "${OUTDIR}/environment.txt" \
  "${OUTDIR}/slurm_job.txt" \
  "${SOURCE_PREFLIGHT}" \
  "${PARTIAL_OUTPUTS[@]}" \
  "${SHARD_OUTPUTS[@]}" \
  "${NORMALIZATION}" \
  "${RESULT}" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 model-dataset build finished at $(date --iso-8601=seconds)"
echo "Result directory: ${OUTDIR}"
echo "Training remains closed pending the matched O1/O2 model protocol."
