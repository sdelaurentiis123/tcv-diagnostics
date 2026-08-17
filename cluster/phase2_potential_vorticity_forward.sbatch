#!/bin/bash
# Source-matched potential-to-vorticity forward closure on five frozen frames.
# CPU-only; all predecessor inputs and results are reused read-only.
#SBATCH --job-name=p2_vort_forward
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_vort_forward_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly GRID="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly CANONICAL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_elliptic/job_6892446/canonical_potential_inputs.nc"
readonly EXTRACTION_RECORD="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_elliptic/job_6892446/extraction_record.json"
readonly ACCEPTED_INVERSE_JOB="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_elliptic_runtime_pressure/job_6892641"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_potential_vorticity_forward"
readonly OUTDIR="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${OUTDIR}/build"
readonly RUN_DIR="${OUTDIR}/run"
readonly ORACLE_SOURCE="${PAPER0_ROOT}/paper0/oracles/potential_vorticity_forward"
readonly DRIVER="${ORACLE_SOURCE}/potential_vorticity_forward_oracle.cxx"
readonly BOUT_INPUT="${PAPER0_ROOT}/paper0/oracles/potential_elliptic/BOUT.inp"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_potential_vorticity_forward_85604.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_POTENTIAL_VORTICITY_FORWARD_PROTOCOL.md"
readonly ACCEPTED_INVERSE_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_elliptic_runtime_pressure_6892641.json"
readonly COMPARATOR="${PAPER0_ROOT}/paper0/tools/compare_potential_vorticity_forward_oracle.py"
readonly FINAL_COMPARISON="${OUTDIR}/potential_vorticity_forward_comparison.json"
readonly COMPARISON_ARRAYS="${OUTDIR}/comparison_arrays.npz"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing result directory: ${OUTDIR}" >&2
  exit 2
fi
if [[ ! -d "${ACCEPTED_INVERSE_JOB}" ]]; then
  echo "Missing immutable accepted inverse job: ${ACCEPTED_INVERSE_JOB}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}" "${RUN_DIR}"
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

echo "Phase 2 potential-vorticity forward oracle started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi
if [[ "$(git -C "${HERMES_SOURCE}" rev-parse HEAD)" != "${EXPECTED_HERMES_COMMIT}" ]]; then
  echo "Hermes source revision mismatch" >&2
  exit 2
fi
if [[ "$(git -C "${BOUT_SOURCE}" rev-parse HEAD)" != "${EXPECTED_BOUT_COMMIT}" ]]; then
  echo "BOUT++ source revision mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${HERMES_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
  echo "Hermes tracked source is dirty" >&2
  exit 2
fi
if [[ -n "$(git -C "${BOUT_SOURCE}" status --porcelain --untracked-files=no)" ]]; then
  echo "BOUT++ tracked source is dirty" >&2
  exit 2
fi

check_sha256 "4d672514d7d8106e84e39610ee9c18a66d48b7efc6f478a5ba1cc6d530b66789" "${MANIFEST}"
check_sha256 "67d2dfe95eb4e3c24d55dc271fd8d624b514869ae7cf002cba338fecaead2b1d" "${PROTOCOL}"
check_sha256 "516527bea146d2ccc258e210e4058f233ecb66ca030fa48298030224919ac487" "${DRIVER}"
check_sha256 "c92774a770adf1788aafe3eadf33aabc6e130d0b893ffb737ab19c21efbeeaca" "${ORACLE_SOURCE}/CMakeLists.txt"
check_sha256 "c95674608a80a5b5aab0f177c9d8ce067c58dbfe9ae20c1437634ffef722d52d" "${COMPARATOR}"
check_sha256 "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2" "${BOUT_INPUT}"
check_sha256 "ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a" "${ACCEPTED_INVERSE_RESULT}"
check_sha256 "ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a" "${ACCEPTED_INVERSE_JOB}/potential_elliptic_runtime_pressure_comparison.json"
check_sha256 "e090b3a23fa6eedf8c37e74421c08bafd3eb513039fa7621b5d612a7e1cbba3e" "${CANONICAL}"
check_sha256 "e30e4f14dddfdff369387f9e8657b31a3e56bb4dc628c1e9ef5d86bd5bfd68be" "${EXTRACTION_RECORD}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GRID}"

check_sha256 "d3abc5e32cdad3ea9c42faf432dcaed465070ee5792cb62e550b9baaad9953e6" "${HERMES_SOURCE}/src/evolve_pressure.cxx"
check_sha256 "5e3b8055ae068481319a90329a3bdf3605b82b9024ee8273b210fd2be915bd85" "${HERMES_SOURCE}/include/hermes_utils.hxx"
check_sha256 "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a" "${HERMES_SOURCE}/src/vorticity.cxx"
check_sha256 "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded" "${HERMES_SOURCE}/include/vorticity.hxx"
check_sha256 "858752dd39c55caf7afd00fa84583b5e1c7944763f6497f6ffaa2530ba0e1fc3" "${BOUT_SOURCE}/include/bout/invert_laplace.hxx"
check_sha256 "ca8167cab0cf0ab71a9c135240d383394455bfc469a33858ccd90673da85c5b1" "${BOUT_SOURCE}/include/bout/fft.hxx"
check_sha256 "c7da57b5dcbda9f0e93cae7b847d4b5f43a21426ee8c938ff2b137fbee55310e" "${BOUT_SOURCE}/src/invert/fft_fftw.cxx"
check_sha256 "47c2cbe674c61e510cc96835823ba8a7b55f2a9b6c2a5c1f93d730c8dba5d157" "${BOUT_SOURCE}/src/invert/laplace/invert_laplace.cxx"
check_sha256 "568faa3c9bcf17732779868f28aa219f5e8beea2ad4f98c490b233ce9552d38a" "${BOUT_SOURCE}/src/invert/laplace/impls/cyclic/cyclic_laplace.cxx"
check_sha256 "1309f7eebf5fe076663224b1a66a08b1a3ae6ce7526d0f0b595d6537db5bf296" "${BOUT_SOURCE}/src/invert/laplace/impls/cyclic/cyclic_laplace.hxx"
check_sha256 "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333" "${BOUT_INSTALL}/lib64/libbout++.so.5.2.0"
check_sha256 "b5d86b8adedf11f3fab2783cbc75fd7b1064e87d91c7df596437fd7d26dd11e6" "${BOUT_INSTALL}/bin/bout-config"

module purge
module load gcc/13.3.0
module load openmpi/4.1.8
module load cmake/3.31.6
module load ninja/1.12.1
module load hdf5/1.12.3
module load netcdf-c/4.9.2
module load fftw/3.3.10
module list 2> "${OUTDIR}/modules.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${OUTDIR}/slurm_job.txt"

cp "${BOUT_INPUT}" "${RUN_DIR}/BOUT.inp"
{
  echo "cmake -S ${ORACLE_SOURCE} -B ${BUILD_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${BOUT_INSTALL}"
  echo "cmake --build ${BUILD_DIR} --parallel ${SLURM_CPUS_PER_TASK}"
  echo "srun --ntasks=4 ${BUILD_DIR}/potential_vorticity_forward_oracle -d ${RUN_DIR} paper0:input_file=${CANONICAL}"
  echo "python ${COMPARATOR} --manifest ${MANIFEST} --protocol ${PROTOCOL} --canonical ${CANONICAL} --extraction-record ${EXTRACTION_RECORD} --bout-output ${RUN_DIR}/BOUT.dmp.[0-3].nc --grid ${GRID} --output ${FINAL_COMPARISON} --arrays ${COMPARISON_ARRAYS} --paper0-commit ${PAPER0_COMMIT} --slurm-job-id ${SLURM_JOB_ID} --atol 5e-10 --rtol 5e-10"
} > "${OUTDIR}/commands.sh"

cmake \
  -S "${ORACLE_SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}" \
  -DnetCDF_DIR="$(nc-config --prefix)/lib64/cmake/netCDF"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS=1
srun --ntasks=4 \
  "${BUILD_DIR}/potential_vorticity_forward_oracle" \
  -d "${RUN_DIR}" \
  "paper0:input_file=${CANONICAL}"

mapfile -t BOUT_OUTPUTS < <(find "${RUN_DIR}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V)
if [[ "${#BOUT_OUTPUTS[@]}" -ne 4 ]]; then
  echo "Expected four BOUT++ rank outputs; found ${#BOUT_OUTPUTS[@]}" >&2
  exit 2
fi

set +e
PYTHONPATH="${PAPER0_ROOT}/src" \
  /mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python \
  "${COMPARATOR}" \
  --manifest "${MANIFEST}" \
  --protocol "${PROTOCOL}" \
  --canonical "${CANONICAL}" \
  --extraction-record "${EXTRACTION_RECORD}" \
  --bout-output "${BOUT_OUTPUTS[@]}" \
  --grid "${GRID}" \
  --output "${FINAL_COMPARISON}" \
  --arrays "${COMPARISON_ARRAYS}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --atol 5e-10 \
  --rtol 5e-10
readonly COMPARISON_STATUS=$?
set -e

HASH_ARTIFACTS=(
  "${OUTDIR}/commands.sh"
  "${OUTDIR}/modules.txt"
  "${OUTDIR}/slurm_job.txt"
  "${RUN_DIR}/BOUT.inp"
  "${BUILD_DIR}/potential_vorticity_forward_oracle"
  "${BOUT_OUTPUTS[@]}"
)
for optional in "${FINAL_COMPARISON}" "${COMPARISON_ARRAYS}"; do
  if [[ -f "${optional}" ]]; then
    HASH_ARTIFACTS+=("${optional}")
  fi
done
sha256sum "${HASH_ARTIFACTS[@]}" > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 potential-vorticity forward oracle finished at $(date --iso-8601=seconds)"
echo "Comparison exit status: ${COMPARISON_STATUS}"
echo "Result directory: ${OUTDIR}"
exit "${COMPARISON_STATUS}"
