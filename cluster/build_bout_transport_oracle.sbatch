#!/bin/bash
# Build the exact BOUT++ revision needed for the Phase 2 transport oracle.
# This is a CPU-only build: it reads no shot data and launches no model.
#SBATCH --job-name=p2_bout_oracle_build
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_bout_build_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set to the committed launcher revision}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly EXTERNAL_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external"
readonly BOUT_SOURCE="${EXTERNAL_ROOT}/BOUT-dev-7d28d67"
readonly NETCDF_CXX_SOURCE="${EXTERNAL_ROOT}/netcdf-cxx4-a43d6d4"
readonly RESULT_BASE="${BOUT_BUILD_RESULT_BASE:-${EXTERNAL_ROOT}/builds}"
readonly OUTDIR="${RESULT_BASE}/bout_7d28d67_job_${SLURM_JOB_ID}"
readonly INSTALL_PREFIX="${OUTDIR}/install"

readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"
readonly EXPECTED_NETCDF_CXX_COMMIT="a43d6d4d415d407712c246faca553bd951730dc1"
readonly EXPECTED_CPPTRACE_COMMIT="027f9aee2d34dbe1c98f26224e1fbe1654cb4aae"
readonly EXPECTED_FMT_COMMIT="407c905e45ad75fc29bf0f9bb7c5c2fd3475976f"
readonly EXPECTED_MPARK_COMMIT="23cb94f027d4ef33bf48133acc2695c7e5c6f1e7"

if [[ -e "${OUTDIR}" ]]; then
  echo "Refusing to overwrite existing build directory: ${OUTDIR}" >&2
  exit 2
fi
mkdir -p "${OUTDIR}"
exec > >(tee -a "${OUTDIR}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch: got ${PAPER0_COMMIT}, expected ${PAPER0_EXPECTED_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 worktree is dirty; refusing provenance-ambiguous build" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi

check_revision() {
  local label="$1"
  local repository="$2"
  local expected="$3"
  local actual
  actual="$(git -C "${repository}" rev-parse HEAD)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "${label} revision mismatch: got ${actual}, expected ${expected}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${repository}" status --porcelain --untracked-files=no)" ]]; then
    echo "${label} tracked worktree is dirty: ${repository}" >&2
    git -C "${repository}" status --short >&2
    exit 2
  fi
  echo "verified ${label} revision ${actual}"
}

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

echo "Phase 2 BOUT++ oracle build started at $(date --iso-8601=seconds)"
echo "Paper 0 commit: ${PAPER0_COMMIT}"
echo "SLURM job: ${SLURM_JOB_ID}"
echo "Host: $(hostname --fqdn)"
cat /etc/os-release
source /etc/os-release
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This launcher is locked to Rocky 9; found VERSION_ID=${VERSION_ID}" >&2
  exit 2
fi

check_revision "BOUT++" "${BOUT_SOURCE}" "${EXPECTED_BOUT_COMMIT}"
check_revision "netCDF C++" "${NETCDF_CXX_SOURCE}" "${EXPECTED_NETCDF_CXX_COMMIT}"
check_revision "cpptrace submodule" "${BOUT_SOURCE}/externalpackages/cpptrace" "${EXPECTED_CPPTRACE_COMMIT}"
check_revision "fmt submodule" "${BOUT_SOURCE}/externalpackages/fmt" "${EXPECTED_FMT_COMMIT}"
check_revision "mpark.variant submodule" "${BOUT_SOURCE}/externalpackages/mpark.variant" "${EXPECTED_MPARK_COMMIT}"

check_sha256 "e00afb05e00ddfc1631c9afbd5d3e00ab50911e14399e4814e14ba9e8c1f336f" "${BOUT_SOURCE}/src/mesh/coordinates.cxx"
check_sha256 "7b89f0b7b859c23ac504dde62bbf7b58dbe81fa6b5eea73cbadaa14d422eb6e9" "${BOUT_SOURCE}/src/mesh/difops.cxx"
check_sha256 "5e237be2344dd588e110304dd6aee0e370d6cbf4b12ae20485653545d4f373f6" "${BOUT_SOURCE}/include/bout/coordinates.hxx"
check_sha256 "761bf5b8488edb21feb8c512860bf2f2a9283e4d6d9c7200bc0f6b1598fafe56" "${BOUT_SOURCE}/src/mesh/parallel/shiftedmetric.cxx"
check_sha256 "55319fa4197c3904f8bf63ee27e0332a1c218558cdf0552838b931bb4560bd8a" "${BOUT_SOURCE}/src/sys/derivs.cxx"

module purge
module load gcc/13.3.0
module load openmpi/4.1.8
module load cmake/3.31.6
module load ninja/1.12.1
module load hdf5/1.12.3
module load netcdf-c/4.9.2
module load fftw/3.3.10

{
  echo "cc=$(command -v gcc)"
  echo "cxx=$(command -v g++)"
  echo "mpicxx=$(command -v mpicxx)"
  echo "cmake=$(command -v cmake)"
  echo "ninja=$(command -v ninja)"
  gcc --version | head -1
  mpicxx --showme:version
  cmake --version | head -1
  nc-config --all
  pkg-config --modversion fftw3
  module list
} > "${OUTDIR}/build_environment.txt" 2>&1

NETCDF_CONFIGURE=(
  cmake
  -S "${NETCDF_CXX_SOURCE}"
  -B "${OUTDIR}/netcdf-cxx4-build"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  -DNCXX_ENABLE_TESTS=OFF
  -DBUILD_SHARED_LIBS=ON
)
printf '%q ' "${NETCDF_CONFIGURE[@]}" > "${OUTDIR}/configure_netcdf_cxx.sh"
printf '\n' >> "${OUTDIR}/configure_netcdf_cxx.sh"
"${NETCDF_CONFIGURE[@]}"
cmake --build "${OUTDIR}/netcdf-cxx4-build" --parallel "${SLURM_CPUS_PER_TASK}"
cmake --install "${OUTDIR}/netcdf-cxx4-build"

BOUT_CONFIGURE=(
  cmake
  -S "${BOUT_SOURCE}"
  -B "${OUTDIR}/bout-build"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
  -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}"
  -DBOUT_ENABLE_MPI=ON
  -DBOUT_ENABLE_OPENMP=OFF
  -DBOUT_USE_NETCDF=ON
  -DBOUT_USE_FFTW=ON
  -DBOUT_USE_PVODE=OFF
  -DBOUT_USE_PETSC=OFF
  -DBOUT_USE_SUNDIALS=OFF
  -DBOUT_USE_LAPACK=OFF
  -DBOUT_USE_NLS=OFF
)
printf '%q ' "${BOUT_CONFIGURE[@]}" > "${OUTDIR}/configure_bout.sh"
printf '\n' >> "${OUTDIR}/configure_bout.sh"
"${BOUT_CONFIGURE[@]}"
cmake --build "${OUTDIR}/bout-build" --parallel "${SLURM_CPUS_PER_TASK}"
cmake --install "${OUTDIR}/bout-build"

if [[ ! -x "${INSTALL_PREFIX}/bin/bout-config" ]]; then
  echo "Installed bout-config is missing: ${INSTALL_PREFIX}/bin/bout-config" >&2
  exit 2
fi
"${INSTALL_PREFIX}/bin/bout-config" --version > "${OUTDIR}/bout_version.txt"

sha256sum \
  "${OUTDIR}/build_environment.txt" \
  "${OUTDIR}/configure_netcdf_cxx.sh" \
  "${OUTDIR}/configure_bout.sh" \
  "${OUTDIR}/bout_version.txt" \
  > "${OUTDIR}/artifact_sha256.txt"

echo "Phase 2 BOUT++ oracle build completed at $(date --iso-8601=seconds)"
echo "Install prefix: ${INSTALL_PREFIX}"
