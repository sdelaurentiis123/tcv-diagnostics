#!/bin/bash
# Exact causal potential reconstruction for predicted E6B bounded forecasts.
# Each of the seven frozen forecast candidates is solved sequentially with the
# validated Hermes/BOUT++ elliptic operator.  Candidate mode uses predicted
# Ne/Pe/Pi/Vort and predicted Bphi boundaries; it never reads target phi.
#SBATCH --job-name=p0_old_state_phi
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/old_85604_state_exact_phi_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"
: "${PAPER0_E6B_GENERATION_RESULT:?submit with E6B generation result set}"
: "${PAPER0_E6B_GENERATION_RESULT_SHA256:?submit with result SHA-256 set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly SOURCE="${PAPER0_ROOT}/paper0/oracles/matched_e6b_elliptic"
readonly BOUT_INPUT="${PAPER0_ROOT}/paper0/oracles/potential_elliptic/BOUT.inp"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_matched_e6b_phi.py"
readonly ASSEMBLER="${PAPER0_ROOT}/paper0/tools/assemble_matched_state_exact_phi.py"
readonly TRUTH_REPLAY="${PAPER0_ROOT}/paper0/results/phase2_matched_e6b_truth_replay_6894325.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_exact_phi"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly PHI_ROOT="${JOB_ROOT}/candidates"
readonly BUILD_DIR="${JOB_ROOT}/build"
readonly FINAL_RESULT="${JOB_ROOT}/result.json"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing exact-phi result: ${JOB_ROOT}" >&2
  exit 2
fi
mkdir -p "${PHI_ROOT}"
exec > >(tee -a "${JOB_ROOT}/job.log") 2>&1

readonly PAPER0_COMMIT="$(git -C "${PAPER0_ROOT}" rev-parse HEAD)"
if [[ "${PAPER0_COMMIT}" != "${PAPER0_EXPECTED_COMMIT}" ]]; then
  echo "Paper 0 commit mismatch" >&2
  exit 2
fi
if [[ -n "$(git -C "${PAPER0_ROOT}" status --porcelain --untracked-files=all)" ]]; then
  echo "Paper 0 checkout is dirty; refusing provenance-ambiguous solve" >&2
  git -C "${PAPER0_ROOT}" status --short >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" || ! -f "${PAPER0_E6B_GENERATION_RESULT}" ]]; then
  echo "Audited Python or E6B generation result is unavailable" >&2
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
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "The pinned exact-phi stack requires Rocky 9" >&2
  exit 2
fi
if [[ "$(git -C "${BOUT_SOURCE}" rev-parse HEAD)" != "${EXPECTED_BOUT_COMMIT}" ]]; then
  echo "BOUT++ source revision differs" >&2
  exit 2
fi
if [[ "$(git -C "${HERMES_SOURCE}" rev-parse HEAD)" != "${EXPECTED_HERMES_COMMIT}" ]]; then
  echo "Hermes source revision differs" >&2
  exit 2
fi

check_sha256 "${PAPER0_E6B_GENERATION_RESULT_SHA256}" "${PAPER0_E6B_GENERATION_RESULT}"
check_sha256 "7c13a60c26238acbe0db0ed6eb0e3c18f8b18f570bb7f22913642bae9bce22e5" "${SOURCE}/CMakeLists.txt"
check_sha256 "d3d1addaa421ddc005cd4ecfedfe303370034c43de6137ef648e3b116ceb8ab4" "${SOURCE}/matched_e6b_elliptic_oracle.cxx"
check_sha256 "622841b559c3c4132444c0466915aecc35cbd6801242d94fafb0042a0acda5e4" "${EXTRACTOR}"
check_sha256 "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2" "${BOUT_INPUT}"
check_sha256 "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27" "${TRUTH_REPLAY}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333" "${BOUT_INSTALL}/lib64/libbout++.so.5.2.0"
check_sha256 "b5d86b8adedf11f3fab2783cbc75fd7b1064e87d91c7df596437fd7d26dd11e6" "${BOUT_INSTALL}/bin/bout-config"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src:${PAPER0_ROOT}"
export OMP_NUM_THREADS=1
export PAPER0_PREFLIGHT_GENERATION_RESULT="${PAPER0_E6B_GENERATION_RESULT}"
export PAPER0_PREFLIGHT_GENERATION_SHA="${PAPER0_E6B_GENERATION_RESULT_SHA256}"
"${PYTHON}" - <<'PY' > "${JOB_ROOT}/preflight.json"
import hashlib
import json
import os
from pathlib import Path

path = Path(os.environ["PAPER0_PREFLIGHT_GENERATION_RESULT"])
result = json.loads(path.read_text(encoding="utf-8"))
expected_names = [
    "h4_direct_predicted_e6b_native81.h5",
    "h4_autoregressive_lead1_predicted_e6b_native81.h5",
    "h4_autoregressive_lead2_predicted_e6b_native81.h5",
    "h8_direct_predicted_e6b_native81.h5",
    "h8_autoregressive_lead1_predicted_e6b_native81.h5",
    "h8_autoregressive_lead2_predicted_e6b_native81.h5",
    "h8_autoregressive_lead4_predicted_e6b_native81.h5",
]
actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
if actual_sha != os.environ["PAPER0_PREFLIGHT_GENERATION_SHA"]:
    raise RuntimeError("generation-result hash differs")
if (
    result.get("scope")
    != "post_ecrd_old_85604_matched_state_bounded_generation"
    or result.get("status") != "completed"
    or result.get("development_run") != "85604"
    or result.get("family") != "e6b"
    or result.get("held_out_85606_read") is not False
    or result.get("new_nersc_data_read") is not False
    or result.get("target_truth_used_during_generation") is not False
):
    raise RuntimeError("E6B generation-result contract differs")
candidates = result.get("elliptic_candidates", [])
if [Path(str(item.get("path", ""))).name for item in candidates] != expected_names:
    raise RuntimeError("E6B elliptic-candidate set or order differs")
print(json.dumps({
    "candidate_count": len(candidates),
    "development_run": "85604",
    "held_out_85606_read": False,
    "new_nersc_data_read": False,
    "target_truth_phi_read": False,
    "boundary_policy": "predicted_Bphi_no_truth_bypass",
}, sort_keys=True))
PY

module purge
module load gcc/13.3.0
module load openmpi/4.1.8
module load cmake/3.31.6
module load ninja/1.12.1
module load hdf5/1.12.3
module load netcdf-c/4.9.2
module load fftw/3.3.10
module list 2> "${JOB_ROOT}/modules.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "operating_system=${PRETTY_NAME}"
  echo "paper0_commit=${PAPER0_COMMIT}"
  echo "generation_result=${PAPER0_E6B_GENERATION_RESULT}"
  echo "generation_result_sha256=${PAPER0_E6B_GENERATION_RESULT_SHA256}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,netCDF4,numpy; print("h5py="+h5py.__version__); print("netCDF4="+netCDF4.__version__); print("numpy="+numpy.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q \
  tests/test_matched_state_exact_phi_assembly.py \
  tests/test_phase2_matched_e6b_candidate_phi_r2_launcher.py \
  | tee "${JOB_ROOT}/test_output.txt"

cmake \
  -S "${SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}" \
  -DnetCDF_DIR="$(nc-config --prefix)/lib64/cmake/netCDF"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"
readonly EXECUTABLE="${BUILD_DIR}/matched_e6b_elliptic_oracle"

mapfile -t CANDIDATE_RECORDS < <(
  "${PYTHON}" -c '
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for record in result["elliptic_candidates"]:
    print(str(record["path"]) + "\t" + str(record["sha256"]))
' "${PAPER0_E6B_GENERATION_RESULT}"
)
if [[ "${#CANDIDATE_RECORDS[@]}" -ne 7 ]]; then
  echo "Expected seven frozen E6B candidates" >&2
  exit 2
fi

for record in "${CANDIDATE_RECORDS[@]}"; do
  IFS=$'\t' read -r candidate candidate_sha <<< "${record}"
  check_sha256 "${candidate_sha}" "${candidate}"
  stem="$(basename "${candidate}" .h5)"
  task_root="${PHI_ROOT}/${stem}"
  run_dir="${task_root}/run"
  phi="${task_root}/derived_phi.h5"
  result="${task_root}/result.json"
  mkdir -p "${run_dir}"
  cp "${BOUT_INPUT}" "${run_dir}/BOUT.inp"
  echo "Solving exact phi for ${stem}"
  srun --ntasks=4 \
    "${EXECUTABLE}" \
    -d "${run_dir}" \
    "paper0:input_file=${candidate}" \
    "paper0:truth_layout=false"
  mapfile -t outputs < <(
    find "${run_dir}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V
  )
  if [[ "${#outputs[@]}" -ne 4 ]]; then
    echo "Expected four BOUT outputs for ${stem}" >&2
    exit 2
  fi
  "${PYTHON}" \
    "${EXTRACTOR}" \
    --input "${candidate}" \
    --bout-output "${outputs[@]}" \
    --output-phi "${phi}" \
    --output-result "${result}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:${stem}"
  jq -e '.status == "completed" and .held_out_85606_read == false and .truth_layout == false and .truth_replay_gate == null' \
    "${result}" >/dev/null
  sha256sum "${candidate}" "${run_dir}/BOUT.inp" "${outputs[@]}" "${phi}" "${result}" \
    > "${task_root}/artifact_sha256.txt"
done

"${PYTHON}" \
  "${ASSEMBLER}" \
  --generation-result "${PAPER0_E6B_GENERATION_RESULT}" \
  --generation-result-sha256 "${PAPER0_E6B_GENERATION_RESULT_SHA256}" \
  --phi-root "${PHI_ROOT}" \
  --paper0-commit "${PAPER0_COMMIT}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --output "${FINAL_RESULT}"

jq -e '.status == "completed" and .candidate_count == 7 and .target_truth_phi_read == false and .held_out_85606_read == false and .new_nersc_data_read == false and .paired_common_view_physics_scoring_authorized == true' \
  "${FINAL_RESULT}" >/dev/null
sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/preflight.json" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${FINAL_RESULT}" \
  > "${JOB_ROOT}/artifact_sha256.txt"

jq -c '{status,candidate_count,target_truth_phi_read,paired_common_view_physics_scoring_authorized}' \
  "${FINAL_RESULT}"
echo "Exact causal phi reconstruction completed at $(date --iso-8601=seconds)"
echo "Results: ${JOB_ROOT}"
