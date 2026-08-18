#!/bin/bash
# Recover exact phi for the three frozen R2 E6B codec candidates.
# CPU-only and sequential: every solve consumes the full four-rank allocation.
# This stage cannot decide O1 or authorize O2; transport finalization follows.
#SBATCH --job-name=p2_e6b_phi_r2
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_e6b_candidate_phi_r2_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics-eval-8a6e20d}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly SOURCE="${PAPER0_ROOT}/paper0/oracles/matched_e6b_elliptic"
readonly BOUT_INPUT="${PAPER0_ROOT}/paper0/oracles/potential_elliptic/BOUT.inp"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_matched_e6b_phi.py"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_e6b_candidate_phi_r2.json"
readonly RECONSTRUCTION_SUMMARY="${PAPER0_ROOT}/paper0/results/phase2_matched_o1_reconstruction_r2_6894838.json"
readonly TRUTH_REPLAY="${PAPER0_ROOT}/paper0/results/phase2_matched_e6b_truth_replay_6894325.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly R2_RUN_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_o1_codec_r2_runs.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${JOB_ROOT}/build"
readonly FINAL_RESULT="${JOB_ROOT}/candidate_phi_summary.json"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

readonly -a run_indices=(1 3 5)
readonly -a seeds=(1701 1702 1703)
readonly -a reconstruction_results=(
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_1_e6b_seed_1701/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_3_e6b_seed_1702/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_5_e6b_seed_1703/result.json"
)
readonly -a reconstruction_result_hashes=(
  9d0caf7dbede9c9db588d84602f0b8024bfd2cffe0e60b23bca3163a5fdc77d8
  8c016ae3ee64ca7f715d8a4aa90dfdcc6cfd99729972dc603277daec9f9a5924
  84748bacc58d96354c9dfedd87707f657d75d2214814190fe92a6028440538c5
)
readonly -a training_candidates=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_1_e6b_seed_1701/train_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_3_e6b_seed_1702/train_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_5_e6b_seed_1703/train_candidate_native81.h5"
)
readonly -a training_candidate_hashes=(
  f59e913ec43e3e40163a5ce70abbac4d7512e736fea86f9111314b9236f0748c
  ecaeb325253baf4d9beaa4050ee5d62a8fd3e5805c5ba7e230ce3122b1f7b7ab
  d0ecba1db9fc6a9e51a27efd2f4b58f76db1195b44355fd1777f4c85321de92b
)
readonly -a validation_candidates=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_1_e6b_seed_1701/validation_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_3_e6b_seed_1702/validation_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_5_e6b_seed_1703/validation_candidate_native81.h5"
)
readonly -a validation_candidate_hashes=(
  0dcc7bc88ff251729ef37608bd842cda4174d5df08d2b18c9aa4b3c2ad353535
  0728369290f2037ce28376a2185655bf4195a5d6a8dd41f66e927c551557333e
  133d41f2b65309abb11a14c237a43d272bba65ae9fb3ef018af80b67ddea89be
)

if [[ -e "${JOB_ROOT}" ]]; then
  echo "Refusing to overwrite existing result directory: ${JOB_ROOT}" >&2
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
if [[ "${VERSION_ID%%.*}" != "9" ]]; then
  echo "This exact-phi evaluation requires Rocky 9" >&2
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

check_sha256 "7c13a60c26238acbe0db0ed6eb0e3c18f8b18f570bb7f22913642bae9bce22e5" "${SOURCE}/CMakeLists.txt"
check_sha256 "d3d1addaa421ddc005cd4ecfedfe303370034c43de6137ef648e3b116ceb8ab4" "${SOURCE}/matched_e6b_elliptic_oracle.cxx"
check_sha256 "622841b559c3c4132444c0466915aecc35cbd6801242d94fafb0042a0acda5e4" "${EXTRACTOR}"
check_sha256 "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2" "${BOUT_INPUT}"
check_sha256 "ef90f8865f84aaad06b4c0ac61430b8edd683828ffdb0a4ebee7dead152fe0fe" "${MANIFEST}"
check_sha256 "c091326385aa77ba245759dccb4a7dc6e55b7a905c317e4b3b7706ce1c8bf321" "${RECONSTRUCTION_SUMMARY}"
check_sha256 "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27" "${TRUTH_REPLAY}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "e3e73f6ca5df1699aa86f634988a19568471cfb310b97cf14a3b85d84dc1eca8" "${R2_RUN_MANIFEST}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333" "${BOUT_INSTALL}/lib64/libbout++.so.5.2.0"
check_sha256 "b5d86b8adedf11f3fab2783cbc75fd7b1064e87d91c7df596437fd7d26dd11e6" "${BOUT_INSTALL}/bin/bout-config"
for ((position = 0; position < 3; position++)); do
  check_sha256 "${reconstruction_result_hashes[position]}" "${reconstruction_results[position]}"
  check_sha256 "${training_candidate_hashes[position]}" "${training_candidates[position]}"
  check_sha256 "${validation_candidate_hashes[position]}" "${validation_candidates[position]}"
done

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
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export OMP_NUM_THREADS=1

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,netCDF4,numpy; print("h5py="+h5py.__version__); print("netCDF4="+netCDF4.__version__); print("numpy="+numpy.__version__)'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PAPER0_CANDIDATE_PHI_MANIFEST="${MANIFEST}"
export PAPER0_RECONSTRUCTION_SUMMARY="${RECONSTRUCTION_SUMMARY}"
export PAPER0_TRUTH_REPLAY="${TRUTH_REPLAY}"
"${PYTHON}" - <<'PY' > "${JOB_ROOT}/preflight.json"
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["PAPER0_CANDIDATE_PHI_MANIFEST"]).read_text())
reconstruction = json.loads(Path(os.environ["PAPER0_RECONSTRUCTION_SUMMARY"]).read_text())
truth = json.loads(Path(os.environ["PAPER0_TRUTH_REPLAY"]).read_text())
expected = [(1, 1701), (3, 1702), (5, 1703)]
if manifest["development_run"] != "85604" or manifest["held_out_85606_access_allowed"] is not False:
    raise RuntimeError("candidate-phi manifest development-data contract differs")
if [(int(run["run_index"]), int(run["seed"])) for run in manifest["runs"]] != expected:
    raise RuntimeError("candidate-phi manifest run order differs")
if reconstruction["status"] != "completed_pending_exact_e6b_phi_and_authoritative_transport":
    raise RuntimeError("R2 reconstruction summary status differs")
if reconstruction["complete_O1_decision_allowed"] is not False:
    raise RuntimeError("R2 reconstruction prematurely permits a complete decision")
selected = [
    (int(run["run_index"]), int(run["seed"]))
    for run in reconstruction["runs"]
    if run["family"] == "e6b"
]
if selected != expected:
    raise RuntimeError("R2 reconstruction E6B run order differs")
if truth["status"] != "pass" or truth["all_frames_passed"] is not True:
    raise RuntimeError("zero-seed exact-phi truth replay did not pass")
if truth["coverage"] != [0, 624] or int(truth["frame_count"]) != 624:
    raise RuntimeError("zero-seed exact-phi truth replay coverage differs")
if manifest["output_contract"]["complete_o1_decision_allowed"] is not False:
    raise RuntimeError("candidate-phi manifest prematurely permits an O1 decision")
if manifest["output_contract"]["o2_launch_allowed_from_this_stage_alone"] is not False:
    raise RuntimeError("candidate-phi manifest prematurely permits O2")
print(json.dumps({
    "development_run": "85604",
    "held_out_85606_read": False,
    "e6b_runs": expected,
    "zero_seed_truth_replay": "pass",
    "truth_phi_read": False,
    "complete_o1_decision_allowed": False,
    "o2_launch_allowed": False,
}, sort_keys=True))
PY

cmake \
  -S "${SOURCE}" \
  -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${BOUT_INSTALL}" \
  -DnetCDF_DIR="$(nc-config --prefix)/lib64/cmake/netCDF"
cmake --build "${BUILD_DIR}" --parallel "${SLURM_CPUS_PER_TASK}"
readonly EXECUTABLE="${BUILD_DIR}/matched_e6b_elliptic_oracle"

run_candidate() {
  local position="$1"
  local split="$2"
  local input
  local expected_input_hash
  local run_index="${run_indices[position]}"
  local seed="${seeds[position]}"
  if [[ "${split}" == "training" ]]; then
    input="${training_candidates[position]}"
    expected_input_hash="${training_candidate_hashes[position]}"
  elif [[ "${split}" == "validation" ]]; then
    input="${validation_candidates[position]}"
    expected_input_hash="${validation_candidate_hashes[position]}"
  else
    echo "Unknown candidate split: ${split}" >&2
    exit 2
  fi
  local task_root="${JOB_ROOT}/task_${run_index}_e6b_seed_${seed}/${split}"
  local run_dir="${task_root}/run"
  local phi="${task_root}/derived_phi.h5"
  local result="${task_root}/result.json"
  mkdir -p "${run_dir}"
  cp "${BOUT_INPUT}" "${run_dir}/BOUT.inp"
  echo "Solving exact phi: run=${run_index}, seed=${seed}, split=${split}"
  srun --ntasks=4 \
    "${EXECUTABLE}" \
    -d "${run_dir}" \
    "paper0:input_file=${input}" \
    "paper0:truth_layout=false"
  mapfile -t outputs < <(
    find "${run_dir}" -maxdepth 1 -name 'BOUT.dmp.*.nc' -type f | sort -V
  )
  if [[ "${#outputs[@]}" -ne 4 ]]; then
    echo "Expected four BOUT outputs for run ${run_index} ${split}" >&2
    exit 2
  fi
  "${PYTHON}" \
    "${EXTRACTOR}" \
    --input "${input}" \
    --bout-output "${outputs[@]}" \
    --output-phi "${phi}" \
    --output-result "${result}" \
    --paper0-commit "${PAPER0_COMMIT}" \
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}:${split}"
  "${PYTHON}" -m json.tool "${result}" >/dev/null
  check_sha256 "${expected_input_hash}" "${input}"
  sha256sum "${run_dir}/BOUT.inp" "${outputs[@]}" "${phi}" "${result}" \
    > "${task_root}/artifact_sha256.txt"
}

for ((position = 0; position < 3; position++)); do
  run_candidate "${position}" training
  run_candidate "${position}" validation
done

export PAPER0_CANDIDATE_PHI_JOB_ROOT="${JOB_ROOT}"
export PAPER0_CANDIDATE_PHI_COMMIT="${PAPER0_COMMIT}"
export PAPER0_CANDIDATE_PHI_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_CANDIDATE_PHI_JOB_ROOT"])
manifest_path = Path(os.environ["PAPER0_CANDIDATE_PHI_MANIFEST"])
manifest = json.loads(manifest_path.read_text())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

runs = []
for source in manifest["runs"]:
    run_index = int(source["run_index"])
    seed = int(source["seed"])
    task = root / f"task_{run_index}_e6b_seed_{seed}"
    split_results = {}
    for split, interval, candidate_key in (
        ("training", [0, 432], "training_candidate"),
        ("validation", [496, 624], "validation_candidate"),
    ):
        result_path = task / split / "result.json"
        record = json.loads(result_path.read_text())
        if (
            record["scope"] != "phase2_matched_e6b_elliptic_output"
            or record["status"] != "completed"
            or record["development_run"] != "85604"
            or record["held_out_85606_read"] is not False
            or record["truth_layout"] is not False
            or record["truth_replay_gate"] is not None
            or record["frame_interval"] != interval
            or record["source_input"]["sha256"] != source[candidate_key]["sha256"]
        ):
            raise RuntimeError(f"exact-phi result identity differs for run {run_index} {split}")
        phi_path = Path(record["derived_phi"]["path"])
        if sha256(phi_path) != record["derived_phi"]["sha256"]:
            raise RuntimeError(f"exact-phi artifact hash differs for run {run_index} {split}")
        split_results[split] = {
            "candidate": source[candidate_key],
            "elliptic_result": {"path": str(result_path), "sha256": sha256(result_path)},
            "derived_phi": record["derived_phi"],
        }
    runs.append({
        "run_index": run_index,
        "family": "e6b",
        "codec": "dcae_l10",
        "seed": seed,
        "reconstruction_result": source["reconstruction_result"],
        "splits": split_results,
    })

payload = {
    "schema_version": 1,
    "scope": "phase2_matched_e6b_candidate_phi_r2_matrix",
    "status": "completed_pending_authoritative_transport",
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_CANDIDATE_PHI_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_CANDIDATE_PHI_JOB_ID"],
    "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
    "run_count": len(runs),
    "candidate_phi_result_count": sum(len(run["splits"]) for run in runs),
    "runs": runs,
    "truth_phi_read": False,
    "interior_phi_seed": "zero",
    "complete_O1_decision_allowed": False,
    "O2_launch_allowed": False,
}
temporary = root / ".candidate_phi_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "candidate_phi_summary.json")
PY

sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/modules.txt" \
  "${JOB_ROOT}/preflight.json" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${EXECUTABLE}" \
  "${FINAL_RESULT}" \
  "${JOB_ROOT}"/task_*/training/artifact_sha256.txt \
  "${JOB_ROOT}"/task_*/validation/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Frozen R2 E6B exact candidate-phi matrix completed"
echo "Summary: ${FINAL_RESULT}"
echo "Complete O1 transport finalization is still required before any O2 launch"

