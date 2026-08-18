#!/bin/bash
# Recover exact phi for the three frozen R1 E6B codec candidates.
# CPU-only and sequential: every solve consumes the full four-rank allocation.
# This stage cannot decide O1 or authorize R2; transport finalization follows.
#SBATCH --job-name=p2_e6b_phi_r1
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_e6b_candidate_phi_r1_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-data/bin/python"
readonly BOUT_INSTALL="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890766/install"
readonly BOUT_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/BOUT-dev-7d28d67"
readonly HERMES_SOURCE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hermes-3-920ba829"
readonly SOURCE="${PAPER0_ROOT}/paper0/oracles/matched_e6b_elliptic"
readonly BOUT_INPUT="${PAPER0_ROOT}/paper0/oracles/potential_elliptic/BOUT.inp"
readonly EXTRACTOR="${PAPER0_ROOT}/paper0/tools/extract_matched_e6b_phi.py"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_e6b_candidate_phi_r1.json"
readonly RECONSTRUCTION_SUMMARY="${PAPER0_ROOT}/paper0/results/phase2_matched_o1_reconstruction_r1_6894345.json"
readonly TRUTH_REPLAY="${PAPER0_ROOT}/paper0/results/phase2_matched_e6b_truth_replay_6894325.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly R1_RUN_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_o1_codec_r1_runs.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r1"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly BUILD_DIR="${JOB_ROOT}/build"
readonly FINAL_RESULT="${JOB_ROOT}/candidate_phi_summary.json"
readonly EXPECTED_BOUT_COMMIT="7d28d67c3f12c24ec281c0982e870f5369c65a6f"
readonly EXPECTED_HERMES_COMMIT="920ba829cc78cdab0dbf6101c69fecc4689bd8dd"

readonly -a run_indices=(1 3 5)
readonly -a seeds=(1701 1702 1703)
readonly -a reconstruction_results=(
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_1_e6b_seed_1701/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_3_e6b_seed_1702/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_5_e6b_seed_1703/result.json"
)
readonly -a reconstruction_result_hashes=(
  5fde882773aab9378131231b9adfb9556e75bc67d6446a8b93e602d8b4a99f39
  9f108d4560d5b5dfe3fe8ee46900eefcd6710acae4ba0662540388b1ee1eca2e
  620d840a70c4f0ce2ed052221276758edb1d38d33294fcaf1598307f5e713133
)
readonly -a training_candidates=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_1_e6b_seed_1701/train_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_3_e6b_seed_1702/train_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_5_e6b_seed_1703/train_candidate_native81.h5"
)
readonly -a training_candidate_hashes=(
  a03bddb359dfff720b5ed226d7612892d905d77b64be1a8f59d4795282d070e2
  88cb4cbd0bdd3885bb66a76d3ea7d52b92453670fec513df7c872fce2a6810db
  8aa9833e8510fc5c47590ab546b409031f28d4de21de6e1c64231230e4e89e4c
)
readonly -a validation_candidates=(
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_1_e6b_seed_1701/validation_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_3_e6b_seed_1702/validation_candidate_native81.h5"
  "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r1/job_6894345/task_5_e6b_seed_1703/validation_candidate_native81.h5"
)
readonly -a validation_candidate_hashes=(
  660a233085ec07a7188e3ad2c255e06f452f28611fa037950982854ed99bb421
  153eeed402e2043e54841dc2ee1eef048740867ca7f7c886253ed51a60cf58c9
  b2fc25b790ba7e9c4cdeb1c3ce2f52e19820e8cc5f4d8c83f2048174000c80d9
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
check_sha256 "ca814409445247afa097d03a8c322b6f6e415a722f9eda92178d249fb4f37a7d" "${MANIFEST}"
check_sha256 "47fb0904d462d6802f44296b9105f108e59683e9f82a42c9c8f14f74a5c6a7eb" "${RECONSTRUCTION_SUMMARY}"
check_sha256 "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27" "${TRUTH_REPLAY}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "22bda830182e9869769e8212fca5e0e50ef6c4314910e6e251249b317bd8c721" "${R1_RUN_MANIFEST}"
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
    raise RuntimeError("R1 reconstruction summary status differs")
if reconstruction["complete_O1_decision_allowed"] is not False:
    raise RuntimeError("R1 reconstruction prematurely permits a complete decision")
selected = [
    (int(run["run_index"]), int(run["seed"]))
    for run in reconstruction["runs"]
    if run["family"] == "e6b"
]
if selected != expected:
    raise RuntimeError("R1 reconstruction E6B run order differs")
if truth["status"] != "pass" or truth["all_frames_passed"] is not True:
    raise RuntimeError("zero-seed exact-phi truth replay did not pass")
if truth["coverage"] != [0, 624] or int(truth["frame_count"]) != 624:
    raise RuntimeError("zero-seed exact-phi truth replay coverage differs")
if manifest["output_contract"]["complete_o1_decision_allowed"] is not False:
    raise RuntimeError("candidate-phi manifest prematurely permits an O1 decision")
if manifest["output_contract"]["r2_launch_allowed_from_this_stage_alone"] is not False:
    raise RuntimeError("candidate-phi manifest prematurely permits R2")
print(json.dumps({
    "development_run": "85604",
    "held_out_85606_read": False,
    "e6b_runs": expected,
    "zero_seed_truth_replay": "pass",
    "truth_phi_read": False,
    "complete_o1_decision_allowed": False,
    "r2_launch_allowed": False,
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
        "codec": "dcae_l20",
        "seed": seed,
        "reconstruction_result": source["reconstruction_result"],
        "splits": split_results,
    })

payload = {
    "schema_version": 1,
    "scope": "phase2_matched_e6b_candidate_phi_r1_matrix",
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
    "R2_launch_allowed": False,
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

echo "Frozen R1 E6B exact candidate-phi matrix completed"
echo "Summary: ${FINAL_RESULT}"
echo "Complete O1 transport finalization is still required before any R2 launch"
