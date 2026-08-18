#!/bin/bash
# Complete all six frozen R2 O1 evaluations with exact common-view phi and
# authoritative native-81 transport. Scientific gate failures return results;
# only missing/corrupt inputs or execution errors fail this Slurm job.
#SBATCH --job-name=p2_o1_final_r2
#SBATCH --partition=gen
#SBATCH --qos=gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/phase2_matched_o1_finalize_r2_%j.log

set -euo pipefail

: "${PAPER0_EXPECTED_COMMIT:?submit with PAPER0_EXPECTED_COMMIT set}"

readonly PAPER0_ROOT="${PAPER0_ROOT:-/mnt/home/sdelaurentiis/tcv-diagnostics-eval-8a6e20d}"
readonly PYTHON="/mnt/home/sdelaurentiis/tcv-gaot-3d/.venv-lola/bin/python"
readonly ENTRYPOINT="${PAPER0_ROOT}/paper0/tools/finalize_matched_o1_codec.py"
readonly MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_matched_o1_r2_transport_finalization.json"
readonly RECONSTRUCTION_SUMMARY="${PAPER0_ROOT}/paper0/results/phase2_matched_o1_reconstruction_r2_6894838.json"
readonly CANDIDATE_PHI_SUMMARY="${PAPER0_ROOT}/paper0/results/phase2_matched_e6b_candidate_phi_r2_6894852.json"
readonly TRUTH_REPLAY="${PAPER0_ROOT}/paper0/results/phase2_matched_e6b_truth_replay_6894325.json"
readonly NATIVE_TRUTH_RESULT="${PAPER0_ROOT}/paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
readonly GEOMETRY_MANIFEST="${PAPER0_ROOT}/paper0/manifests/phase2_85604_geometry_units.json"
readonly PROTOCOL="${PAPER0_ROOT}/paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md"
readonly GEOMETRY="/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc"
readonly MODEL_DATA_ROOT="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
readonly RESULT_BASE="/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_finalize_r2"
readonly JOB_ROOT="${RESULT_BASE}/job_${SLURM_JOB_ID}"
readonly FINAL_RESULT="${JOB_ROOT}/complete_o1_summary.json"

readonly -a families=(c5p e6b c5p e6b c5p e6b)
readonly -a seeds=(1701 1701 1702 1702 1703 1703)
readonly -a reconstruction_results=(
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_0_c5p_seed_1701/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_1_e6b_seed_1701/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_2_c5p_seed_1702/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_3_e6b_seed_1702/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_4_c5p_seed_1703/result.json"
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_o1_reconstruction_r2/job_6894838/task_5_e6b_seed_1703/result.json"
)
readonly -a reconstruction_result_hashes=(
  c2f7de21ea9fee1efc1a0b1a823b1066a629ee121a52d290a2eef344cc0dd47f
  9d0caf7dbede9c9db588d84602f0b8024bfd2cffe0e60b23bca3163a5fdc77d8
  ed348f895c97351b5c139e8ef90e0eea1b5eaeae174ec2987c8b2693d09ef688
  8c016ae3ee64ca7f715d8a4aa90dfdcc6cfd99729972dc603277daec9f9a5924
  174cae80360cb2dfb4cdc3040f2da7ba69ee9e54dde12f4967f3d801c97bf4f3
  84748bacc58d96354c9dfedd87707f657d75d2214814190fe92a6028440538c5
)
readonly -a training_phi_results=(
  ""
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2/job_6894852/task_1_e6b_seed_1701/training/result.json"
  ""
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2/job_6894852/task_3_e6b_seed_1702/training/result.json"
  ""
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2/job_6894852/task_5_e6b_seed_1703/training/result.json"
)
readonly -a training_phi_result_hashes=(
  ""
  5c2ece8a1b3d96f675641159ea4f0b93c0b73a83b766010f1541418ac221cd76
  ""
  bdb3bf29c29be124c2eec08a20e9b183c5b55a207b435224eab4c25b2fdc7e8f
  ""
  e19092fd908489dfd3a3cb9ab18dbf4a2b609c1a89b84bbbba6542969d0c19ac
)
readonly -a validation_phi_results=(
  ""
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2/job_6894852/task_1_e6b_seed_1701/validation/result.json"
  ""
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2/job_6894852/task_3_e6b_seed_1702/validation/result.json"
  ""
  "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_matched_e6b_candidate_phi_r2/job_6894852/task_5_e6b_seed_1703/validation/result.json"
)
readonly -a validation_phi_result_hashes=(
  ""
  36c839a8d9f4ba49f9c5189a143829892c21e845efda5670ff1ee53f12a67ee5
  ""
  72fa05f06ad44d8e3b3c348cc224242737d09ed08b3450000e62d4967b0b98cd
  ""
  e94a2dcf6c5fba6401fcbcd4670e78f07587845aac3d9bd6acd83f5457095b7b
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
  echo "This transport finalization requires Rocky 9" >&2
  exit 2
fi

check_sha256 "73c8a227b1bdb36c3c3936e5ee98ddae6b960ff695ec2d8d29ad9aab0c64acfd" "${ENTRYPOINT}"
check_sha256 "201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_transport.py"
check_sha256 "4338c734e73c1e94ece6b6269f0cbfe2b077d6346d0058243a9734d7844cd28b" "${PAPER0_ROOT}/src/tcv_diagnostics/data_protocol.py"
check_sha256 "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c" "${PAPER0_ROOT}/src/tcv_diagnostics/geometry.py"
check_sha256 "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e" "${PAPER0_ROOT}/src/tcv_diagnostics/resampling.py"
check_sha256 "09e99af7d87f7a0b23433d79bb1f6198ffb49cfd48ee2cf6e1c252beeae5c85a" "${PAPER0_ROOT}/src/tcv_diagnostics/transport.py"
check_sha256 "e060674d9afa0d8fd4d709bc927ef2c144fb40d284f7e672fc37636f820d23bf" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_codec_evaluation.py"
check_sha256 "57b40c3d1be4b065c71d1598876fe2edb69e76f25fe6ce4e87c6dea08d9a6690" "${PAPER0_ROOT}/src/tcv_diagnostics/codec_training.py"
check_sha256 "4d6847b1f7dcae3043df07d405d9dad07cc86c96b1f63c6fb8ccdbcf745a37ed" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_codec_metrics.py"
check_sha256 "af3fc3be94101338a796e19a751014bbcb033f95a42598b947a97a6f66be9af2" "${PAPER0_ROOT}/src/tcv_diagnostics/model_training_data.py"
check_sha256 "8eb062349ab78d8f163c46bb3539e7fd3791cc095ac3b8ed17ae9c000a5be512" "${PAPER0_ROOT}/src/tcv_diagnostics/models/__init__.py"
check_sha256 "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b" "${PAPER0_ROOT}/src/tcv_diagnostics/models/layers.py"
check_sha256 "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554" "${PAPER0_ROOT}/src/tcv_diagnostics/models/dcae.py"
check_sha256 "b54d6660f5c0605086d5215c516aa5c1c40e5f37249ea39544073782b45170c1" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_o1_evaluation.py"
check_sha256 "1eba2a8e0c54928264c2c84d5fda08ef6080cf4d73017d40193e3c23e2d2854a" "${PAPER0_ROOT}/src/tcv_diagnostics/matched_o1_transport.py"
check_sha256 "3312d6ac093e7b9c6ad69a3d7b97fe6aa06695f03c2f10bed7a75f6095795416" "${PAPER0_ROOT}/src/tcv_diagnostics/model_data.py"
check_sha256 "0a7540a3e1b9698dea2293e729fd1f956986924cb3be6cc15a546add98521b64" "${PAPER0_ROOT}/src/tcv_diagnostics/metrics.py"
check_sha256 "26046a0dfa53a1a858ea9fdd69e1c6c1afa240df4d21f35d962e7295ff504390" "${MANIFEST}"
check_sha256 "c091326385aa77ba245759dccb4a7dc6e55b7a905c317e4b3b7706ce1c8bf321" "${RECONSTRUCTION_SUMMARY}"
check_sha256 "f26e85ea0a305d83672f69d4302858af6fb0db8a3ec54ff2236894707d00f559" "${CANDIDATE_PHI_SUMMARY}"
check_sha256 "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27" "${TRUTH_REPLAY}"
check_sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae" "${NATIVE_TRUTH_RESULT}"
check_sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460" "${GEOMETRY_MANIFEST}"
check_sha256 "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4" "${PROTOCOL}"
check_sha256 "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8" "${GEOMETRY}"
check_sha256 "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18" "${MODEL_DATA_ROOT}/model_dataset_manifest.json"
check_sha256 "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7" "${MODEL_DATA_ROOT}/normalization.json"
check_sha256 "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01" "${MODEL_DATA_ROOT}/artifact_sha256.txt"
for ((run_index = 0; run_index < 6; run_index++)); do
  check_sha256 "${reconstruction_result_hashes[run_index]}" "${reconstruction_results[run_index]}"
  if [[ "${families[run_index]}" == "e6b" ]]; then
    check_sha256 "${training_phi_result_hashes[run_index]}" "${training_phi_results[run_index]}"
    check_sha256 "${validation_phi_result_hashes[run_index]}" "${validation_phi_results[run_index]}"
  elif [[ -n "${training_phi_results[run_index]}" || -n "${validation_phi_results[run_index]}" ]]; then
    echo "C5P run ${run_index} unexpectedly has E6B phi inputs" >&2
    exit 2
  fi
done

module purge
module list 2> "${JOB_ROOT}/modules.txt"
scontrol show job -dd "${SLURM_JOB_ID}" > "${JOB_ROOT}/slurm_job.txt"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PAPER0_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

{
  echo "captured_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname --fqdn)"
  echo "paper0_commit=${PAPER0_COMMIT}"
  "${PYTHON}" --version
  "${PYTHON}" -c 'import h5py,numpy,scipy,torch; print("h5py="+h5py.__version__); print("numpy="+numpy.__version__); print("scipy="+scipy.__version__); print("torch="+torch.__version__); print("cuda_available="+str(torch.cuda.is_available()))'
} > "${JOB_ROOT}/environment.txt" 2>&1

export PAPER0_FINALIZE_MANIFEST="${MANIFEST}"
export PAPER0_RECONSTRUCTION_SUMMARY="${RECONSTRUCTION_SUMMARY}"
export PAPER0_CANDIDATE_PHI_SUMMARY="${CANDIDATE_PHI_SUMMARY}"
export PAPER0_TRUTH_REPLAY="${TRUTH_REPLAY}"
"${PYTHON}" - <<'PY' > "${JOB_ROOT}/preflight.json"
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["PAPER0_FINALIZE_MANIFEST"]).read_text())
reconstruction = json.loads(Path(os.environ["PAPER0_RECONSTRUCTION_SUMMARY"]).read_text())
candidate_phi = json.loads(Path(os.environ["PAPER0_CANDIDATE_PHI_SUMMARY"]).read_text())
truth = json.loads(Path(os.environ["PAPER0_TRUTH_REPLAY"]).read_text())
expected = [(0, "c5p", 1701), (1, "e6b", 1701), (2, "c5p", 1702),
            (3, "e6b", 1702), (4, "c5p", 1703), (5, "e6b", 1703)]
observed = [(int(run["run_index"]), run["family"], int(run["seed"]))
            for run in manifest["runs"]]
if observed != expected:
    raise RuntimeError("transport-finalization run order differs")
if manifest["development_run"] != "85604" or manifest["held_out_85606_access_allowed"] is not False:
    raise RuntimeError("transport-finalization development-data contract differs")
if manifest["evaluation"]["guard_frames_read"] is not False:
    raise RuntimeError("transport-finalization manifest permits guard reads")
if manifest["codec"] != "dcae_l10":
    raise RuntimeError("transport-finalization codec differs")
if reconstruction["status"] != "completed_pending_exact_e6b_phi_and_authoritative_transport":
    raise RuntimeError("reconstruction matrix status differs")
if reconstruction["R2_accepted"] is not False or reconstruction["complete_O1_decision_allowed"] is not False:
    raise RuntimeError("reconstruction matrix prematurely decides R2")
if candidate_phi["status"] != "completed_pending_authoritative_transport":
    raise RuntimeError("candidate-phi matrix status differs")
if candidate_phi["truth_phi_read"] is not False or candidate_phi["candidate_phi_result_count"] != 6:
    raise RuntimeError("candidate-phi matrix contract differs")
if candidate_phi["O2_launch_allowed"] is not False:
    raise RuntimeError("candidate-phi matrix prematurely permits O2")
if truth["status"] != "pass" or truth["all_frames_passed"] is not True:
    raise RuntimeError("exact-phi truth replay did not pass")
if manifest["decision_rule"]["o2_launch_allowed_before_all_six_complete"] is not False:
    raise RuntimeError("transport manifest prematurely permits O2")
if manifest["decision_rule"]["if_any_checkpoint_fails"] != "stop_before_O2_and_report_representation_failure":
    raise RuntimeError("transport manifest failure rule differs")
print(json.dumps({
    "development_run": "85604",
    "held_out_85606_read": False,
    "run_order": expected,
    "exact_candidate_phi_results": 6,
    "authoritative_transport": True,
    "o2_launch_allowed_before_completion": False,
}, sort_keys=True))
PY

cd "${PAPER0_ROOT}"
"${PYTHON}" -m pytest -p no:cacheprovider -q | tee "${JOB_ROOT}/test_output.txt"

declare -a pids

task_root() {
  local run_index="$1"
  echo "${JOB_ROOT}/task_${run_index}_${families[run_index]}_seed_${seeds[run_index]}"
}

run_task() {
  local run_index="$1"
  local output
  output="$(task_root "${run_index}")"
  mkdir -p "${output}"
  local -a command=(
    "${PYTHON}" -u "${ENTRYPOINT}"
    --reconstruction-result "${reconstruction_results[run_index]}"
    --reconstruction-result-sha256 "${reconstruction_result_hashes[run_index]}"
    --native-truth-result "${NATIVE_TRUTH_RESULT}"
    --native-truth-result-sha256 "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
    --geometry-manifest "${GEOMETRY_MANIFEST}"
    --geometry-manifest-sha256 "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
    --geometry "${GEOMETRY}"
    --artifact-root "${MODEL_DATA_ROOT}"
    --output "${output}/result.json"
    --paper0-commit "${PAPER0_COMMIT}"
    --slurm-job-id "${SLURM_JOB_ID}:run${run_index}"
    --chunk-frames 4
  )
  if [[ "${families[run_index]}" == "e6b" ]]; then
    command+=(
      --train-phi-result "${training_phi_results[run_index]}"
      --train-phi-result-sha256 "${training_phi_result_hashes[run_index]}"
      --validation-phi-result "${validation_phi_results[run_index]}"
      --validation-phi-result-sha256 "${validation_phi_result_hashes[run_index]}"
      --truth-replay-summary "${TRUTH_REPLAY}"
      --truth-replay-summary-sha256 "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27"
    )
  fi
  printf '%q ' "${command[@]}" > "${output}/command.sh"
  printf '\n' >> "${output}/command.sh"
  echo "Starting complete O1 finalization ${run_index}: family=${families[run_index]}, seed=${seeds[run_index]}"
  "${command[@]}"
  "${PYTHON}" -m json.tool "${output}/result.json" >/dev/null
  sha256sum "${output}/command.sh" "${output}/result.json" > "${output}/artifact_sha256.txt"
  echo "Completed O1 finalization ${run_index}: family=${families[run_index]}, seed=${seeds[run_index]}"
}

for run_index in 0 1 2 3 4 5; do
  run_task "${run_index}" > >(tee -a "$(task_root "${run_index}").log") 2>&1 &
  pids[run_index]=$!
done

failed=0
for run_index in 0 1 2 3 4 5; do
  if ! wait "${pids[run_index]}"; then
    echo "O1 finalization ${run_index} had an infrastructure failure" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one O1 finalizer failed to produce a scientific result" >&2
  exit 1
fi

export PAPER0_FINALIZE_JOB_ROOT="${JOB_ROOT}"
export PAPER0_FINALIZE_COMMIT="${PAPER0_COMMIT}"
export PAPER0_FINALIZE_JOB_ID="${SLURM_JOB_ID}"
"${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PAPER0_FINALIZE_JOB_ROOT"])
manifest_path = Path(os.environ["PAPER0_FINALIZE_MANIFEST"])
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
    family = source["family"]
    seed = int(source["seed"])
    result_path = root / f"task_{run_index}_{family}_seed_{seed}" / "result.json"
    result = json.loads(result_path.read_text())
    if (
        result["scope"] != "phase2_matched_o1_codec_complete"
        or result["development_run"] != "85604"
        or result["held_out_85606_read"] is not False
        or result["identity"]["family"] != family
        or int(result["identity"]["seed"]) != seed
        or result["identity"]["codec"] != "dcae_l10"
    ):
        raise RuntimeError(f"complete O1 result identity differs for run {run_index}")
    passes = bool(result["complete_o1_gate"]["passes"])
    if result["status"] != ("pass" if passes else "fail"):
        raise RuntimeError(f"complete O1 status differs for run {run_index}")
    runs.append({
        "run_index": run_index,
        "family": family,
        "codec": result["identity"]["codec"],
        "seed": seed,
        "status": result["status"],
        "passes": passes,
        "conditions": result["complete_o1_gate"]["conditions"],
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "validation_gate_status": {
            "native_predicted_view": result["source_reconstruction_result"]["native_view_pretransport_gate"]["status"],
            "common_transport_view": result["common_view"]["gate"]["status"],
            "authoritative_native81_transport": "pass" if result["transport"]["gate"]["passes"] else "fail",
        },
    })

r2_accepted = len(runs) == 6 and all(run["passes"] for run in runs)
payload = {
    "schema_version": 1,
    "scope": "phase2_matched_o1_r2_complete_matrix",
    "status": "pass" if r2_accepted else "fail",
    "development_run": "85604",
    "held_out_85606_read": False,
    "paper0_commit": os.environ["PAPER0_FINALIZE_COMMIT"],
    "slurm_job_id": os.environ["PAPER0_FINALIZE_JOB_ID"],
    "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
    "run_count": len(runs),
    "runs": runs,
    "R2_accepted": r2_accepted,
    "all_six_complete": len(runs) == 6,
    "seed_averaging_used": False,
    "representation_failure": not r2_accepted,
    "stop_before_O2": not r2_accepted,
    "O2_launch_allowed": r2_accepted,
}
temporary = root / ".complete_o1_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(root / "complete_o1_summary.json")
PY

sha256sum \
  "${JOB_ROOT}/environment.txt" \
  "${JOB_ROOT}/modules.txt" \
  "${JOB_ROOT}/preflight.json" \
  "${JOB_ROOT}/slurm_job.txt" \
  "${JOB_ROOT}/test_output.txt" \
  "${FINAL_RESULT}" \
  "${JOB_ROOT}"/task_*/artifact_sha256.txt \
  > "${JOB_ROOT}/artifact_sha256.txt"

echo "Complete frozen R2 O1 matrix finalized"
echo "Summary: ${FINAL_RESULT}"
