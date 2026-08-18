"""Safety and provenance locks for frozen R1 E6B exact-phi recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_matched_e6b_candidate_phi_r1.sbatch"
MANIFEST = ROOT / "paper0/manifests/phase2_matched_e6b_candidate_phi_r1.json"
RECONSTRUCTION = (
    ROOT / "paper0/results/phase2_matched_o1_reconstruction_r1_6894345.json"
)
LOCKED = (
    ROOT / "paper0/oracles/matched_e6b_elliptic/CMakeLists.txt",
    ROOT / "paper0/oracles/matched_e6b_elliptic/matched_e6b_elliptic_oracle.cxx",
    ROOT / "paper0/tools/extract_matched_e6b_phi.py",
    ROOT / "paper0/oracles/potential_elliptic/BOUT.inp",
    MANIFEST,
    RECONSTRUCTION,
    ROOT / "paper0/results/phase2_matched_e6b_truth_replay_6894325.json",
    ROOT / "paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md",
    ROOT / "paper0/manifests/phase2_o1_codec_r1_runs.json",
    ROOT / "paper0/manifests/phase2_85604_geometry_units.json",
)


def test_candidate_phi_launcher_is_clean_cpu_only_rocky9_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=gen",
        "#SBATCH --qos=gen",
        "#SBATCH --ntasks=4",
        "#SBATCH --no-requeue",
        '"${VERSION_ID%%.*}" != "9"',
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "paper0:truth_layout=false",
        "run_candidate \"${position}\" training",
        "run_candidate \"${position}\" validation",
        '"complete_O1_decision_allowed": False',
        '"R2_launch_allowed": False',
    ):
        assert required in text
    assert "#SBATCH --gres=gpu" not in text
    assert "paper0:truth_layout=true" not in text
    assert "--truth-layout" not in text


def test_candidate_phi_launcher_hash_locks_every_local_dependency() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text


def test_candidate_phi_manifest_matches_frozen_reconstruction() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reconstruction = json.loads(RECONSTRUCTION.read_text(encoding="utf-8"))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["operator"]["truth_phi_read"] is False
    assert manifest["operator"]["interior_phi_seed"] == "zero"
    assert manifest["identity"]["zperiod"] == 5
    assert manifest["identity"]["mode_mapping"] == "n=5k"
    expected = [
        run for run in reconstruction["runs"] if run["family"] == "e6b"
    ]
    assert [
        (run["run_index"], run["seed"])
        for run in manifest["runs"]
    ] == [(1, 1701), (3, 1702), (5, 1703)]
    for frozen, source in zip(manifest["runs"], expected):
        assert frozen["reconstruction_result"] == source["result"]
        assert frozen["training_candidate"] == {
            **source["training_candidate"],
            "frames": [0, 432],
        }
        assert frozen["validation_candidate"] == {
            **source["validation_candidate"],
            "frames": [496, 624],
        }
    output = manifest["output_contract"]
    assert output["complete_o1_decision_allowed"] is False
    assert output["r2_launch_allowed_from_this_stage_alone"] is False
    assert output["authoritative_transport_finalization_required_next"] is True
