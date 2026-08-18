"""Safety locks for complete frozen R2 O1 transport finalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_matched_o1_finalize_r2.sbatch"
MANIFEST = ROOT / "paper0/manifests/phase2_matched_o1_r2_transport_finalization.json"
RECONSTRUCTION = (
    ROOT / "paper0/results/phase2_matched_o1_reconstruction_r2_6894838.json"
)
CANDIDATE_PHI = (
    ROOT / "paper0/results/phase2_matched_e6b_candidate_phi_r2_6894852.json"
)
LOCKED = (
    ROOT / "paper0/tools/finalize_matched_o1_codec.py",
    ROOT / "src/tcv_diagnostics/codec_transport.py",
    ROOT / "src/tcv_diagnostics/data_protocol.py",
    ROOT / "src/tcv_diagnostics/geometry.py",
    ROOT / "src/tcv_diagnostics/resampling.py",
    ROOT / "src/tcv_diagnostics/transport.py",
    ROOT / "src/tcv_diagnostics/matched_codec_evaluation.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/matched_codec_metrics.py",
    ROOT / "src/tcv_diagnostics/model_training_data.py",
    ROOT / "src/tcv_diagnostics/models/__init__.py",
    ROOT / "src/tcv_diagnostics/models/layers.py",
    ROOT / "src/tcv_diagnostics/models/dcae.py",
    ROOT / "src/tcv_diagnostics/matched_o1_evaluation.py",
    ROOT / "src/tcv_diagnostics/matched_o1_transport.py",
    ROOT / "src/tcv_diagnostics/model_data.py",
    ROOT / "src/tcv_diagnostics/metrics.py",
    MANIFEST,
    RECONSTRUCTION,
    CANDIDATE_PHI,
    ROOT / "paper0/results/phase2_matched_e6b_truth_replay_6894325.json",
    ROOT / "paper0/results/phase2_potential_vorticity_all_frame_6893033.json",
    ROOT / "paper0/manifests/phase2_85604_geometry_units.json",
    ROOT / "paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md",
)


def test_finalize_launcher_is_clean_cpu_only_rocky9_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=gen",
        "#SBATCH --qos=gen",
        "#SBATCH --cpus-per-task=24",
        "#SBATCH --no-requeue",
        '"${VERSION_ID%%.*}" != "9"',
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "families=(c5p e6b c5p e6b c5p e6b)",
        "seeds=(1701 1701 1702 1702 1703 1703)",
        "for run_index in 0 1 2 3 4 5",
        '"R2_accepted": r2_accepted',
        '"representation_failure": not r2_accepted',
        '"stop_before_O2": not r2_accepted',
        '"O2_launch_allowed": r2_accepted',
    ):
        assert required in text
    assert "#SBATCH --gres=gpu" not in text
    assert "nvidia-smi" not in text
    assert "WANDB_MODE" not in text


def test_finalize_launcher_hash_locks_every_local_dependency() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text


def test_transport_manifest_matches_both_frozen_matrices() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reconstruction = json.loads(RECONSTRUCTION.read_text(encoding="utf-8"))
    candidate_phi = json.loads(CANDIDATE_PHI.read_text(encoding="utf-8"))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["evaluation"]["guard_frames_read"] is False
    assert manifest["codec"] == "dcae_l10"
    assert manifest["evaluation"]["zperiod"] == 5
    assert manifest["evaluation"]["mode_mapping"] == "n=5k"
    assert [
        (run["run_index"], run["family"], run["seed"])
        for run in manifest["runs"]
    ] == [
        (0, "c5p", 1701),
        (1, "e6b", 1701),
        (2, "c5p", 1702),
        (3, "e6b", 1702),
        (4, "c5p", 1703),
        (5, "e6b", 1703),
    ]
    for frozen, source in zip(manifest["runs"], reconstruction["runs"]):
        assert frozen["reconstruction_result"] == source["result"]
    phi_by_run = {run["run_index"]: run for run in candidate_phi["runs"]}
    for run in manifest["runs"]:
        if run["family"] == "e6b":
            source = phi_by_run[run["run_index"]]
            assert run["training_phi_result"] == source["splits"]["training"][
                "elliptic_result"
            ]
            assert run["validation_phi_result"] == source["splits"][
                "validation"
            ]["elliptic_result"]
        else:
            assert "training_phi_result" not in run
            assert "validation_phi_result" not in run
    decision = manifest["decision_rule"]
    assert decision["seed_averaging_can_rescue_failure"] is False
    assert decision["o2_launch_allowed_before_all_six_complete"] is False
    assert decision["if_any_checkpoint_fails"] == "stop_before_O2_and_report_representation_failure"


def test_finalize_launcher_runs_all_results_before_o2_decision() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    launch = text.index("for run_index in 0 1 2 3 4 5; do")
    wait = text.index("if ! wait", launch)
    summarize = text.index('"O2_launch_allowed"', wait)
    assert launch < wait < summarize
    assert "Scientific gate failures return results" in text
    assert "scientific result" in text


def test_finalize_launcher_uses_all_frozen_r2_result_hashes() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for run in manifest["runs"]:
        assert run["reconstruction_result"]["sha256"] in text
        if run["family"] == "e6b":
            assert run["training_phi_result"]["sha256"] in text
            assert run["validation_phi_result"]["sha256"] in text

