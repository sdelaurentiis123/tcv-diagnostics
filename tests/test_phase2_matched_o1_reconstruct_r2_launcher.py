"""Static safety locks for the six-checkpoint matched O1 R2 reconstruction job."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_matched_o1_reconstruct_r2.sbatch"
MANIFEST = ROOT / "paper0/manifests/phase2_matched_o1_r2_reconstruction.json"
FREEZE = ROOT / "paper0/results/phase2_o1_codec_r2_freeze_6894703.json"
LOCKED = (
    FREEZE,
    ROOT / "paper0/results/phase2_matched_e6b_truth_replay_6894325.json",
    MANIFEST,
    ROOT / "paper0/manifests/phase2_matched_o1_o2_85604.json",
    ROOT / "paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md",
    ROOT / "paper0/results/phase2_model_dataset_6893525.json",
    ROOT / "paper0/results/phase2_model_dataset_normalization_6893525.json",
    ROOT / "src/tcv_diagnostics/models/layers.py",
    ROOT / "src/tcv_diagnostics/models/dcae.py",
    ROOT / "src/tcv_diagnostics/model_data.py",
    ROOT / "src/tcv_diagnostics/model_training_data.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/matched_codec_evaluation.py",
    ROOT / "src/tcv_diagnostics/matched_codec_metrics.py",
    ROOT / "paper0/tools/evaluate_matched_codec_reconstruction.py",
)


def test_reconstruction_launcher_is_valid_four_h100_rocky9_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=gpuxl",
        "#SBATCH --qos=gen",
        "#SBATCH --gres=gpu:h100:4",
        "#SBATCH --constraint=h100",
        "#SBATCH --no-requeue",
        '"${VERSION_ID%%.*}" != "9"',
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
    ):
        assert required in text
    assert "#SBATCH --partition=gpupreempt" not in text


def test_reconstruction_launcher_runs_exactly_six_frozen_candidates() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "families=(c5p e6b c5p e6b c5p e6b)" in text
    assert "seeds=(1701 1701 1702 1702 1703 1703)" in text
    assert "wait_wave 0 1 2 3" in text
    assert "wait_wave 4 5" in text
    assert text.count("--checkpoint-sha256") == 2
    assert text.count("--training-result-sha256") == 2
    assert "--chunk-frames 4" in text
    assert "--device cuda:0" in text
    assert text.count("--codec dcae_l10") == 2
    assert '"R2_accepted": False' in text
    assert '"complete_O1_decision_allowed": False' in text


def test_reconstruction_launcher_stages_only_verified_development_data() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "MODEL_DATA_STAGED" in text
    assert "Refusing to reuse existing node-local directory" in text
    assert "verified_shards=$((verified_shards + 1))" in text
    assert '"${verified_shards}" -ne 8' in text
    assert "85606/data" not in text
    assert '"held_out_85606_read": False' in text


def test_reconstruction_launcher_hash_locks_every_local_dependency() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text


def test_reconstruction_launcher_uses_every_frozen_r2_artifact() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["stage"] == "R2"
    assert freeze["codec"] == "dcae_l10"
    for run in freeze["runs"]:
        assert run["selected_checkpoint"]["sha256"] in text
        assert run["training_result"]["sha256"] in text


def test_reconstruction_manifest_freezes_run_order_and_no_decision() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["codec"] == "dcae_l10"
    assert manifest["training_matrix"]["training_job_id"] == "6894463"
    assert manifest["training_matrix"]["freeze_job_id"] == "6894703"
    assert manifest["training_matrix"][
        "checkpoint_choice_frozen_before_physics_metrics"
    ] is True
    assert [
        (run["run_index"], run["family"], run["seed"])
        for run in manifest["run_order"]
    ] == [
        (0, "c5p", 1701),
        (1, "e6b", 1701),
        (2, "c5p", 1702),
        (3, "e6b", 1702),
        (4, "c5p", 1703),
        (5, "e6b", 1703),
    ]
    assert manifest["output_contract"]["complete_o1_decision_allowed"] is False
    assert manifest["output_contract"][
        "authoritative_native81_transport_required_after_reconstruction"
    ] is True
