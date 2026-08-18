"""Static safety locks for the authorized six-run DCAE-L10 R2 job."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o1_train_codecs_r2.sbatch"
MANIFEST = ROOT / "paper0/manifests/phase2_o1_codec_r2_runs.json"
R1_DECISION = ROOT / "paper0/results/phase2_matched_o1_finalize_r1_6894445.json"
LOCKED = (
    MANIFEST,
    R1_DECISION,
    ROOT / "paper0/manifests/phase2_matched_o1_o2_85604.json",
    ROOT / "paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md",
    ROOT / "paper0/results/phase2_model_dataset_6893525.json",
    ROOT / "paper0/results/phase2_model_dataset_normalization_6893525.json",
    ROOT / "src/tcv_diagnostics/models/layers.py",
    ROOT / "src/tcv_diagnostics/models/dcae.py",
    ROOT / "src/tcv_diagnostics/model_training_data.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/wandb_tracking.py",
    ROOT / "paper0/tools/train_codec.py",
)


def test_r2_launcher_is_valid_four_h100_rocky9_bash() -> None:
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


def test_r2_launcher_runs_exactly_the_frozen_l10_matrix() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "families=(c5p e6b c5p e6b c5p e6b)" in text
    assert "seeds=(1701 1701 1702 1702 1703 1703)" in text
    assert text.count("--codec dcae_l10") == 2
    assert "--codec dcae_l20" not in text
    assert "wait_wave 0 1 2 3" in text
    assert "wait_wave 4 5" in text
    assert "--mode full" in text
    assert '"completed_epochs"] != 200' in text
    assert '"completed_optimizer_steps"] != 5400' in text
    assert '"physics_derived_loss_used"] is not False' in text


def test_r2_launcher_requires_finished_online_wandb() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "WANDB_MODE=online",
        "WANDB_REQUIRE_SERVICE=true",
        'WANDB_GROUP="o1-dcae-l10-r2"',
        'p0o1r2-${SLURM_JOB_ID}-${run_index}',
        'o1-r2-${family}-s${seed}-j${SLURM_JOB_ID}-t${run_index}',
        'tracking["remote_state_after_finish"] != "finished"',
        'tracking["epochs_logged"] != 200',
    ):
        assert required in text


def test_r2_launcher_hash_locks_every_local_dependency() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text


def test_r2_manifest_and_authorization_match_complete_r1_decision() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decision = json.loads(R1_DECISION.read_text(encoding="utf-8"))
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["model"]["stage"] == "R2"
    assert manifest["model"]["codec"] == "dcae_l10"
    assert manifest["model"]["from_scratch"] is True
    assert manifest["model"]["physics_derived_loss_allowed"] is False
    assert manifest["training"]["schedule_identical_to_r1"] is True
    assert manifest["training"]["total_optimizer_steps"] == 5400
    assert [
        (task["run_index"], task["family"], task["seed"])
        for task in manifest["tasks"]
    ] == [
        (0, "c5p", 1701),
        (1, "e6b", 1701),
        (2, "c5p", 1702),
        (3, "e6b", 1702),
        (4, "c5p", 1703),
        (5, "e6b", 1703),
    ]
    authorization = manifest["r2_authorization"]
    assert authorization["sha256"] == hashlib.sha256(
        R1_DECISION.read_bytes()
    ).hexdigest()
    assert decision["all_six_complete"] is True
    assert decision["R1_accepted"] is False
    assert decision["R2_required"] is True
    assert decision["R2_launch_allowed"] is True
    assert decision["O2_launch_allowed"] is False
    assert len(decision["runs"]) == 6


def test_r2_launcher_stages_only_verified_development_data() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "MODEL_DATA_STAGED" in text
    assert "Refusing to reuse existing node-local directory" in text
    assert "verified_shards=$((verified_shards + 1))" in text
    assert '"${verified_shards}" -ne 8' in text
    assert "85606/data" not in text
    assert '"held_out_85606_read": False' in text
    assert '"O2_launch_allowed": False' in text
