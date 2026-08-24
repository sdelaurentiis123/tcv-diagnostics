"""Protocol and machinery tests for the old-85604 full Stage-1 matrix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/post_ecrd_old_85604_stage1_full.json"
PROTOCOL = (
    ROOT
    / "paper0/protocol/POST_ECRD_OLD_85604_STAGE1_FULL_PROTOCOL_2026-08-24.md"
)
BLOCK_AMENDMENT = (
    ROOT
    / "paper0/protocol/POST_ECRD_OLD_85604_STAGE1_BLOCK_EVALUATION_AMENDMENT_2026-08-24.md"
)
TRAINER = ROOT / "paper0/tools/train_codec_free_stage1_full.py"
SUMMARIZER = ROOT / "paper0/tools/summarize_codec_free_stage1_seed.py"
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_stage1_full.sbatch"


def test_full_manifest_freezes_old_data_and_three_seed_budget() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_read"] is False
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["pilot_training_authorized"] is False
    assert manifest["full_training_authorized"] is True
    assert manifest["split"]["training_pair_count"] == 431
    assert manifest["split"]["validation_pair_count"] == 127
    assert manifest["optimization"]["seeds"] == [1701, 1702, 1703]
    assert manifest["optimization"]["epochs"] == 12
    assert manifest["architecture"]["zero_initialize_output"] is True
    assert manifest["loss"]["scale_fit_split"] == "train"
    assert manifest["loss"]["physics_derived_quantities_used"] is False


def test_full_protocol_records_direct_field_scaling_and_decision_rule() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "training split only" in protocol
    assert "zero-initialized" in protocol
    assert "10%" in protocol
    assert "No flux, spectrum, cross-phase" in protocol
    assert "85606" in protocol and "unopened and prohibited" in protocol
    amendment = BLOCK_AMENDMENT.read_text(encoding="utf-8")
    assert "V00 = [498,540)" in amendment
    assert "V01 = [540,582)" in amendment
    assert "V02 = [582,624)" in amendment
    assert "before any" in amendment and "chronological block" in amendment
    assert "85606" in amendment and "unopened and prohibited" in amendment


def test_full_launcher_requests_one_gpu_at_a_time_and_both_families() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --array=0-2%1" in launcher
    assert "#SBATCH --mem=16G" in launcher
    assert "for FAMILY in c5p e6b" in launcher
    assert "WANDB_MODE=online" in launcher
    assert "tests/test_codec_free_stage1_full.py" in launcher
    assert "85606" not in launcher


def test_full_tools_are_scoped_and_importable() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    summarizer = SUMMARIZER.read_text(encoding="utf-8")
    compile(trainer, str(TRAINER), "exec")
    compile(summarizer, str(SUMMARIZER), "exec")
    assert "persistence_normalized_state_derivative_loss" in trainer
    assert "fit_training_derivative_rms" in trainer
    assert '"held_out_85606_read": False' in trainer
    assert '"physics_derived_loss_used": False' in trainer
