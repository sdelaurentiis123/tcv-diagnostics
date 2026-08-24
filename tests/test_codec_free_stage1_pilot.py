import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0" / "tools" / "train_codec_free_stage1_pilot.py"
MANIFEST = ROOT / "paper0" / "manifests" / "post_ecrd_old_85604_stage1_pilot.json"
PROTOCOL = (
    ROOT
    / "paper0"
    / "protocol"
    / "POST_ECRD_OLD_85604_STAGE1_PILOT_PROTOCOL_2026-08-24.md"
)
LAUNCHER = ROOT / "cluster" / "post_ecrd_old_85604_stage1_pilot.sbatch"


def load_namespace():
    namespace = {"__name__": "stage1_pilot_test"}
    exec(compile(TOOL.read_text(encoding="utf-8"), str(TOOL), "exec"), namespace)
    return namespace


def test_learning_rate_has_frozen_warmup_and_cosine_endpoints() -> None:
    function = load_namespace()["learning_rate"]
    assert function(1, total_updates=10, warmup_updates=2, peak=2e-4, minimum=2e-5) == pytest.approx(1e-4)
    assert function(2, total_updates=10, warmup_updates=2, peak=2e-4, minimum=2e-5) == pytest.approx(2e-4)
    assert function(10, total_updates=10, warmup_updates=2, peak=2e-4, minimum=2e-5) == pytest.approx(2e-5)
    with pytest.raises(ValueError, match="schedule"):
        function(0, total_updates=10, warmup_updates=2, peak=2e-4, minimum=2e-5)


def test_manifest_freezes_old_data_pilot_only() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_read"] is False
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["pilot_training_authorized"] is True
    assert manifest["full_training_authorized"] is False
    assert manifest["split"]["training_pair_count"] == 431
    assert manifest["split"]["validation_pair_count"] == 127
    assert manifest["optimization"]["epochs"] == 2
    assert manifest["loss"]["physics_derived_quantities_used"] is False


def test_protocol_and_launcher_preserve_scope_and_pairing() -> None:
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "C5P" in protocol and "E6B" in protocol
    assert "component-balanced" in protocol
    assert "full Stage-1" in protocol
    assert "for FAMILY in c5p e6b" in launcher
    assert "--gres=gpu:1" in launcher
    assert "WANDB_MODE=online" in launcher
    assert "tests/test_codec_free_stage1_pilot.py" in launcher
