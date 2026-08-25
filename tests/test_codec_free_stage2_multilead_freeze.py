"""Known-answer checks for freezing the old-85604 multi-lead screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import paper0.tools.freeze_codec_free_stage2_multilead as freezer
from paper0.tools.train_codec_free_stage2_multilead import build_model


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return _sha(path)


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "development_85604_multilead"
    root.mkdir()
    reduction = root / "stage1_reduction.json"
    reduction_sha = _write_json(
        reduction,
        {
            "development_run": "85604",
            "held_out_85606_read": False,
            "decision": (
                "retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation"
            ),
        },
    )
    architecture = {
        "base_channels": 24,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "lead_embedding_channels": 128,
        "group_norm_maximum_groups": 8,
        "kernel_size": 3,
        "zero_initialize_output": True,
    }
    model, config = build_model(architecture)
    checkpoint = root / "checkpoint_epoch_012.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_record(),
            "family": "c5p",
            "seed": 1701,
            "epoch": 12,
            "optimizer_updates": 1296,
            "selection_metric": freezer.PARENT_ONE_STEP_SHARED_MSE,
            "paper0_commit": "a" * 40,
        },
        checkpoint,
    )
    checkpoint_sha = _sha(checkpoint)
    monkeypatch.setattr(freezer, "PARENT_CHECKPOINT_SHA256", checkpoint_sha)
    result = root / "parent_result.json"
    result_sha = _write_json(
        result,
        {
            "scope": "post_ecrd_old_85604_stage1_codec_free_full",
            "development_run": "85604",
            "held_out_85606_read": False,
            "physics_derived_loss_used": False,
            "family": "c5p",
            "seed": 1701,
            "status": "passed",
            "training_gate": {"passed": True},
            "best_checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "selection_metric": freezer.PARENT_ONE_STEP_SHARED_MSE,
            },
        },
    )
    return {
        "stage1_reduction": reduction,
        "stage1_reduction_sha256": reduction_sha,
        "parent_result": result,
        "parent_result_sha256": result_sha,
        "parent_checkpoint": checkpoint,
        "parent_checkpoint_sha256": checkpoint_sha,
        "paper0_commit": "b" * 40,
    }


def test_freeze_authorizes_exactly_one_c5p_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = freezer.freeze_manifest(**_inputs(tmp_path, monkeypatch))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["state_family"] == "c5p"
    assert manifest["split"]["lead_steps"] == [1, 2, 4, 8, 16]
    assert manifest["optimization"]["expected_optimizer_updates"] == 2132
    assert manifest["three_seed_scaling_authorized"] is False
    assert manifest["screen_gates"]["maximum_lead1_shared_mse"] == pytest.approx(
        freezer.PARENT_ONE_STEP_SHARED_MSE * 1.05
    )


def test_freeze_rejects_checkpoint_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    inputs["parent_checkpoint"].write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA-256"):
        freezer.freeze_manifest(**inputs)
