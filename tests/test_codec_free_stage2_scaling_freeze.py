"""Known-answer checks for freezing Stage-2 seed confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import paper0.tools.freeze_codec_free_stage2_scaling as freezer
from paper0.tools.train_codec_free_stage2_multilead import build_model


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return _sha(path)


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "development_85604_scaling"
    root.mkdir()
    reduction = root / "reduction.json"
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
    screen = root / "seed1701_screen.json"
    screen_sha = _write_json(
        screen,
        {
            "scope": "post_ecrd_old_85604_stage2_multilead_screen",
            "development_run": "85604",
            "held_out_85606_read": False,
            "seed": 1701,
            "advance_to_three_seed_scaling": True,
            "screen_gates": {
                "at_least_three_longer_leads_improve": True,
                "every_c5p_field_positive_skill_at_every_lead": True,
                "lead1_shared_mse_at_most_five_percent_above_parent": True,
                "mean_multilead_ratio_improves_at_least_ten_percent": True,
                "training_gate_passed": True,
            },
        },
    )
    monkeypatch.setattr(freezer, "SCREEN_RESULT_SHA256", screen_sha)

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
    legacy_config = config.to_record()
    legacy_config.pop("auxiliary_context_channels")
    values: dict[str, object] = {
        "stage1_reduction": reduction,
        "stage1_reduction_sha256": reduction_sha,
        "seed1701_screen_result": screen,
        "seed1701_screen_result_sha256": screen_sha,
        "paper0_commit": "d" * 40,
    }
    for seed in (1702, 1703):
        checkpoint = root / f"seed{seed}_checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": legacy_config,
                "family": "c5p",
                "seed": seed,
                "epoch": 12,
                "optimizer_updates": 1296,
                "selection_metric": freezer.PARENT_METRICS[seed],
                "paper0_commit": "a" * 40,
            },
            checkpoint,
        )
        checkpoint_sha = _sha(checkpoint)
        monkeypatch.setitem(
            freezer.PARENT_CHECKPOINT_SHA256, seed, checkpoint_sha
        )
        result = root / f"seed{seed}_result.json"
        result_sha = _write_json(
            result,
            {
                "scope": "post_ecrd_old_85604_stage1_codec_free_full",
                "development_run": "85604",
                "held_out_85606_read": False,
                "physics_derived_loss_used": False,
                "family": "c5p",
                "seed": seed,
                "status": "passed",
                "training_gate": {"passed": True},
                "best_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "selection_metric": freezer.PARENT_METRICS[seed],
                },
            },
        )
        values[f"seed{seed}_result"] = result
        values[f"seed{seed}_result_sha256"] = result_sha
        values[f"seed{seed}_checkpoint"] = checkpoint
        values[f"seed{seed}_checkpoint_sha256"] = checkpoint_sha
    return values


def test_freeze_authorizes_exactly_two_confirmation_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = freezer.freeze_scaling_manifest(**_inputs(tmp_path, monkeypatch))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["new_nersc_data_access_allowed"] is False
    assert manifest["optimization"]["authorized_seeds"] == [1702, 1703]
    assert set(manifest["parents"]) == {"1702", "1703"}
    assert manifest["three_seed_scaling_authorized"] is True
    assert manifest["conditional_bounded_rollout_authorized"] is True
    for seed in (1702, 1703):
        assert manifest["parents"][str(seed)]["checkpoint_config_validation"][
            "inserted_legacy_defaults"
        ] == ["auxiliary_context_channels"]


def test_freeze_rejects_screen_without_scaling_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    path = inputs["seed1701_screen_result"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["advance_to_three_seed_scaling"] = False
    inputs["seed1701_screen_result_sha256"] = _write_json(path, record)
    monkeypatch.setattr(
        freezer,
        "SCREEN_RESULT_SHA256",
        inputs["seed1701_screen_result_sha256"],
    )
    with pytest.raises(ValueError, match="did not authorize"):
        freezer.freeze_scaling_manifest(**inputs)
