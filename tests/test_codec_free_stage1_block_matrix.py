"""Known-answer tests for the sequential Stage-1 block-matrix driver."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper0.tools.run_codec_free_stage1_block_matrix import (
    evaluation_command,
    validate_block_result,
    validate_frozen_manifest,
)


COMMIT = "a" * 40


def _arm(family: str, seed: int) -> dict:
    return {
        "family": family,
        "seed": seed,
        "training_result": {
            "path": f"/development/{family}/{seed}.json",
            "sha256": "1" * 64,
        },
        "checkpoint": {
            "path": f"/development/{family}/{seed}.pt",
            "sha256": "2" * 64,
        },
    }


def _manifest() -> dict:
    return {
        "scope": "post_ecrd_old_85604_stage1_block_evaluation_input_freeze",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "physics_derived_loss_used": False,
        "evaluation_commit": COMMIT,
        "blocks": {"V00": {}, "V01": {}, "V02": {}},
        "arms": [
            _arm(family, seed)
            for family in ("c5p", "e6b")
            for seed in (1701, 1702, 1703)
        ],
    }


def test_matrix_requires_exact_six_arm_frozen_scope() -> None:
    arms = validate_frozen_manifest(_manifest(), paper0_commit=COMMIT)
    assert [(arm["seed"], arm["family"]) for arm in arms] == [
        (1701, "c5p"),
        (1701, "e6b"),
        (1702, "c5p"),
        (1702, "e6b"),
        (1703, "c5p"),
        (1703, "e6b"),
    ]
    broken = _manifest()
    broken["arms"].pop()
    with pytest.raises(ValueError, match="exactly six"):
        validate_frozen_manifest(broken, paper0_commit=COMMIT)


def test_command_uses_only_locked_arm_inputs() -> None:
    command = evaluation_command(
        python="/audited/python",
        evaluator=Path("/repo/evaluator.py"),
        artifact_root=Path("/development/85604"),
        arm=_arm("e6b", 1702),
        output=Path("/development/output.json"),
        paper0_root=Path("/repo"),
        paper0_commit=COMMIT,
    )
    assert command[0] == "/audited/python"
    assert command[command.index("--family") + 1] == "e6b"
    assert command[command.index("--seed") + 1] == "1702"
    assert command[command.index("--checkpoint-sha256") + 1] == "2" * 64
    assert "85606" not in " ".join(command)


def test_block_result_validation_fails_closed() -> None:
    result = {
        "scope": "post_ecrd_old_85604_stage1_chronological_block_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": "c5p",
        "seed": 1701,
        "paper0_evaluation_commit": COMMIT,
        "blocks": {"V00": {}, "V01": {}, "V02": {}},
    }
    validate_block_result(result, family="c5p", seed=1701, commit=COMMIT)
    result["held_out_85606_read"] = True
    with pytest.raises(ValueError, match="held-out"):
        validate_block_result(result, family="c5p", seed=1701, commit=COMMIT)
