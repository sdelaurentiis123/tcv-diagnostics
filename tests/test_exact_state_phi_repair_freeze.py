"""Known-answer tests for freezing the exact-state current-phi screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper0.tools.freeze_exact_state_phi_repair_screen import freeze_manifest


def _write(path: Path, record: dict) -> str:
    path.write_text(json.dumps(record), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict:
    reduction = tmp_path / "development_85604_reduction.json"
    smoke = tmp_path / "development_85604_axial_smoke.json"
    baseline = tmp_path / "development_85604_e6b_seed1701.json"
    return {
        "stage1_reduction": reduction,
        "stage1_reduction_sha256": _write(
            reduction,
            {
                "development_run": "85604",
                "held_out_85606_read": False,
                "decision": (
                    "retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation"
                ),
            },
        ),
        "axial_smoke": smoke,
        "axial_smoke_sha256": _write(
            smoke,
            {
                "development_run": "85604",
                "held_out_85606_read": False,
                "status": "passed",
                "mechanical_gates": {"reload": True, "equivariance": True},
            },
        ),
        "baseline_e6b": baseline,
        "baseline_e6b_sha256": _write(
            baseline,
            {
                "scope": "post_ecrd_old_85604_stage1_codec_free_full",
                "development_run": "85604",
                "held_out_85606_read": False,
                "family": "e6b",
                "seed": 1701,
                "best_checkpoint": {"selection_metric": 0.007772147896373167},
            },
        ),
        "paper0_commit": "a" * 40,
    }


def test_freeze_authorizes_only_matched_one_seed_screen(tmp_path: Path) -> None:
    manifest = freeze_manifest(**_inputs(tmp_path))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_read"] is False
    assert manifest["optimization"]["screen_seed"] == 1701
    assert manifest["optimization"]["epochs"] == 12
    assert manifest["three_seed_scaling_authorized"] is False
    assert manifest["parameter_count_relative_gap"] < 0.03
    assert manifest["state"]["future_auxiliary_context_allowed"] is False


def test_freeze_rejects_failed_axial_smoke(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    smoke = inputs["axial_smoke"]
    inputs["axial_smoke_sha256"] = _write(
        smoke,
        {
            "development_run": "85604",
            "held_out_85606_read": False,
            "status": "failed",
            "mechanical_gates": {"reload": False},
        },
    )
    with pytest.raises(ValueError, match="did not pass"):
        freeze_manifest(**inputs)
