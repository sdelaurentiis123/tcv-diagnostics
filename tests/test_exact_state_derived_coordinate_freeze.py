"""Known-answer tests for freezing the phi-plus-Vi coordinate screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper0.tools.freeze_exact_state_derived_coordinate_screen import (
    AXIAL_PHI_SHARED_MSE,
    BASELINE_SHARED_MSE,
    LOCAL_PHI_SHARED_MSE,
    freeze_manifest,
)


def _write(path: Path, record: dict) -> str:
    path.write_text(json.dumps(record), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repair_result(architecture: str, metric: float, parameter_count: int) -> dict:
    return {
        "scope": "post_ecrd_old_85604_exact_state_phi_repair_screen",
        "development_run": "85604",
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "architecture_kind": architecture,
        "seed": 1701,
        "status": "passed",
        "training_gate": {"passed": True},
        "advance_to_three_seed_scaling": False,
        "screen_gates": {
            "at_least_15_percent_shared_mse_improvement_over_seed1701_e6b": False
        },
        "best_checkpoint": {"selection_metric": metric},
        "architecture": {"parameter_count": parameter_count},
    }


def _inputs(tmp_path: Path) -> dict:
    reduction = tmp_path / "development_85604_reduction.json"
    baseline = tmp_path / "development_85604_e6b_seed1701.json"
    local = tmp_path / "development_85604_local_phi.json"
    axial = tmp_path / "development_85604_axial_phi.json"
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
        "baseline_e6b": baseline,
        "baseline_e6b_sha256": _write(
            baseline,
            {
                "scope": "post_ecrd_old_85604_stage1_codec_free_full",
                "development_run": "85604",
                "held_out_85606_read": False,
                "family": "e6b",
                "seed": 1701,
                "best_checkpoint": {"selection_metric": BASELINE_SHARED_MSE},
            },
        ),
        "local_phi": local,
        "local_phi_sha256": _write(
            local,
            _repair_result("local_current_phi", LOCAL_PHI_SHARED_MSE, 2_182_352),
        ),
        "axial_phi": axial,
        "axial_phi_sha256": _write(
            axial,
            _repair_result("axial_current_phi", AXIAL_PHI_SHARED_MSE, 2_131_544),
        ),
        "paper0_commit": "b" * 40,
    }


def test_freeze_authorizes_one_causal_phi_vi_arm(tmp_path: Path) -> None:
    manifest = freeze_manifest(**_inputs(tmp_path))
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["state"]["auxiliary_context_fields"] == ["phi", "Vi"]
    assert manifest["state"]["future_auxiliary_context_allowed"] is False
    assert manifest["architectures"].keys() == {"local_current_phi_vi"}
    assert manifest["optimization"]["screen_seed"] == 1701
    assert manifest["three_seed_scaling_authorized"] is False
    assert manifest["parameter_count_relative_gap"] < 0.03
    assert manifest["screen_gates"]["maximum_shared_mse"] == pytest.approx(
        BASELINE_SHARED_MSE * 0.85
    )


def test_freeze_rejects_phi_result_that_advanced(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    local = inputs["local_phi"]
    record = _repair_result("local_current_phi", LOCAL_PHI_SHARED_MSE, 2_182_352)
    record["advance_to_three_seed_scaling"] = True
    inputs["local_phi_sha256"] = _write(local, record)
    with pytest.raises(ValueError, match="unexpectedly advanced"):
        freeze_manifest(**inputs)
