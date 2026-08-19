"""Contract tests for the frozen B5 H1 residual-audit decision."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "paper0/manifests/phase3_b5_residual_audit_85604.json"
PROTOCOL_PATH = ROOT / "paper0/protocol/PHASE3_B5_RESIDUAL_AUDIT_PROTOCOL.md"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_B5_residual_audit_keeps_validation_guard_and_85606_closed() -> None:
    manifest = _manifest()
    assert manifest["development_run"] == "85604"
    assert manifest["sequestered_run"] == "85606"
    assert manifest["held_out_85606_access_allowed"] is False
    data = manifest["data"]
    assert data["training_frames"] == [0, 432]
    assert data["training_targets"] == [2, 432]
    assert data["training_target_count"] == 430
    assert data["guard_frames"] == [432, 496]
    assert data["validation_frames"] == [496, 624]
    assert data["validation_frames_read_allowed"] is False
    assert data["guard_frames_read_allowed"] is False
    assert data["target_truth_input_allowed_during_forecast_generation"] is False


def test_B5_residual_audit_uses_exact_H1_and_physical_state_channels() -> None:
    manifest = _manifest()
    data = manifest["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["input_channels"] == "physically_valid_complete_C5P_state"
    assert data["absolute_time_input_allowed"] is False
    assert data["volume_shape"] == [5, 64, 32, 88]
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    parent = manifest["deterministic_mean"]
    assert parent["arm"] == "C5P-H1"
    assert parent["seed"] == 1701
    assert parent["checkpoint_sha256"] == (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
    )
    assert parent["retraining_allowed"] is False
    assert parent["reselection_allowed"] is False


def test_B5_residual_definition_is_truth_separated_joint_and_nonzero_mean() -> None:
    manifest = _manifest()
    residual = manifest["residual"]
    assert residual["definition"] == (
        "standardized_truth_minus_frozen_H1_standardized_forecast"
    )
    assert residual["truth_separated_forecast_generation_required"] is True
    assert residual["forecast_closed_and_hashed_before_truth_read"] is True
    assert residual["nonzero_mean_preserved"] is True
    assert residual["irreducible_aleatoric_interpretation_allowed"] is False
    rules = manifest["architecture_rules_after_audit"]
    assert rules["primary_residual_representation"] == (
        "joint_five_field_decoded_standardized_coordinates"
    )
    assert rules["independent_per_field_primary_allowed"] is False
    assert rules["latent_branch_allowed"] is False


def test_B5_residual_audit_preserves_full_toroidal_support() -> None:
    manifest = _manifest()
    rules = manifest["architecture_rules_after_audit"]
    assert rules["complete_stored_toroidal_axis_required"] is True
    assert rules["nonperiodic_patch_minimum_span"] == (
        "min(domain_extent,2*stable_near_zero_lag+1)"
    )
    bands = manifest["measurements"]["toroidal_bands"]
    assert [item["label"] for item in bands] == [
        "k0",
        "k1_3",
        "k4_5",
        "k6_7",
        "k_ge_8",
    ]
    assert bands[3]["stored_k"] == [6, 7]
    assert bands[3]["full_torus_n"] == [30, 35]


def test_B5_residual_audit_authorizes_no_training_or_downstream_claim() -> None:
    manifest = _manifest()
    assert manifest["execution"]["training_performed"] is False
    assert manifest["execution"]["accelerator"] == "one_H100"
    assert manifest["execution"]["submission"] == "sbatch_from_rusty9"
    assert manifest["execution"]["wandb"]["required"] is True
    post = manifest["post_audit"]
    assert post == {
        "B5_training_authorized": False,
        "B5_implementation_protocol_may_be_written": True,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "held_out_85606_access_allowed": False,
    }


def test_B5_residual_audit_protocol_contains_formal_definitions_and_cautions() -> None:
    text = PROTOCOL_PATH.read_text()
    for required in (
        "r_{t,c,x,y,z}",
        "r'_{t,c,x,y,z}",
        "\\rho_{c,z}(\\ell)",
        "C_{ij}",
        "in-sample residuals",
        "cannot be separated uniquely",
        "full toroidal axis",
        "CorrDiff",
        "GenCast",
    ):
        assert required in text
