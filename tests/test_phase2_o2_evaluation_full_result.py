"""Regression lock for the completed full C5P O2 evaluation on job 6896117."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase2_o2_evaluation_full_6896117.json"
RESULT_SHA256 = "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb"


@pytest.fixture(scope="module")
def record() -> dict:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_full_o2_result_is_scientific_complete_and_blind(record: dict):
    assert record["status"] == "completed_failed_acceptance_gate"
    assert record["scientific_authority"] is True
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["paper0_commit"] == (
        "5183023c4df3a38bd6821f7e7cf587507aacc241"
    )
    assert record["slurm_job_id"] == "6896117"
    assert record["experiment"]["codec"]["name"] == "C5P-dcae_l10"
    assert record["experiment"]["target_frames"] == [498, 624]
    assert record["experiment"]["target_count"] == 126
    assert record["experiment"]["validation_blocks"] == 6
    assert record["experiment"]["target_truth_read_during_forecast_generation"] is False
    assert record["experiment"]["physics_derived_training_loss_used"] is False
    assert record["experiment"]["toroidal_period_count"] == 5


def test_all_six_models_have_real_field_skill_but_fail_physics_gate(record: dict):
    seeds = record["per_seed"]
    assert [(item["arm"], item["seed"]) for item in seeds] == [
        ("C5P-H1", 1701),
        ("C5P-H1", 1702),
        ("C5P-H1", 1703),
        ("C5P-H2", 1701),
        ("C5P-H2", 1702),
        ("C5P-H2", 1703),
    ]
    assert all(item["reference_skill_pass"] is True for item in seeds)
    assert all(item["all_field_gate_pass"] is True for item in seeds)
    assert all(item["cross_field_gate_pass"] is True for item in seeds)
    assert all(item["spectral_gate_pass"] is False for item in seeds)
    assert all(item["transport_gate_pass"] is False for item in seeds)
    assert all(item["O2_seed_pass"] is False for item in seeds)

    summary = record["arm_summary"]
    assert summary["C5P-H1"]["aggregate_RMSE_standardized_mean"] == pytest.approx(
        0.08003711132356371
    )
    assert summary["C5P-H2"]["aggregate_RMSE_standardized_mean"] == pytest.approx(
        0.08061870571782304
    )
    assert summary["history_comparison"]["H2_minus_H1_RMSE_fraction"] > 0


def test_failure_anatomy_and_stop_decision_are_locked(record: dict):
    physics = record["cross_seed_physics_summary"]
    assert physics["field_error"][
        "seeds_beating_best_applicable_reference_on_RMSE_and_MAE"
    ] == 6
    assert physics["cross_field"][
        "seeds_passing_every_cross_field_check_overall_and_in_all_six_blocks"
    ] == 6
    assert physics["spectral"]["seeds_passing_complete_spectral_gate"] == 0
    assert physics["transport"]["seeds_passing_complete_transport_gate"] == 0
    assert physics["transport"][
        "strict_face_complete_pass_count_across_24_seed_quantity_checks"
    ] == 0

    decision = record["decision"]
    assert decision["accepted_arms"] == []
    assert decision["disposition"] == "stop_and_report_deterministic_one_step_failure"
    assert decision["new_O3_protocol_may_be_frozen"] is False
    assert decision["O3_launch_allowed"] is False
    assert decision["stochastic_model_authorized_by_this_result"] is False
    assert decision["held_out_85606_access_allowed"] is False


def test_external_result_and_tracking_records_are_hash_anchored(record: dict):
    external = record["external_artifacts"]
    assert external["final_matrix"]["sha256"] == (
        "a60299ec59400910198b6ffd64aa99412c7658b9b773e6a7908cdc03a0b73821"
    )
    assert external["artifact_index"]["sha256"] == (
        "3513e44439182086b0df6c8dc2fe8e070ca5ce0e44d5948fe498bff3913cddd5"
    )
    assert record["verification"][
        "all_six_forecast_score_result_and_nested_index_hashes_reverified"
    ] is True
    assert record["verification"]["wandb_online_and_finished"] is True
    assert record["wandb"]["remote_state_after_finish"] == "finished"
