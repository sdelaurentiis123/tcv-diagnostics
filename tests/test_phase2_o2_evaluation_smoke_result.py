"""Regression lock for completed non-scientific O2 evaluator smoke 6895931."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase2_o2_evaluation_smoke_6895931.json"
RESULT_SHA256 = "dc53d9561d0ef0f00cbb41eb14f510fd4a19cf5427acd17be2947baff4211273"


def test_completed_evaluator_smoke_is_immutable_bounded_and_non_scientific():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    record = json.loads(RESULT.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["scientific_authority"] is False
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["paper0_commit"] == (
        "6fb0c1455dac0dc7b75953c2fc6f091788bbc7a5"
    )
    assert record["slurm_job_id"] == "6895931"
    assert record["compute"]["state"] == "COMPLETED"
    assert record["compute"]["exit_code"] == "0:0"
    assert record["authorized_work"]["target_frames"] == [498, 502]
    assert record["authorized_work"]["target_count"] == 4
    assert record["authorized_work"]["target_truth_read_during_forecast_generation"] is False
    assert record["authorized_work"]["native_geometry_aware_transport_scored"] is True


def test_smoke_verification_releases_only_full_85604_evaluation():
    record = json.loads(RESULT.read_text(encoding="utf-8"))
    verification = record["verification"]
    assert verification["complete_Rocky9_CPU_suite"]["pytest_passed"] == 610
    assert verification["all_reference_artifact_hashes_reverified"] is True
    assert verification["all_candidate_artifact_hashes_reverified"] is True
    assert verification["wandb_online_and_finished"] is True
    assert record["descriptive_only"][
        "may_be_used_for_scientific_model_acceptance"
    ] is False
    decision = record["decision"]
    assert decision["smoke_passed"] is True
    assert decision["O2_scientific_gate_evaluated"] is False
    assert decision["full_85604_O2_evaluation_may_be_launched"] is True
    assert decision["O3_launch_allowed"] is False
    assert decision["stochastic_model_authorized"] is False
    assert decision["held_out_85606_access_allowed"] is False
