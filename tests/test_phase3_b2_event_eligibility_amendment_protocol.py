"""Regression checks for the prospective B2 event-eligibility amendment."""

from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B2_EVENT_ELIGIBILITY_AMENDMENT.md"
MANIFEST = (
    ROOT / "paper0/manifests/phase3_b2_event_eligibility_amendment_85604.json"
)


def manifest() -> dict:
    return load_strict_json(MANIFEST)


def test_amendment_is_outcome_informed_and_original_result_is_immutable() -> None:
    record = manifest()
    assert record["status"] == (
        "frozen_before_amended_evaluator_implementation_or_execution"
    )
    assert record["outcome_informed_amendment"] is True
    original = record["original_result"]
    assert original["job_id"] == "6897564"
    assert original["status"] == "completed_failed_frozen_one_step_gate"
    assert original["sha256"] == (
        "cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6"
    )
    assert original["retained_immutable"] is True
    assert original["forecast_or_score_recomputation_allowed"] is False


def test_protocol_hash_and_truth_only_eligibility_are_frozen() -> None:
    record = manifest()
    assert sha256_path(PROTOCOL) == (
        "ddfff77f48ae11117ccc8f0bd0f27043421ee700046e975c8263087e0142e2a9"
    )
    assert record["protocol"]["sha256"] == sha256_path(PROTOCOL)
    event = record["event_eligibility"]
    assert event["source"] == "truth_validation_event_count_only"
    assert event["eligible_expression"] == "validation_event_count > 0"
    assert event["forecast_dependent_eligibility_allowed"] is False
    assert event["minimum_eligible_blocks_per_quantity"] == 5
    assert event["eligible_block_requirements"][
        "every_eligible_block_must_pass"
    ] is True
    assert event["zero_event_block_requirements"] == {
        "defined": False,
        "event_accuracy_status": "not_applicable",
        "magnitude_relative_error": None,
        "truth_magnitude_weighted_sign_disagreement": None,
    }


def test_amendment_permits_only_consistent_gate_reduction_on_85604() -> None:
    record = manifest()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    rerun = record["consistent_rerun"]
    assert rerun["seeds"] == [1701, 1702, 1703]
    assert rerun["inference_allowed"] is False
    assert rerun["training_allowed"] is False
    assert rerun["truth_scoring_allowed"] is False
    assert rerun["gate_only_reduction_allowed"] is True
    for forbidden in (
        "85606_access",
        "model_or_checkpoint_tuning",
        "forecast_regeneration",
        "score_recomputation",
        "O3_or_longer_rollout",
        "assimilation",
        "diagnostic_ranking",
    ):
        assert forbidden in record["forbidden_scope"]
