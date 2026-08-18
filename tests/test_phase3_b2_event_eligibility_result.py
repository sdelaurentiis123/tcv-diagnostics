from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT / "paper0/results/phase3_b2_event_eligibility_amendment_6898348.json"
)


def result() -> dict:
    return load_strict_json(RESULT)


def test_A016_compact_result_is_hash_locked_and_85604_only() -> None:
    record = result()
    assert sha256_path(RESULT) == (
        "15bcfc0b4c9cec2a858848e1d8fbc0fdf2da6f8a3bc6fd001e5c233abae7b397"
    )
    assert record["status"] == "completed_failed_amended_one_step_gate"
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["exit_code"] == "0:0"
    assert record["full_result"]["sha256"] == (
        "4f054365d32d3e1725091ba58c8fa014f104e204748217dda482045a6c0df600"
    )


def test_A016_fixes_only_zero_event_finiteness_without_changing_decision() -> None:
    record = result()
    amendment = record["amendment"]
    assert amendment["zero_event_block_target_frames"] == [540, 561]
    assert amendment["eligible_block_indices_for_every_transport_quantity"] == [
        0,
        1,
        3,
        4,
        5,
    ]
    assert amendment["all_eligible_event_metrics_pass_every_seed"] is True
    for forbidden_change in (
        "raw_forecasts_changed",
        "raw_scores_changed",
        "metrics_recomputed",
        "training_performed",
        "inference_performed",
        "truth_scoring_performed",
    ):
        assert amendment[forbidden_change] is False
    for seed in ("1701", "1702", "1703"):
        item = record["per_seed_gate_effect"][seed]
        assert item["all_required_numeric_metrics_finite_before_A016"] is False
        assert item["all_required_numeric_metrics_finite_after_A016"] is True
        assert item["complete_gate_passed"] is False
    decision = record["decision"]
    assert decision["architecture_decision_changed"] is False
    assert decision["complete_seed_gate_pass_count"] == 0
    assert decision["median_numerical_failed_check_count"] == 106
    assert decision["median_numerical_checks_changed_by_A016"] == 0
    assert decision["O3_launch_allowed"] is False
    assert decision["assimilation_allowed"] is False


def test_B2_summary_distinguishes_point_skill_calibration_and_physics() -> None:
    summary = result()["median_scientific_summary"]
    assert summary["field"][
        "ensemble_mean_RMSE_relative_to_paired_deterministic_H2"
    ] < 1.0
    assert summary["field"]["fair_CRPS_relative_to_paired_deterministic_H2_MAE"] < 1.0
    assert summary["field"]["primary_spread_skill_count_in_0p80_to_1p25"] == 0
    assert summary["spectral"]["material_field_power_checks_passing"] == 11
    assert summary["spectral"]["material_realization_coherence_checks_passing"] == 4
    assert summary["spectral"]["cross_phase_checks_passing"] == 9
    assert summary["spectral"]["cross_coherence_change_checks_passing"] == 7
    assert summary["transport"]["separatrix"][
        "all_relative_l2_pass_0p30"
    ] is True
    assert summary["transport"]["separatrix"][
        "probabilistically_calibrated_count"
    ] == 0
    assert summary["M16_vs_M32_primary_stability_checks_all_pass"] is True
