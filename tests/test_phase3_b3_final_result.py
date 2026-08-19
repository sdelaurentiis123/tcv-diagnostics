"""Regression lock for the complete B3 FGN one-seed evaluation and gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = (
    ROOT / "paper0/results/phase3_b3_fgn_evaluation_full_6899073.json"
)
GATE = ROOT / "paper0/results/phase3_b3_fgn_one_seed_gate_6899224.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_evaluation_manifest_is_byte_identical_and_truth_separated() -> None:
    assert digest(EVALUATION) == (
        "87b6ea353bfe9928404f01d1b494c94bfd2491395c28c0ec0a46105f0ee5e20c"
    )
    record = load(EVALUATION)
    assert record["status"] == "completed_pending_frozen_acceptance_gate"
    assert record["scope"] == "phase3_B3_FGN_H1_full_probabilistic_evaluation"
    assert record["slurm_job_id"] == "6899073"
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["target_frames"] == [498, 624]
    assert record["target_count"] == 126
    assert record["ensemble_members"] == 32
    assert record["direct_network_evaluations_per_member"] == 1
    assert record["truth_opened_only_after_forecast_hash"] is True
    assert record["forecast"]["bytes"] == 14_535_252_816
    assert record["forecast"]["sha256"] == (
        "0f5c97b20fbf7ef32f2bd2b9695dc173d78155dcde356ef5b1a451dc4276e3ef"
    )
    assert record["score"]["sha256"] == (
        "c32508a85a68859aa676d2fada4f76a304984fea5988c81fb106ae6f724654d0"
    )
    assert record["scientific_noise"]["sha256"] == (
        "1449777a61d40af49ccb3bd6bed5edcba0fd8afe24d113e6175218c04865aa9c"
    )
    assert record["wandb"]["remote_state_after_finish"] == "finished"


def test_compact_gate_is_byte_locked_integrity_clean_and_85604_only() -> None:
    assert digest(GATE) == (
        "f8ac75e65586aaa40b905ad4d447f15cad218deaa1119246d518b7730ede0dd3"
    )
    record = load(GATE)
    assert record["status"] == "completed_failed_frozen_one_seed_gate"
    assert record["development_run"] == "85604"
    assert record["held_out_85606_read"] is False
    assert record["seed"] == 1701
    assert record["gate"]["job_id"] == "6899224"
    assert record["gate"]["integrity_passes"] is True
    assert record["gate"]["integrity_failed_check_count"] == 0
    assert record["gate"]["all_required_numeric_metrics_finite"] is True
    assert record["gate"]["full_result"]["sha256"] == (
        "882ba10898bbf132eea7713098202d8e814e3f709e8693c5b25366c52ffbc391"
    )


def test_path_alias_amendment_changed_only_integrity_identity() -> None:
    amendment = load(GATE)["gate_path_alias_amendment"]
    assert amendment["original_gate_job_id"] == "6899154"
    assert amendment["original_gate_sha256"] == (
        "ad5a957254484d1c95228117dc0911f16bcef742f194c8f53dba6b87d9281f4c"
    )
    assert amendment["old_integrity_failed_check_count"] == 1
    assert amendment["new_integrity_failed_check_count"] == 0
    assert amendment["family_numerical_values_changed"] is False
    assert amendment["family_summaries_changed"] is False
    assert amendment["decision_changed"] is False


def test_every_B3_metric_family_failed_the_frozen_gate() -> None:
    families = load(GATE)["family_summary"]
    assert families == {
        "field": {
            "blocks_passing": 0,
            "blocks_required": 5,
            "check_count": 54,
            "failed_check_count": 6,
            "passes": False,
        },
        "spectral": {
            "blocks_passing": 0,
            "blocks_required": 5,
            "check_count": 148,
            "failed_check_count": 59,
            "passes": False,
        },
        "transport": {
            "blocks_passing": 0,
            "blocks_required": 5,
            "check_count": 77,
            "failed_check_count": 6,
            "passes": False,
        },
    }


def test_marginal_improvement_does_not_hide_joint_physics_failure() -> None:
    record = load(GATE)
    aggregate = record["field_and_marginal"]["aggregate"]
    assert aggregate["ensemble_mean_MAE_relative_to_parent_H1"] == (
        1.0030825630261506
    )
    assert aggregate["ensemble_mean_RMSE_relative_to_parent_H1"] == (
        0.9972214817968783
    )
    assert aggregate["fair_CRPS_relative_to_parent_H1_MAE"] == (
        0.7230480491810231
    )
    assert record["field_and_marginal"]["fields_meeting_primary_spread_skill"] == 1
    assert record["field_and_marginal"]["fields_meeting_primary_coverage"] == 1
    spectral = record["spectral_and_cross_field"]
    assert spectral["material_field_power_checks"] == {"passing": 11, "total": 15}
    assert spectral["material_realization_coherence_checks"] == {
        "passing": 4,
        "total": 15,
    }
    assert spectral["cross_phase_absolute_error_degrees"]["passing"] == 9
    assert spectral["cross_coherence_change"]["passing"] == 9
    assert record["transport"]["separatrix_calibrated_count"] == 0
    assert record["transport"][
        "separatrix_fair_CRPS_better_than_parent_H1_count"
    ] == 4


def test_mode_mapping_monte_carlo_stability_and_stop_decision_are_frozen() -> None:
    record = load(GATE)
    assert record["mode_mapping"] == {
        "k1_3": "n=5..15",
        "k4_5": "n=20..25",
        "k6_7": "n=30..35",
        "rule": "n=5k",
        "zperiod": 5,
    }
    assert record["monte_carlo_stability"]["relative_difference"] < 0.001
    decision = record["decision"]
    assert decision["passes_complete_one_seed_gate"] is False
    assert decision["post_gate_instruction"] == (
        "stop_B3_and_diagnose_before_replication"
    )
    assert decision["seed1702_1703_training_authorized"] is False
    assert decision["seed1702_1703_replication_protocol_may_be_written"] is False
    assert decision["O3_launch_allowed"] is False
    assert decision["assimilation_allowed"] is False
    assert decision["diagnostic_ranking_allowed"] is False
    assert record["interpretation"]["next_action"] == (
        "freeze a separately justified B4 PDE-Refiner protocol; do not replicate B3"
    )
