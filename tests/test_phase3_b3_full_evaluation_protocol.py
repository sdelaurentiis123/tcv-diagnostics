"""Regression locks for the prospective B3 full-training/evaluation decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B3_FULL_EVALUATION_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase3_b3_full_evaluation_85604.json"

PROTOCOL_SHA256 = (
    "db717c5605ad9653d2b051ec13254b43bf230f514cb173d295e95d3c68af8030"
)
MANIFEST_SHA256 = (
    "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_b3_full_protocol_and_manifest_bytes_are_frozen() -> None:
    assert _sha256(PROTOCOL) == PROTOCOL_SHA256
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    record = _manifest()
    assert record["protocol"] == {
        "path": "paper0/protocol/PHASE3_B3_FULL_EVALUATION_PROTOCOL.md",
        "sha256": PROTOCOL_SHA256,
    }


def test_b3_full_scope_authorizes_one_seed_and_keeps_85606_closed() -> None:
    record = _manifest()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["full_training_authorized"] is True
    assert record["probabilistic_evaluation_authorized"] is True
    assert "B3_FGN_H1_seed1701_full_training_85604" in record["authorized_scope"]
    forbidden = set(record["forbidden_scope"])
    assert {
        "85606_access",
        "B3_seeds_1702_or_1703_training",
        "B3_architecture_or_noise_ablation",
        "O3_or_longer_rollout",
        "assimilation",
        "diagnostic_ranking",
    } <= forbidden
    assert record["decision_rule"]["O3_authorized_by_this_result"] is False
    assert record["decision_rule"]["85606_authorized_by_this_result"] is False


def test_b3_full_training_budget_matches_predeclared_candidate() -> None:
    record = _manifest()
    training = record["training"]
    assert training == {
        "seed": 1701,
        "epochs": 100,
        "targets_per_epoch": 430,
        "validation_targets": 126,
        "microbatch": 1,
        "ensemble_members_per_target": 2,
        "gradient_accumulation": 16,
        "final_partial_accumulation": 14,
        "optimizer_steps_per_epoch": 27,
        "total_optimizer_steps": 2700,
        "optimizer": "AdamW",
        "betas": [0.9, 0.99],
        "weight_decay": 0.0,
        "common_parameter_peak_learning_rate": 3e-05,
        "new_parameter_peak_learning_rate": 0.0001,
        "warmup_epochs": 10,
        "warmup_optimizer_steps": 270,
        "scheduler": (
            "independent_linear_warmup_cosine_to_zero_per_optimizer_step"
        ),
        "gradient_clip": 1.0,
        "precision": "bfloat16_autocast",
        "early_stopping": False,
        "objective": "equal_channel_decoded_standardized_field_fair_CRPS_M2",
        "physics_derived_loss_allowed": False,
        "checkpoint_selection": (
            "earliest_numerically_lowest_fixed_noise_all126_validation_"
            "equal_channel_decoded_field_fair_CRPS_after_100_epochs"
        ),
    }
    assert record["data"]["training_targets"] == [2, 432]
    assert record["data"]["guard_frames"] == [432, 496]
    assert record["data"]["validation_targets"] == [498, 624]
    assert record["data"]["absolute_time_input_allowed"] is False
    assert record["data"]["future_truth_input_allowed"] is False


def test_b3_selection_and_scientific_noise_are_independent_and_prefix_stable() -> None:
    record = _manifest()
    selection = record["selection_noise"]
    evaluation = record["scientific_ensemble"]
    assert selection["seed"] == 31003
    assert selection["shape"] == [126, 2, 32]
    assert evaluation["seed"] == 31032
    assert evaluation["noise_shape"] == [126, 32, 32]
    assert evaluation["independent_of_checkpoint_selection_noise"] is True
    assert evaluation["forecast_shape"] == [126, 32, 1, 5, 64, 32, 88]
    assert evaluation["member_prefix_sensitivity"] == [4, 8, 16, 32]
    assert evaluation["regeneration_for_member_prefixes_allowed"] is False
    assert evaluation["truth_loaded_after_forecast_hash_only"] is True
    assert evaluation["posthoc_calibration_allowed"] is False


def test_b3_full_uses_matched_H1_parent_and_secondary_B2_only() -> None:
    record = _manifest()
    parent = record["deterministic_parent"]
    assert parent["arm"] == "C5P-H1"
    assert parent["seed"] == 1701
    assert parent["checkpoint_sha256"] == (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
    )
    comparators = record["comparators"]
    assert comparators["primary_deterministic_parent"]["arm"] == "C5P-H1"
    assert comparators["primary_deterministic_parent"]["forecast_sha256"] == (
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
    )
    assert comparators["secondary_B2"]["primary_gate_comparator"] is False
    assert comparators["secondary_B2"]["forecast_sha256"] == (
        "0e3f1f2ea7dc733293dab526d0f7312d83f4d62fd9cd6708744900c5cbdb5e18"
    )


def test_b3_physics_and_calibration_gate_is_not_CRPS_only() -> None:
    record = _manifest()
    gates = record["gates"]
    assert gates["blocks_required_passing"] == 5
    assert gates["field"]["primary_spread_skill_range"] == [0.8, 1.25]
    assert gates["spectral"]["member_expected_power_ratio_range"] == [0.75, 1.3]
    assert gates["spectral"]["ensemble_mean_realization_coherence_min"] == 0.8
    assert gates["spectral"]["cross_phase_error_degrees_max"] == 20.0
    assert gates["transport"]["strict_faces"]["relative_l2_max"] == 0.4
    assert gates["transport"]["separatrix"]["relative_l2_max"] == 0.3
    assert gates["monte_carlo"] == {
        "comparison": "M16_vs_M32",
        "relative_difference_max": 0.1,
        "absolute_floor": 1e-08,
    }
    assert record["spectral"]["zperiod"] == 5
    assert record["spectral"]["mode_mapping"] == "n=5k"
    assert record["transport"]["memberwise_first"] is True
    assert record["transport"][
        "ensemble_mean_fields_for_transport_allowed"
    ] is False


def test_b3_locked_metric_source_hashes_match_repository() -> None:
    for relative, expected in _manifest()["locked_metric_sources"].items():
        assert _sha256(ROOT / relative) == expected
