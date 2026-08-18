from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tcv_diagnostics.b2_acceptance_gate import (
    evaluate_b2_architecture_acceptance,
    evaluate_b2_seed_acceptance,
)
from tcv_diagnostics.b2_field_metrics import (
    B2_FIELDS,
    B2_INTERVALS,
    B2_PRIMARY_REGIONS,
)
from tcv_diagnostics.b2_spectral_metrics import B2_CROSS_PAIRS, B2_MODE_BANDS
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "paper0/manifests/phase3_b2_full_evaluation_85604.json").read_text()
)


def _intervals() -> dict[str, dict[str, float]]:
    return {
        name: {
            "nominal_coverage": (upper - lower) / 33.0,
            "empirical_coverage": (upper - lower) / 33.0,
        }
        for name, (lower, upper) in B2_INTERVALS.items()
    }


def _field_record() -> dict[str, object]:
    return {
        "ensemble_mean": {"rmse": 0.8, "mae": 0.8},
        "fair_crps": 0.5,
        "corrected_spread_skill": {"ratio": 1.0},
        "spread_integrity": {"nonzero_spread": True},
        "order_statistic_intervals": _intervals(),
    }


def _field_scope() -> dict[str, object]:
    return {
        "fields": {field: _field_record() for field in B2_FIELDS},
        "aggregate": {
            "equal_channel_ensemble_mean_rmse": 0.8,
            "equal_channel_ensemble_mean_mae": 0.8,
            "equal_channel_fair_crps": 0.5,
        },
    }


def _deterministic_field_scope() -> dict[str, object]:
    return {
        "fields": {field: {"rmse": 1.0, "mae": 1.0} for field in B2_FIELDS},
        "aggregate_equal_channel_rmse_standardized": 1.0,
        "aggregate_equal_channel_mae_standardized": 1.0,
    }


def _field_score() -> dict[str, object]:
    prefixes = {}
    for members in (16, 32):
        prefixes[f"M{members}"] = {
            "fields": {field: {"fair_crps": 0.5} for field in B2_FIELDS},
            "aggregate": {"equal_channel_fair_crps": 0.5},
        }
    return {
        "regions": {
            region: _field_scope() for region in ("eligible_union", *B2_PRIMARY_REGIONS)
        },
        "chronological_blocks_eligible_union": [_field_scope() for _ in range(6)],
        "member_prefix_sensitivity_eligible_union": prefixes,
    }


def _field_comparator() -> dict[str, object]:
    return {
        "overall": _deterministic_field_scope(),
        "chronological_blocks": [_deterministic_field_scope() for _ in range(6)],
    }


def _calibration() -> dict[str, object]:
    return {
        "primary_M32": {"corrected_spread_skill": {"spread_skill_ratio": 1.0}},
        "order_statistic_intervals": {"I31": {"empirical_coverage": 31.0 / 33.0}},
        "M16_vs_M32_stability": {
            "fair_crps": {
                "absolute_difference": 0.0,
                "tolerance": 0.05,
                "passes": True,
            }
        },
    }


def _materiality() -> dict[str, object]:
    fields = {}
    for field in B2_FIELDS:
        fields[field] = {
            "bands": {
                label: {"material": field == "Ne" and label == "k1_3"}
                for label, _, _ in B2_MODE_BANDS
            }
        }
    crosses = {}
    for first, second in B2_CROSS_PAIRS:
        pair = f"{first}-{second}"
        crosses[pair] = {
            "bands": {
                label: {"material": pair == "Ne-phi" and label == "k1_3"}
                for label, _, _ in B2_MODE_BANDS
            }
        }
    return {"fields": fields, "cross_fields": crosses}


def _spectral_scope() -> dict[str, object]:
    calibration = _calibration()
    return {
        "toroidal_field_power": {
            "Ne": {
                "bands": {
                    "k1_3": {
                        "member_expected_power_ratio": 1.0,
                        "ensemble_mean_realization_coherence_with_truth": 0.9,
                        "per_target_band_power_calibration": calibration,
                    }
                }
            }
        },
        "toroidal_cross_field": {
            "Ne-phi": {
                "bands": {
                    "k1_3": {
                        "truth_amplitude_weighted_absolute_phase_error_degrees": 5.0,
                        "truth_amplitude_weighted_absolute_coherence_change": 0.05,
                        "per_target_cross_projection_calibration": {
                            "real": deepcopy(calibration),
                            "imaginary": deepcopy(calibration),
                        },
                    }
                }
            }
        },
    }


def _paired_metrics() -> dict[str, object]:
    return {
        "relative_l2": 0.1,
        "normalized_bias": 0.01,
        "pearson_correlation": 0.9,
        "weighted_sign_disagreement": 0.01,
    }


def _probabilistic_transport() -> dict[str, object]:
    return {
        "fair_crps": 0.5,
        "corrected_spread_skill": {"ratio": 1.0},
        "spread_integrity": {"nonzero_spread": True},
        "order_statistic_intervals": _intervals(),
    }


def _transport_scope(*, detailed: bool) -> dict[str, object]:
    quantities = {}
    for quantity in TRANSPORT_QUANTITIES:
        sep = {
            "ensemble_expected_paired_metrics": _paired_metrics(),
            "ensemble_probabilistic_metrics": _probabilistic_transport(),
        }
        if detailed:
            sep["M16_vs_M32_stability"] = {
                "fair_crps": {
                    "absolute_difference": 0.0,
                    "tolerance": 0.05,
                    "passes": True,
                }
            }
        quantities[quantity] = {
            "reductions": {
                "strict_face_contributions": {
                    "ensemble_expected_paired_metrics": _paired_metrics(),
                    "ensemble_probabilistic_metrics": _probabilistic_transport(),
                },
                "separatrix_wedge": sep,
            },
            "upper_decile_event_conditioned": {
                "defined": True,
                "magnitude_relative_error": 0.1,
                "truth_magnitude_weighted_sign_disagreement": 0.01,
            },
        }
    return {"quantities": quantities}


def _transport_score() -> dict[str, object]:
    return {
        "overall": _transport_scope(detailed=True),
        "chronological_blocks": [_transport_scope(detailed=False) for _ in range(6)],
    }


def _transport_comparator() -> dict[str, object]:
    scope = {
        "quantities": {
            quantity: {"separatrix_absolute_error": 1.0}
            for quantity in TRANSPORT_QUANTITIES
        }
    }
    return {
        "overall": deepcopy(scope),
        "chronological_blocks": [deepcopy(scope) for _ in range(6)],
    }


def _result(seed: int) -> dict[str, object]:
    return {
        "scope": "B2_LDM_H2_full_probabilistic_evaluation_85604",
        "status": "completed_pending_frozen_acceptance_gate",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_forecast_hash": True,
        "target_frames": [498, 624],
        "target_count": 126,
        "ensemble_members": 32,
        "physics_derived_training_loss_used": False,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "seed": seed,
    }


def _score(seed: int) -> dict[str, object]:
    return {
        "scope": "B2_truth_separated_probabilistic_scoring_85604",
        "bounded_non_scientific_smoke": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "truth_opened_only_after_forecast_was_closed_and_hash_verified": True,
        "training_performed": False,
        "physics_derived_training_loss_used": False,
        "target_frames": [498, 624],
        "target_count": 126,
        "model_seed": seed,
        "transport_event_thresholds": {"spectral_materiality": _materiality()},
        "field_and_marginal_calibration": _field_score(),
        "spectral_and_cross_field": {
            "overall": _spectral_scope(),
            "chronological_blocks": [_spectral_scope() for _ in range(6)],
        },
        "memberwise_transport": _transport_score(),
    }


def _seed_record(seed: int) -> dict[str, object]:
    return evaluate_b2_seed_acceptance(
        result=_result(seed),
        score=_score(seed),
        training_run={
            "seed": seed,
            "training_complete": True,
            "scientific_acceptance_evaluated": False,
        },
        comparator_run={
            "seed": seed,
            "field": _field_comparator(),
            "transport": _transport_comparator(),
        },
        best_uncompressed={"field": _field_comparator()},
        manifest=MANIFEST,
    )


def test_complete_seed_gate_passes_known_answer_record() -> None:
    record = _seed_record(1701)
    assert record["integrity"]["passes"] is True
    assert all(family["passes"] for family in record["families"].values())
    assert record["passes_complete_per_seed_gate"] is True
    assert record["catastrophic_bounds"]["passes"] is True


def test_seed_gate_fails_material_cross_phase_without_hiding_other_families() -> None:
    score = _score(1701)
    score["spectral_and_cross_field"]["overall"]["toroidal_cross_field"]["Ne-phi"][
        "bands"
    ]["k1_3"]["truth_amplitude_weighted_absolute_phase_error_degrees"] = 25.0
    record = evaluate_b2_seed_acceptance(
        result=_result(1701),
        score=score,
        training_run={
            "seed": 1701,
            "training_complete": True,
            "scientific_acceptance_evaluated": False,
        },
        comparator_run={
            "seed": 1701,
            "field": _field_comparator(),
            "transport": _transport_comparator(),
        },
        best_uncompressed={"field": _field_comparator()},
        manifest=MANIFEST,
    )
    assert record["families"]["field"]["passes"] is True
    assert record["families"]["spectral"]["passes"] is False
    assert record["passes_complete_per_seed_gate"] is False


def test_fourth_transport_calibration_must_remain_finite_and_noncollapsed() -> None:
    score = _score(1701)
    quantity = TRANSPORT_QUANTITIES[-1]
    score["memberwise_transport"]["overall"]["quantities"][quantity]["reductions"][
        "separatrix_wedge"
    ]["ensemble_probabilistic_metrics"]["corrected_spread_skill"]["ratio"] = None
    record = evaluate_b2_seed_acceptance(
        result=_result(1701),
        score=score,
        training_run={
            "seed": 1701,
            "training_complete": True,
            "scientific_acceptance_evaluated": False,
        },
        comparator_run={
            "seed": 1701,
            "field": _field_comparator(),
            "transport": _transport_comparator(),
        },
        best_uncompressed={"field": _field_comparator()},
        manifest=MANIFEST,
    )
    assert record["families"]["transport"]["passes"] is False
    assert record["catastrophic_bounds"]["all_required_numeric_metrics_finite"] is False


def test_architecture_gate_allows_one_noncatastrophic_failure_but_requires_median() -> (
    None
):
    records = [_seed_record(seed) for seed in (1701, 1702, 1703)]
    records[2]["passes_complete_per_seed_gate"] = False
    accepted = evaluate_b2_architecture_acceptance(records)
    assert accepted["complete_seed_gate_pass_count"] == 2
    assert accepted["architecture_passes_one_step_B2_gate"] is True
    assert accepted["short_O3_protocol_may_be_frozen"] is True
    assert accepted["O3_launch_allowed"] is False

    name = "field.overall.aggregate_rmse_relative_to_paired_H2"
    for record in records[:2]:
        check = record["numeric_checks_for_architecture_median"][name]
        check["value"] = 2.0
        check["passes"] = False
    rejected = evaluate_b2_architecture_acceptance(records)
    assert rejected["median_numerical_gate"]["passes"] is False
    assert rejected["architecture_passes_one_step_B2_gate"] is False


def test_architecture_gate_rejects_catastrophic_remaining_seed() -> None:
    records = [_seed_record(seed) for seed in (1701, 1702, 1703)]
    records[2]["passes_complete_per_seed_gate"] = False
    records[2]["catastrophic_bounds"]["passes"] = False
    result = evaluate_b2_architecture_acceptance(records)
    assert result["nonpassing_seed_catastrophic_bounds_pass"] is False
    assert result["architecture_passes_one_step_B2_gate"] is False
