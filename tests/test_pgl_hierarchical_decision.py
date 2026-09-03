from __future__ import annotations

from copy import deepcopy

import pytest

from tcv_diagnostics.pgl_hierarchical_decision import evaluate_two_epoch_decision


def _gate(*, integrated: float, covariance: float, stable: bool = True) -> dict:
    family = {
        "cross_field": stable,
        "field_distribution": stable,
        "integrated_transport_calibration": integrated >= 0.6,
        "integrated_transport_mean": stable,
        "local_transport_calibration": True,
        "spatial_transport_covariance": covariance < 0.9,
        "spectral_retention": stable,
    }
    return {
        "passed": all(family.values()),
        "family_pass": family,
        "integrated_transport_calibration": {
            "median_corrected_spread_skill": integrated
        },
        "spatial_transport_covariance": {
            "median_relative_frobenius_error": covariance
        },
        "local_transport_calibration": {
            "corrected_spread_skill_by_quantity": {
                "particle": 0.9,
                "electron_internal_energy": 0.8,
                "ion_internal_energy": 1.0,
                "total_internal_energy": 0.95,
            }
        },
        "integrated_transport_mean": {"candidate_median_relative_L2": 0.3},
        "spectral_retention": {
            "candidate_median_absolute_log_power_ratio_error": 0.25
        },
        "cross_field": {
            "candidate": {
                "normalized_complex_cross_spectrum_error_k1_7": 0.2,
                "truth_amplitude_weighted_absolute_phase_error_degrees_k1_7": 2.0,
            }
        },
        "field_distribution": {
            "h4": {"candidate_equal_field_fair_CRPS": 0.04}
        },
    }


def _records() -> dict:
    result = {}
    for arm in ("CONTROL", "TRANSPORT"):
        for update in (107, 214, 428):
            result[(arm, update)] = {"gate": _gate(integrated=0.3, covariance=0.98)}
    return result


def test_extension_requires_both_material_deltas_and_stability() -> None:
    records = _records()
    records[("TRANSPORT", 428)] = {
        "gate": _gate(integrated=0.36, covariance=0.96)
    }
    decision = evaluate_two_epoch_decision(records)
    assert decision["longer_4_to_8_epoch_extension_authorized"] is True
    assert decision["confirmation_seeds_authorized"] is False
    assert decision["next_action"] == "write_dated_longer_duration_amendment"

    unstable = deepcopy(records)
    unstable[("TRANSPORT", 428)] = {
        "gate": _gate(integrated=0.36, covariance=0.96, stable=False)
    }
    assert not evaluate_two_epoch_decision(unstable)[
        "longer_4_to_8_epoch_extension_authorized"
    ]


def test_missing_checkpoint_blocks_decision() -> None:
    records = _records()
    records.pop(("CONTROL", 107))
    with pytest.raises(ValueError, match="six"):
        evaluate_two_epoch_decision(records)


def test_only_matched_reduction_authorizes_confirmation_seeds() -> None:
    records = _records()
    records[("TRANSPORT", 428)] = {
        "gate": _gate(integrated=0.7, covariance=0.8)
    }
    decision = evaluate_two_epoch_decision(records)
    assert decision["transport_arm_passed_production_gate"] is True
    assert decision["confirmation_seeds_authorized"] is True
    assert decision["next_action"] == "advance_to_confirmation_seeds_without_extending_duration"
