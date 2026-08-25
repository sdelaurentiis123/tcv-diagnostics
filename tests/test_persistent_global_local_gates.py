from __future__ import annotations

import copy

from tcv_diagnostics.b2_field_metrics import B2_FIELDS
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.persistent_global_local_gates import (
    PGL_SPECTRAL_BANDS,
    evaluate_pgl_physics_gates,
)


def _field(*, fair: float, mae: float, spread: float):
    fields = {
        name: {"corrected_spread_skill": {"ratio": spread}} for name in B2_FIELDS
    }
    return {
        "regions": {
            "eligible_union": {
                "aggregate": {
                    "equal_channel_fair_crps": fair,
                    "equal_channel_ensemble_mean_mae": mae,
                },
                "fields": fields,
            }
        }
    }


def _spectral(*, power_ratio: float, cross_scale: float, phase: float = 0.0):
    field_power = {
        field: {
            "bands": {
                band: {"member_expected_power_ratio": power_ratio}
                for band in PGL_SPECTRAL_BANDS
            }
        }
        for field in B2_FIELDS
    }
    truth = {"real": [0.0] + [1.0] * 7, "imag": [0.0] * 8}
    forecast = {
        "real": [0.0] + [cross_scale] * 7,
        "imag": [0.0] + [phase] * 7,
    }
    return {
        "toroidal_field_power": field_power,
        "toroidal_cross_field": {
            "Ne-phi": {
                "curves": {
                    "truth_cross_spectrum": truth,
                    "member_expected_cross_spectrum": forecast,
                    "truth_coherence": [0.0] + [0.8] * 7,
                    "member_expected_coherence": [0.0] + [0.78] * 7,
                }
            }
        },
    }


def _transport(relative_l2: float):
    return {
        "quantities": {
            quantity: {
                "reductions": {
                    "separatrix_wedge": {
                        "ensemble_expected_paired_metrics": {
                            "relative_l2": relative_l2
                        }
                    }
                }
            }
            for quantity in TRANSPORT_QUANTITIES
        }
    }


def _covariance(local: float, integrated: float):
    return {
        "quantities": {
            quantity: {
                "covariance_decomposition": {
                    "local_corrected_spread_skill_ratio": local,
                    "integrated_corrected_spread_skill_ratio": integrated,
                }
            }
            for quantity in TRANSPORT_QUANTITIES
        }
    }


def _sketch(error: float):
    return {
        "quantities": {
            quantity: {"relative_frobenius_error_sketch": error}
            for quantity in TRANSPORT_QUANTITIES
        }
    }


def _passing_kwargs():
    return {
        "candidate_h1_field": _field(fair=0.6, mae=0.0, spread=1.0),
        "candidate_h4_field": _field(fair=0.7, mae=0.0, spread=1.1),
        "selected_h1_field": _field(fair=0.0, mae=0.8, spread=0.0),
        "selected_h4_field": _field(fair=0.0, mae=0.9, spread=0.0),
        "candidate_h4_spectral": _spectral(power_ratio=1.05, cross_scale=0.95),
        "parent_h4_spectral": _spectral(power_ratio=1.10, cross_scale=0.80),
        "candidate_h4_transport": _transport(0.50),
        "parent_h4_transport": _transport(0.55),
        "candidate_h4_covariance": _covariance(1.0, 0.75),
        "candidate_h4_spatial_sketch": _sketch(0.7),
    }


def test_all_seven_frozen_gate_families_pass_without_compensation():
    result = evaluate_pgl_physics_gates(**_passing_kwargs())
    assert result["passed"] is True
    assert len(result["family_pass"]) == 7
    assert all(result["family_pass"].values())
    assert result["spectral_retention"]["candidate_over_parent"] < 1.0
    assert result["cross_field"]["complex_error_strictly_improves"] is True


def test_one_failed_family_fails_the_complete_pilot():
    inputs = _passing_kwargs()
    inputs["candidate_h4_spatial_sketch"] = _sketch(0.91)
    result = evaluate_pgl_physics_gates(**inputs)
    assert result["spatial_transport_covariance"]["passes"] is False
    assert result["passed"] is False
    inputs = copy.deepcopy(_passing_kwargs())
    inputs["candidate_h1_field"]["regions"]["eligible_union"]["fields"]["Ne"][
        "corrected_spread_skill"
    ]["ratio"] = 1.51
    assert evaluate_pgl_physics_gates(**inputs)["passed"] is False
