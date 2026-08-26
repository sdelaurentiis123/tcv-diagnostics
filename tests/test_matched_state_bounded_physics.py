"""Known-answer checks for matched state-view bounded physics scoring."""

from __future__ import annotations

import math

import numpy as np

from paper0.tools.score_matched_state_bounded_physics import (
    EXPECTED_PRIMARY_COUNTS,
    complex_cross_band_relative_l2,
    decide,
    e6b_common_physical,
)
from tcv_diagnostics.model_training_data import VOLUME_SHAPE


def test_e6b_common_view_uses_exact_soft_floor_without_clipping() -> None:
    state = np.ones((1, 6, *VOLUME_SHAPE), dtype=np.float32)
    state[:, 0] = 2.0
    state[:, 1] = 3.0
    state[:, 2] = 4.0
    state[:, 4] = 12.0
    phi = np.full((1, *VOLUME_SHAPE), 5.0, dtype=np.float32)
    common, diagnostics = e6b_common_physical(state, phi)
    assert common.shape == (1, 5, *VOLUME_SHAPE)
    assert float(common[0, 4, 0, 0, 0]) == 3.0
    assert diagnostics["nonpositive_density_count"] == 0

    state[:, 0, 0, 0, 0] = -1.0
    common, diagnostics = e6b_common_physical(state, phi)
    assert diagnostics["nonpositive_density_count"] == 1
    assert float(common[0, 0, 0, 0, 0]) == -1.0
    assert float(common[0, 4, 0, 0, 0]) == 12.0 / (2.0e-7)


def test_complex_cross_spectrum_error_uses_complex_band_norm() -> None:
    truth = np.asarray([0, 1 + 2j, 2 - 1j, 1j, 4, 5, 6, 7], dtype=np.complex128)
    candidate = truth.copy()
    candidate[4] += 1.0 + 2.0j
    metrics = {
        "cross_field_curves_physical": {
            "Ne-phi": {
                "truth_cross_spectrum_sum": {
                    "real": truth.real.tolist(),
                    "imag": truth.imag.tolist(),
                },
                "reconstruction_cross_spectrum_sum": {
                    "real": candidate.real.tolist(),
                    "imag": candidate.imag.tolist(),
                },
            }
        }
    }
    expected = abs(1.0 + 2.0j) / math.sqrt(abs(truth[4]) ** 2 + abs(truth[5]) ** 2)
    assert math.isclose(
        complex_cross_band_relative_l2(metrics, "Ne-phi", "k4_5"),
        expected,
    )


def _rows(e6b_factors: dict[str, float]) -> list[dict[str, object]]:
    rows = []
    for family in ("c5p", "e6b"):
        for metric, count in EXPECTED_PRIMARY_COUNTS.items():
            value = 1.0 if family == "c5p" else e6b_factors[metric]
            rows.extend(
                {
                    "family": family,
                    "metric": metric,
                    "value": value,
                }
                for _ in range(count)
            )
    return rows


def test_decision_applies_all_four_preregistered_ratios() -> None:
    passing = {
        "separatrix_transport_relative_l2": 0.90,
        "complex_cross_spectrum_relative_l2": 0.99,
        "shared_state_standardized_rmse": 1.10,
        "spectral_power_absolute_log_ratio": 1.10,
    }
    result = decide(_rows(passing), causal_phi_passed=True)
    assert result["favor_e6b_saved_state"]
    assert result["three_seed_confirmation_authorized"]

    passing["complex_cross_spectrum_relative_l2"] = 1.0
    result = decide(_rows(passing), causal_phi_passed=True)
    assert not result["favor_e6b_saved_state"]
    assert result["next_action"] == (
        "retain_c5p_control_and_stop_saved_state_branch"
    )
