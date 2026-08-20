"""Formula and scope locks for the prospective ECRD acceptance reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tcv_diagnostics.b2_field_metrics import B2_FIELDS
from tcv_diagnostics.b5_covariance_localization import B5_FINITE_MEMBER_FACTOR
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES
from tcv_diagnostics.ecrd_acceptance import (
    ECRD_BOOTSTRAP_BLOCK_LENGTH,
    ECRD_BOOTSTRAP_REPLICATES,
    ECRD_BOOTSTRAP_SEED,
    ECRD_MATERIAL_POWER_RANGE,
    _cross_summary,
    _integrated_ratios,
    _material_power_summary,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "paper0/protocol/ECRD_EVALUATION_IMPLEMENTATION_FREEZE.md"


def _curve(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "real": np.real(values).tolist(),
        "imag": np.imag(values).tolist(),
    }


def _spectral_score(seed: int, forecast_scale: float = 0.9) -> dict:
    truth_cross = np.ones(45, dtype=np.complex128)
    forecast_cross = forecast_scale * truth_cross
    truth_power = np.full(45, 2.0)
    forecast_power = np.full(45, 2.0)
    root = {
        "toroidal_cross_field": {
            "Ne-phi": {
                "curves": {
                    "truth_cross_spectrum": _curve(truth_cross),
                    "member_expected_cross_spectrum": _curve(forecast_cross),
                }
            }
        },
        "toroidal_field_power": {
            field: {
                "curves": {
                    "truth_power": truth_power.tolist(),
                    "member_expected_power": forecast_power.tolist(),
                },
                "bands": {
                    label: {"member_expected_power_ratio": 1.0}
                    for label in ("k1_3", "k4_5", "k6_7")
                },
            }
            for field in B2_FIELDS
        },
    }
    return {
        "model_seed": seed,
        "spectral_and_cross_field": {"overall": root},
    }


def test_cross_summary_uses_complex_spectrum_and_truth_amplitude_weights() -> None:
    records = {seed: _spectral_score(seed) for seed in (1701, 1702, 1703)}
    result = _cross_summary(records, None)
    assert np.isclose(result["complex_cross_spectrum_relative_L1_error"], 0.1)
    assert np.isclose(
        result["truth_amplitude_weighted_absolute_coherence_error"],
        0.25 - 0.9**2 / 4.0,
    )
    assert result["truth_amplitude_weighted_absolute_phase_error_degrees"] == 0.0


def test_integrated_spread_skill_pools_seed_target_moments_before_ratio() -> None:
    standard_deviation = 1.0 / np.sqrt(B5_FINITE_MEMBER_FACTOR)
    records = {}
    for seed in (1701, 1702, 1703):
        quantities = {}
        for name in TRANSPORT_QUANTITIES:
            quantities[name] = {
                "separatrix_time_series": {
                    "target_frame": list(range(498, 624)),
                    "truth": [0.0] * 126,
                    "ensemble_expected": [1.0] * 126,
                    "member_standard_deviation_ddof1": [
                        standard_deviation
                    ]
                    * 126,
                }
            }
        records[seed] = {
            "model_seed": seed,
            "memberwise_transport": {"overall": {"quantities": quantities}},
        }
    ratios = _integrated_ratios(records)
    assert all(np.isclose(value, 1.0) for value in ratios.values())


def test_material_power_summary_uses_training_material_flags_only() -> None:
    records = {seed: _spectral_score(seed, 1.0) for seed in (1701, 1702, 1703)}
    materiality = {
        "fields": {
            field: {
                "bands": {
                    "k1_3": {"material": field == "Ne"},
                    "k4_5": {"material": False},
                    "k6_7": {"material": False},
                }
            }
            for field in B2_FIELDS
        }
    }
    result = _material_power_summary(records, materiality)
    assert result["material_check_count"] == 1
    assert result["passing_count"] == 1
    assert result["median_absolute_log_power_ratio_error"] == 0.0


def test_ecrd_evaluation_freeze_records_exact_uncertainty_and_power_rules() -> None:
    text = FREEZE.read_text()
    compact = " ".join(text.split())
    assert "Simulation 85606 remains unopened" in text
    assert "64 normalized Rademacher probes" in compact
    assert "PCG64 seed 85604350" in compact
    assert "noncircular blocks of 12 consecutive frames" in compact
    assert "2,000 replicates" in compact
    assert "PCG64 seed 85604351" in compact
    assert "[0.75,1.30]" in text
    assert ECRD_BOOTSTRAP_BLOCK_LENGTH == 12
    assert ECRD_BOOTSTRAP_REPLICATES == 2_000
    assert ECRD_BOOTSTRAP_SEED == 85_604_351
    assert ECRD_MATERIAL_POWER_RANGE == (0.75, 1.30)
