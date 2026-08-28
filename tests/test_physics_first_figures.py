from __future__ import annotations

import numpy as np

from tcv_diagnostics.physics_first_figures import (
    BOOTSTRAP_REPLICATES,
    NePhiCrossSpectrumAccumulator,
    bootstrap_curve_mean,
    first_order_toroidal_variogram,
    lower_sample_median_target,
    moving_block_bootstrap_indices,
    standardized_toroidal_fluctuations,
)


def _score_record(target: int, error: float) -> dict:
    return {
        "target_frame": target,
        "quantities": {
            "particle": {
                "separatrix_wedge": {
                    "ensemble_mean_relative_l2": error / 2.0,
                    "truth_rms": 2.0,
                }
            }
        },
    }


def test_lower_sample_median_uses_rank_18_of_36() -> None:
    targets = tuple(range(100, 136))
    records = [_score_record(target, float(index)) for index, target in enumerate(targets)]
    selected, ordered = lower_sample_median_target(records, allowed_targets=targets)
    assert selected == targets[17]
    assert len(ordered) == 36


def test_fluctuation_preparation_removes_toroidal_mean_and_phi_gauge() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(2, 5, 64, 32, 88)).astype(np.float32)
    mask = np.zeros((64, 32), dtype=bool)
    mask[2:62, 1:31] = True
    result = standardized_toroidal_fluctuations(values, eligible_xy=mask)
    np.testing.assert_allclose(np.mean(result, axis=-1), 0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        np.mean(result[:, 3][:, mask, :], axis=(1, 2)), 0.0, atol=2.0e-7
    )


def test_known_ne_phi_mode_has_expected_physical_n_phase_and_coherence() -> None:
    mask = np.zeros((64, 32), dtype=bool)
    mask[10, 10] = True
    coordinate = 2.0 * np.pi * np.arange(88) / 88.0
    phase = np.deg2rad(37.0)
    values = np.zeros((3, 5, 64, 32, 88), dtype=np.float32)
    values[:, 0, 10, 10] = np.cos(4.0 * coordinate)
    values[:, 3, 10, 10] = np.cos(4.0 * coordinate - phase)
    accumulator = NePhiCrossSpectrumAccumulator(mask)
    accumulator.update(values)
    result = accumulator.finalize()
    assert int(result["physical_n"][4]) == 20
    assert float(result["coherence"][4]) > 1.0 - 1.0e-12
    np.testing.assert_allclose(float(result["phase_degrees"][4]), phase * 180 / np.pi, atol=2.0e-5)


def test_known_periodic_transport_mode_variogram() -> None:
    coordinate = 2.0 * np.pi * np.arange(81) / 81.0
    values = np.broadcast_to(np.cos(3.0 * coordinate), (2, 16, 81)).copy()
    result = first_order_toroidal_variogram(values, lags=(1, 2, 4))
    expected = [
        np.mean(np.abs(values[0] - np.roll(values[0], -lag, axis=-1)))
        for lag in (1, 2, 4)
    ]
    np.testing.assert_allclose(result[0], expected, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_array_equal(result[0], result[1])


def test_moving_block_bootstrap_is_reproducible_and_constant_exact() -> None:
    first = moving_block_bootstrap_indices(36)
    second = moving_block_bootstrap_indices(36)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (BOOTSTRAP_REPLICATES, 36)
    curves = np.full((36, 7), 2.5)
    interval = bootstrap_curve_mean(curves, first)
    for value in interval.values():
        np.testing.assert_array_equal(value, np.full(7, 2.5))

