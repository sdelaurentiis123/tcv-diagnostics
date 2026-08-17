from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.metrics import (  # noqa: E402
    apply_memberwise,
    central_interval_coverage,
    cross_spectral_metrics,
    fair_crps,
    one_sided_power_spectrum,
    ordinary_crps,
    spread_skill_summary,
    toroidal_mode_numbers,
    validate_canonical_forecast,
)


class CanonicalAxisTests(unittest.TestCase):
    def test_canonical_forecast_and_truth_shapes_are_enforced(self) -> None:
        forecast = np.zeros((2, 3, 4, 5, 6, 7, 8), dtype=np.float64)
        truth = np.zeros((2, 4, 5, 6, 7, 8), dtype=np.float64)
        checked_forecast, checked_truth = validate_canonical_forecast(
            forecast, truth
        )
        self.assertEqual(checked_forecast.shape, forecast.shape)
        self.assertEqual(checked_truth.shape, truth.shape)

        with self.assertRaisesRegex(ValueError, r"\[B, M, T, C, X, Y, Z\]"):
            validate_canonical_forecast(forecast[:, 0], truth)
        with self.assertRaisesRegex(ValueError, "non-member axes disagree"):
            validate_canonical_forecast(forecast, truth[:, :-1])
        contaminated = forecast.copy()
        contaminated[0, 0, 0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_canonical_forecast(contaminated, truth)


class ToroidalSpectrumTests(unittest.TestCase):
    def test_mode_mapping_uses_zperiod_five(self) -> None:
        stored_k, full_torus_n = toroidal_mode_numbers(88, zperiod=5)
        np.testing.assert_array_equal(stored_k[:8], np.arange(8))
        np.testing.assert_array_equal(full_torus_n[4:8], [20, 25, 30, 35])

    def test_known_modes_and_parseval_for_even_and_odd_grids(self) -> None:
        for n_z in (88, 87):
            z = np.arange(n_z, dtype=np.float64)
            signal = (
                0.2
                + 1.5 * np.cos(2.0 * np.pi * 4.0 * z / n_z)
                + 0.25 * np.sin(2.0 * np.pi * 7.0 * z / n_z)
            )
            power = one_sided_power_spectrum(signal)
            self.assertEqual(int(np.argmax(power[1:]) + 1), 4)
            self.assertAlmostEqual(
                float(np.sum(power)), float(np.mean(signal**2)), places=13
            )

            fluctuation_power = one_sided_power_spectrum(
                signal, remove_mean=True
            )
            self.assertAlmostEqual(float(fluctuation_power[0]), 0.0, places=28)
            self.assertAlmostEqual(
                float(np.sum(fluctuation_power)),
                float(np.mean((signal - np.mean(signal)) ** 2)),
                places=13,
            )

    def test_cross_spectrum_recovers_signed_phase_and_unit_coherence(self) -> None:
        n_z = 88
        mode = 5
        delta = 0.7
        z = np.arange(n_z, dtype=np.float64)
        sample_phases = np.asarray([0.1, -0.8, 1.4, 2.1])
        a = np.stack(
            [np.cos(2.0 * np.pi * mode * z / n_z + phase) for phase in sample_phases]
        )
        b = np.stack(
            [
                np.cos(2.0 * np.pi * mode * z / n_z + phase + delta)
                for phase in sample_phases
            ]
        )
        result = cross_spectral_metrics(a, b, sample_axes=(0,), zperiod=5)
        self.assertAlmostEqual(float(result["coherence"][mode]), 1.0, places=13)
        self.assertAlmostEqual(
            float(result["phase_radians"][mode]), -delta, places=13
        )
        self.assertEqual(int(result["full_torus_n"][mode]), 25)

        zero_result = cross_spectral_metrics(
            a, np.zeros_like(a), sample_axes=(0,), zperiod=5
        )
        self.assertTrue(np.all(np.isnan(zero_result["coherence"])))


class EnsembleScoreTests(unittest.TestCase):
    def test_ordinary_and_fair_crps_match_hand_calculation(self) -> None:
        forecast = np.asarray([[[0.0], [1.0], [2.0]]])
        truth = np.asarray([[1.0]])
        ordinary = ordinary_crps(forecast, truth, member_axis=1)
        fair = fair_crps(forecast, truth, member_axis=1)
        self.assertAlmostEqual(float(ordinary[0, 0]), 2.0 / 9.0, places=14)
        self.assertAlmostEqual(float(fair[0, 0]), 0.0, places=14)

    def test_sorted_crps_matches_direct_pairwise_definition(self) -> None:
        rng = np.random.default_rng(20260817)
        for members in (2, 3, 8, 16):
            forecast = rng.normal(size=(3, members, 5))
            truth = rng.normal(size=(3, 5))
            observation_term = np.mean(
                np.abs(forecast - truth[:, None, :]), axis=1
            )
            pair_sum = np.sum(
                np.abs(forecast[:, :, None, :] - forecast[:, None, :, :]),
                axis=(1, 2),
            )
            expected_ordinary = observation_term - pair_sum / (
                2.0 * members * members
            )
            expected_fair = observation_term - pair_sum / (
                2.0 * members * (members - 1)
            )
            np.testing.assert_allclose(
                ordinary_crps(forecast, truth),
                expected_ordinary,
                rtol=1e-13,
                atol=1e-13,
            )
            np.testing.assert_allclose(
                fair_crps(forecast, truth),
                expected_fair,
                rtol=1e-13,
                atol=1e-13,
            )

    def test_one_member_crps_is_absolute_error_and_fair_form_rejects_it(self) -> None:
        forecast = np.asarray([[[3.0, -2.0]]])
        truth = np.asarray([[1.0, 1.0]])
        np.testing.assert_allclose(
            ordinary_crps(forecast, truth), np.asarray([[2.0, 3.0]])
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            fair_crps(forecast, truth)

    def test_coverage_and_spread_skill_have_known_values(self) -> None:
        member_values = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
        forecast = np.broadcast_to(member_values[None, :, None], (1, 5, 4))
        truth = np.asarray([[-1.0, 0.0, 1.0, 3.0]])
        coverage = central_interval_coverage(
            forecast, truth, nominal_coverage=0.8
        )
        self.assertAlmostEqual(coverage["empirical_coverage"], 0.75, places=14)
        self.assertAlmostEqual(coverage["mean_interval_width"], 3.2, places=14)

        skill_forecast = np.asarray([[[-1.0, -1.0], [1.0, 1.0]]])
        skill_truth = np.asarray([[1.0, -1.0]])
        summary = spread_skill_summary(skill_forecast, skill_truth)
        self.assertAlmostEqual(summary["rms_spread"], 1.0, places=14)
        self.assertAlmostEqual(summary["rmse_of_ensemble_mean"], 1.0, places=14)
        self.assertAlmostEqual(summary["spread_skill_ratio"], 1.0, places=14)

    def test_nonlinear_diagnostic_is_applied_before_member_reduction(self) -> None:
        a = np.asarray([1.0, -1.0])[None, :, None, None, None, None]
        b = np.asarray([1.0, -1.0])[None, :, None, None, None, None]
        products = apply_memberwise(lambda left, right: left * right, a, b)
        self.assertEqual(products.shape, a.shape)
        self.assertAlmostEqual(float(np.mean(products, axis=1).item()), 1.0)
        product_of_means = np.mean(a, axis=1) * np.mean(b, axis=1)
        self.assertAlmostEqual(float(product_of_means.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
