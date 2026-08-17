from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.signal import resample

from tcv_diagnostics.resampling import (
    finalize_paired_statistics,
    linear_quantile,
    materiality_label,
    merge_paired_sufficient_statistics,
    paired_frame_metrics,
    paired_sufficient_statistics,
    periodic_resample_float32,
    relative_l2,
)


class PeriodicResamplingTests(unittest.TestCase):
    def test_known_native_modes_preserve_amplitude_phase_and_padding(self) -> None:
        native_n = 81
        target_n = 88
        index = np.arange(native_n, dtype=np.float64)
        signal = (
            1.7
            + 0.8 * np.cos(2.0 * math.pi * 7 * index / native_n + 0.31)
            - 0.2 * np.sin(2.0 * math.pi * 40 * index / native_n - 0.27)
        ).astype(np.float32)
        got = periodic_resample_float32(signal, target_n)
        target_index = np.arange(target_n, dtype=np.float64)
        expected = (
            1.7
            + 0.8 * np.cos(2.0 * math.pi * 7 * target_index / target_n + 0.31)
            - 0.2 * np.sin(2.0 * math.pi * 40 * target_index / target_n - 0.27)
        )
        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)
        normalized_spectrum = np.fft.rfft(got.astype(np.float64)) / target_n
        self.assertLess(float(np.max(np.abs(normalized_spectrum[41:]))), 2e-8)

    def test_wrapper_is_bitwise_scipy_call_and_round_trip_is_small(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=(3, 4, 81)).astype(np.float32)
        expected = resample(
            values, 88, axis=-1, window=None, domain="time"
        ).astype(np.float32)
        got = periodic_resample_float32(values, 88)
        np.testing.assert_array_equal(got, expected)
        round_trip = periodic_resample_float32(got, 81)
        self.assertLess(relative_l2(values, round_trip), 2e-6)

    def test_invalid_resampling_inputs_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "real-valued"):
            periodic_resample_float32(np.asarray([1.0 + 1.0j, 2.0]), 3)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            periodic_resample_float32(np.asarray([1.0, np.nan]), 3)
        with self.assertRaisesRegex(ValueError, "target_samples"):
            periodic_resample_float32(np.ones(3), 1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            periodic_resample_float32(np.ones((2, 1)), 3)


class PairedStatisticsTests(unittest.TestCase):
    def test_known_metrics_and_weighted_sign_disagreement(self) -> None:
        reference = np.asarray([1.0, -2.0, 3.0, -4.0])
        candidate = np.asarray([2.0, 2.0, 3.0, -2.0])
        sufficient = paired_sufficient_statistics(reference, candidate)
        metrics = finalize_paired_statistics(sufficient)
        expected_relative = math.sqrt((1.0 + 16.0 + 0.0 + 4.0) / 30.0)
        self.assertAlmostEqual(float(metrics["relative_l2"]), expected_relative)
        self.assertAlmostEqual(
            float(metrics["weighted_sign_disagreement"]), 2.0 / 10.0
        )
        self.assertEqual(metrics["point_count"], 4)
        self.assertTrue(metrics["pearson_correlation_defined"])

    def test_disjoint_merge_matches_single_pass(self) -> None:
        rng = np.random.default_rng(11)
        reference = rng.normal(size=(6, 5))
        candidate = reference + rng.normal(scale=0.2, size=reference.shape)
        whole = paired_sufficient_statistics(reference, candidate)
        merged = merge_paired_sufficient_statistics(
            [
                paired_sufficient_statistics(reference[:2], candidate[:2]),
                paired_sufficient_statistics(reference[2:], candidate[2:]),
            ]
        )
        for key in whole:
            self.assertAlmostEqual(float(merged[key]), float(whole[key]), places=12)
        whole_metrics = finalize_paired_statistics(whole)
        merged_metrics = finalize_paired_statistics(merged)
        for key in (
            "relative_l2",
            "normalized_bias",
            "rms_ratio",
            "pearson_correlation",
            "weighted_sign_disagreement",
        ):
            self.assertAlmostEqual(
                float(merged_metrics[key]), float(whole_metrics[key]), places=12
            )

    def test_frame_metrics_include_profile_and_linear_tail_ratios(self) -> None:
        reference = np.arange(1.0, 25.0).reshape(2, 3, 4)
        candidate = 2.0 * reference
        result = paired_frame_metrics(reference, candidate)
        self.assertAlmostEqual(float(result["metrics"]["relative_l2"]), 1.0)
        self.assertAlmostEqual(
            float(result["toroidal_mean_profile_relative_l2"]), 1.0
        )
        self.assertAlmostEqual(float(result["absolute_value_p95_ratio"]), 2.0)
        self.assertAlmostEqual(float(result["absolute_value_p99_ratio"]), 2.0)
        self.assertEqual(result["metrics"]["weighted_sign_disagreement"], 0.0)

    def test_degenerate_or_invalid_statistics_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero L2"):
            relative_l2(np.zeros(3), np.ones(3))
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            paired_sufficient_statistics(np.ones(2), np.ones(3))
        with self.assertRaisesRegex(ValueError, "empty"):
            merge_paired_sufficient_statistics([])

    def test_materiality_boundaries_and_quantile_convention(self) -> None:
        self.assertEqual(materiality_label(0.009999), "negligible")
        self.assertEqual(materiality_label(0.01), "small")
        self.assertEqual(materiality_label(0.05), "material")
        self.assertEqual(materiality_label(0.10), "severe")
        self.assertAlmostEqual(linear_quantile([0.0, 10.0], 0.25), 2.5)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            materiality_label(-0.1)


if __name__ == "__main__":
    unittest.main()
