from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.data_protocol import (  # noqa: E402
    C5_FIELDS,
    DEFAULT_SPLIT,
    RunningMoments,
    first_threshold_crossing,
    inverse_model_transform,
    inverse_standardize,
    model_transform,
    operational_steady_screen,
    path_is_allowed,
    pattern_autocorrelation,
    standardize,
    summarize_autocorrelation,
    summarize_stationarity_series,
    toroidal_variability_decomposition,
)
from tcv_diagnostics.well import VirtualWellTrajectory  # noqa: E402


def write_shard(path: Path, start: int, frames: int) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        dimensions = handle.create_group("dimensions")
        dimensions.create_dataset(
            "time", data=np.arange(start, start + frames, dtype=np.float64) * 300.0
        )
        dimensions.create_dataset("x", data=np.arange(2))
        dimensions.create_dataset("y", data=np.arange(3))
        dimensions.create_dataset("z", data=np.arange(4))
        fields = handle.create_group("t0_fields")
        fields.attrs.create(
            "field_names", np.asarray(C5_FIELDS, dtype=object), dtype=string_dtype
        )
        for field_index, field in enumerate(C5_FIELDS):
            values = np.empty((1, frames, 2, 3, 4), dtype=np.float32)
            for local_time in range(frames):
                values[0, local_time] = field_index * 100 + start + local_time
            if field == "Ne":
                values += 1.0
            fields.create_dataset(field, data=values)


class SplitAndTransformTests(unittest.TestCase):
    def test_default_split_has_strict_guard_and_contained_windows(self) -> None:
        self.assertGreater(
            DEFAULT_SPLIT.guard.frames, DEFAULT_SPLIT.max_window_frames
        )
        starts = DEFAULT_SPLIT.train.window_starts(32)
        self.assertEqual(starts.start, 0)
        self.assertEqual(starts.stop - 1, 400)
        for start in starts:
            self.assertTrue(DEFAULT_SPLIT.train.contains_window(start, 32))
            self.assertLessEqual(start + 32, DEFAULT_SPLIT.guard.start)
        validation_starts = DEFAULT_SPLIT.validation.window_starts(32)
        self.assertEqual(validation_starts.start, 496)

    def test_density_and_standardization_round_trips(self) -> None:
        density = np.asarray([0.01, 0.2, 3.0], dtype=np.float64)
        transformed = model_transform("Ne", density)
        restored = inverse_model_transform("Ne", transformed)
        np.testing.assert_allclose(restored, density, rtol=1e-12, atol=1e-12)

        standardized = standardize(transformed, mean=-1.0, std=2.5)
        unstandardized = inverse_standardize(standardized, mean=-1.0, std=2.5)
        np.testing.assert_allclose(unstandardized, transformed, rtol=1e-12)

    def test_sequestered_paths_are_rejected(self) -> None:
        self.assertFalse(path_is_allowed(Path("/tmp/85606/fields.h5")))
        self.assertFalse(path_is_allowed(Path("/tmp/test/fields.h5")))
        self.assertTrue(path_is_allowed(Path("/tmp/85604/validation/fields.h5")))


class VirtualTrajectoryTests(unittest.TestCase):
    def test_shards_are_concatenated_in_time_and_cross_boundary_reads_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "85604_train_storage.h5"
            second = root / "85604_valid_storage.h5"
            write_shard(first, start=0, frames=3)
            write_shard(second, start=3, frames=2)
            trajectory = VirtualWellTrajectory([first, second])

            self.assertEqual(trajectory.total_frames, 5)
            self.assertEqual(trajectory.spatial_shape, (2, 3, 4))
            np.testing.assert_array_equal(
                trajectory.time, np.asarray([0, 300, 600, 900, 1200])
            )
            values = trajectory.read_field("Te", 2, 5, chunk_frames=1)
            self.assertEqual(values.shape, (3, 2, 3, 4))
            self.assertTrue(np.all(values[0] == 102))
            self.assertTrue(np.all(values[1] == 103))
            self.assertTrue(np.all(values[2] == 104))


class StatisticsTests(unittest.TestCase):
    def test_running_moments_match_numpy(self) -> None:
        values = np.linspace(-3.0, 7.0, 101, dtype=np.float64)
        moments = RunningMoments()
        moments.update(values[:17])
        moments.update(values[17:83])
        moments.update(values[83:])
        result = moments.finalize()
        self.assertEqual(result["count"], values.size)
        self.assertAlmostEqual(result["mean"], float(np.mean(values)), places=14)
        self.assertAlmostEqual(result["std"], float(np.std(values)), places=14)

    def test_frozen_stationarity_rules_pass_stationary_and_reject_drift(self) -> None:
        stationary = np.tile(np.asarray([-1.0, 0.0, 1.0, 0.0]), 156)
        self.assertTrue(summarize_stationarity_series(stationary)["passes"])
        drift = np.linspace(0.0, 10.0, 624)
        result = summarize_stationarity_series(drift)
        self.assertFalse(result["passes"])
        self.assertFalse(result["criteria_pass"]["absolute_normalized_drift"])

        means = {field: stationary.copy() for field in C5_FIELDS}
        rms = {field: np.ones(624) for field in C5_FIELDS}
        screen = operational_steady_screen(means, rms)
        self.assertTrue(screen["passes"])
        self.assertNotIn("phi.spatial_mean", screen["series"])

    def test_autocorrelation_shape_and_crossing_are_explicit(self) -> None:
        rng = np.random.default_rng(7)
        frames = rng.normal(size=(64, 4, 3))
        curve = pattern_autocorrelation(frames, max_lag=8)
        self.assertEqual(curve.shape, (9,))
        self.assertAlmostEqual(float(curve[0]), 1.0, places=12)

        known = np.asarray([1.0, 0.6, 0.2, -0.1])
        crossing = first_threshold_crossing(known, np.exp(-1.0))
        self.assertIsNotNone(crossing)
        self.assertGreater(crossing, 1.0)
        self.assertLess(crossing, 2.0)
        summary = summarize_autocorrelation(known, cadence_microseconds=3.0)
        self.assertFalse(summary["one_over_e_right_censored"])
        self.assertFalse(summary["first_nonpositive_right_censored"])
        self.assertAlmostEqual(
            summary["one_over_e_crossing_microseconds"], crossing * 3.0
        )

    def test_toroidal_decomposition_removes_k_zero_and_partitions_energy(self) -> None:
        time = np.arange(6, dtype=np.float64)[:, None, None]
        axisymmetric = time * np.ones((1, 2, 4), dtype=np.float64)
        alternating = np.asarray([1.0, -1.0, 1.0, -1.0])[None, None, :]
        nonaxisymmetric = (
            0.5 * time * alternating * np.ones((1, 2, 1), dtype=np.float64)
        )
        frames = axisymmetric + nonaxisymmetric

        residual, energy = toroidal_variability_decomposition(frames)
        np.testing.assert_allclose(np.mean(residual, axis=-1), 0.0, atol=1e-14)
        np.testing.assert_allclose(residual, nonaxisymmetric, atol=1e-14)
        self.assertAlmostEqual(
            energy["axisymmetric_fraction"]
            + energy["nonaxisymmetric_fraction"],
            1.0,
            places=14,
        )
        self.assertAlmostEqual(energy["orthogonality_cross_term"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
