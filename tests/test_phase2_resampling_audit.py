from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper0" / "tools"
AUDITOR_PATH = TOOLS / "audit_85604_resampling.py"
MERGER_PATH = TOOLS / "merge_85604_resampling_shards.py"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
AUDIT = load_module("paper0_resampling_audit", AUDITOR_PATH)
MERGE = load_module("paper0_resampling_merge", MERGER_PATH)


class ResamplingAuditImplementationTests(unittest.TestCase):
    def test_shard_intervals_are_complete_unique_and_chunk_aligned(self) -> None:
        intervals = AUDIT.SHARD_INTERVALS
        self.assertEqual(len(intervals), 17)
        self.assertEqual(intervals[0], (0, 40))
        self.assertEqual(intervals[12], (480, 500))
        self.assertEqual(intervals[13], (500, 540))
        self.assertEqual(intervals[-1], (620, 624))
        frames = [frame for first, stop in intervals for frame in range(first, stop)]
        self.assertEqual(frames, list(range(624)))

    def test_source_mapping_uses_global_half_open_intervals(self) -> None:
        sources = [
            {"global_start_inclusive": 0, "global_stop_exclusive": 500},
            {"global_start_inclusive": 500, "global_stop_exclusive": 624},
        ]
        source, local = AUDIT.source_for_frame(sources, 499)
        self.assertIs(source, sources[0])
        self.assertEqual(local, 499)
        source, local = AUDIT.source_for_frame(sources, 500)
        self.assertIs(source, sources[1])
        self.assertEqual(local, 0)
        with self.assertRaisesRegex(ValueError, "0 source matches"):
            AUDIT.source_for_frame(sources, 624)

    def test_held_out_path_guard_is_explicit(self) -> None:
        AUDIT.assert_development_path(Path("/tmp/85604/data.h5"))
        with self.assertRaisesRegex(ValueError, "held-out"):
            AUDIT.assert_development_path(Path("/tmp/85606/data.h5"))

    def test_primary_quantity_map_has_exact_energy_factors(self) -> None:
        arrays = {
            "Ne": np.asarray([1.0, 2.0]),
            "Pe": np.asarray([3.0, 4.0]),
            "Pi": np.asarray([5.0, 6.0]),
        }
        quantities = AUDIT.primary_quantity_map(arrays)
        np.testing.assert_array_equal(quantities["particle"], [1.0, 2.0])
        np.testing.assert_array_equal(
            quantities["electron_internal_energy"], [4.5, 6.0]
        )
        np.testing.assert_array_equal(
            quantities["ion_internal_energy"], [7.5, 9.0]
        )
        np.testing.assert_array_equal(
            quantities["total_internal_energy"], [12.0, 15.0]
        )

    @staticmethod
    def synthetic_transport_inputs(n_z: int = 9):
        x = np.arange(64, dtype=np.float64)[:, None, None]
        y = np.arange(32, dtype=np.float64)[None, :, None]
        z = np.arange(n_z, dtype=np.float64)[None, None, :]
        phase = 2.0 * np.pi * z / n_z
        phi = (
            0.01 * x * np.cos(phase)
            + 0.02 * y * np.sin(phase)
            + 0.0003 * x * y
        )
        base = 1.0 + 0.003 * x + 0.002 * y + 0.05 * np.cos(phase + 0.1 * x)
        fields = {
            "Ne": base,
            "Pe": 1.2 * base + 0.01 * np.sin(phase),
            "Pi": 0.9 * base - 0.02 * np.cos(phase),
            "phi": phi,
        }
        geometry = {
            "jacobian": np.ones((64, 32)),
            "dx": np.ones((64, 32)),
            "g11": np.ones((64, 32)),
            "g23": np.ones((64, 32)),
            "bxy": np.ones((64, 32)),
            "z_shift": np.zeros((64, 32)),
            "dy": np.ones((64, 32)),
        }
        return fields, geometry, np.zeros(64)

    def test_transport_bundle_has_frozen_scopes_and_nonzero_components(self) -> None:
        fields, geometry, shift_angle = self.synthetic_transport_inputs()
        bundle = AUDIT.compute_transport_bundle(fields, geometry, shift_angle)
        self.assertEqual(set(bundle), set(AUDIT.COMPARISON_CATEGORIES))
        for category, quantities in bundle.items():
            self.assertEqual(set(quantities), set(AUDIT.PRIMARY_QUANTITIES))
            expected_shape = (60, 30, 9) if category == "divergence_total" else (61, 30, 9)
            for values in quantities.values():
                self.assertEqual(values.shape, expected_shape)
                self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreater(float(np.max(np.abs(bundle["face_xz"]["particle"]))), 0.0)
        self.assertGreater(float(np.max(np.abs(bundle["face_xy"]["particle"]))), 0.0)

    def test_bundle_comparison_reports_exact_scaling(self) -> None:
        fields, geometry, shift_angle = self.synthetic_transport_inputs()
        reference = AUDIT.compute_transport_bundle(fields, geometry, shift_angle)
        candidate = {
            category: {
                quantity: 2.0 * values for quantity, values in quantities.items()
            }
            for category, quantities in reference.items()
        }
        comparison = AUDIT.compare_bundles(
            reference, candidate, resample_reference_to=None
        )
        for category in AUDIT.COMPARISON_CATEGORIES:
            for quantity in AUDIT.PRIMARY_QUANTITIES:
                leaf = comparison[category][quantity]
                self.assertAlmostEqual(float(leaf["metrics"]["relative_l2"]), 1.0)
                self.assertAlmostEqual(float(leaf["metrics"]["rms_ratio"]), 2.0)
                self.assertAlmostEqual(float(leaf["absolute_value_p99_ratio"]), 2.0)

    def test_strict_json_refuses_overwrite_and_nonfinite(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            AUDIT.strict_json_write(output, {"valid": 1.0})
            self.assertEqual(json.loads(output.read_text()), {"valid": 1.0})
            with self.assertRaises(FileExistsError):
                AUDIT.strict_json_write(output, {"valid": 2.0})
            with self.assertRaises(ValueError):
                AUDIT.strict_json_write(Path(directory) / "nan.json", {"x": np.nan})


class ResamplingMergeTests(unittest.TestCase):
    def test_scalar_summary_and_blocks_use_linear_quantiles(self) -> None:
        summary = MERGE.scalar_summary([0.0, 10.0])
        self.assertEqual(summary["median"], 5.0)
        blocks = MERGE.temporal_block_summaries(
            [0, 1, 2, 3], [1.0, 3.0, 5.0, 7.0], [[0, 1], [2, 3]]
        )
        self.assertEqual(blocks[0]["mean"], 2.0)
        self.assertEqual(blocks[1]["mean"], 6.0)
        with self.assertRaisesRegex(ValueError, "empty"):
            MERGE.scalar_summary([])

    def test_digest_records_is_order_sensitive_and_repeatable(self) -> None:
        first = [{"shard": 0, "digest": "a" * 64}]
        second = [{"shard": 1, "digest": "a" * 64}]
        self.assertEqual(MERGE.digest_records(first), MERGE.digest_records(first))
        self.assertNotEqual(MERGE.digest_records(first), MERGE.digest_records(second))


if __name__ == "__main__":
    unittest.main()
