from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper0.tools.compare_potential_vorticity_all_frame_shard import (
    sufficient_statistics,
)
from paper0.tools.compare_potential_vorticity_forward_oracle import (
    mode_residual_summary,
)
from paper0.tools.extract_potential_vorticity_all_frame_85604 import (
    EXPECTED_SHARDS,
    create_canonical_files,
    discrepancy_count,
    expected_times,
    update_pressure_inventory,
)
from paper0.tools.merge_potential_vorticity_all_frame_shards import (
    merge_mode_power,
    merge_sufficient_statistics,
    metrics_from_sufficient,
    write_strict_json,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTED_DRIVER = (
    ROOT / "paper0/oracles/potential_vorticity_forward/potential_vorticity_forward_oracle.cxx"
)
DRIVER = (
    ROOT
    / "paper0/oracles/potential_vorticity_all_frame"
    / "potential_vorticity_all_frame_oracle.cxx"
)
CMAKE = ROOT / "paper0/oracles/potential_vorticity_all_frame/CMakeLists.txt"
EXTRACTOR = ROOT / "paper0/tools/extract_potential_vorticity_all_frame_85604.py"
COMPARATOR = ROOT / "paper0/tools/compare_potential_vorticity_all_frame_shard.py"
MERGER = ROOT / "paper0/tools/merge_potential_vorticity_all_frame_shards.py"
SELECTED_DRIVER_SHA256 = (
    "516527bea146d2ccc258e210e4058f233ecb66ca030fa48298030224919ac487"
)
FILE_HASHES = {
    DRIVER: "79daf7925cb6a8b7d8751eee51f3fa9f5e6139289700654999d659a8bce6d254",
    CMAKE: "b2ac21ea37793e24417320b7fef8c143f0db6a1ff80f65a0e6663328f019169e",
    EXTRACTOR: "5840704805b04c8586b1835064a5ae34b62ad22a95257722e6a416db5e2c191a",
    COMPARATOR: "ffd5efbb59abb0eef3bd9a187431f85d5a9c9700fd976627c1d9b89c8b64e967",
    MERGER: "3bedce322efbf22e4942afc84101422f807b179e82c9fa739f1c6800ab4e6fa6",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AllFrameExtractionTests(unittest.TestCase):
    def test_canonical_chunk_matches_rank_write_slab_and_round_trips(self) -> None:
        try:
            import netCDF4
        except ImportError:
            self.skipTest("netCDF4 is unavailable")
        manifest = {
            "canonical_extraction": {
                "volume_axes": ["frame", "x", "y", "z"],
                "boundary_axes": ["frame", "side", "y"],
            },
            "frame_scope": {"physical_cadence_microseconds": 3.131905426352636},
        }
        times = 285000.0 + 300.0 * np.arange(624, dtype=np.float64)
        block = np.arange(78 * 4 * 2 * 81, dtype=np.float64).reshape(
            78, 4, 2, 81
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with ExitStack() as stack:
                _, variables = create_canonical_files(
                    stack,
                    netcdf4=netCDF4,
                    output_dir=output_dir,
                    manifest=manifest,
                    times=times,
                )
                variables[0]["Ne"][:, 0:4, 0:2, :] = block
            with netCDF4.Dataset(output_dir / "canonical_shard_0.nc", "r") as data:
                self.assertEqual(data.variables["Ne"].chunking(), [78, 4, 2, 81])
                np.testing.assert_array_equal(
                    np.asarray(data.variables["Ne"][:, 0:4, 0:2, :]), block
                )

    def test_shard_partition_is_value_independent_and_complete(self) -> None:
        self.assertEqual(
            EXPECTED_SHARDS,
            (
                (0, 78),
                (78, 156),
                (156, 234),
                (234, 312),
                (312, 390),
                (390, 468),
                (468, 546),
                (546, 624),
            ),
        )
        covered = [
            frame for start, stop in EXPECTED_SHARDS for frame in range(start, stop)
        ]
        self.assertEqual(covered, list(range(624)))

    def test_expected_time_sequence_matches_frozen_endpoints(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "paper0/manifests/phase2_potential_vorticity_all_frame_85604.json"
            ).read_text(encoding="utf-8")
        )
        times = expected_times(manifest["frame_scope"])
        self.assertEqual(times.shape, (624,))
        self.assertEqual(times[0], 285000.0)
        self.assertEqual(times[1] - times[0], 300.0)
        self.assertEqual(times[-1], 471900.0)

    def test_pressure_inventory_localizes_known_synthetic_negative(self) -> None:
        inventory = {
            "negative_raw_Pe_count": 0,
            "negative_raw_Pi_count": 0,
            "negative_raw_Pe_count_by_shard": [0] * 8,
            "negative_raw_Pi_count_by_shard": [0] * 8,
            "minimum_raw_Pi": float("inf"),
            "minimum_raw_Pi_location_txyz": None,
        }
        block = np.ones((4, 2, 1, 3), dtype=np.float64)
        block[3, 1, 0, 2] = -2.5
        update_pressure_inventory(
            inventory, field="Pi", block=block, pe_x=4, pe_y=6
        )
        self.assertEqual(inventory["negative_raw_Pi_count"], 1)
        self.assertEqual(inventory["negative_raw_Pi_count_by_shard"], [1] + [0] * 7)
        self.assertEqual(inventory["minimum_raw_Pi"], -2.5)
        self.assertEqual(
            inventory["minimum_raw_Pi_location_txyz"], [3, 9, 6, 2]
        )

    def test_boundary_discrepancy_uses_frozen_scale_aware_rule(self) -> None:
        reference = np.array([1.0, 10.0])
        candidate = np.array([1.0 + 1e-13, 10.0 + 2e-11])
        self.assertEqual(
            discrepancy_count(reference, candidate, atol=1e-12, rtol=1e-12),
            1,
        )
        candidate[0] = np.nan
        self.assertEqual(
            discrepancy_count(reference, candidate, atol=1e-12, rtol=1e-12),
            2,
        )


class AllFrameReductionTests(unittest.TestCase):
    def test_disjoint_sufficient_merge_matches_one_pass_metrics(self) -> None:
        reference = np.arange(24, dtype=np.float64).reshape(2, 2, 2, 3) + 1.0
        candidate = reference.copy()
        candidate[0, 1, 1, 2] += 0.25
        candidate[1, 0, 1, 1] -= 0.5
        mask = np.ones((2, 2), dtype=bool)
        records = [
            sufficient_statistics(
                candidate[index : index + 1],
                reference[index : index + 1],
                mask,
                frame_start=index,
                location_axes=("shard_frame_position", "x", "y", "z"),
            )
            for index in range(2)
        ]
        merged = metrics_from_sufficient(merge_sufficient_statistics(records))
        expected_error = candidate - reference
        self.assertAlmostEqual(
            merged["rmse"], float(np.sqrt(np.mean(expected_error**2)))
        )
        self.assertAlmostEqual(
            merged["relative_l2"],
            float(np.linalg.norm(expected_error) / np.linalg.norm(reference)),
        )
        self.assertAlmostEqual(merged["bias"], float(np.mean(expected_error)))
        self.assertEqual(merged["maximum_absolute_difference"], 0.5)
        self.assertEqual(merged["maximum_location"]["frame_index"], 1)

    def test_mode_power_merge_matches_direct_concatenation(self) -> None:
        z = 2.0 * np.pi * np.arange(81) / 81.0
        reference = np.stack(
            [np.cos(z), 2.0 * np.cos(2.0 * z)], axis=0
        )[:, None, None, :]
        candidate = reference.copy()
        candidate[0] += 0.01 * np.sin(3.0 * z)
        candidate[1] += 0.02 * np.sin(4.0 * z)
        records = []
        for index in range(2):
            summary = mode_residual_summary(
                candidate[index], reference[index], zperiod=5
            )
            records.append(
                {
                    "mode_reference_power": summary["reference_power"],
                    "mode_residual_power": summary["residual_power"],
                }
            )
        merged = merge_mode_power(records, zperiod=5)
        direct = mode_residual_summary(candidate, reference, zperiod=5)
        np.testing.assert_allclose(merged["reference_power"], direct["reference_power"])
        np.testing.assert_allclose(merged["residual_power"], direct["residual_power"])
        self.assertEqual(merged["toroidal_mode_n"], [5 * k for k in range(41)])

    def test_strict_json_refuses_overwrite_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_strict_json(path, {"value": 1.0})
            with self.assertRaises(FileExistsError):
                write_strict_json(path, {"value": 2.0})
            with self.assertRaises(ValueError):
                write_strict_json(Path(directory) / "bad.json", {"value": np.nan})


class AllFrameCompiledImplementationTests(unittest.TestCase):
    def test_new_files_are_locked_and_selected_driver_is_unchanged(self) -> None:
        self.assertEqual(sha256(SELECTED_DRIVER), SELECTED_DRIVER_SHA256)
        for path, expected in FILE_HASHES.items():
            with self.subTest(path=path):
                self.assertEqual(sha256(path), expected)

    def test_driver_uses_exact_forward_path_and_78_contiguous_frames(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("constexpr int FRAME_COUNT = 78", source)
        self.assertIn("shard_start % FRAME_COUNT", source)
        self.assertIn("solver.tridagCoefs", source)
        self.assertIn("rfft(communicated", source)
        self.assertIn("irfft(forward_line.data()", source)
        self.assertIn('output["input_Vort_" + label]', source)
        self.assertIn('output["runtime_Pe_" + label]', source)
        self.assertIn('output["runtime_Pi_" + label]', source)
        self.assertIn('output["forward_Vort_" + label]', source)
        self.assertNotIn("FV::Div_a_Grad_perp", source)
        self.assertNotIn('output["input_phi_" + label]', source)
        cmake = CMAKE.read_text(encoding="utf-8")
        self.assertIn("PAPER0_RUNTIME_PRESSURE_CORRECTION=1", cmake)
        self.assertIn("potential_vorticity_all_frame_oracle", cmake)

    def test_extractor_streams_one_rank_loop_into_eight_files(self) -> None:
        source = EXTRACTOR.read_text(encoding="utf-8")
        self.assertIn("for path in paths:", source)
        self.assertIn("for shard_index, (start, stop) in enumerate(EXPECTED_SHARDS)", source)
        self.assertIn("rank_files_traversed_once", source)
        self.assertIn("chunksizes=(78, 4, 2, 81)", source)
        self.assertIn('"canonical_volume_chunks": [78, 4, 2, 81]', source)
        self.assertIn("refusing to overwrite all-frame extraction artifacts", source)
        self.assertNotIn("85606/", source)

    def test_comparator_blocks_source_until_preliminary_gates(self) -> None:
        source = COMPARATOR.read_text(encoding="utf-8")
        preliminary = source.index("if preliminary_passed:")
        forward = source.index('f"forward_Vort_{label}"')
        self.assertLess(preliminary, forward)
        self.assertIn('"status": "blocked_by_preliminary_gate"', source)
        self.assertIn("merge_sufficient_statistics", source)


if __name__ == "__main__":
    unittest.main()
