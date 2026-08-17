from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from paper0.tools.summarize_potential_vorticity_all_frame import (
    FRAME_COUNT,
    frame_rows,
    summarize_frames,
    summarize_runtime,
    write_strict_json,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_TOOL = ROOT / "paper0/tools/summarize_potential_vorticity_all_frame.py"
RESULT = ROOT / "paper0/results/phase2_potential_vorticity_all_frame_6893033.json"
SUMMARY_TOOL_SHA256 = (
    "af48f2f93287daaeecf125a6fe5f6f35eb1e07967d45efe6f37945ec68e70126"
)
RESULT_SHA256 = "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(*, maximum: float = 1.0, tolerance: float = 2.0) -> dict[str, object]:
    return {
        "acceptance_tolerance": tolerance,
        "bias": 0.0,
        "correlation": 1.0,
        "max_abs_reference": 1.0,
        "maximum_absolute_difference": maximum,
        "maximum_location": {"x": 0, "y": 0, "z": 0},
        "nonfinite_count": 0,
        "passed": maximum <= tolerance,
        "point_count": 1,
        "relative_l2": maximum / 10.0,
        "rmse": maximum / 20.0,
    }


class AllFrameSummaryTests(unittest.TestCase):
    def test_tracked_result_is_hash_locked_and_passes_exact_gates(self) -> None:
        self.assertEqual(sha256(SUMMARY_TOOL), SUMMARY_TOOL_SHA256)
        self.assertEqual(sha256(RESULT), RESULT_SHA256)
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(result["slurm_job_id"], 6893033)
        self.assertEqual(
            result["paper0_commit"], "d3c73231e2a2d0cf49fd3d0c87a8155a3cc20d75"
        )
        self.assertTrue(all(result["ordered_gates"].values()))
        self.assertTrue(
            result["decision"]["all_frame_bidirectional_closure_validated"]
        )
        self.assertFalse(result["decision"]["automatic_training_authorized"])
        self.assertFalse(
            result["decision"]["automatic_held_out_access_authorized"]
        )
        self.assertEqual(result["runtime_pressure_gate"]["negative_raw_Pi_count"], 3412)
        source = result["source_forward_closure_gate"]
        self.assertTrue(source["passed"])
        self.assertEqual(source["pooled"]["point_count"], 103514112)
        self.assertAlmostEqual(source["pooled"]["relative_l2"], 6.502783244122983e-13)
        self.assertAlmostEqual(
            source["frame_extrema"]["maximum_gate_fraction"],
            0.0007981173775212814,
        )
        self.assertEqual(
            source["frame_extrema"]["maximum_absolute_difference"]["frame_index"],
            169,
        )
        self.assertEqual(
            result["external_artifacts"]["full_result"]["sha256"],
            "407d6a46387e22c0af8f279e2292974d2aa9f73394cec02005c8cc026ec60cfc",
        )

    def test_frame_summary_separates_absolute_and_gate_extrema(self) -> None:
        per_frame = {
            f"f{frame:03d}": metric() for frame in range(FRAME_COUNT)
        }
        per_frame["f005"] = metric(maximum=0.9, tolerance=1.0)
        per_frame["f006"] = metric(maximum=100.0, tolerance=1000.0)
        intervals = [[start, start + 78] for start in range(0, FRAME_COUNT, 78)]
        summary = summarize_frames(frame_rows(per_frame), intervals)
        self.assertTrue(summary["all_frames_passed"])
        self.assertEqual(summary["maximum_gate_fraction_frame"], 5)
        self.assertEqual(summary["maximum_gate_fraction"], 0.9)
        self.assertEqual(summary["maximum_absolute_difference"]["frame_index"], 6)
        self.assertEqual(len(summary["by_predeclared_temporal_block"]), 8)

    def test_frame_rows_requires_exact_order_and_coverage(self) -> None:
        per_frame = {
            f"f{frame:03d}": metric() for frame in range(FRAME_COUNT)
        }
        per_frame["f624"] = per_frame.pop("f623")
        with self.assertRaises(ValueError):
            frame_rows(per_frame)

    def test_runtime_summary_retains_per_field_extrema(self) -> None:
        per_frame = {
            f"f{frame:03d}": {"Pe": metric(maximum=0.0), "Pi": metric(maximum=0.0)}
            for frame in range(FRAME_COUNT)
        }
        per_frame["f101"]["Pi"] = metric(maximum=0.25)
        summary = summarize_runtime(per_frame)
        self.assertTrue(summary["Pe"]["all_frames_passed"])
        self.assertTrue(summary["Pi"]["all_frames_passed"])
        self.assertEqual(
            summary["Pi"]["maximum_absolute_difference"]["frame_index"], 101
        )

    def test_strict_summary_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.json"
            write_strict_json(output, {"value": 1.0})
            with self.assertRaises(FileExistsError):
                write_strict_json(output, {"value": 2.0})
            with self.assertRaises(ValueError):
                write_strict_json(
                    Path(directory) / "nonfinite.json", {"value": float("nan")}
                )


if __name__ == "__main__":
    unittest.main()
