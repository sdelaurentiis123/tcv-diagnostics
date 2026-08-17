from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0" / "tools" / "summarize_resampling_audit.py"
SPEC = importlib.util.spec_from_file_location("summarize_resampling_audit", TOOL)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {TOOL}")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def paired_result(frame_indices: list[int]) -> dict:
    statistics = {
        "point_count": 2,
        "relative_l2": 0.1,
        "normalized_bias": 0.0,
        "rms_ratio": 1.0,
        "pearson_correlation": 1.0,
        "pearson_correlation_defined": True,
        "weighted_sign_disagreement": 0.0,
        "weighted_sign_disagreement_defined": True,
        "reference_rms": 1.0,
        "candidate_rms": 1.0,
    }
    sufficient = {
        "count": 2,
        "sum_reference": 0.0,
        "sum_candidate": 0.0,
        "sum_reference_squared": 2.0,
        "sum_candidate_squared": 2.0,
        "sum_reference_candidate": 2.0,
        "sum_difference": 0.0,
        "sum_difference_squared": 0.02,
        "sum_absolute_reference": 2.0,
        "sum_absolute_reference_sign_disagreement": 0.0,
    }
    scalar_summary = {
        "count": len(frame_indices),
        "minimum": 0.1,
        "maximum": 0.1,
        "mean": 0.1,
        "median": 0.1,
        "p95": 0.1,
        "p99": 0.1,
    }
    return {
        "frame_indices": frame_indices,
        "aggregate": statistics,
        "aggregate_sufficient_statistics": sufficient,
        "toroidal_mean_profile_aggregate": statistics,
        "toroidal_mean_profile_aggregate_sufficient_statistics": sufficient,
        "per_frame_relative_l2_summary": scalar_summary,
        "per_frame_toroidal_mean_profile_relative_l2_summary": scalar_summary,
        "per_frame_absolute_value_p95_ratio_summary": scalar_summary,
        "per_frame_absolute_value_p99_ratio_summary": scalar_summary,
        "relative_l2_by_temporal_block": [],
    }


def source_record() -> dict:
    field_result = paired_result([0, 1])
    field_result = {
        key: field_result[key]
        for key in (
            "aggregate",
            "aggregate_sufficient_statistics",
            "per_frame_relative_l2_summary",
            "relative_l2_by_temporal_block",
        )
    }
    return {
        "schema_version": 1,
        "phase": "phase2_85604_native81_resampled88_sensitivity",
        "paper0_commit": "a" * 40,
        "slurm_job_id": 42,
        "audit_completed": True,
        "rank_shard_completed": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": "/immutable/phase2_manifest.json",
        "manifest_sha256": "b" * 64,
        "frame_count": 624,
        "frame_indices_complete_and_unique": True,
        "native_shape_per_frame": [64, 32, 81],
        "resampled_shape_per_frame": [64, 32, 88],
        "zperiod": 5,
        "parallel_execution": {
            "shard_count": 17,
            "chunk_aligned_intervals": [[0, 40], [620, 624]],
            "all_shards_completed_before_merge": True,
        },
        "field_stream_digest_tree": {field: "c" * 64 for field in summary.FIELDS},
        "structural_checks": {
            "selected_frames": [0, 156, 312, 467, 623],
            "raw_equals_native_after_float32_cast": True,
            "legacy_c5t_resampling_bitwise_exact": True,
        },
        "field_round_trip": {
            field: copy.deepcopy(field_result) for field in summary.FIELDS
        },
        "comparisons": {
            comparison: {
                category: {
                    quantity: paired_result([0, 1])
                    for quantity in summary.PRIMARY_QUANTITIES
                }
                for category in summary.COMPARISON_CATEGORIES
            }
            for comparison in summary.COMPARISON_PATHS
        },
        "acceptance": {"overall_passed": True},
        "scientific_findings": {
            "primary_transport_evaluator": (
                "downsample_each_88_cell_member_to_native_81_then_apply_Q81"
            )
        },
    }


class CompactResamplingResultTests(unittest.TestCase):
    def test_valid_source_compacts_without_frame_arrays(self) -> None:
        raw = source_record()
        summary.validate_source(
            raw, expected_commit="a" * 40, expected_job=42
        )
        compact = summary.compact_record(
            raw, artifact_path="/immutable/full.json", digest="d" * 64, size_bytes=9
        )

        self.assertTrue(compact["acceptance"]["overall_passed"])
        self.assertFalse(compact["data_scope"]["held_out_85606_read"])
        self.assertEqual(compact["data_scope"]["zperiod"], 5)
        particle = compact["transport_comparisons"]["round_trip"][
            "face_total"
        ]["particle"]
        self.assertEqual(particle["frame_coverage"]["count"], 2)
        self.assertTrue(particle["frame_coverage"]["strictly_increasing"])
        self.assertNotIn("per_frame_relative_l2", particle)

    def test_held_out_access_and_incomplete_merge_fail_closed(self) -> None:
        for mutation in ("held_out", "incomplete"):
            with self.subTest(mutation=mutation):
                raw = source_record()
                if mutation == "held_out":
                    raw["held_out_85606_read"] = True
                else:
                    raw["parallel_execution"][
                        "all_shards_completed_before_merge"
                    ] = False
                with self.assertRaises(ValueError):
                    summary.validate_source(
                        raw, expected_commit="a" * 40, expected_job=42
                    )

    def test_schema_drift_fails_closed(self) -> None:
        raw = source_record()
        del raw["comparisons"]["direct_88"]["face_total"]["particle"]
        with self.assertRaises(ValueError):
            summary.validate_source(raw, expected_commit="a" * 40, expected_job=42)


if __name__ == "__main__":
    unittest.main()
