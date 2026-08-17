from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper0" / "tools" / "summarize_state_completeness.py"
SPEC = importlib.util.spec_from_file_location("state_summary", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load state summary tool")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def point(value: float = 0.0):
    return {"value": value, "location_txyz": [0, 0, 0, 0]}


def field_statistics():
    return {
        "scopes": {
            scope: {
                "total_count": 1,
                "finite_count": 1,
                "nonfinite_count": 0,
                "rms": 1.0,
                "minimum": point(),
                "maximum": point(1.0),
            }
            for scope in SUMMARY.SCOPES
        }
    }


def closure_statistics():
    return {
        "scopes": {
            scope: {
                "total_count": 1,
                "nonfinite_count": 0,
                "point_discrepancy_count": 0,
                "relative_l2_error": 0.0,
                "maximum_error": {
                    "absolute_error": 0.0,
                    "location_txyz": [0, 0, 0, 0],
                    "reference": 1.0,
                    "candidate": 1.0,
                },
                "frame_pass_count": 624,
                "frame_fail_count": 0,
                "failed_frame_indices": [],
            }
            for scope in SUMMARY.SCOPES
        },
        "point_discrepancy_count_by_temporal_block": [0] * 8,
    }


def raw_result():
    digest = "a" * 64
    return {
        "phase": "phase2_85604_state_completeness_audit",
        "audit_completed": True,
        "rank_shard_completed": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "rank_file_count": 256,
        "rank_indices": list(range(256)),
        "shape_per_field": [624, 64, 32, 81],
        "zperiod": 5,
        "native_z_samples": 81,
        "field_statistics": {
            field: field_statistics() for field in SUMMARY.EXPECTED_FIELDS
        },
        "closure_statistics": {
            "atol": 1e-12,
            "rtol": 1e-12,
            "relations": {
                relation: closure_statistics()
                for relation in SUMMARY.EXPECTED_RELATIONS
            },
        },
        "variable_metadata": {str(index): {} for index in range(11)},
        "guard_stripped_stream_digest_tree": {
            field: digest for field in SUMMARY.EXPECTED_FIELDS
        },
        "paper0_commit": "b" * 40,
        "slurm_job_id": 123,
        "manifest": "/manifest.json",
        "manifest_sha256": digest,
        "rank_shard_digest_tree_sha256": digest,
        "frame_count": 624,
        "total_points_per_stream": 624 * 64 * 32 * 81,
        "processor_coverage": {"complete": True, "unique_coordinates": 256},
        "normalized_time": {"first": 285000.0, "last": 471900.0},
        "density_floor_statistics": {
            "density_floor": 1e-7,
            "scope_counts": {},
            "count_by_temporal_block": {},
        },
        "scientific_findings": {
            "source_exact_velocity_momentum_equivalence": True,
            "automatic_channel_change_authorized": False,
        },
    }


class StateSummaryTests(unittest.TestCase):
    def test_compactor_keeps_frozen_primary_metrics_and_provenance(self) -> None:
        compact = SUMMARY.compact_result(
            raw_result(),
            raw_path="/raw.json",
            raw_sha256="c" * 64,
            compactor_commit="d" * 40,
        )
        self.assertEqual(compact["raw_artifact"]["executed_paper0_commit"], "b" * 40)
        self.assertEqual(compact["coverage"]["metadata_inventory_field_count"], 11)
        self.assertEqual(set(compact["field_statistics"]), set(SUMMARY.EXPECTED_FIELDS))
        self.assertEqual(
            set(compact["closure_statistics"]["relations"]),
            set(SUMMARY.EXPECTED_RELATIONS),
        )
        self.assertFalse(compact["held_out_85606_read"])

    def test_compactor_rejects_incomplete_or_blind_input(self) -> None:
        raw = raw_result()
        raw["rank_file_count"] = 255
        with self.assertRaisesRegex(ValueError, "all-rank"):
            SUMMARY.compact_result(
                raw,
                raw_path="/raw.json",
                raw_sha256="c" * 64,
                compactor_commit="d" * 40,
            )
        raw = raw_result()
        raw["held_out_85606_read"] = True
        with self.assertRaisesRegex(ValueError, "85604-only"):
            SUMMARY.compact_result(
                raw,
                raw_path="/raw.json",
                raw_sha256="c" * 64,
                compactor_commit="d" * 40,
            )


if __name__ == "__main__":
    unittest.main()
