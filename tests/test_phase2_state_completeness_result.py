from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "paper0/results/phase2_state_completeness_6891855.json"
RAW_SHA256 = "9fec0426a97fab9e15b0029d80f1f6c6464d0d7e34aac4216ec4a76ceb3bda93"
EXECUTED_COMMIT = "4913361b4f1ee5f04f8fd3e95ac9240b3941c9fc"
COMPACTOR_COMMIT = "54d2bba33cf4a5458bc8e61cb794024de0849d7f"
POINTS_PER_STREAM = 103_514_112


class StateCompletenessResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_provenance_and_scope_are_exact(self) -> None:
        result = self.result
        self.assertEqual(result["phase"], "phase2_85604_state_completeness_compact")
        self.assertEqual(result["development_run"], "85604")
        self.assertFalse(result["held_out_85606_read"])
        raw = result["raw_artifact"]
        self.assertEqual(raw["slurm_job_id"], 6_891_855)
        self.assertEqual(raw["executed_paper0_commit"], EXECUTED_COMMIT)
        self.assertEqual(raw["sha256"], RAW_SHA256)
        self.assertRegex(raw["sha256"], re.compile(r"^[0-9a-f]{64}$"))
        self.assertEqual(result["compactor_commit"], COMPACTOR_COMMIT)

    def test_full_archive_coverage_is_preserved(self) -> None:
        coverage = self.result["coverage"]
        self.assertEqual(coverage["rank_file_count"], 256)
        self.assertEqual(coverage["frame_count"], 624)
        self.assertEqual(coverage["shape_per_field"], [624, 64, 32, 81])
        self.assertEqual(coverage["total_points_per_stream"], POINTS_PER_STREAM)
        self.assertEqual(coverage["metadata_inventory_field_count"], 11)
        self.assertEqual(coverage["native_z_samples"], 81)
        self.assertEqual(coverage["zperiod"], 5)
        self.assertEqual(coverage["processor_coverage"]["unique_coordinates"], 256)
        self.assertTrue(coverage["processor_coverage"]["complete"])
        self.assertEqual(
            coverage["normalized_time"]["physical_cadence_microseconds"],
            3.131905426352636,
        )

    def test_all_streamed_fields_are_finite(self) -> None:
        expected = {"Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "Ve", "Vi"}
        fields = self.result["field_statistics"]
        self.assertEqual(set(fields), expected)
        for field, statistics in fields.items():
            full = statistics["full_physical_domain"]
            self.assertEqual(full["total_count"], POINTS_PER_STREAM, field)
            self.assertEqual(full["finite_count"], POINTS_PER_STREAM, field)
            self.assertEqual(full["nonfinite_count"], 0, field)

    def test_density_floor_is_inactive_everywhere(self) -> None:
        floor = self.result["density_floor_statistics"]
        self.assertEqual(floor["density_floor"], 1e-7)
        for scope, counts in floor["scope_counts"].items():
            self.assertEqual(
                counts,
                {"below_density_floor": 0, "below_zero": 0, "softfloor_changed": 0},
                scope,
            )
        for counts in floor["count_by_temporal_block"].values():
            self.assertEqual(counts, [0] * 8)

    def test_velocity_momentum_closures_pass_all_frames(self) -> None:
        relations = self.result["closure_statistics"]["relations"]
        self.assertEqual(
            set(relations),
            {
                "NVe_from_softfloor_Ne_Ve",
                "NVi_from_softfloor_Ne_Vi",
                "NVe_from_plain_Ne_Ve",
                "NVi_from_plain_Ne_Vi",
            },
        )
        for relation, statistics in relations.items():
            for scope, metrics in statistics["scopes"].items():
                self.assertEqual(metrics["nonfinite_count"], 0, (relation, scope))
                self.assertEqual(metrics["point_discrepancy_count"], 0, (relation, scope))
                self.assertEqual(metrics["frame_pass_count"], 624, (relation, scope))
                self.assertEqual(metrics["frame_fail_count"], 0, (relation, scope))
                self.assertEqual(metrics["failed_frame_indices"], [], (relation, scope))
                self.assertLess(metrics["relative_l2_error"], 6e-15, (relation, scope))
            self.assertEqual(
                statistics["point_discrepancy_count_by_temporal_block"],
                [0] * 8,
            )

    def test_findings_do_not_overreach_unfinished_state_gate(self) -> None:
        findings = self.result["scientific_findings"]
        self.assertTrue(findings["all_relevant_fields_finite"])
        self.assertTrue(findings["source_exact_velocity_momentum_equivalence"])
        self.assertFalse(findings["historical_c5_contains_electron_velocity_or_momentum"])
        self.assertFalse(findings["historical_c5_is_complete_evolved_state"])
        self.assertFalse(findings["potential_vorticity_gate_completed"])
        self.assertFalse(findings["automatic_channel_change_authorized"])


if __name__ == "__main__":
    unittest.main()
