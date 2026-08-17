from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "paper0/results/phase2_phi_boundary_state_6891890.json"
RESULT_SHA256 = (
    "79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc"
)
EXECUTED_COMMIT = "cee2264a88ae7a912f8a70a06086137bf16d4e76"


class PhiBoundaryStateResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_tracked_result_is_the_verified_raw_artifact(self) -> None:
        digest = hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, RESULT_SHA256)

    def test_provenance_scope_and_coverage_are_exact(self) -> None:
        result = self.result
        self.assertEqual(result["phase"], "phase2_85604_phi_boundary_state_audit")
        self.assertTrue(result["audit_completed"])
        self.assertEqual(result["paper0_commit"], EXECUTED_COMMIT)
        self.assertEqual(result["slurm_job_id"], 6_891_890)
        self.assertEqual(result["development_run"], "85604")
        self.assertFalse(result["held_out_85606_read"])
        self.assertEqual(result["archive_rank_file_count"], 256)
        self.assertEqual(result["selected_rank_count"], 32)
        self.assertEqual(result["frame_count"], 624)
        self.assertEqual(result["native_z_samples"], 81)
        self.assertEqual(result["zperiod"], 5)
        self.assertTrue(result["processor_coordinate_coverage_complete"])
        self.assertTrue(result["boundary_global_y_coverage_complete"])

    def test_source_policy_is_recorded_without_misreading_the_weight(self) -> None:
        policy = self.result["source_boundary_policy"]
        self.assertTrue(policy["phi_boundary_relax"])
        self.assertEqual(policy["phi_boundary_timescale_microseconds"], 1.0)
        self.assertFalse(policy["phi_core_averagey"])
        self.assertEqual(
            policy["homogeneous_saved_frame_memory_weight"],
            0.043634575521405435,
        )
        self.assertIn("not an empirical lag-one prediction", policy["weight_interpretation"])

    def test_saved_boundary_structure_passes_exact_checks(self) -> None:
        for side, statistics in self.result["per_side"].items():
            self.assertEqual(
                statistics["plane_nonfinite_counts"],
                {
                    "outermost_guard": 0,
                    "adjacent_guard": 0,
                    "adjacent_interior": 0,
                },
                side,
            )
            for name in ("outer_guard_copy", "midpoint_toroidal_constancy"):
                check = statistics[name]
                self.assertEqual(check["total_count"], 1_617_408, (side, name))
                self.assertEqual(check["nonfinite_count"], 0, (side, name))
                self.assertEqual(check["point_discrepancy_count"], 0, (side, name))
                self.assertLess(
                    check["maximum_error"]["absolute_error"],
                    2e-14,
                    (side, name),
                )

    def test_instantaneous_neumann_state_fails_every_saved_location(self) -> None:
        for side, statistics in self.result["per_side"].items():
            check = statistics["instantaneous_neumann"]
            self.assertEqual(check["total_count"], 19_968, side)
            self.assertEqual(check["point_discrepancy_count"], 19_968, side)
            self.assertEqual(
                check["point_discrepancy_count_by_temporal_block"],
                [2_496] * 8,
                side,
            )

    def test_departure_amplitudes_and_context_are_exact(self) -> None:
        inner = self.result["per_side"]["inner"]["departure"]
        outer = self.result["per_side"]["outer"]["departure"]
        self.assertAlmostEqual(inner["rms_physical"], 1.0726069667800957)
        self.assertAlmostEqual(outer["rms_physical"], 0.5129863055535508)
        self.assertAlmostEqual(inner["maximum_absolute_physical"], 8.117106253382596)
        self.assertAlmostEqual(outer["maximum_absolute_physical"], 1.9916897329591565)
        self.assertAlmostEqual(
            inner["rms_to_interior_toroidal_fluctuation_rms"],
            0.7448348896333629,
        )
        self.assertAlmostEqual(
            outer["rms_to_interior_toroidal_fluctuation_rms"],
            1.2814007050453133,
        )
        self.assertEqual(inner["nonfinite_count"], 0)
        self.assertEqual(outer["nonfinite_count"], 0)

    def test_lag_summaries_remain_descriptive(self) -> None:
        inner = self.result["per_side"]["inner"]["lag_one_correlation_summary"]
        outer = self.result["per_side"]["outer"]["lag_one_correlation_summary"]
        self.assertEqual(inner["finite_y_count"], 32)
        self.assertEqual(outer["finite_y_count"], 32)
        self.assertAlmostEqual(inner["mean_across_y"], 0.43729005646879027)
        self.assertAlmostEqual(outer["mean_across_y"], 0.9708878315233679)

    def test_findings_require_the_paired_elliptic_solve(self) -> None:
        findings = self.result["scientific_findings"]
        self.assertTrue(findings["all_boundary_planes_finite"])
        self.assertTrue(findings["outer_guard_copy_passes"])
        self.assertTrue(findings["midpoint_toroidal_constancy_passes"])
        self.assertTrue(findings["saved_compact_boundary_value_structurally_valid"])
        self.assertTrue(findings["nonzero_saved_boundary_state_detected"])
        self.assertFalse(findings["instantaneous_neumann_passes_everywhere"])
        self.assertTrue(findings["paired_elliptic_solve_required_for_materiality"])
        self.assertFalse(findings["materiality_established"])
        self.assertFalse(findings["potential_vorticity_gate_completed"])
        self.assertFalse(findings["automatic_state_change_authorized"])


if __name__ == "__main__":
    unittest.main()
