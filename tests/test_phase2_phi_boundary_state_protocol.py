from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0" / "manifests" / "phase2_85604_phi_boundary_state.json"
PROTOCOL = ROOT / "paper0" / "protocol" / "PHASE2_PHI_BOUNDARY_STATE_PROTOCOL.md"


class PhiBoundaryStateProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_scope_is_prospective_85604_only(self) -> None:
        self.assertEqual(self.manifest["development_run"], "85604")
        self.assertEqual(self.manifest["sequestered_run"], "85606")
        self.assertFalse(self.manifest["held_out_85606_access_allowed"])
        self.assertFalse(self.manifest["frame_scope"]["selection_uses_field_values"])
        self.assertIn("no all-frame", self.manifest["prior_reads"]["current_freeze_read"])

    def test_source_policy_and_theoretical_weight_are_exact(self) -> None:
        policy = self.manifest["source_boundary_policy"]
        self.assertTrue(policy["phi_boundary_relax"])
        self.assertEqual(policy["phi_boundary_timescale_microseconds"], 1.0)
        self.assertFalse(policy["phi_core_averagey"])
        self.assertAlmostEqual(policy["homogeneous_saved_frame_memory_weight"], 0.043634575521405435)
        self.assertEqual(self.manifest["raw_archive"]["zperiod"], 5)

    def test_boundary_rank_and_guard_indices_are_predeclared(self) -> None:
        scope = self.manifest["boundary_rank_scope"]
        self.assertEqual(scope["expected_selected_rank_count"], 32)
        self.assertFalse(scope["selection_uses_field_values"])
        self.assertEqual(scope["inner"]["PE_XIND"], 0)
        self.assertEqual(scope["outer"]["PE_XIND"], 15)
        self.assertEqual(
            [scope["inner"][key] for key in ("outermost_guard_x", "adjacent_guard_x", "adjacent_interior_x")],
            [0, 1, 2],
        )
        self.assertEqual(
            [scope["outer"][key] for key in ("outermost_guard_x", "adjacent_guard_x", "adjacent_interior_x")],
            [7, 6, 5],
        )
        self.assertEqual(scope["local_physical_y_slice"], [2, 4])

    def test_exact_checks_do_not_become_posthoc_materiality_gate(self) -> None:
        checks = self.manifest["exact_checks"]
        self.assertEqual((checks["atol"], checks["rtol"]), (1e-12, 1e-12))
        self.assertFalse(checks["instantaneous_neumann_is_materiality_threshold"])
        decisions = self.manifest["decision_rules"]
        self.assertTrue(decisions["materiality_requires_paired_elliptic_solve"])
        self.assertFalse(decisions["automatic_state_or_channel_change_authorized"])
        self.assertFalse(decisions["posthoc_materiality_threshold_authorized"])

    def test_human_protocol_contains_decisive_definitions(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        for required in (
            "0.043634575521405435",
            "inner: PE_XIND = 0",
            "outer: PE_XIND = 15",
            "midpoint(k) = 0.5 * (adjacent_guard(k) + adjacent_interior(k))",
            "departure(k) = midpoint(k) - target",
            "atol = 1e-12",
            "rtol = 1e-12",
            "No post hoc cutoff",
            "later paired exact elliptic",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
