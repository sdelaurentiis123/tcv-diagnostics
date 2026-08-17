from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase2_85604_pressure_closure_audit.json"
)
PROTOCOL = (
    ROOT
    / "paper0"
    / "protocol"
    / "PHASE2_PRESSURE_CLOSURE_AUDIT_PROTOCOL.md"
)


class PressureClosureAuditProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_full_85604_scope_is_frozen_without_blind_run(self) -> None:
        self.assertEqual(self.manifest["development_run"], "85604")
        self.assertEqual(self.manifest["sequestered_run"], "85606")
        self.assertFalse(self.manifest["held_out_85606_access_allowed"])
        scope = self.manifest["frame_scope"]
        self.assertEqual((scope["first_index"], scope["last_index"]), (0, 623))
        self.assertEqual(scope["frame_count"], 624)
        self.assertFalse(scope["selection_uses_field_values"])

    def test_native_shape_time_and_toroidal_metadata_are_exact(self) -> None:
        archive = self.manifest["raw_archive"]
        cells = self.manifest["canonical_cells"]
        self.assertEqual(archive["expected_rank_file_count"], 256)
        self.assertEqual(archive["native_z_samples"], 81)
        self.assertEqual(archive["zperiod"], 5)
        self.assertEqual(cells["shape_per_field"], [624, 64, 32, 81])
        self.assertEqual(cells["total_points_per_field"], 624 * 64 * 32 * 81)
        self.assertEqual(
            self.manifest["frame_scope"]["expected_normalized_cadence"],
            300.0,
        )

    def test_blocks_fields_scopes_and_relations_are_predeclared(self) -> None:
        blocks = self.manifest["temporal_blocks"]["inclusive_index_ranges"]
        self.assertEqual(len(blocks), 8)
        self.assertEqual(blocks[0], [0, 77])
        self.assertEqual(blocks[-1], [546, 623])
        self.assertEqual(
            self.manifest["fields"], ["Ne", "Ni", "Te", "Ti", "Pe", "Pi"]
        )
        scopes = self.manifest["spatial_scopes"]
        self.assertEqual(
            scopes["guard_independent_transport_interior"]["included_y_indices"],
            [1, 30],
        )
        self.assertEqual(
            scopes["target_dependent_rows"]["included_y_indices"], [0, 31]
        )
        self.assertEqual(
            set(self.manifest["closure_statistics"]["relations"]),
            {
                "Ni_equals_Ne",
                "Pe_equals_Ne_times_Te",
                "Pi_equals_Ni_times_Ti",
                "Pi_equals_Ne_times_Ti",
            },
        )

    def test_tolerances_and_no_posthoc_channel_change_are_frozen(self) -> None:
        closure = self.manifest["closure_statistics"]
        self.assertEqual((closure["atol"], closure["rtol"]), (1e-12, 1e-12))
        decisions = self.manifest["decision_rules"]
        self.assertFalse(decisions["automatic_channel_change_authorized"])
        self.assertFalse(decisions["posthoc_prevalence_threshold_authorized"])
        completion = self.manifest["completion_requirements"]
        self.assertTrue(completion["scientific_findings_do_not_change_process_exit_status"])

    def test_human_protocol_contains_the_decisive_machine_rules(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        for required in (
            "103,514,112",
            "z=81",
            "y=1..30",
            "y in {0,31}",
            "Ni == Ne",
            "Pe == Ne * Te",
            "Pi == Ni * Ti",
            "Pi == Ne * Ti",
            "atol = 1e-12",
            "rtol = 1e-12",
            "No frequency or magnitude discovered by this job automatically",
            "rank modulo 16 == s",
            "all rank indices `0..255`",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
