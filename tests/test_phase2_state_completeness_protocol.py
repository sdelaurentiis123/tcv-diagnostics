from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0" / "manifests" / "phase2_85604_state_completeness.json"
PROTOCOL = ROOT / "paper0" / "protocol" / "PHASE2_STATE_COMPLETENESS_PROTOCOL.md"


class StateCompletenessProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_scope_is_85604_only_and_prospective_for_all_frame_momentum(self) -> None:
        self.assertEqual(self.manifest["development_run"], "85604")
        self.assertEqual(self.manifest["sequestered_run"], "85606")
        self.assertFalse(self.manifest["held_out_85606_access_allowed"])
        self.assertIn(
            "no all-frame raw momentum values",
            self.manifest["prior_reads"]["current_freeze_read"],
        )
        self.assertFalse(self.manifest["frame_scope"]["selection_uses_field_values"])

    def test_native_geometry_time_and_rank_scope_are_locked(self) -> None:
        archive = self.manifest["raw_archive"]
        self.assertEqual(archive["expected_rank_file_count"], 256)
        self.assertEqual(archive["zperiod"], 5)
        self.assertEqual(archive["native_z_samples"], 81)
        self.assertEqual(archive["rank_dimensions"], {"t": 624, "x": 8, "y": 6, "z": 81})
        self.assertEqual(self.manifest["canonical_cells"]["shape_per_field"], [624, 64, 32, 81])
        self.assertEqual(self.manifest["canonical_cells"]["total_points_per_field"], 624 * 64 * 32 * 81)

    def test_evolved_and_derived_inventory_is_explicit(self) -> None:
        inventory = self.manifest["field_inventory"]
        self.assertEqual(inventory["evolved_fields"], ["Ne", "Pe", "Pi", "NVe", "NVi", "Vort"])
        self.assertEqual(inventory["derived_fields"], ["Te", "Ti", "Ve", "Vi", "phi"])
        self.assertEqual(len(inventory["metadata_inventory_fields"]), 11)
        self.assertEqual(len(inventory["value_stream_fields"]), 8)
        self.assertEqual(inventory["expected_common"]["dimensions"], ["t", "x", "y", "z"])

    def test_exact_and_attribution_relations_cannot_be_confused(self) -> None:
        formula = self.manifest["momentum_formula"]
        self.assertEqual(formula["density_floor"], 1e-7)
        self.assertEqual(formula["electron_atomic_mass"], {"numerator": 1, "denominator": 1836})
        self.assertEqual(formula["ion_atomic_mass"], {"numerator": 2, "denominator": 1})
        relations = formula["relations"]
        self.assertTrue(relations["NVe_from_softfloor_Ne_Ve"]["source_exact"])
        self.assertTrue(relations["NVi_from_softfloor_Ne_Vi"]["source_exact"])
        self.assertFalse(relations["NVe_from_plain_Ne_Ve"]["source_exact"])
        self.assertFalse(relations["NVi_from_plain_Ne_Vi"]["source_exact"])

    def test_gate_and_nonautomatic_interpretation_are_frozen(self) -> None:
        closure = self.manifest["closure_statistics"]
        self.assertEqual((closure["atol"], closure["rtol"]), (1e-12, 1e-12))
        decisions = self.manifest["decision_rules"]
        self.assertTrue(decisions["plain_density_relations_are_attribution_only"])
        self.assertTrue(decisions["potential_vorticity_closure_is_a_separate_gate"])
        self.assertFalse(decisions["automatic_channel_change_authorized"])
        self.assertFalse(decisions["posthoc_threshold_change_authorized"])

    def test_human_protocol_contains_decisive_rules(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        for required in (
            "103,514,112",
            "zperiod=5",
            "NVe == (1 / 1836) * softFloor(Ne, 1e-7) * Ve",
            "NVi == 2 * softFloor(Ne, 1e-7) * Vi",
            "atol = 1e-12",
            "rtol = 1e-12",
            "rank modulo 16 == s",
            "all rank indices `0..255`",
            "No result automatically changes the training channels",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
