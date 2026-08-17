from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "paper0/manifests/phase2_potential_elliptic_85604.json"
)
PROTOCOL_PATH = ROOT / "paper0/protocol/PHASE2_POTENTIAL_ELLIPTIC_PROTOCOL.md"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PotentialEllipticProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    def test_scope_is_development_only_and_nontraining(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["development_run"], "85604")
        self.assertEqual(manifest["sequestered_run"], "85606")
        self.assertFalse(manifest["held_out_85606_access_allowed"])
        self.assertFalse(manifest["training_allowed"])
        self.assertFalse(
            manifest["scientific_scope"]["evaluates_learned_model"]
        )
        self.assertFalse(
            manifest["scientific_scope"]["establishes_held_out_generalization"]
        )

    def test_frame_selection_reuses_the_prior_value_independent_rule(self) -> None:
        selection = self.manifest["frame_selection"]
        self.assertTrue(selection["selection_predates_boundary_amplitude_result"])
        self.assertEqual(selection["indices"], [0, 156, 312, 467, 623])
        self.assertEqual(selection["fractions"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(
            selection["posthoc_boundary_maximum_frames_excluded_from_selection_rule"],
            [586, 591],
        )
        self.assertFalse(selection["physical_independence_claimed"])
        locks = self.manifest["provenance_locks"]
        self.assertEqual(
            locks["value_independent_frame_selection"]["sha256"],
            "f99c540e00b1f8f99ae0712a3b161e851b2b72199cff4ad609a7026b27b0ee5e",
        )
        self.assertEqual(
            locks["all_frame_phi_boundary_audit"]["sha256"],
            "79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc",
        )
        self.assertEqual(
            locks["all_frame_phi_boundary_audit"]["rusty_job_id"], 6_891_890
        )

    def test_exact_equation_includes_electron_pressure_term(self) -> None:
        equation = self.manifest["elliptic_equation"]
        self.assertEqual(equation["pressure_correction"], "Pi_hat = Pi - Pe / 3672")
        self.assertEqual(equation["coefficient_C"], "2 / Bxy^2")
        self.assertEqual(equation["right_hand_side"], "Vort * Bxy^2 / 2")
        self.assertEqual(equation["solver_type"], "cyclic")
        self.assertEqual(equation["inner_boundary_flags"], "INVERT_SET")
        self.assertEqual(equation["outer_boundary_flags"], "INVERT_SET")
        self.assertTrue(equation["source_half_cell_adjustment_required"])
        self.assertIn(r"P_i-\frac{P_e}{3672}", self.protocol)

    def test_metric_and_source_are_fully_locked(self) -> None:
        normalization = self.manifest["normalization"]
        self.assertEqual(normalization["Bnorm_tesla"], 1.0)
        self.assertEqual(
            normalization["rho_s0_meters"], 0.0007224847664314034
        )
        self.assertTrue(
            normalization["requires_exact_hermes_metric_normalization_block"]
        )
        source = self.manifest["source_lock"]
        self.assertEqual(
            source["hermes_revision"],
            "920ba829cc78cdab0dbf6101c69fecc4689bd8dd",
        )
        self.assertEqual(
            source["bout_revision"],
            "7d28d67c3f12c24ec281c0982e870f5369c65a6f",
        )
        for key, value in source.items():
            if key.endswith("_sha256"):
                self.assertRegex(value, SHA256, key)
        dependency = self.manifest["bout_dependency"]
        self.assertEqual(dependency["build_job_id"], 6_890_766)
        self.assertRegex(dependency["shared_library_sha256"], SHA256)
        self.assertFalse(dependency["mixed_hdf5_abi_detected"])

    def test_canonical_shapes_and_paired_arms_are_unambiguous(self) -> None:
        extraction = self.manifest["canonical_extraction"]
        self.assertEqual(
            extraction["volume_fields"], ["Ne", "Pe", "Pi", "Vort", "phi"]
        )
        self.assertEqual(extraction["volume_shape_per_field"], [5, 64, 32, 81])
        self.assertEqual(extraction["boundary_shape_per_array"], [5, 2, 32])
        self.assertEqual(extraction["boundary_sides"], ["inner", "outer"])
        self.assertTrue(extraction["refuse_overwrite"])
        arms = self.manifest["paired_arms"]
        self.assertEqual(arms["retained"]["boundary_midpoint"], "saved_midpoint")
        self.assertFalse(
            arms["instantaneous"]["self_consistent_iterated_neumann_claimed"]
        )

    def test_reconstruction_gate_precedes_counterfactual(self) -> None:
        gate = self.manifest["source_reconstruction_gate"]
        self.assertTrue(gate["required_before_paired_effect_interpretation"])
        self.assertTrue(gate["input_echo_requires_bitwise_equality"])
        self.assertTrue(gate["boundary_echo_requires_bitwise_equality"])
        self.assertTrue(gate["requires_all_values_finite"])
        self.assertEqual(gate["continuous_atol"], 5e-10)
        self.assertEqual(gate["continuous_rtol"], 5e-10)
        self.assertTrue(gate["constant_shift_aligned_error_is_diagnostic_only"])
        self.assertIn("stop", self.protocol.lower())

    def test_materiality_and_state_decisions_do_not_overreach(self) -> None:
        metrics = self.manifest["paired_effect_metrics"]
        self.assertFalse(metrics["posthoc_materiality_threshold_allowed"])
        self.assertFalse(metrics["internal_energy_called_total_heat_flux"])
        decisions = self.manifest["decision_rules"]
        self.assertFalse(decisions["automatic_model_state_change_authorized"])
        self.assertFalse(decisions["automatic_training_authorized"])
        self.assertEqual(decisions["exact_source_state_candidate"], "S6+Bphi")
        self.assertFalse(
            decisions["selected_frame_pass_establishes_all_frame_stability"]
        )

    def test_protocol_declares_fixed_target_not_iterated_neumann(self) -> None:
        self.assertIn("one-step counterfactual", self.protocol)
        self.assertIn("not an iterated self-consistent Neumann solve", self.protocol)
        self.assertIn("No post hoc materiality label", self.protocol)


if __name__ == "__main__":
    unittest.main()
