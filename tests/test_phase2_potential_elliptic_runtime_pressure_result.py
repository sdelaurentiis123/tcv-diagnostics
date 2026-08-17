from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT
    / "paper0/results/phase2_potential_elliptic_runtime_pressure_6892641.json"
)
RESULT_SHA256 = (
    "ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a"
)
EXECUTED_COMMIT = "df7fa7d7464c3e91fbcc12c228c6c9d3c5aad6f0"


class PotentialEllipticRuntimePressureResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_tracked_result_is_the_verified_raw_artifact(self) -> None:
        digest = hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, RESULT_SHA256)

    def test_scope_and_provenance_are_exact(self) -> None:
        result = self.result
        self.assertEqual(
            result["phase"],
            "phase2_potential_elliptic_85604_runtime_pressure_correction",
        )
        self.assertEqual(result["paper0_commit"], EXECUTED_COMMIT)
        self.assertEqual(result["slurm_job_id"], 6_892_641)
        self.assertEqual(result["development_run"], "85604")
        self.assertFalse(result["held_out_85606_read"])
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["native_z_samples"], 81)
        self.assertEqual(result["zperiod"], 5)
        self.assertEqual(result["frame_indices"], [0, 156, 312, 467, 623])

    def test_runtime_pressure_transformation_passes_first(self) -> None:
        gate = self.result["runtime_pressure_transformation_gate"]
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["all_runtime_field_gates_passed"])
        self.assertTrue(gate["known_negative_support_passed"])
        self.assertTrue(gate["known_runtime_Pi_zero_passed"])
        self.assertEqual(gate["selected_frame_negative_raw_Pe_count"], 0)
        self.assertEqual(gate["selected_frame_negative_raw_Pi_count"], 1)
        self.assertAlmostEqual(
            gate["known_negative_raw_Pi"], -5.799512988032478e-05
        )
        self.assertEqual(gate["compiled_runtime_Pi_at_known_point"], 0.0)
        self.assertEqual(gate["continuous_atol"], 1e-12)
        self.assertEqual(gate["continuous_rtol"], 1e-12)
        for frame in gate["per_frame"].values():
            for field_name, metrics in frame.items():
                self.assertTrue(metrics["passed"], field_name)
                if field_name in {"Pe", "Pi"}:
                    self.assertEqual(metrics["maximum_absolute_difference"], 0.0)
                else:
                    self.assertLessEqual(
                        metrics["maximum_absolute_difference"],
                        4.440892098500626e-16,
                    )

    def test_source_reconstruction_gate_passes_unchanged_tolerance(self) -> None:
        gate = self.result["source_reconstruction_gate"]
        self.assertTrue(gate["requires_runtime_pressure_transformation_gate"])
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["all_frame_reconstructions_passed"])
        self.assertTrue(gate["volume_input_echoes_passed"])
        self.assertTrue(gate["boundary_input_echoes_passed"])
        self.assertEqual(gate["continuous_atol"], 5e-10)
        self.assertEqual(gate["continuous_rtol"], 5e-10)
        self.assertFalse(gate["constant_shift_alignment_can_change_gate"])
        maxima = []
        for frame in gate["per_frame_reconstruction"].values():
            self.assertTrue(frame["passed"])
            self.assertTrue(frame["raw_full_physical_domain"]["passed"])
            maxima.append(
                frame["raw_full_physical_domain"][
                    "maximum_absolute_difference"
                ]
            )
        self.assertLessEqual(max(maxima), 2.8110846983508964e-13)

    def test_paired_arms_and_potential_effect_are_locked(self) -> None:
        effect = self.result["paired_boundary_effect"]
        self.assertEqual(
            effect["status"], "computed_after_reconstruction_gate_passed"
        )
        self.assertFalse(effect["materiality_label_assigned"])
        potential = effect["potential"]
        self.assertEqual(potential["reference_arm"], "retained_saved_midpoint")
        self.assertEqual(
            potential["candidate_arm"],
            "instantaneous_stored_adjacent_interior_target",
        )
        pooled = potential["normalized"]["pooled"]
        self.assertAlmostEqual(pooled["relative_l2"], 0.004268032092031723)
        self.assertAlmostEqual(
            potential["volts"]["pooled"]["rmse"], 0.6605450390500373
        )
        self.assertAlmostEqual(
            potential["volts"]["pooled"]["maximum_absolute_difference"],
            3.521169288740875,
        )
        regions = potential["normalized"]["by_geometry_region_pooled"]
        self.assertAlmostEqual(
            regions["private_flux"]["relative_l2"], 0.03199092791067865
        )
        self.assertAlmostEqual(
            regions["outboard_midplane"]["relative_l2"],
            0.0028665917971728693,
        )

    def test_transport_effect_is_memberwise_style_and_small_on_sample(self) -> None:
        transport = self.result["paired_boundary_effect"]["transport"]
        self.assertEqual(transport["reference_arm"], "retained_saved_midpoint")
        self.assertEqual(
            transport["candidate_arm"],
            "instantaneous_stored_adjacent_interior_target",
        )
        definition = transport["transport_definition"]
        self.assertTrue(definition["nonlinear_operator_applied_separately_to_each_arm"])
        self.assertFalse(definition["called_total_heat_flux"])

        quantities = transport["quantities"]
        particle = quantities["particle"]["strict_local_faces"]["pooled"]
        self.assertAlmostEqual(particle["relative_l2"], 0.0002342495547119932)
        self.assertAlmostEqual(
            particle["sign_error_fraction"], 0.00147067987785879
        )
        for quantity in quantities.values():
            pooled = quantity["strict_local_faces"]["pooled"]
            self.assertLess(pooled["relative_l2"], 2.4e-4)
            self.assertLess(pooled["sign_error_fraction"], 0.0016)

        wedge = quantities["particle"]["confined_separatrix_wedge"]
        relative_changes = [
            delta / retained
            for delta, retained in zip(
                wedge["instantaneous_minus_retained_normalized"],
                wedge["retained_normalized"],
                strict=True,
            )
        ]
        self.assertAlmostEqual(min(relative_changes), -0.0010189079715469616)
        self.assertAlmostEqual(max(relative_changes), 0.001671722621892974)

    def test_result_does_not_authorize_automatic_next_steps(self) -> None:
        decision = self.result["decision"]
        self.assertTrue(decision["paired_effect_interpretable"])
        self.assertFalse(decision["selected_frames_establish_all_frame_stability"])
        self.assertFalse(decision["automatic_state_change_authorized"])
        self.assertFalse(decision["automatic_training_authorized"])
        self.assertFalse(decision["automatic_held_out_access_authorized"])
        artifacts = self.result["artifacts"]
        self.assertEqual(
            artifacts["comparison_arrays_sha256"],
            "2846bcb9b252fd4565ee657ecff3a7dfc6f47fedbcfcec0e309a1c3895ebbb4b",
        )
        self.assertEqual(
            artifacts["base_comparison_sha256"],
            "863df594e5c41a157f2f98abdfbed00392715ea84bb12eeccbbc2873dbd7e32e",
        )


if __name__ == "__main__":
    unittest.main()
