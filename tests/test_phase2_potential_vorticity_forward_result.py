from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = (
    ROOT / "paper0/results/phase2_potential_vorticity_forward_6892764.json"
)
RESULT_SHA256 = (
    "42b3aa3d56ff6f4dbfda9b2cf7317c1f3c0080c3a229f331ff75d58a95406871"
)
EXECUTED_COMMIT = "ab1a5e85633c0e96bc513b32d074d11a3c9356e5"


class PotentialVorticityForwardResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_tracked_result_is_the_verified_raw_artifact(self) -> None:
        digest = hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, RESULT_SHA256)

    def test_scope_and_provenance_are_exact(self) -> None:
        result = self.result
        self.assertEqual(result["phase"], "phase2_potential_vorticity_forward_85604")
        self.assertEqual(result["paper0_commit"], EXECUTED_COMMIT)
        self.assertEqual(result["slurm_job_id"], 6_892_764)
        self.assertEqual(result["development_run"], "85604")
        self.assertFalse(result["held_out_85606_read"])
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["native_z_samples"], 81)
        self.assertEqual(result["zperiod"], 5)
        self.assertEqual(result["physical_shape_xyz"], [64, 32, 81])
        self.assertEqual(result["frame_indices"], [0, 156, 312, 467, 623])
        artifacts = result["artifacts"]
        self.assertEqual(
            artifacts["accepted_inverse_result_sha256"],
            "ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a",
        )
        self.assertEqual(
            artifacts["comparison_arrays_sha256"],
            "0f4ff5a15b92b308abcbbc7b952f6fbfae0b12fb7639fd8be870fddad5c1c289",
        )

    def test_equation_is_the_executed_cyclic_closure(self) -> None:
        equation = self.result["equation"]
        self.assertEqual(equation["C"], "2/Bxy^2")
        self.assertEqual(equation["u"], "phi+Pi_hat")
        self.assertEqual(equation["forward_vorticity"], "C*L_C(u)")
        self.assertEqual(
            equation["discrete_operator"],
            "BOUT++ Laplacian::tridagCoefs with rfft/irfft",
        )
        self.assertFalse(equation["alternative_relax_potential_fv_operator_used"])

    def test_preliminary_gates_pass_before_source_comparison(self) -> None:
        result = self.result
        self.assertTrue(result["input_echo_gate"]["passed"])
        self.assertTrue(result["input_echo_gate"]["volume_passed"])
        self.assertTrue(result["input_echo_gate"]["boundary_passed"])

        pressure = result["runtime_pressure_gate"]
        self.assertTrue(pressure["passed"])
        self.assertTrue(pressure["all_runtime_field_gates_passed"])
        self.assertTrue(pressure["known_negative_support_passed"])
        self.assertEqual(pressure["selected_frame_negative_raw_Pe_count"], 0)
        self.assertEqual(pressure["selected_frame_negative_raw_Pi_count"], 1)
        self.assertEqual(pressure["compiled_runtime_Pi_at_known_point"], 0.0)

        compiled = result["compiled_implementation_gate"]
        self.assertTrue(compiled["passed"])
        self.assertTrue(compiled["constant_null"]["passed"])
        self.assertLessEqual(
            compiled["constant_null"]["maximum_absolute_difference"],
            2.014812859646082e-14,
        )
        self.assertTrue(compiled["manufactured_modes_k0_k3_present"])
        round_trip = compiled["manufactured_forward_inverse_round_trip"]
        self.assertTrue(round_trip["passed"])
        self.assertLessEqual(round_trip["relative_l2"], 1.832083290908516e-14)

    def test_source_forward_closure_passes_comfortably_on_all_selected_points(self) -> None:
        gate = self.result["source_forward_closure_gate"]
        self.assertEqual(
            gate["status"], "evaluated_after_preliminary_gates_passed"
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["atol"], 5e-10)
        self.assertEqual(gate["rtol"], 5e-10)
        self.assertEqual(
            gate["scope"],
            "all_64x32x81_physical_points_for_each_selected_frame",
        )
        pooled = gate["pooled"]
        self.assertTrue(pooled["passed"])
        self.assertEqual(pooled["point_count"], 829_440)
        self.assertEqual(pooled["nonfinite_count"], 0)
        self.assertAlmostEqual(pooled["relative_l2"], 6.362849294223249e-13)
        self.assertAlmostEqual(pooled["rmse"], 2.498689876807738e-14)
        self.assertAlmostEqual(
            pooled["maximum_absolute_difference"], 4.738154313344012e-13
        )
        self.assertLess(
            pooled["maximum_absolute_difference"],
            pooled["acceptance_tolerance"] / 1_000.0,
        )
        for frame in gate["per_frame"].values():
            self.assertTrue(frame["passed"])
            self.assertEqual(frame["point_count"], 165_888)
            self.assertEqual(frame["nonfinite_count"], 0)
            self.assertLess(frame["relative_l2"], 8e-13)

    def test_regions_and_all_native_toroidal_modes_are_reported(self) -> None:
        gate = self.result["source_forward_closure_gate"]
        regions = gate["by_geometry_region_pooled"]
        self.assertEqual(
            set(regions),
            {
                "confined_edge",
                "inner_divertor_leg",
                "outboard_midplane",
                "outer_divertor_leg",
                "private_flux",
                "scrape_off_layer",
                "separatrix_cell_band",
                "x_point_topology_stencil",
            },
        )
        for metrics in regions.values():
            self.assertEqual(metrics["nonfinite_count"], 0)
            self.assertLess(metrics["relative_l2"], 2e-12)

        modes = gate["toroidal_mode_residual"]["pooled"]
        self.assertEqual(modes["fourier_index_k"], list(range(41)))
        self.assertEqual(modes["toroidal_mode_n"], [5 * k for k in range(41)])
        self.assertEqual(len(modes["reference_power"]), 41)
        self.assertEqual(len(modes["residual_power"]), 41)
        self.assertEqual(len(modes["relative_residual_power"]), 41)
        self.assertEqual(
            max(modes["residual_power"]), 7.580775214927354e-21
        )
        finite_relative = [
            value
            for value in modes["relative_residual_power"]
            if value is not None
        ]
        self.assertLessEqual(max(finite_relative), 3.4794793246095317e-23)

    def test_result_closes_only_the_selected_frame_deterministic_gate(self) -> None:
        decision = self.result["decision"]
        self.assertTrue(decision["selected_frame_bidirectional_closure_validated"])
        self.assertFalse(decision["selected_frames_establish_all_frame_stability"])
        self.assertFalse(decision["automatic_state_change_authorized"])
        self.assertFalse(decision["automatic_training_authorized"])
        self.assertFalse(decision["automatic_held_out_access_authorized"])


if __name__ == "__main__":
    unittest.main()
