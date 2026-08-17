from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "paper0/manifests/phase2_potential_vorticity_forward_85604.json"
)
PROTOCOL_PATH = (
    ROOT / "paper0/protocol/PHASE2_POTENTIAL_VORTICITY_FORWARD_PROTOCOL.md"
)
MANIFEST_SHA256 = (
    "4d672514d7d8106e84e39610ee9c18a66d48b7efc6f478a5ba1cc6d530b66789"
)
PROTOCOL_SHA256 = (
    "67d2dfe95eb4e3c24d55dc271fd8d624b514869ae7cf002cba338fecaead2b1d"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PotentialVorticityForwardProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    def test_protocol_and_manifest_are_hash_locked(self) -> None:
        self.assertEqual(sha256(MANIFEST_PATH), MANIFEST_SHA256)
        self.assertEqual(sha256(PROTOCOL_PATH), PROTOCOL_SHA256)
        self.assertEqual(self.manifest["protocol"]["sha256"], PROTOCOL_SHA256)

    def test_scope_is_five_frozen_85604_frames_only(self) -> None:
        scope = self.manifest["scope"]
        self.assertEqual(scope["development_run"], "85604")
        self.assertFalse(scope["held_out_85606_access_allowed"])
        self.assertFalse(scope["training_allowed"])
        self.assertFalse(scope["model_evaluation"])
        self.assertFalse(scope["codec_evaluation"])
        self.assertFalse(scope["assimilation_evaluation"])
        self.assertFalse(scope["selected_frames_are_independent_runs"])
        self.assertEqual(scope["frame_indices"], [0, 156, 312, 467, 623])
        self.assertEqual(scope["physical_shape_xyz"], [64, 32, 81])
        self.assertEqual(scope["native_z_samples"], 81)
        self.assertEqual(scope["zperiod"], 5)
        self.assertEqual(scope["toroidal_mode_mapping"], "n=5k")

    def test_tracked_predecessor_locks_match_actual_files(self) -> None:
        inputs = self.manifest["immutable_inputs"]
        for key in (
            "accepted_inverse_result",
            "runtime_pressure_correction_manifest",
            "runtime_pressure_correction_protocol",
            "bout_input",
        ):
            record = inputs[key]
            self.assertEqual(sha256(ROOT / record["path"]), record["sha256"], key)
        self.assertEqual(
            inputs["accepted_inverse_result"]["sha256"],
            "ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a",
        )
        self.assertEqual(inputs["accepted_inverse_result"]["slurm_job_id"], 6_892_641)

    def test_state_equation_uses_runtime_pressure_and_boussinesq_coefficient(self) -> None:
        state = self.manifest["state_contract"]
        self.assertEqual(
            state["runtime_pressure"], "N*max(P_raw,0)/softFloor(N,1e-7)"
        )
        self.assertEqual(state["pi_hat"], "Pi_runtime-Pe_runtime/3672")
        self.assertEqual(state["average_atomic_mass"], 2.0)
        self.assertEqual(state["coefficient_C"], "2/Bxy^2")
        self.assertEqual(state["forward_field_u"], "phi+Pi_hat")
        self.assertEqual(state["forward_vorticity"], "C*L_C(u)")
        for required in (
            "Vort}_{\\mathrm{forward}",
            "P_s^{\\mathrm{runtime}",
            "\\widehat P_i",
            "3672",
        ):
            self.assertIn(required, self.protocol)

    def test_primary_discretization_is_the_executed_cyclic_matrix(self) -> None:
        operator = self.manifest["discrete_operator"]
        self.assertIn("Laplacian::tridagCoefs", operator["primary"])
        self.assertFalse(operator["alternative_fv_operator_is_primary"])
        self.assertEqual(operator["A"], 0.0)
        self.assertEqual(operator["D"], 1.0)
        self.assertEqual(operator["C1"], "C")
        self.assertEqual(operator["C2"], "C")
        self.assertTrue(operator["all_terms"])
        self.assertTrue(operator["nonuniform"])
        self.assertEqual(operator["fft"], "BOUT++ rfft/irfft")
        self.assertEqual(operator["fourier_indices"], [0, 40])
        self.assertFalse(operator["spectral_filtering_allowed"])
        self.assertFalse(operator["constant_alignment_allowed"])
        self.assertIn("not `FV::Div_a_Grad_perp`", self.protocol)

    def test_boundary_contract_keeps_every_physical_row(self) -> None:
        boundary = self.manifest["boundary_contract"]
        self.assertEqual(boundary["arm"], "retained_saved_midpoint")
        self.assertEqual(boundary["phi_second_guard"], "copy_first_guard")
        self.assertEqual(boundary["pi_hat_boundary"], "Hermes neumann")
        self.assertIn("guard rows", boundary["solver_replaced_rows"])
        self.assertEqual(boundary["primary_physical_x_indices"], [0, 63])
        self.assertEqual(boundary["primary_physical_y_indices"], [0, 31])
        self.assertEqual(boundary["primary_physical_z_indices"], [0, 80])
        self.assertIn("covers the complete\nstored physical domain", self.protocol)

    def test_ordered_gates_are_strict_and_prospective(self) -> None:
        gates = self.manifest["ordered_gates"]
        self.assertEqual(gates["input_echo"]["rule"], "bitwise_exact")
        self.assertEqual(gates["runtime_pressure"]["atol"], 1e-12)
        self.assertEqual(gates["runtime_pressure"]["rtol"], 1e-12)
        self.assertEqual(
            gates["runtime_pressure"]["known_negative_raw_pi_location"],
            {"frame_index": 312, "x": 6, "y": 31, "z": 73},
        )
        compiled = gates["compiled_implementation"]
        self.assertTrue(compiled["constant_null"])
        self.assertTrue(compiled["nonzero_constant_gauge_invariance"])
        self.assertEqual(compiled["manufactured_modes_k"], [0, 3])
        self.assertTrue(compiled["manufactured_forward_inverse_round_trip"])
        source = gates["source_forward_closure"]
        self.assertEqual(source["atol"], 5e-10)
        self.assertEqual(source["rtol"], 5e-10)
        self.assertEqual(source["nonfinite_count"], 0)
        self.assertTrue(source["all_frames_must_pass"])
        self.assertFalse(source["correlation_can_override"])
        self.assertFalse(source["regional_subset_can_override"])
        self.assertFalse(source["alignment_can_override"])

    def test_execution_and_decisions_fail_closed(self) -> None:
        execution = self.manifest["execution"]
        self.assertEqual(execution["os_major"], 9)
        self.assertTrue(execution["cpu_only"])
        self.assertEqual(execution["mpi_ranks"], 4)
        self.assertLessEqual(execution["time_limit_minutes"], 20)
        self.assertTrue(execution["clean_exact_commit_required"])
        self.assertTrue(execution["existing_results_read_only"])
        decision = self.manifest["decision_rules"]
        self.assertFalse(decision["selected_frames_establish_all_frame_stability"])
        self.assertFalse(decision["automatic_state_change_authorized"])
        self.assertFalse(decision["automatic_training_authorized"])
        self.assertFalse(decision["automatic_held_out_access_authorized"])
        self.assertFalse(decision["posthoc_materiality_label_allowed"])


if __name__ == "__main__":
    unittest.main()
