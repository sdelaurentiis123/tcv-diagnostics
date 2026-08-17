from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "paper0/manifests/phase2_potential_elliptic_runtime_pressure_correction.json"
)
PROTOCOL_PATH = (
    ROOT
    / "paper0/protocol/PHASE2_POTENTIAL_ELLIPTIC_RUNTIME_PRESSURE_CORRECTION.md"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PotentialEllipticCorrectionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.protocol = PROTOCOL_PATH.read_text(encoding="utf-8")

    def test_scope_is_85604_only_nontraining_and_single_correction(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["development_run"], "85604")
        self.assertEqual(manifest["sequestered_run"], "85606")
        self.assertFalse(manifest["held_out_85606_access_allowed"])
        self.assertFalse(manifest["training_allowed"])
        scope = manifest["scientific_scope"]
        self.assertFalse(scope["evaluates_learned_model"])
        self.assertFalse(scope["establishes_held_out_generalization"])
        self.assertTrue(scope["changes_only_raw_to_runtime_pressure_contract"])

    def test_every_predecessor_lock_matches_the_tracked_artifact(self) -> None:
        for name, lock in self.manifest["predecessor_locks"].items():
            path = ROOT / lock["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, lock["sha256"], name)
            self.assertRegex(lock["sha256"], SHA256)

    def test_known_failure_is_frozen_without_relabeling(self) -> None:
        failure = self.manifest["known_failed_point"]
        self.assertEqual(failure["frame_index"], 312)
        self.assertEqual(failure["model_indices_xyz"], [6, 31, 73])
        self.assertEqual(failure["selected_frame_negative_raw_Pe_count"], 0)
        self.assertEqual(failure["selected_frame_negative_raw_Pi_count"], 1)
        self.assertAlmostEqual(
            failure["raw_evolved_Pi"], -5.799512988032478e-05
        )
        self.assertAlmostEqual(
            failure["replay_minus_stored_phi"], 5.7995129900123565e-05
        )
        self.assertIn("remains a failed result", self.protocol)

    def test_exact_runtime_pressure_contract_is_explicit(self) -> None:
        correction = self.manifest["source_contract_correction"]
        self.assertEqual(correction["density_floor"], 1e-7)
        self.assertEqual(
            correction["soft_floor_formula"],
            "softFloor(N,f) = max(N,0) + f * exp(-max(N,0)/f)",
        )
        self.assertEqual(
            correction["runtime_pressure_formula"],
            "P_runtime = N * max(P_evolved,0) / softFloor(N,1e-7)",
        )
        self.assertEqual(correction["ion_density"], "Ni = Ne")
        self.assertEqual(
            correction["pressure_correction"],
            "Pi_hat = Pi_runtime - Pe_runtime / 3672",
        )
        self.assertTrue(correction["raw_evolved_pressure_retained_for_transport"])
        self.assertIn(r"P_s^{\mathrm{runtime}}", self.protocol)

    def test_source_and_reused_canonical_artifacts_are_locked(self) -> None:
        for key, value in self.manifest["source_locks"].items():
            if key.endswith("sha256"):
                self.assertRegex(value, SHA256, key)
        reuse = self.manifest["canonical_reuse"]
        self.assertEqual(reuse["producer_job_id"], 6_892_446)
        self.assertRegex(reuse["canonical_sha256"], SHA256)
        self.assertRegex(reuse["extraction_record_sha256"], SHA256)
        self.assertTrue(reuse["read_only"])
        self.assertFalse(reuse["predecessor_directory_may_be_modified"])

    def test_every_nonpressure_setting_and_both_gates_remain_frozen(self) -> None:
        unchanged = self.manifest["unchanged_experiment"]
        self.assertEqual(unchanged["frame_indices"], [0, 156, 312, 467, 623])
        self.assertEqual(unchanged["native_z_samples"], 81)
        self.assertEqual(unchanged["zperiod"], 5)
        self.assertEqual(
            (unchanged["mpi_ranks"], unchanged["NXPE"], unchanged["NYPE"]),
            (4, 1, 4),
        )
        self.assertEqual(unchanged["coefficient_C"], "2 / Bxy^2")
        self.assertFalse(unchanged["instantaneous_arm_is_iterated_neumann"])
        pressure_gate = self.manifest["runtime_pressure_gate"]
        self.assertEqual(pressure_gate["continuous_atol"], 1e-12)
        self.assertEqual(pressure_gate["continuous_rtol"], 1e-12)
        source_gate = self.manifest["source_reconstruction_gate"]
        self.assertEqual(source_gate["continuous_atol"], 5e-10)
        self.assertEqual(source_gate["continuous_rtol"], 5e-10)
        self.assertTrue(source_gate["constant_shift_aligned_error_is_diagnostic_only"])

    def test_decisions_fail_closed_without_authorizing_model_work(self) -> None:
        decisions = self.manifest["decision_rules"]
        self.assertEqual(decisions["runtime_pressure_failure_action"], "stop")
        self.assertEqual(
            decisions["source_reconstruction_failure_action"],
            "stop_without_tolerance_change",
        )
        self.assertFalse(decisions["automatic_model_state_change_authorized"])
        self.assertFalse(decisions["automatic_training_authorized"])
        self.assertFalse(decisions["automatic_held_out_access_authorized"])
        paired = self.manifest["paired_effect_policy"]
        self.assertFalse(paired["posthoc_materiality_threshold_allowed"])
        self.assertFalse(paired["selected_frames_establish_all_frame_stability"])


if __name__ == "__main__":
    unittest.main()
