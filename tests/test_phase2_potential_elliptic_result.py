from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "paper0/results/phase2_potential_elliptic_6892446.json"
NATIVE_RESULT_PATH = ROOT / "paper0/results/phase2_hermes_native_frames_6891379.json"
RESULT_SHA256 = (
    "71b9d2942f8ff943f412f86ee7c6fa729194c53a883a2df6c8ecc6894bb6917d"
)
EXECUTED_COMMIT = "47737c7f807aa16b74e556aaaef3fba45edec8ab"


class PotentialEllipticResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        cls.native = json.loads(NATIVE_RESULT_PATH.read_text(encoding="utf-8"))

    def test_tracked_result_is_the_verified_raw_artifact(self) -> None:
        digest = hashlib.sha256(RESULT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, RESULT_SHA256)

    def test_provenance_scope_and_nontraining_status_are_exact(self) -> None:
        result = self.result
        self.assertEqual(
            result["phase"], "phase2_potential_elliptic_85604_paired_oracle"
        )
        self.assertEqual(result["paper0_commit"], EXECUTED_COMMIT)
        self.assertEqual(result["slurm_job_id"], 6_892_446)
        self.assertEqual(result["development_run"], "85604")
        self.assertFalse(result["held_out_85606_read"])
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["native_z_samples"], 81)
        self.assertEqual(result["zperiod"], 5)
        self.assertEqual(result["frame_indices"], [0, 156, 312, 467, 623])
        self.assertEqual(
            result["artifacts"]["canonical_sha256"],
            "e090b3a23fa6eedf8c37e74421c08bafd3eb513039fa7621b5d612a7e1cbba3e",
        )

    def test_all_input_echoes_are_bitwise_exact(self) -> None:
        gate = self.result["source_reconstruction_gate"]
        self.assertTrue(gate["volume_input_echoes_passed"])
        self.assertTrue(gate["boundary_input_echoes_passed"])
        for family in ("volume_input_echoes", "boundary_input_echoes"):
            for frame in gate[family].values():
                for echo in frame.values():
                    self.assertTrue(echo["passed"])
                    self.assertEqual(echo["bitwise_mismatch_count"], 0)
                    self.assertEqual(echo["nonfinite_candidate_count"], 0)
                    self.assertEqual(echo["nonfinite_reference_count"], 0)

    def test_only_frame_312_full_domain_fails(self) -> None:
        reconstructions = self.result["source_reconstruction_gate"][
            "per_frame_reconstruction"
        ]
        self.assertEqual(
            {frame for frame, metrics in reconstructions.items() if not metrics["passed"]},
            {"f312"},
        )
        for frame, metrics in reconstructions.items():
            self.assertTrue(
                metrics["raw_guard_independent_transport_interior"]["passed"],
                frame,
            )
        failure = reconstructions["f312"]["raw_full_physical_domain"]
        self.assertFalse(failure["passed"])
        self.assertEqual(failure["maximum_location"], {"x": 6, "y": 31, "z": 73})
        self.assertAlmostEqual(
            failure["maximum_absolute_difference"], 5.7995129900123565e-05
        )
        self.assertEqual(
            reconstructions["f312"]["raw_guard_independent_transport_interior"][
                "maximum_absolute_difference"
            ],
            2.424727085781342e-13,
        )

    def test_failure_matches_previously_audited_negative_raw_pi(self) -> None:
        closure = self.native["five_channel_closure"]
        point = closure["failure"]
        self.assertEqual(point["frame_index"], 312)
        self.assertEqual(point["model_indices_xyz"], [6, 31, 73])
        negative_pi = point["Pi"]
        maximum = self.result["source_reconstruction_gate"][
            "per_frame_reconstruction"
        ]["f312"]["raw_full_physical_domain"]["maximum_absolute_difference"]
        self.assertAlmostEqual(maximum, -negative_pi, places=13)
        self.assertIn("floor(P, 0)", closure["source_backed_diagnosis"]["cause"])

    def test_paired_effect_remains_blocked_without_posthoc_relaxation(self) -> None:
        gate = self.result["source_reconstruction_gate"]
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["all_frame_reconstructions_passed"])
        self.assertEqual(gate["continuous_atol"], 5e-10)
        self.assertEqual(gate["continuous_rtol"], 5e-10)
        self.assertFalse(gate["constant_shift_alignment_can_change_gate"])
        effect = self.result["paired_boundary_effect"]
        self.assertEqual(effect["status"], "blocked_by_source_reconstruction_gate")
        self.assertIsNone(effect["potential"])
        self.assertIsNone(effect["transport"])
        self.assertFalse(effect["materiality_label_assigned"])
        self.assertFalse(self.result["decision"]["paired_effect_interpretable"])
        self.assertFalse(self.result["decision"]["automatic_state_change_authorized"])
        self.assertFalse(self.result["decision"]["automatic_training_authorized"])


if __name__ == "__main__":
    unittest.main()
