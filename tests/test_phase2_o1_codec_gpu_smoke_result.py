"""Regression locks for bounded GPU smoke job 6893713."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "paper0/results/phase2_o1_codec_gpu_smoke_6893713.json"
CHILDREN = {
    "c5p": ROOT / "paper0/results/phase2_o1_codec_gpu_smoke_6893713_c5p.json",
    "e6b": ROOT / "paper0/results/phase2_o1_codec_gpu_smoke_6893713_e6b.json",
}


class TestO1CodecGpuSmokeResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary_raw = SUMMARY.read_bytes()
        cls.summary = json.loads(cls.summary_raw)
        cls.children = {name: json.loads(path.read_text()) for name, path in CHILDREN.items()}

    def test_exact_result_digests(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.summary_raw).hexdigest(),
            "f16bb41c2ce03d9ae3ec0128d759748a594bd0fb8120a38a24825f373cb725db",
        )
        expected = {
            "c5p": "56c8e158cf19e927ab61c2530c9228285d9edeac55061cc33611b9da589839ef",
            "e6b": "e0ae356f3e56a70d23bc8118440a8a41de3247cc2bb7eb9ec9176925dc7c30c1",
        }
        for family, path in CHILDREN.items():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected[family])
            self.assertEqual(self.summary["runs"][family]["result_sha256"], actual)

    def test_execution_identity(self) -> None:
        self.assertEqual(self.summary["slurm_job_id"], "6893713")
        self.assertEqual(self.summary["rocky_major"], 9)
        self.assertEqual(
            self.summary["paper0_commit"],
            "c3d03289dd1e1e1a80eb03579febbb4255022a75",
        )
        self.assertEqual(self.summary["codec"], "dcae_l20")
        self.assertEqual(self.summary["seed"], 1701)

    def test_both_real_models_complete_bounded_optimizer_and_reload(self) -> None:
        expected_parameters = {"c5p": 123007365, "e6b": 123010822}
        for family, child in self.children.items():
            config = child["config"]
            self.assertEqual(child["completed_epochs"], 2)
            self.assertEqual(child["completed_optimizer_steps"], 2)
            self.assertLessEqual(config["train_stop"] - config["train_start"], 16)
            self.assertLessEqual(
                config["validation_stop"] - config["validation_start"], 16
            )
            self.assertEqual(child["parameter_count"], expected_parameters[family])
            self.assertTrue(child["checkpoint_reload_bitwise_exact"])
            self.assertGreater(child["peak_cuda_bytes"], 0)
            self.assertFalse(child["physics_derived_loss_used"])
            self.assertFalse(child["tf32_allowed"])
            self.assertEqual(child["development_run"], "85604")
            self.assertFalse(child["held_out_85606_read"])

    def test_smoke_scores_are_not_scientific_acceptance(self) -> None:
        self.assertEqual(
            self.summary["scope"], "bounded_non_scientific_O1_codec_gpu_smoke"
        )
        self.assertFalse(self.summary["training_result_accepted"])
        self.assertFalse(self.summary["O1_scientific_gate_evaluated"])
        self.assertFalse(self.summary["held_out_85606_read"])


if __name__ == "__main__":
    unittest.main()
