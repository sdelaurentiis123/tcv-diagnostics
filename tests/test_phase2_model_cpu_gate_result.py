"""Regression lock for the immutable Rocky 9 codec CPU-gate result."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase2_model_cpu_gate_6893674.json"


class TestModelCpuGateResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = RESULT.read_bytes()
        cls.result = json.loads(cls.raw)

    def test_exact_immutable_result_digest(self) -> None:
        self.assertEqual(
            hashlib.sha256(self.raw).hexdigest(),
            "d46bca572c52b8beab1f0bda18b3b869fe63ec6627b76382c350c52a972ed6f9",
        )

    def test_clean_rocky9_pass(self) -> None:
        self.assertEqual(self.result["status"], "passed")
        self.assertEqual(self.result["rocky_major"], 9)
        self.assertEqual(self.result["slurm_job_id"], "6893674")
        self.assertEqual(
            self.result["paper0_commit"],
            "0d633164050281160f37b79f33d4332c536e7970",
        )

    def test_no_scientific_or_compute_claim_leaks_from_cpu_gate(self) -> None:
        self.assertFalse(self.result["gpu_requested"])
        self.assertFalse(self.result["simulation_data_read"])
        self.assertFalse(self.result["training_performed"])
        self.assertFalse(self.result["held_out_85606_read"])
        self.assertEqual(
            self.result["scope"], "deterministic_codec_cpu_implementation_gate"
        )


if __name__ == "__main__":
    unittest.main()
