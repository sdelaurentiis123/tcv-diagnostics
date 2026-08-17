"""Prospective locks for the six full R1 codec training tasks."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from tcv_diagnostics.codec_training import CodecRunConfig


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/phase2_o1_codec_r1_runs.json"
SMOKE = ROOT / "paper0/results/phase2_o1_codec_gpu_smoke_6893745.json"


class TestO1CodecR1RunManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.smoke = json.loads(SMOKE.read_text())

    def test_current_checkpoint_regression_smoke_is_exact_and_passed(self) -> None:
        self.assertEqual(
            hashlib.sha256(SMOKE.read_bytes()).hexdigest(),
            "fc5a1509097a5b5a87f1c86dd25edab49bf2b7c386befdd52c70d3dc9ce41644",
        )
        self.assertEqual(
            self.smoke["paper0_commit"],
            "b2465d5b2480d489daf9425c8692c0ed46a70e91",
        )
        self.assertTrue(
            all(run["checkpoint_reload_bitwise_exact"] for run in self.smoke["runs"].values())
        )
        self.assertFalse(self.smoke["training_result_accepted"])

    def test_exact_family_seed_product_is_frozen(self) -> None:
        tasks = self.manifest["tasks"]
        self.assertEqual([task["array_index"] for task in tasks], list(range(6)))
        self.assertEqual(
            [(task["family"], task["seed"]) for task in tasks],
            [
                ("c5p", 1701),
                ("e6b", 1701),
                ("c5p", 1702),
                ("e6b", 1702),
                ("c5p", 1703),
                ("e6b", 1703),
            ],
        )

    def test_every_task_resolves_to_the_frozen_full_config(self) -> None:
        frozen = self.manifest["training"]
        for task in self.manifest["tasks"]:
            config = CodecRunConfig.frozen(
                mode="full",
                codec=self.manifest["model"]["codec"],
                family=task["family"],
                seed=task["seed"],
            )
            self.assertEqual(config.epochs, frozen["epochs"])
            self.assertEqual(len(config.train_frames), frozen["examples_per_epoch"])
            self.assertEqual(
                config.optimizer_steps_per_epoch,
                frozen["optimizer_steps_per_epoch"],
            )
            self.assertEqual(config.total_optimizer_steps, frozen["total_optimizer_steps"])

    def test_execution_is_nonpreemptible_and_concurrency_limited(self) -> None:
        execution = self.manifest["execution"]
        self.assertEqual(execution["os_major"], 9)
        self.assertEqual(execution["partition"], "gpuxl")
        self.assertEqual(execution["qos"], "gpuxl")
        self.assertEqual(execution["array_concurrency_cap"], 2)
        self.assertEqual(execution["gpu_per_task"], 1)

    def test_training_cannot_decide_o1_or_open_later_phases(self) -> None:
        self.assertFalse(self.manifest["held_out_85606_access_allowed"])
        self.assertFalse(self.manifest["model"]["physics_derived_loss_allowed"])
        post = self.manifest["post_training"]
        self.assertFalse(post["O1_scientific_evaluation_in_training_job"])
        self.assertFalse(post["R1_acceptance_from_training_loss_allowed"])
        self.assertFalse(post["R2_launch_allowed_before_complete_R1_O1_evaluation"])
        self.assertFalse(post["O2_launch_allowed"])


if __name__ == "__main__":
    unittest.main()
