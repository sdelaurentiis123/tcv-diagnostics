"""Static safety locks for the full R1 codec launcher."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o1_train_codecs.sbatch"


class TestO1TrainCodecsLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text()

    def test_rocky9_nonpreemptible_four_h100_request_is_explicit(self) -> None:
        self.assertIn("#SBATCH --partition=gpuxl", self.source)
        self.assertIn("#SBATCH --qos=gpuxl", self.source)
        self.assertIn("#SBATCH --gres=gpu:h100:4", self.source)
        self.assertIn("#SBATCH --constraint=h100", self.source)
        self.assertIn("#SBATCH --no-requeue", self.source)
        self.assertNotIn("#SBATCH --partition=gpupreempt", self.source)
        self.assertIn('"${VERSION_ID%%.*}" != "9"', self.source)

    def test_six_runs_execute_in_two_frozen_waves(self) -> None:
        self.assertIn("families=(c5p e6b c5p e6b c5p e6b)", self.source)
        self.assertIn("seeds=(1701 1701 1702 1702 1703 1703)", self.source)
        self.assertIn("wait_wave 0 1 2 3", self.source)
        self.assertIn("wait_wave 4 5", self.source)
        self.assertIn("--mode full", self.source)
        self.assertIn("--codec dcae_l20", self.source)

    def test_data_are_staged_and_every_shard_is_verified(self) -> None:
        self.assertIn("MODEL_DATA_STAGED", self.source)
        self.assertIn("SLURM_TMPDIR", self.source)
        self.assertIn('verified_shards=$((verified_shards + 1))', self.source)
        self.assertIn('"${verified_shards}" -ne 8', self.source)
        self.assertIn("artifact_sha256.txt", self.source)

    def test_wandb_is_required_for_every_run(self) -> None:
        self.assertIn("export WANDB_MODE=online", self.source)
        self.assertIn('readonly WANDB_PROJECT="tcv-diagnostics-paper0"', self.source)
        self.assertIn('readonly WANDB_GROUP="o1-dcae-l20-r1"', self.source)
        self.assertGreaterEqual(self.source.count("--wandb-project"), 2)
        self.assertGreaterEqual(self.source.count("--wandb-run-id"), 2)
        self.assertIn('tracking["remote_state_after_finish"] != "finished"', self.source)
        self.assertIn('tracking["epochs_logged"] != 200', self.source)

    def test_training_cannot_open_held_out_or_claim_o1_acceptance(self) -> None:
        self.assertNotIn("85606/data", self.source)
        self.assertIn('"held_out_85606_read": False', self.source)
        self.assertIn('"training_result_accepted": False', self.source)
        self.assertIn('"O1_scientific_evaluation_completed": False', self.source)


if __name__ == "__main__":
    unittest.main()
