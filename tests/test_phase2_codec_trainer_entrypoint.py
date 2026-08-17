"""Static scope locks for the O1 codec training entrypoint."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_codec.py"
TRAINING = ROOT / "src/tcv_diagnostics/codec_training.py"


class TestCodecTrainerEntrypoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entrypoint = ENTRYPOINT.read_text()
        cls.training = TRAINING.read_text()

    def test_cli_choices_are_frozen(self) -> None:
        self.assertIn('choices=("smoke", "full")', self.entrypoint)
        self.assertIn('choices=("dcae_l20", "dcae_l10")', self.entrypoint)
        self.assertIn('choices=("c5p", "e6b")', self.entrypoint)
        self.assertIn('choices=(1701, 1702, 1703)', self.entrypoint)

    def test_clean_exact_checkout_and_cuda_are_required(self) -> None:
        self.assertIn('"rev-parse", "HEAD"', self.entrypoint)
        self.assertIn('"--untracked-files=all"', self.entrypoint)
        self.assertIn("torch.cuda.is_available()", self.entrypoint)
        self.assertNotIn('torch.device("mps")', self.entrypoint)

    def test_only_volume_enters_o1_codec(self) -> None:
        self.assertIn('batch["volume"]', self.training)
        self.assertNotIn('batch["boundary"]', self.training)
        self.assertIn("equal_channel_mae(target, reconstruction)", self.training)
        self.assertNotIn("85606/data", self.entrypoint)

    def test_selected_checkpoint_uses_strict_earliest_improvement(self) -> None:
        self.assertIn("if validation_loss < selected_loss:", self.training)
        self.assertNotIn("if validation_loss <= selected_loss:", self.training)
        self.assertEqual(self.training.count("save_torch_atomic(\n        selected_path"), 1)
        self.assertIn('value.detach().to("cpu").clone()', self.training)
        self.assertIn("checkpoint reload changed model output", self.training)


if __name__ == "__main__":
    unittest.main()
