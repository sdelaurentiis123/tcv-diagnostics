"""Static gates for the bounded Rocky 9 O1 codec GPU smoke."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o1_codec_gpu_smoke.sbatch"


class TestO1CodecGpuSmokeLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text()

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_is_bounded_h100_or_h200_rocky9_job(self) -> None:
        self.assertIn("#SBATCH --time=00:45:00", self.source)
        self.assertIn("#SBATCH --gres=gpu:1", self.source)
        self.assertIn("#SBATCH --constraint=h100|h200", self.source)
        self.assertIn('"${VERSION_ID%%.*}" != "9"', self.source)
        self.assertNotIn("--mode full", self.source)
        self.assertEqual(self.source.count("--mode smoke"), 2)

    def test_both_state_families_use_one_frozen_smoke_configuration(self) -> None:
        self.assertIn("for family in c5p e6b", self.source)
        self.assertIn("--codec dcae_l20", self.source)
        self.assertIn("--seed 1701", self.source)
        self.assertIn('config["epochs"] > 2', self.source)
        self.assertIn("> 16", self.source)

    def test_requires_clean_exact_commit(self) -> None:
        self.assertIn("PAPER0_EXPECTED_COMMIT", self.source)
        self.assertIn("status --porcelain --untracked-files=all", self.source)
        self.assertIn("refusing provenance-ambiguous execution", self.source)

    def test_historical_launcher_retains_its_executed_source_hashes(self) -> None:
        locked = {
            "src/tcv_diagnostics/models/layers.py":
                "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b",
            "src/tcv_diagnostics/models/dcae.py":
                "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554",
            "src/tcv_diagnostics/model_training_data.py":
                "e4cf03475e37e808acc21beead82fa7c0849b2857f6bac16f93538611a261ed1",
            "src/tcv_diagnostics/codec_training.py":
                "727408e524a1c1b9fbe644327e6fa5ec096e6b7eea4373bc4ee7c6e4d5546f7a",
            "paper0/tools/train_codec.py":
                "ddea107b6d6ed79557e05aa55cb63069441e36fa9510e10ae63673ec23c1607d",
        }
        # This launcher already ran at commit b2465d5.  Its recorded hashes must
        # remain immutable even when the current source legitimately advances.
        for relative, expected in locked.items():
            self.assertIn(expected, self.source, relative)

    def test_summary_cannot_be_mistaken_for_scientific_result(self) -> None:
        self.assertIn('"training_result_accepted": False', self.source)
        self.assertIn('"O1_scientific_gate_evaluated": False', self.source)
        self.assertIn('"held_out_85606_read": False', self.source)
        self.assertIn("not an accepted training result", self.source)


if __name__ == "__main__":
    unittest.main()
