"""Static gates for the Rocky 9 deterministic-model CPU launcher."""

from __future__ import annotations

import hashlib
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_model_cpu_gate.sbatch"


class TestModelCpuGateLauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text()

    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_is_short_cpu_only_rocky9_job(self) -> None:
        self.assertIn("#SBATCH --partition=gen", self.source)
        self.assertIn("#SBATCH --time=00:10:00", self.source)
        self.assertNotIn("#SBATCH --gres", self.source)
        self.assertIn('"${VERSION_ID%%.*}" != "9"', self.source)

    def test_requires_clean_exact_commit(self) -> None:
        self.assertIn("PAPER0_EXPECTED_COMMIT", self.source)
        self.assertIn("status --porcelain --untracked-files=all", self.source)
        self.assertIn("refusing provenance-ambiguous execution", self.source)

    def test_hash_locks_implementation_and_protocol(self) -> None:
        locked = {
            "paper0/manifests/phase2_matched_o1_o2_85604.json":
                "6cbbf991e311565ab2cc2d3ae0eb4f1d2572f4c23fb804723c990d5b2401c562",
            "paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md":
                "2148880b998f71b309e553d0712170b2f25a0d5c69c204ea1a089a1407a03de4",
            "src/tcv_diagnostics/models/layers.py":
                "87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b",
            "src/tcv_diagnostics/models/dcae.py":
                "0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554",
            "src/tcv_diagnostics/models/LOLA_LICENSE.txt":
                "6a483108d787c61c7e2306216ecaa2e15f80a0a2e7fb44cb70d200bbbab63605",
            "tests/test_phase2_dcae.py":
                "d41b174a874da8ffef31f872db3d713d8c0613fa43ece74cf2de2374cf71bc3a",
        }
        for relative, expected in locked.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)
            self.assertIn(expected, self.source)

    def test_cannot_train_or_read_simulation_data(self) -> None:
        self.assertNotIn("phase2_model_dataset/job_", self.source)
        self.assertNotIn("optimizer", self.source.lower())
        self.assertNotIn("backward(", self.source)
        self.assertIn('"simulation_data_read": False', self.source)
        self.assertIn('"training_performed": False', self.source)
        self.assertIn('"held_out_85606_read": False', self.source)

    def test_runs_complete_pytest_suite_without_cache(self) -> None:
        self.assertIn("-m pytest -p no:cacheprovider -q", self.source)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", self.source)


if __name__ == "__main__":
    unittest.main()
