from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "build_bout_transport_oracle.sbatch"


class TransportOracleBuildLauncherTests(unittest.TestCase):
    def test_launcher_is_syntax_valid_and_provenance_locked(self) -> None:
        text = LAUNCHER.read_text()
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "7d28d67c3f12c24ec281c0982e870f5369c65a6f",
            "a43d6d4d415d407712c246faca553bd951730dc1",
            "027f9aee2d34dbe1c98f26224e1fbe1654cb4aae",
            "407c905e45ad75fc29bf0f9bb7c5c2fd3475976f",
            "23cb94f027d4ef33bf48133acc2695c7e5c6f1e7",
            "761bf5b8488edb21feb8c512860bf2f2a9283e4d6d9c7200bc0f6b1598fafe56",
            "BOUT_ENABLE_MPI=ON",
            "BOUT_USE_NETCDF=ON",
            "BOUT_USE_FFTW=ON",
            "VERSION_ID%%.*",
            "--no-requeue",
            "Refusing to overwrite",
        ):
            self.assertIn(required, text)
        self.assertNotIn("85606", text)
        self.assertNotIn("--gres=gpu", text)


if __name__ == "__main__":
    unittest.main()
