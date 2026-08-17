from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_85604_phi_boundary_state.sbatch"


class PhiBoundaryLauncherTests(unittest.TestCase):
    def test_launcher_is_rusty_cpu_only_clean_locked_and_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--partition=preempt",
            "--qos=preempt",
            "--cpus-per-task=1",
            "--mem=8G",
            "--no-requeue",
            "VERSION_ID%%.*",
            "Refusing to overwrite",
            "status --porcelain --untracked-files=all",
            "phi_boundary_state_audit.json",
            "artifact_sha256.txt",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_launcher_locks_protocol_code_source_and_data_controls(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for digest in (
            "34462a32af60d8f683da49435803e12ffcb5e8bfcfc07a472b3866c6bad8ffb3",
            "1a1f5f099d380997413792132dfe493b647e081d67ef9eaeabf28eed5e0fd9b6",
            "c2264c7acc5765078192336f4de9a5542bbc23d174580b16d798ca5b95a82cc1",
            "c78bc34b82e225a32cb0c0764d17304c1c8e21cae58f0987d8a82c9d55869f99",
            "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a",
            "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded",
            "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad",
            "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57",
            "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
        ):
            self.assertIn(digest, text)


if __name__ == "__main__":
    unittest.main()
