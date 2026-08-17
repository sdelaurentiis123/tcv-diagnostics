from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_85604_state_completeness.sbatch"


class StateCompletenessLauncherTests(unittest.TestCase):
    def test_launcher_is_cpu_only_clean_locked_and_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--partition=preempt",
            "--qos=preempt",
            "--cpus-per-task=16",
            "--mem=64G",
            "--no-requeue",
            "SHARD_COUNT=16",
            "--exclusive",
            "--exact",
            "--mem=4G",
            "VERSION_ID%%.*",
            "Refusing to overwrite",
            "status --porcelain --untracked-files=all",
            "state_completeness_audit.json",
            "artifact_sha256.txt",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_launcher_locks_protocol_code_source_and_data_controls(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for digest in (
            "8ce098ddc4f75a6c81d2d74c6e4190884c2f4eb1273436e522fff0326e86bdc5",
            "d134f46ce517457299fece0e978d8ed032283529b69dd2facf3a7634cda9da22",
            "200c2fd96a788eec16e9e3c495e310f8c0af36376656fbd20ea5207f4b46030b",
            "d049f33ab42e97229ca6689bf482223686278d3c84a5effaa06135446952f6b1",
            "579197834948fbd60d17fb10feca71c35a83fe433cd310e691928fb4a8aa5aeb",
            "83d56e8335aeb3c7595f26b1ecbbed7a9ad016bb7bc4df3280a7b752ba6061ae",
            "5e3b8055ae068481319a90329a3bdf3605b82b9024ee8273b210fd2be915bd85",
            "c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad",
            "57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57",
            "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
        ):
            self.assertIn(digest, text)


if __name__ == "__main__":
    unittest.main()
