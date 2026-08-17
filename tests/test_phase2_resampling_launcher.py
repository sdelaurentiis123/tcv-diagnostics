from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_85604_resampling_audit.sbatch"
FROZEN_INTERNAL_HASHES = (
    "875164259879621c3273a043b7fc529e8bef4875cf51b8c4d23764b95b00ce91",
    "8b36c516ad3002268261c169cc95b07a5f1e349386ea522c4e05b6066824f229",
    "f56ac1b366cc8a278032068cdc718ed604c9b79f01e920c3cb6d1b46e35a4c63",
    "12612b2cd65ac807ef4e55996712f6dde49dfca1b956f449ed189386eb5ea04e",
    "e61f007bb6268fdcd754bc975e1b0cb04133d471d86eda9d3bab01927fe8401e",
)


class ResamplingLauncherTests(unittest.TestCase):
    def test_launcher_is_clean_locked_cpu_only_and_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--partition=gen",
            "--qos=gen",
            "--cpus-per-task=17",
            "--mem=68G",
            "--no-requeue",
            "SHARD_COUNT=17",
            "--exclusive",
            "--exact",
            "--mem=4G",
            "Refusing to overwrite",
            "status --porcelain --untracked-files=all",
            *FROZEN_INTERNAL_HASHES,
            "resampling_sensitivity.json",
            "artifact_sha256.txt",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_launcher_locks_every_external_input_hash(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for digest in (
            "f446c0d6f068de0c4d190c77292467efa7312731343239082cf7a5920aa595a3",
            "843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777",
            "a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba",
            "f4aae5c13ecd944f51cec0c3539f57ff669cb0bb0405cb813c3f50ff6cb83817",
            "eed18a7f7a356a4f8d437647b73d4f8078a5309e1e8583a1089e622196ce4d43",
            "61dee30a92bb1a3eefcece80faa42d143271bfe200f4024b9747aeb06747bc21",
            "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
        ):
            self.assertIn(digest, text)


if __name__ == "__main__":
    unittest.main()
