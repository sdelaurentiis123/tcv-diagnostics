from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_85604_resampling_audit.sbatch"
MANIFEST = ROOT / "paper0" / "manifests" / "phase2_85604_resampling_sensitivity.json"
AUDITOR = ROOT / "paper0" / "tools" / "audit_85604_resampling.py"
MERGER = ROOT / "paper0" / "tools" / "merge_85604_resampling_shards.py"
RESAMPLING = ROOT / "src" / "tcv_diagnostics" / "resampling.py"
TRANSPORT = ROOT / "src" / "tcv_diagnostics" / "transport.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            sha256(MANIFEST),
            sha256(AUDITOR),
            sha256(MERGER),
            sha256(RESAMPLING),
            sha256(TRANSPORT),
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
