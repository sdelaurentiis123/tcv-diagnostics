from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_potential_vorticity_all_frame.sbatch"
LOCAL_LOCKS = (
    ROOT / "paper0/manifests/phase2_potential_vorticity_all_frame_85604.json",
    ROOT / "paper0/protocol/PHASE2_POTENTIAL_VORTICITY_ALL_FRAME_PROTOCOL.md",
    ROOT / "paper0/tools/extract_potential_vorticity_all_frame_85604.py",
    ROOT
    / "paper0/oracles/potential_vorticity_all_frame/potential_vorticity_all_frame_oracle.cxx",
    ROOT / "paper0/oracles/potential_vorticity_all_frame/CMakeLists.txt",
    ROOT / "paper0/tools/compare_potential_vorticity_all_frame_shard.py",
    ROOT / "paper0/tools/merge_potential_vorticity_all_frame_shards.py",
)
LAUNCHER_SHA256 = (
    "f8672e4b32aa79f8d263b903539c4c6e9aa408286bf9063cd9000e2bf545deea"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PotentialVorticityAllFrameLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_is_hash_locked_and_shell_syntax_is_valid(self) -> None:
        self.assertEqual(sha256(LAUNCHER), LAUNCHER_SHA256)
        completed = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_job_is_cpu_only_rocky9_and_resource_bounded(self) -> None:
        self.assertIn("#SBATCH --partition=gen", self.source)
        self.assertIn("#SBATCH --qos=gen", self.source)
        self.assertIn("#SBATCH --ntasks=4", self.source)
        self.assertIn("#SBATCH --mem=64G", self.source)
        self.assertIn("#SBATCH --time=01:00:00", self.source)
        self.assertIn("#SBATCH --no-requeue", self.source)
        self.assertNotIn("#SBATCH --gres", self.source)
        self.assertNotIn("#SBATCH --gpus", self.source)
        self.assertIn('VERSION_ID%%.*}" != "9"', self.source)
        self.assertIn("export OMP_NUM_THREADS=1", self.source)

    def test_execution_is_sequential_and_extracts_once(self) -> None:
        self.assertIn("for ((shard = 0; shard < 8; shard++)); do", self.source)
        self.assertIn("start=$((shard * 78))", self.source)
        self.assertNotRegex(self.source, re.compile(r"srun[^\n]*&\s*$", re.MULTILINE))
        self.assertIn("rank_files_traversed_once", (
            ROOT / "paper0/tools/extract_potential_vorticity_all_frame_85604.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("Shard comparator statuses", self.source)

    def test_refuses_ambiguous_or_overwriting_execution(self) -> None:
        self.assertIn("PAPER0_EXPECTED_COMMIT", self.source)
        self.assertIn("Paper 0 commit mismatch", self.source)
        self.assertIn("worktree is dirty", self.source)
        self.assertIn("Refusing to overwrite existing result directory", self.source)
        self.assertIn("Refusing to overwrite existing shard directory", self.source)
        self.assertNotIn("git reset", self.source)
        self.assertNotIn("git checkout", self.source)
        self.assertNotIn("rm -", self.source)

    def test_every_local_scientific_dependency_is_hash_locked(self) -> None:
        for path in LOCAL_LOCKS:
            with self.subTest(path=path):
                self.assertIn(sha256(path), self.source)

    def test_scope_excludes_training_and_held_out_data(self) -> None:
        self.assertIn("tcv-fresh-proj/85604", self.source)
        self.assertNotIn("85606", self.source)
        self.assertNotIn("train.py", self.source)
        self.assertNotIn("torchrun", self.source)
        self.assertIn("--atol 5e-10", self.source)
        self.assertIn("--rtol 5e-10", self.source)
        self.assertIn("artifact_sha256.txt", self.source)


if __name__ == "__main__":
    unittest.main()
