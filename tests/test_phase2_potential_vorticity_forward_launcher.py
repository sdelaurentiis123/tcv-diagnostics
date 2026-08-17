from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_potential_vorticity_forward.sbatch"
MANIFEST = ROOT / "paper0/manifests/phase2_potential_vorticity_forward_85604.json"
PROTOCOL = (
    ROOT / "paper0/protocol/PHASE2_POTENTIAL_VORTICITY_FORWARD_PROTOCOL.md"
)
DRIVER = (
    ROOT
    / "paper0/oracles/potential_vorticity_forward"
    / "potential_vorticity_forward_oracle.cxx"
)
CMAKE = ROOT / "paper0/oracles/potential_vorticity_forward/CMakeLists.txt"
COMPARATOR = ROOT / "paper0/tools/compare_potential_vorticity_forward_oracle.py"
ACCEPTED_INVERSE_RESULT = (
    ROOT / "paper0/results/phase2_potential_elliptic_runtime_pressure_6892641.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PotentialVorticityForwardLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_is_short_cpu_only_rocky9_job(self) -> None:
        self.assertIn("#SBATCH --partition=gen", self.source)
        self.assertIn("#SBATCH --qos=gen", self.source)
        self.assertIn("#SBATCH --ntasks=4", self.source)
        self.assertIn("#SBATCH --time=00:20:00", self.source)
        self.assertIn("#SBATCH --no-requeue", self.source)
        self.assertNotIn("#SBATCH --gres", self.source)
        self.assertNotIn("#SBATCH --gpus", self.source)
        self.assertIn('VERSION_ID%%.*}" != "9"', self.source)
        self.assertIn("srun --ntasks=4", self.source)
        self.assertIn("export OMP_NUM_THREADS=1", self.source)

    def test_refuses_ambiguous_or_overwriting_execution(self) -> None:
        self.assertIn("PAPER0_EXPECTED_COMMIT", self.source)
        self.assertIn("Paper 0 commit mismatch", self.source)
        self.assertIn("worktree is dirty", self.source)
        self.assertIn("Refusing to overwrite existing result directory", self.source)
        self.assertIn("set -euo pipefail", self.source)
        self.assertNotIn("git reset", self.source)
        self.assertNotIn("git checkout", self.source)
        self.assertNotIn("rm -", self.source)

    def test_uses_only_frozen_85604_inputs_and_no_training(self) -> None:
        self.assertIn("tcv_85604_adjusted.nc", self.source)
        self.assertNotIn("85606", self.source)
        self.assertNotIn("train.py", self.source)
        self.assertNotIn("torchrun", self.source)
        self.assertIn("--atol 5e-10", self.source)
        self.assertIn("--rtol 5e-10", self.source)
        self.assertIn("phase2_potential_vorticity_forward_85604.json", self.source)
        self.assertIn("PHASE2_POTENTIAL_VORTICITY_FORWARD_PROTOCOL.md", self.source)

    def test_locks_every_local_scientific_input_by_hash(self) -> None:
        for path in (
            MANIFEST,
            PROTOCOL,
            DRIVER,
            CMAKE,
            COMPARATOR,
            ACCEPTED_INVERSE_RESULT,
        ):
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                self.assertIn(sha256(path), self.source)

    def test_keeps_accepted_inverse_result_read_only(self) -> None:
        self.assertIn("ACCEPTED_INVERSE_JOB", self.source)
        self.assertIn("Missing immutable accepted inverse job", self.source)
        self.assertIn(
            '${ACCEPTED_INVERSE_JOB}/potential_elliptic_runtime_pressure_comparison.json',
            self.source,
        )
        self.assertNotIn('cp "${ACCEPTED_INVERSE_JOB}', self.source)

    def test_requires_four_rank_outputs_and_hashes_artifacts(self) -> None:
        self.assertIn("Expected four BOUT++ rank outputs", self.source)
        self.assertIn("artifact_sha256.txt", self.source)
        self.assertIn('exit "${COMPARISON_STATUS}"', self.source)


if __name__ == "__main__":
    unittest.main()
