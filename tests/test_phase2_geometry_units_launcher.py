from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_85604_geometry_units.sbatch"
AUDITOR = ROOT / "paper0" / "tools" / "audit_85604_geometry_units.py"

FROZEN_HASHES = (
    "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460",
    "ef26a2f88a73826259be95fbe0f24cadece4dbf04a28e0eb116159d7afd7c478",
    "4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c",
    "b788ca25d2aaa7991120e84be4235250f6331bb35710b3c76fd8a35f28584518",
    "0a7540a3e1b9698dea2293e729fd1f956986924cb3be6cc15a546add98521b64",
    "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
    "3c4a3d8f5b94ab728650726fbf010af70f63ae6452a83e024460d34ab99336e3",
    "4a89ceb00a66799668b1b73d3598e2995d9e171680be0d5ce0d20fe6b33e63b2",
    "458eeecbd6da1afb882d0de2b652271fc2c2ca142c39a636a52f3adc5c16ef3f",
    "3c766083078ec17d737a7ac595868adf1706e0596a9e614bb3ac73f071c1834d",
)


class GeometryUnitsLauncherTests(unittest.TestCase):
    def test_launcher_is_clean_locked_cpu_only_and_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--partition=gen",
            "--qos=gen",
            "--cpus-per-task=1",
            "--mem=8G",
            "--no-requeue",
            "Refusing to overwrite",
            "status --porcelain --untracked-files=all",
            "VERSION_ID%%.*",
            "geometry_units.json",
            "artifact_sha256.txt",
            *FROZEN_HASHES,
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_auditor_is_parseable_and_writes_failure_artifact(self) -> None:
        source = AUDITOR.read_text(encoding="utf-8")
        ast.parse(source)
        for required in (
            "held_out_85606_accessed",
            "confined_separatrix_surface_mask",
            "integrate_radial_surface_flow",
            "hermes_transport_scales",
            "apply_memberwise",
            "refusing to overwrite existing output",
            '"status": "error"',
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
