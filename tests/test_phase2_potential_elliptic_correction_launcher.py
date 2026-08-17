from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_potential_elliptic_runtime_pressure.sbatch"
LOCKED_LOCAL_FILES = (
    ROOT
    / "paper0/manifests/phase2_potential_elliptic_runtime_pressure_correction.json",
    ROOT
    / "paper0/protocol/PHASE2_POTENTIAL_ELLIPTIC_RUNTIME_PRESSURE_CORRECTION.md",
    ROOT / "paper0/manifests/phase2_potential_elliptic_85604.json",
    ROOT / "paper0/protocol/PHASE2_POTENTIAL_ELLIPTIC_PROTOCOL.md",
    ROOT / "paper0/results/phase2_potential_elliptic_6892446.json",
    ROOT / "paper0/oracles/potential_elliptic/potential_elliptic_oracle.cxx",
    ROOT / "paper0/oracles/potential_elliptic_runtime_pressure/CMakeLists.txt",
    ROOT / "paper0/oracles/potential_elliptic/BOUT.inp",
    ROOT / "paper0/tools/compare_potential_elliptic_oracle.py",
    ROOT / "paper0/tools/compare_potential_elliptic_runtime_pressure_oracle.py",
)


class PotentialEllipticCorrectionLauncherTests(unittest.TestCase):
    def test_launcher_is_cpu_only_short_rocky9_clean_and_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--partition=gen",
            "--qos=gen",
            "--ntasks=4",
            "--cpus-per-task=2",
            "--time=00:20:00",
            "--no-requeue",
            "VERSION_ID%%.*",
            "status --porcelain --untracked-files=all",
            "Refusing to overwrite",
            "potential_elliptic_runtime_pressure_comparison.json",
            "base_potential_elliptic_comparison.json",
            "artifact_sha256.txt",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_launcher_reuses_predecessor_inputs_read_only(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("phase2_potential_elliptic/job_6892446", text)
        self.assertIn('${PREDECESSOR_DIR}/canonical_potential_inputs.nc', text)
        self.assertIn(
            "e090b3a23fa6eedf8c37e74421c08bafd3eb513039fa7621b5d612a7e1cbba3e",
            text,
        )
        self.assertIn(
            "e30e4f14dddfdff369387f9e8657b31a3e56bb4dc628c1e9ef5d86bd5bfd68be",
            text,
        )
        self.assertNotIn("EXTRACTOR=", text)
        self.assertNotIn('cp "${CANONICAL}"', text)
        self.assertIn("phase2_potential_elliptic_runtime_pressure", text)

    def test_launcher_hash_locks_every_local_dependency(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for path in LOCKED_LOCAL_FILES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn(digest, text, str(path))

    def test_launcher_locks_corrected_hermes_sources_and_exact_abi(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "d3abc5e32cdad3ea9c42faf432dcaed465070ee5792cb62e550b9baaad9953e6",
            "5e3b8055ae068481319a90329a3bdf3605b82b9024ee8273b210fd2be915bd85",
            "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a",
            "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded",
            "7d28d67c3f12c24ec281c0982e870f5369c65a6f",
            "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333",
            "hdf5/1.12.3",
            "netcdf-c/4.9.2",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
