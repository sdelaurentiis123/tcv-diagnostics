from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_potential_elliptic_oracle.sbatch"
LOCKED_LOCAL_FILES = (
    ROOT / "paper0/manifests/phase2_potential_elliptic_85604.json",
    ROOT / "paper0/protocol/PHASE2_POTENTIAL_ELLIPTIC_PROTOCOL.md",
    ROOT / "paper0/tools/extract_potential_elliptic_85604_frames.py",
    ROOT / "paper0/tools/compare_potential_elliptic_oracle.py",
    ROOT / "paper0/oracles/potential_elliptic/BOUT.inp",
    ROOT / "paper0/oracles/potential_elliptic/CMakeLists.txt",
    ROOT / "paper0/oracles/potential_elliptic/potential_elliptic_oracle.cxx",
    ROOT / "paper0/tools/extract_native_85604_frames.py",
    ROOT / "paper0/tools/audit_85604_phi_boundary_state.py",
    ROOT / "src/tcv_diagnostics/phi_boundary.py",
    ROOT / "paper0/tools/compare_hermes_radial_flow_oracle.py",
    ROOT / "paper0/tools/compare_shifted_ddy_oracle.py",
    ROOT / "src/tcv_diagnostics/codec_transport.py",
    ROOT / "src/tcv_diagnostics/geometry.py",
    ROOT / "src/tcv_diagnostics/transport.py",
    ROOT / "paper0/manifests/phase2_native_frame_oracle.json",
    ROOT / "paper0/results/phase2_phi_boundary_state_6891890.json",
)


class PotentialEllipticLauncherTests(unittest.TestCase):
    def test_launcher_is_cpu_only_rocky9_clean_locked_and_syntax_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "--partition=gen",
            "--qos=inter",
            "--ntasks=4",
            "--cpus-per-task=2",
            "--mem=32G",
            "--time=00:50:00",
            "--no-requeue",
            "VERSION_ID%%.*",
            "Refusing to overwrite",
            "status --porcelain --untracked-files=all",
            "potential_elliptic_comparison.json",
            "comparison_arrays.npz",
            "artifact_sha256.txt",
            "--atol 5e-10",
            "--rtol 5e-10",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_launcher_hash_locks_every_local_dependency(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for path in LOCKED_LOCAL_FILES:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn(digest, text, str(path))

    def test_launcher_locks_exact_external_source_and_abi(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for required in (
            "920ba829cc78cdab0dbf6101c69fecc4689bd8dd",
            "7d28d67c3f12c24ec281c0982e870f5369c65a6f",
            "843ea048baa90b4a55ead6aa6ed546c6746e81c2718a8ade3ed977381364f65a",
            "7643af10d23ce4efebf58b835fbb27cb8a7d7ed11545d450182916013f3a002a",
            "276a90682298e2b20594fc52b241518f197aa1a3310ada47fb9f7852b3deaded",
            "858752dd39c55caf7afd00fa84583b5e1c7944763f6497f6ffaa2530ba0e1fc3",
            "47c2cbe674c61e510cc96835823ba8a7b55f2a9b6c2a5c1f93d730c8dba5d157",
            "568faa3c9bcf17732779868f28aa219f5e8beea2ad4f98c490b233ce9552d38a",
            "1309f7eebf5fe076663224b1a66a08b1a3ae6ce7526d0f0b595d6537db5bf296",
            "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333",
            "hdf5/1.12.3",
            "netcdf-c/4.9.2",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
