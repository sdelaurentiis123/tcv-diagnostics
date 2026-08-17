"""Static gates for the source-matched E6B elliptic oracle."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = (
    ROOT
    / "paper0/oracles/matched_e6b_elliptic"
    / "matched_e6b_elliptic_oracle.cxx"
)
CMAKE = ROOT / "paper0/oracles/matched_e6b_elliptic/CMakeLists.txt"
RECONSTRUCTION_TOOL = (
    ROOT / "paper0/tools/evaluate_matched_codec_reconstruction.py"
)


class TestMatchedE6BEllipticOracle(unittest.TestCase):
    def test_candidate_path_uses_only_e6b_fields_and_boundary(self) -> None:
        source = ORACLE.read_text(encoding="utf-8")
        self.assertIn('for (const char* field : {"Ne", "Pe", "Pi", "Vort"})', source)
        self.assertIn('return truth_layout ? field : "candidate/" + field;', source)
        self.assertIn('return truth_layout ? "saved_midpoint" : "boundary/Bphi";', source)
        self.assertIn("Field3D boundary_only_seed{0.0};", source)
        self.assertIn("set_radial_phi_ghosts(boundary_only_seed", source)
        self.assertNotIn('input.load_field("phi", position)', source)

    def test_oracle_retains_source_pressure_and_cyclic_solve(self) -> None:
        source = ORACLE.read_text(encoding="utf-8")
        self.assertIn("PRESSURE_DENSITY_FLOOR = 1.0e-7", source)
        self.assertIn("std::max(evolved_pressure(x, y, z), 0.0)", source)
        self.assertIn("runtime_pi - runtime_pe / ELECTRON_PRESSURE_DENOMINATOR", source)
        self.assertIn("solver->setCoefC(2.0 / SQ", source)
        self.assertIn("solver->setInnerBoundaryFlags(INVERT_SET)", source)
        self.assertIn("solver.solve(vorticity * (bsq / 2.0)", source)
        self.assertNotIn("FV::Div_a_Grad_perp", source)

    def test_build_and_interchange_format_are_explicit(self) -> None:
        cmake = CMAKE.read_text(encoding="utf-8")
        self.assertIn("find_package(HDF5 REQUIRED COMPONENTS C)", cmake)
        self.assertIn("HDF5::HDF5", cmake)
        tool = RECONSTRUCTION_TOOL.read_text(encoding="utf-8")
        self.assertIn('compression="gzip"', tool)
        self.assertNotIn('compression="lzf"', tool)


if __name__ == "__main__":
    unittest.main()
