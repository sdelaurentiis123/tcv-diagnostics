from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper0" / "tools"
COMPARATOR = TOOLS / "compare_hermes_radial_flow_oracle.py"
LAUNCHER = ROOT / "cluster" / "phase2_hermes_radial_flow_oracle.sbatch"
ORACLE_DIR = ROOT / "paper0" / "oracles" / "hermes_radial_flow"
PROTOCOL = ROOT / "paper0" / "protocol" / "PHASE2_TRANSPORT_PROTOCOL.md"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("hermes_radial_comparator", COMPARATOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyVariable:
    def __init__(self, values: np.ndarray, dimensions: tuple[str, ...], name: str):
        self._values = values
        self.dimensions = dimensions
        self.name = name

    def __getitem__(self, item):
        return self._values[item]


class CompiledHermesRadialFlowOracleTests(unittest.TestCase):
    def test_launcher_is_cpu_only_source_locked_and_syntax_valid(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "920ba829cc78cdab0dbf6101c69fecc4689bd8dd",
            "458eeecbd6da1afb882d0de2b652271fc2c2ca142c39a636a52f3adc5c16ef3f",
            "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
            "hdf5/1.12.3",
            "--ntasks=4",
            "--atol 5e-10",
            "--rtol 5e-10",
            "Refusing to overwrite",
            "--no-requeue",
        ):
            self.assertIn(required, text)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_gpl_driver_exposes_components_sum_and_divergence(self) -> None:
        source = (ORACLE_DIR / "hermes_radial_flow_oracle.cxx").read_text(
            encoding="utf-8"
        )
        options = (ORACLE_DIR / "BOUT.inp").read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: GPL-3.0-or-later", source)
        self.assertIn("corner_plus - corner_minus", source)
        self.assertIn("mc_slope", source)
        self.assertIn("Field3D dfdy = DDY(phi);", source)
        self.assertIn("flow_xz + flow_xy", source)
        self.assertIn("coord->J(i, j) * coord->dx(i, j)", source)
        for output in (
            "xz_flow",
            "xy_flow",
            "total_radial_flow",
            "radial_divergence",
        ):
            self.assertIn(output, source)
        for case in MODULE.CASES:
            self.assertIn(f'"{case}"', source)
            self.assertIn(f"q_{case} =", options)
            self.assertIn(f"phi_{case} =", options)
        self.assertIn("zperiod = 5", options)

    def test_xy_reader_and_partition_assembly_preserve_rank_order(self) -> None:
        values = np.arange(12 * 3, dtype=np.float64).reshape(12, 3)
        variable = DummyVariable(values.T, ("y", "x"), "dz")
        np.testing.assert_array_equal(MODULE.canonical_xy(variable), values)
        partitions = []
        for pe_y in (2, 0, 3, 1):
            local = np.full((3, 12), -99.0)
            local[:, 2:-2] = float(pe_y)
            partitions.append((pe_y, local))
        assembled = MODULE.assemble_xy_partitions(partitions)
        self.assertEqual(assembled.shape, (3, 32))
        for pe_y in range(4):
            np.testing.assert_array_equal(
                assembled[:, pe_y * 8 : (pe_y + 1) * 8], float(pe_y)
            )

    def test_cell_regions_cover_separatrix_branches_and_sol(self) -> None:
        cells = np.arange(2, 62, dtype=np.int64)
        regions = MODULE.cell_regions(cells)
        self.assertEqual(regions["all_valid"].shape, (60, 32))
        separatrix_row = int(np.flatnonzero(cells == 16)[0])
        self.assertTrue(regions["separatrix_cell"][separatrix_row, 12])
        self.assertTrue(regions["inner_core_branch_connection"][0, 8])
        self.assertTrue(regions["inner_private_flux_connection"][0, 24])
        self.assertTrue(regions["open_sol_interior"][separatrix_row, 12])
        self.assertFalse(regions["all_valid"][0, 0])
        for name, mask in regions.items():
            self.assertGreater(int(np.count_nonzero(mask)), 0, name)

    def test_volume_weighted_conservation_metric_accepts_known_identity(self) -> None:
        rng = np.random.default_rng(17)
        face_flow = rng.normal(size=(4, 2, 3))
        jacobian = np.ones((6, 2)) * 2.0
        dx = np.ones((6, 2)) * 0.25
        cells = np.asarray([2, 3, 4])
        divergence = (face_flow[1:] - face_flow[:-1]) / (
            jacobian[cells] * dx[cells]
        )[..., None]
        mask = np.ones((3, 2), dtype=bool)
        accepted = MODULE.conservation_metrics(
            divergence, face_flow, jacobian, dx, cells, mask
        )
        self.assertTrue(accepted["passed"])
        broken = divergence.copy()
        broken[1, 0, 0] += 1.0e-3
        self.assertFalse(
            MODULE.conservation_metrics(
                broken, face_flow, jacobian, dx, cells, mask
            )["passed"]
        )

    def test_all_acceptance_constants_are_frozen_in_protocol(self) -> None:
        self.assertEqual(MODULE.DEFAULT_ATOL, 5.0e-10)
        self.assertEqual(MODULE.DEFAULT_RTOL, 5.0e-10)
        self.assertEqual(MODULE.CONSERVATION_ATOL, 5.0e-12)
        self.assertEqual(MODULE.CONSERVATION_RTOL, 5.0e-12)
        self.assertEqual(MODULE.DZ_ATOL, 1.0e-15)
        protocol = PROTOCOL.read_text(encoding="utf-8")
        for required in (
            "Combined radial flow and conservation",
            "left-cell indices `1:62`",
            "divergence uses cells `2:62`",
            "2*pi/(5*81)",
            "max_abs_error <= 5e-10 + 5e-10 * max_abs_reference",
            "max_abs_residual <= 5e-12 + 5e-12 * max_abs_face_difference",
            "both positive and negative `xz` flow",
        ):
            self.assertIn(required, protocol)


if __name__ == "__main__":
    unittest.main()
