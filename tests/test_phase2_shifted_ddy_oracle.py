from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster" / "phase2_shifted_ddy_oracle.sbatch"
COMPARATOR = ROOT / "paper0" / "tools" / "compare_shifted_ddy_oracle.py"
ORACLE_DIR = ROOT / "paper0" / "oracles" / "bout_shifted_ddy"

SPEC = importlib.util.spec_from_file_location("shifted_ddy_comparator", COMPARATOR)
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


class CompiledShiftedDdyOracleTests(unittest.TestCase):
    def test_launcher_is_cpu_only_clean_commit_locked_and_syntax_valid(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        for required in (
            "PAPER0_EXPECTED_COMMIT",
            "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
            "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333",
            "hdf5/1.12.3",
            "zperiod = 5",
            "--atol 5e-10",
            "--rtol 5e-10",
            "--ntasks=4",
            "NXPE = 1",
            "Refusing to overwrite",
            "--no-requeue",
        ):
            source = text + (ORACLE_DIR / "BOUT.inp").read_text(encoding="utf-8")
            self.assertIn(required, source)
        self.assertNotIn("--gres=gpu", text)
        self.assertNotIn("85606", text)

    def test_oracle_source_contains_all_frozen_cases_and_explicit_c2(self) -> None:
        source = (ORACLE_DIR / "shifted_ddy_oracle.cxx").read_text(
            encoding="utf-8"
        )
        for case in MODULE.CASES:
            self.assertIn(f'"{case}"', source)
        self.assertIn('DDY(input, CELL_CENTRE, "C2")', source)
        self.assertIn("mesh->communicate(input)", source)

    def test_canonical_reader_reorders_axes_and_selects_last_time(self) -> None:
        values = np.arange(2 * 81 * 4 * 3, dtype=np.float64).reshape(2, 81, 4, 3)
        variable = DummyVariable(values, ("t", "z", "x", "y"), "field")
        canonical = MODULE.canonical_xyz(variable)
        self.assertEqual(canonical.shape, (4, 3, 81))
        self.assertEqual(canonical[2, 1, 7], values[-1, 7, 2, 1])

    def test_guard_stripping_and_regions_are_unambiguous(self) -> None:
        values = np.zeros((68, 36, 81))
        self.assertEqual(MODULE.strip_bout_y_guards(values).shape, (68, 32, 81))
        with self.assertRaisesRegex(ValueError, "symmetrically"):
            MODULE.strip_bout_y_guards(np.zeros((68, 35, 81)))
        regions = MODULE.comparison_regions()
        self.assertTrue(regions["inner_private_flux_connection"][0, 7])
        self.assertTrue(regions["inner_core_branch_connection"][0, 23])
        self.assertTrue(regions["open_sol_interior"][16, 12])
        self.assertFalse(regions["all_valid"][3, 0])
        self.assertFalse(regions["all_valid"][3, 31])

    def test_rank_partitions_are_reassembled_by_explicit_y_index(self) -> None:
        partitions = []
        for pe_y in (2, 0, 3, 1):
            local = np.full((3, 12, 5), -99.0)
            local[:, 2:-2, :] = float(pe_y)
            partitions.append((pe_y, local))
        assembled = MODULE.assemble_y_partitions(partitions)
        self.assertEqual(assembled.shape, (3, 32, 5))
        for pe_y in range(4):
            np.testing.assert_array_equal(
                assembled[:, pe_y * 8 : (pe_y + 1) * 8, :], float(pe_y)
            )
        with self.assertRaisesRegex(ValueError, "exactly"):
            MODULE.assemble_y_partitions([(0, partitions[0][1])] * 4)

    def test_acceptance_rule_is_scale_aware_and_rejects_nonfinite(self) -> None:
        reference = np.ones((1, 1, 3)) * 2.0
        mask = np.ones((1, 1), dtype=bool)
        accepted = MODULE.error_metrics(
            reference + 1e-9,
            reference,
            mask,
            atol=5e-10,
            rtol=5e-10,
        )
        self.assertTrue(accepted["passed"])
        rejected = MODULE.error_metrics(
            np.asarray([[[np.nan, 2.0, 2.0]]]),
            reference,
            mask,
            atol=5e-10,
            rtol=5e-10,
        )
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["nonfinite_count"], 1)


if __name__ == "__main__":
    unittest.main()
