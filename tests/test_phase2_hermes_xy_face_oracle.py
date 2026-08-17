from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper0" / "tools"
COMPARATOR = TOOLS / "compare_hermes_xy_face_oracle.py"
LAUNCHER = ROOT / "cluster" / "phase2_hermes_xy_face_oracle.sbatch"
ORACLE_DIR = ROOT / "paper0" / "oracles" / "hermes_xy_face"
PROTOCOL = ROOT / "paper0" / "protocol" / "PHASE2_TRANSPORT_PROTOCOL.md"
RESULT = ROOT / "paper0" / "results" / "phase2_hermes_xy_face_6891343.json"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("hermes_xy_comparator", COMPARATOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompiledHermesXyFaceOracleTests(unittest.TestCase):
    def test_tracked_execution_passes_frozen_rule_without_scope_creep(
        self,
    ) -> None:
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(
            result["paper0_commit"],
            "ee2b04ff381466ae62054616f7e59410b868ed08",
        )
        self.assertEqual(result["slurm"]["state"], "COMPLETED")
        self.assertEqual(result["slurm"]["exit_code"], "0:0")
        self.assertTrue(result["acceptance"]["overall_passed"])
        self.assertEqual(result["acceptance"]["atol"], MODULE.DEFAULT_ATOL)
        self.assertEqual(result["acceptance"]["rtol"], MODULE.DEFAULT_RTOL)
        self.assertEqual(set(result["acceptance"]["cases"]), set(MODULE.CASES))
        self.assertEqual(result["acceptance"]["clip_mismatch_count"], 0)
        self.assertFalse(result["data_access"]["held_out_85606_read"])
        self.assertEqual(result["data_access"]["plasma_state_frames_read"], 0)
        self.assertEqual(
            result["scientific_status"], "accepted_shifted_xy_face_stage_only"
        )

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

    def test_gpl_driver_locks_exact_face_intermediates_and_cases(self) -> None:
        source = (ORACLE_DIR / "hermes_xy_face_oracle.cxx").read_text(
            encoding="utf-8"
        )
        options = (ORACLE_DIR / "BOUT.inp").read_text(encoding="utf-8")
        self.assertIn("SPDX-License-Identifier: GPL-3.0-or-later", source)
        self.assertIn("Field3D dfdy = DDY(phi);", source)
        self.assertIn("0.25 * (q(i + 1, j, k) - q(i - 1, j, k))", source)
        self.assertIn("0.25 * (q(i + 2, j, k) - q(i, j, k))", source)
        for output in ("face_velocity", "face_state", "face_flow", "face_clipped"):
            self.assertIn(output, source)
        for case in MODULE.CASES:
            self.assertIn(f'"{case}"', source)
            self.assertIn(f"q_{case} =", options)
            self.assertIn(f"phi_{case} =", options)
        self.assertIn("zperiod = 5", options)
        self.assertIn("NXPE = 1", options)

    def test_face_regions_include_separatrix_branches_and_sol(self) -> None:
        left_indices = np.arange(1, 62, dtype=np.int64)
        regions = MODULE.comparison_regions(left_indices)
        self.assertEqual(regions["all_valid"].shape, (61, 32))
        separatrix_row = int(np.flatnonzero(left_indices == 15)[0])
        sol_row = int(np.flatnonzero(left_indices == 16)[0])
        self.assertTrue(regions["separatrix_radial_face"][separatrix_row, 12])
        self.assertTrue(regions["inner_core_branch_connection"][0, 8])
        self.assertTrue(regions["inner_private_flux_connection"][0, 24])
        self.assertTrue(regions["open_sol_interior"][sol_row, 12])
        self.assertFalse(regions["all_valid"][0, 0])
        self.assertFalse(regions["all_valid"][0, 31])
        for name, mask in regions.items():
            self.assertGreater(int(np.count_nonzero(mask)), 0, name)

    def test_input_and_binary_clip_gates_reject_collapsed_or_wrong_data(self) -> None:
        constant = np.full((2, 3, 4), 2.5)
        phi = np.full((2, 3, 4), 4.0)
        self.assertTrue(MODULE.field_input_metrics(constant, phi, "constant")["passed"])
        self.assertFalse(
            MODULE.field_input_metrics(np.zeros_like(constant), phi, "constant")[
                "passed"
            ]
        )
        varying = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        self.assertTrue(MODULE.field_input_metrics(varying, varying, "smooth")["passed"])
        self.assertFalse(
            MODULE.field_input_metrics(np.zeros_like(varying), varying, "smooth")[
                "passed"
            ]
        )

        mask = np.ones((1, 2), dtype=bool)
        reference = np.asarray([[[0.0, 1.0], [1.0, 0.0]]])
        candidate = reference.astype(bool)
        accepted = MODULE.discrete_clip_metrics(candidate, reference, mask)
        self.assertTrue(accepted["passed"])
        candidate[0, 1, 1] = True
        self.assertFalse(
            MODULE.discrete_clip_metrics(candidate, reference, mask)["passed"]
        )
        nonbinary = reference.copy()
        nonbinary[0, 0, 0] = 0.25
        self.assertFalse(
            MODULE.discrete_clip_metrics(reference.astype(bool), nonbinary, mask)[
                "passed"
            ]
        )

    def test_tolerance_and_coverage_are_frozen_in_code_and_protocol(self) -> None:
        self.assertEqual(MODULE.DEFAULT_ATOL, 5.0e-10)
        self.assertEqual(MODULE.DEFAULT_RTOL, 5.0e-10)
        self.assertEqual(
            MODULE.CASES, ("constant", "smooth", "signed", "clipping")
        )
        protocol = PROTOCOL.read_text(encoding="utf-8")
        for required in (
            "Shifted-`xy` radial face flow",
            "safe radial faces whose model-local",
            "left cells are `1:62`",
            "max_abs_error <= 5e-10 + 5e-10 * s",
            "both positive and negative face",
            "one clipped and one",
        ):
            self.assertIn(required, protocol)


if __name__ == "__main__":
    unittest.main()
