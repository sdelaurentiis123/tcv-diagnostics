from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper0" / "tools"
EXTRACTOR = TOOLS / "extract_potential_elliptic_85604_frames.py"
COMPARATOR = TOOLS / "compare_potential_elliptic_oracle.py"
ORACLE_DIR = ROOT / "paper0" / "oracles" / "potential_elliptic"
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase2_potential_elliptic_85604.json"
)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


EXTRACT = load_module("paper0_potential_extract", EXTRACTOR)
COMPARE = load_module("paper0_potential_compare", COMPARATOR)


class PotentialEllipticOracleImplementationTests(unittest.TestCase):
    def test_full_time_sequence_is_implied_by_frozen_selected_locks(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        times = EXTRACT.expected_time_sequence(manifest["frame_selection"])
        self.assertEqual(times.shape, (624,))
        self.assertEqual(float(times[0]), 285000.0)
        self.assertEqual(float(times[-1]), 471900.0)
        np.testing.assert_array_equal(np.diff(times), np.full(623, 300.0))

    def test_boundary_products_recover_midpoint_target_and_departure(self) -> None:
        frames, ny, nz = 3, 32, 81
        z = np.arange(nz, dtype=np.float64)[None, None, :]
        y = np.arange(ny, dtype=np.float64)[None, :, None]
        time = np.arange(frames, dtype=np.float64)[:, None, None]
        interior = 2.0 + 0.01 * time + 0.001 * y + np.sin(z) * 0.1
        streams = {}
        for side_index, side in enumerate(("inner", "outer")):
            midpoint_value = (
                4.0
                + side_index
                + 0.02 * time[..., 0]
                + 0.003 * y[..., 0]
            )
            adjacent = 2.0 * midpoint_value[..., None] - interior
            streams[side] = {
                "outermost_guard": adjacent.copy(),
                "adjacent_guard": adjacent,
                "adjacent_interior": interior,
            }
        products, checks = EXTRACT.derive_boundary_products(
            streams,
            frame_indices=[0, 2],
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        self.assertEqual(products["saved_midpoint"].shape, (2, 2, 32))
        np.testing.assert_array_equal(
            products["midpoint_departure"],
            products["saved_midpoint"] - products["instantaneous_target"],
        )
        for side in ("inner", "outer"):
            self.assertEqual(
                checks[side]["outer_guard_copy_count_by_frame"], [0, 0, 0]
            )
            self.assertEqual(
                checks[side]["midpoint_constancy_count_by_frame"], [0, 0, 0]
            )

    def test_bitwise_echo_is_stricter_than_numeric_equality(self) -> None:
        reference = np.asarray([0.0, 1.0], dtype=np.float64)
        candidate = np.asarray([-0.0, 1.0], dtype=np.float64)
        metrics = COMPARE.bitwise_metrics(candidate, reference)
        self.assertEqual(metrics["bitwise_mismatch_count"], 1)
        self.assertFalse(metrics["passed"])
        self.assertTrue(
            COMPARE.bitwise_metrics(reference.copy(), reference)["passed"]
        )

    def test_reconstruction_gate_uses_frozen_continuous_formula(self) -> None:
        reference = np.ones((2, 3, 4), dtype=np.float64)
        mask = np.ones((2, 3), dtype=bool)
        within = reference + 9.9e-10
        outside = reference + 1.01e-9
        self.assertTrue(
            COMPARE.gate_metrics(
                within,
                reference,
                mask,
                location_axes=("x", "y", "z"),
                atol=5.0e-10,
                rtol=5.0e-10,
            )["passed"]
        )
        self.assertFalse(
            COMPARE.gate_metrics(
                outside,
                reference,
                mask,
                location_axes=("x", "y", "z"),
                atol=5.0e-10,
                rtol=5.0e-10,
            )["passed"]
        )

    def test_continuous_metrics_and_constant_shift_are_labeled(self) -> None:
        reference = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        candidate = reference + 2.0
        mask = np.ones((2, 3), dtype=bool)
        raw = COMPARE.continuous_metrics(
            candidate,
            reference,
            mask,
            location_axes=("x", "y", "z"),
            include_sign_error=True,
        )
        self.assertAlmostEqual(raw["rmse"], 2.0)
        self.assertAlmostEqual(raw["bias"], 2.0)
        self.assertEqual(raw["maximum_absolute_difference"], 2.0)
        aligned = COMPARE.constant_shift_aligned_metrics(
            candidate,
            reference,
            mask,
            location_axes=("x", "y", "z"),
        )
        self.assertTrue(aligned["diagnostic_only"])
        self.assertAlmostEqual(
            aligned["removed_candidate_minus_reference_constant"], 2.0
        )
        self.assertAlmostEqual(aligned["maximum_absolute_difference"], 0.0)

    def test_gpl_driver_contains_exact_source_equation_and_two_arms(self) -> None:
        source = (
            ORACLE_DIR / "potential_elliptic_oracle.cxx"
        ).read_text(encoding="utf-8")
        options = (ORACLE_DIR / "BOUT.inp").read_text(encoding="utf-8")
        cmake = (ORACLE_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        for required in (
            "SPDX-License-Identifier: GPL-3.0-or-later",
            "EXPECTED_FRAMES{0, 156, 312, 467, 623}",
            'INPUT_FIELDS{"Ne", "Pe", "Pi", "Vort",',
            "pi - pe / ELECTRON_PRESSURE_DENOMINATOR",
            "solver->setCoefC(2.0 / SQ",
            "setInnerBoundaryFlags(INVERT_SET)",
            "setOuterBoundaryFlags(INVERT_SET)",
            "vorticity * (bsq / 2.0)",
            "0.5 * (phi_plus_pi",
            "retained_solver",
            "instantaneous_solver",
            "coord->g_23 /= SQ(rho_s0)",
            "BoutComm::size() != 4",
            "mesh->getNYPE() != 4",
        ):
            self.assertIn(required, source)
        self.assertIn("type = cyclic", options)
        self.assertIn("zperiod = 5", options)
        self.assertIn("NXPE = 1", options)
        self.assertIn("netCDF::netcdf", cmake)

    def test_comparator_blocks_effect_until_raw_reconstruction_passes(self) -> None:
        text = COMPARATOR.read_text(encoding="utf-8")
        blocked = text.index('"status": "blocked_by_source_reconstruction_gate"')
        gate = text.index("if reconstruction_gate_passed:")
        self.assertGreater(blocked, gate)
        self.assertIn("constant_shift_alignment_can_change_gate", text)
        self.assertIn('"materiality_label_assigned": False', text)
        self.assertIn('"called_total_heat_flux": False', text)


if __name__ == "__main__":
    unittest.main()
