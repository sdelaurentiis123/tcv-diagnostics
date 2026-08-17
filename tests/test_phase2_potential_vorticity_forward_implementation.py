from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper0.tools.compare_potential_vorticity_forward_oracle import (
    apply_complex_tridiagonal,
    mode_residual_summary,
    write_strict_json,
)


ROOT = Path(__file__).resolve().parents[1]
OLD_DRIVER = (
    ROOT / "paper0/oracles/potential_elliptic/potential_elliptic_oracle.cxx"
)
DRIVER = (
    ROOT
    / "paper0/oracles/potential_vorticity_forward"
    / "potential_vorticity_forward_oracle.cxx"
)
CMAKE = ROOT / "paper0/oracles/potential_vorticity_forward/CMakeLists.txt"
COMPARATOR = (
    ROOT / "paper0/tools/compare_potential_vorticity_forward_oracle.py"
)
OLD_DRIVER_SHA256 = (
    "8690f015a246348cabe4725824a4181fb42ec41c139486e6ca1c2a48de7de687"
)
DRIVER_SHA256 = (
    "516527bea146d2ccc258e210e4058f233ecb66ca030fa48298030224919ac487"
)
CMAKE_SHA256 = (
    "c92774a770adf1788aafe3eadf33aabc6e130d0b893ffb737ab19c21efbeeaca"
)
COMPARATOR_SHA256 = (
    "c95674608a80a5b5aab0f177c9d8ce067c58dbfe9ae20c1437634ffef722d52d"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PotentialVorticityForwardImplementationTests(unittest.TestCase):
    def test_new_files_are_locked_and_accepted_driver_is_unchanged(self) -> None:
        self.assertEqual(sha256(OLD_DRIVER), OLD_DRIVER_SHA256)
        self.assertEqual(sha256(DRIVER), DRIVER_SHA256)
        self.assertEqual(sha256(CMAKE), CMAKE_SHA256)
        self.assertEqual(sha256(COMPARATOR), COMPARATOR_SHA256)

    def test_driver_uses_all_native_modes_and_public_cyclic_coefficients(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("const int mode_count = mesh->LocalNz / 2 + 1", source)
        self.assertIn("mode_count != 41", source)
        self.assertIn("solver.tridagCoefs", source)
        self.assertIn("rfft(communicated", source)
        self.assertIn("irfft(forward_line.data()", source)
        self.assertIn("2.0 / SQ(mesh->getCoordinates()->Bxy)", source)
        self.assertNotIn("FV::Div_a_Grad_perp", source)

    def test_driver_contains_every_ordered_compiled_gate(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        self.assertIn("PAPER0_RUNTIME_PRESSURE_CORRECTION", source)
        self.assertIn("runtime_species_pressure", source)
        self.assertIn("CONSTANT_NULL_VALUE", source)
        self.assertIn("GAUGE_SHIFT", source)
        self.assertIn("MANUFACTURED_TOROIDAL_MODE = 3", source)
        self.assertIn("manufactured_reconstructed_u", source)
        self.assertIn("saved_midpoint_inner_", source)
        self.assertIn("forward_Vort_", source)
        cmake = CMAKE.read_text(encoding="utf-8")
        self.assertIn("PAPER0_RUNTIME_PRESSURE_CORRECTION=1", cmake)
        self.assertIn("potential_vorticity_forward_oracle", cmake)

    def test_comparator_blocks_source_metrics_until_preliminary_pass(self) -> None:
        source = COMPARATOR.read_text(encoding="utf-8")
        gate_position = source.index("if preliminary_passed:")
        assembly_position = source.index('f"forward_Vort_{label}"')
        self.assertLess(gate_position, assembly_position)
        self.assertIn('"status": "blocked_by_preliminary_gate"', source)
        self.assertIn("all_64x32x81_physical_points", source)
        self.assertIn("alternative_relax_potential_fv_operator_used", source)

    def test_complex_tridiagonal_known_answer(self) -> None:
        result = apply_complex_tridiagonal(
            np.array([0.0, 2.0, 3.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([4.0, 5.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(result, np.array([9.0, 19.0, 9.0]))

    def test_complex_tridiagonal_rejects_bad_or_nonfinite_input(self) -> None:
        with self.assertRaises(ValueError):
            apply_complex_tridiagonal(
                np.ones(2), np.ones(3), np.ones(3), np.ones(3)
            )
        with self.assertRaises(ValueError):
            apply_complex_tridiagonal(
                np.ones(3),
                np.ones(3),
                np.ones(3),
                np.array([1.0, np.nan, 2.0]),
            )

    def test_mode_residual_localizes_known_fourier_mode_and_n_mapping(self) -> None:
        z = 2.0 * np.pi * np.arange(81) / 81.0
        reference = np.cos(2.0 * z)[None, None, :]
        candidate = reference + 0.5 * np.sin(3.0 * z)[None, None, :]
        summary = mode_residual_summary(candidate, reference, zperiod=5)
        self.assertEqual(summary["fourier_index_k"], list(range(41)))
        self.assertEqual(summary["toroidal_mode_n"][2], 10)
        self.assertEqual(summary["toroidal_mode_n"][3], 15)
        residual = np.asarray(summary["residual_power"])
        self.assertEqual(int(np.argmax(residual)), 3)
        self.assertGreater(summary["reference_power"][2], 0.0)
        self.assertLess(summary["relative_residual_power"][2], 1e-28)
        self.assertIsNone(summary["relative_residual_power"][3])

    def test_mode_residual_rejects_wrong_native_contract(self) -> None:
        values = np.zeros((2, 81))
        with self.assertRaises(ValueError):
            mode_residual_summary(values, values, zperiod=1)
        with self.assertRaises(ValueError):
            mode_residual_summary(values[:, :-1], values[:, :-1], zperiod=5)
        values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            mode_residual_summary(values, values, zperiod=5)

    def test_strict_json_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_strict_json(path, {"finite": 1.0})
            self.assertEqual(json.loads(path.read_text()), {"finite": 1.0})
            with self.assertRaises(FileExistsError):
                write_strict_json(path, {"finite": 2.0})


if __name__ == "__main__":
    unittest.main()
