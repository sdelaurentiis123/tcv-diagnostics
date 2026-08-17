from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "paper0/oracles/potential_elliptic/potential_elliptic_oracle.cxx"
CMAKE = ROOT / "paper0/oracles/potential_elliptic_runtime_pressure/CMakeLists.txt"
COMPARATOR = (
    ROOT
    / "paper0/tools/compare_potential_elliptic_runtime_pressure_oracle.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_potential_elliptic_runtime_pressure_oracle", COMPARATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load corrected potential comparator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORRECTED = load_module()


class PotentialEllipticCorrectionImplementationTests(unittest.TestCase):
    def test_runtime_pressure_matches_executed_scalar_formula(self) -> None:
        density = np.array([4.0e-3, 5.0e-8, -2.0e-8], dtype=np.float64)
        pressure = np.array([2.0, -3.0, 4.0], dtype=np.float64)
        floor = 1.0e-7
        nonnegative_density = np.maximum(density, 0.0)
        expected = density * np.maximum(pressure, 0.0) / (
            nonnegative_density
            + floor * np.exp(-nonnegative_density / floor)
        )
        actual = CORRECTED.runtime_pressure(pressure, density, floor)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[1], 0.0)
        self.assertLess(actual[2], 0.0)

    def test_runtime_pressure_rejects_invalid_contract_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            CORRECTED.runtime_pressure(np.ones(2), np.ones(3), 1e-7)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            CORRECTED.runtime_pressure(np.ones(2), np.ones(2), 0.0)

    def test_driver_preserves_old_default_and_adds_explicit_corrected_branch(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        for required in (
            "#ifdef PAPER0_RUNTIME_PRESSURE_CORRECTION",
            "runtime_species_pressure",
            "nonnegative_density",
            "PRESSURE_DENSITY_FLOOR",
            "std::exp(-nonnegative_density / PRESSURE_DENSITY_FLOOR)",
            "runtime_pi - runtime_pe / ELECTRON_PRESSURE_DENOMINATOR",
            "pi - pe / ELECTRON_PRESSURE_DENOMINATOR",
            'output["runtime_Pe_" + label]',
            'output["runtime_Pi_" + label]',
            'output["paper0_runtime_pressure_correction"] = 1',
        ):
            self.assertIn(required, source)
        cmake = CMAKE.read_text(encoding="utf-8")
        self.assertIn("PAPER0_RUNTIME_PRESSURE_CORRECTION=1", cmake)
        self.assertIn("potential_elliptic_runtime_pressure_oracle", cmake)

    def test_runtime_gate_precedes_unchanged_base_comparator(self) -> None:
        text = COMPARATOR.read_text(encoding="utf-8")
        validate = text.index("runtime_gate = validate_runtime_pressure")
        stop = text.index('if not runtime_gate["passed"]')
        run_base = text.index("command, base_status = run_base_comparator")
        self.assertLess(validate, stop)
        self.assertLess(stop, run_base)
        self.assertIn(
            'BASE_COMPARATOR = TOOLS / "compare_potential_elliptic_oracle.py"',
            text,
        )
        self.assertIn("refusing to overwrite", text)
        self.assertIn("automatic_held_out_access_authorized", text)

    def test_corrected_metadata_distinguishes_raw_and_runtime_pressure(self) -> None:
        text = COMPARATOR.read_text(encoding="utf-8")
        for required in (
            '"raw_evolved_pressure_fields": ["Pe", "Pi"]',
            "P_runtime = N * max(P_evolved,0) / softFloor(N,1e-7)",
            "Pi_hat = Pi_runtime - Pe_runtime / 3672",
            '"runtime_pressure_transformation_gate"',
            '"predecessor_failed_result"',
            '"base_comparison_sha256"',
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
