from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcv_diagnostics import phi_boundary as boundary  # noqa: E402


BLOCKS = [(index * 2, index * 2 + 1) for index in range(8)]


class PhiBoundaryTests(unittest.TestCase):
    def test_extract_planes_uses_frozen_side_specific_indices(self) -> None:
        phi = np.arange(16 * 8 * 6 * 81, dtype=np.float64).reshape(16, 8, 6, 81)
        inner = boundary.extract_boundary_planes(phi, side="inner")
        outer = boundary.extract_boundary_planes(phi, side="outer")
        np.testing.assert_array_equal(inner["outermost_guard"], phi[:, 0, 2:4, :])
        np.testing.assert_array_equal(inner["adjacent_interior"], phi[:, 2, 2:4, :])
        np.testing.assert_array_equal(outer["outermost_guard"], phi[:, 7, 2:4, :])
        np.testing.assert_array_equal(outer["adjacent_interior"], phi[:, 5, 2:4, :])

    @staticmethod
    def planes(*, departure: float, gauge: float = 0.0):
        time = np.arange(16, dtype=np.float64)[:, None, None]
        y = np.arange(32, dtype=np.float64)[None, :, None]
        z = np.arange(81, dtype=np.float64)[None, None, :]
        interior = gauge + 2.0 + 0.01 * time + 0.02 * y + np.sin(2 * np.pi * z / 81)
        target = np.mean(interior, axis=-1, keepdims=True)
        midpoint = target + departure
        adjacent = 2.0 * midpoint - interior
        return {
            "outermost_guard": adjacent.copy(),
            "adjacent_guard": adjacent,
            "adjacent_interior": interior,
        }

    def analyze(self, planes):
        return boundary.analyze_side(
            planes,
            atol=1e-12,
            rtol=1e-12,
            conversion_volts=50.0,
            percentiles=(50.0, 90.0, 95.0, 99.0, 100.0),
            temporal_blocks=BLOCKS,
        )

    def test_nonzero_constant_midpoint_is_structurally_valid_memory_state(self) -> None:
        result = self.analyze(self.planes(departure=0.25))
        self.assertEqual(result["outer_guard_copy"]["point_discrepancy_count"], 0)
        self.assertEqual(
            result["midpoint_toroidal_constancy"]["point_discrepancy_count"], 0
        )
        self.assertEqual(
            result["instantaneous_neumann"]["point_discrepancy_count"], 16 * 32
        )
        self.assertAlmostEqual(result["departure"]["rms"], 0.25)
        self.assertAlmostEqual(result["departure"]["rms_physical"], 12.5)
        findings = boundary.derive_findings({"inner": result, "outer": result})
        self.assertTrue(findings["saved_compact_boundary_value_structurally_valid"])
        self.assertTrue(findings["nonzero_saved_boundary_state_detected"])
        self.assertFalse(findings["materiality_established"])

    def test_instantaneous_neumann_has_no_distinct_saved_boundary_value(self) -> None:
        result = self.analyze(self.planes(departure=0.0))
        findings = boundary.derive_findings({"inner": result, "outer": result})
        self.assertTrue(findings["instantaneous_neumann_passes_everywhere"])
        self.assertFalse(findings["nonzero_saved_boundary_state_detected"])

    def test_departure_and_ratio_are_gauge_invariant(self) -> None:
        base = self.analyze(self.planes(departure=0.125, gauge=0.0))
        shifted = self.analyze(self.planes(departure=0.125, gauge=1e6))
        self.assertAlmostEqual(base["departure"]["rms"], shifted["departure"]["rms"])
        self.assertAlmostEqual(
            base["departure"]["rms_to_interior_toroidal_fluctuation_rms"],
            shifted["departure"]["rms_to_interior_toroidal_fluctuation_rms"],
            places=9,
        )

    def test_outer_copy_failure_is_not_mislabeled_as_boundary_memory(self) -> None:
        planes = self.planes(departure=0.25)
        planes["outermost_guard"][0, 0, 0] += 1.0
        result = self.analyze(planes)
        findings = boundary.derive_findings({"inner": result, "outer": result})
        self.assertFalse(findings["outer_guard_copy_passes"])
        self.assertFalse(findings["saved_compact_boundary_value_structurally_valid"])
        self.assertFalse(findings["nonzero_saved_boundary_state_detected"])


if __name__ == "__main__":
    unittest.main()
