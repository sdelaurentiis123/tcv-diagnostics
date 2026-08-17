from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcv_diagnostics import state_completeness as state


class StateCompletenessUnitTests(unittest.TestCase):
    def test_soft_floor_matches_executed_scalar_definition(self) -> None:
        values = np.array([-2.0, 0.0, 1e-7, 1.0], dtype=np.float64)
        result = state.soft_floor(values, 1e-7)
        expected = np.maximum(values, 0.0) + 1e-7 * np.exp(
            -np.maximum(values, 0.0) / 1e-7
        )
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result[0], 1e-7)
        self.assertEqual(result[1], 1e-7)
        self.assertEqual(result[-1], 1.0)
        with self.assertRaises(ValueError):
            state.soft_floor(values, 0.0)

    def test_frozen_relations_separate_exact_floor_from_plain_density(self) -> None:
        ne = np.array([[[[-1e-8, 0.0, 1e-8, 1.0]]]], dtype=np.float64)
        ve = np.array([[[[1.0, -2.0, 3.0, -4.0]]]], dtype=np.float64)
        vi = np.array([[[[-0.5, 0.25, 1.5, 2.0]]]], dtype=np.float64)
        nlim = state.soft_floor(ne, 1e-7)
        fields = {
            "Ne": ne,
            "Ve": ve,
            "Vi": vi,
            "NVe": (1.0 / 1836.0) * nlim * ve,
            "NVi": 2.0 * nlim * vi,
        }
        for relation in state.SOURCE_EXACT_RELATIONS:
            reference, candidate = state.relation_arrays(
                relation,
                fields,
                density_floor=1e-7,
                electron_atomic_mass=1.0 / 1836.0,
                ion_atomic_mass=2.0,
            )
            np.testing.assert_array_equal(reference, candidate)
        for relation in state.RELATIONS[2:]:
            reference, candidate = state.relation_arrays(
                relation,
                fields,
                density_floor=1e-7,
                electron_atomic_mass=1.0 / 1836.0,
                ion_atomic_mass=2.0,
            )
            self.assertGreater(float(np.max(np.abs(reference - candidate))), 0.0)

    def test_field_and_density_floor_accumulators_use_physical_coordinates(self) -> None:
        values = np.array(
            [
                [[[0.0], [2e-7]], [[-1e-8], [1.0]]],
                [[[1e-8], [2.0]], [[3.0], [4.0]]],
            ],
            dtype=np.float64,
        )
        field = state.FieldAccumulator()
        field.update(values, x0=4, y0=0)
        result = field.result()["scopes"]["full_physical_domain"]
        self.assertEqual(result["total_count"], values.size)
        self.assertEqual(result["nonfinite_count"], 0)
        self.assertEqual(result["minimum"]["location_txyz"], [0, 5, 0, 0])
        self.assertTrue(math.isclose(result["rms"], float(np.sqrt(np.mean(values**2)))))

        floor = state.DensityFloorAccumulator(
            frame_count=2,
            nx=8,
            ny=32,
            density_floor=1e-7,
            temporal_blocks=[(0, 0), (1, 1)],
        )
        floor.update(values, x0=4, y0=0)
        floor_result = floor.result()
        self.assertEqual(
            floor_result["scope_counts"]["full_physical_domain"]["below_zero"],
            1,
        )
        self.assertEqual(
            floor_result["scope_counts"]["full_physical_domain"]["below_density_floor"],
            3,
        )
        self.assertEqual(floor_result["count_by_x"]["below_zero"][5], 1)
        self.assertEqual(floor_result["count_by_temporal_block"]["below_zero"], [1, 0])

    def test_closure_accumulator_reports_passes_l2_and_localized_failure(self) -> None:
        reference = np.ones((2, 2, 2, 1), dtype=np.float64)
        candidate = reference.copy()
        candidate[1, 1, 0, 0] += 1e-5
        accumulator = state.ClosureAccumulator(
            frame_count=2,
            nx=8,
            ny=32,
            temporal_blocks=[(0, 0), (1, 1)],
            atol=1e-12,
            rtol=1e-12,
        )
        accumulator.update(reference, candidate, x0=3, y0=0)
        result = accumulator.result()
        full = result["scopes"]["full_physical_domain"]
        self.assertEqual(full["frame_passed"], [True, False])
        self.assertEqual(full["point_discrepancy_count"], 1)
        self.assertEqual(full["maximum_error"]["location_txyz"], [1, 4, 0, 0])
        self.assertTrue(math.isclose(full["relative_l2_error"], 1e-5 / math.sqrt(8)))
        self.assertEqual(result["point_discrepancy_count_by_x"][4], 1)

    def test_findings_never_authorize_channel_change(self) -> None:
        fields = {
            name: {
                "scopes": {"full_physical_domain": {"nonfinite_count": 0}}
            }
            for name in ("Ne", "NVe", "NVi", "Ve", "Vi")
        }
        closures = {
            name: {
                "scopes": {"full_physical_domain": {"frame_fail_count": 0}}
            }
            for name in state.SOURCE_EXACT_RELATIONS
        }
        findings = state.derive_findings(fields, closures)
        self.assertTrue(findings["source_exact_velocity_momentum_equivalence"])
        self.assertFalse(findings["historical_c5_is_complete_evolved_state"])
        self.assertFalse(findings["automatic_channel_change_authorized"])
        self.assertFalse(findings["potential_vorticity_gate_completed"])


if __name__ == "__main__":
    unittest.main()
