from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.transport import (  # noqa: E402
    PartialRadialFaceFlow,
    divergence_from_xz_face_flow_partial,
    mc_radial_face_states_partial,
    monotonized_central_slope,
    radial_exb_xz_face_flow_partial,
    toroidal_wedge_spacing,
)


class ToroidalSpacingTests(unittest.TestCase):
    def test_one_fifth_wedge_spacing_is_exact(self) -> None:
        self.assertAlmostEqual(
            toroidal_wedge_spacing(81, zperiod=5),
            2.0 * math.pi / (5.0 * 81.0),
            places=16,
        )
        self.assertAlmostEqual(
            toroidal_wedge_spacing(88),
            2.0 * math.pi / (5.0 * 88.0),
            places=16,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            toroidal_wedge_spacing(88, zperiod=0)


class LimiterTests(unittest.TestCase):
    def test_mc_slope_matches_hand_calculated_stencils(self) -> None:
        minus = np.asarray([1.0, 1.0, 3.0, 2.0])
        center = np.asarray([2.0, 3.0, 2.0, 2.0])
        plus = np.asarray([3.0, 2.0, 1.0, 3.0])
        expected = np.asarray([1.0, 0.0, -1.0, 0.0])
        np.testing.assert_allclose(
            monotonized_central_slope(minus, center, plus), expected
        )

    def test_left_and_right_face_states_use_different_upwind_stencils(self) -> None:
        radial_line = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0])
        field = np.broadcast_to(radial_line[:, None, None], (5, 1, 3))
        from_left, from_right, left_indices = mc_radial_face_states_partial(field)
        np.testing.assert_array_equal(left_indices, np.asarray([1, 2]))
        np.testing.assert_allclose(from_left[:, 0, 0], [1.5, 2.75])
        np.testing.assert_allclose(from_right[:, 0, 0], [1.25, 2.5])


class PartialFaceFlowTests(unittest.TestCase):
    def test_constant_potential_gives_zero_flow(self) -> None:
        rng = np.random.default_rng(20260817)
        advected = rng.uniform(0.2, 2.0, size=(2, 6, 3, 12))
        potential = np.full_like(advected, 4.25)
        jacobian = rng.uniform(0.5, 2.0, size=(6, 3))
        result = radial_exb_xz_face_flow_partial(
            advected,
            potential,
            jacobian,
            dz=toroidal_wedge_spacing(12),
        )
        self.assertEqual(result.flow.shape, (2, 3, 3, 12))
        np.testing.assert_array_equal(result.left_cell_indices, [1, 2, 3])
        np.testing.assert_array_equal(result.velocity_factor, 0.0)
        np.testing.assert_array_equal(result.flow, 0.0)
        self.assertIn("partial", result.component)

    def test_known_sinusoid_matches_corner_formula_and_periodic_wrap(self) -> None:
        n_x, n_y, n_z = 6, 2, 16
        mode = 3
        z_index = np.arange(n_z, dtype=np.float64)
        theta = 2.0 * math.pi * mode * z_index / n_z
        potential_line = np.sin(theta)
        potential = np.broadcast_to(
            potential_line, (n_x, n_y, n_z)
        ).copy()
        advected = np.full_like(potential, 3.0)
        jacobian = np.full((n_x, n_y), 2.0)
        dz = toroidal_wedge_spacing(n_z)

        result = radial_exb_xz_face_flow_partial(
            advected, potential, jacobian, dz=dz
        )
        expected_velocity = 2.0 * np.cos(theta) * np.sin(
            2.0 * math.pi * mode / n_z
        ) / dz
        expected = np.broadcast_to(
            expected_velocity, result.velocity_factor.shape
        )
        np.testing.assert_allclose(
            result.velocity_factor, expected, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            result.flow, 3.0 * expected, rtol=1e-13, atol=1e-13
        )
        self.assertAlmostEqual(
            float(result.velocity_factor[0, 0, -1]),
            float(expected_velocity[-1]),
            places=13,
        )

    def test_geometry_and_shape_fail_loudly(self) -> None:
        field = np.ones((6, 2, 8))
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            radial_exb_xz_face_flow_partial(
                field, field[:, :, :-1], np.ones((6, 2)), dz=0.1
            )
        with self.assertRaisesRegex(ValueError, "jacobian"):
            radial_exb_xz_face_flow_partial(
                field, field, np.ones((5, 2)), dz=0.1
            )
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            radial_exb_xz_face_flow_partial(
                field, field, np.ones((6, 2)), dz=0.0
            )
        with self.assertRaisesRegex(ValueError, "four radial"):
            mc_radial_face_states_partial(np.ones((3, 2, 8)))

    def test_partial_divergence_is_conservative_on_safe_cells(self) -> None:
        rng = np.random.default_rng(44)
        n_x, n_y, n_z = 8, 3, 10
        advected = rng.uniform(0.5, 2.0, size=(n_x, n_y, n_z))
        potential = rng.normal(size=(n_x, n_y, n_z))
        jacobian = rng.uniform(0.7, 1.8, size=(n_x, n_y))
        dx = rng.uniform(0.2, 0.6, size=(n_x, n_y))
        faces = radial_exb_xz_face_flow_partial(
            advected,
            potential,
            jacobian,
            dz=toroidal_wedge_spacing(n_z),
        )
        result = divergence_from_xz_face_flow_partial(
            faces, jacobian, dx=dx
        )
        np.testing.assert_array_equal(result.cell_indices, [2, 3, 4, 5])
        volume_weighted = result.divergence * (
            jacobian[result.cell_indices] * dx[result.cell_indices]
        )[..., None]
        np.testing.assert_allclose(
            np.sum(volume_weighted, axis=-3),
            faces.flow[-1] - faces.flow[0],
            rtol=1e-14,
            atol=1e-14,
        )

    def test_divergence_rejects_nonconsecutive_face_metadata(self) -> None:
        bad = PartialRadialFaceFlow(
            flow=np.zeros((2, 1, 4)),
            velocity_factor=np.zeros((2, 1, 4)),
            left_cell_indices=np.asarray([1, 3]),
        )
        with self.assertRaisesRegex(ValueError, "consecutive"):
            divergence_from_xz_face_flow_partial(
                bad, np.ones((5, 1)), dx=1.0
            )


if __name__ == "__main__":
    unittest.main()
