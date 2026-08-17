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
    PartialCombinedRadialDivergence,
    PartialCombinedRadialFaceFlow,
    PartialRadialFaceFlow,
    SingleNullTopology,
    divergence_from_radial_face_flow_partial,
    divergence_from_xz_face_flow_partial,
    fromm_radial_face_states_partial,
    mc_radial_face_states_partial,
    monotonized_central_slope,
    radial_exb_face_flow_partial,
    radial_exb_xy_face_flow_partial,
    radial_exb_xz_face_flow_partial,
    shifted_ddy_single_null_partial,
    single_null_y_neighbors,
    spectral_shift_z,
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

    def test_fromm_states_match_executed_four_cell_formula(self) -> None:
        radial_line = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
        field = np.broadcast_to(radial_line[:, None, None], (5, 1, 3))
        states = fromm_radial_face_states_partial(field, positive=False)
        np.testing.assert_array_equal(states.left_cell_indices, [1, 2])
        np.testing.assert_allclose(states.state_from_left[:, 0, 0], [2.75, 5.5])
        np.testing.assert_allclose(states.state_from_right[:, 0, 0], [2.5, 5.0])
        self.assertFalse(np.any(states.clipped_from_left))
        self.assertFalse(np.any(states.clipped_from_right))

    def test_fromm_positivity_clips_only_negative_candidate_states(self) -> None:
        radial_line = np.asarray([10.0, 1.0, 0.1, 10.0, 10.0])
        field = np.broadcast_to(radial_line[:, None, None], (5, 1, 3))
        unclipped = fromm_radial_face_states_partial(field, positive=False)
        clipped = fromm_radial_face_states_partial(field, positive=True)
        self.assertLess(float(unclipped.state_from_left[0, 0, 0]), 0.0)
        self.assertLess(float(unclipped.state_from_right[0, 0, 0]), 0.0)
        self.assertEqual(float(clipped.state_from_left[0, 0, 0]), 0.0)
        self.assertEqual(float(clipped.state_from_right[0, 0, 0]), 0.0)
        self.assertTrue(bool(clipped.clipped_from_left[0, 0, 0]))
        self.assertTrue(bool(clipped.clipped_from_right[0, 0, 0]))
        self.assertGreater(float(clipped.state_from_left[1, 0, 0]), 0.0)
        with self.assertRaisesRegex(TypeError, "boolean"):
            fromm_radial_face_states_partial(field, positive=1)


class ShiftedDerivativeTests(unittest.TestCase):
    @staticmethod
    def topology(separatrix_x_index: int) -> SingleNullTopology:
        return SingleNullTopology(
            separatrix_x_index=separatrix_x_index,
            core_lower_y=8,
            core_upper_y=23,
            pfr_lower_y=7,
            pfr_upper_y=24,
        )

    def test_fourier_shift_recovers_known_mode_phase_and_inverse(self) -> None:
        n_z = 81
        stored_k = 4
        shift = 0.071
        z_index = np.arange(n_z, dtype=np.float64)
        signal = np.cos(2.0 * math.pi * stored_k * z_index / n_z)
        shifted = spectral_shift_z(signal, shift, zperiod=5)
        expected = np.cos(
            2.0 * math.pi * stored_k * z_index / n_z
            + 5.0 * stored_k * shift
        )
        np.testing.assert_allclose(shifted, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            spectral_shift_z(shifted, -shift, zperiod=5),
            signal,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_single_null_neighbor_map_separates_core_pfr_and_sol(self) -> None:
        minus, plus, valid = single_null_y_neighbors(
            3, 32, self.topology(separatrix_x_index=2)
        )
        self.assertEqual(int(minus[0, 8]), 23)
        self.assertEqual(int(plus[0, 23]), 8)
        self.assertEqual(int(plus[0, 7]), 24)
        self.assertEqual(int(minus[0, 24]), 7)
        self.assertEqual(int(minus[2, 8]), 7)
        self.assertEqual(int(plus[2, 23]), 24)
        self.assertFalse(bool(valid[1, 0]))
        self.assertFalse(bool(valid[1, 31]))
        self.assertTrue(bool(valid[1, 1]))

    def test_open_sol_manufactured_aligned_gradient_is_recovered(self) -> None:
        n_x, n_y, n_z = 2, 32, 81
        stored_k = 3
        full_torus_n = 5 * stored_k
        z = np.arange(n_z, dtype=np.float64) * toroidal_wedge_spacing(n_z)
        y = np.arange(n_y, dtype=np.float64)
        dy = np.broadcast_to(np.asarray([0.2, 0.35])[:, None], (n_x, n_y)).copy()
        z_shift = (
            0.03 * np.arange(n_x, dtype=np.float64)[:, None]
            + 0.002 * y[None, :]
        )
        aligned_amplitude = y[None, :] * dy
        potential = aligned_amplitude[..., None] * np.cos(
            full_torus_n * (z[None, None, :] - z_shift[..., None])
        )
        result = shifted_ddy_single_null_partial(
            potential,
            z_shift,
            dy,
            np.zeros(n_x),
            topology=self.topology(separatrix_x_index=1),
            zperiod=5,
        )
        expected = np.cos(
            full_torus_n * (z[None, :] - z_shift[1, :, None])
        )
        np.testing.assert_allclose(
            result.values[1, 1:-1],
            expected[1:-1],
            rtol=2e-13,
            atol=2e-13,
        )
        self.assertTrue(np.all(np.isnan(result.values[:, (0, -1), :])))
        self.assertIn("partial", result.component)

    def test_core_branch_shift_angle_signs_match_source_guard_correction(self) -> None:
        n_x, n_y, n_z = 1, 32, 81
        stored_k = 2
        full_torus_n = 5 * stored_k
        branch_shift = 0.09
        z = np.arange(n_z, dtype=np.float64) * toroidal_wedge_spacing(n_z)
        mode = np.cos(full_torus_n * z)
        potential = np.broadcast_to(mode, (n_x, n_y, n_z)).copy()
        result = shifted_ddy_single_null_partial(
            potential,
            np.zeros((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.asarray([branch_shift]),
            topology=self.topology(separatrix_x_index=1),
        )
        expected_lower = 0.5 * (
            np.cos(full_torus_n * z)
            - np.cos(full_torus_n * (z - branch_shift))
        )
        expected_upper = 0.5 * (
            np.cos(full_torus_n * (z + branch_shift))
            - np.cos(full_torus_n * z)
        )
        np.testing.assert_allclose(
            result.values[0, 8], expected_lower, rtol=2e-13, atol=2e-13
        )
        np.testing.assert_allclose(
            result.values[0, 23], expected_upper, rtol=2e-13, atol=2e-13
        )

    def test_y_code_exposes_all_four_inner_connection_stencils(self) -> None:
        n_x, n_y, n_z = 1, 32, 9
        y_code = np.arange(n_y, dtype=np.float64)[None, :, None]
        potential = np.broadcast_to(y_code, (n_x, n_y, n_z)).copy()
        result = shifted_ddy_single_null_partial(
            potential,
            np.zeros((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.asarray([0.0]),
            topology=self.topology(separatrix_x_index=1),
        )
        self.assertAlmostEqual(float(result.values[0, 7, 0]), 9.0)
        self.assertAlmostEqual(float(result.values[0, 24, 0]), 9.0)
        self.assertAlmostEqual(float(result.values[0, 8, 0]), -7.0)
        self.assertAlmostEqual(float(result.values[0, 23, 0]), -7.0)
        self.assertAlmostEqual(float(result.values[0, 12, 0]), 1.0)

    def test_shifted_derivative_geometry_errors_fail_loudly(self) -> None:
        field = np.ones((2, 32, 9))
        topology = self.topology(separatrix_x_index=1)
        with self.assertRaisesRegex(ValueError, "z_shift"):
            shifted_ddy_single_null_partial(
                field,
                np.zeros((2, 31)),
                np.ones((2, 32)),
                np.zeros(2),
                topology=topology,
            )
        with self.assertRaisesRegex(ValueError, "shift_angle"):
            shifted_ddy_single_null_partial(
                field,
                np.zeros((2, 32)),
                np.ones((2, 32)),
                np.zeros(1),
                topology=topology,
            )

    def test_shift_angle_nan_is_allowed_only_outside_twisted_core(self) -> None:
        field = np.ones((2, 32, 9))
        topology = self.topology(separatrix_x_index=1)
        result = shifted_ddy_single_null_partial(
            field,
            np.zeros((2, 32)),
            np.ones((2, 32)),
            np.asarray([0.2, np.nan]),
            topology=topology,
        )
        np.testing.assert_allclose(result.values[:, 1:-1], 0.0)
        with self.assertRaisesRegex(ValueError, "inner branch"):
            shifted_ddy_single_null_partial(
                field,
                np.zeros((2, 32)),
                np.ones((2, 32)),
                np.asarray([np.nan, 0.0]),
                topology=topology,
            )


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


class ShiftedFaceFlowTests(unittest.TestCase):
    @staticmethod
    def topology() -> SingleNullTopology:
        return SingleNullTopology(
            separatrix_x_index=0,
            core_lower_y=8,
            core_upper_y=23,
            pfr_lower_y=7,
            pfr_upper_y=24,
        )

    @staticmethod
    def fields(
        radial_values: np.ndarray,
        *,
        n_y: int = 32,
        n_z: int = 9,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_x = radial_values.size
        advected = np.broadcast_to(
            radial_values[:, None, None], (n_x, n_y, n_z)
        ).copy()
        potential = np.broadcast_to(
            np.arange(n_y, dtype=np.float64)[None, :, None],
            (n_x, n_y, n_z),
        ).copy()
        return advected, potential

    def evaluate(
        self,
        radial_values: np.ndarray,
        *,
        g23: float,
        positive: bool = True,
    ):
        advected, potential = self.fields(radial_values)
        n_x, n_y, _ = advected.shape
        return radial_exb_xy_face_flow_partial(
            advected,
            potential,
            np.full((n_x, n_y), 4.0),
            np.full((n_x, n_y), 2.0),
            np.full((n_x, n_y), g23),
            np.full((n_x, n_y), 2.0),
            np.zeros((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.zeros(n_x),
            topology=self.topology(),
            zperiod=5,
            positive=positive,
        )

    def test_known_y_gradient_matches_geometry_formula_and_masks_targets(self) -> None:
        result = self.evaluate(np.full(6, 2.0), g23=3.0)
        np.testing.assert_allclose(result.velocity_factor[:, 1:-1], 6.0)
        np.testing.assert_allclose(result.upwind_state[:, 1:-1], 2.0)
        np.testing.assert_allclose(result.flow[:, 1:-1], 12.0)
        self.assertTrue(np.all(np.isnan(result.flow[:, (0, -1), :])))
        self.assertTrue(np.all(result.valid_mask[:, 1:-1]))
        self.assertFalse(np.any(result.valid_mask[:, (0, -1)]))
        self.assertEqual(result.component, "radial_exb_xy_component_partial")

    def test_velocity_sign_selects_the_matching_fromm_side(self) -> None:
        radial = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
        positive = self.evaluate(radial, g23=3.0)
        negative = self.evaluate(radial, g23=-3.0)
        np.testing.assert_allclose(
            positive.upwind_state[:, 1:-1, 0],
            np.broadcast_to(np.asarray([2.75, 5.5])[:, None], (2, 30)),
        )
        np.testing.assert_allclose(
            negative.upwind_state[:, 1:-1, 0],
            np.broadcast_to(np.asarray([2.5, 5.0])[:, None], (2, 30)),
        )
        self.assertTrue(np.all(positive.flow[:, 1:-1] > 0.0))
        self.assertTrue(np.all(negative.flow[:, 1:-1] < 0.0))

    def test_selected_negative_fromm_state_is_clipped_before_flux(self) -> None:
        radial = np.asarray([10.0, 1.0, 0.1, 10.0, 10.0])
        result = self.evaluate(radial, g23=3.0, positive=True)
        self.assertTrue(np.all(result.positivity_clipped_mask[0, 1:-1]))
        np.testing.assert_array_equal(result.flow[0, 1:-1], 0.0)
        unbounded = self.evaluate(radial, g23=3.0, positive=False)
        self.assertTrue(np.all(unbounded.flow[0, 1:-1] < 0.0))

    def test_constant_potential_and_invalid_geometry_fail_safely(self) -> None:
        advected, potential = self.fields(np.ones(6))
        potential.fill(2.0)
        n_x, n_y, _ = potential.shape
        kwargs = {
            "topology": self.topology(),
            "zperiod": 5,
            "positive": True,
        }
        result = radial_exb_xy_face_flow_partial(
            advected,
            potential,
            np.ones((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.zeros((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.zeros(n_x),
            **kwargs,
        )
        np.testing.assert_array_equal(result.flow[:, 1:-1], 0.0)
        with self.assertRaisesRegex(ValueError, "bxy must be nonzero"):
            radial_exb_xy_face_flow_partial(
                advected,
                potential,
                np.ones((n_x, n_y)),
                np.ones((n_x, n_y)),
                np.ones((n_x, n_y)),
                np.zeros((n_x, n_y)),
                np.zeros((n_x, n_y)),
                np.ones((n_x, n_y)),
                np.zeros(n_x),
                **kwargs,
            )


class PartialCombinedFaceFlowTests(unittest.TestCase):
    @staticmethod
    def topology() -> SingleNullTopology:
        return SingleNullTopology(
            separatrix_x_index=0,
            core_lower_y=8,
            core_upper_y=23,
            pfr_lower_y=7,
            pfr_upper_y=24,
        )

    def test_combined_flow_is_exact_component_sum_on_valid_cells(self) -> None:
        rng = np.random.default_rng(85604)
        n_x, n_y, n_z = 8, 32, 11
        q = rng.uniform(0.4, 2.0, size=(2, n_x, n_y, n_z))
        phi = rng.normal(size=q.shape)
        jacobian = rng.uniform(0.8, 1.4, size=(n_x, n_y))
        g11 = rng.uniform(0.7, 1.3, size=(n_x, n_y))
        g23 = rng.uniform(-0.4, 0.4, size=(n_x, n_y))
        bxy = rng.uniform(0.9, 1.5, size=(n_x, n_y))
        z_shift = rng.uniform(-0.1, 0.1, size=(n_x, n_y))
        dy = rng.uniform(0.2, 0.6, size=(n_x, n_y))
        result = radial_exb_face_flow_partial(
            q,
            phi,
            jacobian,
            g11,
            g23,
            bxy,
            z_shift,
            dy,
            np.zeros(n_x),
            dz=toroidal_wedge_spacing(n_z),
            topology=self.topology(),
            positive=True,
        )
        valid = np.broadcast_to(
            result.valid_mask.reshape(1, *result.valid_mask.shape, 1),
            result.flow.shape,
        )
        np.testing.assert_allclose(
            result.flow[valid],
            (result.xz_flow + result.xy_flow)[valid],
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(np.all(np.isnan(result.flow[..., (0, -1), :])))
        self.assertIsInstance(result, PartialCombinedRadialFaceFlow)
        self.assertEqual(result.component, "radial_exb_xz_plus_xy_partial")

    def test_combined_divergence_telescopes_in_volume_weighted_form(self) -> None:
        rng = np.random.default_rng(44)
        n_x, n_y, n_z = 9, 32, 13
        q = rng.uniform(0.5, 1.8, size=(n_x, n_y, n_z))
        phi = rng.normal(size=q.shape)
        jacobian = rng.uniform(0.7, 1.7, size=(n_x, n_y))
        dx = rng.uniform(0.2, 0.5, size=(n_x, n_y))
        faces = radial_exb_face_flow_partial(
            q,
            phi,
            jacobian,
            np.ones((n_x, n_y)),
            rng.uniform(-0.3, 0.3, size=(n_x, n_y)),
            np.ones((n_x, n_y)),
            np.zeros((n_x, n_y)),
            np.ones((n_x, n_y)),
            np.zeros(n_x),
            dz=toroidal_wedge_spacing(n_z),
            topology=self.topology(),
            positive=True,
        )
        divergence = divergence_from_radial_face_flow_partial(
            faces,
            jacobian,
            dx=dx,
        )
        self.assertIsInstance(divergence, PartialCombinedRadialDivergence)
        self.assertEqual(
            divergence.component,
            "radial_exb_xz_plus_xy_divergence_partial",
        )
        self.assertTrue(np.all(divergence.valid_mask[:, 1:-1]))
        self.assertFalse(np.any(divergence.valid_mask[:, (0, -1)]))
        weighted = divergence.divergence[:, 1:-1] * (
            jacobian[divergence.cell_indices, 1:-1]
            * dx[divergence.cell_indices, 1:-1]
        )[..., None]
        np.testing.assert_allclose(
            np.sum(weighted, axis=0),
            faces.flow[-1, 1:-1] - faces.flow[0, 1:-1],
            rtol=2e-14,
            atol=2e-14,
        )

    def test_constant_potential_gives_zero_combined_flow_and_divergence(self) -> None:
        n_x, n_y, n_z = 7, 32, 9
        q = np.ones((n_x, n_y, n_z))
        phi = np.full_like(q, 3.5)
        geometry = np.ones((n_x, n_y))
        faces = radial_exb_face_flow_partial(
            q,
            phi,
            geometry,
            geometry,
            geometry,
            geometry,
            np.zeros_like(geometry),
            geometry,
            np.zeros(n_x),
            dz=toroidal_wedge_spacing(n_z),
            topology=self.topology(),
        )
        np.testing.assert_array_equal(faces.flow[:, 1:-1], 0.0)
        divergence = divergence_from_radial_face_flow_partial(
            faces,
            geometry,
            dx=geometry,
        )
        np.testing.assert_array_equal(divergence.divergence[:, 1:-1], 0.0)


if __name__ == "__main__":
    unittest.main()
