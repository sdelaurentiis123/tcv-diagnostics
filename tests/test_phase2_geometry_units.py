from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcv_diagnostics.geometry import (  # noqa: E402
    build_single_null_region_masks,
    confined_separatrix_surface_mask,
)
from tcv_diagnostics.metrics import apply_memberwise  # noqa: E402
from tcv_diagnostics.transport import (  # noqa: E402
    PartialCombinedRadialFaceFlow,
    SingleNullTopology,
    hermes_transport_scales,
    integrate_radial_surface_flow,
    normalized_particle_flow_to_si,
    normalized_pressure_flow_to_si,
    radial_exb_face_flow_partial,
    toroidal_wedge_spacing,
)


class GeometryMaskTests(unittest.TestCase):
    @staticmethod
    def topology() -> SingleNullTopology:
        return SingleNullTopology(
            separatrix_x_index=2,
            core_lower_y=2,
            core_upper_y=5,
            pfr_lower_y=1,
            pfr_upper_y=6,
        )

    @staticmethod
    def inputs() -> tuple[np.ndarray, np.ndarray]:
        penalty = np.zeros((4, 8), dtype=np.float64)
        penalty[3, 1] = 0.25
        penalty[3, 6] = 1.0
        separatrix_radius = np.asarray(
            [0.2, 0.4, 0.8, 1.0, 1.3, 0.9, 0.5, 0.3],
            dtype=np.float64,
        )
        return penalty, separatrix_radius

    def test_regions_form_exact_strict_wall_partition(self) -> None:
        penalty, radius = self.inputs()
        masks = build_single_null_region_masks(
            penalty, radius, topology=self.topology()
        )
        self.assertEqual(masks.outboard_midplane_y, 4)
        self.assertEqual(masks.separatrix_x_index, 2)
        self.assertEqual(masks.separatrix_face_left_cell_index, 1)
        self.assertEqual(int(np.sum(masks.wall_crossing)), 1)
        self.assertEqual(int(np.sum(masks.wall_exterior)), 1)
        self.assertEqual(int(np.sum(masks.confined_edge)), 8)
        self.assertEqual(int(np.sum(masks.private_flux)), 4)
        self.assertEqual(int(np.sum(masks.scrape_off_layer)), 10)
        eligible = masks.strict_wall_interior & masks.operator_interior
        np.testing.assert_array_equal(masks.primary_partition(), eligible.astype(np.int8))
        self.assertEqual(int(np.sum(masks.separatrix_cell_band)), 12)
        self.assertEqual(int(np.sum(masks.x_point_topology_stencil)), 8)
        self.assertEqual(int(np.sum(masks.outboard_midplane)), 4)

    def test_exact_confined_separatrix_surface_uses_both_adjacent_cells(self) -> None:
        penalty, radius = self.inputs()
        masks = build_single_null_region_masks(
            penalty, radius, topology=self.topology()
        )
        left_indices = np.asarray([0, 1, 2], dtype=np.int64)
        valid = np.ones((3, 8), dtype=bool)
        valid[:, (0, 7)] = False
        surface = confined_separatrix_surface_mask(
            masks, left_indices, operator_valid_mask=valid
        )
        expected = np.zeros((3, 8), dtype=bool)
        expected[1, 2:6] = True
        np.testing.assert_array_equal(surface, expected)

        penalty[2, 4] = 0.1
        penalized = build_single_null_region_masks(
            penalty, radius, topology=self.topology()
        )
        with self.assertRaisesRegex(ValueError, "lost a core row"):
            confined_separatrix_surface_mask(
                penalized, left_indices, operator_valid_mask=valid
            )

        penalty, radius = self.inputs()
        penalty[1, 3] = 0.1
        penalized_left = build_single_null_region_masks(
            penalty, radius, topology=self.topology()
        )
        with self.assertRaisesRegex(ValueError, "lost a core row"):
            confined_separatrix_surface_mask(
                penalized_left, left_indices, operator_valid_mask=valid
            )

    def test_geometry_ambiguity_and_invalid_penalty_fail_loudly(self) -> None:
        penalty, radius = self.inputs()
        penalty[0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "closed interval"):
            build_single_null_region_masks(
                penalty, radius, topology=self.topology()
            )
        penalty[0, 0] = 0.0
        radius[3] = radius[4]
        with self.assertRaisesRegex(ValueError, "not uniquely"):
            build_single_null_region_masks(
                penalty, radius, topology=self.topology()
            )


class SurfaceIntegrationTests(unittest.TestCase):
    @staticmethod
    def result() -> PartialCombinedRadialFaceFlow:
        flow = np.empty((2, 2, 4, 5), dtype=np.float64)
        flow[0].fill(2.0)
        flow[1].fill(-3.0)
        return PartialCombinedRadialFaceFlow(
            flow=flow,
            xz_flow=flow.copy(),
            xy_flow=np.zeros_like(flow),
            valid_mask=np.ones((2, 4), dtype=bool),
            left_cell_indices=np.asarray([1, 2], dtype=np.int64),
        )

    def test_nonuniform_dy_wedge_sum_and_explicit_replication(self) -> None:
        dy = np.broadcast_to(
            np.asarray([1.0, 2.0, 3.0, 4.0])[None, :], (5, 4)
        ).copy()
        selected = np.zeros((2, 4), dtype=bool)
        selected[0, 1:3] = True
        wedge = integrate_radial_surface_flow(
            self.result(), dy, dz=0.25, face_mask=selected
        )
        np.testing.assert_allclose(
            wedge, np.asarray([12.5, -18.75]), rtol=0, atol=0
        )
        full = integrate_radial_surface_flow(
            self.result(),
            dy,
            dz=0.25,
            face_mask=selected,
            toroidal_replication=5,
        )
        np.testing.assert_allclose(full, 5.0 * wedge, rtol=0, atol=0)

    def test_invalid_or_nonfinite_selected_surface_fails(self) -> None:
        result = self.result()
        dy = np.ones((5, 4))
        selected = np.zeros((2, 4), dtype=bool)
        selected[0, 1] = True
        invalid = PartialCombinedRadialFaceFlow(
            flow=result.flow,
            xz_flow=result.xz_flow,
            xy_flow=result.xy_flow,
            valid_mask=np.zeros((2, 4), dtype=bool),
            left_cell_indices=result.left_cell_indices,
        )
        with self.assertRaisesRegex(ValueError, "operator-invalid"):
            integrate_radial_surface_flow(
                invalid, dy, dz=0.2, face_mask=selected
            )
        result.flow[:, 0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            integrate_radial_surface_flow(
                result, dy, dz=0.2, face_mask=selected
            )


class HermesUnitTests(unittest.TestCase):
    def test_source_constants_reproduce_frozen_run_normalization(self) -> None:
        scales = hermes_transport_scales()
        expected = {
            "sound_speed_m_per_s": 69205.61141651045,
            "ion_cyclotron_frequency_per_s": 95788333.03066081,
            "sound_gyroradius_m": 0.0007224847664314034,
            "particle_rate_scale_per_s": 3.612423832157018e17,
            "pressure_flow_scale_w": 2.893870527993356,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(getattr(scales, name) / value, 1.0, places=14)

    def test_unit_converters_preserve_sign_and_do_not_hide_energy_factor(self) -> None:
        scales = hermes_transport_scales()
        normalized = np.asarray([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(
            normalized_particle_flow_to_si(normalized, scales),
            normalized * scales.particle_rate_scale_per_s,
            rtol=0,
            atol=0,
        )
        pressure = normalized_pressure_flow_to_si(normalized, scales)
        np.testing.assert_allclose(
            pressure,
            normalized * scales.pressure_flow_scale_w,
            rtol=0,
            atol=0,
        )
        internal_energy = normalized_pressure_flow_to_si(1.5 * normalized, scales)
        np.testing.assert_allclose(internal_energy, 1.5 * pressure, rtol=0, atol=0)


class MemberwiseTransportTests(unittest.TestCase):
    @staticmethod
    def topology() -> SingleNullTopology:
        return SingleNullTopology(
            separatrix_x_index=0,
            core_lower_y=8,
            core_upper_y=23,
            pfr_lower_y=7,
            pfr_upper_y=24,
        )

    def test_actual_nonlinear_face_transport_is_evaluated_memberwise(self) -> None:
        n_batch, n_member, n_x, n_y, n_z = 1, 2, 5, 32, 9
        angle = 2.0 * np.pi * np.arange(n_z) / n_z
        cosine = np.cos(angle)
        sine = np.sin(angle)
        q = np.empty((n_batch, n_member, n_x, n_y, n_z))
        phi = np.empty_like(q)
        q[:, 0] = 2.0 + 0.5 * cosine
        q[:, 1] = 2.0 + 0.25 * cosine
        phi[:, 0] = sine
        phi[:, 1] = -sine
        geometry = np.ones((n_x, n_y), dtype=np.float64)
        zeros = np.zeros((n_x, n_y), dtype=np.float64)
        dz = toroidal_wedge_spacing(n_z, zperiod=5)
        selected = np.zeros((2, n_y), dtype=bool)
        selected[0, 1:-1] = True

        def diagnostic(q_member: np.ndarray, phi_member: np.ndarray) -> np.ndarray:
            faces = radial_exb_face_flow_partial(
                q_member,
                phi_member,
                geometry,
                geometry,
                zeros,
                geometry,
                zeros,
                geometry,
                np.zeros(n_x),
                dz=dz,
                topology=self.topology(),
                zperiod=5,
                positive=True,
            )
            return integrate_radial_surface_flow(
                faces, geometry, dz=dz, face_mask=selected
            )

        memberwise = apply_memberwise(diagnostic, q, phi, member_axis=1)
        central0 = 0.5 * (np.roll(sine, -1) - np.roll(sine, 1))
        central1 = -central0
        expected = np.asarray(
            [
                30.0 * np.sum((2.0 + 0.5 * cosine) * central0),
                30.0 * np.sum((2.0 + 0.25 * cosine) * central1),
            ]
        )[None, :]
        np.testing.assert_allclose(memberwise, expected, rtol=1e-14, atol=1e-13)

        mean_field_transport = diagnostic(np.mean(q, axis=1), np.mean(phi, axis=1))
        np.testing.assert_allclose(mean_field_transport, 0.0, rtol=0, atol=0)
        self.assertGreater(abs(float(np.mean(memberwise))), 1.0)


if __name__ == "__main__":
    unittest.main()
