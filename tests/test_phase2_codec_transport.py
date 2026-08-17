from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcv_diagnostics.codec_transport import (  # noqa: E402
    MATCHED_O1_COMPARISON,
    MATCHED_O1_TRANSPORT_THRESHOLDS,
    O1_COMPARISONS,
    STATE_PATHS,
    TRANSPORT_QUANTITIES,
    TransportComparisonAccumulator,
    build_codec_transport_geometry,
    build_matched_o1_transport_gate,
    build_o1_transport_gate,
    c5t_transport_state,
    direct_pressure_transport_state,
    evaluate_transport_state,
    per_frame_relative_l2,
)
from tcv_diagnostics.transport import (  # noqa: E402
    SingleNullTopology,
    toroidal_wedge_spacing,
)


class CodecTransportStateTests(unittest.TestCase):
    @staticmethod
    def topology() -> SingleNullTopology:
        return SingleNullTopology(
            separatrix_x_index=2,
            core_lower_y=2,
            core_upper_y=5,
            pfr_lower_y=1,
            pfr_upper_y=6,
        )

    @classmethod
    def geometry(cls, n_z: int = 9):
        n_x, n_y = 6, 8
        ones = np.ones((n_x, n_y), dtype=np.float64)
        zeros = np.zeros((n_x, n_y), dtype=np.float64)
        radius = np.asarray([0.1, 0.3, 0.8, 1.2, 1.0, 0.6, 0.2, 0.1])
        return build_codec_transport_geometry(
            jacobian=ones,
            g11=ones,
            g23=zeros,
            bxy=ones,
            z_shift=zeros,
            dy=ones,
            shift_angle=np.zeros(n_x),
            penalty_mask=zeros,
            separatrix_face_major_radius=radius,
            dz=toroidal_wedge_spacing(n_z, zperiod=5),
            topology=cls.topology(),
        )

    def test_c5t_and_direct_pressure_paths_are_distinct_and_unclipped(self) -> None:
        ne = np.asarray([[[[2.0, 3.0, 4.0]]]])
        te = np.asarray([[[[5.0, 6.0, 7.0]]]])
        ti = np.asarray([[[[-1.0, 2.0, 3.0]]]])
        phi = np.zeros_like(ne)
        c5t = c5t_transport_state(ne, te, ti, phi)
        np.testing.assert_array_equal(c5t["Pe"], ne * te)
        np.testing.assert_array_equal(c5t["Pi"], ne * ti)
        self.assertLess(float(np.min(c5t["Pi"])), 0.0)

        direct = direct_pressure_transport_state(ne, ne * te, -ne, phi)
        np.testing.assert_array_equal(direct["Pi"], -ne)
        self.assertFalse(np.array_equal(direct["Pi"], c5t["Pi"]))

    def test_geometry_constructs_strict_and_exact_separatrix_faces(self) -> None:
        geometry = self.geometry()
        np.testing.assert_array_equal(
            geometry.left_cell_indices, np.asarray([1, 2, 3])
        )
        self.assertEqual(int(np.sum(geometry.separatrix_face_mask)), 4)
        self.assertTrue(np.all(geometry.separatrix_face_mask[0, 2:6]))
        self.assertTrue(np.all(~geometry.operator_valid_mask[:, (0, 7)]))
        self.assertEqual(int(np.sum(geometry.strict_face_mask)), 18)

    def test_transport_is_reduced_after_each_state_path(self) -> None:
        n_time, n_x, n_y, n_z = 2, 6, 8, 9
        angle = 2.0 * np.pi * np.arange(n_z) / n_z
        shape = (n_time, n_x, n_y, n_z)
        ne = np.broadcast_to(2.0 + 0.2 * np.cos(angle), shape).copy()
        te = np.broadcast_to(3.0 + 0.1 * np.cos(angle), shape).copy()
        ti = np.broadcast_to(4.0 + 0.05 * np.cos(angle), shape).copy()
        phi = np.broadcast_to(np.sin(angle), shape).copy()
        outputs = evaluate_transport_state(
            c5t_transport_state(ne, te, ti, phi), self.geometry(n_z)
        )
        self.assertEqual(set(outputs), set(TRANSPORT_QUANTITIES))
        for quantity in TRANSPORT_QUANTITIES:
            self.assertEqual(
                outputs[quantity]["strict_face_contributions"].shape,
                (n_time, 18, n_z),
            )
            self.assertEqual(
                outputs[quantity]["separatrix_wedge"].shape, (n_time,)
            )
        for reduction in ("strict_face_contributions", "separatrix_wedge"):
            np.testing.assert_array_equal(
                outputs["total_internal_energy"][reduction],
                outputs["electron_internal_energy"][reduction]
                + outputs["ion_internal_energy"][reduction],
            )

    def test_per_frame_relative_l2_does_not_mix_time(self) -> None:
        reference = np.asarray([[[1.0, 0.0]], [[0.0, 2.0]]])
        candidate = np.asarray([[[2.0, 0.0]], [[0.0, 1.0]]])
        np.testing.assert_allclose(
            per_frame_relative_l2(reference, candidate), [1.0, 0.5]
        )


def fake_outputs(scale_by_path: dict[str, float], frames: int = 4):
    time = np.arange(1, frames + 1, dtype=np.float64)
    base_face = np.stack(
        [time, 0.5 * time + 0.2, -0.25 * time - 0.1], axis=1
    )[..., None]
    base_surface = time * time + 0.5 * time
    return {
        path: {
            quantity: {
                "strict_face_contributions": (
                    base_face * scale_by_path[path] * (index + 1)
                ),
                "separatrix_wedge": (
                    base_surface * scale_by_path[path] * (index + 1)
                ),
            }
            for index, quantity in enumerate(TRANSPORT_QUANTITIES)
        }
        for path in scale_by_path
    }


class CodecTransportAccumulatorTests(unittest.TestCase):
    def test_matched_gate_uses_frozen_truth_reconstruction_thresholds(self) -> None:
        def summary(reconstruction_scale: float):
            accumulator = TransportComparisonAccumulator(MATCHED_O1_COMPARISON)
            accumulator.update(
                fake_outputs(
                    {"truth": 1.0, "reconstruction": reconstruction_scale}
                )
            )
            return accumulator.finalize()

        identity = summary(1.0)
        gate = build_matched_o1_transport_gate(
            overall=identity,
            temporal_blocks=[identity] * 8,
        )
        self.assertTrue(gate["passes"])
        self.assertEqual(
            gate["thresholds"]["strict_faces"]["relative_l2_max"],
            0.25,
        )
        self.assertEqual(
            gate["thresholds"]["separatrix"]["rms_ratio"],
            [0.8, 1.2],
        )
        self.assertEqual(
            MATCHED_O1_TRANSPORT_THRESHOLDS["separatrix_block"][
                "required_passing_blocks"
            ],
            7,
        )

        biased = summary(1.3)
        failed = build_matched_o1_transport_gate(
            overall=biased,
            temporal_blocks=[biased] * 8,
        )
        self.assertFalse(failed["passes"])

    def test_accumulator_keeps_all_four_attribution_comparisons(self) -> None:
        accumulator = TransportComparisonAccumulator()
        accumulator.update(fake_outputs({path: 1.0 for path in STATE_PATHS}))
        result = accumulator.finalize()
        self.assertEqual(result["frames"], 4)
        self.assertEqual(set(result["comparisons"]), set(O1_COMPARISONS))
        for comparison in O1_COMPARISONS:
            for quantity in TRANSPORT_QUANTITIES:
                face = result["comparisons"][comparison]["quantities"][quantity][
                    "strict_faces"
                ]["metrics"]
                surface = result["comparisons"][comparison]["quantities"][
                    quantity
                ]["separatrix"]
                self.assertEqual(face["relative_l2"], 0.0)
                self.assertEqual(face["pearson_correlation"], 1.0)
                self.assertEqual(surface["metrics"]["relative_l2"], 0.0)
                self.assertEqual(surface["absolute_value_p95_ratio"], 1.0)

    def test_gate_can_pass_subgates_but_prior_failure_blocks_full_acceptance(self) -> None:
        accumulator = TransportComparisonAccumulator()
        accumulator.update(fake_outputs({path: 1.0 for path in STATE_PATHS}))
        overall = accumulator.finalize()
        blocks = []
        for _ in range(8):
            block = TransportComparisonAccumulator()
            block.update(fake_outputs({path: 1.0 for path in STATE_PATHS}))
            blocks.append(block.finalize())
        manifest = json.loads(
            (ROOT / "paper0/manifests/phase2_o1_transport_85604.json").read_text(
                encoding="utf-8"
            )
        )
        alignment = {field: 0.0 for field in ("Ne", "Te", "Ti", "phi", "Vi")}
        failed_prior = build_o1_transport_gate(
            overall=overall,
            temporal_blocks=blocks,
            input_field_max_relative_l2=alignment,
            preliminary_status="fail",
            thresholds=manifest["acceptance_gates"],
        )
        self.assertTrue(failed_prior["input_alignment"]["passes"])
        self.assertTrue(failed_prior["input_roundtrip"]["passes"])
        self.assertTrue(failed_prior["c5t_state_adequacy"]["passes"])
        self.assertTrue(failed_prior["codec_only_transport"]["passes"])
        self.assertTrue(failed_prior["authoritative_transport"]["passes"])
        self.assertFalse(failed_prior["full_codec_acceptance"]["passes"])

        passed_prior = build_o1_transport_gate(
            overall=overall,
            temporal_blocks=blocks,
            input_field_max_relative_l2=alignment,
            preliminary_status="pass",
            thresholds=manifest["acceptance_gates"],
        )
        self.assertTrue(passed_prior["full_codec_acceptance"]["passes"])

    def test_gate_rejects_missing_input_field_and_requires_eight_blocks(self) -> None:
        accumulator = TransportComparisonAccumulator()
        accumulator.update(fake_outputs({path: 1.0 for path in STATE_PATHS}))
        overall = accumulator.finalize()
        manifest = json.loads(
            (ROOT / "paper0/manifests/phase2_o1_transport_85604.json").read_text(
                encoding="utf-8"
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly the five"):
            build_o1_transport_gate(
                overall=overall,
                temporal_blocks=[overall] * 8,
                input_field_max_relative_l2={"Ne": 0.0},
                preliminary_status="fail",
                thresholds=manifest["acceptance_gates"],
            )
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            build_o1_transport_gate(
                overall=overall,
                temporal_blocks=[overall] * 7,
                input_field_max_relative_l2={
                    field: 0.0 for field in ("Ne", "Te", "Ti", "phi", "Vi")
                },
                preliminary_status="fail",
                thresholds=manifest["acceptance_gates"],
            )


if __name__ == "__main__":
    unittest.main()
