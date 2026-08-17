from __future__ import annotations

import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "paper0" / "manifests" / "phase2_native_frame_oracle.json"
)
PROTOCOL = (
    ROOT / "paper0" / "protocol" / "PHASE2_NATIVE_FRAME_PROTOCOL.md"
)


class NativeFrameProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_selection_is_value_independent_quartile_rule(self) -> None:
        selection = self.manifest["frame_selection"]
        last = selection["total_frames"] - 1
        expected = [
            math.floor(fraction * last + 0.5)
            for fraction in selection["fractions"]
        ]
        self.assertEqual(expected, [0, 156, 312, 467, 623])
        self.assertEqual(selection["indices"], expected)
        self.assertFalse(selection["selection_uses_field_values"])

    def test_archive_and_toroidal_metadata_are_locked(self) -> None:
        archive = self.manifest["raw_archive"]
        self.assertEqual(archive["expected_rank_file_count"], 256)
        self.assertEqual(
            archive["mpi_decomposition"],
            {
                "NXPE": 16,
                "NYPE": 16,
                "MXSUB": 4,
                "MYSUB": 2,
                "MXG": 2,
                "MYG": 2,
            },
        )
        self.assertEqual(archive["native_z_samples"], 81)
        self.assertEqual(archive["zperiod"], 5)
        for key in (
            "bout_input_sha256",
            "bout_settings_sha256",
            "geometry_sha256",
        ):
            self.assertEqual(len(archive[key]), 64)

    def test_five_channel_closure_and_internal_energy_are_explicit(self) -> None:
        variables = self.manifest["advected_variables"]
        self.assertEqual(variables["particle"]["direct_q"], "Ne")
        self.assertEqual(
            variables["electron_pressure_advection"][
                "five_channel_reconstruction"
            ],
            "Ne * Te",
        )
        self.assertIn(
            "Ni == Ne",
            variables["ion_pressure_advection"][
                "five_channel_reconstruction"
            ],
        )
        self.assertEqual(
            variables["advected_internal_energy"]["electron"],
            "1.5 * face_flow(Pe, phi)",
        )
        self.assertFalse(
            variables["advected_internal_energy"][
                "released_as_total_heat_flux"
            ]
        )

    def test_tolerances_scope_and_blind_run_are_frozen(self) -> None:
        closure = self.manifest["closure_acceptance"]
        operator = self.manifest["operator_acceptance"]
        self.assertEqual((closure["atol"], closure["rtol"]), (1e-12, 1e-12))
        self.assertEqual(
            (operator["continuous_atol"], operator["continuous_rtol"]),
            (5e-10, 5e-10),
        )
        self.assertEqual(
            (operator["conservation_atol"], operator["conservation_rtol"]),
            (5e-12, 5e-12),
        )
        self.assertFalse(self.manifest["held_out_85606_access_allowed"])
        self.assertEqual(self.manifest["development_run"], "85604")
        scope = self.manifest["scientific_scope"]
        self.assertTrue(scope["proves_native_dynamic_range_operator_agreement"])
        self.assertFalse(scope["proves_resampling_fidelity"])
        self.assertFalse(scope["evaluates_a_learned_model"])

    def test_human_protocol_contains_machine_frozen_rules(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        for required in (
            "[0, 156, 312, 467, 623]",
            "Ni == Ne",
            "Pe == Ne * Te",
            "Pi == Ni * Ti",
            "Pi == Ne * Ti",
            "max_abs_error <= 1e-12 + 1e-12 * max_abs_reference",
            "max_abs_error <= 5e-10 + 5e-10 * max_abs_reference",
            "max_abs_residual <= 5e-12 + 5e-12 * max_abs_face_difference",
            "zperiod=5",
            "not yet released as an experimental or total",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
