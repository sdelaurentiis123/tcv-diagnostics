"""Known-answer tests for the matched O1 codec metric core."""

from __future__ import annotations

import unittest

import numpy as np

from tcv_diagnostics.matched_codec_metrics import (
    CodecViewSpec,
    MatchedCodecAccumulator,
    build_matched_o1_view_gate,
    training_materiality,
)


C5P = CodecViewSpec(
    name="c5p_common",
    fields=("Ne", "Pe", "Pi", "phi", "Vi"),
    spectral_fields=("Ne", "Pe", "Pi", "phi", "Vi"),
    cross_pairs=(("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi")),
)


def synthetic_fields(*, frames: int = 16, n_z: int = 88) -> np.ndarray:
    z = np.arange(n_z, dtype=np.float64)
    output = np.empty((frames, 5, 2, 3, n_z), dtype=np.float64)
    offsets = np.asarray([2.0, 1.2, 1.5, 0.3, -0.2])
    amplitudes = np.asarray([0.12, 0.08, 0.10, 0.06, 0.05])
    phases = np.asarray([0.0, 0.2, -0.3, 0.5, -0.1])
    for time in range(frames):
        for channel in range(5):
            line = np.full(n_z, offsets[channel], dtype=np.float64)
            for mode, scale in ((1, 1.0), (4, 0.8), (6, 0.7)):
                line += amplitudes[channel] * scale * np.cos(
                    2.0 * np.pi * mode * z / n_z
                    + 0.37 * time
                    + phases[channel]
                )
            output[time, channel] = line[None, None, :] + np.asarray(
                [[0.0, -0.001, -0.002], [0.002, 0.001, 0.0]]
            )[..., None]
    return output


def metrics(truth: np.ndarray, reconstruction: np.ndarray) -> dict:
    accumulator = MatchedCodecAccumulator(spec=C5P, n_z=truth.shape[-1])
    accumulator.update(truth, reconstruction, truth, reconstruction)
    return accumulator.finalize()


def eight_blocks(truth: np.ndarray, reconstruction: np.ndarray) -> list[dict]:
    return [metrics(truth, reconstruction) for _ in range(8)]


class TestMatchedCodecMetrics(unittest.TestCase):
    def test_identity_passes_with_training_frozen_materiality(self) -> None:
        truth = synthetic_fields()
        training = metrics(truth, truth)
        validation = metrics(truth, truth.copy())
        gate = build_matched_o1_view_gate(
            validation_overall=validation,
            validation_blocks=eight_blocks(truth, truth.copy()),
            materiality=training_materiality(training),
        )
        self.assertTrue(gate["passes"])
        self.assertEqual(gate["materiality_source_split"], "85604_training_[0,432)")
        self.assertEqual(gate["spectral_transfer"]["applicable_check_count"], 15)
        self.assertEqual(gate["cross_field"]["applicable_check_count"], 9)
        self.assertEqual(validation["full_torus_n"][4], 20)
        self.assertEqual(validation["full_torus_n"][7], 35)

    def test_phase_shift_preserves_power_but_fails_cross_phase(self) -> None:
        truth = synthetic_fields()
        shifted = truth.copy()
        shifted[:, 3] = np.roll(shifted[:, 3], 2, axis=-1)
        validation = metrics(truth, shifted)
        gate = build_matched_o1_view_gate(
            validation_overall=validation,
            validation_blocks=eight_blocks(truth, shifted),
            materiality=training_materiality(metrics(truth, truth)),
        )
        self.assertAlmostEqual(
            validation["field_band_summaries"]["phi"]["k4_5"]["power_ratio"],
            1.0,
            places=13,
        )
        self.assertFalse(gate["cross_field"]["passes"])
        self.assertFalse(gate["passes"])

    def test_training_material_band_remains_required_on_validation(self) -> None:
        truth = synthetic_fields()
        coefficients = np.fft.rfft(truth, axis=-1)
        coefficients[..., 4:8] = 0.0
        reconstruction = np.fft.irfft(
            coefficients,
            n=truth.shape[-1],
            axis=-1,
        )
        materiality = training_materiality(metrics(truth, truth))
        gate = build_matched_o1_view_gate(
            validation_overall=metrics(truth, reconstruction),
            validation_blocks=eight_blocks(truth, reconstruction),
            materiality=materiality,
        )
        check = gate["spectral_transfer"]["checks"]["Ne"]["k4_5"]
        self.assertTrue(check["material_from_training_truth"])
        self.assertFalse(check["overall_pass_if_material"])
        self.assertFalse(gate["spectral_transfer"]["passes"])

    def test_native_view_without_phi_has_no_cross_gate(self) -> None:
        spec = CodecViewSpec(
            name="e6b_native",
            fields=("Ne", "Pe", "Pi", "NVe", "NVi", "Vort"),
            spectral_fields=("Ne", "Pe", "Pi", "NVe", "NVi", "Vort"),
            cross_pairs=(),
        )
        truth = np.concatenate(
            [synthetic_fields(), synthetic_fields()[:, :1]],
            axis=1,
        )
        accumulator = MatchedCodecAccumulator(spec=spec)
        accumulator.update(truth, truth, truth, truth)
        record = accumulator.finalize()
        gate = build_matched_o1_view_gate(
            validation_overall=record,
            validation_blocks=[record] * 8,
            materiality=training_materiality(record),
        )
        self.assertFalse(gate["cross_field"]["required"])
        self.assertTrue(gate["cross_field"]["passes"])
        self.assertTrue(gate["passes"])

    def test_merge_matches_single_pass_and_nonpositive_density_fails(self) -> None:
        truth = synthetic_fields()
        first = MatchedCodecAccumulator(spec=C5P)
        second = MatchedCodecAccumulator(spec=C5P)
        first.update(truth[:8], truth[:8], truth[:8], truth[:8])
        second.update(truth[8:], truth[8:], truth[8:], truth[8:])
        first.merge(second)
        direct = metrics(truth, truth)
        merged = first.finalize()
        self.assertEqual(merged["frames"], direct["frames"])
        self.assertAlmostEqual(
            merged["field_metrics_standardized"]["Pe"]["truth_variance"],
            direct["field_metrics_standardized"]["Pe"]["truth_variance"],
            places=13,
        )

        broken = truth.copy()
        broken[0, 0, 0, 0, 0] = -0.1
        validation = metrics(truth, broken)
        gate = build_matched_o1_view_gate(
            validation_overall=validation,
            validation_blocks=eight_blocks(truth, broken),
            materiality=training_materiality(direct),
        )
        self.assertEqual(
            validation["density_physical_reconstruction"][
                "nonpositive_cell_count"
            ],
            1,
        )
        self.assertFalse(gate["density_positivity"]["passes"])
        self.assertFalse(gate["passes"])


if __name__ == "__main__":
    unittest.main()
