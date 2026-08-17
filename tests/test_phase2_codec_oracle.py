from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_oracle import (  # noqa: E402
    CodecMetricAccumulator,
    build_preliminary_gate,
)


def synthetic_fields(*, frames: int = 4, n_z: int = 88) -> np.ndarray:
    """Return [T,C,X,Y,Z] fields with known material k=1,4,6 bands."""

    z = np.arange(n_z, dtype=np.float64)
    fields = np.empty((frames, 5, 2, 3, n_z), dtype=np.float64)
    offsets = np.asarray([2.0, 1.2, 1.5, 0.3, -0.2])
    amplitudes = np.asarray([0.12, 0.08, 0.10, 0.06, 0.05])
    field_phases = np.asarray([0.0, 0.2, -0.3, 0.5, -0.1])
    for time in range(frames):
        temporal_phase = 0.37 * time
        for channel in range(5):
            line = offsets[channel]
            for mode, scale in ((1, 1.0), (4, 0.8), (6, 0.7)):
                line = line + amplitudes[channel] * scale * np.cos(
                    2.0 * np.pi * mode * z / n_z
                    + temporal_phase
                    + field_phases[channel]
                )
            for x in range(2):
                for y in range(3):
                    fields[time, channel, x, y] = line + 0.002 * x - 0.001 * y
    return fields


def block_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
) -> tuple[dict, list[dict]]:
    block_records = []
    overall_accumulator = CodecMetricAccumulator(n_z=truth.shape[-1])
    for _ in range(8):
        accumulator = CodecMetricAccumulator(n_z=truth.shape[-1])
        accumulator.update(truth, reconstruction, truth, reconstruction)
        overall_accumulator.merge(accumulator)
        block_records.append(accumulator.finalize())
    return overall_accumulator.finalize(), block_records


class CodecAccumulatorTests(unittest.TestCase):
    def test_identity_round_trip_passes_and_obeys_parseval(self) -> None:
        truth = synthetic_fields()
        overall, blocks = block_metrics(truth, truth.copy())
        gate = build_preliminary_gate(overall, blocks)
        self.assertEqual(gate["preliminary_status"], "pass")
        self.assertEqual(
            gate["full_codec_acceptance"],
            "blocked_pending_authoritative_transport",
        )
        for field in ("Ne", "Te", "Ti", "phi", "Vi"):
            metrics = overall["field_metrics_legacy_standardized"][field]
            self.assertAlmostEqual(metrics["rmse"], 0.0, places=15)
            self.assertAlmostEqual(metrics["variance_ratio"], 1.0, places=13)

        density_power = np.asarray(
            overall["toroidal_spectral_curves_linear_coordinates"]["Ne"][
                "truth_power"
            ]
        )
        self.assertAlmostEqual(
            float(np.sum(density_power)),
            float(np.mean(truth[:, 0] ** 2)),
            places=13,
        )
        transfer_coherence = np.asarray(
            overall["toroidal_spectral_curves_linear_coordinates"]["Ne"][
                "truth_to_reconstruction_coherence"
            ]
        )
        np.testing.assert_allclose(
            transfer_coherence[[1, 4, 6]],
            1.0,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_merge_matches_one_pass_accumulation(self) -> None:
        truth = synthetic_fields(frames=6)
        first = CodecMetricAccumulator(n_z=truth.shape[-1])
        second = CodecMetricAccumulator(n_z=truth.shape[-1])
        first.update(truth[:2], truth[:2], truth[:2], truth[:2])
        second.update(truth[2:], truth[2:], truth[2:], truth[2:])
        first.merge(second)

        direct = CodecMetricAccumulator(n_z=truth.shape[-1])
        direct.update(truth, truth, truth, truth)
        merged_result = first.finalize()
        direct_result = direct.finalize()
        self.assertEqual(merged_result["frames"], direct_result["frames"])
        for field in ("Ne", "Te", "Ti", "phi", "Vi"):
            for metric in ("rmse", "mae", "bias", "truth_variance"):
                self.assertAlmostEqual(
                    merged_result["field_metrics_legacy_standardized"][field][
                        metric
                    ],
                    direct_result["field_metrics_legacy_standardized"][field][
                        metric
                    ],
                    places=13,
                )
            np.testing.assert_allclose(
                merged_result["toroidal_spectral_curves_linear_coordinates"][
                    field
                ]["truth_power"],
                direct_result["toroidal_spectral_curves_linear_coordinates"][
                    field
                ]["truth_power"],
                rtol=1e-13,
                atol=1e-13,
            )

    def test_removing_material_modes_fails_spectral_gate(self) -> None:
        truth = synthetic_fields()
        coefficients = np.fft.rfft(truth, axis=-1)
        coefficients[..., 4:] = 0.0
        reconstruction = np.fft.irfft(
            coefficients, n=truth.shape[-1], axis=-1
        )
        overall, blocks = block_metrics(truth, reconstruction)
        gate = build_preliminary_gate(overall, blocks)
        self.assertFalse(gate["spectral_transfer"]["pass"])
        self.assertEqual(gate["preliminary_status"], "fail")
        check = gate["spectral_transfer"]["checks"]["Ne"]["coherent_study"]
        self.assertTrue(check["material_overall"])
        self.assertFalse(check["overall_pass_if_material"])

    def test_phase_shift_preserves_power_but_fails_cross_field_gate(self) -> None:
        truth = synthetic_fields()
        reconstruction = truth.copy()
        reconstruction[:, 3] = np.roll(reconstruction[:, 3], 2, axis=-1)
        overall, blocks = block_metrics(truth, reconstruction)
        gate = build_preliminary_gate(overall, blocks)
        ne_phi = overall["cross_field_band_summaries"]["Ne-phi"]
        self.assertAlmostEqual(
            overall["field_band_summaries"]["phi"]["coherent_study"][
                "power_ratio"
            ],
            1.0,
            places=13,
        )
        self.assertGreater(
            ne_phi["upper_study"][
                "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
            ],
            15.0,
        )
        self.assertFalse(gate["cross_field"]["pass"])

    def test_nonpositive_density_is_counted_without_clipping(self) -> None:
        truth = synthetic_fields(frames=1)
        reconstruction = truth.copy()
        reconstruction[0, 0, 0, 0, 0] = -0.25
        accumulator = CodecMetricAccumulator(n_z=truth.shape[-1])
        accumulator.update(truth, reconstruction, truth, reconstruction)
        density = accumulator.finalize()["density_linear_reconstruction"]
        self.assertEqual(density["nonpositive_cell_count"], 1)
        self.assertAlmostEqual(density["minimum"], -0.25)


if __name__ == "__main__":
    unittest.main()
