"""Known-answer tests for the frozen B5 residual-audit measurements."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tcv_diagnostics.b5_residual_audit import (
    B5_FIELDS,
    audit_training_residual,
    axisymmetric_residual_bias,
    cross_field_statistics,
    curve_length_summary,
    residual_fluctuation,
    spatial_autocorrelation,
    toroidal_power_statistics,
    write_residual_audit_figures,
)


def _signal(*, targets: int = 12, x: int = 4, y: int = 3, z: int = 16) -> np.ndarray:
    rng = np.random.default_rng(240315)
    return rng.normal(size=(targets, len(B5_FIELDS), x, y, z)).astype(np.float32)


def test_axisymmetric_bias_and_fluctuation_preserve_nonzero_target_mean() -> None:
    residual = _signal()
    imposed = np.arange(len(B5_FIELDS), dtype=np.float32)[:, None, None] + 0.75
    residual += imposed[None, ..., None]
    bias = axisymmetric_residual_bias(residual)
    fluctuation = residual_fluctuation(residual, bias)
    np.testing.assert_allclose(
        np.mean(fluctuation, axis=(0, 4), dtype=np.float64),
        0.0,
        atol=2.0e-7,
    )
    assert np.max(np.abs(bias)) > 1.0


def test_circular_toroidal_acf_recovers_known_k1_cosine() -> None:
    targets, x, y, z = 4, 2, 2, 16
    phase = 2.0 * np.pi * np.arange(z) / z
    values = np.broadcast_to(
        np.cos(phase), (targets, len(B5_FIELDS), x, y, z)
    ).astype(np.float32)
    record, raw = spatial_autocorrelation(values, axis="stored_toroidal_z")
    expected = np.cos(2.0 * np.pi * np.arange(z // 2 + 1) / z)
    np.testing.assert_allclose(
        raw["correlation"],
        np.broadcast_to(expected, raw["correlation"].shape),
        atol=2.0e-7,
    )
    assert record["fields"]["Ne"]["length_summary"]["first_nonpositive_lag"] == 4


def test_curve_summary_uses_three_consecutive_near_zero_lags() -> None:
    summary = curve_length_summary(np.asarray([1.0, 0.4, 0.09, -0.08, 0.05, 0.2]))
    assert summary["first_at_or_below_one_over_e_lag"] == 2
    assert summary["first_nonpositive_lag"] == 3
    assert summary["first_stable_near_zero_lag"] == 2
    assert summary["stable_near_zero_censored"] is False


def test_cross_field_matrix_detects_joint_residual_direction() -> None:
    base = _signal(targets=8)
    base[:, 1] = 2.0 * base[:, 0]
    fluctuation = residual_fluctuation(base, axisymmetric_residual_bias(base))
    mask = np.ones(base.shape[2:4], dtype=bool)
    records, raw = cross_field_statistics(fluctuation, region_masks_xy={"all": mask})
    assert records["global"]["correlation_matrix"][0][1] > 0.999999
    assert records["all"]["entropy_effective_rank"] < len(B5_FIELDS)
    assert raw["cross_field__global__uncentered_gram"].shape == (5, 5)


def test_toroidal_power_uses_n_equals_5k_and_correct_band() -> None:
    targets, x, y, z = 3, 2, 2, 16
    grid = np.arange(z)
    truth = np.empty((targets, 5, x, y, z), dtype=np.float32)
    residual = np.empty_like(truth)
    for channel in range(5):
        truth[:, channel] = np.cos(2 * np.pi * 2 * grid / z)
        residual[:, channel] = np.cos(2 * np.pi * 6 * grid / z)
    record, _ = toroidal_power_statistics(residual, truth, chunk_targets=2)
    ne = record["fields"]["Ne"]["bands"]
    assert ne["k1_3"]["truth_power_fraction"] > 0.999999
    assert ne["k6_7"]["residual_power_fraction"] > 0.999999
    assert ne["k6_7"]["full_torus_n_inclusive"] == [30, 35]


def test_complete_small_audit_is_jsonable_and_writes_labeled_figures(tmp_path: Path) -> None:
    forecast = _signal(targets=70)
    truth = forecast + 0.2 * _signal(targets=70)
    x, y = truth.shape[2:4]
    masks = {
        "eligible_union": np.ones((x, y), dtype=bool),
        "left": np.indices((x, y))[0] < x // 2,
    }
    product = audit_training_residual(
        truth=truth,
        forecast=forecast,
        region_masks_xy=masks,
        cadence_microseconds=3.131905426352636,
        training_decorrelation_frames=2.2443947105846638,
        target_start=2,
        target_stop=72,
    )
    assert product.record["training_performed"] is False
    assert product.record["held_out_85606_read"] is False
    assert product.record["target_count"] == 70
    assert product.record["toroidal_support"]["mode_mapping"] == "n=5k"
    assert product.record["temporal_autocorrelation"]["pattern"]["maximum_lag_frames"] == 64
    assert math.isfinite(product.record["cross_field"]["global"]["entropy_effective_rank"])
    paths = write_residual_audit_figures(product.record, output_directory=tmp_path)
    assert len(paths) == 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
