"""Frozen training-residual measurements for Paper 0 B5.

This module contains no checkpoint loading, model inference, dataset routing,
or validation access.  It measures an already constructed canonical residual
tensor with axes ``[target, field, x, y, stored_toroidal_z]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


B5_FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
B5_ABSOLUTE_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
B5_TOROIDAL_BANDS = {
    "k0": (0, 0),
    "k1_3": (1, 3),
    "k4_5": (4, 5),
    "k6_7": (6, 7),
    "k_ge_8": (8, None),
}


@dataclass(frozen=True)
class ResidualAuditProduct:
    """Compact JSON record plus sufficient statistics for immutable storage."""

    record: dict[str, Any]
    raw_accumulators: dict[str, np.ndarray]


def _canonical(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    if array.ndim != 5 or array.shape[1] != len(B5_FIELDS):
        raise ValueError(
            f"{name} must have axes [target,{len(B5_FIELDS)},x,y,z]"
        )
    if array.shape[0] < 2 or any(length < 2 for length in array.shape[2:]):
        raise ValueError(f"{name} dimensions are too short for an audit")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def axisymmetric_residual_bias(residual: np.ndarray) -> np.ndarray:
    """Return ``mean(target, z)`` with axes ``[field,x,y]``."""

    values = _canonical("residual", residual)
    return np.mean(values, axis=(0, 4), dtype=np.float64)


def residual_fluctuation(
    residual: np.ndarray,
    axisymmetric_bias: np.ndarray,
) -> np.ndarray:
    """Subtract the frozen axisymmetric residual bias without changing axes."""

    values = _canonical("residual", residual)
    bias = np.asarray(axisymmetric_bias, dtype=np.float64)
    if bias.shape != values.shape[1:4] or not np.all(np.isfinite(bias)):
        raise ValueError("axisymmetric residual bias shape or values differ")
    return np.asarray(values - bias[None, ..., None], dtype=np.float32)


def _selected_values(values: np.ndarray, mask_xy: np.ndarray | None) -> np.ndarray:
    if mask_xy is None:
        return values.reshape(-1)
    mask = np.asarray(mask_xy, dtype=bool)
    if mask.shape != values.shape[1:3] or not np.any(mask):
        raise ValueError("region mask shape differs or is empty")
    return values[:, mask, :].reshape(-1)


def _scale_record(residual: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    residual_values = np.asarray(residual, dtype=np.float32).reshape(-1)
    truth_values = np.asarray(truth, dtype=np.float32).reshape(-1)
    if residual_values.shape != truth_values.shape or residual_values.size == 0:
        raise ValueError("residual/truth scale arrays differ or are empty")
    mean = float(np.mean(residual_values, dtype=np.float64))
    variance = float(np.var(residual_values, dtype=np.float64))
    truth_variance = float(np.var(truth_values, dtype=np.float64))
    absolute = np.abs(residual_values)
    quantiles = np.quantile(absolute, B5_ABSOLUTE_QUANTILES)
    return {
        "sample_count": int(residual_values.size),
        "bias": mean,
        "MAE": float(np.mean(absolute, dtype=np.float64)),
        "RMS": float(
            np.sqrt(np.mean(residual_values * residual_values, dtype=np.float64))
        ),
        "population_standard_deviation": math.sqrt(max(0.0, variance)),
        "absolute_quantiles": {
            f"q{int(round(1000 * quantile)):03d}": float(value)
            for quantile, value in zip(B5_ABSOLUTE_QUANTILES, quantiles)
        },
        "maximum_absolute_value": float(np.max(absolute)),
        "target_population_variance": truth_variance,
        "residual_to_target_variance_ratio": (
            variance / truth_variance if truth_variance > 0.0 else None
        ),
    }


def residual_scale_statistics(
    residual: np.ndarray,
    truth: np.ndarray,
    *,
    region_masks_xy: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return global/region scale summaries and pointwise residual spread."""

    residual_values = _canonical("residual", residual)
    truth_values = _canonical("truth", truth)
    if residual_values.shape != truth_values.shape:
        raise ValueError("residual and truth tensors differ")
    masks = {
        str(name): np.asarray(mask, dtype=bool)
        for name, mask in region_masks_xy.items()
    }
    if not masks:
        raise ValueError("at least one geometry region is required")
    for mask in masks.values():
        if mask.shape != residual_values.shape[2:4] or not np.any(mask):
            raise ValueError("geometry region shape differs or is empty")

    global_records: dict[str, Any] = {}
    region_records: dict[str, Any] = {name: {} for name in masks}
    for channel, field in enumerate(B5_FIELDS):
        global_records[field] = _scale_record(
            residual_values[:, channel], truth_values[:, channel]
        )
        for name, mask in masks.items():
            region_records[name][field] = _scale_record(
                _selected_values(residual_values[:, channel], mask),
                _selected_values(truth_values[:, channel], mask),
            )

    pointwise = np.std(
        residual_values,
        axis=(0, 4),
        dtype=np.float64,
        ddof=0,
    )
    percentiles = (0.0, 0.05, 0.5, 0.9, 0.95, 1.0)
    heteroscedasticity: dict[str, Any] = {}
    for channel, field in enumerate(B5_FIELDS):
        values = pointwise[channel].reshape(-1)
        quantiles = np.quantile(values, percentiles)
        q05 = float(quantiles[1])
        heteroscedasticity[field] = {
            "pointwise_population_standard_deviation_percentiles": {
                f"q{int(round(100 * quantile)):03d}": float(value)
                for quantile, value in zip(percentiles, quantiles)
            },
            "q95_to_q05_ratio": (
                float(quantiles[4]) / q05 if q05 > 0.0 else None
            ),
        }
    return (
        {
            "global": global_records,
            "regions": region_records,
            "heteroscedasticity": heteroscedasticity,
        },
        pointwise,
        np.asarray(percentiles, dtype=np.float64),
    )


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.einsum(
            "...,...->",
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
            dtype=np.float64,
            optimize=True,
        )
    )


def _normalized_curve(
    numerator: np.ndarray,
    left_energy: np.ndarray,
    right_energy: np.ndarray,
) -> np.ndarray:
    denominator = np.sqrt(left_energy * right_energy)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("autocorrelation has zero or invalid overlap energy")
    result = numerator / denominator
    if result.ndim == 1:
        result[0] = 1.0
    else:
        result[..., 0] = 1.0
    return np.clip(result, -1.0, 1.0)


def curve_length_summary(
    correlation: np.ndarray,
    *,
    stable_threshold: float = 0.10,
    stable_consecutive: int = 3,
) -> dict[str, Any]:
    """Summarize a lag curve using the frozen crossing conventions."""

    curve = np.asarray(correlation, dtype=np.float64)
    if (
        curve.ndim != 1
        or curve.size < 2
        or not np.all(np.isfinite(curve))
        or not math.isclose(float(curve[0]), 1.0, abs_tol=1.0e-8)
    ):
        raise ValueError("correlation curve must be finite, one-dimensional, and start at 1")
    if not 0.0 < float(stable_threshold) < 1.0 or stable_consecutive < 1:
        raise ValueError("stable-near-zero convention is invalid")

    nonpositive = np.flatnonzero(curve[1:] <= 0.0)
    one_over_e = np.flatnonzero(curve[1:] <= math.exp(-1.0))
    stable = None
    for start in range(1, curve.size - stable_consecutive + 1):
        if np.all(np.abs(curve[start : start + stable_consecutive]) <= stable_threshold):
            stable = start
            break
    first_nonpositive = int(nonpositive[0] + 1) if nonpositive.size else None
    first_one_over_e = int(one_over_e[0] + 1) if one_over_e.size else None
    positive_stop = first_nonpositive if first_nonpositive is not None else curve.size - 1
    positive_curve = np.maximum(curve[: positive_stop + 1], 0.0)
    integral = float(np.trapz(positive_curve, dx=1.0))
    return {
        "first_nonpositive_lag": first_nonpositive,
        "first_at_or_below_one_over_e_lag": first_one_over_e,
        "first_stable_near_zero_lag": stable,
        "stable_near_zero_absolute_threshold": float(stable_threshold),
        "stable_near_zero_consecutive_lags": int(stable_consecutive),
        "positive_lobe_integral_scale_lags": integral,
        "nonpositive_crossing_censored": first_nonpositive is None,
        "one_over_e_crossing_censored": first_one_over_e is None,
        "stable_near_zero_censored": stable is None,
    }


def spatial_autocorrelation(
    fluctuation: np.ndarray,
    *,
    axis: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pooled valid-overlap (x/y) or circular (z) spatial correlation."""

    values = _canonical("residual fluctuation", fluctuation)
    axis_map = {"x": 2, "y": 3, "stored_toroidal_z": 4}
    if axis not in axis_map:
        raise ValueError(f"unsupported residual spatial axis {axis!r}")
    canonical_axis = axis_map[axis]
    field_axis = canonical_axis - 1
    extent = values.shape[canonical_axis]
    max_lag = extent // 2
    numerator = np.zeros((len(B5_FIELDS), max_lag + 1), dtype=np.float64)
    left_energy = np.zeros_like(numerator)
    right_energy = np.zeros_like(numerator)

    for channel in range(len(B5_FIELDS)):
        field_values = values[:, channel]
        for lag in range(max_lag + 1):
            if lag == 0:
                left = right = field_values
                numerator[channel, lag] = _dot(left, right)
                left_energy[channel, lag] = numerator[channel, lag]
                right_energy[channel, lag] = numerator[channel, lag]
                continue
            if axis == "stored_toroidal_z":
                numerator[channel, lag] = _dot(
                    field_values[..., :-lag], field_values[..., lag:]
                ) + _dot(field_values[..., -lag:], field_values[..., :lag])
                total = _dot(field_values, field_values)
                left_energy[channel, lag] = total
                right_energy[channel, lag] = total
                continue
            left_slice = [slice(None)] * field_values.ndim
            right_slice = [slice(None)] * field_values.ndim
            left_slice[field_axis] = slice(None, -lag)
            right_slice[field_axis] = slice(lag, None)
            left = field_values[tuple(left_slice)]
            right = field_values[tuple(right_slice)]
            numerator[channel, lag] = _dot(left, right)
            left_energy[channel, lag] = _dot(left, left)
            right_energy[channel, lag] = _dot(right, right)

    correlation = _normalized_curve(numerator, left_energy, right_energy)
    lags = np.arange(max_lag + 1, dtype=np.int64)
    fields = {
        field: {
            "correlation": correlation[channel].tolist(),
            "length_summary": curve_length_summary(correlation[channel]),
        }
        for channel, field in enumerate(B5_FIELDS)
    }
    return (
        {
            "axis": axis,
            "estimator": (
                "pooled_circular_normalized_correlation"
                if axis == "stored_toroidal_z"
                else "pooled_valid_overlap_normalized_correlation"
            ),
            "axis_extent_cells": int(extent),
            "maximum_lag_cells": int(max_lag),
            "lags_cells": lags.tolist(),
            "fields": fields,
        },
        {
            "lags": lags,
            "numerator": numerator,
            "left_energy": left_energy,
            "right_energy": right_energy,
            "correlation": correlation,
        },
    )


def temporal_pattern_autocorrelation(
    fluctuation: np.ndarray,
    *,
    maximum_lag: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Pooled field-pattern correlation through target time."""

    values = _canonical("residual fluctuation", fluctuation)
    max_lag = min(int(maximum_lag), values.shape[0] - 1)
    if max_lag < 1:
        raise ValueError("temporal maximum lag must be positive")
    numerator = np.zeros((len(B5_FIELDS), max_lag + 1), dtype=np.float64)
    left_energy = np.zeros_like(numerator)
    right_energy = np.zeros_like(numerator)
    for channel in range(len(B5_FIELDS)):
        series = values[:, channel]
        for lag in range(max_lag + 1):
            left = series if lag == 0 else series[:-lag]
            right = series if lag == 0 else series[lag:]
            numerator[channel, lag] = _dot(left, right)
            left_energy[channel, lag] = _dot(left, left)
            right_energy[channel, lag] = _dot(right, right)
    correlation = _normalized_curve(numerator, left_energy, right_energy)
    lags = np.arange(max_lag + 1, dtype=np.int64)
    return (
        {
            "estimator": "pooled_valid_overlap_normalized_pattern_correlation",
            "maximum_lag_frames": int(max_lag),
            "lags_frames": lags.tolist(),
            "fields": {
                field: {
                    "correlation": correlation[channel].tolist(),
                    "length_summary": curve_length_summary(correlation[channel]),
                }
                for channel, field in enumerate(B5_FIELDS)
            },
        },
        {
            "lags": lags,
            "numerator": numerator,
            "left_energy": left_energy,
            "right_energy": right_energy,
            "correlation": correlation,
        },
    )


def residual_rms_autocorrelation(
    residual: np.ndarray,
    *,
    maximum_lag: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Mean-centered ACF of each field's per-target residual RMS."""

    values = _canonical("residual", residual)
    series = np.sqrt(np.mean(values * values, axis=(2, 3, 4), dtype=np.float64))
    centered = series - np.mean(series, axis=0, keepdims=True, dtype=np.float64)
    max_lag = min(int(maximum_lag), values.shape[0] - 1)
    numerator = np.zeros((len(B5_FIELDS), max_lag + 1), dtype=np.float64)
    left_energy = np.zeros_like(numerator)
    right_energy = np.zeros_like(numerator)
    for channel in range(len(B5_FIELDS)):
        field_series = centered[:, channel]
        for lag in range(max_lag + 1):
            left = field_series if lag == 0 else field_series[:-lag]
            right = field_series if lag == 0 else field_series[lag:]
            numerator[channel, lag] = _dot(left, right)
            left_energy[channel, lag] = _dot(left, left)
            right_energy[channel, lag] = _dot(right, right)
    correlation = _normalized_curve(numerator, left_energy, right_energy)
    lags = np.arange(max_lag + 1, dtype=np.int64)
    return (
        {
            "estimator": "mean_centered_per_target_RMS_valid_overlap_correlation",
            "maximum_lag_frames": int(max_lag),
            "lags_frames": lags.tolist(),
            "fields": {
                field: {
                    "correlation": correlation[channel].tolist(),
                    "length_summary": curve_length_summary(correlation[channel]),
                }
                for channel, field in enumerate(B5_FIELDS)
            },
        },
        {
            "lags": lags,
            "RMS_series": series,
            "numerator": numerator,
            "left_energy": left_energy,
            "right_energy": right_energy,
            "correlation": correlation,
        },
    )


def _cross_field_one(
    fluctuation: np.ndarray,
    mask_xy: np.ndarray | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    selected = []
    for channel in range(len(B5_FIELDS)):
        selected.append(_selected_values(fluctuation[:, channel], mask_xy))
    count = int(selected[0].size)
    if any(values.size != count for values in selected):
        raise ValueError("cross-field selections differ")
    sums = np.asarray(
        [np.sum(values, dtype=np.float64) for values in selected], dtype=np.float64
    )
    gram = np.empty((len(B5_FIELDS), len(B5_FIELDS)), dtype=np.float64)
    for left in range(len(B5_FIELDS)):
        for right in range(left, len(B5_FIELDS)):
            value = _dot(selected[left], selected[right])
            gram[left, right] = gram[right, left] = value
    centered = gram - np.outer(sums, sums) / count
    diagonal = np.diag(centered)
    if np.any(diagonal <= 0.0):
        raise ValueError("cross-field residual variance is nonpositive")
    correlation = centered / np.sqrt(np.outer(diagonal, diagonal))
    correlation = np.clip((correlation + correlation.T) * 0.5, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    nonnegative = np.clip(eigenvalues, 0.0, None)
    probability = nonnegative / np.sum(nonnegative)
    entropy = -float(np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
    effective_rank = math.exp(entropy)
    participation = float(np.sum(nonnegative) ** 2 / np.sum(nonnegative**2))
    return (
        {
            "sample_count": count,
            "field_order": list(B5_FIELDS),
            "correlation_matrix": correlation.tolist(),
            "eigenvalues_descending": eigenvalues.tolist(),
            "entropy_effective_rank": effective_rank,
            "participation_ratio_effective_rank": participation,
        },
        {
            "sample_count": np.asarray([count], dtype=np.int64),
            "sums": sums,
            "uncentered_gram": gram,
            "centered_gram": centered,
            "correlation": correlation,
            "eigenvalues_descending": eigenvalues,
        },
    )


def cross_field_statistics(
    fluctuation: np.ndarray,
    *,
    region_masks_xy: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Global and geometry-resolved five-field residual correlations."""

    values = _canonical("residual fluctuation", fluctuation)
    records: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    selections: dict[str, np.ndarray | None] = {"global": None}
    selections.update(
        {str(name): np.asarray(mask, dtype=bool) for name, mask in region_masks_xy.items()}
    )
    for name, mask in selections.items():
        record, accumulators = _cross_field_one(values, mask)
        records[name] = record
        for key, array in accumulators.items():
            raw[f"cross_field__{name}__{key}"] = array
    return records, raw


def toroidal_power_statistics(
    residual: np.ndarray,
    truth: np.ndarray,
    *,
    chunk_targets: int = 8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Report Parseval-weighted stored-toroidal power by frozen k band."""

    residual_values = _canonical("residual", residual)
    truth_values = _canonical("truth", truth)
    if residual_values.shape != truth_values.shape:
        raise ValueError("residual and truth tensors differ for toroidal power")
    if int(chunk_targets) < 1:
        raise ValueError("toroidal FFT chunk size must be positive")
    n_z = residual_values.shape[-1]
    n_k = n_z // 2 + 1
    residual_power = np.zeros((len(B5_FIELDS), n_k), dtype=np.float64)
    truth_power = np.zeros_like(residual_power)
    for start in range(0, residual_values.shape[0], int(chunk_targets)):
        stop = min(start + int(chunk_targets), residual_values.shape[0])
        for source, accumulator in (
            (residual_values, residual_power),
            (truth_values, truth_power),
        ):
            coefficients = np.fft.rfft(source[start:stop], axis=-1)
            accumulator += np.sum(
                coefficients.real * coefficients.real
                + coefficients.imag * coefficients.imag,
                axis=(0, 2, 3),
                dtype=np.float64,
            )
    weights = np.full(n_k, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if n_z % 2 == 0:
        weights[-1] = 1.0
    residual_weighted = residual_power * weights[None]
    truth_weighted = truth_power * weights[None]

    fields: dict[str, Any] = {}
    for channel, field in enumerate(B5_FIELDS):
        residual_total = float(np.sum(residual_weighted[channel]))
        truth_total = float(np.sum(truth_weighted[channel]))
        if residual_total <= 0.0 or truth_total <= 0.0:
            raise ValueError("toroidal power is nonpositive")
        bands: dict[str, Any] = {}
        for label, (start, frozen_stop) in B5_TOROIDAL_BANDS.items():
            if start >= n_k:
                residual_band = truth_band = 0.0
                stop = n_k - 1
            else:
                stop = n_k - 1 if frozen_stop is None else min(frozen_stop, n_k - 1)
                residual_band = float(np.sum(residual_weighted[channel, start : stop + 1]))
                truth_band = float(np.sum(truth_weighted[channel, start : stop + 1]))
            bands[label] = {
                "stored_k_inclusive": [int(start), int(stop)],
                "full_torus_n_inclusive": [int(5 * start), int(5 * stop)],
                "residual_power_fraction": residual_band / residual_total,
                "truth_power_fraction": truth_band / truth_total,
                "residual_to_truth_power_ratio": (
                    residual_band / truth_band if truth_band > 0.0 else None
                ),
            }
        fields[field] = {"bands": bands}
    return (
        {
            "stored_toroidal_cells": int(n_z),
            "stored_k_maximum": int(n_k - 1),
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "parseval_positive_frequency_weighting": True,
            "fields": fields,
        },
        {
            "stored_k": np.arange(n_k, dtype=np.int64),
            "parseval_weights": weights,
            "residual_unweighted_positive_frequency_power": residual_power,
            "truth_unweighted_positive_frequency_power": truth_power,
        },
    )


def _add_time_units(
    temporal_record: dict[str, Any],
    *,
    cadence_microseconds: float,
    decorrelation_frames: float,
) -> None:
    for field in B5_FIELDS:
        summary = temporal_record["fields"][field]["length_summary"]
        for key in (
            "first_nonpositive_lag",
            "first_at_or_below_one_over_e_lag",
            "first_stable_near_zero_lag",
            "positive_lobe_integral_scale_lags",
        ):
            value = summary[key]
            stem = key[:-4] if key.endswith("_lag") else key.replace("_lags", "")
            summary[f"{stem}_microseconds"] = (
                None if value is None else float(value) * cadence_microseconds
            )
            summary[f"{stem}_training_decorrelation_times"] = (
                None if value is None else float(value) / decorrelation_frames
            )


def audit_training_residual(
    *,
    truth: np.ndarray,
    forecast: np.ndarray,
    region_masks_xy: Mapping[str, np.ndarray],
    cadence_microseconds: float,
    training_decorrelation_frames: float,
    target_start: int,
    target_stop: int,
) -> ResidualAuditProduct:
    """Run every prospectively frozen B5 training-residual measurement."""

    truth_values = _canonical("truth", truth)
    forecast_values = _canonical("forecast", forecast)
    if truth_values.shape != forecast_values.shape:
        raise ValueError("truth and forecast canonical shapes differ")
    if target_stop - target_start != truth_values.shape[0]:
        raise ValueError("target interval/count differs")
    if not math.isfinite(cadence_microseconds) or cadence_microseconds <= 0.0:
        raise ValueError("cadence must be finite and positive")
    if not math.isfinite(training_decorrelation_frames) or training_decorrelation_frames <= 0.0:
        raise ValueError("training decorrelation time must be finite and positive")

    residual = np.asarray(truth_values - forecast_values, dtype=np.float32)
    bias = axisymmetric_residual_bias(residual)
    fluctuation = residual_fluctuation(residual, bias)
    scale, pointwise, pointwise_percentiles = residual_scale_statistics(
        residual,
        truth_values,
        region_masks_xy=region_masks_xy,
    )
    raw: dict[str, np.ndarray] = {
        "axisymmetric_residual_bias__field_x_y": bias,
        "pointwise_residual_population_standard_deviation__field_x_y": pointwise,
        "pointwise_standard_deviation_percentiles": pointwise_percentiles,
    }

    spatial: dict[str, Any] = {}
    for axis in ("x", "y", "stored_toroidal_z"):
        record, accumulators = spatial_autocorrelation(fluctuation, axis=axis)
        spatial[axis] = record
        for key, array in accumulators.items():
            raw[f"spatial__{axis}__{key}"] = array

    temporal_pattern, temporal_pattern_raw = temporal_pattern_autocorrelation(
        fluctuation, maximum_lag=64
    )
    temporal_rms, temporal_rms_raw = residual_rms_autocorrelation(
        residual, maximum_lag=64
    )
    _add_time_units(
        temporal_pattern,
        cadence_microseconds=float(cadence_microseconds),
        decorrelation_frames=float(training_decorrelation_frames),
    )
    _add_time_units(
        temporal_rms,
        cadence_microseconds=float(cadence_microseconds),
        decorrelation_frames=float(training_decorrelation_frames),
    )
    for key, array in temporal_pattern_raw.items():
        raw[f"temporal_pattern__{key}"] = array
    for key, array in temporal_rms_raw.items():
        raw[f"temporal_RMS__{key}"] = array

    cross_field, cross_raw = cross_field_statistics(
        fluctuation, region_masks_xy=region_masks_xy
    )
    raw.update(cross_raw)
    toroidal, toroidal_raw = toroidal_power_statistics(residual, truth_values)
    raw.update({f"toroidal__{key}": value for key, value in toroidal_raw.items()})

    record = {
        "schema_version": 1,
        "scope": "B5_frozen_H1_training_residual_audit_85604",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_frames_read": False,
        "training_performed": False,
        "B5_training_authorized": False,
        "residual_definition": "standardized_truth_minus_frozen_H1_standardized_forecast",
        "residual_interpretation": "in_sample_parent_model_error_not_identified_aleatoric_noise",
        "field_order": list(B5_FIELDS),
        "target_frames": [int(target_start), int(target_stop)],
        "target_count": int(truth_values.shape[0]),
        "canonical_shape": [int(value) for value in truth_values.shape],
        "cadence_microseconds": float(cadence_microseconds),
        "training_decorrelation_frames": float(training_decorrelation_frames),
        "axisymmetric_bias_definition": "mean_over_target_and_stored_toroidal_z_per_field_x_y",
        "scale": scale,
        "spatial_autocorrelation": spatial,
        "temporal_autocorrelation": {
            "pattern": temporal_pattern,
            "per_target_residual_RMS": temporal_rms,
        },
        "cross_field": cross_field,
        "toroidal_support": toroidal,
        "scientific_boundaries": {
            "validation_used": False,
            "held_out_85606_used": False,
            "irreducible_uncertainty_identified": False,
            "physics_metric_used_as_training_loss": False,
            "architecture_selected": False,
        },
    }
    return ResidualAuditProduct(record=record, raw_accumulators=raw)


def write_residual_audit_figures(
    record: Mapping[str, Any],
    *,
    output_directory: Path,
) -> list[Path]:
    """Write the five required fully labeled audit figures without a browser."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = plt.get_cmap("tab10").colors

    scale = record["scale"]["global"]
    x = np.arange(len(B5_FIELDS), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for offset, (label, key) in enumerate(
        (("RMS", "RMS"), ("MAE", "MAE"), ("|bias|", "bias"))
    ):
        values = [abs(float(scale[field][key])) for field in B5_FIELDS]
        axis.bar(x + (offset - 1) * width, values, width=width, label=label)
    axis.set_xticks(x, B5_FIELDS)
    axis.set_ylabel("Standardized field units")
    axis.set_title("Frozen H1 in-sample residual scale on 85604 training targets")
    axis.legend(title="Residual statistic")
    axis.grid(axis="y", alpha=0.25)
    path = output / "residual_scale.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for panel, axis_name in zip(axes, ("x", "y", "stored_toroidal_z")):
        source = record["spatial_autocorrelation"][axis_name]
        lags = source["lags_cells"]
        for index, field in enumerate(B5_FIELDS):
            panel.plot(
                lags,
                source["fields"][field]["correlation"],
                label=field,
                color=colors[index],
            )
        panel.axhline(0.1, color="black", linestyle="--", linewidth=1, label="|ρ|=0.10")
        panel.axhline(0.0, color="black", linewidth=0.8)
        panel.set_xlabel(f"Lag along {axis_name} (cells)")
        panel.set_ylabel("Pooled normalized residual correlation ρ")
        panel.set_title(f"Spatial residual ACF: {axis_name}")
        panel.grid(alpha=0.2)
    axes[-1].legend(title="Field / threshold", fontsize=8)
    path = output / "spatial_autocorrelation.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for panel, (key, title) in zip(
        axes,
        (
            ("pattern", "Residual-pattern temporal correlation"),
            ("per_target_residual_RMS", "Residual-RMS temporal correlation"),
        ),
    ):
        source = record["temporal_autocorrelation"][key]
        for index, field in enumerate(B5_FIELDS):
            panel.plot(
                source["lags_frames"],
                source["fields"][field]["correlation"],
                label=field,
                color=colors[index],
            )
        panel.axhline(0.1, color="black", linestyle="--", linewidth=1)
        panel.axhline(0.0, color="black", linewidth=0.8)
        panel.set_xlabel(
            f"Lag (saved frames; {record['cadence_microseconds']:.6g} μs/frame)"
        )
        panel.set_ylabel("Normalized residual correlation ρ")
        panel.set_title(title)
        panel.grid(alpha=0.2)
    axes[-1].legend(title="Field")
    path = output / "temporal_autocorrelation.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    matrix = np.asarray(record["cross_field"]["global"]["correlation_matrix"])
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xticks(range(len(B5_FIELDS)), B5_FIELDS)
    axis.set_yticks(range(len(B5_FIELDS)), B5_FIELDS)
    for row in range(len(B5_FIELDS)):
        for column in range(len(B5_FIELDS)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    axis.set_title("Joint residual cross-field correlation (85604 training)")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    path = output / "cross_field_correlation.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    labels = list(B5_TOROIDAL_BANDS)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for panel, (quantity, title) in zip(
        axes,
        (
            ("truth_power_fraction", "Training-truth toroidal power fraction"),
            ("residual_power_fraction", "H1 residual toroidal power fraction"),
        ),
    ):
        width = 0.15
        locations = np.arange(len(labels))
        for index, field in enumerate(B5_FIELDS):
            values = [
                record["toroidal_support"]["fields"][field]["bands"][label][quantity]
                for label in labels
            ]
            panel.bar(
                locations + (index - 2) * width,
                values,
                width=width,
                label=field,
                color=colors[index],
            )
        panel.set_xticks(locations, labels)
        panel.set_ylabel("Fraction of Parseval-weighted toroidal power")
        panel.set_xlabel("Stored Fourier band (full-torus mapping n=5k)")
        panel.set_title(title)
        panel.grid(axis="y", alpha=0.2)
    axes[-1].legend(title="Field")
    path = output / "toroidal_support.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths
