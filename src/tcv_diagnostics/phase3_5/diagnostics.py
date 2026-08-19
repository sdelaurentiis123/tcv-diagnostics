"""Physics-facing evaluation summaries used by Phase 3.5.

All transforms are evaluation-only.  Stored Fourier index ``k`` is retained
in the records together with the physical mapping ``n=5k``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
MODE_BANDS = {
    "k0_n0": (0, 0),
    "k1_3_n5_15": (1, 3),
    "k4_5_n20_25": (4, 5),
    "k6_7_n30_35": (6, 7),
    "k_ge_8_n_ge_40": (8, None),
}
CROSS_PAIRS = (("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi"))


def _canonical(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 5 or array.shape[1] != len(FIELDS):
        raise ValueError("diagnostics require [sample,5,x,y,z]")
    if not np.all(np.isfinite(array)):
        raise ValueError("diagnostic tensor contains non-finite values")
    return array


def parseval_toroidal_power(values: np.ndarray) -> np.ndarray:
    """Per-sample/field/k one-sided power, averaged over x-y."""

    array = _canonical(values)
    coefficients = np.fft.rfft(array, axis=-1, norm="ortho")
    power = np.abs(coefficients) ** 2
    weights = np.full(power.shape[-1], 2.0, dtype=np.float64)
    weights[0] = 1.0
    if array.shape[-1] % 2 == 0:
        weights[-1] = 1.0
    return np.mean(power, axis=(2, 3)) * weights[None, None, :]


def band_power(values: np.ndarray) -> dict[str, np.ndarray]:
    power = parseval_toroidal_power(values)
    output = {}
    for name, (lower, upper) in MODE_BANDS.items():
        stop = power.shape[-1] if upper is None else min(int(upper) + 1, power.shape[-1])
        output[name] = np.sum(power[..., int(lower) : stop], axis=-1)
    return output


def cross_spectrum_summary(
    values: np.ndarray,
    *,
    pairs: Sequence[tuple[str, str]] = CROSS_PAIRS,
) -> dict[str, dict[str, dict[str, float]]]:
    array = _canonical(values)
    coefficients = np.fft.rfft(array, axis=-1, norm="ortho")
    output: dict[str, dict[str, dict[str, float]]] = {}
    for first_name, second_name in pairs:
        first = FIELDS.index(first_name)
        second = FIELDS.index(second_name)
        first_coefficient = coefficients[:, first]
        second_coefficient = coefficients[:, second]
        cross = np.mean(first_coefficient * np.conjugate(second_coefficient), axis=(0, 1, 2))
        first_power = np.mean(np.abs(first_coefficient) ** 2, axis=(0, 1, 2))
        second_power = np.mean(np.abs(second_coefficient) ** 2, axis=(0, 1, 2))
        pair_record: dict[str, dict[str, float]] = {}
        for band, (lower, upper) in MODE_BANDS.items():
            if lower == 0 or lower >= cross.size:
                continue
            stop = cross.size if upper is None else min(int(upper) + 1, cross.size)
            selected_cross = np.sum(cross[int(lower) : stop])
            denominator = float(
                np.sum(first_power[int(lower) : stop])
                * np.sum(second_power[int(lower) : stop])
            )
            coherence = abs(selected_cross) ** 2 / denominator if denominator > 0.0 else math.nan
            pair_record[band] = {
                "phase_radians": float(np.angle(selected_cross)),
                "phase_degrees": float(np.degrees(np.angle(selected_cross))),
                "coherence_squared": float(np.clip(coherence, 0.0, 1.0)),
                "cross_magnitude": float(abs(selected_cross)),
            }
        output[f"{first_name}_{second_name}"] = pair_record
    return output


def per_sample_phase_coherence_error(
    truth: np.ndarray,
    forecast: np.ndarray,
    *,
    pair: tuple[str, str] = ("Ne", "phi"),
) -> tuple[np.ndarray, np.ndarray]:
    observed = _canonical(truth)
    candidate = _canonical(forecast)
    if observed.shape != candidate.shape:
        raise ValueError("phase/coherence forecast and truth shapes differ")
    first, second = (FIELDS.index(name) for name in pair)
    truth_fft = np.fft.rfft(observed, axis=-1, norm="ortho")
    forecast_fft = np.fft.rfft(candidate, axis=-1, norm="ortho")
    phase_errors = np.zeros(observed.shape[0], dtype=np.float64)
    coherence_errors = np.zeros(observed.shape[0], dtype=np.float64)
    material_bands = ((4, 5), (6, 7))
    for sample in range(observed.shape[0]):
        phase_values = []
        coherence_values = []
        for lower, upper in material_bands:
            records = []
            for transformed in (truth_fft, forecast_fft):
                a = transformed[sample, first, ..., lower : upper + 1]
                b = transformed[sample, second, ..., lower : upper + 1]
                cross = np.sum(a * np.conjugate(b))
                denom = float(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
                records.append((float(np.angle(cross)), abs(cross) ** 2 / denom if denom > 0 else 0.0))
            delta = np.angle(np.exp(1j * (records[1][0] - records[0][0])))
            phase_values.append(abs(float(delta)))
            coherence_values.append(abs(float(records[1][1] - records[0][1])))
        phase_errors[sample] = float(np.mean(phase_values))
        coherence_errors[sample] = float(np.mean(coherence_values))
    return phase_errors, coherence_errors


def raw_scalar_series(values: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    array = _canonical(values)
    series: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}
    for channel, field in enumerate(FIELDS):
        selected = array[:, channel]
        if field == "phi":
            selected = selected - np.mean(selected, axis=(1, 2, 3), keepdims=True)
        mean = np.mean(selected, axis=(1, 2, 3))
        fluctuation = selected - np.mean(selected, axis=(1, 2, 3), keepdims=True)
        radial_profile = np.mean(selected, axis=(2, 3))
        radial_fluctuation_rms = np.sqrt(np.mean(fluctuation * fluctuation, axis=(2, 3)))
        series[f"{field}.global_mean"] = mean
        series[f"{field}.global_fluctuation_RMS"] = np.sqrt(
            np.mean(fluctuation * fluctuation, axis=(1, 2, 3))
        )
        series[f"{field}.radial_profile_RMS"] = np.sqrt(np.mean(radial_profile**2, axis=1))
        profiles[f"{field}.radial_mean"] = radial_profile
        profiles[f"{field}.radial_fluctuation_RMS"] = radial_fluctuation_rms
    for band, power in band_power(array).items():
        for channel, field in enumerate(FIELDS):
            series[f"{field}.mode_power.{band}"] = power[:, channel]
    return series, profiles


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("correlation matrix input must be [sample,feature]")
    centered = matrix - np.mean(matrix, axis=0)
    covariance = centered.T @ centered / (matrix.shape[0] - 1)
    scale = np.sqrt(np.clip(np.diag(covariance), np.finfo(float).tiny, None))
    result = covariance / np.outer(scale, scale)
    return np.clip(0.5 * (result + result.T), -1.0, 1.0)


def covariance_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("covariance matrix input must be [sample,feature]")
    centered = matrix - np.mean(matrix, axis=0)
    return centered.T @ centered / (matrix.shape[0] - 1)


def matrix_relative_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("matrix distance shapes differ")
    denominator = float(np.linalg.norm(left.reshape(-1)))
    return float(np.linalg.norm((right - left).reshape(-1)) / denominator) if denominator > 0 else math.nan


def cross_field_covariance(values: np.ndarray) -> np.ndarray:
    array = _canonical(values)
    flattened = np.moveaxis(array, 1, -1).reshape(-1, len(FIELDS))
    return covariance_matrix(flattened)


def spectral_band_covariance(values: np.ndarray) -> np.ndarray:
    powers = band_power(values)
    matrix = np.column_stack([powers[name] for name in MODE_BANDS])
    return covariance_matrix(np.log(np.maximum(matrix, np.finfo(float).tiny)))


def transport_covariance_summary(
    local: Mapping[str, np.ndarray],
    integrated: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    quantities = tuple(sorted(local))
    if set(quantities) != set(integrated):
        raise ValueError("local and integrated transport quantities differ")
    local_features = np.column_stack(
        [np.mean(np.asarray(local[name]) ** 2, axis=tuple(range(1, np.asarray(local[name]).ndim))) for name in quantities]
    )
    integrated_features = np.column_stack([np.asarray(integrated[name]).reshape(-1) for name in quantities])
    return {
        "local_covariance": covariance_matrix(local_features),
        "integrated_covariance": covariance_matrix(integrated_features),
        "integrated_variance": np.var(integrated_features, axis=0, ddof=1),
    }


def reconstruction_diagnostics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    truth = _canonical(reference)
    reconstruction = _canonical(candidate)
    if truth.shape != reconstruction.shape:
        raise ValueError("reconstruction diagnostic shapes differ")
    residual = reconstruction - truth
    total = float(np.sum(truth * truth))
    result: dict[str, object] = {
        "variance_capture_about_zero": 1.0 - float(np.sum(residual * residual)) / total,
        "field_RMSE": {
            field: float(np.sqrt(np.mean(residual[:, channel] ** 2)))
            for channel, field in enumerate(FIELDS)
        },
        "spectral_band_power_ratio": {},
        "cross_field_covariance_relative_error": matrix_relative_distance(
            cross_field_covariance(truth), cross_field_covariance(reconstruction)
        ),
    }
    truth_power = band_power(truth)
    candidate_power = band_power(reconstruction)
    ratios = result["spectral_band_power_ratio"]
    if not isinstance(ratios, dict):
        raise AssertionError
    for band in MODE_BANDS:
        ratios[band] = {
            field: float(
                np.mean(candidate_power[band][:, channel])
                / np.mean(truth_power[band][:, channel])
            )
            for channel, field in enumerate(FIELDS)
        }
    result["cross_spectrum"] = cross_spectrum_summary(reconstruction)
    return result
