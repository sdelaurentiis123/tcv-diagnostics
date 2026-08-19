"""Toroidal-only translation diagnostics for Paper 0 Phase 3.5.

The sign convention is explicit: a positive displacement ``s`` is the value
passed to ``numpy.roll(earlier, s, axis=-1)`` to align the earlier state to
the later state.  No helper in this module permits wrapping x or y.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


TOROIDAL_AXIS = -1


def circular_toroidal_roll(values: np.ndarray, shift: int, *, axis: int = -1) -> np.ndarray:
    array = np.asarray(values)
    normalized_axis = axis if axis >= 0 else array.ndim + axis
    if normalized_axis != array.ndim - 1:
        raise ValueError("Phase 3.5 permits circular shifts only on the final z axis")
    return np.roll(array, int(shift), axis=-1)


def remove_toroidal_mean(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("translation input must include channel and toroidal axes")
    return array - np.mean(array, axis=-1, keepdims=True)


def training_field_rms(values: np.ndarray) -> np.ndarray:
    """RMS after per-(x,y) toroidal-mean removal for ``[sample,field,...,z]``."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError("training states must have sample, field, and z axes")
    centered = remove_toroidal_mean(array)
    reduction = tuple(index for index in range(array.ndim) if index != 1)
    rms = np.sqrt(np.mean(centered * centered, axis=reduction))
    if np.any(~np.isfinite(rms)) or np.any(rms <= 0.0):
        raise ValueError("training toroidal fluctuation RMS must be finite and positive")
    return rms


def _whiten_fields(values: np.ndarray, field_rms: Sequence[float]) -> np.ndarray:
    array = remove_toroidal_mean(values)
    scale = np.asarray(field_rms, dtype=np.float64)
    if array.ndim < 2 or scale.shape != (array.shape[0],):
        raise ValueError("field RMS must match the first/channel axis")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("field RMS must be finite and positive")
    shape = (scale.size,) + (1,) * (array.ndim - 1)
    return array / scale.reshape(shape)


def normalized_circular_correlation(
    earlier: np.ndarray,
    later: np.ndarray,
    *,
    field_rms: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return shared and per-field normalized correlations for every z shift."""

    first = np.asarray(earlier)
    second = np.asarray(later)
    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("translation states must share [field,...,z] shape")
    a = _whiten_fields(first, field_rms)
    b = _whiten_fields(second, field_rms)
    cross = np.conjugate(np.fft.rfft(a, axis=-1)) * np.fft.rfft(b, axis=-1)
    correlation = np.fft.irfft(cross, n=a.shape[-1], axis=-1)
    spatial_axes = tuple(range(1, correlation.ndim - 1))
    numerator = np.sum(correlation, axis=spatial_axes) if spatial_axes else correlation
    norm_axes = tuple(range(1, a.ndim))
    denominator = np.sqrt(
        np.sum(a * a, axis=norm_axes) * np.sum(b * b, axis=norm_axes)
    )
    if np.any(denominator <= 0.0):
        raise ValueError("translation correlation received a zero-energy field")
    per_field = numerator / denominator[:, None]
    shared_numerator = np.sum(numerator, axis=0)
    shared_denominator = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    shared = shared_numerator / shared_denominator
    return np.asarray(shared, dtype=np.float64), np.asarray(per_field, dtype=np.float64)


def signed_shift(index: int, size: int) -> int:
    value = int(index) % int(size)
    return value - int(size) if value > int(size) // 2 else value


def _peak_summary(curve: np.ndarray, *, neighbor_exclusion: int) -> dict[str, float | int]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 4 or not np.all(np.isfinite(values)):
        raise ValueError("correlation curve must be one finite vector")
    peak_index = int(np.argmax(values))
    distance = np.minimum(
        (np.arange(values.size) - peak_index) % values.size,
        (peak_index - np.arange(values.size)) % values.size,
    )
    eligible = distance > int(neighbor_exclusion)
    second = float(np.max(values[eligible])) if np.any(eligible) else math.nan
    weights = values - float(np.min(values))
    weights = weights + np.finfo(np.float64).eps
    weights /= float(np.sum(weights))
    entropy = -float(np.sum(weights * np.log(weights))) / math.log(values.size)
    return {
        "integer_shift_index": peak_index,
        "signed_integer_shift": signed_shift(peak_index, values.size),
        "peak_correlation": float(values[peak_index]),
        "second_peak_correlation": second,
        "peak_margin": float(values[peak_index] - second),
        "normalized_surface_entropy": entropy,
    }


def fourier_subcell_shift(
    earlier: np.ndarray,
    later: np.ndarray,
    *,
    field_rms: Sequence[float],
    integer_shift: int,
    modes: tuple[int, int] = (1, 7),
    minimum_modes: int = 4,
    minimum_weighted_r2: float = 0.8,
) -> dict[str, float | int | bool | None]:
    """Refine an integer shift with a weighted cross-spectral phase slope."""

    a = _whiten_fields(earlier, field_rms)
    b = _whiten_fields(later, field_rms)
    cross = np.sum(
        np.conjugate(np.fft.rfft(a, axis=-1)) * np.fft.rfft(b, axis=-1),
        axis=tuple(range(a.ndim - 1)),
    )
    lower, upper = (int(value) for value in modes)
    ks = np.arange(lower, min(upper, cross.size - 1) + 1, dtype=np.float64)
    selected = cross[ks.astype(np.int64)]
    weights = np.abs(selected)
    finite = np.isfinite(weights) & (weights > np.finfo(np.float64).eps)
    if int(np.sum(finite)) < int(minimum_modes):
        return {"available": False, "subcell_shift": None, "weighted_r2": None,
                "mode_count": int(np.sum(finite))}
    ks = ks[finite]
    selected = selected[finite]
    weights = weights[finite]
    size = a.shape[-1]
    compensated = selected * np.exp(1j * 2.0 * np.pi * ks * int(integer_shift) / size)
    phase = np.unwrap(np.angle(compensated))
    if float(np.max(np.abs(phase - np.mean(phase)))) <= 1e-10:
        return {
            "available": True,
            "subcell_shift": float(integer_shift),
            "weighted_r2": 1.0,
            "mode_count": int(ks.size),
        }
    design = np.column_stack((np.ones(ks.size), ks))
    root_weight = np.sqrt(weights / np.max(weights))
    coefficient, *_ = np.linalg.lstsq(design * root_weight[:, None], phase * root_weight, rcond=None)
    fitted = design @ coefficient
    mean = float(np.sum(weights * phase) / np.sum(weights))
    total = float(np.sum(weights * (phase - mean) ** 2))
    error = float(np.sum(weights * (phase - fitted) ** 2))
    r2 = 1.0 - error / total if total > 0.0 else 1.0
    estimate = float(integer_shift - coefficient[1] * size / (2.0 * np.pi))
    while estimate - integer_shift > size / 2:
        estimate -= size
    while estimate - integer_shift < -size / 2:
        estimate += size
    available = bool(np.isfinite(estimate) and r2 >= float(minimum_weighted_r2))
    return {
        "available": available,
        "subcell_shift": estimate if available else None,
        "weighted_r2": float(r2),
        "mode_count": int(ks.size),
    }


@dataclass(frozen=True)
class TranslationEstimate:
    shared_curve: np.ndarray
    per_field_curves: np.ndarray
    shared: dict[str, float | int | bool | None]
    per_field: tuple[dict[str, float | int], ...]


def estimate_toroidal_displacement(
    earlier: np.ndarray,
    later: np.ndarray,
    *,
    field_rms: Sequence[float],
    neighbor_exclusion: int = 1,
    subcell_modes: tuple[int, int] = (1, 7),
    minimum_subcell_modes: int = 4,
    minimum_subcell_r2: float = 0.8,
) -> TranslationEstimate:
    shared_curve, field_curves = normalized_circular_correlation(
        earlier, later, field_rms=field_rms
    )
    shared = _peak_summary(shared_curve, neighbor_exclusion=neighbor_exclusion)
    shared.update(
        fourier_subcell_shift(
            earlier,
            later,
            field_rms=field_rms,
            integer_shift=int(shared["signed_integer_shift"]),
            modes=subcell_modes,
            minimum_modes=minimum_subcell_modes,
            minimum_weighted_r2=minimum_subcell_r2,
        )
    )
    per_field = tuple(
        _peak_summary(curve, neighbor_exclusion=neighbor_exclusion)
        for curve in field_curves
    )
    return TranslationEstimate(shared_curve, field_curves, shared, per_field)


def normalized_equivariance_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    truth = np.asarray(reference, dtype=np.float64)
    test = np.asarray(candidate, dtype=np.float64)
    if truth.shape != test.shape:
        raise ValueError("equivariance tensors differ in shape")
    denominator = float(np.linalg.norm(truth.reshape(-1)))
    if denominator <= 0.0:
        raise ValueError("equivariance reference has zero norm")
    return float(np.linalg.norm((test - truth).reshape(-1)) / denominator)


def equal_field_relative_error(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, np.ndarray]:
    truth = np.asarray(reference, dtype=np.float64)
    test = np.asarray(candidate, dtype=np.float64)
    if truth.shape != test.shape or truth.ndim < 2:
        raise ValueError("equal-field error expects matching [field,...] tensors")
    field = np.empty(truth.shape[0], dtype=np.float64)
    for index in range(truth.shape[0]):
        denominator = float(np.linalg.norm(truth[index].reshape(-1)))
        field[index] = (
            float(np.linalg.norm((test[index] - truth[index]).reshape(-1)) / denominator)
            if denominator > 0.0
            else math.nan
        )
    return float(np.nanmean(field)), field
