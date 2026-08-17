"""Validated metric primitives for Paper 0.

The conventions implemented here are frozen in
``paper0/protocol/PHASE2_METRIC_PROTOCOL.md``.  These functions contain no
shot-specific constants other than the explicit default ``zperiod=5`` and do
not read simulation data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np


CANONICAL_FORECAST_AXES = (
    "batch",
    "member",
    "time",
    "channel",
    "x",
    "y",
    "z",
)
CANONICAL_TRUTH_AXES = ("batch", "time", "channel", "x", "y", "z")


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array of rank {ndim}")
    return normalized


def _normalize_axes(axes: Sequence[int], ndim: int) -> tuple[int, ...]:
    normalized = tuple(_normalize_axis(axis, ndim) for axis in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("axes must be unique")
    return normalized


def _require_real_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_canonical_forecast(
    forecast: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return canonical forecast and truth arrays.

    Forecast axes are ``[B, M, T, C, X, Y, Z]`` and truth axes are
    ``[B, T, C, X, Y, Z]``.
    """

    forecast_array = _require_real_finite("forecast", forecast)
    truth_array = _require_real_finite("truth", truth)
    if forecast_array.ndim != len(CANONICAL_FORECAST_AXES):
        raise ValueError(
            "forecast must have axes [B, M, T, C, X, Y, Z]; "
            f"received shape {forecast_array.shape}"
        )
    if truth_array.ndim != len(CANONICAL_TRUTH_AXES):
        raise ValueError(
            "truth must have axes [B, T, C, X, Y, Z]; "
            f"received shape {truth_array.shape}"
        )
    expected_truth_shape = (
        forecast_array.shape[0],
        *forecast_array.shape[2:],
    )
    if truth_array.shape != expected_truth_shape:
        raise ValueError(
            "forecast/truth non-member axes disagree: expected truth shape "
            f"{expected_truth_shape}, received {truth_array.shape}"
        )
    if forecast_array.shape[1] < 1:
        raise ValueError("forecast must contain at least one ensemble member")
    if forecast_array.shape[2] < 1:
        raise ValueError("forecast must contain at least one future time")
    return forecast_array, truth_array


def toroidal_mode_numbers(n_z: int, *, zperiod: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return stored indices ``k`` and full-torus modes ``n=zperiod*k``."""

    if not isinstance(n_z, (int, np.integer)) or n_z < 2:
        raise ValueError("n_z must be an integer of at least two")
    if not isinstance(zperiod, (int, np.integer)) or zperiod <= 0:
        raise ValueError("zperiod must be a positive integer")
    stored_k = np.arange(n_z // 2 + 1, dtype=np.int64)
    return stored_k, stored_k * int(zperiod)


def one_sided_power_spectrum(
    values: np.ndarray,
    *,
    axis: int = -1,
    remove_mean: bool = False,
) -> np.ndarray:
    """Return Parseval-normalized one-sided mean-square mode contributions."""

    array = _require_real_finite("spectrum input", values)
    transform_axis = _normalize_axis(axis, array.ndim)
    n_z = array.shape[transform_axis]
    if n_z < 2:
        raise ValueError("Fourier axis must contain at least two cells")
    if remove_mean:
        array = array - np.mean(array, axis=transform_axis, keepdims=True)

    coefficients = np.fft.rfft(array, axis=transform_axis)
    power = np.abs(coefficients) ** 2 / float(n_z * n_z)
    weights = np.ones(n_z // 2 + 1, dtype=np.float64)
    if n_z % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    broadcast_shape = [1] * power.ndim
    broadcast_shape[transform_axis] = weights.size
    return power * weights.reshape(broadcast_shape)


def cross_spectral_metrics(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sample_axes: Sequence[int],
    fourier_axis: int = -1,
    remove_mean: bool = False,
    zperiod: int = 5,
) -> dict[str, np.ndarray]:
    """Compute cross-spectrum, coherence, and phase with explicit reductions.

    The convention is ``S_ab = mean(A * conjugate(B))``.  Therefore a relation
    ``B = A * exp(i delta)`` has returned phase ``-delta``.
    """

    a_array = _require_real_finite("cross-spectrum input a", a)
    b_array = _require_real_finite("cross-spectrum input b", b)
    if a_array.shape != b_array.shape:
        raise ValueError(
            f"cross-spectrum inputs must have equal shape, got "
            f"{a_array.shape} and {b_array.shape}"
        )
    transform_axis = _normalize_axis(fourier_axis, a_array.ndim)
    reduction_axes = _normalize_axes(sample_axes, a_array.ndim)
    if not reduction_axes:
        raise ValueError("sample_axes must contain at least one reduction axis")
    if transform_axis in reduction_axes:
        raise ValueError("the Fourier axis cannot also be a sample axis")
    n_z = a_array.shape[transform_axis]
    if n_z < 2:
        raise ValueError("Fourier axis must contain at least two cells")
    if remove_mean:
        a_array = a_array - np.mean(a_array, axis=transform_axis, keepdims=True)
        b_array = b_array - np.mean(b_array, axis=transform_axis, keepdims=True)

    a_coefficients = np.fft.rfft(a_array, axis=transform_axis)
    b_coefficients = np.fft.rfft(b_array, axis=transform_axis)
    cross = np.mean(
        a_coefficients * np.conjugate(b_coefficients), axis=reduction_axes
    )
    auto_a = np.mean(np.abs(a_coefficients) ** 2, axis=reduction_axes)
    auto_b = np.mean(np.abs(b_coefficients) ** 2, axis=reduction_axes)
    denominator = auto_a * auto_b
    coherence = np.full(denominator.shape, np.nan, dtype=np.float64)
    np.divide(
        np.abs(cross) ** 2,
        denominator,
        out=coherence,
        where=denominator > 0,
    )
    stored_k, full_torus_n = toroidal_mode_numbers(n_z, zperiod=zperiod)
    return {
        "cross_spectrum": cross,
        "auto_spectrum_a": auto_a,
        "auto_spectrum_b": auto_b,
        "coherence": coherence,
        "phase_radians": np.angle(cross),
        "stored_k": stored_k,
        "full_torus_n": full_torus_n,
    }


def _validate_scalar_ensemble(
    forecast: np.ndarray,
    truth: np.ndarray,
    member_axis: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    forecast_array = _require_real_finite("ensemble forecast", forecast)
    truth_array = _require_real_finite("ensemble truth", truth)
    if forecast_array.ndim < 1:
        raise ValueError("ensemble forecast must have at least one axis")
    normalized_member_axis = _normalize_axis(member_axis, forecast_array.ndim)
    expected_truth_shape = (
        forecast_array.shape[:normalized_member_axis]
        + forecast_array.shape[normalized_member_axis + 1 :]
    )
    if truth_array.shape != expected_truth_shape:
        raise ValueError(
            "truth shape must equal forecast shape without the member axis: "
            f"expected {expected_truth_shape}, received {truth_array.shape}"
        )
    members = forecast_array.shape[normalized_member_axis]
    if members < 1:
        raise ValueError("ensemble forecast must contain at least one member")
    return forecast_array, truth_array, normalized_member_axis


def ensemble_crps(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
    fair: bool = False,
) -> np.ndarray:
    """Return pointwise ordinary or finite-ensemble fair CRPS.

    The sorted-member identity avoids allocating an ``M x M`` pairwise array.
    No axes other than the member axis are reduced.
    """

    forecast_array, truth_array, member_axis = _validate_scalar_ensemble(
        forecast, truth, member_axis
    )
    members = forecast_array.shape[member_axis]
    if fair and members < 2:
        raise ValueError("fair CRPS requires at least two ensemble members")

    observation_term = np.mean(
        np.abs(forecast_array - np.expand_dims(truth_array, axis=member_axis)),
        axis=member_axis,
    )
    if members == 1:
        return observation_term

    sorted_forecast = np.sort(forecast_array, axis=member_axis)
    coefficients = 2.0 * np.arange(members, dtype=np.float64) - members + 1.0
    coefficient_shape = [1] * forecast_array.ndim
    coefficient_shape[member_axis] = members
    weighted_order_sum = np.sum(
        sorted_forecast * coefficients.reshape(coefficient_shape),
        axis=member_axis,
    )
    denominator = members * (members - 1) if fair else members * members
    return observation_term - weighted_order_sum / float(denominator)


def ordinary_crps(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
) -> np.ndarray:
    """Return the CRPS of the finite empirical ensemble distribution."""

    return ensemble_crps(forecast, truth, member_axis=member_axis, fair=False)


def fair_crps(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
) -> np.ndarray:
    """Return the fair CRPS estimator for random-sample ensemble members."""

    return ensemble_crps(forecast, truth, member_axis=member_axis, fair=True)


def spread_skill_summary(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
) -> dict[str, float | int]:
    """Return globally reduced population spread, RMSE, and their ratio."""

    forecast_array, truth_array, member_axis = _validate_scalar_ensemble(
        forecast, truth, member_axis
    )
    ensemble_mean = np.mean(forecast_array, axis=member_axis)
    pointwise_spread = np.std(forecast_array, axis=member_axis, ddof=0)
    rms_spread = float(np.sqrt(np.mean(pointwise_spread**2)))
    rmse = float(np.sqrt(np.mean((ensemble_mean - truth_array) ** 2)))
    ratio = math.nan if rmse == 0.0 else rms_spread / rmse
    return {
        "ensemble_size": int(forecast_array.shape[member_axis]),
        "rms_spread": rms_spread,
        "rmse_of_ensemble_mean": rmse,
        "spread_skill_ratio": ratio,
        "member_standard_deviation_ddof": 0,
    }


def central_interval_coverage(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    nominal_coverage: float,
    member_axis: int = 1,
    quantile_method: str = "linear",
) -> dict[str, Any]:
    """Return global empirical coverage for a central ensemble interval."""

    forecast_array, truth_array, member_axis = _validate_scalar_ensemble(
        forecast, truth, member_axis
    )
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be strictly between zero and one")
    tail = (1.0 - nominal_coverage) / 2.0
    bounds = np.quantile(
        forecast_array,
        [tail, 1.0 - tail],
        axis=member_axis,
        method=quantile_method,
    )
    lower, upper = bounds[0], bounds[1]
    covered = (truth_array >= lower) & (truth_array <= upper)
    return {
        "ensemble_size": int(forecast_array.shape[member_axis]),
        "nominal_coverage": float(nominal_coverage),
        "empirical_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "quantile_method": quantile_method,
        "lower": lower,
        "upper": upper,
        "covered": covered,
    }


def apply_memberwise(
    diagnostic: Callable[..., np.ndarray],
    *ensemble_fields: np.ndarray,
    member_axis: int = 1,
) -> np.ndarray:
    """Apply a nonlinear diagnostic independently to each ensemble member.

    Inputs must share a shape.  The diagnostic receives arrays with the member
    axis removed and must return the same shape for every member.  The returned
    array restores the member axis in its original position.
    """

    if not ensemble_fields:
        raise ValueError("at least one ensemble field is required")
    arrays = tuple(
        _require_real_finite("ensemble field", field) for field in ensemble_fields
    )
    reference_shape = arrays[0].shape
    if any(array.shape != reference_shape for array in arrays[1:]):
        raise ValueError("all ensemble fields must have identical shape")
    if arrays[0].ndim < 2:
        raise ValueError("ensemble fields must include batch and member axes")
    normalized_member_axis = _normalize_axis(member_axis, arrays[0].ndim)
    members = arrays[0].shape[normalized_member_axis]
    if members < 1:
        raise ValueError("ensemble fields must contain at least one member")

    member_results: list[np.ndarray] = []
    result_shape: tuple[int, ...] | None = None
    for member in range(members):
        inputs = tuple(
            np.take(array, member, axis=normalized_member_axis) for array in arrays
        )
        result = _require_real_finite(
            "member-wise diagnostic result", diagnostic(*inputs)
        )
        if result_shape is None:
            result_shape = result.shape
            if normalized_member_axis > result.ndim:
                raise ValueError(
                    "diagnostic removed axes before the member position; cannot "
                    "restore member semantics"
                )
            expected_leading_shape = reference_shape[:normalized_member_axis]
            if result.shape[:normalized_member_axis] != expected_leading_shape:
                raise ValueError(
                    "diagnostic must preserve every axis before the member axis"
                )
        elif result.shape != result_shape:
            raise ValueError("diagnostic returned inconsistent shapes across members")
        member_results.append(result)
    return np.stack(member_results, axis=normalized_member_axis)
