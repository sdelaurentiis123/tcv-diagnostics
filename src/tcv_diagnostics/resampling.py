"""Auditable periodic resampling and paired-grid comparison statistics.

The canonical Paper 0 model-grid transform uses ``scipy.signal.resample`` on
the final periodic axis and stores float32. The exact scientific scope and
acceptance rules are frozen in
``paper0/protocol/PHASE2_STATE_RESAMPLING_PROTOCOL.md``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.signal import resample


PAIR_SUM_KEYS = (
    "count",
    "sum_reference",
    "sum_candidate",
    "sum_reference_squared",
    "sum_candidate_squared",
    "sum_reference_candidate",
    "sum_difference",
    "sum_difference_squared",
    "sum_absolute_reference",
    "sum_absolute_reference_sign_disagreement",
)


def _finite_real_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one axis")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def periodic_resample_float32(
    values: np.ndarray,
    target_samples: int,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Apply the frozen unwindowed Fourier resampler and return float32.

    This is intentionally a narrow wrapper around the public SciPy operation
    used by the predecessor converter. No window, smoothing, coordinate
    interpolation, or phase shift is introduced.
    """

    array = _finite_real_array("periodic resampling input", values)
    if not isinstance(target_samples, (int, np.integer)) or target_samples < 2:
        raise ValueError("target_samples must be an integer of at least two")
    normalized_axis = np.core.numeric.normalize_axis_index(axis, array.ndim)
    if array.shape[normalized_axis] < 2:
        raise ValueError("periodic input axis must contain at least two samples")
    output = resample(
        array,
        int(target_samples),
        axis=normalized_axis,
        window=None,
        domain="time",
    )
    output = np.asarray(output, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("periodic resampling produced non-finite values")
    return output


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return ``sqrt(sum((candidate-reference)^2)/sum(reference^2))``."""

    stats = paired_sufficient_statistics(reference, candidate)
    return float(finalize_paired_statistics(stats)["relative_l2"])


def paired_sufficient_statistics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, int | float]:
    """Return additive sufficient statistics for a finite paired comparison."""

    reference_array = np.asarray(
        _finite_real_array("paired reference", reference), dtype=np.float64
    )
    candidate_array = np.asarray(
        _finite_real_array("paired candidate", candidate), dtype=np.float64
    )
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            "paired reference and candidate shapes differ: "
            f"{reference_array.shape} versus {candidate_array.shape}"
        )
    difference = candidate_array - reference_array
    disagreement = np.sign(reference_array) != np.sign(candidate_array)
    absolute_reference = np.abs(reference_array)
    return {
        "count": int(reference_array.size),
        "sum_reference": float(np.sum(reference_array, dtype=np.float64)),
        "sum_candidate": float(np.sum(candidate_array, dtype=np.float64)),
        "sum_reference_squared": float(
            np.sum(np.square(reference_array), dtype=np.float64)
        ),
        "sum_candidate_squared": float(
            np.sum(np.square(candidate_array), dtype=np.float64)
        ),
        "sum_reference_candidate": float(
            np.sum(reference_array * candidate_array, dtype=np.float64)
        ),
        "sum_difference": float(np.sum(difference, dtype=np.float64)),
        "sum_difference_squared": float(
            np.sum(np.square(difference), dtype=np.float64)
        ),
        "sum_absolute_reference": float(
            np.sum(absolute_reference, dtype=np.float64)
        ),
        "sum_absolute_reference_sign_disagreement": float(
            np.sum(absolute_reference[disagreement], dtype=np.float64)
        ),
    }


def merge_paired_sufficient_statistics(
    statistics: Iterable[dict[str, int | float]],
) -> dict[str, int | float]:
    """Merge disjoint paired-statistic records without changing semantics."""

    records = list(statistics)
    if not records:
        raise ValueError("cannot merge an empty paired-statistic collection")
    for record in records:
        if set(record) != set(PAIR_SUM_KEYS):
            raise ValueError("paired-statistic keys do not match the schema")
        if int(record["count"]) <= 0:
            raise ValueError("paired-statistic count must be positive")
        if not all(math.isfinite(float(record[key])) for key in PAIR_SUM_KEYS[1:]):
            raise ValueError("paired-statistic record contains a non-finite sum")
    return {
        key: (
            int(sum(int(record[key]) for record in records))
            if key == "count"
            else float(sum(float(record[key]) for record in records))
        )
        for key in PAIR_SUM_KEYS
    }


def finalize_paired_statistics(
    statistics: dict[str, int | float],
) -> dict[str, int | float | bool | None]:
    """Derive the frozen paired metrics from additive sufficient statistics."""

    if set(statistics) != set(PAIR_SUM_KEYS):
        raise ValueError("paired-statistic keys do not match the schema")
    count = int(statistics["count"])
    if count <= 0:
        raise ValueError("paired-statistic count must be positive")
    values = {key: float(statistics[key]) for key in PAIR_SUM_KEYS[1:]}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("paired-statistic record contains a non-finite sum")
    reference_squared = values["sum_reference_squared"]
    candidate_squared = values["sum_candidate_squared"]
    if reference_squared <= 0.0:
        raise ValueError("paired reference has zero L2 norm")
    reference_rms = math.sqrt(reference_squared / count)
    candidate_rms = math.sqrt(max(candidate_squared, 0.0) / count)
    relative_error = math.sqrt(
        max(values["sum_difference_squared"], 0.0) / reference_squared
    )
    reference_centered = reference_squared - (
        values["sum_reference"] ** 2 / count
    )
    candidate_centered = candidate_squared - (
        values["sum_candidate"] ** 2 / count
    )
    covariance = values["sum_reference_candidate"] - (
        values["sum_reference"] * values["sum_candidate"] / count
    )
    correlation_defined = reference_centered > 0.0 and candidate_centered > 0.0
    correlation = (
        covariance / math.sqrt(reference_centered * candidate_centered)
        if correlation_defined
        else None
    )
    absolute_reference = values["sum_absolute_reference"]
    sign_defined = absolute_reference > 0.0
    weighted_sign_disagreement = (
        values["sum_absolute_reference_sign_disagreement"] / absolute_reference
        if sign_defined
        else None
    )
    return {
        "point_count": count,
        "reference_rms": reference_rms,
        "candidate_rms": candidate_rms,
        "relative_l2": relative_error,
        "normalized_bias": values["sum_difference"] / count / reference_rms,
        "rms_ratio": candidate_rms / reference_rms,
        "pearson_correlation": correlation,
        "pearson_correlation_defined": correlation_defined,
        "weighted_sign_disagreement": weighted_sign_disagreement,
        "weighted_sign_disagreement_defined": sign_defined,
    }


def paired_frame_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    z_axis: int = -1,
) -> dict[str, Any]:
    """Return frozen full-array, toroidal-profile, and tail metrics."""

    reference_array = np.asarray(
        _finite_real_array("frame reference", reference), dtype=np.float64
    )
    candidate_array = np.asarray(
        _finite_real_array("frame candidate", candidate), dtype=np.float64
    )
    if reference_array.shape != candidate_array.shape:
        raise ValueError("frame reference and candidate shapes differ")
    normalized_axis = np.core.numeric.normalize_axis_index(z_axis, reference_array.ndim)
    full_sufficient = paired_sufficient_statistics(reference_array, candidate_array)
    profile_reference = np.mean(reference_array, axis=normalized_axis)
    profile_candidate = np.mean(candidate_array, axis=normalized_axis)
    profile_sufficient = paired_sufficient_statistics(
        profile_reference, profile_candidate
    )
    absolute_reference = np.abs(reference_array)
    absolute_candidate = np.abs(candidate_array)
    tail_ratios: dict[str, float] = {}
    tail_values: dict[str, dict[str, float]] = {}
    for quantile in (0.95, 0.99):
        label = f"p{int(quantile * 100)}"
        reference_quantile = float(
            np.quantile(absolute_reference, quantile, method="linear")
        )
        candidate_quantile = float(
            np.quantile(absolute_candidate, quantile, method="linear")
        )
        if reference_quantile <= 0.0:
            raise ValueError(f"reference absolute {label} is zero")
        tail_ratios[f"absolute_value_{label}_ratio"] = (
            candidate_quantile / reference_quantile
        )
        tail_values[label] = {
            "reference": reference_quantile,
            "candidate": candidate_quantile,
        }
    return {
        "sufficient_statistics": full_sufficient,
        "metrics": finalize_paired_statistics(full_sufficient),
        "toroidal_mean_profile_sufficient_statistics": profile_sufficient,
        "toroidal_mean_profile_relative_l2": finalize_paired_statistics(
            profile_sufficient
        )["relative_l2"],
        "tail_quantiles": tail_values,
        **tail_ratios,
    }


def materiality_label(relative_error: float) -> str:
    """Apply the prospectively frozen direct-88 materiality intervals."""

    value = float(relative_error)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("relative error must be finite and nonnegative")
    if value < 0.01:
        return "negligible"
    if value < 0.05:
        return "small"
    if value < 0.10:
        return "material"
    return "severe"


def linear_quantile(values: Iterable[float], probability: float) -> float:
    """Return the frozen NumPy-linear quantile for finite scalar values."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantile values must form a nonempty one-dimensional set")
    if not np.all(np.isfinite(array)):
        raise ValueError("quantile values contain non-finite entries")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0,1]")
    return float(np.quantile(array, probability, method="linear"))
