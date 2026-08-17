"""Immutable 85604 data-protocol primitives.

The functions here contain no model code and have no knowledge of shot 85606.
They implement the rules frozen in ``paper0/protocol/PHASE1_DATA_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


RUN_ID = "85604"
C5_FIELDS = ("Ne", "Te", "Ti", "phi", "Vi")
DENSITY_EPSILON = 1e-6
FIELD_TRANSFORMS = {
    "Ne": "ln(x + 1e-6)",
    "Te": "identity",
    "Ti": "identity",
    "phi": "identity",
    "Vi": "identity",
}


def path_is_allowed(path: Path) -> bool:
    """Return false for any path that could name the sequestered test run."""

    candidates = (path, path.expanduser().resolve(strict=False))
    for candidate in candidates:
        if "85606" in str(candidate).lower():
            return False
        if any(part.lower() == "test" for part in candidate.parts):
            return False
    return True


def require_allowed_file(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path_is_allowed(path):
        raise ValueError(f"refusing sequestered input path: {path}")
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return path


@dataclass(frozen=True)
class IndexRegion:
    name: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"invalid {self.name} region [{self.start}, {self.stop})")

    @property
    def frames(self) -> int:
        return self.stop - self.start

    def contains_window(self, start: int, total_frames: int) -> bool:
        if total_frames <= 0:
            return False
        return self.start <= start and start + total_frames <= self.stop

    def window_starts(self, total_frames: int) -> range:
        if total_frames <= 0 or total_frames > self.frames:
            raise ValueError(
                f"window length {total_frames} does not fit {self.name} region"
            )
        return range(self.start, self.stop - total_frames + 1)


@dataclass(frozen=True)
class SplitProtocol:
    total_frames: int
    train: IndexRegion
    guard: IndexRegion
    validation: IndexRegion
    max_window_frames: int

    def __post_init__(self) -> None:
        if self.train.start != 0:
            raise ValueError("training region must begin at global frame zero")
        if self.train.stop != self.guard.start:
            raise ValueError("training and guard boundaries are not contiguous")
        if self.guard.stop != self.validation.start:
            raise ValueError("guard and validation boundaries are not contiguous")
        if self.validation.stop != self.total_frames:
            raise ValueError("validation region must end at total frame count")
        if self.guard.frames <= self.max_window_frames:
            raise ValueError("guard must be strictly longer than the maximum window")

    def to_dict(self) -> dict[str, object]:
        def region_dict(region: IndexRegion) -> dict[str, object]:
            return {
                "name": region.name,
                "start_inclusive": region.start,
                "stop_exclusive": region.stop,
                "frames": region.frames,
            }

        return {
            "total_frames": self.total_frames,
            "max_window_frames": self.max_window_frames,
            "train": region_dict(self.train),
            "guard": region_dict(self.guard),
            "validation": region_dict(self.validation),
        }


DEFAULT_SPLIT = SplitProtocol(
    total_frames=624,
    train=IndexRegion("train", 0, 432),
    guard=IndexRegion("guard", 432, 496),
    validation=IndexRegion("validation", 496, 624),
    max_window_frames=32,
)


def model_transform(field: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if field == "Ne":
        shifted = array + DENSITY_EPSILON
        if np.any(shifted <= 0):
            minimum = float(np.min(array))
            raise ValueError(f"Ne contains value {minimum} <= -density epsilon")
        return np.log(shifted)
    if field not in FIELD_TRANSFORMS:
        raise KeyError(f"no model transform declared for field {field}")
    return array


def inverse_model_transform(field: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if field == "Ne":
        return np.exp(array) - DENSITY_EPSILON
    if field not in FIELD_TRANSFORMS:
        raise KeyError(f"no model transform declared for field {field}")
    return array


@dataclass
class RunningMoments:
    """Numerically stable population moments combined one array batch at a time."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return
        batch_count = int(array.size)
        batch_mean = float(np.mean(array, dtype=np.float64))
        centered = array - batch_mean
        batch_m2 = float(np.sum(centered * centered, dtype=np.float64))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    def finalize(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("cannot finalize empty moments")
        variance = self.m2 / self.count
        if variance < 0 and abs(variance) < 1e-15:
            variance = 0.0
        if variance < 0:
            raise ValueError(f"negative population variance {variance}")
        return {
            "count": self.count,
            "mean": self.mean,
            "std": math.sqrt(variance),
            "variance": variance,
            "ddof": 0,
        }


def standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not math.isfinite(std) or std <= 0:
        raise ValueError(f"normalization std must be positive, got {std}")
    return (np.asarray(values) - mean) / std


def inverse_standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not math.isfinite(std) or std <= 0:
        raise ValueError(f"normalization std must be positive, got {std}")
    return np.asarray(values) * std + mean


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else math.copysign(math.inf, numerator)


def summarize_stationarity_series(
    values: np.ndarray,
    *,
    block_count: int = 8,
    max_abs_drift: float = 0.5,
    max_abs_half_shift: float = 0.5,
    max_block_range: float = 1.0,
) -> dict[str, object]:
    """Apply the frozen operational steady-state screen to one scalar series."""

    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size == 0 or series.size % block_count:
        raise ValueError(
            f"series length {series.size} must be a positive multiple of {block_count}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("stationarity series contains non-finite values")

    temporal_std = float(np.std(series, ddof=0))
    centered_time = np.arange(series.size, dtype=np.float64)
    centered_time -= float(np.mean(centered_time))
    centered_values = series - float(np.mean(series))
    time_square_sum = float(np.dot(centered_time, centered_time))
    slope = (
        float(np.dot(centered_time, centered_values)) / time_square_sum
        if time_square_sum > 0
        else 0.0
    )
    total_drift = slope * (series.size - 1)
    normalized_drift = _safe_ratio(total_drift, temporal_std)

    midpoint = series.size // 2
    first = series[:midpoint]
    second = series[midpoint:]
    pooled_std = math.sqrt(
        0.5
        * (
            float(np.var(first, ddof=0))
            + float(np.var(second, ddof=0))
        )
    )
    half_shift = float(np.mean(first) - np.mean(second))
    normalized_half_shift = _safe_ratio(half_shift, pooled_std)

    block_means = [
        float(np.mean(block)) for block in np.split(series, block_count)
    ]
    block_range = max(block_means) - min(block_means)
    normalized_block_range = _safe_ratio(block_range, temporal_std)

    criteria = {
        "absolute_normalized_drift": abs(normalized_drift) <= max_abs_drift,
        "absolute_normalized_half_shift": (
            abs(normalized_half_shift) <= max_abs_half_shift
        ),
        "normalized_block_mean_range": normalized_block_range <= max_block_range,
    }
    return {
        "frames": int(series.size),
        "block_count": block_count,
        "block_frames": int(series.size // block_count),
        "mean": float(np.mean(series)),
        "temporal_std": temporal_std,
        "block_means": block_means,
        "fitted_slope_per_frame": slope,
        "normalized_total_fitted_drift": normalized_drift,
        "normalized_first_minus_second_half_shift": normalized_half_shift,
        "normalized_block_mean_range": normalized_block_range,
        "thresholds": {
            "max_absolute_normalized_drift": max_abs_drift,
            "max_absolute_normalized_half_shift": max_abs_half_shift,
            "max_normalized_block_mean_range": max_block_range,
        },
        "criteria_pass": criteria,
        "passes": all(criteria.values()),
    }


def operational_steady_screen(
    frame_means: Mapping[str, np.ndarray],
    fluctuation_rms: Mapping[str, np.ndarray],
) -> dict[str, object]:
    expected = set(C5_FIELDS)
    if set(frame_means) != expected or set(fluctuation_rms) != expected:
        raise ValueError("steady-state inputs must contain exactly the C5 fields")
    series_results: dict[str, object] = {}
    for field in C5_FIELDS:
        if field != "phi":
            series_results[f"{field}.spatial_mean"] = summarize_stationarity_series(
                frame_means[field]
            )
        series_results[f"{field}.fluctuation_rms"] = summarize_stationarity_series(
            fluctuation_rms[field]
        )
    failures = [
        name for name, result in series_results.items() if not result["passes"]
    ]
    return {
        "definition": "predeclared operational screen; not a proof of stationarity",
        "series": series_results,
        "failures": failures,
        "passes": not failures,
    }


def pattern_autocorrelation(frames: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized Eulerian pattern autocorrelation from fluctuation fields."""

    array = np.asarray(frames, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("frames must have time and at least one spatial axis")
    if max_lag < 0 or max_lag >= array.shape[0]:
        raise ValueError("max_lag must be in [0, number of frames)")
    flat = array.reshape(array.shape[0], -1)
    flat = flat - np.mean(flat, axis=1, keepdims=True)
    flat = flat - np.mean(flat, axis=0, keepdims=True)
    if not np.all(np.isfinite(flat)):
        raise ValueError("decorrelation input contains non-finite values")

    frame_energy = np.einsum("td,td->t", flat, flat, optimize=True)
    result = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        left = flat if lag == 0 else flat[:-lag]
        right = flat if lag == 0 else flat[lag:]
        numerator = float(np.einsum("td,td->", left, right, optimize=True))
        left_energy = frame_energy if lag == 0 else frame_energy[:-lag]
        right_energy = frame_energy if lag == 0 else frame_energy[lag:]
        denominator = math.sqrt(float(np.sum(left_energy) * np.sum(right_energy)))
        if denominator <= 0:
            raise ValueError("decorrelation field has zero fluctuation energy")
        result[lag] = numerator / denominator
    return result


def first_threshold_crossing(curve: np.ndarray, threshold: float) -> float | None:
    """Return the linearly interpolated first downward threshold crossing."""

    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return None
    for upper_index in range(1, values.size):
        left = float(values[upper_index - 1])
        right = float(values[upper_index])
        if left > threshold and right <= threshold:
            if left == right:
                return float(upper_index)
            fraction = (left - threshold) / (left - right)
            return float(upper_index - 1 + fraction)
    return None


def summarize_autocorrelation(
    curve: np.ndarray, cadence_microseconds: float
) -> dict[str, object]:
    values = np.asarray(curve, dtype=np.float64).reshape(-1)
    one_over_e = first_threshold_crossing(values, math.exp(-1.0))
    first_nonpositive = first_threshold_crossing(values, 0.0)
    positive_values: list[float] = []
    nonpositive_observed = False
    for value in values[1:]:
        if value <= 0:
            nonpositive_observed = True
            break
        positive_values.append(float(value))
    integrated = 1.0 + 2.0 * sum(positive_values)

    def physical(frames: float | None) -> float | None:
        return None if frames is None else frames * cadence_microseconds

    return {
        "rho": [float(value) for value in values],
        "max_lag_frames": int(values.size - 1),
        "one_over_e_crossing_frames": one_over_e,
        "one_over_e_crossing_microseconds": physical(one_over_e),
        "one_over_e_right_censored": one_over_e is None,
        "first_nonpositive_crossing_frames": first_nonpositive,
        "first_nonpositive_crossing_microseconds": physical(first_nonpositive),
        "first_nonpositive_right_censored": first_nonpositive is None,
        "initial_positive_sequence_integrated_frames": integrated,
        "initial_positive_sequence_integrated_microseconds": physical(integrated),
        "integrated_time_right_censored": not nonpositive_observed,
    }


def representative_decorrelation(
    per_field: Mapping[str, Mapping[str, object]], cadence_microseconds: float
) -> dict[str, object]:
    missing = set(C5_FIELDS) - set(per_field)
    if missing:
        raise ValueError(f"missing decorrelation fields: {sorted(missing)}")
    finite = [
        float(per_field[field]["one_over_e_crossing_frames"])
        for field in C5_FIELDS
        if per_field[field]["one_over_e_crossing_frames"] is not None
    ]
    if not finite:
        return {
            "status": "all_fields_right_censored",
            "median_one_over_e_frames": None,
            "median_one_over_e_microseconds": None,
            "maximum_one_over_e_frames": None,
            "maximum_one_over_e_microseconds": None,
        }
    median = float(np.median(np.asarray(finite, dtype=np.float64)))
    maximum = max(finite)
    return {
        "status": "finite" if len(finite) == len(C5_FIELDS) else "partially_censored",
        "finite_field_count": len(finite),
        "median_one_over_e_frames": median,
        "median_one_over_e_microseconds": median * cadence_microseconds,
        "maximum_one_over_e_frames": maximum,
        "maximum_one_over_e_microseconds": maximum * cadence_microseconds,
    }

