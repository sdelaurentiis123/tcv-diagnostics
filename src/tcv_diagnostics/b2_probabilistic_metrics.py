"""Finite-ensemble conventions frozen specifically for Paper 0 B2.

This module is separate from the historical Phase 2 metric primitives so that
adding B2 calibration conventions cannot change any hash-locked O1 launcher.
It reads no simulation data and contains no training loss.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is out of bounds for rank {ndim}")
    return normalized


def _real_finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_scalar_ensemble(
    forecast: np.ndarray,
    truth: np.ndarray,
    member_axis: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    forecast_array = _real_finite("ensemble forecast", forecast)
    truth_array = _real_finite("ensemble truth", truth)
    if forecast_array.ndim < 1:
        raise ValueError("ensemble forecast must have at least one axis")
    axis = _normalize_axis(member_axis, forecast_array.ndim)
    expected_truth_shape = forecast_array.shape[:axis] + forecast_array.shape[axis + 1 :]
    if truth_array.shape != expected_truth_shape:
        raise ValueError(
            "truth shape must equal forecast shape without the member axis: "
            f"expected {expected_truth_shape}, received {truth_array.shape}"
        )
    if forecast_array.shape[axis] < 1:
        raise ValueError("ensemble forecast must contain at least one member")
    if truth_array.size < 1:
        raise ValueError("ensemble truth must contain at least one value")
    return forecast_array, truth_array, axis


def corrected_spread_skill_summary(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
) -> dict[str, float | int]:
    """Return the frozen finite-member B2 spread-skill summary.

    Member variance uses ``ddof=1``. Multiplication by ``(M+1)/M`` matches
    the expected squared error of an ensemble mean against one additional
    exchangeable realization. Numerator and denominator are reduced before
    division.
    """

    forecast_array, truth_array, axis = _validate_scalar_ensemble(
        forecast, truth, member_axis
    )
    members = int(forecast_array.shape[axis])
    if members < 2:
        raise ValueError("corrected spread-skill requires at least two members")
    ensemble_mean = np.mean(forecast_array, axis=axis)
    unbiased_variance = np.var(forecast_array, axis=axis, ddof=1)
    mean_unbiased_variance = float(np.mean(unbiased_variance))
    correction = float((members + 1) / members)
    corrected_rms_spread = float(np.sqrt(correction * mean_unbiased_variance))
    rmse = float(np.sqrt(np.mean((ensemble_mean - truth_array) ** 2)))
    ratio = math.nan if rmse == 0.0 else corrected_rms_spread / rmse
    return {
        "ensemble_size": members,
        "mean_unbiased_member_variance": mean_unbiased_variance,
        "finite_member_variance_factor": correction,
        "corrected_rms_spread": corrected_rms_spread,
        "rmse_of_ensemble_mean": rmse,
        "spread_skill_ratio": ratio,
        "member_variance_ddof": 1,
    }


def order_statistic_interval_coverage(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    lower_order_one_indexed: int,
    upper_order_one_indexed: int,
    member_axis: int = 1,
) -> dict[str, Any]:
    """Evaluate one exact finite-ensemble order-statistic interval.

    For continuous exchangeable truth and ``M`` members, the interval
    ``[x_(lower), x_(upper)]`` has nominal coverage
    ``(upper-lower)/(M+1)``. No interpolated quantile is used.
    """

    forecast_array, truth_array, axis = _validate_scalar_ensemble(
        forecast, truth, member_axis
    )
    members = int(forecast_array.shape[axis])
    lower = int(lower_order_one_indexed)
    upper = int(upper_order_one_indexed)
    if not 1 <= lower < upper <= members:
        raise ValueError(
            "order statistics must satisfy 1 <= lower < upper <= members"
        )
    sorted_forecast = np.sort(forecast_array, axis=axis)
    lower_values = np.take(sorted_forecast, lower - 1, axis=axis)
    upper_values = np.take(sorted_forecast, upper - 1, axis=axis)
    covered = (truth_array >= lower_values) & (truth_array <= upper_values)
    return {
        "ensemble_size": members,
        "lower_order_one_indexed": lower,
        "upper_order_one_indexed": upper,
        "nominal_coverage": float((upper - lower) / (members + 1)),
        "empirical_coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper_values - lower_values)),
        "lower": lower_values,
        "upper": upper_values,
        "covered": covered,
        "quantile_method": "exact_order_statistics_no_interpolation",
    }


def _nonnegative_integer_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return np.asarray(array, dtype=np.uint64)


def deterministic_tie_uniform(
    target_frame: np.ndarray,
    channel_index: np.ndarray,
    spatial_cell_index: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Map frozen rank keys to reproducible uniforms in ``[0,1)``.

    The vectorized SplitMix64 finalizer is stateless, so chunking a field does
    not change a cell's tie-breaking draw.
    """

    if not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**64:
        raise ValueError("tie-breaking seed must be an unsigned 64-bit integer")
    target = _nonnegative_integer_array("target_frame", target_frame)
    channel = _nonnegative_integer_array("channel_index", channel_index)
    cell = _nonnegative_integer_array("spatial_cell_index", spatial_cell_index)
    try:
        target, channel, cell = np.broadcast_arrays(target, channel, cell)
    except ValueError as error:
        raise ValueError("rank-key arrays are not broadcast-compatible") from error

    with np.errstate(over="ignore"):
        key = np.full(target.shape, np.uint64(seed), dtype=np.uint64)
        key ^= target * np.uint64(0xD2B74407B1CE6E93)
        key ^= channel * np.uint64(0xCA5A826395121157)
        key ^= cell * np.uint64(0x9E3779B97F4A7C15)
        key += np.uint64(0x9E3779B97F4A7C15)
        key = (key ^ (key >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        key = (key ^ (key >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        key ^= key >> np.uint64(31)
    return (key >> np.uint64(11)).astype(np.float64) * (2.0**-53)


def ensemble_rank_histogram(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    member_axis: int = 1,
    tie_uniform: np.ndarray | None = None,
    return_ranks: bool = False,
) -> dict[str, Any]:
    """Return a finite-ensemble rank histogram with explicit tie handling.

    If truth equals one or more members, ``tie_uniform`` must provide one
    frozen uniform per truth value. The truth is inserted uniformly among all
    tied positions, producing ranks from zero through ``M``.
    """

    forecast_array, truth_array, axis = _validate_scalar_ensemble(
        forecast, truth, member_axis
    )
    members = int(forecast_array.shape[axis])
    expanded_truth = np.expand_dims(truth_array, axis=axis)
    less = np.sum(forecast_array < expanded_truth, axis=axis, dtype=np.int64)
    equal = np.sum(forecast_array == expanded_truth, axis=axis, dtype=np.int64)
    tied = equal > 0
    offsets = np.zeros(truth_array.shape, dtype=np.int64)
    if np.any(tied):
        if tie_uniform is None:
            raise ValueError("tie_uniform is required when ensemble ties are present")
        uniforms = _real_finite("tie_uniform", tie_uniform)
        if uniforms.shape != truth_array.shape:
            raise ValueError("tie_uniform shape must match truth")
        if np.any((uniforms < 0.0) | (uniforms >= 1.0)):
            raise ValueError("tie_uniform values must lie in [0,1)")
        offsets[tied] = np.floor(
            uniforms[tied] * (equal[tied] + 1)
        ).astype(np.int64)
    ranks = less + offsets
    if np.any((ranks < 0) | (ranks > members)):
        raise RuntimeError("computed ensemble rank lies outside 0..M")
    counts = np.bincount(ranks.ravel(), minlength=members + 1).astype(np.int64)
    total = int(np.sum(counts))
    frequencies = counts.astype(np.float64) / total
    uniform_frequency = 1.0 / (members + 1)
    rank_values = ranks.astype(np.float64)
    uniform_variance = float(members * (members + 2) / 12.0)
    observed_variance = float(np.var(rank_values, ddof=0))
    record: dict[str, Any] = {
        "ensemble_size": members,
        "bins": members + 1,
        "counts": counts,
        "frequencies": frequencies,
        "total": total,
        "tied_truth_values": int(np.sum(tied)),
        "normalized_rank_mean": float(np.mean(rank_values) / members),
        "rank_variance": observed_variance,
        "uniform_rank_variance": uniform_variance,
        "rank_variance_ratio": float(observed_variance / uniform_variance),
        "total_variation_from_uniform": float(
            0.5 * np.sum(np.abs(frequencies - uniform_frequency))
        ),
        "pixel_iid_p_value_reported": False,
    }
    if return_ranks:
        record["ranks"] = ranks
    return record


def member_prefix_views(
    forecast: np.ndarray,
    prefixes: Sequence[int],
    *,
    member_axis: int = 1,
) -> dict[int, np.ndarray]:
    """Return ordered prefix views from one already-generated ensemble."""

    array = np.asarray(forecast)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError("ensemble forecast must be real numeric data")
    if not np.all(np.isfinite(array)):
        raise ValueError("ensemble forecast contains non-finite values")
    axis = _normalize_axis(member_axis, array.ndim)
    requested = tuple(int(value) for value in prefixes)
    if not requested or any(value <= 0 for value in requested):
        raise ValueError("member prefixes must be positive and nonempty")
    if tuple(sorted(set(requested))) != requested:
        raise ValueError("member prefixes must be strictly increasing")
    if requested[-1] > array.shape[axis]:
        raise ValueError("a member prefix exceeds the stored ensemble")
    views: dict[int, np.ndarray] = {}
    for count in requested:
        index = [slice(None)] * array.ndim
        index[axis] = slice(0, count)
        views[count] = array[tuple(index)]
    return views


def monte_carlo_stability(
    value_m16: float,
    value_m32: float,
    *,
    relative_difference_max: float = 0.1,
    absolute_floor: float = 1.0e-8,
) -> dict[str, float | bool]:
    """Apply the frozen B2 M=16 versus M=32 scalar stability rule."""

    first = float(value_m16)
    second = float(value_m32)
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("Monte Carlo stability values must be finite")
    if relative_difference_max < 0.0 or absolute_floor < 0.0:
        raise ValueError("Monte Carlo tolerances must be nonnegative")
    difference = abs(first - second)
    tolerance = relative_difference_max * abs(second) + absolute_floor
    return {
        "M16": first,
        "M32": second,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "relative_difference_max": float(relative_difference_max),
        "absolute_floor": float(absolute_floor),
        "passes": difference <= tolerance,
    }


def moving_block_bootstrap_indices(
    length: int,
    *,
    block_length: int,
    replicates: int,
    seed: int,
    blocks_per_replicate: int | None = None,
) -> np.ndarray:
    """Return reproducible moving-block bootstrap target indices.

    Blocks are sampled with replacement from every valid contiguous start,
    concatenated, and truncated to ``length``. This returns temporal indices
    only; nonlinear metrics must be recomputed from indexed primitives.
    """

    sample_length = int(length)
    block = int(block_length)
    repeats = int(replicates)
    if sample_length <= 0 or block <= 0 or repeats <= 0:
        raise ValueError(
            "bootstrap length, block length, and replicates must be positive"
        )
    if block > sample_length:
        raise ValueError("bootstrap block cannot exceed the series length")
    required_blocks = math.ceil(sample_length / block)
    blocks = (
        required_blocks
        if blocks_per_replicate is None
        else int(blocks_per_replicate)
    )
    if blocks < required_blocks:
        raise ValueError("too few bootstrap blocks to cover the requested length")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    valid_start_count = sample_length - block + 1
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    starts = generator.integers(
        0,
        valid_start_count,
        size=(repeats, blocks),
        dtype=np.int64,
    )
    offsets = np.arange(block, dtype=np.int64)
    indices = (starts[..., None] + offsets).reshape(repeats, blocks * block)
    return indices[:, :sample_length]
