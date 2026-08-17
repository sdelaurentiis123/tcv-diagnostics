"""Gauge-invariant helpers for the frozen saved-potential boundary audit."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SIDES = ("inner", "outer")
PLANE_NAMES = ("outermost_guard", "adjacent_guard", "adjacent_interior")
SIDE_INDICES = {
    "inner": {"outermost_guard": 0, "adjacent_guard": 1, "adjacent_interior": 2},
    "outer": {"outermost_guard": 7, "adjacent_guard": 6, "adjacent_interior": 5},
}


def extract_boundary_planes(
    phi: np.ndarray,
    *,
    side: str,
    physical_y_slice: tuple[int, int] = (2, 4),
) -> dict[str, np.ndarray]:
    """Extract the three predeclared x planes while retaining native z."""

    if side not in SIDE_INDICES:
        raise KeyError(side)
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 4:
        raise ValueError("phi must use [time,x,y,z]")
    first_y, stop_y = physical_y_slice
    if not (0 <= first_y < stop_y <= phi.shape[2]):
        raise ValueError("physical y slice lies outside phi")
    if phi.shape[1] != 8:
        raise ValueError("frozen raw-rank phi must have x=8")
    return {
        name: phi[:, x_index, first_y:stop_y, :]
        for name, x_index in SIDE_INDICES[side].items()
    }


def _safe_relative(numerator: float, denominator: float) -> float | None:
    if denominator > 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return 0.0
    return None


def _finite_summary(
    values: np.ndarray,
    *,
    conversion: float,
    percentiles: tuple[float, ...],
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    selected = values[finite]
    result: dict[str, Any] = {
        "total_count": int(values.size),
        "finite_count": int(selected.size),
        "nonfinite_count": int(values.size - selected.size),
    }
    if selected.size == 0:
        return {
            **result,
            "mean": None,
            "rms": None,
            "maximum_absolute": None,
            "absolute_percentiles": {str(value): None for value in percentiles},
            "physical_conversion": conversion,
            "mean_physical": None,
            "rms_physical": None,
            "maximum_absolute_physical": None,
        }
    absolute = np.abs(selected)
    mean = float(np.mean(selected, dtype=np.float64))
    rms = float(np.sqrt(np.mean(selected * selected, dtype=np.float64)))
    maximum = float(np.max(absolute))
    return {
        **result,
        "mean": mean,
        "rms": rms,
        "maximum_absolute": maximum,
        "absolute_percentiles": {
            str(value): float(np.percentile(absolute, value)) for value in percentiles
        },
        "physical_conversion": conversion,
        "mean_physical": mean * conversion,
        "rms_physical": rms * conversion,
        "maximum_absolute_physical": maximum * conversion,
    }


def _maximum_error_record(
    error: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    location_axes: tuple[str, ...],
) -> dict[str, Any] | None:
    finite = np.isfinite(error) & np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(finite):
        return None
    safe = np.where(finite, np.abs(error), -np.inf)
    flat_index = int(np.argmax(safe))
    location = np.unravel_index(flat_index, error.shape)
    return {
        "absolute_error": float(safe.flat[flat_index]),
        "reference": float(reference.flat[flat_index]),
        "candidate": float(candidate.flat[flat_index]),
        "location": {
            axis: int(value) for axis, value in zip(location_axes, location)
        },
    }


def _pointwise_check(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
    temporal_blocks: list[tuple[int, int]],
    location_axes: tuple[str, ...],
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    with np.errstate(invalid="ignore", over="ignore"):
        error = candidate - reference
    finite &= np.isfinite(error)
    discrepancy = finite & (np.abs(error) > atol + rtol * np.abs(reference))
    nonfinite = ~finite
    frame_axes = tuple(range(1, discrepancy.ndim))
    by_frame = np.sum(discrepancy, axis=frame_axes, dtype=np.int64)
    y_axes = tuple(axis for axis in range(discrepancy.ndim) if axis != 1)
    by_global_y = np.sum(discrepancy, axis=y_axes, dtype=np.int64)
    return {
        "total_count": int(reference.size),
        "nonfinite_count": int(np.count_nonzero(nonfinite)),
        "point_discrepancy_count": int(np.count_nonzero(discrepancy)),
        "point_discrepancy_count_by_frame": by_frame.astype(int).tolist(),
        "point_discrepancy_count_by_global_y": by_global_y.astype(int).tolist(),
        "point_discrepancy_count_by_temporal_block": [
            int(np.sum(by_frame[first : last + 1]))
            for first, last in temporal_blocks
        ],
        "maximum_error": _maximum_error_record(
            error,
            reference,
            candidate,
            location_axes=location_axes,
        ),
    }


def _lag_one_correlation(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    first = values[:-1]
    second = values[1:]
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 3:
        return None
    first = first[finite]
    second = second[finite]
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(np.sqrt(np.sum(first * first) * np.sum(second * second)))
    if denominator == 0.0:
        return None
    return float(np.sum(first * second) / denominator)


def analyze_side(
    planes: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
    conversion_volts: float,
    percentiles: tuple[float, ...],
    temporal_blocks: list[tuple[int, int]],
) -> dict[str, Any]:
    """Evaluate one side with arrays shaped [time,global_y,z]."""

    if set(planes) != set(PLANE_NAMES):
        raise ValueError("boundary planes are incomplete")
    shapes = {np.asarray(values).shape for values in planes.values()}
    if len(shapes) != 1:
        raise ValueError("boundary planes must have identical shapes")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[1:] != (32, 81):
        raise ValueError("assembled boundary planes must use [time,32,81]")
    outermost = np.asarray(planes["outermost_guard"], dtype=np.float64)
    adjacent = np.asarray(planes["adjacent_guard"], dtype=np.float64)
    interior = np.asarray(planes["adjacent_interior"], dtype=np.float64)
    midpoint = 0.5 * (adjacent + interior)
    midpoint_mean = np.mean(midpoint, axis=-1)
    target = np.mean(interior, axis=-1)
    departure_scalar = midpoint_mean - target
    departure = midpoint - target[..., None]
    interior_fluctuation = interior - target[..., None]

    outer_copy = _pointwise_check(
        adjacent,
        outermost,
        atol=atol,
        rtol=rtol,
        temporal_blocks=temporal_blocks,
        location_axes=("frame", "global_y", "z"),
    )
    midpoint_reference = np.broadcast_to(midpoint_mean[..., None], midpoint.shape)
    midpoint_constancy = _pointwise_check(
        midpoint_reference,
        midpoint,
        atol=atol,
        rtol=rtol,
        temporal_blocks=temporal_blocks,
        location_axes=("frame", "global_y", "z"),
    )
    instantaneous_neumann = _pointwise_check(
        target,
        midpoint_mean,
        atol=atol,
        rtol=rtol,
        temporal_blocks=temporal_blocks,
        location_axes=("frame", "global_y"),
    )
    departure_summary = _finite_summary(
        departure,
        conversion=conversion_volts,
        percentiles=percentiles,
    )
    fluctuation_summary = _finite_summary(
        interior_fluctuation,
        conversion=conversion_volts,
        percentiles=percentiles,
    )
    departure_rms = departure_summary["rms"]
    fluctuation_rms = fluctuation_summary["rms"]
    departure_summary["rms_to_interior_toroidal_fluctuation_rms"] = (
        None
        if departure_rms is None or fluctuation_rms is None
        else _safe_relative(float(departure_rms), float(fluctuation_rms))
    )
    by_y = []
    for global_y in range(32):
        summary = _finite_summary(
            departure[:, global_y, :],
            conversion=conversion_volts,
            percentiles=percentiles,
        )
        summary["global_y"] = global_y
        summary["lag_one_correlation_of_midpoint_departure"] = _lag_one_correlation(
            departure_scalar[:, global_y]
        )
        by_y.append(summary)
    by_block = []
    for block_index, (first, last) in enumerate(temporal_blocks):
        summary = _finite_summary(
            departure[first : last + 1],
            conversion=conversion_volts,
            percentiles=percentiles,
        )
        summary.update(
            {"block_index": block_index, "first_frame": first, "last_frame": last}
        )
        by_block.append(summary)
    finite_correlations = [
        record["lag_one_correlation_of_midpoint_departure"]
        for record in by_y
        if record["lag_one_correlation_of_midpoint_departure"] is not None
    ]
    return {
        "plane_nonfinite_counts": {
            name: int(np.count_nonzero(~np.isfinite(values)))
            for name, values in planes.items()
        },
        "outer_guard_copy": outer_copy,
        "midpoint_toroidal_constancy": midpoint_constancy,
        "instantaneous_neumann": instantaneous_neumann,
        "departure": departure_summary,
        "interior_toroidal_fluctuation": fluctuation_summary,
        "by_global_y": by_y,
        "by_temporal_block": by_block,
        "lag_one_correlation_summary": {
            "finite_y_count": len(finite_correlations),
            "mean_across_y": (
                float(np.mean(finite_correlations)) if finite_correlations else None
            ),
            "minimum_across_y": min(finite_correlations) if finite_correlations else None,
            "maximum_across_y": max(finite_correlations) if finite_correlations else None,
        },
    }


def derive_findings(per_side: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(per_side) != set(SIDES):
        raise ValueError("both radial sides are required")
    all_finite = all(
        all(count == 0 for count in result["plane_nonfinite_counts"].values())
        for result in per_side.values()
    )
    outer_copy = all(
        result["outer_guard_copy"]["point_discrepancy_count"] == 0
        and result["outer_guard_copy"]["nonfinite_count"] == 0
        for result in per_side.values()
    )
    midpoint_constant = all(
        result["midpoint_toroidal_constancy"]["point_discrepancy_count"] == 0
        and result["midpoint_toroidal_constancy"]["nonfinite_count"] == 0
        for result in per_side.values()
    )
    neumann_everywhere = all(
        result["instantaneous_neumann"]["point_discrepancy_count"] == 0
        and result["instantaneous_neumann"]["nonfinite_count"] == 0
        for result in per_side.values()
    )
    structural = all_finite and outer_copy and midpoint_constant
    return {
        "all_boundary_planes_finite": all_finite,
        "outer_guard_copy_passes": outer_copy,
        "midpoint_toroidal_constancy_passes": midpoint_constant,
        "saved_compact_boundary_value_structurally_valid": structural,
        "instantaneous_neumann_passes_everywhere": neumann_everywhere,
        "nonzero_saved_boundary_state_detected": structural and not neumann_everywhere,
        "materiality_established": False,
        "paired_elliptic_solve_required_for_materiality": True,
        "automatic_state_change_authorized": False,
        "potential_vorticity_gate_completed": False,
    }
