"""Pure helpers for the 85604 evolved-state and momentum-closure audit."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


EVOLVED_FIELDS = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")
DERIVED_FIELDS = ("Te", "Ti", "Ve", "Vi", "phi")
INVENTORY_FIELDS = EVOLVED_FIELDS + DERIVED_FIELDS
STREAM_FIELDS = EVOLVED_FIELDS + ("Ve", "Vi")
RELATIONS = (
    "NVe_from_softfloor_Ne_Ve",
    "NVi_from_softfloor_Ne_Vi",
    "NVe_from_plain_Ne_Ve",
    "NVi_from_plain_Ne_Vi",
)
SOURCE_EXACT_RELATIONS = RELATIONS[:2]
SCOPE_NAMES = (
    "full_physical_domain",
    "guard_independent_transport_interior",
    "target_dependent_rows",
)


def soft_floor(values: np.ndarray, minimum: float) -> np.ndarray:
    """Apply the exact scalar Hermes softFloor expression elementwise."""

    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("soft-floor minimum must be positive and finite")
    values = np.asarray(values, dtype=np.float64)
    nonnegative = np.maximum(values, 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return nonnegative + minimum * np.exp(-nonnegative / minimum)


def relation_arrays(
    name: str,
    fields: dict[str, np.ndarray],
    *,
    density_floor: float,
    electron_atomic_mass: float,
    ion_atomic_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference and candidate arrays for one frozen momentum relation."""

    if name not in RELATIONS:
        raise KeyError(name)
    density = fields["Ne"]
    if "softfloor" in name:
        density = soft_floor(density, density_floor)
    if name.startswith("NVe"):
        reference = fields["NVe"]
        candidate = electron_atomic_mass * density * fields["Ve"]
    else:
        reference = fields["NVi"]
        candidate = ion_atomic_mass * density * fields["Vi"]
    return reference, candidate


def scope_y_indices(global_y: np.ndarray, scope_name: str) -> np.ndarray:
    global_y = np.asarray(global_y)
    if scope_name == "full_physical_domain":
        mask = np.ones(global_y.shape, dtype=bool)
    elif scope_name == "guard_independent_transport_interior":
        mask = (global_y >= 1) & (global_y <= 30)
    elif scope_name == "target_dependent_rows":
        mask = (global_y == 0) | (global_y == 31)
    else:
        raise KeyError(scope_name)
    return np.flatnonzero(mask)


def selected_location(
    flat_index: int,
    shape: tuple[int, int, int, int],
    *,
    x0: int,
    global_y: np.ndarray,
) -> tuple[int, int, int, int]:
    frame, local_x, selected_y, z_index = np.unravel_index(flat_index, shape)
    return (
        int(frame),
        int(x0 + local_x),
        int(global_y[selected_y]),
        int(z_index),
    )


def _point(value: float, location: tuple[int, int, int, int]) -> dict[str, Any]:
    return {"value": float(value), "location_txyz": [int(item) for item in location]}


def _relative_l2(sum_squared_error: float, sum_squared_reference: float) -> float | None:
    if sum_squared_reference > 0.0:
        return math.sqrt(sum_squared_error / sum_squared_reference)
    if sum_squared_error == 0.0:
        return 0.0
    return None


class FieldScopeAccumulator:
    def __init__(self) -> None:
        self.total_count = 0
        self.finite_count = 0
        self.nonfinite_count = 0
        self.sum = 0.0
        self.sum_squares = 0.0
        self.minimum: dict[str, Any] | None = None
        self.maximum: dict[str, Any] | None = None

    def update(self, values: np.ndarray, *, x0: int, global_y: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.total_count += int(values.size)
        finite = np.isfinite(values)
        finite_values = values[finite]
        self.finite_count += int(finite_values.size)
        self.nonfinite_count += int(values.size - finite_values.size)
        if finite_values.size == 0:
            return
        self.sum += float(np.sum(finite_values, dtype=np.float64))
        self.sum_squares += float(
            np.sum(finite_values * finite_values, dtype=np.float64)
        )
        minimum_index = int(np.argmin(np.where(finite, values, np.inf)))
        maximum_index = int(np.argmax(np.where(finite, values, -np.inf)))
        minimum = _point(
            float(values.flat[minimum_index]),
            selected_location(minimum_index, values.shape, x0=x0, global_y=global_y),
        )
        maximum = _point(
            float(values.flat[maximum_index]),
            selected_location(maximum_index, values.shape, x0=x0, global_y=global_y),
        )
        if self.minimum is None or minimum["value"] < self.minimum["value"]:
            self.minimum = minimum
        if self.maximum is None or maximum["value"] > self.maximum["value"]:
            self.maximum = maximum

    def result(self) -> dict[str, Any]:
        rms = (
            math.sqrt(self.sum_squares / self.finite_count)
            if self.finite_count
            else None
        )
        return {
            "total_count": self.total_count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "sum": self.sum,
            "sum_squares": self.sum_squares,
            "rms": rms,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


class FieldAccumulator:
    def __init__(self) -> None:
        self.scopes = {name: FieldScopeAccumulator() for name in SCOPE_NAMES}

    def update(self, values: np.ndarray, *, x0: int, y0: int) -> None:
        global_y = y0 + np.arange(values.shape[2])
        for name, accumulator in self.scopes.items():
            local_y = scope_y_indices(global_y, name)
            if local_y.size:
                accumulator.update(
                    values[:, :, local_y, :],
                    x0=x0,
                    global_y=global_y[local_y],
                )

    def result(self) -> dict[str, Any]:
        return {"scopes": {name: value.result() for name, value in self.scopes.items()}}


class DensityFloorAccumulator:
    EVENT_NAMES = ("below_zero", "below_density_floor", "softfloor_changed")

    def __init__(
        self,
        *,
        frame_count: int,
        nx: int,
        ny: int,
        density_floor: float,
        temporal_blocks: list[tuple[int, int]],
    ) -> None:
        self.density_floor = density_floor
        self.temporal_blocks = temporal_blocks
        self.scope_counts = {
            scope: {event: 0 for event in self.EVENT_NAMES} for scope in SCOPE_NAMES
        }
        self.by_frame = {
            event: np.zeros(frame_count, dtype=np.int64) for event in self.EVENT_NAMES
        }
        self.by_x = {
            event: np.zeros(nx, dtype=np.int64) for event in self.EVENT_NAMES
        }
        self.by_y = {
            event: np.zeros(ny, dtype=np.int64) for event in self.EVENT_NAMES
        }

    def _events(self, values: np.ndarray) -> dict[str, np.ndarray]:
        finite = np.isfinite(values)
        floored = soft_floor(values, self.density_floor)
        return {
            "below_zero": finite & (values < 0.0),
            "below_density_floor": finite & (values < self.density_floor),
            "softfloor_changed": finite & (floored != values),
        }

    def update(self, values: np.ndarray, *, x0: int, y0: int) -> None:
        values = np.asarray(values, dtype=np.float64)
        events = self._events(values)
        global_y = y0 + np.arange(values.shape[2])
        for scope in SCOPE_NAMES:
            local_y = scope_y_indices(global_y, scope)
            if local_y.size:
                for event, mask in events.items():
                    self.scope_counts[scope][event] += int(
                        np.count_nonzero(mask[:, :, local_y, :])
                    )
        for event, mask in events.items():
            self.by_frame[event] += np.sum(mask, axis=(1, 2, 3), dtype=np.int64)
            self.by_x[event][x0 : x0 + values.shape[1]] += np.sum(
                mask, axis=(0, 2, 3), dtype=np.int64
            )
            self.by_y[event][y0 : y0 + values.shape[2]] += np.sum(
                mask, axis=(0, 1, 3), dtype=np.int64
            )

    def result(self) -> dict[str, Any]:
        return {
            "density_floor": self.density_floor,
            "scope_counts": self.scope_counts,
            "count_by_frame": {name: values.tolist() for name, values in self.by_frame.items()},
            "count_by_x": {name: values.tolist() for name, values in self.by_x.items()},
            "count_by_y": {name: values.tolist() for name, values in self.by_y.items()},
            "count_by_temporal_block": {
                name: [
                    int(np.sum(values[first : last + 1]))
                    for first, last in self.temporal_blocks
                ]
                for name, values in self.by_frame.items()
            },
        }


class ClosureScopeAccumulator:
    def __init__(self, *, frame_count: int, atol: float, rtol: float) -> None:
        self.atol = atol
        self.rtol = rtol
        self.total_count = 0
        self.nonfinite_count = 0
        self.point_discrepancy_count = 0
        self.sum_squared_error = 0.0
        self.sum_squared_reference = 0.0
        self.maximum_error: dict[str, Any] | None = None
        self.frame_max_abs_error = np.zeros(frame_count, dtype=np.float64)
        self.frame_max_abs_reference = np.zeros(frame_count, dtype=np.float64)
        self.frame_nonfinite_count = np.zeros(frame_count, dtype=np.int64)
        self.frame_point_discrepancy_count = np.zeros(frame_count, dtype=np.int64)

    def update(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        *,
        x0: int,
        global_y: np.ndarray,
    ) -> np.ndarray:
        reference = np.asarray(reference, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        self.total_count += int(reference.size)
        finite = np.isfinite(reference) & np.isfinite(candidate)
        with np.errstate(invalid="ignore", over="ignore"):
            error = candidate - reference
            absolute_error = np.abs(error)
        finite &= np.isfinite(absolute_error)
        nonfinite = ~finite
        self.nonfinite_count += int(np.count_nonzero(nonfinite))
        self.frame_nonfinite_count += np.sum(nonfinite, axis=(1, 2, 3), dtype=np.int64)

        safe_error = np.where(finite, error, 0.0)
        safe_absolute_error = np.abs(safe_error)
        safe_reference = np.where(finite, reference, 0.0)
        self.sum_squared_error += float(
            np.sum(safe_error * safe_error, dtype=np.float64)
        )
        self.sum_squared_reference += float(
            np.sum(safe_reference * safe_reference, dtype=np.float64)
        )
        self.frame_max_abs_error = np.maximum(
            self.frame_max_abs_error,
            np.max(safe_absolute_error, axis=(1, 2, 3)),
        )
        self.frame_max_abs_reference = np.maximum(
            self.frame_max_abs_reference,
            np.max(np.abs(safe_reference), axis=(1, 2, 3)),
        )
        discrepancy = finite & (
            absolute_error > self.atol + self.rtol * np.abs(reference)
        )
        self.point_discrepancy_count += int(np.count_nonzero(discrepancy))
        self.frame_point_discrepancy_count += np.sum(
            discrepancy, axis=(1, 2, 3), dtype=np.int64
        )

        if np.any(finite):
            maximum_index = int(np.argmax(safe_absolute_error))
            maximum = {
                "absolute_error": float(safe_absolute_error.flat[maximum_index]),
                "location_txyz": list(
                    selected_location(
                        maximum_index,
                        reference.shape,
                        x0=x0,
                        global_y=global_y,
                    )
                ),
                "reference": float(reference.flat[maximum_index]),
                "candidate": float(candidate.flat[maximum_index]),
            }
            if (
                self.maximum_error is None
                or maximum["absolute_error"] > self.maximum_error["absolute_error"]
            ):
                self.maximum_error = maximum
        return discrepancy

    def result(self) -> dict[str, Any]:
        tolerance = self.atol + self.rtol * self.frame_max_abs_reference
        passed = (self.frame_nonfinite_count == 0) & (
            self.frame_max_abs_error <= tolerance
        )
        return {
            "total_count": self.total_count,
            "nonfinite_count": self.nonfinite_count,
            "point_discrepancy_count": self.point_discrepancy_count,
            "sum_squared_error": self.sum_squared_error,
            "sum_squared_reference": self.sum_squared_reference,
            "relative_l2_error": _relative_l2(
                self.sum_squared_error, self.sum_squared_reference
            ),
            "maximum_error": self.maximum_error,
            "frame_pass_count": int(np.count_nonzero(passed)),
            "frame_fail_count": int(np.count_nonzero(~passed)),
            "failed_frame_indices": np.flatnonzero(~passed).astype(int).tolist(),
            "frame_max_abs_error": self.frame_max_abs_error.tolist(),
            "frame_max_abs_reference": self.frame_max_abs_reference.tolist(),
            "frame_tolerance": tolerance.tolist(),
            "frame_passed": passed.tolist(),
            "frame_nonfinite_count": self.frame_nonfinite_count.tolist(),
            "frame_point_discrepancy_count": self.frame_point_discrepancy_count.tolist(),
        }


class ClosureAccumulator:
    def __init__(
        self,
        *,
        frame_count: int,
        nx: int,
        ny: int,
        temporal_blocks: list[tuple[int, int]],
        atol: float,
        rtol: float,
    ) -> None:
        self.scopes = {
            name: ClosureScopeAccumulator(frame_count=frame_count, atol=atol, rtol=rtol)
            for name in SCOPE_NAMES
        }
        self.temporal_blocks = temporal_blocks
        self.discrepancy_by_frame = np.zeros(frame_count, dtype=np.int64)
        self.discrepancy_by_x = np.zeros(nx, dtype=np.int64)
        self.discrepancy_by_y = np.zeros(ny, dtype=np.int64)

    def update(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        *,
        x0: int,
        y0: int,
    ) -> None:
        global_y = y0 + np.arange(reference.shape[2])
        full_discrepancy: np.ndarray | None = None
        for name, accumulator in self.scopes.items():
            local_y = scope_y_indices(global_y, name)
            if not local_y.size:
                continue
            discrepancy = accumulator.update(
                reference[:, :, local_y, :],
                candidate[:, :, local_y, :],
                x0=x0,
                global_y=global_y[local_y],
            )
            if name == "full_physical_domain":
                full_discrepancy = discrepancy
        if full_discrepancy is None:
            raise RuntimeError("full-domain closure scope is empty")
        self.discrepancy_by_frame += np.sum(
            full_discrepancy, axis=(1, 2, 3), dtype=np.int64
        )
        self.discrepancy_by_x[x0 : x0 + reference.shape[1]] += np.sum(
            full_discrepancy, axis=(0, 2, 3), dtype=np.int64
        )
        self.discrepancy_by_y[y0 : y0 + reference.shape[2]] += np.sum(
            full_discrepancy, axis=(0, 1, 3), dtype=np.int64
        )

    def result(self) -> dict[str, Any]:
        return {
            "scopes": {name: value.result() for name, value in self.scopes.items()},
            "point_discrepancy_count_by_frame": self.discrepancy_by_frame.tolist(),
            "point_discrepancy_count_by_x": self.discrepancy_by_x.tolist(),
            "point_discrepancy_count_by_y": self.discrepancy_by_y.tolist(),
            "point_discrepancy_count_by_temporal_block": [
                int(np.sum(self.discrepancy_by_frame[first : last + 1]))
                for first, last in self.temporal_blocks
            ],
        }


def initialize_stream_digests(fields: Iterable[str], shape: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in fields:
        digest = hashlib.sha256()
        header = json.dumps(
            {
                "field": field,
                "dtype": "<f8",
                "global_shape": shape,
                "stream_order": "ascending rank with rank,PE_XIND,PE_YIND headers",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(header)
        digest.update(b"\0")
        result[field] = digest
    return result


def update_stream_digest(
    digest: Any,
    values: np.ndarray,
    *,
    rank: int,
    pe_x: int,
    pe_y: int,
) -> None:
    header = json.dumps(
        {"rank": rank, "PE_XIND": pe_x, "PE_YIND": pe_y},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))


def derive_findings(
    field_statistics: dict[str, Any],
    closures: dict[str, Any],
) -> dict[str, Any]:
    relevant_fields = ("Ne", "NVe", "NVi", "Ve", "Vi")
    all_relevant_finite = all(
        field_statistics[field]["scopes"]["full_physical_domain"]["nonfinite_count"]
        == 0
        for field in relevant_fields
    )
    exact_pass = all(
        closures[name]["scopes"]["full_physical_domain"]["frame_fail_count"] == 0
        for name in SOURCE_EXACT_RELATIONS
    )
    if not all_relevant_finite:
        recommendation = "resolve_nonfinite_momentum_state_before_channel_choice"
    elif not exact_pass:
        recommendation = "resolve_momentum_velocity_source_or_dump_mismatch"
    else:
        recommendation = "velocity_and_momentum_pairs_are_algebraically_equivalent_under_executed_floor"
    return {
        "all_relevant_fields_finite": all_relevant_finite,
        "source_exact_velocity_momentum_equivalence": all_relevant_finite and exact_pass,
        "historical_c5_contains_electron_velocity_or_momentum": False,
        "historical_c5_is_complete_evolved_state": False,
        "recommendation": recommendation,
        "automatic_channel_change_authorized": False,
        "potential_vorticity_gate_completed": False,
    }
