#!/usr/bin/env python3
"""Stream the prospectively frozen all-frame 85604 pressure-closure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


RANK_PATTERN = re.compile(r"BOUT\.dmp\.(\d+)\.nc")
FIELD_NAMES = ("Ne", "Ni", "Te", "Ti", "Pe", "Pi")
RELATIONS = {
    "Ni_equals_Ne": ("Ni", ("Ne",)),
    "Pe_equals_Ne_times_Te": ("Pe", ("Ne", "Te")),
    "Pi_equals_Ni_times_Ti": ("Pi", ("Ni", "Ti")),
    "Pi_equals_Ne_times_Ti": ("Pi", ("Ne", "Ti")),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_number(path: Path) -> int:
    match = RANK_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a BOUT rank file: {path}")
    return int(match.group(1))


def scalar_value(dataset: Any, name: str) -> Any:
    if name not in dataset.variables:
        raise ValueError(f"missing scalar metadata variable {name}")
    value = dataset.variables[name][...]
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"metadata variable {name} is not scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return item.item() if hasattr(item, "item") else item


def variable_metadata(variable: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("cell_location", "conversion", "source", "species", "units"):
        if hasattr(variable, name):
            value = getattr(variable, name)
            if isinstance(value, np.generic):
                value = value.item()
            metadata[name] = value
    return metadata


def strict_json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary file {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def point_record(value: float, location: tuple[int, int, int, int]) -> dict[str, Any]:
    return {"value": float(value), "location_txyz": [int(item) for item in location]}


def selected_location(
    flat_index: int,
    selected_shape: tuple[int, int, int, int],
    *,
    x0: int,
    global_y: np.ndarray,
) -> tuple[int, int, int, int]:
    frame, local_x, selected_y, z_index = np.unravel_index(
        flat_index, selected_shape
    )
    return (
        int(frame),
        int(x0 + local_x),
        int(global_y[selected_y]),
        int(z_index),
    )


def scope_y_indices(global_y: np.ndarray, scope_name: str) -> np.ndarray:
    if scope_name == "full_physical_domain":
        mask = np.ones(global_y.shape, dtype=bool)
    elif scope_name == "guard_independent_transport_interior":
        mask = (global_y >= 1) & (global_y <= 30)
    elif scope_name == "target_dependent_rows":
        mask = (global_y == 0) | (global_y == 31)
    else:  # pragma: no cover - construction guard
        raise KeyError(scope_name)
    return np.flatnonzero(mask)


class ValueScopeAccumulator:
    def __init__(self, *, keep_most_negative: int = 20) -> None:
        self.total_count = 0
        self.nonfinite_count = 0
        self.negative_count = 0
        self.zero_count = 0
        self.minimum: dict[str, Any] | None = None
        self.maximum: dict[str, Any] | None = None
        self.keep_most_negative = keep_most_negative
        self.most_negative: list[dict[str, Any]] = []

    def update(
        self,
        values: np.ndarray,
        *,
        x0: int,
        global_y: np.ndarray,
    ) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.total_count += int(values.size)
        finite = np.isfinite(values)
        self.nonfinite_count += int(values.size - np.count_nonzero(finite))
        negative = finite & (values < 0.0)
        self.negative_count += int(np.count_nonzero(negative))
        self.zero_count += int(np.count_nonzero(finite & (values == 0.0)))

        if np.any(finite):
            minimum_index = int(np.argmin(np.where(finite, values, np.inf)))
            maximum_index = int(np.argmax(np.where(finite, values, -np.inf)))
            minimum_value = float(values.flat[minimum_index])
            maximum_value = float(values.flat[maximum_index])
            minimum_location = selected_location(
                minimum_index, values.shape, x0=x0, global_y=global_y
            )
            maximum_location = selected_location(
                maximum_index, values.shape, x0=x0, global_y=global_y
            )
            if self.minimum is None or minimum_value < self.minimum["value"]:
                self.minimum = point_record(minimum_value, minimum_location)
            if self.maximum is None or maximum_value > self.maximum["value"]:
                self.maximum = point_record(maximum_value, maximum_location)

        negative_indices = np.flatnonzero(negative)
        if negative_indices.size:
            negative_values = values.ravel()[negative_indices]
            retained = min(self.keep_most_negative, int(negative_indices.size))
            if negative_indices.size > retained:
                selection = np.argpartition(negative_values, retained - 1)[:retained]
                negative_indices = negative_indices[selection]
            for flat_index in negative_indices:
                location = selected_location(
                    int(flat_index), values.shape, x0=x0, global_y=global_y
                )
                self.most_negative.append(
                    point_record(float(values.flat[flat_index]), location)
                )
            self.most_negative.sort(key=lambda item: (item["value"], item["location_txyz"]))
            del self.most_negative[self.keep_most_negative :]

    def result(self) -> dict[str, Any]:
        denominator = self.total_count
        return {
            "total_count": denominator,
            "nonfinite_count": self.nonfinite_count,
            "nonfinite_fraction": self.nonfinite_count / denominator,
            "negative_count": self.negative_count,
            "negative_fraction": self.negative_count / denominator,
            "zero_count": self.zero_count,
            "zero_fraction": self.zero_count / denominator,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "most_negative_points": self.most_negative,
        }


class ValueAccumulator:
    SCOPE_NAMES = (
        "full_physical_domain",
        "guard_independent_transport_interior",
        "target_dependent_rows",
    )

    def __init__(
        self,
        *,
        frame_count: int,
        nx: int,
        ny: int,
        temporal_blocks: list[tuple[int, int]],
    ) -> None:
        self.scopes = {name: ValueScopeAccumulator() for name in self.SCOPE_NAMES}
        self.negative_by_frame = np.zeros(frame_count, dtype=np.int64)
        self.negative_by_x = np.zeros(nx, dtype=np.int64)
        self.negative_by_y = np.zeros(ny, dtype=np.int64)
        self.temporal_blocks = temporal_blocks

    def update(self, values: np.ndarray, *, x0: int, y0: int) -> None:
        global_y_all = y0 + np.arange(values.shape[2])
        for name, accumulator in self.scopes.items():
            local_y = scope_y_indices(global_y_all, name)
            if local_y.size == 0:
                continue
            accumulator.update(
                values[:, :, local_y, :],
                x0=x0,
                global_y=global_y_all[local_y],
            )

        finite_negative = np.isfinite(values) & (values < 0.0)
        self.negative_by_frame += np.sum(finite_negative, axis=(1, 2, 3), dtype=np.int64)
        per_x = np.sum(finite_negative, axis=(0, 2, 3), dtype=np.int64)
        self.negative_by_x[x0 : x0 + values.shape[1]] += per_x
        per_y = np.sum(finite_negative, axis=(0, 1, 3), dtype=np.int64)
        self.negative_by_y[y0 : y0 + values.shape[2]] += per_y

    def result(self) -> dict[str, Any]:
        by_frame = self.negative_by_frame.tolist()
        return {
            "scopes": {name: value.result() for name, value in self.scopes.items()},
            "negative_count_by_frame": by_frame,
            "negative_count_by_x": self.negative_by_x.tolist(),
            "negative_count_by_y": self.negative_by_y.tolist(),
            "negative_count_by_temporal_block": [
                int(np.sum(self.negative_by_frame[first : last + 1]))
                for first, last in self.temporal_blocks
            ],
        }


class ClosureScopeAccumulator:
    def __init__(self, *, frame_count: int, atol: float, rtol: float) -> None:
        self.total_count = 0
        self.nonfinite_count = 0
        self.point_discrepancy_count = 0
        self.negative_reference_discrepancy_count = 0
        self.nonnegative_reference_discrepancy_count = 0
        self.frame_max_abs_error = np.zeros(frame_count, dtype=np.float64)
        self.frame_max_abs_reference = np.zeros(frame_count, dtype=np.float64)
        self.frame_nonfinite_count = np.zeros(frame_count, dtype=np.int64)
        self.frame_point_discrepancy_count = np.zeros(frame_count, dtype=np.int64)
        self.maximum_error: dict[str, Any] | None = None
        self.atol = atol
        self.rtol = rtol

    def update(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        *,
        x0: int,
        global_y: np.ndarray,
    ) -> np.ndarray:
        self.total_count += int(reference.size)
        finite = np.isfinite(reference) & np.isfinite(candidate)
        nonfinite = ~finite
        with np.errstate(invalid="ignore", over="ignore"):
            absolute_error = np.abs(reference - candidate)
        finite &= np.isfinite(absolute_error)
        nonfinite = ~finite
        self.nonfinite_count += int(np.count_nonzero(nonfinite))
        self.frame_nonfinite_count += np.sum(nonfinite, axis=(1, 2, 3), dtype=np.int64)

        safe_error = np.where(finite, absolute_error, 0.0)
        safe_reference = np.where(np.isfinite(reference), np.abs(reference), 0.0)
        self.frame_max_abs_error = np.maximum(
            self.frame_max_abs_error, np.max(safe_error, axis=(1, 2, 3))
        )
        self.frame_max_abs_reference = np.maximum(
            self.frame_max_abs_reference,
            np.max(safe_reference, axis=(1, 2, 3)),
        )

        threshold = self.atol + self.rtol * np.abs(reference)
        discrepancy = finite & (absolute_error > threshold)
        discrepancy_count = int(np.count_nonzero(discrepancy))
        self.point_discrepancy_count += discrepancy_count
        self.frame_point_discrepancy_count += np.sum(
            discrepancy, axis=(1, 2, 3), dtype=np.int64
        )
        self.negative_reference_discrepancy_count += int(
            np.count_nonzero(discrepancy & (reference < 0.0))
        )
        self.nonnegative_reference_discrepancy_count += int(
            np.count_nonzero(discrepancy & (reference >= 0.0))
        )

        if np.any(finite):
            maximum_index = int(np.argmax(safe_error))
            maximum_value = float(safe_error.flat[maximum_index])
            location = selected_location(
                maximum_index, reference.shape, x0=x0, global_y=global_y
            )
            if (
                self.maximum_error is None
                or maximum_value > self.maximum_error["absolute_error"]
            ):
                self.maximum_error = {
                    "absolute_error": maximum_value,
                    "location_txyz": [int(item) for item in location],
                    "reference": float(reference.flat[maximum_index]),
                    "candidate": float(candidate.flat[maximum_index]),
                }
        return discrepancy

    def result(self) -> dict[str, Any]:
        tolerance = self.atol + self.rtol * self.frame_max_abs_reference
        passed = (self.frame_nonfinite_count == 0) & (
            self.frame_max_abs_error <= tolerance
        )
        failed_indices = np.flatnonzero(~passed).astype(int).tolist()
        return {
            "total_count": self.total_count,
            "nonfinite_count": self.nonfinite_count,
            "point_discrepancy_count": self.point_discrepancy_count,
            "negative_reference_discrepancy_count": self.negative_reference_discrepancy_count,
            "nonnegative_reference_discrepancy_count": self.nonnegative_reference_discrepancy_count,
            "maximum_error": self.maximum_error,
            "frame_pass_count": int(np.count_nonzero(passed)),
            "frame_fail_count": int(np.count_nonzero(~passed)),
            "failed_frame_indices": failed_indices,
            "frame_max_abs_error": self.frame_max_abs_error.tolist(),
            "frame_max_abs_reference": self.frame_max_abs_reference.tolist(),
            "frame_tolerance": tolerance.tolist(),
            "frame_passed": passed.tolist(),
            "frame_nonfinite_count": self.frame_nonfinite_count.tolist(),
            "frame_point_discrepancy_count": self.frame_point_discrepancy_count.tolist(),
        }


class ClosureAccumulator:
    SCOPE_NAMES = ValueAccumulator.SCOPE_NAMES

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
            name: ClosureScopeAccumulator(
                frame_count=frame_count, atol=atol, rtol=rtol
            )
            for name in self.SCOPE_NAMES
        }
        self.discrepancy_by_frame = np.zeros(frame_count, dtype=np.int64)
        self.discrepancy_by_x = np.zeros(nx, dtype=np.int64)
        self.discrepancy_by_y = np.zeros(ny, dtype=np.int64)
        self.temporal_blocks = temporal_blocks

    def update(
        self,
        reference: np.ndarray,
        candidate: np.ndarray,
        *,
        x0: int,
        y0: int,
    ) -> None:
        global_y_all = y0 + np.arange(reference.shape[2])
        full_discrepancy: np.ndarray | None = None
        for name, accumulator in self.scopes.items():
            local_y = scope_y_indices(global_y_all, name)
            if local_y.size == 0:
                continue
            discrepancy = accumulator.update(
                reference[:, :, local_y, :],
                candidate[:, :, local_y, :],
                x0=x0,
                global_y=global_y_all[local_y],
            )
            if name == "full_physical_domain":
                full_discrepancy = discrepancy
        assert full_discrepancy is not None
        self.discrepancy_by_frame += np.sum(
            full_discrepancy, axis=(1, 2, 3), dtype=np.int64
        )
        per_x = np.sum(full_discrepancy, axis=(0, 2, 3), dtype=np.int64)
        self.discrepancy_by_x[x0 : x0 + reference.shape[1]] += per_x
        per_y = np.sum(full_discrepancy, axis=(0, 1, 3), dtype=np.int64)
        self.discrepancy_by_y[y0 : y0 + reference.shape[2]] += per_y

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


def relation_candidate(
    relation_name: str, fields: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    reference_name, factors = RELATIONS[relation_name]
    reference = fields[reference_name]
    if len(factors) == 1:
        candidate = fields[factors[0]]
    else:
        with np.errstate(invalid="ignore", over="ignore"):
            candidate = fields[factors[0]] * fields[factors[1]]
    return reference, candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    return parser.parse_args()


def _verify_path_is_development_run(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    if "85606" in lowered_parts:
        raise ValueError("held-out run path is prohibited")
    if "85604" not in lowered_parts:
        raise ValueError("raw root must identify development run 85604")


def initialize_stream_digests(
    fields: Iterable[str], shape: list[int]
) -> dict[str, Any]:
    result = {}
    for field in fields:
        digest = hashlib.sha256()
        header = json.dumps(
            {
                "field": field,
                "dtype": "<f8",
                "global_shape": shape,
                "stream_order": "ascending rank number with explicit rank,PE_XIND,PE_YIND headers",
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
    rank_header = json.dumps(
        {"rank": rank, "PE_XIND": pe_x, "PE_YIND": pe_y},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(rank_header)
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the pressure-closure audit") from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["development_run"] != "85604":
        raise ValueError("pressure-closure audit accepts development run 85604 only")
    if manifest["held_out_85606_access_allowed"]:
        raise ValueError("manifest must prohibit held-out access")
    _verify_path_is_development_run(args.raw_root.resolve())
    if args.raw_root.resolve() != Path(manifest["raw_archive"]["root"]).resolve():
        raise ValueError("raw root differs from the frozen manifest")

    archive = manifest["raw_archive"]
    locked_files = {
        "BOUT.inp": archive["bout_input_sha256"],
        "BOUT.settings": archive["bout_settings_sha256"],
        "tcv_85604_adjusted.nc": archive["geometry_sha256"],
    }
    for name, expected_digest in locked_files.items():
        if sha256_file(args.raw_root / name) != expected_digest:
            raise ValueError(f"hash mismatch for raw control {name}")

    paths = sorted(args.raw_root.glob("BOUT.dmp.*.nc"), key=rank_number)
    expected_rank_count = int(archive["expected_rank_file_count"])
    if len(paths) != expected_rank_count:
        raise ValueError(f"expected {expected_rank_count} rank files, found {len(paths)}")
    ranks = [rank_number(path) for path in paths]
    if ranks != list(range(expected_rank_count)):
        raise ValueError("rank filenames must cover exactly 0 through 255")

    decomposition = {
        key: int(value) for key, value in archive["mpi_decomposition"].items()
    }
    frame_scope = manifest["frame_scope"]
    frame_count = int(frame_scope["frame_count"])
    first_time = float(frame_scope["expected_first_normalized_time"])
    cadence = float(frame_scope["expected_normalized_cadence"])
    expected_times = first_time + cadence * np.arange(frame_count, dtype=np.float64)
    if expected_times[-1] != float(frame_scope["expected_last_normalized_time"]):
        raise ValueError("frozen time endpoints and cadence are inconsistent")

    nx = decomposition["NXPE"] * decomposition["MXSUB"]
    ny = decomposition["NYPE"] * decomposition["MYSUB"]
    native_z = int(archive["native_z_samples"])
    expected_shape = [frame_count, nx, ny, native_z]
    if expected_shape != manifest["canonical_cells"]["shape_per_field"]:
        raise ValueError("manifest canonical shape is inconsistent")
    if int(np.prod(expected_shape, dtype=np.int64)) != int(
        manifest["canonical_cells"]["total_points_per_field"]
    ):
        raise ValueError("manifest point count is inconsistent")

    temporal_blocks = [
        (int(first), int(last))
        for first, last in manifest["temporal_blocks"]["inclusive_index_ranges"]
    ]
    closure_definition = manifest["closure_statistics"]
    atol = float(closure_definition["atol"])
    rtol = float(closure_definition["rtol"])
    value_accumulators = {
        field: ValueAccumulator(
            frame_count=frame_count,
            nx=nx,
            ny=ny,
            temporal_blocks=temporal_blocks,
        )
        for field in FIELD_NAMES
    }
    closure_accumulators = {
        relation: ClosureAccumulator(
            frame_count=frame_count,
            nx=nx,
            ny=ny,
            temporal_blocks=temporal_blocks,
            atol=atol,
            rtol=rtol,
        )
        for relation in RELATIONS
    }
    stream_digests = initialize_stream_digests(FIELD_NAMES, expected_shape)
    coverage = np.zeros((decomposition["NXPE"], decomposition["NYPE"]), dtype=np.int64)
    reference_metadata: dict[str, dict[str, Any]] | None = None

    for path in paths:
        rank = rank_number(path)
        with netCDF4.Dataset(path, "r") as dataset:
            dimensions = {
                name: int(len(dimension))
                for name, dimension in dataset.dimensions.items()
            }
            if dimensions != archive["rank_dimensions"]:
                raise ValueError(f"unexpected dimensions in {path.name}: {dimensions}")
            for name, expected in decomposition.items():
                if int(scalar_value(dataset, name)) != expected:
                    raise ValueError(f"{name} mismatch in {path.name}")
            if int(scalar_value(dataset, "MZ")) != native_z:
                raise ValueError(f"MZ mismatch in {path.name}")
            if int(scalar_value(dataset, "zperiod")) != int(archive["zperiod"]):
                raise ValueError(f"zperiod mismatch in {path.name}")
            if scalar_value(dataset, "HERMES_REVISION") != archive["hermes_revision"]:
                raise ValueError(f"Hermes revision mismatch in {path.name}")
            if scalar_value(dataset, "HERMES_SLOPE_LIMITER") != archive["slope_limiter"]:
                raise ValueError(f"slope limiter mismatch in {path.name}")

            pe_x = int(scalar_value(dataset, "PE_XIND"))
            pe_y = int(scalar_value(dataset, "PE_YIND"))
            if not (0 <= pe_x < coverage.shape[0] and 0 <= pe_y < coverage.shape[1]):
                raise ValueError(f"processor coordinate outside decomposition in {path.name}")
            coverage[pe_x, pe_y] += 1

            times = np.asarray(dataset.variables["t"][:], dtype=np.float64)
            if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"time sequence mismatch in {path.name}")

            metadata = {
                name: variable_metadata(dataset.variables[name]) for name in FIELD_NAMES
            }
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise ValueError(f"variable metadata disagree in {path.name}")

            mxg = decomposition["MXG"]
            myg = decomposition["MYG"]
            mxsub = decomposition["MXSUB"]
            mysub = decomposition["MYSUB"]
            rank_fields: dict[str, np.ndarray] = {}
            for name in FIELD_NAMES:
                variable = dataset.variables[name]
                if tuple(variable.dimensions) != ("t", "x", "y", "z"):
                    raise ValueError(f"unexpected axis order for {name} in {path.name}")
                values = np.ma.filled(
                    variable[
                        :,
                        mxg : mxg + mxsub,
                        myg : myg + mysub,
                        :,
                    ],
                    np.nan,
                ).astype(np.float64, copy=False)
                if values.shape != (frame_count, mxsub, mysub, native_z):
                    raise ValueError(f"unexpected physical shape for {name} in {path.name}")
                rank_fields[name] = values
                update_stream_digest(
                    stream_digests[name],
                    values,
                    rank=rank,
                    pe_x=pe_x,
                    pe_y=pe_y,
                )
                value_accumulators[name].update(
                    values,
                    x0=pe_x * mxsub,
                    y0=pe_y * mysub,
                )

            for relation in RELATIONS:
                reference, candidate = relation_candidate(relation, rank_fields)
                closure_accumulators[relation].update(
                    reference,
                    candidate,
                    x0=pe_x * mxsub,
                    y0=pe_y * mysub,
                )

    if not np.array_equal(coverage, np.ones_like(coverage)):
        raise ValueError("processor-coordinate coverage is incomplete or duplicated")
    assert reference_metadata is not None

    value_results = {
        field: accumulator.result()
        for field, accumulator in value_accumulators.items()
    }
    closure_results = {
        relation: accumulator.result()
        for relation, accumulator in closure_accumulators.items()
    }
    expected_total = int(manifest["canonical_cells"]["total_points_per_field"])
    expected_interior = frame_count * nx * 30 * native_z
    expected_targets = frame_count * nx * 2 * native_z
    expected_scope_counts = {
        "full_physical_domain": expected_total,
        "guard_independent_transport_interior": expected_interior,
        "target_dependent_rows": expected_targets,
    }
    for field, field_result in value_results.items():
        for scope_name, expected_count in expected_scope_counts.items():
            actual = field_result["scopes"][scope_name]["total_count"]
            if actual != expected_count:
                raise ValueError(
                    f"incomplete value accounting for {field}/{scope_name}: {actual}"
                )
    for relation, relation_result in closure_results.items():
        for scope_name, expected_count in expected_scope_counts.items():
            actual = relation_result["scopes"][scope_name]["total_count"]
            if actual != expected_count:
                raise ValueError(
                    f"incomplete closure accounting for {relation}/{scope_name}: {actual}"
                )

    all_fields_finite = all(
        value_results[field]["scopes"]["full_physical_domain"]["nonfinite_count"] == 0
        for field in FIELD_NAMES
    )
    all_full_closures_pass = all(
        closure_results[relation]["scopes"]["full_physical_domain"]["frame_fail_count"] == 0
        for relation in RELATIONS
    )
    pressure_relations = ("Pe_equals_Ne_times_Te", "Pi_equals_Ne_times_Ti")
    interior_pressure_closures_pass = all(
        closure_results[relation]["scopes"]["guard_independent_transport_interior"]["frame_fail_count"] == 0
        for relation in pressure_relations
    )
    pressure_undershoots_present = any(
        value_results[field]["scopes"]["full_physical_domain"]["negative_count"] > 0
        for field in ("Pe", "Pi")
    )
    target_only_pressure_undershoots = all(
        value_results[field]["scopes"]["guard_independent_transport_interior"]["negative_count"] == 0
        for field in ("Pe", "Pi")
    )
    pressure_discrepancies_confined_to_targets = all(
        closure_results[relation]["scopes"]["guard_independent_transport_interior"]["point_discrepancy_count"] == 0
        for relation in pressure_relations
    )
    if not all_fields_finite:
        recommendation = "resolve_nonfinite_state_before_channel_choice"
    elif not interior_pressure_closures_pass:
        recommendation = "prefer_direct_evolved_pressure_or_define_and_validate_floor_policy"
    elif all_full_closures_pass:
        recommendation = "historical_temperature_channels_close_evolved_pressure_on_85604"
    else:
        recommendation = "temperature_channels_match_guard_independent_operator_scope_but_not_full_evolved_state"

    result = {
        "schema_version": 1,
        "phase": "phase2_85604_all_frame_pressure_closure_audit",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "audit_completed": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "raw_root": str(args.raw_root),
        "raw_control_digests": locked_files,
        "rank_file_count": len(paths),
        "processor_coverage": {
            "NXPE": decomposition["NXPE"],
            "NYPE": decomposition["NYPE"],
            "unique_coordinates": int(np.count_nonzero(coverage)),
        },
        "frame_count": frame_count,
        "normalized_time": {
            "first": float(expected_times[0]),
            "last": float(expected_times[-1]),
            "cadence": cadence,
            "physical_cadence_microseconds": float(
                frame_scope["physical_cadence_microseconds"]
            ),
        },
        "native_z_samples": native_z,
        "zperiod": int(archive["zperiod"]),
        "shape_per_field": expected_shape,
        "total_points_per_field": expected_total,
        "variable_metadata": reference_metadata,
        "guard_stripped_rank_stream_sha256": {
            field: digest.hexdigest() for field, digest in stream_digests.items()
        },
        "value_statistics": value_results,
        "closure_statistics": {
            "atol": atol,
            "rtol": rtol,
            "relations": closure_results,
        },
        "scientific_findings": {
            "all_fields_finite": all_fields_finite,
            "exact_full_state_compatibility": all_fields_finite
            and all_full_closures_pass,
            "temperature_state_reproduces_guard_independent_pressure_transport": all_fields_finite
            and interior_pressure_closures_pass,
            "pressure_undershoots_present": pressure_undershoots_present,
            "target_only_pressure_undershoots": target_only_pressure_undershoots,
            "pressure_closure_discrepancies_confined_to_targets": pressure_discrepancies_confined_to_targets,
            "recommendation": recommendation,
            "automatic_channel_change_authorized": False,
        },
    }
    strict_json_write(args.output, result)
    print(json.dumps(result["scientific_findings"], indent=2, sort_keys=True))
    print(f"Wrote complete audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
