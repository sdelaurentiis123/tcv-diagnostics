#!/usr/bin/env python3
"""Inventory the newly supplied 85604 squash without touching held-out data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


EVOLVED_FIELDS = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort")
C5P_FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
SUMMARY_FIELDS = tuple(dict.fromkeys((*EVOLVED_FIELDS, "Te", "Ti", "phi", "Vi")))
SELECTED_ATTRIBUTES = (
    "bout_type",
    "cell_location",
    "conversion",
    "direction_y",
    "direction_z",
    "long_name",
    "source",
    "standard_name",
    "time_dimension",
    "units",
)
SELECTED_SETTINGS = {
    "": ("MZ", "nout", "restart", "timestep", "zperiod"),
    "Ne": ("flux", "function", "shape_factor", "source", "source_only_in_core"),
    "Pe": ("function", "heating", "source", "source_only_in_core"),
    "Pi": ("function", "source", "source_only_in_core"),
    "e": ("AA", "charge", "damp_p_nt", "low_T_diffuse_perp", "low_n_diffuse_perp", "type"),
    "hermes": ("Bnorm", "Nnorm", "Tnorm", "components", "normalise_metric"),
    "i": ("AA", "charge", "damp_p_nt", "low_T_diffuse_perp", "low_n_diffuse_perp", "type"),
    "mesh": ("file", "extrapolate_y"),
    "mesh:paralleltransform": ("type",),
    "run": ("revision", "started", "version"),
    "vorticity": (
        "diamagnetic",
        "exb_advection_simplified",
        "phi_boundary_relax",
        "phi_boundary_timescale",
        "phi_dissipation",
        "phi_sheath_dissipation",
        "poloidal_flows",
        "split_n0",
        "vort_dissipation",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squash", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--squash-sha256", required=True)
    parser.add_argument("--squash-hash-job-id", type=int, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--chunk-frames", type=int, default=16)
    return parser.parse_args()


def verify_development_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    parts = {part.lower() for part in resolved.parts}
    if "85606" in parts or "85606" in str(resolved).lower():
        raise ValueError("held-out 85606 paths are prohibited")
    if "85604" not in parts:
        raise ValueError(f"development source must identify 85604: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(name: str, values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values)
    header = json.dumps(
        {"name": name, "dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_scalar(dataset: Any, name: str) -> Any:
    if name not in dataset.variables:
        raise ValueError(f"missing scalar {name}")
    value = dataset.variables[name][...]
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} is not scalar")
    return json_value(array.reshape(-1)[0])


def filled(values: Any) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)


class StreamingSummary:
    def __init__(self) -> None:
        self.total = 0
        self.finite = 0
        self.sum = 0.0
        self.sum_squares = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64)
        self.total += int(array.size)
        selected = array[np.isfinite(array)]
        self.finite += int(selected.size)
        if not selected.size:
            return
        self.sum += float(np.sum(selected, dtype=np.float64))
        self.sum_squares += float(np.sum(selected * selected, dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(selected)))
        self.maximum = max(self.maximum, float(np.max(selected)))

    def finalize(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_count": self.total,
            "finite_count": self.finite,
            "nonfinite_count": self.total - self.finite,
        }
        if self.finite == 0:
            return {**result, "minimum": None, "maximum": None, "mean": None, "rms": None}
        return {
            **result,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.sum / self.finite,
            "rms": math.sqrt(self.sum_squares / self.finite),
        }


def parse_settings(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {"": {}}
    section = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            sections.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split("#", 1)[0].strip()
        sections.setdefault(section, {})[key.strip()] = value
    return sections


def selected_settings(sections: dict[str, dict[str, str]]) -> dict[str, dict[str, str | None]]:
    return {
        section or "root": {
            key: sections.get(section, {}).get(key) for key in keys
        }
        for section, keys in SELECTED_SETTINGS.items()
    }


def relative_step(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.linalg.norm((second - first).reshape(-1)))
    denominator = float(np.linalg.norm(first.reshape(-1)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def reset_records(
    dataset: Any,
    *,
    reset_indices: Iterable[int],
    x_slice: slice,
    y_slice: slice,
    z_slice: slice,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    frame_count = len(dataset.dimensions["t"])
    for reset in reset_indices:
        first = max(0, reset - 17)
        stop = min(frame_count, reset + 17)
        by_field: dict[str, Any] = {}
        for name in EVOLVED_FIELDS:
            values = filled(dataset.variables[name][first:stop, x_slice, y_slice, z_slice])
            steps = np.asarray(
                [relative_step(values[index - 1], values[index]) for index in range(1, len(values))],
                dtype=np.float64,
            )
            global_transition_indices = np.arange(first + 1, stop)
            reset_positions = np.flatnonzero(global_transition_indices == reset)
            if reset_positions.size != 1:
                raise ValueError("reset transition not covered exactly once")
            position = int(reset_positions[0])
            reset_value = float(steps[position])
            controls = np.delete(steps, position)
            by_field[name] = {
                "relative_l2_at_reset": reset_value,
                "nearby_transition_count": int(controls.size),
                "nearby_minimum": float(np.min(controls)),
                "nearby_median": float(np.median(controls)),
                "nearby_maximum": float(np.max(controls)),
                "reset_to_nearby_median": (
                    reset_value / float(np.median(controls))
                    if float(np.median(controls)) > 0.0
                    else None
                ),
                "within_nearby_range": bool(
                    np.min(controls) <= reset_value <= np.max(controls)
                ),
            }
        records.append({"transition_target_frame": int(reset), "fields": by_field})
    return records


def strict_json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.squash_sha256):
        raise ValueError("squash SHA-256 must contain 64 lowercase hex characters")
    if args.chunk_frames <= 0:
        raise ValueError("chunk size must be positive")
    squash = verify_development_path(args.squash)
    settings = verify_development_path(args.settings)
    grid = verify_development_path(args.grid)
    protocol = args.protocol.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency gate
        raise RuntimeError("netCDF4 is required") from error

    settings_text = settings.read_text(encoding="utf-8")
    blockers: list[str] = []
    grid_sha256 = None
    grid_error = None
    try:
        grid_sha256 = sha256_file(grid)
    except OSError as error:
        grid_error = f"{type(error).__name__}: {error}"
        blockers.append("supplied_grid_is_not_readable")

    with netCDF4.Dataset(squash, "r") as dataset:
        dimensions = {name: len(value) for name, value in dataset.dimensions.items()}
        if set(("t", "x", "y", "z")) - set(dimensions):
            raise ValueError(f"missing canonical dimensions: {dimensions}")
        frame_count = int(dimensions["t"])
        mxg = int(read_scalar(dataset, "MXG"))
        myg = int(read_scalar(dataset, "MYG"))
        zperiod = int(read_scalar(dataset, "zperiod"))
        if zperiod != 5:
            raise ValueError("zperiod differs from frozen value five")
        times = filled(dataset.variables["t_array"][:]).reshape(-1)
        iterations = np.asarray(dataset.variables["iteration"][:], dtype=np.int64).reshape(-1)
        if times.shape != (frame_count,) or iterations.shape != (frame_count,):
            raise ValueError("time or iteration shape differs from frame count")
        time_steps = np.diff(times)
        if not np.all(np.isfinite(times)) or not np.all(time_steps > 0.0):
            raise ValueError("normalized time is nonfinite or nonmonotonic")
        reset_indices = np.flatnonzero(np.diff(iterations) <= 0) + 1

        x_slice = slice(mxg, dimensions["x"] - mxg)
        y_slice = slice(myg, dimensions["y"] - myg)
        endpoint: dict[str, Any] = {}
        for name in SUMMARY_FIELDS:
            if name not in dataset.variables:
                raise ValueError(f"missing required field {name}")
            first = filled(dataset.variables[name][:, x_slice, y_slice, 0])
            last = filled(dataset.variables[name][:, x_slice, y_slice, -1])
            difference = last - first
            denominator = math.sqrt(float(np.mean(first * first)))
            endpoint[name] = {
                "bitwise_equal": bool(np.array_equal(first, last)),
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "normalized_rms_error": (
                    math.sqrt(float(np.mean(difference * difference))) / denominator
                    if denominator > 0.0
                    else None
                ),
            }
        duplicated_endpoint = all(record["bitwise_equal"] for record in endpoint.values())
        z_slice = slice(0, dimensions["z"] - 1 if duplicated_endpoint else dimensions["z"])

        fields: dict[str, Any] = {}
        boundary_summary = StreamingSummary()
        for name in SUMMARY_FIELDS:
            variable = dataset.variables[name]
            summary = StreamingSummary()
            for start in range(0, frame_count, args.chunk_frames):
                stop = min(frame_count, start + args.chunk_frames)
                values = filled(variable[start:stop, x_slice, y_slice, z_slice])
                summary.update(values)
                if name == "phi":
                    inner = 0.5 * (
                        filled(variable[start:stop, 1, y_slice, z_slice])
                        + filled(variable[start:stop, 2, y_slice, z_slice])
                    )
                    outer = 0.5 * (
                        filled(variable[start:stop, -2, y_slice, z_slice])
                        + filled(variable[start:stop, -3, y_slice, z_slice])
                    )
                    bphi = np.stack(
                        [np.mean(inner, axis=-1), np.mean(outer, axis=-1)], axis=1
                    )
                    boundary_summary.update(bphi)
            fields[name] = {
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "attributes": {
                    attribute: json_value(variable.getncattr(attribute))
                    for attribute in SELECTED_ATTRIBUTES
                    if attribute in variable.ncattrs()
                },
                "physical_unique_z_summary": summary.finalize(),
            }

        embedded_geometry: dict[str, Any] = {}
        for name, variable in dataset.variables.items():
            if "t" in variable.dimensions or variable.ndim == 0:
                continue
            if not set(variable.dimensions).issubset({"x", "y", "z"}):
                continue
            values = np.asarray(variable[:])
            if values.dtype.kind not in "biufc":
                continue
            embedded_geometry[name] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "array_sha256": array_sha256(name, values),
            }

        resets = reset_records(
            dataset,
            reset_indices=reset_indices,
            x_slice=x_slice,
            y_slice=y_slice,
            z_slice=z_slice,
        )
        continuity_pass = all(
            field["within_nearby_range"]
            for reset in resets
            for field in reset["fields"].values()
        )
        if not continuity_pass:
            blockers.append("internal_restart_field_jump_outside_nearby_range")

        omega_ci = float(read_scalar(dataset, "Omega_ci"))
        result = {
            "schema_version": 1,
            "scope": "post_ecrd_nersc_85604_minimal_inventory",
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "development_run": "85604",
            "held_out_85606_read": False,
            "protocol": str(protocol),
            "protocol_sha256": sha256_file(protocol),
            "sources": {
                "squash": {
                    "path": str(squash),
                    "size_bytes": squash.stat().st_size,
                    "mtime_ns": squash.stat().st_mtime_ns,
                    "sha256": args.squash_sha256,
                    "sha256_job_id": args.squash_hash_job_id,
                },
                "settings": {
                    "path": str(settings),
                    "size_bytes": settings.stat().st_size,
                    "mtime_ns": settings.stat().st_mtime_ns,
                    "sha256": sha256_file(settings),
                    "selected_values": selected_settings(parse_settings(settings_text)),
                },
                "grid": {
                    "path": str(grid),
                    "size_bytes": grid.stat().st_size,
                    "mtime_ns": grid.stat().st_mtime_ns,
                    "sha256": grid_sha256,
                    "read_error": grid_error,
                },
            },
            "simulation": {
                "run_id": read_scalar(dataset, "run_id"),
                "restart_parent": read_scalar(dataset, "run_restart_from"),
                "hermes_revision": read_scalar(dataset, "HERMES_REVISION"),
                "bout_version": float(read_scalar(dataset, "BOUT_VERSION")),
                "zperiod": zperiod,
                "dimensions": dimensions,
                "guard_cells": {"x_each_side": mxg, "y_each_side": myg},
                "physical_unique_shape": [
                    dimensions["x"] - 2 * mxg,
                    dimensions["y"] - 2 * myg,
                    z_slice.stop,
                ],
                "toroidal_endpoint_duplicated_bitwise": duplicated_endpoint,
                "toroidal_endpoint_checks": endpoint,
                "embedded_geometry": embedded_geometry,
            },
            "time": {
                "frame_count": frame_count,
                "first_normalized": float(times[0]),
                "last_normalized": float(times[-1]),
                "cadence_normalized_minimum": float(np.min(time_steps)),
                "cadence_normalized_median": float(np.median(time_steps)),
                "cadence_normalized_maximum": float(np.max(time_steps)),
                "cadence_physical_microseconds": float(np.median(time_steps) / omega_ci * 1e6),
                "duration_physical_milliseconds": float((times[-1] - times[0]) / omega_ci * 1e3),
                "iteration_reset_target_frames": reset_indices.astype(int).tolist(),
            },
            "state": {
                "evolved_fields": list(EVOLVED_FIELDS),
                "c5p_fields": list(C5P_FIELDS),
                "fields": fields,
                "Bphi_saved_midpoint_summary": boundary_summary.finalize(),
            },
            "restart_continuity": {
                "rule": "reset relative L2 must lie within 16-neighbor empirical range for every evolved field",
                "records": resets,
                "passed": continuity_pass,
            },
            "blockers": sorted(set(blockers)),
            "released_for_scientific_training": not blockers,
        }

    strict_json_write(args.output, result)
    print(json.dumps({
        "output": str(args.output),
        "frame_count": result["time"]["frame_count"],
        "physical_unique_shape": result["simulation"]["physical_unique_shape"],
        "iteration_resets": result["time"]["iteration_reset_target_frames"],
        "restart_continuity_passed": result["restart_continuity"]["passed"],
        "blockers": result["blockers"],
        "released_for_scientific_training": result["released_for_scientific_training"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
