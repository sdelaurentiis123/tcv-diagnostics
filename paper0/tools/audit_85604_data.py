#!/usr/bin/env python3
"""Read-only audit of the legacy 85604 raw and Well files.

The script deliberately refuses paths that name shot 85606 or a ``test`` split.
It reads metadata and coordinate vectors, never full plasma-field arrays, and
refuses to overwrite an existing result file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_RUN_ID = "85604"


def _path_is_allowed(path: Path) -> bool:
    candidates = (path, path.expanduser().resolve(strict=False))
    for candidate in candidates:
        lowered = str(candidate).lower()
        if "85606" in lowered:
            return False
        if any(part.lower() == "test" for part in candidate.parts):
            return False
    return True


def require_allowed_input(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve(strict=True)
    if not _path_is_allowed(path):
        raise ValueError(f"refusing sequestered input path: {path}")
    if not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def _scalar(handle: h5py.File, name: str) -> float:
    value = np.asarray(handle[name][()])
    if value.size != 1:
        raise ValueError(f"expected scalar {name}, got shape {value.shape}")
    return float(value.reshape(-1)[0])


def _axis_summary(dataset: h5py.Dataset) -> dict[str, Any]:
    values = np.asarray(dataset[...], dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"empty coordinate dataset: {dataset.name}")
    differences = np.diff(values)
    uniform = bool(
        differences.size == 0
        or np.allclose(differences, differences[0], rtol=1e-10, atol=1e-12)
    )
    return {
        "count": int(values.size),
        "first": float(values[0]),
        "last": float(values[-1]),
        "uniform": uniform,
        "step": float(differences[0]) if differences.size and uniform else None,
    }


def audit_raw_bout(path: Path, expected_zperiod: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = (
            "zperiod",
            "ZMIN",
            "ZMAX",
            "Omega_ci",
            "Bnorm",
            "Nnorm",
            "Tnorm",
            "rho_s0",
            "Cs0",
            "t_array",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"raw BOUT file is missing variables: {missing}")

        zperiod = int(round(_scalar(handle, "zperiod")))
        zmin = _scalar(handle, "ZMIN")
        zmax = _scalar(handle, "ZMAX")
        omega_ci = _scalar(handle, "Omega_ci")
        time = _axis_summary(handle["t_array"])
        normalization_scalars = {
            name: _scalar(handle, name)
            for name in ("Bnorm", "Nnorm", "Tnorm", "rho_s0", "Cs0")
        }

        candidate_field = None
        for name in ("Ne", "Te", "Ti", "phi", "Vi", "Vort"):
            if name in handle and handle[name].ndim == 4:
                candidate_field = name
                break
        raw_shape = (
            [int(size) for size in handle[candidate_field].shape]
            if candidate_field is not None
            else None
        )

    if zperiod != expected_zperiod:
        raise ValueError(f"zperiod={zperiod}, expected {expected_zperiod}")
    expected_fraction = 1.0 / expected_zperiod
    if not math.isclose(zmax - zmin, expected_fraction, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"stored z extent {zmax - zmin} is not 1/zperiod={expected_fraction}"
        )
    if time["step"] is None:
        raise ValueError("raw time axis is not uniformly spaced")

    cadence_seconds = float(time["step"] / omega_ci)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "representative_field": candidate_field,
        "representative_field_shape": raw_shape,
        "zperiod": zperiod,
        "zmin": zmin,
        "zmax": zmax,
        "stored_torus_fraction": expected_fraction,
        "mode_mapping": f"n = {zperiod}k",
        "omega_ci_per_second": omega_ci,
        "time": time,
        "frame_cadence_seconds": cadence_seconds,
        "frame_cadence_microseconds": cadence_seconds * 1e6,
        "normalization_scalars": normalization_scalars,
    }


def _well_fields(handle: h5py.File) -> list[str]:
    if "t0_fields" not in handle:
        raise ValueError("Well file has no t0_fields group")
    raw_names = handle["t0_fields"].attrs.get("field_names")
    if raw_names is None:
        return sorted(handle["t0_fields"].keys())
    return [str(_jsonable(name)) for name in np.asarray(raw_names).reshape(-1)]


def audit_well(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        fields = _well_fields(handle)
        if not fields:
            raise ValueError(f"no fields declared in {path}")
        field_info = {}
        for field in fields:
            dataset = handle[f"t0_fields/{field}"]
            attrs = {key: _jsonable(value) for key, value in dataset.attrs.items()}
            field_info[field] = {
                "shape": [int(size) for size in dataset.shape],
                "dtype": str(dataset.dtype),
                "attrs": attrs,
                "units_declared": any("unit" in key.lower() for key in attrs),
            }
        shapes = {tuple(item["shape"]) for item in field_info.values()}
        if len(shapes) != 1:
            raise ValueError(f"field shapes disagree in {path}: {sorted(shapes)}")
        axes = {
            name: _axis_summary(handle[f"dimensions/{name}"])
            for name in ("time", "x", "y", "z")
        }
        boundary_conditions = {
            name: {
                key: _jsonable(value) for key, value in group.attrs.items()
            }
            for name, group in handle["boundary_conditions"].items()
        }
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "fields": fields,
        "field_info": field_info,
        "axes": axes,
        "boundary_conditions": boundary_conditions,
        "all_field_units_declared": all(
            item["units_declared"] for item in field_info.values()
        ),
    }


def compare_time_regions(
    raw: dict[str, Any], train: dict[str, Any], valid: dict[str, Any]
) -> dict[str, Any]:
    train_time = train["axes"]["time"]
    valid_time = valid["axes"]["time"]
    if train_time["step"] is None or valid_time["step"] is None:
        raise ValueError("Well time axes must be uniform")
    if not math.isclose(
        train_time["step"], valid_time["step"], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("train and valid frame steps disagree")
    step = float(train_time["step"])
    gap_steps = (valid_time["first"] - train_time["last"]) / step
    guard_frames = int(round(gap_steps)) - 1
    if guard_frames < 0 or not math.isclose(
        gap_steps, round(gap_steps), rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(f"unexpected train/valid time boundary: gap_steps={gap_steps}")
    combined_frames = train_time["count"] + valid_time["count"]
    return {
        "train_last": train_time["last"],
        "valid_first": valid_time["first"],
        "normalized_frame_step": step,
        "gap_steps": float(gap_steps),
        "guard_frames": guard_frames,
        "combined_frames": combined_frames,
        "matches_raw_frame_count": combined_frames == raw["time"]["count"],
        "matches_raw_endpoints": bool(
            math.isclose(train_time["first"], raw["time"]["first"])
            and math.isclose(valid_time["last"], raw["time"]["last"])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bout", required=True)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--valid-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default=EXPECTED_RUN_ID)
    parser.add_argument("--expected-zperiod", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_id != EXPECTED_RUN_ID:
        raise ValueError(f"this locked audit accepts only run {EXPECTED_RUN_ID}")
    if args.expected_zperiod != 5:
        raise ValueError("Paper 0 protocol locks expected zperiod to 5")

    raw_path = require_allowed_input(args.raw_bout)
    train_path = require_allowed_input(args.train_h5)
    valid_path = require_allowed_input(args.valid_h5)
    output = Path(args.output).expanduser().resolve(strict=False)
    if not _path_is_allowed(output):
        raise ValueError(f"refusing output path with sequestered naming: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")

    raw = audit_raw_bout(raw_path, args.expected_zperiod)
    train = audit_well(train_path)
    valid = audit_well(valid_path)
    if train["fields"] != valid["fields"]:
        raise ValueError("train and valid field order disagree")
    if train["fields"] != ["Ne", "Te", "Ti", "phi", "Vi"]:
        raise ValueError(f"unexpected C5 field order: {train['fields']}")
    boundary = compare_time_regions(raw, train, valid)

    result = {
        "schema_version": "0.1.0",
        "run_id": args.run_id,
        "blind_test_accessed": False,
        "raw": raw,
        "legacy_train": train,
        "legacy_valid": valid,
        "legacy_boundary": boundary,
        "protocol_assertions": {
            "zperiod_is_5": raw["zperiod"] == 5,
            "mode_mapping": "n = 5k",
            "legacy_split_has_any_guard": bool(boundary["guard_frames"] > 0),
            "field_units_available": bool(
                train["all_field_units_declared"]
                and valid["all_field_units_declared"]
            ),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote read-only 85604 audit to {output}")


if __name__ == "__main__":
    main()
