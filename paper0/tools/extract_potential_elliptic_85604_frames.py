#!/usr/bin/env python3
"""Extract the frozen 85604 inputs for the paired potential elliptic oracle.

The extractor assembles five guard-stripped volume fields on the native
64-by-32-by-81 mesh and preserves the saved radial-potential midpoint state.
It also rebuilds the complete boundary streams and requires their SHA-256
digests to match the previously completed all-frame audit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_85604_phi_boundary_state as prior_audit  # noqa: E402
from extract_native_85604_frames import (  # noqa: E402
    nearest_half_up_indices,
    place_rank_block,
    rank_number,
    scalar_value,
    sha256_array,
    sha256_file,
    variable_metadata,
)
from tcv_diagnostics import phi_boundary  # noqa: E402


VOLUME_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")
BOUNDARY_ARRAYS = (
    "saved_midpoint",
    "instantaneous_target",
    "midpoint_departure",
)


def verify_development_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "85606" in lowered:
        raise ValueError("held-out run path is prohibited")
    if "85604" not in lowered:
        raise ValueError("raw root must identify development run 85604")


def expected_time_sequence(selection: dict[str, Any]) -> np.ndarray:
    """Recover the uniquely frozen affine time sequence from selected times."""

    frame_indices = np.asarray(selection["indices"], dtype=np.int64)
    selected_times = np.asarray(
        selection["expected_normalized_times"], dtype=np.float64
    )
    if frame_indices.shape != selected_times.shape or frame_indices.size < 2:
        raise ValueError("selected frame/time lock is malformed")
    differences = np.diff(frame_indices)
    if np.any(differences <= 0):
        raise ValueError("selected frame indices must increase")
    cadences = np.diff(selected_times) / differences
    if not np.allclose(cadences, cadences[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("selected normalized times do not define one cadence")
    first_time = selected_times[0] - frame_indices[0] * cadences[0]
    total_frames = int(selection["total_frames"])
    expected = first_time + cadences[0] * np.arange(
        total_frames, dtype=np.float64
    )
    if not np.allclose(
        expected[frame_indices], selected_times, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("reconstructed time sequence misses selected locks")
    return expected


def discrepancy_count_by_frame(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> list[int]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim < 1:
        raise ValueError("boundary comparison shapes differ")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    difference = candidate - reference
    discrepant = (~finite) | (
        np.abs(difference) > atol + rtol * np.abs(reference)
    )
    axes = tuple(range(1, discrepant.ndim))
    return np.sum(discrepant, axis=axes, dtype=np.int64).astype(int).tolist()


def derive_boundary_products(
    streams: dict[str, dict[str, np.ndarray]],
    *,
    frame_indices: list[int],
    atol: float,
    rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Derive compact boundary state and exact structural checks."""

    expected_sides = set(phi_boundary.SIDES)
    if set(streams) != expected_sides:
        raise ValueError("both radial boundary sides are required")
    selected = {
        name: np.full(
            (len(frame_indices), len(phi_boundary.SIDES), 32),
            np.nan,
            dtype=np.float64,
        )
        for name in BOUNDARY_ARRAYS
    }
    checks: dict[str, Any] = {}
    for side_index, side in enumerate(phi_boundary.SIDES):
        planes = streams[side]
        if set(planes) != set(phi_boundary.PLANE_NAMES):
            raise ValueError(f"boundary planes are incomplete for {side}")
        shapes = {np.asarray(values).shape for values in planes.values()}
        if len(shapes) != 1:
            raise ValueError(f"boundary plane shapes differ for {side}")
        shape = next(iter(shapes))
        if len(shape) != 3 or shape[1:] != (32, 81):
            raise ValueError("boundary streams must use [time,32,81]")

        outermost = np.asarray(planes["outermost_guard"], dtype=np.float64)
        adjacent = np.asarray(planes["adjacent_guard"], dtype=np.float64)
        interior = np.asarray(planes["adjacent_interior"], dtype=np.float64)
        midpoint = 0.5 * (adjacent + interior)
        midpoint_mean = np.mean(midpoint, axis=-1, dtype=np.float64)
        target = np.mean(interior, axis=-1, dtype=np.float64)
        departure = midpoint_mean - target
        midpoint_reference = np.broadcast_to(
            midpoint_mean[..., None], midpoint.shape
        )

        selected["saved_midpoint"][:, side_index] = midpoint_mean[frame_indices]
        selected["instantaneous_target"][:, side_index] = target[frame_indices]
        selected["midpoint_departure"][:, side_index] = departure[frame_indices]
        checks[side] = {
            "outer_guard_copy_count_by_frame": discrepancy_count_by_frame(
                adjacent, outermost, atol=atol, rtol=rtol
            ),
            "midpoint_constancy_count_by_frame": discrepancy_count_by_frame(
                midpoint_reference, midpoint, atol=atol, rtol=rtol
            ),
            "instantaneous_neumann_count_by_frame": discrepancy_count_by_frame(
                target, midpoint_mean, atol=atol, rtol=rtol
            ),
            "selected_outer_guard_copy_max_abs": float(
                np.max(np.abs((outermost - adjacent)[frame_indices]))
            ),
            "selected_midpoint_constancy_max_abs": float(
                np.max(np.abs((midpoint - midpoint_reference)[frame_indices]))
            ),
        }

    for name, values in selected.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"selected boundary array {name} is non-finite")
    if not np.array_equal(
        selected["midpoint_departure"],
        selected["saved_midpoint"] - selected["instantaneous_target"],
    ):
        raise ValueError("midpoint departure is not an exact derived difference")
    return selected, checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite a canonical oracle artifact")
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for oracle extraction") from error

    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["development_run"] != "85604":
        raise ValueError("potential extractor accepts development run 85604 only")
    if manifest["held_out_85606_access_allowed"] or manifest["training_allowed"]:
        raise ValueError("manifest violates the development-only nontraining scope")

    raw_root = args.raw_root.resolve(strict=True)
    verify_development_path(raw_root)
    if raw_root != Path(manifest["raw_archive"]["root"]).resolve():
        raise ValueError("raw root differs from the frozen manifest")

    repository_root = manifest_path.parents[2]
    locked_records: dict[str, dict[str, Any]] = {}
    for name, lock in manifest["provenance_locks"].items():
        locked_path = repository_root / lock["path"]
        actual = sha256_file(locked_path)
        if actual != lock["sha256"]:
            raise ValueError(f"provenance SHA-256 mismatch for {name}")
        locked_records[name] = {
            "path": str(locked_path),
            "sha256": actual,
        }
    prior_result_path = Path(
        locked_records["all_frame_phi_boundary_audit"]["path"]
    )
    prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
    if (
        prior_result["development_run"] != "85604"
        or prior_result["held_out_85606_read"]
        or not prior_result["audit_completed"]
    ):
        raise ValueError("locked boundary audit is not a completed 85604 result")

    selection = manifest["frame_selection"]
    frame_indices = nearest_half_up_indices(
        int(selection["total_frames"]),
        [float(value) for value in selection["fractions"]],
    )
    if frame_indices != selection["indices"]:
        raise ValueError("manifest frame indices violate the frozen rule")
    expected_times = expected_time_sequence(selection)

    archive = manifest["raw_archive"]
    control_digests = {
        "BOUT.inp": archive["bout_input_sha256"],
        "BOUT.settings": archive["bout_settings_sha256"],
        "tcv_85604_adjusted.nc": archive["geometry_sha256"],
    }
    for name, expected in control_digests.items():
        if sha256_file(raw_root / name) != expected:
            raise ValueError(f"hash mismatch for raw control {name}")

    paths = sorted(raw_root.glob("BOUT.dmp.*.nc"), key=rank_number)
    expected_rank_count = int(archive["expected_rank_file_count"])
    if len(paths) != expected_rank_count:
        raise ValueError(
            f"expected {expected_rank_count} rank files, found {len(paths)}"
        )
    if [rank_number(path) for path in paths] != list(range(expected_rank_count)):
        raise ValueError("rank filenames must cover exactly 0 through 255")

    decomposition = {
        key: int(value) for key, value in archive["mpi_decomposition"].items()
    }
    nxpe = decomposition["NXPE"]
    nype = decomposition["NYPE"]
    mxsub = decomposition["MXSUB"]
    mysub = decomposition["MYSUB"]
    mxg = decomposition["MXG"]
    myg = decomposition["MYG"]
    native_z = int(archive["native_z_samples"])
    frame_count = int(selection["total_frames"])
    volume_shape = (
        len(frame_indices),
        nxpe * mxsub,
        nype * mysub,
        native_z,
    )
    fields = {
        name: np.full(volume_shape, np.nan, dtype=np.float64)
        for name in VOLUME_FIELDS
    }
    boundary_streams = {
        side: {
            plane: np.full(
                (frame_count, nype * mysub, native_z),
                np.nan,
                dtype=np.float64,
            )
            for plane in phi_boundary.PLANE_NAMES
        }
        for side in phi_boundary.SIDES
    }
    coverage = np.zeros((nxpe, nype), dtype=np.int64)
    boundary_coverage = {
        side: np.zeros(nype * mysub, dtype=np.int64)
        for side in phi_boundary.SIDES
    }
    reference_metadata: dict[str, dict[str, Any]] | None = None

    for path in paths:
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
            if (
                scalar_value(dataset, "HERMES_SLOPE_LIMITER")
                != archive["slope_limiter"]
            ):
                raise ValueError(f"slope limiter mismatch in {path.name}")

            times = np.asarray(dataset.variables["t"][:], dtype=np.float64)
            if not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"full time sequence mismatch in {path.name}")

            pe_x = int(scalar_value(dataset, "PE_XIND"))
            pe_y = int(scalar_value(dataset, "PE_YIND"))
            if not (0 <= pe_x < nxpe and 0 <= pe_y < nype):
                raise ValueError(f"processor coordinate out of range in {path.name}")
            coverage[pe_x, pe_y] += 1

            metadata = {
                name: variable_metadata(dataset.variables[name])
                for name in VOLUME_FIELDS
            }
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise ValueError(f"field metadata differ in {path.name}")

            for name in VOLUME_FIELDS:
                variable = dataset.variables[name]
                if tuple(variable.dimensions) != ("t", "x", "y", "z"):
                    raise ValueError(f"unexpected axes for {name} in {path.name}")
                block = np.stack(
                    [
                        np.ma.filled(
                            variable[
                                frame,
                                mxg : mxg + mxsub,
                                myg : myg + mysub,
                                :,
                            ],
                            np.nan,
                        )
                        for frame in frame_indices
                    ],
                    axis=0,
                ).astype(np.float64, copy=False)
                place_rank_block(
                    fields[name],
                    block,
                    pe_x=pe_x,
                    pe_y=pe_y,
                    mxsub=mxsub,
                    mysub=mysub,
                )

            side: str | None = None
            if pe_x == 0:
                side = "inner"
            elif pe_x == nxpe - 1:
                side = "outer"
            if side is not None:
                phi = np.ma.filled(
                    dataset.variables["phi"][
                        :,
                        :,
                        myg : myg + mysub,
                        :,
                    ],
                    np.nan,
                ).astype(np.float64, copy=False)
                planes = phi_boundary.extract_boundary_planes(
                    phi,
                    side=side,
                    physical_y_slice=(0, mysub),
                )
                y0 = pe_y * mysub
                y1 = y0 + mysub
                for plane, values in planes.items():
                    boundary_streams[side][plane][:, y0:y1, :] = values
                boundary_coverage[side][y0:y1] += 1

    if not np.array_equal(coverage, np.ones_like(coverage)):
        raise ValueError("processor-coordinate coverage is incomplete or duplicated")
    for side in phi_boundary.SIDES:
        if not np.array_equal(
            boundary_coverage[side],
            np.ones(nype * mysub, dtype=np.int64),
        ):
            raise ValueError(f"boundary y coverage is incomplete for {side}")
    if reference_metadata is None:
        raise RuntimeError("no volume metadata were read")
    for name, values in fields.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"canonical field {name} contains non-finite values")

    stream_digests = {
        side: {
            plane: prior_audit.sha256_array(
                f"{side}:{plane}", boundary_streams[side][plane]
            )
            for plane in phi_boundary.PLANE_NAMES
        }
        for side in phi_boundary.SIDES
    }
    if stream_digests != prior_result["stream_digests"]:
        raise ValueError("rebuilt boundary streams differ from the locked audit")

    atol = float(prior_result["exact_tolerances"]["atol"])
    rtol = float(prior_result["exact_tolerances"]["rtol"])
    boundary_arrays, boundary_checks = derive_boundary_products(
        boundary_streams,
        frame_indices=frame_indices,
        atol=atol,
        rtol=rtol,
    )
    prior_check_names = {
        "outer_guard_copy_count_by_frame": "outer_guard_copy",
        "midpoint_constancy_count_by_frame": "midpoint_toroidal_constancy",
        "instantaneous_neumann_count_by_frame": "instantaneous_neumann",
    }
    for side in phi_boundary.SIDES:
        for local_name, prior_name in prior_check_names.items():
            expected = prior_result["per_side"][side][prior_name][
                "point_discrepancy_count_by_frame"
            ]
            if boundary_checks[side][local_name] != expected:
                raise ValueError(
                    f"selected extractor does not reproduce {side} {prior_name}"
                )
        if any(
            boundary_checks[side][name][frame] != 0
            for name in (
                "outer_guard_copy_count_by_frame",
                "midpoint_constancy_count_by_frame",
            )
            for frame in frame_indices
        ):
            raise ValueError(f"selected structural boundary checks fail for {side}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(args.output, "w", format="NETCDF4") as output:
        output.createDimension("selected_frame", len(frame_indices))
        output.createDimension("x", volume_shape[1])
        output.createDimension("y", volume_shape[2])
        output.createDimension("z", volume_shape[3])
        output.createDimension("side", len(phi_boundary.SIDES))
        output.setncattr(
            "paper0_protocol", "phase2_potential_elliptic_85604"
        )
        output.setncattr("development_run", "85604")
        output.setncattr("hermes_revision", archive["hermes_revision"])
        output.setncattr("zperiod", int(archive["zperiod"]))
        output.setncattr("boundary_side_order", ",".join(phi_boundary.SIDES))
        output.createVariable("frame_index", "i8", ("selected_frame",))[:] = (
            frame_indices
        )
        output.createVariable("normalized_time", "f8", ("selected_frame",))[:] = (
            expected_times[frame_indices]
        )
        output.createVariable(
            "relative_time_microseconds", "f8", ("selected_frame",)
        )[:] = np.asarray(
            selection["expected_relative_microseconds"], dtype=np.float64
        )
        output.createVariable("side_index", "i8", ("side",))[:] = np.arange(
            len(phi_boundary.SIDES), dtype=np.int64
        )
        for name in VOLUME_FIELDS:
            variable = output.createVariable(
                name,
                "f8",
                ("selected_frame", "x", "y", "z"),
                zlib=False,
            )
            variable[:] = fields[name]
            for key, value in reference_metadata[name].items():
                variable.setncattr(key, value)
        for name in BOUNDARY_ARRAYS:
            output.createVariable(
                name,
                "f8",
                ("selected_frame", "side", "y"),
                zlib=False,
            )[:] = boundary_arrays[name]

    record = {
        "schema_version": 1,
        "phase": "phase2_potential_elliptic_85604_extraction",
        "development_run": "85604",
        "held_out_run_read": False,
        "training_performed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "provenance_locks": locked_records,
        "raw_root": str(raw_root),
        "raw_control_digests": control_digests,
        "rank_file_count": len(paths),
        "processor_coverage_complete": True,
        "boundary_y_coverage_complete": True,
        "frame_indices": frame_indices,
        "normalized_times": expected_times[frame_indices].tolist(),
        "relative_time_microseconds": selection[
            "expected_relative_microseconds"
        ],
        "canonical_volume_shape": list(volume_shape),
        "canonical_volume_axes": ["selected_frame", "x", "y", "z"],
        "canonical_boundary_shape": [
            len(frame_indices),
            len(phi_boundary.SIDES),
            nype * mysub,
        ],
        "canonical_boundary_axes": ["selected_frame", "side", "y"],
        "boundary_side_order": list(phi_boundary.SIDES),
        "dtype": "float64",
        "variable_metadata": reference_metadata,
        "volume_array_digests": {
            name: sha256_array(values) for name, values in fields.items()
        },
        "boundary_array_digests": {
            name: sha256_array(values)
            for name, values in boundary_arrays.items()
        },
        "rebuilt_all_frame_boundary_stream_digests": stream_digests,
        "boundary_exact_tolerances": {"atol": atol, "rtol": rtol},
        "boundary_checks": boundary_checks,
        "canonical_file": str(args.output),
        "canonical_file_sha256": sha256_file(args.output),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
