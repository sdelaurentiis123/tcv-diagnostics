#!/usr/bin/env python3
"""Stream the frozen 85604 archive into eight all-frame closure shards."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for search_path in (SRC, TOOLS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from extract_native_85604_frames import (  # noqa: E402
    rank_number,
    scalar_value,
    sha256_array,
    sha256_file,
    variable_metadata,
)
from tcv_diagnostics import phi_boundary  # noqa: E402


VOLUME_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")
EXPECTED_SHARDS = tuple((start, start + 78) for start in range(0, 624, 78))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ),
    )


def verify_development_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "85606" in lowered:
        raise ValueError("held-out run path is prohibited")
    if "85604" not in lowered:
        raise ValueError("raw root must identify development run 85604")


def expected_times(scope: dict[str, Any]) -> np.ndarray:
    frames = int(scope["frame_count"])
    first = float(scope["expected_first_normalized_time"])
    cadence = float(scope["expected_normalized_cadence"])
    result = first + cadence * np.arange(frames, dtype=np.float64)
    if result[-1] != float(scope["expected_last_normalized_time"]):
        raise ValueError("manifest time endpoint is inconsistent")
    return result


def discrepancy_count(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> int:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("boundary structural comparison shapes differ")
    finite = np.isfinite(reference) & np.isfinite(candidate)
    difference = candidate - reference
    return int(
        np.count_nonzero(
            (~finite) | (np.abs(difference) > atol + rtol * np.abs(reference))
        )
    )


def canonical_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"canonical_shard_{shard_index}.nc"


@contextmanager
def staged_rank_file(source: Path, scratch_dir: Path) -> Iterator[Path]:
    """Stage exactly one immutable raw rank file into node-local storage."""

    destination = scratch_dir / source.name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite staged rank {destination}")
    if shutil.disk_usage(scratch_dir).free < source.stat().st_size:
        raise OSError(f"insufficient node-local scratch for {source.name}")
    try:
        shutil.copyfile(source, destination)
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"staged rank size differs for {source.name}")
        yield destination
    finally:
        destination.unlink(missing_ok=True)


def create_canonical_files(
    stack: ExitStack,
    *,
    netcdf4: Any,
    output_dir: Path,
    manifest: dict[str, Any],
    times: np.ndarray,
) -> tuple[list[Any], list[dict[str, Any]]]:
    extraction = manifest["canonical_extraction"]
    cadence_us = float(manifest["frame_scope"]["physical_cadence_microseconds"])
    files: list[Any] = []
    variables: list[dict[str, Any]] = []
    for shard_index, (start, stop) in enumerate(EXPECTED_SHARDS):
        dataset = stack.enter_context(
            netcdf4.Dataset(
                canonical_path(output_dir, shard_index), "w", format="NETCDF4"
            )
        )
        dataset.createDimension("frame", stop - start)
        dataset.createDimension("x", 64)
        dataset.createDimension("y", 32)
        dataset.createDimension("z", 81)
        dataset.createDimension("side", 2)
        dataset.setncattr(
            "paper0_protocol", "phase2_potential_vorticity_all_frame_85604"
        )
        dataset.setncattr("development_run", "85604")
        dataset.setncattr("shard_index", shard_index)
        dataset.setncattr("shard_start", start)
        dataset.setncattr("shard_stop", stop)
        dataset.setncattr("zperiod", 5)
        dataset.setncattr("boundary_side_order", "inner,outer")
        dataset.createVariable("frame_index", "i8", ("frame",))[:] = np.arange(
            start, stop, dtype=np.int64
        )
        dataset.createVariable("normalized_time", "f8", ("frame",))[:] = times[
            start:stop
        ]
        dataset.createVariable(
            "relative_time_microseconds", "f8", ("frame",)
        )[:] = cadence_us * np.arange(start, stop, dtype=np.float64)
        dataset.createVariable("side_index", "i8", ("side",))[:] = np.arange(
            2, dtype=np.int64
        )
        shard_variables = {
            field: dataset.createVariable(
                field,
                "f8",
                tuple(extraction["volume_axes"]),
                zlib=False,
                # One chunk is exactly the 78-frame physical block written by
                # one source rank.  A one-frame time chunk creates 78 small
                # Ceph allocations for every assignment without changing any
                # canonical value; matching the write slab keeps extraction
                # streaming while avoiding that metadata bottleneck.
                chunksizes=(78, 4, 2, 81),
            )
            for field in VOLUME_FIELDS
        }
        shard_variables["saved_midpoint"] = dataset.createVariable(
            "saved_midpoint",
            "f8",
            tuple(extraction["boundary_axes"]),
            zlib=False,
            chunksizes=(78, 1, 2),
        )
        files.append(dataset)
        variables.append(shard_variables)
    return files, variables


def validate_rank_metadata(
    dataset: Any,
    *,
    path: Path,
    archive: dict[str, Any],
) -> tuple[int, int]:
    dimensions = {
        name: int(len(dimension)) for name, dimension in dataset.dimensions.items()
    }
    if dimensions != archive["rank_dimensions"]:
        raise ValueError(f"unexpected dimensions in {path.name}: {dimensions}")
    decomposition = archive["mpi_decomposition"]
    for name, expected in decomposition.items():
        if int(scalar_value(dataset, name)) != int(expected):
            raise ValueError(f"{name} mismatch in {path.name}")
    if int(scalar_value(dataset, "MZ")) != int(archive["native_z_samples"]):
        raise ValueError(f"MZ mismatch in {path.name}")
    if int(scalar_value(dataset, "zperiod")) != int(archive["zperiod"]):
        raise ValueError(f"zperiod mismatch in {path.name}")
    if scalar_value(dataset, "HERMES_REVISION") != archive["hermes_revision"]:
        raise ValueError(f"Hermes revision mismatch in {path.name}")
    if scalar_value(dataset, "HERMES_SLOPE_LIMITER") != archive["slope_limiter"]:
        raise ValueError(f"slope limiter mismatch in {path.name}")
    pe_x = int(scalar_value(dataset, "PE_XIND"))
    pe_y = int(scalar_value(dataset, "PE_YIND"))
    if not (0 <= pe_x < 16 and 0 <= pe_y < 16):
        raise ValueError(f"processor coordinate out of range in {path.name}")
    return pe_x, pe_y


def update_pressure_inventory(
    inventory: dict[str, Any],
    *,
    field: str,
    block: np.ndarray,
    pe_x: int,
    pe_y: int,
    frame_start: int = 0,
) -> None:
    negative = block < 0.0
    count = int(np.count_nonzero(negative))
    inventory[f"negative_raw_{field}_count"] += count
    frame_stop = frame_start + block.shape[0]
    for shard_index, (start, stop) in enumerate(EXPECTED_SHARDS):
        overlap_start = max(start, frame_start)
        overlap_stop = min(stop, frame_stop)
        if overlap_start >= overlap_stop:
            continue
        inventory[f"negative_raw_{field}_count_by_shard"][shard_index] += int(
            np.count_nonzero(
                negative[
                    overlap_start - frame_start : overlap_stop - frame_start
                ]
            )
        )
    if field != "Pi":
        return
    flat_index = int(np.argmin(block))
    value = float(block.flat[flat_index])
    if value < inventory["minimum_raw_Pi"]:
        t, local_x, local_y, z = np.unravel_index(flat_index, block.shape)
        inventory["minimum_raw_Pi"] = value
        inventory["minimum_raw_Pi_location_txyz"] = [
            frame_start + int(t),
            pe_x * block.shape[1] + int(local_x),
            pe_y * block.shape[2] + int(local_y),
            int(z),
        ]


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite all-frame extraction artifacts")
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for all-frame extraction") from error

    manifest_path = args.manifest.resolve(strict=True)
    manifest = load_json(manifest_path)
    if (
        manifest["protocol_status"]
        != "frozen_before_first_all_624_frame_forward_closure_calculation"
        or manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_allowed"]
    ):
        raise ValueError("manifest violates the frozen development-only scope")
    intervals = [tuple(item) for item in manifest["shards"]["half_open_intervals"]]
    if tuple(intervals) != EXPECTED_SHARDS:
        raise ValueError("manifest shard intervals differ from the frozen partition")

    raw_root = args.raw_root.resolve(strict=True)
    verify_development_path(raw_root)
    archive = manifest["raw_archive"]
    if raw_root != Path(archive["root"]).resolve():
        raise ValueError("raw root differs from the frozen manifest")
    for name, expected in (
        ("BOUT.inp", archive["bout_input_sha256"]),
        ("BOUT.settings", archive["bout_settings_sha256"]),
        ("tcv_85604_adjusted.nc", archive["geometry_sha256"]),
    ):
        if sha256_file(raw_root / name) != expected:
            raise ValueError(f"raw control SHA-256 mismatch for {name}")

    repository_root = manifest_path.parents[2]
    protocol_lock = manifest["protocol"]
    protocol_path = repository_root / protocol_lock["path"]
    if sha256_file(protocol_path) != protocol_lock["sha256"]:
        raise ValueError("all-frame protocol SHA-256 mismatch")
    predecessor_locks: dict[str, dict[str, str]] = {}
    for name, lock in manifest["provenance_locks"].items():
        path = repository_root / lock["path"]
        actual = sha256_file(path)
        if actual != lock["sha256"]:
            raise ValueError(f"predecessor SHA-256 mismatch for {name}")
        predecessor_locks[name] = {"path": str(path), "sha256": actual}

    paths = sorted(raw_root.glob("BOUT.dmp.*.nc"), key=rank_number)
    expected_rank_count = int(archive["expected_rank_file_count"])
    if len(paths) != expected_rank_count:
        raise ValueError(
            f"expected {expected_rank_count} rank files, found {len(paths)}"
        )
    if [rank_number(path) for path in paths] != list(range(expected_rank_count)):
        raise ValueError("rank filenames must cover exactly 0 through 255")

    scratch_dir = args.scratch_dir.absolute()
    if scratch_dir.exists():
        raise FileExistsError(f"refusing existing scratch directory {scratch_dir}")
    if raw_root == scratch_dir or raw_root in scratch_dir.parents:
        raise ValueError("node-local scratch may not be inside the raw archive")
    scratch_dir.mkdir(parents=True, exist_ok=False)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    times = expected_times(manifest["frame_scope"])
    coverage = np.zeros((16, 16), dtype=np.int64)
    boundary_coverage = {
        side: np.zeros(32, dtype=np.int64) for side in phi_boundary.SIDES
    }
    boundary_checks = {
        side: {
            "outer_guard_copy_discrepancy_count": 0,
            "midpoint_constancy_discrepancy_count": 0,
            "nonfinite_count": 0,
            "outer_guard_copy_max_abs": 0.0,
            "midpoint_constancy_max_abs": 0.0,
        }
        for side in phi_boundary.SIDES
    }
    pressure_inventory: dict[str, Any] = {
        "negative_raw_Pe_count": 0,
        "negative_raw_Pi_count": 0,
        "negative_raw_Pe_count_by_shard": [0] * 8,
        "negative_raw_Pi_count_by_shard": [0] * 8,
        "minimum_raw_Pi": math.inf,
        "minimum_raw_Pi_location_txyz": None,
    }
    reference_metadata: dict[str, dict[str, Any]] | None = None
    decomposition = {
        key: int(value) for key, value in archive["mpi_decomposition"].items()
    }
    mxsub = decomposition["MXSUB"]
    mysub = decomposition["MYSUB"]
    mxg = decomposition["MXG"]
    myg = decomposition["MYG"]

    with ExitStack() as output_stack:
        canonical_files, shard_variables = create_canonical_files(
            output_stack,
            netcdf4=netCDF4,
            output_dir=args.output_dir,
            manifest=manifest,
            times=times,
        )
        metadata_written = False
        for path in paths:
            with staged_rank_file(path, scratch_dir) as staged_path, netCDF4.Dataset(
                staged_path, "r"
            ) as source:
                pe_x, pe_y = validate_rank_metadata(
                    source, path=path, archive=archive
                )
                coverage[pe_x, pe_y] += 1
                metadata = {
                    name: variable_metadata(source.variables[name])
                    for name in VOLUME_FIELDS
                }
                if reference_metadata is None:
                    reference_metadata = metadata
                elif metadata != reference_metadata:
                    raise ValueError(f"field metadata differ in {path.name}")
                if not metadata_written:
                    for variables in shard_variables:
                        for field in VOLUME_FIELDS:
                            for key, value in metadata[field].items():
                                variables[field].setncattr(key, value)
                    metadata_written = True

                x0 = pe_x * mxsub
                x1 = x0 + mxsub
                y0 = pe_y * mysub
                y1 = y0 + mysub
                side: str | None = None
                side_index = -1
                if pe_x == 0:
                    side = "inner"
                    side_index = 0
                elif pe_x == 15:
                    side = "outer"
                    side_index = 1
                for field in VOLUME_FIELDS:
                    if tuple(source.variables[field].dimensions) != (
                        "t",
                        "x",
                        "y",
                        "z",
                    ):
                        raise ValueError(f"unexpected axes for {field} in {path.name}")

                # BOUT++ allocated one HDF5 chunk per saved frame and wrote
                # the time-dependent variables in frame order.  Read the five
                # required fields, then t, for one frame at a time so the raw
                # file is traversed in its native order.  Buffer only the
                # current 78-frame local-rank slab before one canonical write.
                for shard_index, (start, stop) in enumerate(EXPECTED_SHARDS):
                    blocks = {
                        field: np.empty(
                            (stop - start, mxsub, mysub, 81), dtype=np.float64
                        )
                        for field in VOLUME_FIELDS
                    }
                    raw_phi_boundary = (
                        np.empty(
                            (stop - start, mxsub + 2 * mxg, mysub, 81),
                            dtype=np.float64,
                        )
                        if side is not None
                        else None
                    )
                    for local_frame, frame_index in enumerate(range(start, stop)):
                        for field in VOLUME_FIELDS:
                            variable = source.variables[field]
                            if field == "phi" and raw_phi_boundary is not None:
                                raw_frame = np.ma.filled(
                                    variable[
                                        frame_index,
                                        :,
                                        myg : myg + mysub,
                                        :,
                                    ],
                                    np.nan,
                                ).astype(np.float64, copy=False)
                                if raw_frame.shape != (
                                    mxsub + 2 * mxg,
                                    mysub,
                                    81,
                                ):
                                    raise ValueError(
                                        f"unexpected raw phi frame in {path.name}"
                                    )
                                raw_phi_boundary[local_frame] = raw_frame
                                frame = raw_frame[mxg : mxg + mxsub]
                            else:
                                frame = np.ma.filled(
                                    variable[
                                        frame_index,
                                        mxg : mxg + mxsub,
                                        myg : myg + mysub,
                                        :,
                                    ],
                                    np.nan,
                                ).astype(np.float64, copy=False)
                            if frame.shape != (mxsub, mysub, 81):
                                raise ValueError(
                                    f"unexpected {field} frame in {path.name}"
                                )
                            if not np.all(np.isfinite(frame)):
                                raise ValueError(
                                    f"non-finite {field} frame in {path.name}"
                                )
                            blocks[field][local_frame] = frame

                        actual_time = float(
                            np.ma.filled(
                                source.variables["t"][frame_index], np.nan
                            )
                        )
                        if (
                            not np.isfinite(actual_time)
                            or actual_time != times[frame_index]
                        ):
                            raise ValueError(
                                f"time mismatch at frame {frame_index} in {path.name}"
                            )

                    for field, block in blocks.items():
                        if field in ("Pe", "Pi"):
                            update_pressure_inventory(
                                pressure_inventory,
                                field=field,
                                block=block,
                                pe_x=pe_x,
                                pe_y=pe_y,
                                frame_start=start,
                            )
                        shard_variables[shard_index][field][
                            :, x0:x1, y0:y1, :
                        ] = block

                    if raw_phi_boundary is None:
                        continue
                    planes = phi_boundary.extract_boundary_planes(
                        raw_phi_boundary, side=side, physical_y_slice=(0, mysub)
                    )
                    if any(not np.all(np.isfinite(values)) for values in planes.values()):
                        boundary_checks[side]["nonfinite_count"] += sum(
                            int(np.count_nonzero(~np.isfinite(values)))
                            for values in planes.values()
                        )
                    outermost = planes["outermost_guard"]
                    adjacent = planes["adjacent_guard"]
                    interior = planes["adjacent_interior"]
                    midpoint = 0.5 * (adjacent + interior)
                    midpoint_mean = np.mean(midpoint, axis=-1, dtype=np.float64)
                    midpoint_reference = np.broadcast_to(
                        midpoint_mean[..., None], midpoint.shape
                    )
                    boundary_checks[side][
                        "outer_guard_copy_discrepancy_count"
                    ] += discrepancy_count(
                        adjacent, outermost, atol=1e-12, rtol=1e-12
                    )
                    boundary_checks[side][
                        "midpoint_constancy_discrepancy_count"
                    ] += discrepancy_count(
                        midpoint_reference, midpoint, atol=1e-12, rtol=1e-12
                    )
                    boundary_checks[side]["outer_guard_copy_max_abs"] = max(
                        boundary_checks[side]["outer_guard_copy_max_abs"],
                        float(np.max(np.abs(outermost - adjacent))),
                    )
                    boundary_checks[side]["midpoint_constancy_max_abs"] = max(
                        boundary_checks[side]["midpoint_constancy_max_abs"],
                        float(np.max(np.abs(midpoint - midpoint_reference))),
                    )
                    shard_variables[shard_index]["saved_midpoint"][
                        :, side_index, y0:y1
                    ] = midpoint_mean
                if side is not None:
                    boundary_coverage[side][y0:y1] += 1

        scratch_dir.rmdir()

        if not np.array_equal(coverage, np.ones_like(coverage)):
            raise ValueError("processor-coordinate coverage is incomplete or duplicated")
        for side in phi_boundary.SIDES:
            if not np.array_equal(boundary_coverage[side], np.ones(32, dtype=np.int64)):
                raise ValueError(f"boundary y coverage is incomplete for {side}")
            if any(
                boundary_checks[side][name] != 0
                for name in (
                    "outer_guard_copy_discrepancy_count",
                    "midpoint_constancy_discrepancy_count",
                    "nonfinite_count",
                )
            ):
                raise ValueError(f"boundary structural gate fails for {side}")
        for dataset in canonical_files:
            dataset.sync()

    expected_pressure = manifest["raw_pressure_identity"]
    for key in (
        "negative_raw_Pe_count",
        "negative_raw_Pi_count",
        "negative_raw_Pi_count_by_shard",
        "minimum_raw_Pi",
        "minimum_raw_Pi_location_txyz",
    ):
        if pressure_inventory[key] != expected_pressure[key]:
            raise ValueError(
                f"raw pressure identity {key} differs: "
                f"{pressure_inventory[key]} != {expected_pressure[key]}"
            )

    shard_records: list[dict[str, Any]] = []
    for shard_index, (start, stop) in enumerate(EXPECTED_SHARDS):
        path = canonical_path(args.output_dir, shard_index)
        with netCDF4.Dataset(path, "r") as dataset:
            dimensions = {
                name: int(len(dimension))
                for name, dimension in dataset.dimensions.items()
            }
            expected_dimensions = {
                "frame": 78,
                "x": 64,
                "y": 32,
                "z": 81,
                "side": 2,
            }
            if dimensions != expected_dimensions:
                raise ValueError(f"canonical shard {shard_index} dimensions differ")
            frames = np.asarray(dataset.variables["frame_index"][:], dtype=np.int64)
            if not np.array_equal(frames, np.arange(start, stop, dtype=np.int64)):
                raise ValueError(f"canonical shard {shard_index} frames differ")
            array_digests: dict[str, str] = {}
            for name in (*VOLUME_FIELDS, "saved_midpoint"):
                values = np.asarray(dataset.variables[name][:], dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"canonical shard {shard_index} {name} is non-finite")
                array_digests[name] = sha256_array(values)
        shard_records.append(
            {
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "frame_indices": list(range(start, stop)),
                "canonical_file": str(path),
                "canonical_file_sha256": sha256_file(path),
                "array_sha256": array_digests,
            }
        )

    if reference_metadata is None:
        raise RuntimeError("no raw rank metadata were read")
    record = {
        "schema_version": 1,
        "phase": "phase2_potential_vorticity_all_frame_85604_extraction",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_root": str(raw_root),
        "node_local_scratch": str(scratch_dir),
        "raw_rank_staged_sequentially_once": True,
        "maximum_simultaneous_staged_rank_files": 1,
        "staged_rank_files_retained": False,
        "rank_file_count": len(paths),
        "rank_files_traversed_once": True,
        "raw_rank_read_order": [*VOLUME_FIELDS, "t"],
        "time_major_local_rank_buffer_frames": 78,
        "processor_coverage_complete": True,
        "boundary_y_coverage_complete": True,
        "frame_count": 624,
        "frame_indices": list(range(624)),
        "normalized_times": times.tolist(),
        "shard_intervals": [list(item) for item in EXPECTED_SHARDS],
        "canonical_dtype": "float64",
        "canonical_volume_chunks": [78, 4, 2, 81],
        "canonical_boundary_chunks": [78, 1, 2],
        "canonical_volume_axes": ["frame", "x", "y", "z"],
        "canonical_boundary_axes": ["frame", "side", "y"],
        "boundary_side_order": list(phi_boundary.SIDES),
        "variable_metadata": reference_metadata,
        "raw_pressure_identity": pressure_inventory,
        "boundary_checks": boundary_checks,
        "provenance_locks": predecessor_locks,
        "shards": shard_records,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "frame_count": 624,
                "rank_file_count": len(paths),
                "shard_count": len(shard_records),
                "negative_raw_Pi_count": pressure_inventory[
                    "negative_raw_Pi_count"
                ],
                "output": str(args.record),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
