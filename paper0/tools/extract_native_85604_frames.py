#!/usr/bin/env python3
"""Extract the prospectively frozen native-81 85604 oracle frames.

The extractor validates the original 256-rank decomposition, strips guards,
and assembles only the fields and frame indices declared in the committed
native-frame manifest. It refuses to overwrite either output artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


FIELDS = ("Ne", "Ni", "Te", "Ti", "Pe", "Pi", "phi")
RANK_PATTERN = re.compile(r"BOUT\.dmp\.(\d+)\.nc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    """Hash shape, canonical little-endian float64 dtype, and array bytes."""

    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"dtype": "<f8", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def nearest_half_up_indices(total_frames: int, fractions: list[float]) -> list[int]:
    if total_frames < 2:
        raise ValueError("total_frames must be at least two")
    if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
        raise ValueError("selection fractions must lie in [0, 1]")
    last = total_frames - 1
    return [math.floor(fraction * last + 0.5) for fraction in fractions]


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


def place_rank_block(
    destination: np.ndarray,
    block: np.ndarray,
    *,
    pe_x: int,
    pe_y: int,
    mxsub: int,
    mysub: int,
) -> None:
    """Place one guard-stripped block into canonical global coordinates."""

    expected = (destination.shape[0], mxsub, mysub, destination.shape[-1])
    if block.shape != expected:
        raise ValueError(f"rank block shape {block.shape} does not match {expected}")
    x0 = pe_x * mxsub
    y0 = pe_y * mysub
    x1 = x0 + mxsub
    y1 = y0 + mysub
    if x0 < 0 or y0 < 0 or x1 > destination.shape[1] or y1 > destination.shape[2]:
        raise ValueError("rank block lies outside the canonical destination")
    destination[:, x0:x1, y0:y1, :] = block


def variable_metadata(variable: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("cell_location", "conversion", "source", "species", "units"):
        if hasattr(variable, name):
            value = getattr(variable, name)
            if isinstance(value, np.generic):
                value = value.item()
            metadata[name] = value
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.record.exists():
        raise FileExistsError("refusing to overwrite a canonical frame artifact")

    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for native-frame extraction") from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["development_run"] != "85604":
        raise ValueError("native extractor accepts development run 85604 only")
    declared_root = Path(manifest["raw_archive"]["root"])
    if args.raw_root.resolve() != declared_root.resolve():
        raise ValueError("raw root differs from the prospectively frozen manifest")

    selection = manifest["frame_selection"]
    frame_indices = nearest_half_up_indices(
        int(selection["total_frames"]),
        [float(value) for value in selection["fractions"]],
    )
    if frame_indices != selection["indices"]:
        raise ValueError("manifest indices do not follow the frozen selection rule")

    archive = manifest["raw_archive"]
    locked_files = {
        "BOUT.inp": archive["bout_input_sha256"],
        "BOUT.settings": archive["bout_settings_sha256"],
        "tcv_85604_adjusted.nc": archive["geometry_sha256"],
    }
    for name, expected_digest in locked_files.items():
        actual_digest = sha256_file(args.raw_root / name)
        if actual_digest != expected_digest:
            raise ValueError(f"hash mismatch for raw control {name}")

    paths = sorted(args.raw_root.glob("BOUT.dmp.*.nc"), key=rank_number)
    expected_rank_count = int(archive["expected_rank_file_count"])
    if len(paths) != expected_rank_count:
        raise ValueError(
            f"expected {expected_rank_count} rank files, found {len(paths)}"
        )
    ranks = [rank_number(path) for path in paths]
    if ranks != list(range(expected_rank_count)):
        raise ValueError("rank filenames must cover exactly 0 through 255")

    decomposition = {key: int(value) for key, value in archive["mpi_decomposition"].items()}
    nxpe = decomposition["NXPE"]
    nype = decomposition["NYPE"]
    mxsub = decomposition["MXSUB"]
    mysub = decomposition["MYSUB"]
    mxg = decomposition["MXG"]
    myg = decomposition["MYG"]
    native_z = int(archive["native_z_samples"])
    expected_shape = (
        len(frame_indices),
        nxpe * mxsub,
        nype * mysub,
        native_z,
    )
    fields = {
        name: np.full(expected_shape, np.nan, dtype=np.float64) for name in FIELDS
    }
    coverage = np.zeros((nxpe, nype), dtype=np.int64)
    reference_times: np.ndarray | None = None
    reference_metadata: dict[str, dict[str, Any]] | None = None

    for path in paths:
        with netCDF4.Dataset(path, "r") as dataset:
            dimensions = {name: int(len(dimension)) for name, dimension in dataset.dimensions.items()}
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
            if not (0 <= pe_x < nxpe and 0 <= pe_y < nype):
                raise ValueError(f"processor coordinate outside decomposition in {path.name}")
            coverage[pe_x, pe_y] += 1

            times = np.asarray(dataset.variables["t"][frame_indices], dtype=np.float64)
            if reference_times is None:
                reference_times = times
            elif not np.array_equal(times, reference_times):
                raise ValueError(f"selected times disagree in {path.name}")

            metadata = {
                name: variable_metadata(dataset.variables[name]) for name in FIELDS
            }
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise ValueError(f"variable metadata disagree in {path.name}")

            for name in FIELDS:
                variable = dataset.variables[name]
                if tuple(variable.dimensions) != ("t", "x", "y", "z"):
                    raise ValueError(f"unexpected axis order for {name} in {path.name}")
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

    if not np.array_equal(coverage, np.ones_like(coverage)):
        raise ValueError("processor-coordinate coverage is incomplete or duplicated")
    assert reference_times is not None and reference_metadata is not None
    expected_times = np.asarray(selection["expected_normalized_times"], dtype=np.float64)
    if not np.allclose(reference_times, expected_times, rtol=0.0, atol=1.0e-12):
        raise ValueError("selected normalized times differ from the frozen values")
    for name, values in fields.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"canonical field {name} contains non-finite values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(args.output, "w", format="NETCDF4") as output:
        output.createDimension("selected_frame", len(frame_indices))
        output.createDimension("x", expected_shape[1])
        output.createDimension("y", expected_shape[2])
        output.createDimension("z", expected_shape[3])
        output.setncattr("paper0_protocol", "phase2_native_frame_oracle")
        output.setncattr("development_run", "85604")
        output.setncattr("hermes_revision", archive["hermes_revision"])
        output.setncattr("zperiod", int(archive["zperiod"]))
        output.createVariable("frame_index", "i8", ("selected_frame",))[:] = frame_indices
        output.createVariable("normalized_time", "f8", ("selected_frame",))[:] = reference_times
        output.createVariable("relative_time_microseconds", "f8", ("selected_frame",))[:] = np.asarray(
            selection["expected_relative_microseconds"], dtype=np.float64
        )
        for name in FIELDS:
            variable = output.createVariable(
                name,
                "f8",
                ("selected_frame", "x", "y", "z"),
                zlib=False,
            )
            variable[:] = fields[name]
            for key, value in reference_metadata[name].items():
                variable.setncattr(key, value)

    record = {
        "schema_version": 1,
        "phase": "phase2_native_85604_frame_extraction",
        "development_run": "85604",
        "held_out_run_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "raw_root": str(args.raw_root),
        "raw_control_digests": locked_files,
        "rank_file_count": len(paths),
        "processor_coverage": {
            "NXPE": nxpe,
            "NYPE": nype,
            "unique_coordinates": int(np.count_nonzero(coverage)),
        },
        "frame_indices": frame_indices,
        "normalized_times": reference_times.tolist(),
        "relative_time_microseconds": selection["expected_relative_microseconds"],
        "canonical_shape_per_variable": list(expected_shape),
        "canonical_axes": ["selected_frame", "x", "y", "z"],
        "dtype": "float64",
        "variable_metadata": reference_metadata,
        "variable_digests": {
            name: sha256_array(values) for name, values in fields.items()
        },
        "canonical_file": str(args.output),
        "canonical_file_sha256": sha256_file(args.output),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
