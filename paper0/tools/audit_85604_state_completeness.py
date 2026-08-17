#!/usr/bin/env python3
"""Stream one deterministic shard of the frozen 85604 state audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics import state_completeness as state  # noqa: E402


RANK_PATTERN = re.compile(r"BOUT\.dmp\.(\d+)\.nc")


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


def strict_json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary file {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_development_path(path: Path) -> None:
    lowered = {part.lower() for part in path.parts}
    if "85606" in lowered:
        raise ValueError("held-out run path is prohibited")
    if "85604" not in lowered:
        raise ValueError("raw root must identify development run 85604")


def normalized_attribute(variable: Any, name: str) -> Any:
    if not hasattr(variable, name):
        return None
    value = getattr(variable, name)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def validate_and_record_field_metadata(
    dataset: Any,
    manifest: dict[str, Any],
    *,
    path_name: str,
) -> dict[str, Any]:
    inventory = manifest["field_inventory"]
    expected_common = inventory["expected_common"]
    expected_shape = tuple(manifest["raw_archive"]["rank_dimensions"].values())
    result: dict[str, Any] = {}
    for name in inventory["metadata_inventory_fields"]:
        if name not in dataset.variables:
            raise ValueError(f"missing field {name} in {path_name}")
        variable = dataset.variables[name]
        dimensions = list(variable.dimensions)
        shape = [int(size) for size in variable.shape]
        dtype = str(variable.dtype)
        if dimensions != expected_common["dimensions"]:
            raise ValueError(f"axis order mismatch for {name} in {path_name}")
        if tuple(shape) != expected_shape:
            raise ValueError(f"shape mismatch for {name} in {path_name}: {shape}")
        if dtype != expected_common["dtype"]:
            raise ValueError(f"dtype mismatch for {name} in {path_name}: {dtype}")
        common_attributes = {
            key: normalized_attribute(variable, key)
            for key in ("cell_location", "time_dimension")
        }
        if common_attributes != {
            "cell_location": expected_common["cell_location"],
            "time_dimension": expected_common["time_dimension"],
        }:
            raise ValueError(f"common metadata mismatch for {name} in {path_name}")

        expected_field = inventory["expected_per_field"][name]
        actual_field = {
            "source": normalized_attribute(variable, "source"),
            "species": normalized_attribute(variable, "species"),
            "units": normalized_attribute(variable, "units"),
            "conversion": normalized_attribute(variable, "conversion"),
        }
        if actual_field["source"] != expected_field["source"]:
            raise ValueError(f"source metadata mismatch for {name} in {path_name}")
        if actual_field["species"] != expected_field["species"]:
            raise ValueError(f"species metadata mismatch for {name} in {path_name}")
        if actual_field["units"] != expected_field["units"]:
            raise ValueError(f"units metadata mismatch for {name} in {path_name}")
        if not math.isclose(
            float(actual_field["conversion"]),
            float(expected_field["conversion"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(f"conversion metadata mismatch for {name} in {path_name}")
        result[name] = {
            "dimensions": dimensions,
            "shape": shape,
            "dtype": dtype,
            **common_attributes,
            **actual_field,
        }
    return result


def expected_scope_counts_for_coverage(
    coverage: np.ndarray,
    *,
    frame_count: int,
    mxsub: int,
    mysub: int,
    native_z: int,
) -> dict[str, int]:
    counts = {name: 0 for name in state.SCOPE_NAMES}
    points_per_y = frame_count * mxsub * native_z
    for _, pe_y in np.argwhere(coverage == 1):
        global_y = int(pe_y) * mysub + np.arange(mysub)
        for name in state.SCOPE_NAMES:
            counts[name] += points_per_y * int(
                state.scope_y_indices(global_y, name).size
            )
    return counts


def validate_scope_accounting(
    field_statistics: dict[str, Any],
    closures: dict[str, Any],
    expected: dict[str, int],
) -> None:
    for field, statistics in field_statistics.items():
        for scope, count in expected.items():
            if statistics["scopes"][scope]["total_count"] != count:
                raise ValueError(f"incomplete field accounting for {field}/{scope}")
    for relation, statistics in closures.items():
        for scope, count in expected.items():
            if statistics["scopes"][scope]["total_count"] != count:
                raise ValueError(f"incomplete closure accounting for {relation}/{scope}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the state audit") from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["development_run"] != "85604":
        raise ValueError("state audit accepts development run 85604 only")
    if manifest["held_out_85606_access_allowed"]:
        raise ValueError("manifest must prohibit held-out access")
    raw_root = args.raw_root.resolve()
    verify_development_path(raw_root)
    if raw_root != Path(manifest["raw_archive"]["root"]).resolve():
        raise ValueError("raw root differs from frozen manifest")

    archive = manifest["raw_archive"]
    control_digests = {
        "BOUT.inp": archive["bout_input_sha256"],
        "BOUT.settings": archive["bout_settings_sha256"],
        "tcv_85604_adjusted.nc": archive["geometry_sha256"],
    }
    for name, expected in control_digests.items():
        if sha256_file(raw_root / name) != expected:
            raise ValueError(f"hash mismatch for raw control {name}")

    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    all_paths = sorted(raw_root.glob("BOUT.dmp.*.nc"), key=rank_number)
    rank_count = int(archive["expected_rank_file_count"])
    if len(all_paths) != rank_count:
        raise ValueError(f"expected {rank_count} rank files, found {len(all_paths)}")
    if [rank_number(path) for path in all_paths] != list(range(rank_count)):
        raise ValueError("rank filenames must cover exactly 0 through 255")
    paths = [
        path
        for path in all_paths
        if rank_number(path) % args.shard_count == args.shard_index
    ]
    expected_ranks = [
        rank for rank in range(rank_count) if rank % args.shard_count == args.shard_index
    ]
    if [rank_number(path) for path in paths] != expected_ranks or not paths:
        raise ValueError("rank shard violates frozen modulo rule")

    decomposition = {
        key: int(value) for key, value in archive["mpi_decomposition"].items()
    }
    frame_scope = manifest["frame_scope"]
    frame_count = int(frame_scope["frame_count"])
    expected_times = float(frame_scope["expected_first_normalized_time"]) + float(
        frame_scope["expected_normalized_cadence"]
    ) * np.arange(frame_count, dtype=np.float64)
    if expected_times[-1] != float(frame_scope["expected_last_normalized_time"]):
        raise ValueError("frozen time endpoints are inconsistent")
    native_z = int(archive["native_z_samples"])
    nx = decomposition["NXPE"] * decomposition["MXSUB"]
    ny = decomposition["NYPE"] * decomposition["MYSUB"]
    expected_global_shape = [frame_count, nx, ny, native_z]
    if expected_global_shape != manifest["canonical_cells"]["shape_per_field"]:
        raise ValueError("canonical shape is inconsistent")

    blocks = [
        (int(first), int(last))
        for first, last in manifest["temporal_blocks"]["inclusive_index_ranges"]
    ]
    closure_spec = manifest["closure_statistics"]
    atol = float(closure_spec["atol"])
    rtol = float(closure_spec["rtol"])
    formula = manifest["momentum_formula"]
    density_floor = float(formula["density_floor"])
    electron_mass = float(formula["electron_atomic_mass"]["numerator"]) / float(
        formula["electron_atomic_mass"]["denominator"]
    )
    ion_mass = float(formula["ion_atomic_mass"]["numerator"]) / float(
        formula["ion_atomic_mass"]["denominator"]
    )
    if tuple(formula["relations"]) != state.RELATIONS:
        raise ValueError("manifest relations disagree with implementation")
    if tuple(manifest["field_inventory"]["value_stream_fields"]) != state.STREAM_FIELDS:
        raise ValueError("manifest value streams disagree with implementation")

    fields = {name: state.FieldAccumulator() for name in state.STREAM_FIELDS}
    density = state.DensityFloorAccumulator(
        frame_count=frame_count,
        nx=nx,
        ny=ny,
        density_floor=density_floor,
        temporal_blocks=blocks,
    )
    closures = {
        name: state.ClosureAccumulator(
            frame_count=frame_count,
            nx=nx,
            ny=ny,
            temporal_blocks=blocks,
            atol=atol,
            rtol=rtol,
        )
        for name in state.RELATIONS
    }
    stream_digests = state.initialize_stream_digests(
        state.STREAM_FIELDS, expected_global_shape
    )
    coverage = np.zeros((decomposition["NXPE"], decomposition["NYPE"]), dtype=np.int64)
    reference_metadata: dict[str, Any] | None = None

    for path in paths:
        rank = rank_number(path)
        with netCDF4.Dataset(path, "r") as dataset:
            dimensions = {
                name: int(len(dimension)) for name, dimension in dataset.dimensions.items()
            }
            if dimensions != archive["rank_dimensions"]:
                raise ValueError(f"rank dimensions mismatch in {path.name}: {dimensions}")
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
            if not np.allclose(times, expected_times, rtol=0.0, atol=1e-12):
                raise ValueError(f"time sequence mismatch in {path.name}")

            metadata = validate_and_record_field_metadata(
                dataset, manifest, path_name=path.name
            )
            if reference_metadata is None:
                reference_metadata = metadata
            elif metadata != reference_metadata:
                raise ValueError(f"field metadata differ in {path.name}")

            mxg, myg = decomposition["MXG"], decomposition["MYG"]
            mxsub, mysub = decomposition["MXSUB"], decomposition["MYSUB"]
            rank_fields: dict[str, np.ndarray] = {}
            for name in state.STREAM_FIELDS:
                variable = dataset.variables[name]
                values = np.ma.filled(
                    variable[:, mxg : mxg + mxsub, myg : myg + mysub, :],
                    np.nan,
                ).astype(np.float64, copy=False)
                expected_shape = (frame_count, mxsub, mysub, native_z)
                if values.shape != expected_shape:
                    raise ValueError(f"physical shape mismatch for {name} in {path.name}")
                rank_fields[name] = values
                fields[name].update(values, x0=pe_x * mxsub, y0=pe_y * mysub)
                state.update_stream_digest(
                    stream_digests[name],
                    values,
                    rank=rank,
                    pe_x=pe_x,
                    pe_y=pe_y,
                )
            density.update(rank_fields["Ne"], x0=pe_x * mxsub, y0=pe_y * mysub)
            for name, accumulator in closures.items():
                reference, candidate = state.relation_arrays(
                    name,
                    rank_fields,
                    density_floor=density_floor,
                    electron_atomic_mass=electron_mass,
                    ion_atomic_mass=ion_mass,
                )
                accumulator.update(
                    reference,
                    candidate,
                    x0=pe_x * mxsub,
                    y0=pe_y * mysub,
                )

    if np.any(coverage > 1) or int(np.count_nonzero(coverage)) != len(paths):
        raise ValueError("processor-coordinate coverage is duplicated or incomplete")
    if args.shard_count == 1 and not np.array_equal(coverage, np.ones_like(coverage)):
        raise ValueError("full processor-coordinate coverage is incomplete")
    if reference_metadata is None:
        raise RuntimeError("no rank metadata were read")

    field_results = {name: accumulator.result() for name, accumulator in fields.items()}
    closure_results = {
        name: accumulator.result() for name, accumulator in closures.items()
    }
    expected_scope_counts = expected_scope_counts_for_coverage(
        coverage,
        frame_count=frame_count,
        mxsub=decomposition["MXSUB"],
        mysub=decomposition["MYSUB"],
        native_z=native_z,
    )
    validate_scope_accounting(field_results, closure_results, expected_scope_counts)
    complete = args.shard_count == 1
    result = {
        "schema_version": 1,
        "phase": (
            "phase2_85604_state_completeness_audit"
            if complete
            else "phase2_85604_state_completeness_rank_shard"
        ),
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "audit_completed": complete,
        "rank_shard_completed": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "raw_root": str(raw_root),
        "raw_control_digests": control_digests,
        "archive_rank_file_count": len(all_paths),
        "rank_file_count": len(paths),
        "rank_indices": expected_ranks,
        "rank_shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "rule": "rank modulo shard_count equals shard_index",
        },
        "processor_coverage": {
            "NXPE": decomposition["NXPE"],
            "NYPE": decomposition["NYPE"],
            "unique_coordinates": int(np.count_nonzero(coverage)),
            "coordinates": np.argwhere(coverage == 1).astype(int).tolist(),
            "complete": bool(np.array_equal(coverage, np.ones_like(coverage))),
        },
        "frame_count": frame_count,
        "normalized_time": {
            "first": float(expected_times[0]),
            "last": float(expected_times[-1]),
            "cadence": float(frame_scope["expected_normalized_cadence"]),
            "physical_cadence_microseconds": float(
                frame_scope["physical_cadence_microseconds"]
            ),
        },
        "native_z_samples": native_z,
        "zperiod": int(archive["zperiod"]),
        "shape_per_field": expected_global_shape,
        "points_covered_per_stream": expected_scope_counts["full_physical_domain"],
        "expected_scope_counts": expected_scope_counts,
        "variable_metadata": reference_metadata,
        "guard_stripped_rank_stream_digests": {
            name: digest.hexdigest() for name, digest in stream_digests.items()
        },
        "field_statistics": field_results,
        "density_floor_statistics": density.result(),
        "closure_statistics": {
            "atol": atol,
            "rtol": rtol,
            "relations": closure_results,
        },
    }
    if complete:
        result["scientific_findings"] = state.derive_findings(
            field_results, closure_results
        )
    strict_json_write(args.output, result)
    print(
        f"Wrote {'complete audit' if complete else 'rank shard'}: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
