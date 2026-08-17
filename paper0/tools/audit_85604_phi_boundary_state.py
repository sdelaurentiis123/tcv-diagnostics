#!/usr/bin/env python3
"""Run the frozen all-frame 85604 saved-potential boundary-state audit."""

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

from tcv_diagnostics import phi_boundary as boundary  # noqa: E402


RANK_PATTERN = re.compile(r"BOUT\.dmp\.(\d+)\.nc")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(name: str, values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"name": name, "dtype": "<f8", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
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


def normalized_attribute(variable: Any, name: str) -> Any:
    if not hasattr(variable, name):
        return None
    value = getattr(variable, name)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def validate_phi_metadata(variable: Any, expected: dict[str, Any]) -> dict[str, Any]:
    actual = {
        "dimensions": list(variable.dimensions),
        "shape": [int(size) for size in variable.shape],
        "dtype": str(variable.dtype),
        "cell_location": normalized_attribute(variable, "cell_location"),
        "time_dimension": normalized_attribute(variable, "time_dimension"),
        "source": normalized_attribute(variable, "source"),
        "units": normalized_attribute(variable, "units"),
        "conversion_volts": normalized_attribute(variable, "conversion"),
    }
    for name in (
        "dimensions",
        "dtype",
        "cell_location",
        "time_dimension",
        "source",
        "units",
    ):
        if actual[name] != expected[name]:
            raise ValueError(f"phi metadata mismatch for {name}: {actual[name]!r}")
    if not math.isclose(
        float(actual["conversion_volts"]),
        float(expected["conversion_volts"]),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("phi conversion metadata mismatch")
    return actual


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        import netCDF4
    except ImportError as error:  # pragma: no cover - cluster dependency guard
        raise RuntimeError("netCDF4 is required for the boundary audit") from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["development_run"] != "85604":
        raise ValueError("boundary audit accepts development run 85604 only")
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
    frame_scope = manifest["frame_scope"]
    frame_count = int(frame_scope["frame_count"])
    expected_times = float(frame_scope["expected_first_normalized_time"]) + float(
        frame_scope["expected_normalized_cadence"]
    ) * np.arange(frame_count, dtype=np.float64)
    if expected_times[-1] != float(frame_scope["expected_last_normalized_time"]):
        raise ValueError("frozen time endpoints are inconsistent")
    native_z = int(archive["native_z_samples"])
    rank_dimensions = archive["rank_dimensions"]
    rank_scope = manifest["boundary_rank_scope"]
    local_y_first, local_y_stop = [int(value) for value in rank_scope["local_physical_y_slice"]]

    assembled = {
        side: {
            plane: np.full((frame_count, 32, native_z), np.nan, dtype=np.float64)
            for plane in boundary.PLANE_NAMES
        }
        for side in boundary.SIDES
    }
    selected_ranks: dict[str, list[int]] = {side: [] for side in boundary.SIDES}
    coordinate_coverage = np.zeros((decomposition["NXPE"], decomposition["NYPE"]), dtype=np.int64)
    selected_coverage = {
        side: np.zeros(32, dtype=np.int64) for side in boundary.SIDES
    }
    reference_phi_metadata: dict[str, Any] | None = None

    for path in paths:
        with netCDF4.Dataset(path, "r") as dataset:
            dimensions = {
                name: int(len(dimension)) for name, dimension in dataset.dimensions.items()
            }
            if dimensions != rank_dimensions:
                raise ValueError(f"rank dimensions mismatch in {path.name}")
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
            if not (
                0 <= pe_x < coordinate_coverage.shape[0]
                and 0 <= pe_y < coordinate_coverage.shape[1]
            ):
                raise ValueError(f"processor coordinate outside decomposition in {path.name}")
            coordinate_coverage[pe_x, pe_y] += 1
            side = None
            if pe_x == int(rank_scope["inner"]["PE_XIND"]):
                side = "inner"
            elif pe_x == int(rank_scope["outer"]["PE_XIND"]):
                side = "outer"
            if side is None:
                continue

            times = np.asarray(dataset.variables["t"][:], dtype=np.float64)
            if not np.allclose(times, expected_times, rtol=0.0, atol=1e-12):
                raise ValueError(f"time sequence mismatch in {path.name}")
            phi_variable = dataset.variables["phi"]
            metadata = validate_phi_metadata(phi_variable, manifest["phi_metadata"])
            if metadata["shape"] != [624, 8, 6, 81]:
                raise ValueError(f"phi shape mismatch in {path.name}")
            if reference_phi_metadata is None:
                reference_phi_metadata = metadata
            elif metadata != reference_phi_metadata:
                raise ValueError(f"phi metadata differ in {path.name}")
            phi = np.ma.filled(
                phi_variable[:, :, local_y_first:local_y_stop, :], np.nan
            ).astype(np.float64, copy=False)
            planes = boundary.extract_boundary_planes(
                phi, side=side, physical_y_slice=(0, local_y_stop - local_y_first)
            )
            y0 = pe_y * decomposition["MYSUB"]
            y1 = y0 + decomposition["MYSUB"]
            for plane, values in planes.items():
                assembled[side][plane][:, y0:y1, :] = values
            selected_coverage[side][y0:y1] += 1
            selected_ranks[side].append(rank_number(path))

    if not np.array_equal(coordinate_coverage, np.ones_like(coordinate_coverage)):
        raise ValueError("processor-coordinate coverage is incomplete or duplicated")
    if reference_phi_metadata is None:
        raise RuntimeError("no boundary phi metadata were read")
    for side in boundary.SIDES:
        if not np.array_equal(selected_coverage[side], np.ones(32, dtype=np.int64)):
            raise ValueError(f"global-y boundary coverage is incomplete for {side}")
        if len(selected_ranks[side]) != 16:
            raise ValueError(f"expected 16 selected ranks for {side}")
        for plane in boundary.PLANE_NAMES:
            if assembled[side][plane].shape != (frame_count, 32, native_z):
                raise ValueError("assembled boundary shape is inconsistent")

    blocks = [
        (int(first), int(last))
        for first, last in manifest["temporal_blocks"]["inclusive_index_ranges"]
    ]
    checks = manifest["exact_checks"]
    observables = manifest["observable_definitions"]
    per_side = {
        side: boundary.analyze_side(
            assembled[side],
            atol=float(checks["atol"]),
            rtol=float(checks["rtol"]),
            conversion_volts=float(manifest["phi_metadata"]["conversion_volts"]),
            percentiles=tuple(float(value) for value in observables["absolute_percentiles"]),
            temporal_blocks=blocks,
        )
        for side in boundary.SIDES
    }
    stream_digests = {
        side: {
            plane: sha256_array(f"{side}:{plane}", assembled[side][plane])
            for plane in boundary.PLANE_NAMES
        }
        for side in boundary.SIDES
    }
    result = {
        "schema_version": 1,
        "phase": "phase2_85604_phi_boundary_state_audit",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "audit_completed": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "raw_root": str(raw_root),
        "raw_control_digests": control_digests,
        "archive_rank_file_count": len(paths),
        "selected_rank_count": sum(len(values) for values in selected_ranks.values()),
        "selected_rank_indices_by_side": selected_ranks,
        "processor_coordinate_coverage_complete": True,
        "boundary_global_y_coverage_complete": True,
        "frame_count": frame_count,
        "normalized_time": {
            "first": float(expected_times[0]),
            "last": float(expected_times[-1]),
            "cadence": float(frame_scope["expected_normalized_cadence"]),
            "physical_cadence_microseconds": float(
                frame_scope["physical_cadence_microseconds"]
            ),
        },
        "zperiod": int(archive["zperiod"]),
        "native_z_samples": native_z,
        "phi_metadata": reference_phi_metadata,
        "source_boundary_policy": manifest["source_boundary_policy"],
        "stream_digests": stream_digests,
        "exact_tolerances": {"atol": float(checks["atol"]), "rtol": float(checks["rtol"])},
        "per_side": per_side,
        "scientific_findings": boundary.derive_findings(per_side),
    }
    strict_json_write(args.output, result)
    print(json.dumps(result["scientific_findings"], indent=2, sort_keys=True))
    print(f"Wrote boundary audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
