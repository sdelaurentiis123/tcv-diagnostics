#!/usr/bin/env python3
"""Merge complete deterministic shards of the frozen 85604 state audit."""

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


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def strict_json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def strict_json_write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale temporary file {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sum_lists(records: list[list[int]]) -> list[int]:
    arrays = [np.asarray(record, dtype=np.int64) for record in records]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("integer lists must be nonempty and shape-compatible")
    return np.sum(arrays, axis=0, dtype=np.int64).astype(int).tolist()


def select_point(
    records: list[dict[str, Any] | None], *, key: str, minimum: bool
) -> dict[str, Any] | None:
    available = [record for record in records if record is not None]
    if not available:
        return None
    ordering = lambda record: (float(record[key]), record["location_txyz"])
    return dict(min(available, key=ordering) if minimum else max(available, key=ordering))


def relative_l2(sum_squared_error: float, sum_squared_reference: float) -> float | None:
    if sum_squared_reference > 0.0:
        return math.sqrt(sum_squared_error / sum_squared_reference)
    if sum_squared_error == 0.0:
        return 0.0
    return None


def merge_field_scope(scopes: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(scope["total_count"]) for scope in scopes)
    finite = sum(int(scope["finite_count"]) for scope in scopes)
    nonfinite = sum(int(scope["nonfinite_count"]) for scope in scopes)
    if finite + nonfinite != total:
        raise ValueError("field finite accounting is inconsistent")
    sum_value = sum(float(scope["sum"]) for scope in scopes)
    sum_squares = sum(float(scope["sum_squares"]) for scope in scopes)
    return {
        "total_count": total,
        "finite_count": finite,
        "nonfinite_count": nonfinite,
        "sum": sum_value,
        "sum_squares": sum_squares,
        "rms": math.sqrt(sum_squares / finite) if finite else None,
        "minimum": select_point(
            [scope["minimum"] for scope in scopes], key="value", minimum=True
        ),
        "maximum": select_point(
            [scope["maximum"] for scope in scopes], key="value", minimum=False
        ),
    }


def merge_field(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scopes": {
            name: merge_field_scope([record["scopes"][name] for record in records])
            for name in state.SCOPE_NAMES
        }
    }


def merge_density_floor(records: list[dict[str, Any]]) -> dict[str, Any]:
    density_floor = float(records[0]["density_floor"])
    if any(float(record["density_floor"]) != density_floor for record in records):
        raise ValueError("density-floor values differ across shards")
    result: dict[str, Any] = {
        "density_floor": density_floor,
        "scope_counts": {
            scope: {
                event: sum(
                    int(record["scope_counts"][scope][event]) for record in records
                )
                for event in state.DensityFloorAccumulator.EVENT_NAMES
            }
            for scope in state.SCOPE_NAMES
        },
    }
    for output_key, input_key in (
        ("count_by_frame", "count_by_frame"),
        ("count_by_x", "count_by_x"),
        ("count_by_y", "count_by_y"),
        ("count_by_temporal_block", "count_by_temporal_block"),
    ):
        result[output_key] = {
            event: sum_lists([record[input_key][event] for record in records])
            for event in state.DensityFloorAccumulator.EVENT_NAMES
        }
    return result


def merge_closure_scope(
    scopes: list[dict[str, Any]], *, atol: float, rtol: float
) -> dict[str, Any]:
    frame_max_error = np.max(
        np.asarray([scope["frame_max_abs_error"] for scope in scopes]), axis=0
    )
    frame_max_reference = np.max(
        np.asarray([scope["frame_max_abs_reference"] for scope in scopes]), axis=0
    )
    frame_nonfinite = np.sum(
        np.asarray([scope["frame_nonfinite_count"] for scope in scopes], dtype=np.int64),
        axis=0,
    )
    frame_discrepancy = np.sum(
        np.asarray(
            [scope["frame_point_discrepancy_count"] for scope in scopes],
            dtype=np.int64,
        ),
        axis=0,
    )
    tolerance = atol + rtol * frame_max_reference
    passed = (frame_nonfinite == 0) & (frame_max_error <= tolerance)
    sum_squared_error = sum(float(scope["sum_squared_error"]) for scope in scopes)
    sum_squared_reference = sum(
        float(scope["sum_squared_reference"]) for scope in scopes
    )
    return {
        "total_count": sum(int(scope["total_count"]) for scope in scopes),
        "nonfinite_count": sum(int(scope["nonfinite_count"]) for scope in scopes),
        "point_discrepancy_count": sum(
            int(scope["point_discrepancy_count"]) for scope in scopes
        ),
        "sum_squared_error": sum_squared_error,
        "sum_squared_reference": sum_squared_reference,
        "relative_l2_error": relative_l2(sum_squared_error, sum_squared_reference),
        "maximum_error": select_point(
            [scope["maximum_error"] for scope in scopes],
            key="absolute_error",
            minimum=False,
        ),
        "frame_pass_count": int(np.count_nonzero(passed)),
        "frame_fail_count": int(np.count_nonzero(~passed)),
        "failed_frame_indices": np.flatnonzero(~passed).astype(int).tolist(),
        "frame_max_abs_error": frame_max_error.tolist(),
        "frame_max_abs_reference": frame_max_reference.tolist(),
        "frame_tolerance": tolerance.tolist(),
        "frame_passed": passed.tolist(),
        "frame_nonfinite_count": frame_nonfinite.astype(int).tolist(),
        "frame_point_discrepancy_count": frame_discrepancy.astype(int).tolist(),
    }


def merge_relation(
    records: list[dict[str, Any]],
    *,
    temporal_blocks: list[tuple[int, int]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    discrepancy_by_frame = sum_lists(
        [record["point_discrepancy_count_by_frame"] for record in records]
    )
    return {
        "scopes": {
            name: merge_closure_scope(
                [record["scopes"][name] for record in records],
                atol=atol,
                rtol=rtol,
            )
            for name in state.SCOPE_NAMES
        },
        "point_discrepancy_count_by_frame": discrepancy_by_frame,
        "point_discrepancy_count_by_x": sum_lists(
            [record["point_discrepancy_count_by_x"] for record in records]
        ),
        "point_discrepancy_count_by_y": sum_lists(
            [record["point_discrepancy_count_by_y"] for record in records]
        ),
        "point_discrepancy_count_by_temporal_block": [
            int(sum(discrepancy_by_frame[first : last + 1]))
            for first, last in temporal_blocks
        ],
    }


def digest_tree(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partial-output", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.output.parent.resolve() != args.artifact_root.resolve():
        raise ValueError("merged output must live directly in artifact root")
    manifest = strict_json_load(args.manifest)
    if manifest["development_run"] != "85604" or manifest[
        "held_out_85606_access_allowed"
    ]:
        raise ValueError("merger requires frozen 85604-only manifest")
    manifest_digest = sha256_file(args.manifest)
    pairs = [(path, strict_json_load(path)) for path in args.partial_output]
    pairs.sort(key=lambda pair: int(pair[1]["rank_shard"]["index"]))
    partials = [partial for _, partial in pairs]
    if not partials:
        raise ValueError("no partial outputs supplied")

    shard_count = int(partials[0]["rank_shard"]["count"])
    if shard_count != 16 or len(partials) != shard_count:
        raise ValueError("frozen execution requires exactly 16 shards")
    if [int(partial["rank_shard"]["index"]) for partial in partials] != list(
        range(shard_count)
    ):
        raise ValueError("shard indices must be contiguous and zero based")

    common_keys = (
        "development_run",
        "raw_root",
        "raw_control_digests",
        "archive_rank_file_count",
        "frame_count",
        "normalized_time",
        "native_z_samples",
        "zperiod",
        "shape_per_field",
        "variable_metadata",
    )
    common = {key: partials[0][key] for key in common_keys}
    rank_indices: list[int] = []
    coordinates: list[tuple[int, int]] = []
    for partial in partials:
        shard = partial["rank_shard"]
        if partial["phase"] != "phase2_85604_state_completeness_rank_shard":
            raise ValueError("unexpected partial phase")
        if partial["audit_completed"] or not partial["rank_shard_completed"]:
            raise ValueError("invalid partial completion semantics")
        if "scientific_findings" in partial:
            raise ValueError("rank shards cannot contain scientific findings")
        if partial["paper0_commit"] != args.paper0_commit:
            raise ValueError("partial commit mismatch")
        if int(partial["slurm_job_id"]) != args.slurm_job_id:
            raise ValueError("partial job mismatch")
        if partial["held_out_85606_read"]:
            raise ValueError("partial reports held-out access")
        if partial["manifest_sha256"] != manifest_digest:
            raise ValueError("partial manifest mismatch")
        for key, expected in common.items():
            if partial[key] != expected:
                raise ValueError(f"partial common field differs: {key}")
        index = int(shard["index"])
        if int(shard["count"]) != shard_count:
            raise ValueError("partial shard-count mismatch")
        expected_ranks = [rank for rank in range(256) if rank % shard_count == index]
        if partial["rank_indices"] != expected_ranks:
            raise ValueError("partial ranks violate modulo partition")
        if partial["rank_file_count"] != len(expected_ranks):
            raise ValueError("partial rank count mismatch")
        if partial["processor_coverage"]["complete"]:
            raise ValueError("individual shard cannot claim full processor coverage")
        stream_digests = partial["guard_stripped_rank_stream_digests"]
        if set(stream_digests) != set(state.STREAM_FIELDS):
            raise ValueError("partial stream fields disagree")
        if any(SHA256_PATTERN.fullmatch(value) is None for value in stream_digests.values()):
            raise ValueError("malformed stream digest")
        rank_indices.extend(int(rank) for rank in partial["rank_indices"])
        coordinates.extend(
            tuple(int(value) for value in coordinate)
            for coordinate in partial["processor_coverage"]["coordinates"]
        )

    if sorted(rank_indices) != list(range(256)):
        raise ValueError("merged rank coverage is incomplete or duplicated")
    nxpe = int(manifest["raw_archive"]["mpi_decomposition"]["NXPE"])
    nype = int(manifest["raw_archive"]["mpi_decomposition"]["NYPE"])
    expected_coordinates = {(x, y) for x in range(nxpe) for y in range(nype)}
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected_coordinates:
        raise ValueError("merged processor-coordinate coverage is incomplete or duplicated")

    atol = float(manifest["closure_statistics"]["atol"])
    rtol = float(manifest["closure_statistics"]["rtol"])
    blocks = [
        (int(first), int(last))
        for first, last in manifest["temporal_blocks"]["inclusive_index_ranges"]
    ]
    field_statistics = {
        field: merge_field([partial["field_statistics"][field] for partial in partials])
        for field in state.STREAM_FIELDS
    }
    density_floor_statistics = merge_density_floor(
        [partial["density_floor_statistics"] for partial in partials]
    )
    closures = {
        relation: merge_relation(
            [partial["closure_statistics"]["relations"][relation] for partial in partials],
            temporal_blocks=blocks,
            atol=atol,
            rtol=rtol,
        )
        for relation in state.RELATIONS
    }
    frame_count, nx, ny, native_z = [
        int(value) for value in manifest["canonical_cells"]["shape_per_field"]
    ]
    expected_scope_counts = {
        "full_physical_domain": frame_count * nx * ny * native_z,
        "guard_independent_transport_interior": frame_count * nx * 30 * native_z,
        "target_dependent_rows": frame_count * nx * 2 * native_z,
    }
    for field in state.STREAM_FIELDS:
        for scope, expected in expected_scope_counts.items():
            if field_statistics[field]["scopes"][scope]["total_count"] != expected:
                raise ValueError(f"incomplete merged field accounting for {field}/{scope}")
    for relation in state.RELATIONS:
        for scope, expected in expected_scope_counts.items():
            if closures[relation]["scopes"][scope]["total_count"] != expected:
                raise ValueError(
                    f"incomplete merged closure accounting for {relation}/{scope}"
                )

    partial_records = []
    for path, partial in pairs:
        partial_records.append(
            {
                "shard_index": int(partial["rank_shard"]["index"]),
                "rank_indices": partial["rank_indices"],
                "path": str(path),
                "partial_file_sha256": sha256_file(path),
                "field_stream_digests": partial[
                    "guard_stripped_rank_stream_digests"
                ],
            }
        )
    stream_digest_tree = {
        field: digest_tree(
            [
                {
                    "shard_index": record["shard_index"],
                    "rank_indices": record["rank_indices"],
                    "stream_sha256": record["field_stream_digests"][field],
                }
                for record in partial_records
            ]
        )
        for field in state.STREAM_FIELDS
    }
    result = {
        "schema_version": 1,
        "phase": "phase2_85604_state_completeness_audit",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "audit_completed": True,
        "rank_shard_completed": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_digest,
        "raw_root": common["raw_root"],
        "raw_control_digests": common["raw_control_digests"],
        "archive_rank_file_count": 256,
        "rank_file_count": 256,
        "rank_indices": list(range(256)),
        "rank_shards": partial_records,
        "rank_shard_digest_tree_sha256": digest_tree(partial_records),
        "processor_coverage": {
            "NXPE": nxpe,
            "NYPE": nype,
            "unique_coordinates": 256,
            "complete": True,
        },
        "frame_count": common["frame_count"],
        "normalized_time": common["normalized_time"],
        "native_z_samples": common["native_z_samples"],
        "zperiod": common["zperiod"],
        "shape_per_field": common["shape_per_field"],
        "total_points_per_stream": expected_scope_counts["full_physical_domain"],
        "expected_scope_counts": expected_scope_counts,
        "variable_metadata": common["variable_metadata"],
        "guard_stripped_stream_digest_tree": stream_digest_tree,
        "field_statistics": field_statistics,
        "density_floor_statistics": density_floor_statistics,
        "closure_statistics": {"atol": atol, "rtol": rtol, "relations": closures},
        "scientific_findings": state.derive_findings(field_statistics, closures),
    }
    strict_json_write(args.output, result)
    print(json.dumps(result["scientific_findings"], indent=2, sort_keys=True))
    print(f"Wrote complete audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
