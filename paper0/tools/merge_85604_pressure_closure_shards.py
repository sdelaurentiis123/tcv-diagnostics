#!/usr/bin/env python3
"""Merge complete disjoint 85604 pressure-closure rank shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

import audit_85604_pressure_closure as audit


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ),
    )


def sum_integer_lists(values: list[list[int]]) -> list[int]:
    if not values:
        raise ValueError("cannot sum an empty list collection")
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError("integer-list lengths disagree across shards")
    return np.sum(np.asarray(values, dtype=np.int64), axis=0).astype(int).tolist()


def _minimum_record(records: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    finite = [record for record in records if record is not None]
    if not finite:
        return None
    return min(finite, key=lambda item: (item["value"], item["location_txyz"]))


def _maximum_record(records: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    finite = [record for record in records if record is not None]
    if not finite:
        return None
    return max(finite, key=lambda item: (item["value"], tuple(-v for v in item["location_txyz"])))


def _maximum_error_record(
    records: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    finite = [record for record in records if record is not None]
    if not finite:
        return None
    return max(
        finite,
        key=lambda item: (
            item["absolute_error"],
            tuple(-v for v in item["location_txyz"]),
        ),
    )


def merge_value_scope(scopes: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = sum(int(scope["total_count"]) for scope in scopes)
    if total_count <= 0:
        raise ValueError("merged value scope is empty")
    nonfinite_count = sum(int(scope["nonfinite_count"]) for scope in scopes)
    negative_count = sum(int(scope["negative_count"]) for scope in scopes)
    zero_count = sum(int(scope["zero_count"]) for scope in scopes)
    most_negative = [
        point
        for scope in scopes
        for point in scope["most_negative_points"]
    ]
    most_negative.sort(key=lambda item: (item["value"], item["location_txyz"]))
    return {
        "total_count": total_count,
        "nonfinite_count": nonfinite_count,
        "nonfinite_fraction": nonfinite_count / total_count,
        "negative_count": negative_count,
        "negative_fraction": negative_count / total_count,
        "zero_count": zero_count,
        "zero_fraction": zero_count / total_count,
        "minimum": _minimum_record([scope["minimum"] for scope in scopes]),
        "maximum": _maximum_record([scope["maximum"] for scope in scopes]),
        "most_negative_points": most_negative[:20],
    }


def merge_value_field(
    fields: list[dict[str, Any]], temporal_blocks: list[tuple[int, int]]
) -> dict[str, Any]:
    negative_by_frame = sum_integer_lists(
        [field["negative_count_by_frame"] for field in fields]
    )
    return {
        "scopes": {
            scope_name: merge_value_scope(
                [field["scopes"][scope_name] for field in fields]
            )
            for scope_name in audit.ValueAccumulator.SCOPE_NAMES
        },
        "negative_count_by_frame": negative_by_frame,
        "negative_count_by_x": sum_integer_lists(
            [field["negative_count_by_x"] for field in fields]
        ),
        "negative_count_by_y": sum_integer_lists(
            [field["negative_count_by_y"] for field in fields]
        ),
        "negative_count_by_temporal_block": [
            int(sum(negative_by_frame[first : last + 1]))
            for first, last in temporal_blocks
        ],
    }


def merge_closure_scope(
    scopes: list[dict[str, Any]], *, atol: float, rtol: float
) -> dict[str, Any]:
    frame_max_abs_error = np.max(
        np.asarray([scope["frame_max_abs_error"] for scope in scopes]), axis=0
    )
    frame_max_abs_reference = np.max(
        np.asarray([scope["frame_max_abs_reference"] for scope in scopes]), axis=0
    )
    frame_nonfinite_count = np.sum(
        np.asarray([scope["frame_nonfinite_count"] for scope in scopes], dtype=np.int64),
        axis=0,
    )
    frame_point_discrepancy_count = np.sum(
        np.asarray(
            [scope["frame_point_discrepancy_count"] for scope in scopes],
            dtype=np.int64,
        ),
        axis=0,
    )
    tolerance = atol + rtol * frame_max_abs_reference
    passed = (frame_nonfinite_count == 0) & (frame_max_abs_error <= tolerance)
    return {
        "total_count": sum(int(scope["total_count"]) for scope in scopes),
        "nonfinite_count": sum(int(scope["nonfinite_count"]) for scope in scopes),
        "point_discrepancy_count": sum(
            int(scope["point_discrepancy_count"]) for scope in scopes
        ),
        "negative_reference_discrepancy_count": sum(
            int(scope["negative_reference_discrepancy_count"]) for scope in scopes
        ),
        "nonnegative_reference_discrepancy_count": sum(
            int(scope["nonnegative_reference_discrepancy_count"]) for scope in scopes
        ),
        "maximum_error": _maximum_error_record(
            [scope["maximum_error"] for scope in scopes]
        ),
        "frame_pass_count": int(np.count_nonzero(passed)),
        "frame_fail_count": int(np.count_nonzero(~passed)),
        "failed_frame_indices": np.flatnonzero(~passed).astype(int).tolist(),
        "frame_max_abs_error": frame_max_abs_error.tolist(),
        "frame_max_abs_reference": frame_max_abs_reference.tolist(),
        "frame_tolerance": tolerance.tolist(),
        "frame_passed": passed.tolist(),
        "frame_nonfinite_count": frame_nonfinite_count.astype(int).tolist(),
        "frame_point_discrepancy_count": frame_point_discrepancy_count.astype(int).tolist(),
    }


def merge_closure_relation(
    relations: list[dict[str, Any]],
    *,
    temporal_blocks: list[tuple[int, int]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    discrepancy_by_frame = sum_integer_lists(
        [relation["point_discrepancy_count_by_frame"] for relation in relations]
    )
    return {
        "scopes": {
            scope_name: merge_closure_scope(
                [relation["scopes"][scope_name] for relation in relations],
                atol=atol,
                rtol=rtol,
            )
            for scope_name in audit.ClosureAccumulator.SCOPE_NAMES
        },
        "point_discrepancy_count_by_frame": discrepancy_by_frame,
        "point_discrepancy_count_by_x": sum_integer_lists(
            [relation["point_discrepancy_count_by_x"] for relation in relations]
        ),
        "point_discrepancy_count_by_y": sum_integer_lists(
            [relation["point_discrepancy_count_by_y"] for relation in relations]
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
    parser = argparse.ArgumentParser()
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
        raise ValueError("merged output must live directly in the declared artifact root")
    manifest = strict_json_load(args.manifest)
    if manifest["development_run"] != "85604" or manifest[
        "held_out_85606_access_allowed"
    ]:
        raise ValueError("merger requires the frozen 85604-only manifest")
    manifest_digest = audit.sha256_file(args.manifest)
    partial_pairs = [(path, strict_json_load(path)) for path in args.partial_output]
    partial_pairs.sort(key=lambda pair: pair[1]["rank_shard"]["index"])
    partials = [partial for _, partial in partial_pairs]
    if not partials:
        raise ValueError("no partial outputs supplied")

    shard_count = int(partials[0]["rank_shard"]["count"])
    if shard_count != 16:
        raise ValueError("corrected execution is frozen to exactly 16 shards")
    if shard_count != len(partials):
        raise ValueError("partial count does not match declared shard count")
    if [partial["rank_shard"]["index"] for partial in partials] != list(
        range(shard_count)
    ):
        raise ValueError("shard indices must cover a contiguous zero-based range")

    rank_indices: list[int] = []
    coordinates: list[tuple[int, int]] = []
    reference_metadata = partials[0]["variable_metadata"]
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
    common_values = {key: partials[0][key] for key in common_keys}
    for partial in partials:
        shard = partial["rank_shard"]
        if partial["phase"] != "phase2_85604_pressure_closure_rank_shard":
            raise ValueError("unexpected partial phase")
        if partial["audit_completed"] or not partial["rank_shard_completed"]:
            raise ValueError("partial completion semantics are invalid")
        if "scientific_findings" in partial:
            raise ValueError("rank shard must not contain scientific findings")
        if partial["paper0_commit"] != args.paper0_commit:
            raise ValueError("partial commit mismatch")
        if int(partial["slurm_job_id"]) != args.slurm_job_id:
            raise ValueError("partial Slurm job mismatch")
        if partial["held_out_85606_read"]:
            raise ValueError("partial reports held-out access")
        for key, expected in common_values.items():
            if partial[key] != expected:
                raise ValueError(f"partial common field differs: {key}")
        if partial["manifest_sha256"] != manifest_digest:
            raise ValueError("partial manifest mismatch")
        if int(shard["count"]) != shard_count:
            raise ValueError("shard-count mismatch")
        expected_ranks = [
            rank
            for rank in range(int(partial["archive_rank_file_count"]))
            if rank % shard_count == int(shard["index"])
        ]
        if partial["rank_indices"] != expected_ranks:
            raise ValueError("partial rank indices violate modulo partition")
        if partial["rank_file_count"] != len(expected_ranks):
            raise ValueError("partial rank count mismatch")
        if partial["processor_coverage"]["complete"]:
            raise ValueError("individual rank shard cannot claim complete coverage")
        if partial["processor_coverage"]["unique_coordinates"] != len(
            expected_ranks
        ):
            raise ValueError("partial processor-coordinate count mismatch")
        if partial["variable_metadata"] != reference_metadata:
            raise ValueError("variable metadata differ across shards")
        field_digests = partial["guard_stripped_rank_stream_digests"]
        if set(field_digests) != set(audit.FIELD_NAMES):
            raise ValueError("partial field-stream digest keys disagree")
        if any(SHA256_PATTERN.fullmatch(value) is None for value in field_digests.values()):
            raise ValueError("partial field-stream digest is malformed")
        rank_indices.extend(int(rank) for rank in partial["rank_indices"])
        coordinates.extend(
            tuple(int(item) for item in coordinate)
            for coordinate in partial["processor_coverage"]["coordinates"]
        )

    expected_rank_count = int(manifest["raw_archive"]["expected_rank_file_count"])
    if expected_rank_count != 256:
        raise ValueError("frozen archive must contain exactly 256 ranks")
    if sorted(rank_indices) != list(range(expected_rank_count)):
        raise ValueError("merged rank coverage is incomplete or duplicated")
    nxpe = int(manifest["raw_archive"]["mpi_decomposition"]["NXPE"])
    nype = int(manifest["raw_archive"]["mpi_decomposition"]["NYPE"])
    expected_coordinates = {(pe_x, pe_y) for pe_x in range(nxpe) for pe_y in range(nype)}
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected_coordinates:
        raise ValueError("merged processor-coordinate coverage is incomplete or duplicated")

    temporal_blocks = [
        (int(first), int(last))
        for first, last in manifest["temporal_blocks"]["inclusive_index_ranges"]
    ]
    atol = float(manifest["closure_statistics"]["atol"])
    rtol = float(manifest["closure_statistics"]["rtol"])
    value_results = {
        field: merge_value_field(
            [partial["value_statistics"][field] for partial in partials],
            temporal_blocks,
        )
        for field in audit.FIELD_NAMES
    }
    closure_results = {
        relation: merge_closure_relation(
            [
                partial["closure_statistics"]["relations"][relation]
                for partial in partials
            ],
            temporal_blocks=temporal_blocks,
            atol=atol,
            rtol=rtol,
        )
        for relation in audit.RELATIONS
    }

    frame_count, nx, ny, native_z = [
        int(value) for value in manifest["canonical_cells"]["shape_per_field"]
    ]
    expected_scope_counts = {
        "full_physical_domain": frame_count * nx * ny * native_z,
        "guard_independent_transport_interior": frame_count * nx * 30 * native_z,
        "target_dependent_rows": frame_count * nx * 2 * native_z,
    }
    audit.validate_scope_accounting(
        value_results, closure_results, expected_scope_counts
    )
    scientific_findings = audit.derive_scientific_findings(
        value_results, closure_results
    )

    partial_records = []
    for path, partial in partial_pairs:
        partial_records.append(
            {
                "shard_index": int(partial["rank_shard"]["index"]),
                "rank_indices": partial["rank_indices"],
                "path": str(path),
                "partial_file_sha256": audit.sha256_file(path),
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
        for field in audit.FIELD_NAMES
    }

    first = partials[0]
    result = {
        "schema_version": 1,
        "phase": "phase2_85604_all_frame_pressure_closure_audit",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "audit_completed": True,
        "rank_shard_completed": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_digest,
        "raw_root": first["raw_root"],
        "raw_control_digests": first["raw_control_digests"],
        "rank_file_count": expected_rank_count,
        "processor_coverage": {
            "NXPE": nxpe,
            "NYPE": nype,
            "unique_coordinates": len(expected_coordinates),
            "complete": True,
        },
        "frame_count": frame_count,
        "normalized_time": first["normalized_time"],
        "native_z_samples": native_z,
        "zperiod": int(first["zperiod"]),
        "shape_per_field": [frame_count, nx, ny, native_z],
        "total_points_per_field": expected_scope_counts["full_physical_domain"],
        "expected_scope_counts": expected_scope_counts,
        "variable_metadata": reference_metadata,
        "parallel_execution": {
            "shard_count": shard_count,
            "partition_rule": "rank modulo shard_count equals shard_index",
            "all_shards_completed_before_merge": True,
            "artifact_root": str(args.artifact_root),
            "partials": partial_records,
        },
        "guard_stripped_rank_stream_digest_tree": stream_digest_tree,
        "value_statistics": value_results,
        "closure_statistics": {
            "atol": atol,
            "rtol": rtol,
            "relations": closure_results,
        },
        "scientific_findings": scientific_findings,
    }
    audit.strict_json_write(args.output, result)
    print(json.dumps(scientific_findings, indent=2, sort_keys=True))
    print(f"Wrote merged complete audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
