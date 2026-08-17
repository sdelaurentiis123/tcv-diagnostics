#!/usr/bin/env python3
"""Validate and compact the immutable all-frame 85604 resampling audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
COMPARISON_PATHS = ("round_trip", "direct_88", "raw64_vs_float32")
COMPARISON_CATEGORIES = ("face_xz", "face_xy", "face_total", "divergence_total")
PRIMARY_QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-job", required=True, type=int)
    parser.add_argument("--artifact-path", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def strict_json_write(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_exact_keys(name: str, value: dict[str, Any], expected: tuple[str, ...]) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{name} keys {sorted(value)} differ from expected {sorted(expected)}"
        )


def validate_source(
    raw: dict[str, Any], *, expected_commit: str, expected_job: int
) -> None:
    checks = {
        "schema_version": raw.get("schema_version") == 1,
        "phase": raw.get("phase")
        == "phase2_85604_native81_resampled88_sensitivity",
        "paper0_commit": raw.get("paper0_commit") == expected_commit,
        "slurm_job_id": int(raw.get("slurm_job_id", -1)) == expected_job,
        "audit_completed": raw.get("audit_completed") is True,
        "not_a_rank_shard": raw.get("rank_shard_completed") is False,
        "development_run": raw.get("development_run") == "85604",
        "held_out_excluded": raw.get("held_out_85606_read") is False,
        "frame_count": raw.get("frame_count") == 624,
        "frame_coverage": raw.get("frame_indices_complete_and_unique") is True,
        "native_shape": raw.get("native_shape_per_frame") == [64, 32, 81],
        "resampled_shape": raw.get("resampled_shape_per_frame") == [64, 32, 88],
        "zperiod": raw.get("zperiod") == 5,
        "all_shards_merged": raw.get("parallel_execution", {}).get(
            "all_shards_completed_before_merge"
        )
        is True,
        "shard_count": raw.get("parallel_execution", {}).get("shard_count") == 17,
        "manifest_digest": SHA256_PATTERN.fullmatch(
            str(raw.get("manifest_sha256", ""))
        )
        is not None,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"resampling source identity/integrity checks failed: {failed}")

    _require_exact_keys("field_round_trip", raw["field_round_trip"], FIELDS)
    _require_exact_keys("comparisons", raw["comparisons"], COMPARISON_PATHS)
    for comparison in COMPARISON_PATHS:
        _require_exact_keys(
            f"comparisons.{comparison}",
            raw["comparisons"][comparison],
            COMPARISON_CATEGORIES,
        )
        for category in COMPARISON_CATEGORIES:
            _require_exact_keys(
                f"comparisons.{comparison}.{category}",
                raw["comparisons"][comparison][category],
                PRIMARY_QUANTITIES,
            )


def _compact_field(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate": source["aggregate"],
        "aggregate_sufficient_statistics": source[
            "aggregate_sufficient_statistics"
        ],
        "per_frame_relative_l2_summary": source[
            "per_frame_relative_l2_summary"
        ],
        "relative_l2_by_temporal_block": source["relative_l2_by_temporal_block"],
    }


def _compact_comparison(source: dict[str, Any]) -> dict[str, Any]:
    frame_indices = [int(value) for value in source["frame_indices"]]
    if not frame_indices:
        raise ValueError("comparison contains no frame indices")
    return {
        "frame_coverage": {
            "count": len(frame_indices),
            "first": frame_indices[0],
            "last": frame_indices[-1],
            "strictly_increasing": all(
                right > left for left, right in zip(frame_indices, frame_indices[1:])
            ),
        },
        "aggregate": source["aggregate"],
        "aggregate_sufficient_statistics": source[
            "aggregate_sufficient_statistics"
        ],
        "toroidal_mean_profile_aggregate": source[
            "toroidal_mean_profile_aggregate"
        ],
        "toroidal_mean_profile_aggregate_sufficient_statistics": source[
            "toroidal_mean_profile_aggregate_sufficient_statistics"
        ],
        "per_frame_relative_l2_summary": source[
            "per_frame_relative_l2_summary"
        ],
        "per_frame_toroidal_mean_profile_relative_l2_summary": source[
            "per_frame_toroidal_mean_profile_relative_l2_summary"
        ],
        "per_frame_absolute_value_p95_ratio_summary": source[
            "per_frame_absolute_value_p95_ratio_summary"
        ],
        "per_frame_absolute_value_p99_ratio_summary": source[
            "per_frame_absolute_value_p99_ratio_summary"
        ],
        "relative_l2_by_temporal_block": source["relative_l2_by_temporal_block"],
    }


def compact_record(
    raw: dict[str, Any], *, artifact_path: str, digest: str, size_bytes: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "result_type": "phase2_85604_resampling_sensitivity_compact",
        "execution": {
            "paper0_commit": raw["paper0_commit"],
            "slurm_job_id": raw["slurm_job_id"],
        },
        "data_scope": {
            "development_run": raw["development_run"],
            "held_out_85606_read": raw["held_out_85606_read"],
            "frame_count": raw["frame_count"],
            "frame_indices_complete_and_unique": raw[
                "frame_indices_complete_and_unique"
            ],
            "native_shape_per_frame": raw["native_shape_per_frame"],
            "resampled_shape_per_frame": raw["resampled_shape_per_frame"],
            "zperiod": raw["zperiod"],
        },
        "integrity": {
            "manifest_path": raw["manifest"],
            "manifest_sha256": raw["manifest_sha256"],
            "field_stream_digest_tree": raw["field_stream_digest_tree"],
            "shard_count": raw["parallel_execution"]["shard_count"],
            "chunk_aligned_intervals": raw["parallel_execution"][
                "chunk_aligned_intervals"
            ],
            "all_shards_completed_before_merge": raw["parallel_execution"][
                "all_shards_completed_before_merge"
            ],
            "full_result": {
                "path": artifact_path,
                "sha256": digest,
                "size_bytes": size_bytes,
                "tracked_in_git": False,
            },
        },
        "structural_checks": raw["structural_checks"],
        "field_round_trip": {
            field: _compact_field(raw["field_round_trip"][field])
            for field in FIELDS
        },
        "transport_comparisons": {
            comparison: {
                category: {
                    quantity: _compact_comparison(
                        raw["comparisons"][comparison][category][quantity]
                    )
                    for quantity in PRIMARY_QUANTITIES
                }
                for category in COMPARISON_CATEGORIES
            }
            for comparison in COMPARISON_PATHS
        },
        "acceptance": raw["acceptance"],
        "scientific_findings": raw["scientific_findings"],
    }


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if not SHA256_PATTERN.fullmatch(args.expected_sha256):
        raise ValueError("expected SHA-256 must contain exactly 64 lowercase hex digits")
    actual_digest = sha256_file(source)
    if actual_digest != args.expected_sha256:
        raise ValueError(
            f"input SHA-256 {actual_digest} != expected {args.expected_sha256}"
        )
    raw = strict_json_load(source)
    validate_source(
        raw, expected_commit=args.expected_commit, expected_job=args.expected_job
    )
    record = compact_record(
        raw,
        artifact_path=args.artifact_path,
        digest=actual_digest,
        size_bytes=source.stat().st_size,
    )
    strict_json_write(output, record)
    print(f"wrote compact resampling result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
