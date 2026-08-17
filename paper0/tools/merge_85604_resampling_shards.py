#!/usr/bin/env python3
"""Strictly merge complete 85604 native/resampled frame shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_85604_resampling as audit  # noqa: E402
from tcv_diagnostics.resampling import (  # noqa: E402
    finalize_paired_statistics,
    linear_quantile,
    materiality_label,
    merge_paired_sufficient_statistics,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ),
    )


def digest_records(records: Iterable[dict[str, Any]]) -> str:
    encoded = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def scalar_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = [float(value) for value in values]
    if not array:
        raise ValueError("cannot summarize an empty scalar collection")
    return {
        "count": len(array),
        "minimum": min(array),
        "median": linear_quantile(array, 0.5),
        "p95": linear_quantile(array, 0.95),
        "p99": linear_quantile(array, 0.99),
        "maximum": max(array),
        "mean": sum(array) / len(array),
    }


def temporal_block_summaries(
    frame_indices: list[int],
    values: list[float],
    blocks: list[list[int]],
) -> list[dict[str, Any]]:
    if len(frame_indices) != len(values):
        raise ValueError("frame indices and values have different lengths")
    summaries = []
    for first, last in blocks:
        selected = [
            value
            for frame, value in zip(frame_indices, values)
            if first <= frame <= last
        ]
        summaries.append(
            {
                "first_frame": first,
                "last_frame": last,
                **scalar_summary(selected),
            }
        )
    return summaries


def merge_field_round_trip(
    frame_records: list[dict[str, Any]],
    field: str,
    blocks: list[list[int]],
) -> dict[str, Any]:
    frame_indices = [int(record["frame_index"]) for record in frame_records]
    records = [record["field_round_trip"][field] for record in frame_records]
    relative = [float(record["relative_l2"]) for record in records]
    sufficient = merge_paired_sufficient_statistics(
        [record["sufficient_statistics"] for record in records]
    )
    return {
        "aggregate": finalize_paired_statistics(sufficient),
        "aggregate_sufficient_statistics": sufficient,
        "per_frame_relative_l2": relative,
        "per_frame_relative_l2_summary": scalar_summary(relative),
        "relative_l2_by_temporal_block": temporal_block_summaries(
            frame_indices, relative, blocks
        ),
    }


def merge_comparison_leaf(
    frame_records: list[dict[str, Any]],
    *,
    path: str,
    category: str,
    quantity: str,
    blocks: list[list[int]],
) -> dict[str, Any]:
    pairs = [
        (int(record["frame_index"]), record["comparisons"][path])
        for record in frame_records
        if record["comparisons"][path] is not None
    ]
    if not pairs:
        raise ValueError(f"comparison path {path} has no frames")
    frames = [frame for frame, _ in pairs]
    leaves = [comparison[category][quantity] for _, comparison in pairs]
    sufficient = merge_paired_sufficient_statistics(
        [leaf["sufficient_statistics"] for leaf in leaves]
    )
    profile_sufficient = merge_paired_sufficient_statistics(
        [leaf["toroidal_mean_profile_sufficient_statistics"] for leaf in leaves]
    )
    relative = [float(leaf["metrics"]["relative_l2"]) for leaf in leaves]
    profile_relative = [
        float(leaf["toroidal_mean_profile_relative_l2"]) for leaf in leaves
    ]
    p95_ratio = [float(leaf["absolute_value_p95_ratio"]) for leaf in leaves]
    p99_ratio = [float(leaf["absolute_value_p99_ratio"]) for leaf in leaves]
    result = {
        "frame_indices": frames,
        "aggregate": finalize_paired_statistics(sufficient),
        "aggregate_sufficient_statistics": sufficient,
        "toroidal_mean_profile_aggregate": finalize_paired_statistics(
            profile_sufficient
        ),
        "toroidal_mean_profile_aggregate_sufficient_statistics": profile_sufficient,
        "per_frame_relative_l2": relative,
        "per_frame_relative_l2_summary": scalar_summary(relative),
        "per_frame_toroidal_mean_profile_relative_l2_summary": scalar_summary(
            profile_relative
        ),
        "per_frame_absolute_value_p95_ratio_summary": scalar_summary(p95_ratio),
        "per_frame_absolute_value_p99_ratio_summary": scalar_summary(p99_ratio),
    }
    if len(frames) == 624:
        result["relative_l2_by_temporal_block"] = temporal_block_summaries(
            frames, relative, blocks
        )
    return result


def merge_comparison_path(
    frame_records: list[dict[str, Any]],
    *,
    path: str,
    blocks: list[list[int]],
) -> dict[str, Any]:
    return {
        category: {
            quantity: merge_comparison_leaf(
                frame_records,
                path=path,
                category=category,
                quantity=quantity,
                blocks=blocks,
            )
            for quantity in audit.PRIMARY_QUANTITIES
        }
        for category in audit.COMPARISON_CATEGORIES
    }


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
        raise ValueError("merged output must live in the artifact root")
    manifest = strict_json_load(args.manifest)
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
    ):
        raise ValueError("merger requires the frozen 85604-only manifest")
    manifest_digest = sha256_file(args.manifest)
    partial_pairs = [(path, strict_json_load(path)) for path in args.partial_output]
    partial_pairs.sort(key=lambda pair: int(pair[1]["shard"]["index"]))
    partials = [partial for _, partial in partial_pairs]
    expected_shards = len(audit.SHARD_INTERVALS)
    if len(partials) != expected_shards:
        raise ValueError(f"expected {expected_shards} partials")

    frame_records: list[dict[str, Any]] = []
    partial_artifacts: list[dict[str, Any]] = []
    for expected_index, ((path, partial), expected_interval) in enumerate(
        zip(partial_pairs, audit.SHARD_INTERVALS)
    ):
        if partial["phase"] != "phase2_85604_resampling_rank_shard":
            raise ValueError("unexpected partial phase")
        if not partial["rank_shard_completed"] or partial["audit_completed"]:
            raise ValueError("partial completion semantics are invalid")
        if "scientific_findings" in partial:
            raise ValueError("rank shard may not contain a scientific conclusion")
        if partial["paper0_commit"] != args.paper0_commit:
            raise ValueError("partial commit differs from merger commit")
        if int(partial["slurm_job_id"]) != args.slurm_job_id:
            raise ValueError("partial Slurm job differs from merger job")
        if partial["held_out_85606_read"]:
            raise ValueError("partial reports held-out access")
        if partial["manifest_sha256"] != manifest_digest:
            raise ValueError("partial manifest digest differs")
        shard = partial["shard"]
        interval = (
            int(shard["global_start_inclusive"]),
            int(shard["global_stop_exclusive"]),
        )
        if int(shard["index"]) != expected_index:
            raise ValueError("partial shard index is missing or duplicated")
        if int(shard["count"]) != expected_shards or interval != expected_interval:
            raise ValueError("partial shard interval differs from implementation lock")
        expected_frames = list(range(*expected_interval))
        actual_frames = [int(record["frame_index"]) for record in partial["frame_records"]]
        if actual_frames != expected_frames:
            raise ValueError("partial frame records differ from its declared interval")
        digests = partial["field_stream_sha256"]
        if set(digests) != set(audit.C5P_FIELDS) or any(
            SHA256_PATTERN.fullmatch(value) is None for value in digests.values()
        ):
            raise ValueError("partial field stream digest schema is invalid")
        frame_records.extend(partial["frame_records"])
        partial_artifacts.append(
            {
                "shard_index": expected_index,
                "frame_interval": list(expected_interval),
                "path": str(path),
                "sha256": sha256_file(path),
                "field_stream_sha256": digests,
            }
        )

    frame_indices = [int(record["frame_index"]) for record in frame_records]
    if frame_indices != list(range(624)):
        raise ValueError("merged frame coverage is incomplete or duplicated")
    selected_records = [
        record
        for record in frame_records
        if int(record["frame_index"]) in audit.SELECTED_RAW_FRAMES
    ]
    if [int(record["frame_index"]) for record in selected_records] != list(
        audit.SELECTED_RAW_FRAMES
    ):
        raise ValueError("selected structural frame coverage is incomplete")
    raw_structural_passed = all(
        all(
            record["structural_checks"]
            ["selected_raw_equals_native_after_float32_cast"].values()
        )
        for record in selected_records
    )
    legacy_structural_passed = all(
        all(
            record["structural_checks"]
            ["selected_legacy_c5t_resampling_bitwise_exact"].values()
        )
        for record in selected_records
    )
    for record in frame_records:
        selected = int(record["frame_index"]) in audit.SELECTED_RAW_FRAMES
        checks = record["structural_checks"]
        if selected != (
            checks["selected_raw_equals_native_after_float32_cast"] is not None
        ) or selected != (
            checks["selected_legacy_c5t_resampling_bitwise_exact"] is not None
        ):
            raise ValueError("structural checks appear on the wrong frames")

    blocks = manifest["data"]["temporal_blocks_inclusive"]
    field_round_trip = {
        field: merge_field_round_trip(frame_records, field, blocks)
        for field in audit.C5P_FIELDS
    }
    comparisons = {
        path: merge_comparison_path(frame_records, path=path, blocks=blocks)
        for path in ("round_trip", "direct_88", "raw64_vs_float32")
    }

    gates = manifest["acceptance_gates"]
    field_gate = {
        field: bool(
            result["per_frame_relative_l2_summary"]["maximum"]
            <= float(gates["field_round_trip_max_per_frame_relative_l2"])
        )
        for field, result in field_round_trip.items()
    }
    round_trip_gate: dict[str, dict[str, bool]] = {}
    raw_quantization_gate: dict[str, dict[str, bool]] = {}
    for category in ("face_total", "divergence_total"):
        round_trip_gate[category] = {}
        raw_quantization_gate[category] = {}
        for quantity in audit.PRIMARY_QUANTITIES:
            round_result = comparisons["round_trip"][category][quantity]
            round_trip_gate[category][quantity] = bool(
                round_result["aggregate"]["relative_l2"]
                <= float(gates["transport_round_trip_max_aggregate_relative_l2"])
                and round_result["per_frame_relative_l2_summary"]["p99"]
                <= float(gates["transport_round_trip_max_p99_per_frame_relative_l2"])
            )
            raw_result = comparisons["raw64_vs_float32"][category][quantity]
            raw_quantization_gate[category][quantity] = bool(
                raw_result["aggregate"]["relative_l2"]
                <= float(
                    gates[
                        "selected_raw64_vs_float32_transport_max_aggregate_relative_l2"
                    ]
                )
            )

    direct_materiality = {
        category: {
            quantity: materiality_label(
                comparisons["direct_88"][category][quantity]["aggregate"][
                    "relative_l2"
                ]
            )
            for quantity in audit.PRIMARY_QUANTITIES
        }
        for category in audit.COMPARISON_CATEGORIES
    }
    all_round_trip_passed = all(field_gate.values()) and all(
        passed
        for category in round_trip_gate.values()
        for passed in category.values()
    )
    all_raw_quantization_passed = all(
        passed
        for category in raw_quantization_gate.values()
        for passed in category.values()
    )
    overall_passed = bool(
        raw_structural_passed
        and legacy_structural_passed
        and all_round_trip_passed
        and all_raw_quantization_passed
    )
    stream_digest_tree = {
        field: digest_records(
            [
                {
                    "shard_index": record["shard_index"],
                    "frame_interval": record["frame_interval"],
                    "stream_sha256": record["field_stream_sha256"][field],
                }
                for record in partial_artifacts
            ]
        )
        for field in audit.C5P_FIELDS
    }

    result = {
        "schema_version": 1,
        "phase": "phase2_85604_native81_resampled88_sensitivity",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "audit_completed": True,
        "rank_shard_completed": False,
        "development_run": "85604",
        "held_out_85606_read": False,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_digest,
        "frame_count": 624,
        "frame_indices_complete_and_unique": True,
        "native_shape_per_frame": [64, 32, 81],
        "resampled_shape_per_frame": [64, 32, 88],
        "zperiod": 5,
        "parallel_execution": {
            "shard_count": expected_shards,
            "chunk_aligned_intervals": [list(interval) for interval in audit.SHARD_INTERVALS],
            "all_shards_completed_before_merge": True,
            "artifact_root": str(args.artifact_root),
            "partials": partial_artifacts,
        },
        "field_stream_digest_tree": stream_digest_tree,
        "structural_checks": {
            "selected_frames": list(audit.SELECTED_RAW_FRAMES),
            "raw_equals_native_after_float32_cast": raw_structural_passed,
            "legacy_c5t_resampling_bitwise_exact": legacy_structural_passed,
        },
        "field_round_trip": field_round_trip,
        "comparisons": comparisons,
        "acceptance": {
            "thresholds": gates,
            "field_round_trip": field_gate,
            "primary_transport_round_trip": round_trip_gate,
            "selected_raw64_vs_float32_transport": raw_quantization_gate,
            "all_round_trip_gates_passed": all_round_trip_passed,
            "all_raw_quantization_gates_passed": all_raw_quantization_passed,
            "overall_passed": overall_passed,
        },
        "scientific_findings": {
            "direct_88_materiality": direct_materiality,
            "primary_transport_evaluator": (
                "downsample_each_88_cell_member_to_native_81_then_apply_Q81"
                if overall_passed
                else "blocked_pending_documented_round_trip_or_quantization_failure"
            ),
            "direct_88_is_primary_transport_score": False,
            "automatic_architecture_change_authorized": False,
            "automatic_channel_change_authorized": False,
        },
        "frame_records": frame_records,
    }
    audit.strict_json_write(args.output, result)
    print(json.dumps(result["acceptance"], indent=2, sort_keys=True))
    print(json.dumps(result["scientific_findings"], indent=2, sort_keys=True))
    print(f"Wrote complete resampling audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
