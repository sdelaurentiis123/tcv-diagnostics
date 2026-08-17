#!/usr/bin/env python3
"""Create the compact tracked report for the all-frame closure result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FRAME_COUNT = 624


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--extraction-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-full-sha256", required=True)
    parser.add_argument("--expected-extraction-sha256", required=True)
    parser.add_argument("--artifact-index-sha256", required=True)
    parser.add_argument("--job-log-sha256", required=True)
    parser.add_argument("--slurm-log-sha256", required=True)
    parser.add_argument("--slurm-state", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--batch-max-rss-kib", type=int, required=True)
    parser.add_argument("--allocated-cpus", type=int, required=True)
    parser.add_argument("--requested-tasks", type=int, required=True)
    parser.add_argument("--cpus-per-task", type=int, required=True)
    parser.add_argument("--submit-command", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value} in {path}")
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_strict_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def frame_rows(per_frame: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    labels = [f"f{frame:03d}" for frame in range(FRAME_COUNT)]
    if list(per_frame) != labels:
        raise ValueError("per-frame metrics do not cover f000 through f623 in order")
    return [(frame, per_frame[label]) for frame, label in enumerate(labels)]


def maximum_record(
    rows: list[tuple[int, dict[str, Any]]], key: str
) -> dict[str, Any]:
    frame, metrics = max(rows, key=lambda item: float(item[1][key]))
    return {"frame_index": frame, "metrics": metrics}


def gate_fraction(metrics: dict[str, Any]) -> float:
    return float(metrics["maximum_absolute_difference"]) / float(
        metrics["acceptance_tolerance"]
    )


def summarize_frames(
    rows: list[tuple[int, dict[str, Any]]], intervals: list[list[int]]
) -> dict[str, Any]:
    frame_by_number = dict(rows)
    gate_frame, gate_metrics = max(rows, key=lambda item: gate_fraction(item[1]))
    blocks = []
    for block_index, (start, stop) in enumerate(intervals):
        block_rows = [(frame, frame_by_number[frame]) for frame in range(start, stop)]
        block_gate_frame, block_gate_metrics = max(
            block_rows, key=lambda item: gate_fraction(item[1])
        )
        blocks.append(
            {
                "block_index": block_index,
                "start": start,
                "stop": stop,
                "all_frames_passed": all(bool(row["passed"]) for _, row in block_rows),
                "maximum_gate_fraction": gate_fraction(block_gate_metrics),
                "maximum_gate_fraction_frame": block_gate_frame,
                "maximum_absolute_difference": maximum_record(
                    block_rows, "maximum_absolute_difference"
                ),
                "maximum_relative_l2": maximum_record(block_rows, "relative_l2"),
                "maximum_rmse": maximum_record(block_rows, "rmse"),
            }
        )
    return {
        "all_frames_passed": all(bool(metrics["passed"]) for _, metrics in rows),
        "maximum_gate_fraction": gate_fraction(gate_metrics),
        "maximum_gate_fraction_frame": gate_frame,
        "maximum_absolute_difference": maximum_record(
            rows, "maximum_absolute_difference"
        ),
        "maximum_relative_l2": maximum_record(rows, "relative_l2"),
        "maximum_rmse": maximum_record(rows, "rmse"),
        "by_predeclared_temporal_block": blocks,
    }


def summarize_runtime(per_frame: dict[str, Any]) -> dict[str, Any]:
    labels = [f"f{frame:03d}" for frame in range(FRAME_COUNT)]
    if list(per_frame) != labels:
        raise ValueError("runtime-pressure metrics do not cover all frames")
    result: dict[str, Any] = {}
    for field in ("Pe", "Pi"):
        rows = [(frame, per_frame[label][field]) for frame, label in enumerate(labels)]
        result[field] = {
            "all_frames_passed": all(bool(metrics["passed"]) for _, metrics in rows),
            "maximum_absolute_difference": maximum_record(
                rows, "maximum_absolute_difference"
            ),
            "maximum_relative_l2": maximum_record(rows, "relative_l2"),
            "maximum_nonfinite_count": max(
                int(metrics["nonfinite_count"]) for _, metrics in rows
            ),
        }
    return result


def compact_shards(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "shard_index": shard["shard_index"],
            "start": shard["start"],
            "stop": shard["stop"],
            "canonical_file": shard["canonical_file"],
            "canonical_file_sha256": shard["canonical_file_sha256"],
            "array_sha256": shard["array_sha256"],
        }
        for shard in extraction["shards"]
    ]


def build_summary(
    full: dict[str, Any],
    extraction: dict[str, Any],
    *,
    full_sha256: str,
    extraction_sha256: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if (
        full["phase"] != "phase2_potential_vorticity_all_frame_85604"
        or full["frame_count"] != FRAME_COUNT
        or full["frame_indices"] != list(range(FRAME_COUNT))
        or full["held_out_85606_read"]
        or full["training_performed"]
    ):
        raise ValueError("full result violates the all-frame development scope")
    if (
        extraction["paper0_commit"] != full["paper0_commit"]
        or extraction["slurm_job_id"] != full["slurm_job_id"]
        or extraction["frame_indices"] != list(range(FRAME_COUNT))
        or extraction["held_out_85606_read"]
        or extraction["training_performed"]
    ):
        raise ValueError("extraction identity differs from the full result")
    if full["artifacts"]["extraction_record_sha256"] != extraction_sha256:
        raise ValueError("full result does not lock the supplied extraction record")

    source = full["source_forward_closure_gate"]
    rows = frame_rows(source["per_frame"])
    frame_summary = summarize_frames(rows, extraction["shard_intervals"])
    external_root = full["artifacts"]["artifact_root"]
    full_external = f"{external_root}/potential_vorticity_all_frame_comparison.json"
    artifact_index = f"{external_root}/artifact_sha256.txt"
    job_log = f"{external_root}/job.log"
    slurm_log = (
        "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs/"
        f"phase2_vort_all_{full['slurm_job_id']}.log"
    )
    return {
        "schema_version": 1,
        "phase": "phase2_potential_vorticity_all_frame_85604_compact_summary",
        "paper0_commit": full["paper0_commit"],
        "slurm_job_id": full["slurm_job_id"],
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "scope": {
            "frame_count": FRAME_COUNT,
            "physical_shape_xyz": full["physical_shape_xyz"],
            "point_count_per_volume_field": source["pooled"]["point_count"],
            "native_z_samples": full["native_z_samples"],
            "zperiod": full["zperiod"],
            "toroidal_mode_mapping": "n=5k",
        },
        "compute": {
            "slurm_state": args.slurm_state,
            "node": args.node,
            "start": args.start,
            "end": args.end,
            "elapsed_seconds": args.elapsed_seconds,
            "batch_max_rss_kib": args.batch_max_rss_kib,
            "allocated_cpus_reported_by_sacct": args.allocated_cpus,
            "requested_tasks": args.requested_tasks,
            "cpus_per_task": args.cpus_per_task,
        },
        "extraction": {
            "rank_file_count": extraction["rank_file_count"],
            "rank_files_traversed_once": extraction["rank_files_traversed_once"],
            "raw_rank_staged_sequentially_once": extraction[
                "raw_rank_staged_sequentially_once"
            ],
            "maximum_simultaneous_staged_rank_files": extraction[
                "maximum_simultaneous_staged_rank_files"
            ],
            "staged_rank_files_retained": extraction["staged_rank_files_retained"],
            "raw_rank_read_order": extraction["raw_rank_read_order"],
            "canonical_volume_chunks": extraction["canonical_volume_chunks"],
            "canonical_boundary_chunks": extraction["canonical_boundary_chunks"],
            "raw_pressure_identity": extraction["raw_pressure_identity"],
            "boundary_checks": extraction["boundary_checks"],
            "canonical_shards": compact_shards(extraction),
        },
        "ordered_gates": full["ordered_gates"],
        "runtime_pressure_gate": {
            "negative_raw_Pe_count": full["runtime_pressure_gate"][
                "negative_raw_Pe_count"
            ],
            "negative_raw_Pi_count": full["runtime_pressure_gate"][
                "negative_raw_Pi_count"
            ],
            "negative_raw_Pi_count_by_shard": full["runtime_pressure_gate"][
                "negative_raw_Pi_count_by_shard"
            ],
            "pressure_inventory_passed": full["runtime_pressure_gate"][
                "pressure_inventory_passed"
            ],
            "per_field_extrema": summarize_runtime(
                full["runtime_pressure_gate"]["per_frame"]
            ),
            "passed": full["runtime_pressure_gate"]["passed"],
        },
        "source_forward_closure_gate": {
            "atol": source["atol"],
            "rtol": source["rtol"],
            "scope": source["scope"],
            "pooled": source["pooled"],
            "by_geometry_region_pooled": source["by_geometry_region_pooled"],
            "frame_extrema": frame_summary,
            "toroidal_mode_residual_pooled": source["toroidal_mode_residual"][
                "pooled"
            ],
            "passed": source["passed"],
        },
        "decision": full["decision"],
        "external_artifacts": {
            "artifact_root": external_root,
            "full_result": {
                "path": full_external,
                "bytes": args.full_result.stat().st_size,
                "sha256": full_sha256,
            },
            "extraction_record": {
                "path": full["artifacts"]["extraction_record"],
                "bytes": args.extraction_record.stat().st_size,
                "sha256": extraction_sha256,
            },
            "artifact_index": {
                "path": artifact_index,
                "sha256": args.artifact_index_sha256,
            },
            "job_log": {"path": job_log, "sha256": args.job_log_sha256},
            "slurm_log": {"path": slurm_log, "sha256": args.slurm_log_sha256},
            "manifest": {
                "path": full["artifacts"]["manifest"],
                "sha256": full["artifacts"]["manifest_sha256"],
            },
            "protocol": {
                "path": full["artifacts"]["protocol"],
                "sha256": full["artifacts"]["protocol_sha256"],
            },
        },
        "reproduction": {
            "submit_command": args.submit_command,
            "full_result_is_external_and_immutable": True,
            "summary_tool": "paper0/tools/summarize_potential_vorticity_all_frame.py",
            "summary_tool_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def main() -> int:
    args = parse_args()
    full_sha256 = sha256_file(args.full_result)
    extraction_sha256 = sha256_file(args.extraction_record)
    if full_sha256 != args.expected_full_sha256:
        raise ValueError("full-result SHA-256 differs from the expected artifact")
    if extraction_sha256 != args.expected_extraction_sha256:
        raise ValueError("extraction SHA-256 differs from the expected artifact")
    summary = build_summary(
        load_json(args.full_result),
        load_json(args.extraction_record),
        full_sha256=full_sha256,
        extraction_sha256=extraction_sha256,
        args=args,
    )
    write_strict_json(args.output, summary)
    print(
        json.dumps(
            {
                "all_frame_bidirectional_closure_validated": summary["decision"][
                    "all_frame_bidirectional_closure_validated"
                ],
                "output": str(args.output),
                "pooled_relative_l2": summary["source_forward_closure_gate"][
                    "pooled"
                ]["relative_l2"],
                "maximum_gate_fraction": summary["source_forward_closure_gate"][
                    "frame_extrema"
                ]["maximum_gate_fraction"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
