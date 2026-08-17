#!/usr/bin/env python3
"""Merge eight exact E6B zero-interior-seed truth replay results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, nargs=8, required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def summarize_truth_replay(
    *,
    records: Sequence[Mapping[str, Any]],
    record_paths: Sequence[Path],
    native_truth: Mapping[str, Any],
    paper0_commit: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    if len(records) != 8 or len(record_paths) != 8:
        raise ValueError("truth replay requires exactly eight shard results")
    if (
        native_truth.get("development_run") != "85604"
        or native_truth.get("held_out_85606_read") is not False
        or native_truth.get("decision", {}).get(
            "all_frame_bidirectional_closure_validated"
        )
        is not True
        or not native_truth.get("ordered_gates")
        or not all(native_truth["ordered_gates"].values())
    ):
        raise ValueError("native truth authority did not pass")
    canonical = native_truth.get("extraction", {}).get("canonical_shards", [])
    if len(canonical) != 8:
        raise ValueError("native truth authority does not contain eight shards")

    shards: list[dict[str, Any]] = []
    all_passed = True
    maximum_absolute_difference = 0.0
    maximum_relative_l2 = 0.0
    frame_count = 0
    for index, (record, path, source) in enumerate(
        zip(records, record_paths, canonical)
    ):
        interval = [index * 78, (index + 1) * 78]
        if (
            record.get("scope") != "phase2_matched_e6b_elliptic_output"
            or record.get("status") != "completed"
            or record.get("development_run") != "85604"
            or record.get("held_out_85606_read") is not False
            or record.get("paper0_commit") != paper0_commit
            or record.get("truth_layout") is not True
            or record.get("frame_interval") != interval
            or int(record.get("frame_count", -1)) != 78
            or Path(record.get("source_input", {}).get("path", "")).resolve(
                strict=True
            )
            != Path(source.get("canonical_file", "")).resolve(strict=True)
            or record.get("source_input", {}).get("sha256")
            != source.get("canonical_file_sha256")
        ):
            raise ValueError(f"truth replay shard {index} identity differs")
        gate = record.get("truth_replay_gate")
        if not isinstance(gate, Mapping):
            raise ValueError(f"truth replay shard {index} has no gate")
        per_frame = gate.get("per_frame", [])
        if [int(item["frame_index"]) for item in per_frame] != list(
            range(*interval)
        ):
            raise ValueError(f"truth replay shard {index} frame records differ")
        shard_pass = bool(gate.get("all_frames_passed")) and all(
            bool(item.get("passes")) for item in per_frame
        )
        all_passed = all_passed and shard_pass
        maximum_absolute_difference = max(
            maximum_absolute_difference,
            float(gate["maximum_absolute_difference"]),
        )
        maximum_relative_l2 = max(
            maximum_relative_l2,
            float(gate["maximum_relative_l2"]),
        )
        frame_count += len(per_frame)
        shards.append(
            {
                "shard_index": index,
                "frame_interval": interval,
                "passes": shard_pass,
                "result": {"path": str(path), "sha256": sha256_path(path)},
                "source_input": dict(record["source_input"]),
                "derived_phi": dict(record["derived_phi"]),
                "maximum_absolute_difference": float(
                    gate["maximum_absolute_difference"]
                ),
                "maximum_relative_l2": float(gate["maximum_relative_l2"]),
            }
        )
    if frame_count != 624:
        raise ValueError("truth replay did not cover exactly 624 frames")
    return {
        "schema_version": 1,
        "scope": "phase2_matched_e6b_zero_seed_truth_replay",
        "status": "pass" if all_passed else "fail",
        "development_run": "85604",
        "held_out_85606_read": False,
        "paper0_commit": paper0_commit,
        "slurm_job_id": str(slurm_job_id),
        "coverage": [0, 624],
        "frame_count": frame_count,
        "all_frames_passed": all_passed,
        "boundary_only_zero_interior_seed": True,
        "zperiod": 5,
        "maximum_absolute_difference": maximum_absolute_difference,
        "maximum_relative_l2": maximum_relative_l2,
        "shards": shards,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    for path in (*args.result, args.native_truth_result, args.output):
        assert_development_path(path)
    native_path = args.native_truth_result.resolve(strict=True)
    if sha256_path(native_path) != args.native_truth_result_sha256:
        raise ValueError("native truth result SHA-256 differs")
    paths = [path.resolve(strict=True) for path in args.result]
    output = args.output.resolve(strict=False)
    result = summarize_truth_replay(
        records=[load_strict_json(path) for path in paths],
        record_paths=paths,
        native_truth=load_strict_json(native_path),
        paper0_commit=args.paper0_commit,
        slurm_job_id=args.slurm_job_id,
    )
    write_strict_json_atomic(output, result)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
