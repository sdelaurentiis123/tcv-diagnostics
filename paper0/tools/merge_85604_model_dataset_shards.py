#!/usr/bin/env python3
"""Validate and merge all frozen 85604 model-dataset shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.model_data import (  # noqa: E402
    StreamingMoments,
    VOLUME_FIELDS,
    apply_moment_transform,
    array_sha256,
    load_strict_json,
    records_close,
    sha256_file,
    validate_intervals,
    write_strict_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-preflight", type=Path, required=True)
    parser.add_argument("--partial-output", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization-output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def git_execution_gate(expected_commit: str) -> dict[str, Any]:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    result = {
        "expected_commit": expected_commit,
        "actual_commit": actual,
        "commit_matches": actual == expected_commit,
        "worktree_clean": dirty == "",
    }
    if not result["commit_matches"] or not result["worktree_clean"]:
        raise ValueError("git execution gate failed")
    return result


def rocky_execution_gate(required_major: int) -> dict[str, Any]:
    records: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            records[key] = value.strip().strip('"')
    actual_major = int(records["VERSION_ID"].split(".", 1)[0])
    result = {
        "required_major": required_major,
        "actual_major": actual_major,
        "pretty_name": records.get("PRETTY_NAME"),
        "passed": actual_major == required_major,
    }
    if not result["passed"]:
        raise ValueError("Rocky major-version gate failed")
    return result


def merge_partial_moments(
    partials: list[Mapping[str, Any]],
    key: str,
) -> dict[str, int | float]:
    merged = StreamingMoments()
    found = False
    for partial in partials:
        records = partial["normalization_partial"]["records"]
        if key in records:
            merged.merge(StreamingMoments.from_record(records[key]))
            found = True
    if not found:
        raise ValueError(f"no normalization partial was found for {key}")
    return merged.finalize()


def normalization_comparison(
    partial_record: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    expected_count: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    comparisons = records_close(
        partial_record,
        recomputed,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
    standard_deviation = float(recomputed["population_standard_deviation"])
    positive_scale = math.isfinite(standard_deviation) and (
        standard_deviation > 0.0
    )
    passed = (
        int(partial_record["count"]) == expected_count
        and int(recomputed["count"]) == expected_count
        and all(comparisons.values())
        and positive_scale
    )
    return {
        "expected_count": expected_count,
        "partial_record": dict(partial_record),
        "recomputed_record": dict(recomputed),
        "comparisons": comparisons,
        "standard_deviation_strictly_positive": positive_scale,
        "passed": passed,
    }


def recompute_output_moments(
    shard_paths: list[Path],
    intervals: tuple[tuple[int, int], ...],
    *,
    field: str,
    transforms: Mapping[str, Mapping[str, Any]],
    training_stop: int,
    boundary_side: int | None = None,
) -> dict[str, int | float]:
    moments = StreamingMoments()
    for path, (start, stop) in zip(shard_paths, intervals):
        local_training_count = max(0, min(stop, training_stop) - start)
        if local_training_count == 0:
            continue
        with h5py.File(path, "r") as handle:
            if boundary_side is None:
                dataset = handle[f"fields/{field}"]
                for local_frame in range(local_training_count):
                    values = np.asarray(dataset[local_frame], dtype=np.float32)
                    moments.update(
                        apply_moment_transform(field, values, transforms)
                    )
            else:
                dataset = handle["boundary/Bphi"]
                for local_frame in range(local_training_count):
                    values = np.asarray(
                        dataset[local_frame, boundary_side, :],
                        dtype=np.float32,
                    )
                    moments.update(
                        apply_moment_transform("Bphi", values, transforms)
                    )
    return moments.finalize()


def verify_shard_contents(
    path: Path,
    partial: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    if sha256_file(path) != partial["output"]["sha256"]:
        raise ValueError(f"shard file hash differs for {path}")
    result: dict[str, Any] = {"fields": {}, "boundary": {}}
    with h5py.File(path, "r") as handle:
        start = int(partial["shard"]["global_start_inclusive"])
        stop = int(partial["shard"]["global_stop_exclusive"])
        expected_frames = np.arange(start, stop, dtype=np.int64)
        frame_index = np.asarray(
            handle["coordinates/frame_index"][:], dtype=np.int64
        )
        time = np.asarray(handle["coordinates/time"][:], dtype=np.float64)
        if not np.array_equal(frame_index, expected_frames):
            raise ValueError(f"frame coordinates differ in {path}")
        if not np.all(np.isfinite(time)):
            raise ValueError(f"time contains non-finite values in {path}")
        for field in fields:
            values = np.asarray(handle[f"fields/{field}"][...], dtype=np.float32)
            if values.shape != (stop - start, 64, 32, 88):
                raise ValueError(f"{field} shape differs in {path}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{field} contains non-finite values in {path}")
            digest = array_sha256(values)
            if digest != partial["fields"][field]["output_array_sha256"]:
                raise ValueError(f"{field} array digest differs in {path}")
            result["fields"][field] = digest
        boundary = np.asarray(handle["boundary/Bphi"][...], dtype=np.float32)
        if boundary.shape != (stop - start, 2, 32):
            raise ValueError(f"Bphi shape differs in {path}")
        if not np.all(np.isfinite(boundary)):
            raise ValueError(f"Bphi contains non-finite values in {path}")
        boundary_digest = array_sha256(boundary)
        if boundary_digest != partial["boundary"]["output_array_sha256"]:
            raise ValueError(f"Bphi array digest differs in {path}")
        result["boundary"] = boundary_digest
        result["frame_indices"] = frame_index.tolist()
        result["time"] = time.tolist()
    return result


def main() -> int:
    args = parse_args()
    manifest = load_strict_json(args.manifest)
    manifest_sha256 = sha256_file(args.manifest)
    protocol_sha256 = sha256_file(args.protocol)
    if protocol_sha256 != manifest["protocol"]["sha256"]:
        raise ValueError("protocol hash differs from the manifest lock")
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_authorized"]
    ):
        raise ValueError("reducer requires the frozen 85604-only manifest")
    expected_artifact_root = (
        Path(manifest["output"]["base_directory"])
        / manifest["output"]["job_directory_template"].format(
            slurm_job_id=args.slurm_job_id
        )
    )
    if args.artifact_root.resolve() != expected_artifact_root.resolve():
        raise ValueError("artifact root differs from the frozen job directory")
    if args.output.parent.resolve() != args.artifact_root.resolve():
        raise ValueError("merged output must live in the artifact root")
    if args.normalization_output.parent.resolve() != args.artifact_root.resolve():
        raise ValueError("normalization output must live in the artifact root")
    if args.output.name != manifest["output"]["merged_record_filename"]:
        raise ValueError("merged output filename differs from manifest")
    if args.normalization_output.name != manifest["output"]["normalization_filename"]:
        raise ValueError("normalization filename differs from manifest")

    intervals = validate_intervals(
        manifest["output"]["shard_intervals"],
        expected_start=0,
        expected_stop=624,
    )
    if len(args.partial_output) != len(intervals):
        raise ValueError("reducer requires exactly eight partial outputs")
    loaded_partials = [
        (path, load_strict_json(path)) for path in args.partial_output
    ]
    loaded_partials.sort(key=lambda item: int(item[1]["shard"]["index"]))
    partial_paths = [item[0] for item in loaded_partials]
    partials = [item[1] for item in loaded_partials]
    if [int(item["shard"]["index"]) for item in partials] != list(range(8)):
        raise ValueError("partial shard indices are not exactly 0 through 7")

    preflight_sha256 = sha256_file(args.source_preflight)
    shard_paths: list[Path] = []
    shard_records: list[dict[str, Any]] = []
    field_maximum_round_trip = {field: 0.0 for field in VOLUME_FIELDS}
    legacy_fields = set(
        manifest["integrity_gates"]["legacy_z88_bitwise_fields"]
    )
    all_times: list[float] = []
    all_frames: list[int] = []
    for index, (partial_path, partial, interval) in enumerate(
        zip(partial_paths, partials, intervals)
    ):
        start, stop = interval
        if (
            partial["phase"] != "phase2_85604_model_dataset_shard"
            or partial["paper0_commit"] != args.paper0_commit
            or int(partial["slurm_job_id"]) != args.slurm_job_id
            or partial["manifest_sha256"] != manifest_sha256
            or partial["protocol_sha256"] != protocol_sha256
            or partial["held_out_85606_read"]
            or partial["training_performed"]
            or not partial["all_shard_gates_passed"]
        ):
            raise ValueError(f"partial identity or gate differs for shard {index}")
        if (
            int(partial["shard"]["global_start_inclusive"]) != start
            or int(partial["shard"]["global_stop_exclusive"]) != stop
            or partial["shard"]["frame_indices"] != list(range(start, stop))
        ):
            raise ValueError(f"partial interval differs for shard {index}")
        if (
            partial["source_preflight"]["sha256"] != preflight_sha256
            or not partial["source_preflight"]["passed"]
        ):
            raise ValueError(f"source preflight differs for shard {index}")
        path = Path(partial["output"]["path"])
        expected_name = manifest["output"]["shard_filename_template"].format(
            shard_index=index
        )
        expected_path = (
            args.artifact_root
            / manifest["output"]["shard_directory"]
            / expected_name
        )
        if path.resolve() != expected_path.resolve():
            raise ValueError(f"output path differs for shard {index}")
        verified = verify_shard_contents(
            path,
            partial,
            fields=VOLUME_FIELDS,
        )
        shard_paths.append(path)
        all_frames.extend(verified["frame_indices"])
        all_times.extend(verified["time"])
        for field in VOLUME_FIELDS:
            field_record = partial["fields"][field]
            value = float(
                field_record["round_trip"]["maximum_per_frame_relative_l2"]
            )
            field_maximum_round_trip[field] = max(
                field_maximum_round_trip[field], value
            )
            if field in legacy_fields and not field_record["legacy_z88"][
                "bitwise_exact"
            ]:
                raise ValueError(f"legacy z88 gate failed for {field}")
        if not partial["boundary"]["explicit_float32_cast_bitwise_exact"]:
            raise ValueError(f"Bphi cast gate failed for shard {index}")
        if not partial["writer_echo"]["all_bitwise_exact"]:
            raise ValueError(f"writer echo gate failed for shard {index}")
        shard_records.append(
            {
                "index": index,
                "global_start_inclusive": start,
                "global_stop_exclusive": stop,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": partial["output"]["sha256"],
                "field_array_sha256": verified["fields"],
                "Bphi_array_sha256": verified["boundary"],
                "partial_path": str(partial_path),
                "partial_sha256": sha256_file(partial_path),
            }
        )

    if all_frames != list(range(624)):
        raise ValueError("merged frame coverage is not exactly 0 through 623")
    time = np.asarray(all_times, dtype=np.float64)
    time_specification = manifest["data"]["time"]
    expected_time = time_specification["first"] + (
        time_specification["normalized_step"] * np.arange(624, dtype=np.float64)
    )
    time_exact = bool(np.array_equal(time, expected_time))
    if not time_exact or not np.all(np.diff(time) > 0.0):
        raise ValueError("merged normalized time sequence differs")
    physical_cadence = (
        float(time_specification["normalized_step"])
        / float(time_specification["omega_ci_per_second"])
        * 1e6
    )
    physical_cadence_matches = math.isclose(
        physical_cadence,
        float(time_specification["physical_cadence_microseconds"]),
        rel_tol=1e-15,
        abs_tol=0.0,
    )
    if not physical_cadence_matches:
        raise ValueError("physical cadence conversion differs")

    transforms = manifest["normalization"]["transforms"]
    training_stop = int(manifest["paper0_split"]["normalization_fit_frames"][1])
    relative_tolerance = float(
        manifest["integrity_gates"][
            "normalization_recomputation_relative_tolerance"
        ]
    )
    absolute_tolerance = float(
        manifest["integrity_gates"][
            "normalization_recomputation_absolute_tolerance"
        ]
    )
    normalization_records: dict[str, Any] = {}
    normalization_gate: dict[str, Any] = {}
    normalization_keys = list(VOLUME_FIELDS) + ["Bphi/inner", "Bphi/outer"]
    for key in normalization_keys:
        partial_record = merge_partial_moments(partials, key)
        if key.startswith("Bphi/"):
            side = 0 if key.endswith("inner") else 1
            recomputed = recompute_output_moments(
                shard_paths,
                intervals,
                field="Bphi",
                transforms=transforms,
                training_stop=training_stop,
                boundary_side=side,
            )
            expected_count = int(
                manifest["normalization"]["boundary_count_per_side"]
            )
        else:
            recomputed = recompute_output_moments(
                shard_paths,
                intervals,
                field=key,
                transforms=transforms,
                training_stop=training_stop,
            )
            expected_count = int(
                manifest["normalization"]["volume_count_per_field"]
            )
        comparison = normalization_comparison(
            partial_record,
            recomputed,
            expected_count=expected_count,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if not comparison["passed"]:
            raise ValueError(f"normalization recomputation failed for {key}")
        transform_key = "Bphi" if key.startswith("Bphi/") else key
        normalization_records[key] = {
            **recomputed,
            "transform": transforms[transform_key],
        }
        normalization_gate[key] = comparison

    normalization_artifact = {
        "schema_version": 1,
        "phase": "phase2_85604_model_dataset_normalization",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "fit_frames": manifest["paper0_split"]["normalization_fit_frames"],
        "accumulator_dtype": manifest["normalization"]["accumulator_dtype"],
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "records": normalization_records,
    }
    write_strict_json_atomic(args.normalization_output, normalization_artifact)
    normalization_sha256 = sha256_file(args.normalization_output)

    round_trip_limit = float(
        manifest["integrity_gates"]["field_round_trip_max_per_frame_relative_l2"]
    )
    round_trip_gate = {
        field: {
            "maximum_per_frame_relative_l2": value,
            "limit": round_trip_limit,
            "passed": value <= round_trip_limit,
        }
        for field, value in field_maximum_round_trip.items()
    }
    if not all(item["passed"] for item in round_trip_gate.values()):
        raise ValueError("merged round-trip field gate failed")

    git_gate = git_execution_gate(args.paper0_commit)
    rocky_gate = rocky_execution_gate(
        int(manifest["integrity_gates"]["rocky_major_version_required"])
    )
    result = {
        "schema_version": 1,
        "phase": "phase2_85604_matched_model_dataset",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "manifest": {
            "path": str(args.manifest),
            "sha256": manifest_sha256,
        },
        "protocol": {
            "path": str(args.protocol),
            "sha256": protocol_sha256,
        },
        "source_preflight": {
            "path": str(args.source_preflight),
            "sha256": preflight_sha256,
        },
        "execution": {
            "git": git_gate,
            "os": rocky_gate,
            "hostname": platform.node(),
        },
        "dataset": {
            "frame_count": 624,
            "fields": list(VOLUME_FIELDS),
            "shards": shard_records,
            "total_bytes": sum(item["bytes"] for item in shard_records),
            "normalization": {
                "path": str(args.normalization_output),
                "sha256": normalization_sha256,
            },
        },
        "time": {
            "coordinate": time_specification["coordinate_name"],
            "first": float(time[0]),
            "last": float(time[-1]),
            "normalized_step": float(time[1] - time[0]),
            "omega_ci_per_second": float(
                time_specification["omega_ci_per_second"]
            ),
            "physical_cadence_microseconds": physical_cadence,
        },
        "gates": {
            "source_hash_preflight": True,
            "complete_unique_frame_coverage": True,
            "normalized_time_sequence_exact": time_exact,
            "physical_cadence_conversion": physical_cadence_matches,
            "all_inputs_and_reopened_outputs_finite": True,
            "writer_echo_bitwise_exact": True,
            "legacy_z88_bitwise_all_frames": True,
            "field_round_trip": round_trip_gate,
            "Bphi_explicit_float32_cast_bitwise_exact": True,
            "normalization_recomputation": normalization_gate,
            "clean_git_checkout": True,
            "rocky_major_version": True,
            "all_passed": True,
        },
        "decision": {
            "dataset_gate_passed": True,
            "training_released": False,
            "meaning": "verified_shared_engineering_representation_only",
            "next_required_gate": "committed_matched_O1_O2_model_protocol",
        },
    }
    write_strict_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "normalization": str(args.normalization_output),
                "dataset_gate_passed": True,
                "training_released": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
