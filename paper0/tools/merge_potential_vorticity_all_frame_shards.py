#!/usr/bin/env python3
"""Merge exactly eight frozen all-frame potential/vorticity shard results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FRAME_COUNT = 624
SHARD_INTERVALS = tuple((start, start + 78) for start in range(0, 624, 78))
SUM_KEYS = (
    "sum_candidate",
    "sum_reference",
    "sum_candidate_squared",
    "sum_reference_squared",
    "sum_cross",
    "sum_error",
    "sum_error_squared",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--extraction-record", type=Path, required=True)
    parser.add_argument("--shard-result", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
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


def merge_sufficient_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one sufficient-statistic record is required")
    merged: dict[str, Any] = {
        "point_count": sum(int(record["point_count"]) for record in records),
        "finite_count": sum(int(record["finite_count"]) for record in records),
        "nonfinite_count": sum(int(record["nonfinite_count"]) for record in records),
    }
    for key in SUM_KEYS:
        merged[key] = float(sum(float(record[key]) for record in records))
    merged["max_abs_reference"] = max(
        float(record["max_abs_reference"]) for record in records
    )
    maximum_record = max(
        records, key=lambda record: float(record["maximum_absolute_difference"])
    )
    merged["maximum_absolute_difference"] = float(
        maximum_record["maximum_absolute_difference"]
    )
    merged["maximum_location"] = maximum_record["maximum_location"]
    return merged


def metrics_from_sufficient(record: dict[str, Any]) -> dict[str, Any]:
    count = int(record["finite_count"])
    if count <= 0:
        raise ValueError("merged sufficient statistics contain no finite values")
    sum_error_squared = float(record["sum_error_squared"])
    sum_reference_squared = float(record["sum_reference_squared"])
    candidate_variance_numerator = (
        float(record["sum_candidate_squared"])
        - float(record["sum_candidate"]) ** 2 / count
    )
    reference_variance_numerator = (
        sum_reference_squared - float(record["sum_reference"]) ** 2 / count
    )
    covariance_numerator = (
        float(record["sum_cross"])
        - float(record["sum_candidate"])
        * float(record["sum_reference"])
        / count
    )
    variance_product = max(candidate_variance_numerator, 0.0) * max(
        reference_variance_numerator, 0.0
    )
    correlation = (
        covariance_numerator / math.sqrt(variance_product)
        if variance_product > 0.0
        else None
    )
    return {
        "point_count": int(record["point_count"]),
        "finite_count": count,
        "nonfinite_count": int(record["nonfinite_count"]),
        "relative_l2": (
            math.sqrt(sum_error_squared / sum_reference_squared)
            if sum_reference_squared > 0.0
            else None
        ),
        "rmse": math.sqrt(sum_error_squared / count),
        "bias": float(record["sum_error"]) / count,
        "correlation": correlation,
        "max_abs_reference": float(record["max_abs_reference"]),
        "maximum_absolute_difference": float(
            record["maximum_absolute_difference"]
        ),
        "maximum_location": record["maximum_location"],
    }


def merge_mode_power(records: list[dict[str, Any]], *, zperiod: int) -> dict[str, Any]:
    if zperiod != 5:
        raise ValueError("all-frame mode merge requires zperiod=5")
    reference = np.sum(
        np.asarray([record["mode_reference_power"] for record in records]),
        axis=0,
        dtype=np.float64,
    )
    residual = np.sum(
        np.asarray([record["mode_residual_power"] for record in records]),
        axis=0,
        dtype=np.float64,
    )
    if reference.shape != (41,) or residual.shape != (41,):
        raise ValueError("mode-power shard arrays must contain 41 modes")
    floor = float(
        np.finfo(np.float64).eps * max(float(np.max(reference, initial=0.0)), 1.0)
    )
    relative = np.divide(
        residual,
        reference,
        out=np.full_like(residual, np.nan),
        where=reference > floor,
    )
    return {
        "fourier_index_k": list(range(41)),
        "toroidal_mode_n": [5 * k for k in range(41)],
        "reference_power": reference.tolist(),
        "residual_power": residual.tolist(),
        "relative_power_denominator_floor": floor,
        "relative_residual_power": [
            None if not np.isfinite(value) else float(value) for value in relative
        ],
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite the all-frame merged result")
    if len(args.shard_result) != 8:
        raise ValueError("exactly eight shard results are required")
    artifact_root = args.artifact_root.resolve(strict=True)
    for path in (*args.shard_result, args.extraction_record):
        if not path.resolve(strict=True).is_relative_to(artifact_root):
            raise ValueError(f"artifact lies outside the top-level result: {path}")

    manifest = load_json(args.manifest)
    if (
        manifest["development_run"] != "85604"
        or manifest["held_out_85606_access_allowed"]
        or manifest["training_allowed"]
        or [tuple(item) for item in manifest["shards"]["half_open_intervals"]]
        != list(SHARD_INTERVALS)
    ):
        raise ValueError("merge manifest differs from frozen scope")
    protocol_lock = manifest["protocol"]
    if args.protocol.resolve() != (ROOT / protocol_lock["path"]).resolve():
        raise ValueError("merge protocol path differs")
    if sha256_file(args.protocol) != protocol_lock["sha256"]:
        raise ValueError("merge protocol SHA-256 differs")
    manifest_sha = sha256_file(args.manifest)
    protocol_sha = sha256_file(args.protocol)
    extraction_sha = sha256_file(args.extraction_record)
    extraction = load_json(args.extraction_record)
    if (
        extraction["paper0_commit"] != args.paper0_commit
        or extraction["slurm_job_id"] != args.slurm_job_id
        or extraction["manifest_sha256"] != manifest_sha
        or extraction["frame_indices"] != list(range(FRAME_COUNT))
    ):
        raise ValueError("merge extraction provenance differs")

    loaded = [load_json(path) for path in args.shard_result]
    ordered = sorted(loaded, key=lambda result: int(result["shard_index"]))
    if [result["shard_index"] for result in ordered] != list(range(8)):
        raise ValueError("shard indices must cover exactly 0 through 7")
    all_frames: list[int] = []
    for shard_index, (result, interval) in enumerate(
        zip(ordered, SHARD_INTERVALS, strict=True)
    ):
        start, stop = interval
        if (
            result["phase"]
            != "phase2_potential_vorticity_all_frame_85604_shard"
            or result["paper0_commit"] != args.paper0_commit
            or result["slurm_job_id"] != args.slurm_job_id
            or result["development_run"] != "85604"
            or result["held_out_85606_read"]
            or result["training_performed"]
            or result["start"] != start
            or result["stop"] != stop
            or result["frame_indices"] != list(range(start, stop))
        ):
            raise ValueError(f"shard {shard_index} identity differs")
        artifacts = result["artifacts"]
        if (
            artifacts["manifest_sha256"] != manifest_sha
            or artifacts["protocol_sha256"] != protocol_sha
            or artifacts["extraction_record_sha256"] != extraction_sha
        ):
            raise ValueError(f"shard {shard_index} provenance hashes differ")
        all_frames.extend(result["frame_indices"])
    if all_frames != list(range(FRAME_COUNT)):
        raise ValueError("shard frame coverage is incomplete or duplicated")

    extraction_passed = all(
        result["extraction_gate"]["passed"] for result in ordered
    )
    input_passed = all(
        result["input_vorticity_echo_gate"]["passed"] for result in ordered
    )
    runtime_passed = all(
        result["runtime_pressure_gate"]["passed"] for result in ordered
    )
    compiled_passed = all(
        result["compiled_known_answer_gate"]["passed"] for result in ordered
    )
    source_passed = all(
        result["source_forward_closure_gate"]["passed"] for result in ordered
    )
    negative_pe = sum(
        int(result["runtime_pressure_gate"]["negative_raw_Pe_count"])
        for result in ordered
    )
    negative_pi = sum(
        int(result["runtime_pressure_gate"]["negative_raw_Pi_count"])
        for result in ordered
    )
    pressure_inventory_passed = bool(
        negative_pe == manifest["raw_pressure_identity"]["negative_raw_Pe_count"]
        and negative_pi
        == manifest["raw_pressure_identity"]["negative_raw_Pi_count"]
    )

    source_records = [
        result["source_forward_closure_gate"]["merge_sufficient_statistics"]
        for result in ordered
    ]
    if any(record is None for record in source_records):
        per_frame_source = None
        pooled_source = None
        regional_source = None
        mode_source = None
        source_status = "blocked_by_at_least_one_shard_preliminary_gate"
    else:
        typed_records = [record for record in source_records if record is not None]
        per_frame_source: dict[str, Any] = {}
        per_frame_modes: dict[str, Any] = {}
        for result in ordered:
            gate = result["source_forward_closure_gate"]
            for label, metrics in gate["per_frame"].items():
                if label in per_frame_source:
                    raise ValueError(f"duplicate per-frame source metric {label}")
                per_frame_source[label] = metrics
            for label, metrics in gate["toroidal_mode_residual"]["per_frame"].items():
                if label in per_frame_modes:
                    raise ValueError(f"duplicate per-frame mode metric {label}")
                per_frame_modes[label] = metrics
        expected_labels = [f"f{frame:03d}" for frame in range(FRAME_COUNT)]
        if list(per_frame_source) != expected_labels or list(per_frame_modes) != expected_labels:
            raise ValueError("per-frame source or mode metrics do not cover 0..623")
        merged_full = merge_sufficient_statistics(
            [record["full_domain"] for record in typed_records]
        )
        pooled_source = metrics_from_sufficient(merged_full)
        region_names = list(typed_records[0]["regions"])
        if any(list(record["regions"]) != region_names for record in typed_records):
            raise ValueError("region names or ordering differ across shards")
        regional_source = {
            name: metrics_from_sufficient(
                merge_sufficient_statistics(
                    [record["regions"][name] for record in typed_records]
                )
            )
            for name in region_names
        }
        mode_source = {
            "per_frame": per_frame_modes,
            "pooled": merge_mode_power(typed_records, zperiod=5),
        }
        source_status = "evaluated_after_all_shard_preliminary_gates_passed"

    per_frame_runtime: dict[str, Any] = {}
    for result in ordered:
        for label, metrics in result["runtime_pressure_gate"]["per_frame"].items():
            if label in per_frame_runtime:
                raise ValueError(f"duplicate runtime-pressure frame {label}")
            per_frame_runtime[label] = metrics
    expected_labels = [f"f{frame:03d}" for frame in range(FRAME_COUNT)]
    if list(per_frame_runtime) != expected_labels:
        raise ValueError("runtime-pressure metrics do not cover 0..623")

    final_passed = bool(
        extraction_passed
        and input_passed
        and runtime_passed
        and pressure_inventory_passed
        and compiled_passed
        and source_passed
        and per_frame_source is not None
        and all(metrics["passed"] for metrics in per_frame_source.values())
    )
    source_gate = {
        "status": source_status,
        "atol": manifest["source_forward_gate"]["atol"],
        "rtol": manifest["source_forward_gate"]["rtol"],
        "scope": "all_624x64x32x81_physical_points",
        "per_frame": per_frame_source,
        "pooled": pooled_source,
        "by_geometry_region_pooled": regional_source,
        "toroidal_mode_residual": mode_source,
        "all_624_frames_passed": bool(
            per_frame_source is not None
            and all(metrics["passed"] for metrics in per_frame_source.values())
        ),
        "passed": final_passed,
    }
    result = {
        "schema_version": 1,
        "phase": "phase2_potential_vorticity_all_frame_85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_performed": False,
        "frame_count": FRAME_COUNT,
        "frame_indices": list(range(FRAME_COUNT)),
        "physical_shape_xyz": [64, 32, 81],
        "native_z_samples": 81,
        "zperiod": 5,
        "shards": [
            {
                "shard_index": result["shard_index"],
                "start": result["start"],
                "stop": result["stop"],
                "passed": result["decision"]["shard_passed"],
            }
            for result in ordered
        ],
        "ordered_gates": {
            "G0_provenance_and_extraction": extraction_passed,
            "G1_compiled_known_answers": compiled_passed,
            "G2_compiled_input_and_runtime_pressure": bool(
                input_passed and runtime_passed and pressure_inventory_passed
            ),
            "G3_source_forward_closure": source_passed,
            "G4_exact_eight_shard_merge": True,
        },
        "runtime_pressure_gate": {
            "per_frame": per_frame_runtime,
            "negative_raw_Pe_count": negative_pe,
            "negative_raw_Pi_count": negative_pi,
            "negative_raw_Pi_count_by_shard": [
                result["runtime_pressure_gate"]["negative_raw_Pi_count"]
                for result in ordered
            ],
            "pressure_inventory_passed": pressure_inventory_passed,
            "passed": bool(runtime_passed and pressure_inventory_passed),
        },
        "source_forward_closure_gate": source_gate,
        "decision": {
            "all_frame_bidirectional_closure_validated": final_passed,
            "state_candidate_decision_may_proceed": final_passed,
            "establishes_predictive_sufficiency": False,
            "establishes_stationarity": False,
            "automatic_state_change_authorized": False,
            "automatic_training_authorized": False,
            "automatic_held_out_access_authorized": False,
        },
        "artifacts": {
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_sha,
            "protocol": str(args.protocol),
            "protocol_sha256": protocol_sha,
            "extraction_record": str(args.extraction_record),
            "extraction_record_sha256": extraction_sha,
            "shard_results": [str(path) for path in args.shard_result],
            "shard_result_sha256": {
                str(path): sha256_file(path) for path in args.shard_result
            },
            "merger": str(Path(__file__).resolve()),
            "merger_sha256": sha256_file(Path(__file__).resolve()),
            "artifact_root": str(artifact_root),
        },
    }
    write_strict_json(args.output, result)
    print(
        json.dumps(
            {
                "all_frame_bidirectional_closure_validated": final_passed,
                "frame_count": FRAME_COUNT,
                "negative_raw_Pi_count": negative_pi,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
