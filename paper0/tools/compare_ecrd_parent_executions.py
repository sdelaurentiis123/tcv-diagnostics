#!/usr/bin/env python3
"""Compare CPU-smoke and scientific-H100 ECRD parents frame by frame."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_data import ECRDParentMeanArtifact
from tcv_diagnostics.model_data import (
    assert_development_path,
    write_strict_json_atomic,
)
from tcv_diagnostics.models.field_residual_edm import B5_FIELD_ORDER


ROOT = Path(__file__).resolve().parents[2]
ENGINEERING_GLOBAL_RELATIVE_RMS_GUARD = 5.0e-5
ENGINEERING_MAXIMUM_ABSOLUTE_GUARD = 5.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-train", type=Path, required=True)
    parser.add_argument("--cpu-train-sha256", required=True)
    parser.add_argument("--cpu-validation", type=Path, required=True)
    parser.add_argument("--cpu-validation-sha256", required=True)
    parser.add_argument("--h100-train", type=Path, required=True)
    parser.add_argument("--h100-train-sha256", required=True)
    parser.add_argument("--h100-validation", type=Path, required=True)
    parser.add_argument("--h100-validation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit mismatch: {actual} != {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


class ParentDifferenceAccumulator:
    """Stable per-field summaries for streamed parent tensors."""

    def __init__(self) -> None:
        fields = len(B5_FIELD_ORDER)
        self.element_count = np.zeros(fields, dtype=np.int64)
        self.exact_count = np.zeros(fields, dtype=np.int64)
        self.sum_squared_difference = np.zeros(fields, dtype=np.float64)
        self.sum_squared_h100 = np.zeros(fields, dtype=np.float64)
        self.sum_absolute_difference = np.zeros(fields, dtype=np.float64)
        self.maximum_absolute_difference = np.zeros(fields, dtype=np.float64)
        self.frame_relative_rms: list[float] = []

    def update(self, cpu: np.ndarray, h100: np.ndarray) -> None:
        cpu_values = np.asarray(cpu, dtype=np.float32)
        h100_values = np.asarray(h100, dtype=np.float32)
        if (
            cpu_values.shape != h100_values.shape
            or cpu_values.ndim != 5
            or cpu_values.shape[1] != len(B5_FIELD_ORDER)
            or not np.all(np.isfinite(cpu_values))
            or not np.all(np.isfinite(h100_values))
        ):
            raise ValueError("ECRD parent comparison tensor differs")
        frame_difference_squared = 0.0
        frame_h100_squared = 0.0
        for field in range(len(B5_FIELD_ORDER)):
            cpu_field = cpu_values[:, field].astype(np.float64, copy=False)
            h100_field = h100_values[:, field].astype(np.float64, copy=False)
            difference = cpu_field - h100_field
            absolute = np.abs(difference)
            difference_squared = float(np.sum(difference * difference, dtype=np.float64))
            h100_squared = float(np.sum(h100_field * h100_field, dtype=np.float64))
            count = int(difference.size)
            self.element_count[field] += count
            self.exact_count[field] += int(np.count_nonzero(difference == 0.0))
            self.sum_squared_difference[field] += difference_squared
            self.sum_squared_h100[field] += h100_squared
            self.sum_absolute_difference[field] += float(
                np.sum(absolute, dtype=np.float64)
            )
            self.maximum_absolute_difference[field] = max(
                self.maximum_absolute_difference[field],
                float(np.max(absolute)),
            )
            frame_difference_squared += difference_squared
            frame_h100_squared += h100_squared
        self.frame_relative_rms.append(
            math.sqrt(frame_difference_squared / max(frame_h100_squared, 1.0e-300))
        )

    def to_record(self) -> dict[str, Any]:
        total_elements = int(np.sum(self.element_count))
        total_exact = int(np.sum(self.exact_count))
        total_difference_squared = float(np.sum(self.sum_squared_difference))
        total_h100_squared = float(np.sum(self.sum_squared_h100))
        total_absolute = float(np.sum(self.sum_absolute_difference))
        relative = np.sqrt(
            self.sum_squared_difference / np.maximum(self.sum_squared_h100, 1.0e-300)
        )
        records = {}
        for index, field in enumerate(B5_FIELD_ORDER):
            records[field] = {
                "element_count": int(self.element_count[index]),
                "exact_fraction": float(
                    self.exact_count[index] / self.element_count[index]
                ),
                "mean_absolute_difference": float(
                    self.sum_absolute_difference[index] / self.element_count[index]
                ),
                "maximum_absolute_difference": float(
                    self.maximum_absolute_difference[index]
                ),
                "relative_RMS_difference": float(relative[index]),
            }
        frame_values = np.asarray(self.frame_relative_rms, dtype=np.float64)
        global_relative = math.sqrt(
            total_difference_squared / max(total_h100_squared, 1.0e-300)
        )
        global_maximum = float(np.max(self.maximum_absolute_difference))
        return {
            "frame_count": len(self.frame_relative_rms),
            "element_count": total_elements,
            "exact_fraction": float(total_exact / total_elements),
            "mean_absolute_difference": float(total_absolute / total_elements),
            "maximum_absolute_difference": global_maximum,
            "relative_RMS_difference": global_relative,
            "frame_relative_RMS_difference": {
                "minimum": float(np.min(frame_values)),
                "median": float(np.median(frame_values)),
                "maximum": float(np.max(frame_values)),
            },
            "by_field": records,
            "engineering_consistency_guard": {
                "global_relative_RMS_limit": ENGINEERING_GLOBAL_RELATIVE_RMS_GUARD,
                "maximum_absolute_limit": ENGINEERING_MAXIMUM_ABSOLUTE_GUARD,
                "passed": bool(
                    global_relative <= ENGINEERING_GLOBAL_RELATIVE_RMS_GUARD
                    and global_maximum <= ENGINEERING_MAXIMUM_ABSOLUTE_GUARD
                ),
                "diagnostic_only": True,
                "may_promote_CPU_parent": False,
            },
        }


def compare_split(
    cpu: ECRDParentMeanArtifact,
    h100: ECRDParentMeanArtifact,
) -> dict[str, Any]:
    if cpu.split != h100.split or cpu.target_frames != h100.target_frames:
        raise ValueError("ECRD parent comparison split differs")
    accumulator = ParentDifferenceAccumulator()
    for index in range(len(cpu.target_frames)):
        accumulator.update(cpu.read(index, index + 1), h100.read(index, index + 1))
    return accumulator.to_record()


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    paths = (
        args.cpu_train,
        args.cpu_validation,
        args.h100_train,
        args.h100_validation,
        args.output,
    )
    if any("85606" in str(path).lower() for path in paths):
        raise ValueError("held-out paths are prohibited during parent comparison")
    output = Path(args.output)
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(output)

    sources = {
        "CPU_train": (args.cpu_train, args.cpu_train_sha256, "train"),
        "CPU_validation": (
            args.cpu_validation,
            args.cpu_validation_sha256,
            "validation",
        ),
        "H100_train": (args.h100_train, args.h100_train_sha256, "train"),
        "H100_validation": (
            args.h100_validation,
            args.h100_validation_sha256,
            "validation",
        ),
    }
    with ExitStack() as stack:
        artifacts = {
            name: stack.enter_context(
                ECRDParentMeanArtifact(path, split=split, expected_sha256=digest)
            )
            for name, (path, digest, split) in sources.items()
        }
        for name in ("CPU_train", "CPU_validation"):
            if (
                artifacts[name].artifact_authority
                != "bounded_non_scientific_engineering_smoke_only"
                or artifacts[name].execution_device != "cpu-smoke"
            ):
                raise RuntimeError("CPU-smoke parent authority differs")
        for name in ("H100_train", "H100_validation"):
            if (
                artifacts[name].artifact_authority != "scientific_H100_parent"
                or artifacts[name].execution_device != "h100"
            ):
                raise RuntimeError("scientific H100 parent authority differs")
        comparisons = {
            "train": compare_split(artifacts["CPU_train"], artifacts["H100_train"]),
            "validation": compare_split(
                artifacts["CPU_validation"], artifacts["H100_validation"]
            ),
        }
        metadata = {
            name: {
                "path": str(Path(path).resolve(strict=True)),
                "sha256": digest,
                "artifact_authority": artifacts[name].artifact_authority,
                "execution_device": artifacts[name].execution_device,
            }
            for name, (path, digest, _) in sources.items()
        }

    output.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "scope": "ECRD_CPU_smoke_to_scientific_H100_parent_framewise_comparison_85604",
        "status": "comparison_completed_H100_remains_only_scientific_parent",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "development_run": "85604",
        "sources": metadata,
        "comparisons": comparisons,
        "comparison_completed_for_CPU_smoke_amendment": True,
        "CPU_parent_promoted": False,
        "full_training_parent_authority": "scientific_H100_parent_only",
        "scientific_result": False,
        "physics_metric_evaluated": False,
        "target_truth_read": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "training_performed": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        f"{sha256_path(result_path)}  {result_path.resolve(strict=True)}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
