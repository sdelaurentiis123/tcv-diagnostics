#!/usr/bin/env python3
"""Freeze gauge-consistent B2 comparators from immutable C5P-H2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b2_acceptance import (  # noqa: E402
    DeterministicFieldComparatorAccumulator,
    deterministic_transport_comparator,
)
from tcv_diagnostics.b2_field_scoring import (  # noqa: E402
    B2_VALIDATION_BLOCKS,
    B2_VALIDATION_TARGETS,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.matched_o1_transport import load_transport_geometry  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import load_official_catalog  # noqa: E402
from tcv_diagnostics.o2_forecast import O2ForecastArtifact  # noqa: E402
from tcv_diagnostics.o2_training_data import OneStepWindowDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o2-final-matrix", type=Path, required=True)
    parser.add_argument("--o2-final-matrix-sha256", required=True)
    parser.add_argument("--o2-audit-result", type=Path, required=True)
    parser.add_argument("--o2-audit-result-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
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
    if actual != str(expected_commit):
        raise RuntimeError(f"Paper 0 commit {actual} differs from {expected_commit}")
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
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def verify_input(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    actual = sha256_path(resolved)
    if actual != str(expected_sha256):
        raise ValueError(f"SHA-256 mismatch for {resolved}: {actual}")
    return resolved


def _validate_o2_matrix(matrix: dict[str, object]) -> list[dict[str, object]]:
    if (
        matrix.get("scope") != "phase2_C5P_O2_complete_scientific_matrix"
        or matrix.get("status") != "completed"
        or matrix.get("development_run") != "85604"
        or matrix.get("held_out_85606_read") is not False
        or matrix.get("guard_frames_read") is not False
        or matrix.get("run_count") != 6
    ):
        raise ValueError("frozen O2 final matrix contract differs")
    runs = [
        item
        for item in matrix.get("runs", [])  # type: ignore[union-attr]
        if item.get("arm") == "C5P-H2"
    ]
    if [int(item.get("seed", -1)) for item in runs] != [1701, 1702, 1703]:
        raise ValueError("frozen O2 C5P-H2 seed matrix differs")
    return runs


def _validate_audit(audit: dict[str, object]) -> float:
    if (
        audit.get("scope") != "phase2_C5P_O2_full_85604_scientific_evaluation_audit"
        or audit.get("development_run") != "85604"
        or audit.get("held_out_85606_read") is not False
        or audit.get("guard_frames_read") is not False
    ):
        raise ValueError("frozen O2 audit contract differs")
    references = audit.get("references", {})
    if references.get("best_applicable_for_both_arms") != "spectral_ar1":
        raise ValueError("best frozen uncompressed comparator differs")
    value = float(references["spectral_ar1_aggregate_equal_channel_MAE_standardized"])
    if value != 0.056300767439895476:
        raise ValueError("best frozen uncompressed comparator value differs")
    return value


def main() -> None:
    args = parse_args()
    for path in (
        args.o2_final_matrix,
        args.o2_audit_result,
        args.artifact_root,
        args.geometry_manifest,
        args.geometry,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    matrix_path = verify_input(
        args.o2_final_matrix, args.o2_final_matrix_sha256
    )
    audit_path = verify_input(args.o2_audit_result, args.o2_audit_result_sha256)
    geometry_manifest_path = verify_input(
        args.geometry_manifest, args.geometry_manifest_sha256
    )
    geometry_path = verify_input(args.geometry, args.geometry_sha256)
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 comparators {output}")
    output.mkdir(parents=True)

    matrix = load_strict_json(matrix_path)
    runs = _validate_o2_matrix(matrix)
    best_uncompressed_mae = _validate_audit(load_strict_json(audit_path))
    catalog = load_official_catalog(args.artifact_root)
    geometry = load_transport_geometry(
        geometry_path=geometry_path,
        geometry_manifest=load_strict_json(geometry_manifest_path),
    )
    eligible_2d = (
        geometry.region_masks.strict_wall_interior
        & geometry.region_masks.operator_interior
    )
    eligible_3d = np.broadcast_to(eligible_2d[..., None], (64, 32, 88))

    frozen_runs = []
    for run in runs:
        seed = int(run["seed"])
        score_record = run["score"]
        forecast_record = run["forecast"]
        score_path = verify_input(
            Path(score_record["path"]), str(score_record["sha256"])
        )
        score = load_strict_json(score_path)
        forecast_path = verify_input(
            Path(forecast_record["path"]), str(forecast_record["sha256"])
        )
        if (
            score.get("forecast_artifact", {}).get("path") != str(forecast_path)
            or score.get("forecast_artifact", {}).get("sha256")
            != str(forecast_record["sha256"])
            or score.get("forecast_artifact", {}).get("metadata", {}).get("seed")
            != seed
            or score.get("forecast_artifact", {}).get("metadata", {}).get("arm")
            != "C5P-H2"
        ):
            raise ValueError("deterministic score/forecast provenance differs")
        accumulator = DeterministicFieldComparatorAccumulator(
            target_frames=B2_VALIDATION_TARGETS,
            eligible_mask=eligible_3d,
            validation_blocks=B2_VALIDATION_BLOCKS,
        )
        truth = OneStepWindowDataset(
            catalog,
            split="validation",
            target_frames=B2_VALIDATION_TARGETS,
            context_frames=2,
            augment=False,
            seed=1701,
            return_physical=False,
        )
        try:
            with O2ForecastArtifact(
                forecast_path,
                expected_sha256=str(forecast_record["sha256"]),
                target_frames=B2_VALIDATION_TARGETS,
            ) as artifact:
                for position, target in enumerate(B2_VALIDATION_TARGETS):
                    item = truth[position]
                    if int(item["target_frame_index"]) != target:
                        raise RuntimeError("deterministic comparator truth order differs")
                    accumulator.update(
                        target_frame=target,
                        standardized_forecast=artifact.read(position, position + 1)[0],
                        standardized_truth=item["target"],
                    )
        finally:
            truth.close()
        frozen_runs.append(
            {
                "seed": seed,
                "selected_checkpoint": dict(run["selected_checkpoint"]),
                "forecast": dict(forecast_record),
                "score": dict(score_record),
                "field": accumulator.finalize(),
                "transport": deterministic_transport_comparator(score),
            }
        )

    result = {
        "schema_version": 1,
        "scope": "phase3_B2_frozen_paired_deterministic_comparators_85604",
        "status": "completed_before_B2_scientific_acceptance",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "B2_forecasts_or_scores_read": False,
        "deterministic_model_retrained": False,
        "deterministic_checkpoint_reselected": False,
        "physics_derived_training_loss_used": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "best_uncompressed_aggregate_equal_channel_mae_standardized": (
            best_uncompressed_mae
        ),
        "o2_final_matrix": {
            "path": str(matrix_path),
            "sha256": args.o2_final_matrix_sha256,
        },
        "o2_audit_result": {
            "path": str(audit_path),
            "sha256": args.o2_audit_result_sha256,
        },
        "model_dataset": {
            "root": str(args.artifact_root.resolve(strict=True)),
            "manifest_sha256": sha256_path(
                args.artifact_root / "model_dataset_manifest.json"
            ),
            "normalization_sha256": sha256_path(
                args.artifact_root / "normalization.json"
            ),
        },
        "geometry_manifest": {
            "path": str(geometry_manifest_path),
            "sha256": args.geometry_manifest_sha256,
        },
        "runs": frozen_runs,
        "seed_count": len(frozen_runs),
        "seeds": [item["seed"] for item in frozen_runs],
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    result_path = output / "comparators.json"
    write_strict_json_atomic(result_path, result)
    index_path = output / "artifact_sha256.txt"
    index_path.write_text(
        f"{sha256_path(result_path)}  {result_path.resolve(strict=True)}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(result_path.resolve(strict=True)),
                "sha256": sha256_path(result_path),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
