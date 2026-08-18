#!/usr/bin/env python3
"""Build the frozen gauge-consistent H1 and uncompressed comparators for B3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

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


EXPECTED_MANIFEST_SHA256 = (
    "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
)
EXPECTED_B2_COMPARATORS_SHA256 = (
    "2e96359cf2213d62ea81c7bec33e30551ade6fd081ca88a1aa088f65d84de72e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
    parser.add_argument("--b2-comparators", type=Path, required=True)
    parser.add_argument("--b2-comparators-sha256", required=True)
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


def comparator_inputs_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        manifest.get("protocol_status")
        != (
            "frozen_after_passing_B3_smoke_before_full_training_or_scientific_"
            "evaluation_implementation"
        )
        or manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise ValueError("B3 comparator manifest scope differs")
    parent = manifest.get("comparators", {}).get("primary_deterministic_parent", {})
    if (
        parent.get("arm") != "C5P-H1"
        or parent.get("seed") != 1701
        or parent.get("forecast_sha256")
        != "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
        or parent.get("score_sha256")
        != "ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593"
        or parent.get("retraining_or_reselection_allowed") is not False
    ):
        raise ValueError("B3 deterministic H1 comparator lock differs")
    uncompressed = manifest.get("comparators", {}).get("uncompressed")
    if uncompressed != ["persistence", "training_only_toroidal_spectral_AR1"]:
        raise ValueError("B3 uncompressed comparator lock differs")
    return dict(parent), {"names": list(uncompressed)}


def frozen_best_uncompressed(matrix: Mapping[str, Any]) -> dict[str, Any]:
    if (
        matrix.get("scope")
        != "phase3_B2_frozen_paired_deterministic_comparators_85604"
        or matrix.get("status") != "completed_before_B2_scientific_acceptance"
        or matrix.get("development_run") != "85604"
        or matrix.get("held_out_85606_read") is not False
        or matrix.get("B2_forecasts_or_scores_read") is not False
        or matrix.get("scientific_acceptance_evaluated") is not False
    ):
        raise ValueError("frozen B2 comparator artifact contract differs")
    best = matrix.get("best_uncompressed", {})
    if (
        best.get("name") != "training_only_toroidal_spectral_AR1"
        or best.get("field", {}).get("target_frames") != [498, 624]
        or best.get("field", {}).get("potential_policy")
        != "subtract_full_spatial_mean_separately_per_forecast_and_truth_target"
    ):
        raise ValueError("frozen uncompressed comparator contract differs")
    return dict(best)


def build_h1_field_comparator(
    *,
    catalog: Any,
    eligible_mask: np.ndarray,
    forecast_path: Path,
    forecast_sha256: str,
) -> dict[str, Any]:
    accumulator = DeterministicFieldComparatorAccumulator(
        target_frames=B2_VALIDATION_TARGETS,
        eligible_mask=eligible_mask,
        validation_blocks=B2_VALIDATION_BLOCKS,
    )
    truth = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=B2_VALIDATION_TARGETS,
        context_frames=1,
        augment=False,
        seed=1701,
        return_physical=False,
    )
    try:
        with O2ForecastArtifact(
            forecast_path,
            expected_sha256=forecast_sha256,
            target_frames=B2_VALIDATION_TARGETS,
        ) as artifact:
            for position, target in enumerate(B2_VALIDATION_TARGETS):
                item = truth[position]
                if int(item["target_frame_index"]) != target:
                    raise RuntimeError("H1 comparator truth order differs")
                accumulator.update(
                    target_frame=target,
                    standardized_forecast=artifact.read(position, position + 1)[0],
                    standardized_truth=item["target"],
                )
    finally:
        truth.close()
    record = accumulator.finalize()
    record["scope"] = "gauge_consistent_deterministic_C5P_H1_field_comparator"
    record["context_frames"] = 1
    return record


def main() -> None:
    args = parse_args()
    for path in (
        args.evaluation_manifest,
        args.b2_comparators,
        args.artifact_root,
        args.geometry_manifest,
        args.geometry,
        args.output_directory,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    if args.evaluation_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("B3 comparator manifest hash differs")
    b2_path = verify_input(args.b2_comparators, args.b2_comparators_sha256)
    if args.b2_comparators_sha256 != EXPECTED_B2_COMPARATORS_SHA256:
        raise RuntimeError("B3 reused-comparator hash differs")
    geometry_manifest_path = verify_input(
        args.geometry_manifest, args.geometry_manifest_sha256
    )
    geometry_path = verify_input(args.geometry, args.geometry_sha256)
    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B3 comparators {output}")
    output.mkdir(parents=True)

    manifest = load_strict_json(manifest_path)
    parent, uncompressed_lock = comparator_inputs_from_manifest(manifest)
    best_uncompressed = frozen_best_uncompressed(load_strict_json(b2_path))
    forecast_path = verify_input(
        Path(parent["forecast_path"]), parent["forecast_sha256"]
    )
    score_path = verify_input(Path(parent["score_path"]), parent["score_sha256"])
    score = load_strict_json(score_path)
    forecast_metadata = score.get("forecast_artifact", {}).get("metadata", {})
    if (
        score.get("scope") != "O2_truth_separated_forecast_scoring"
        or score.get("development_run") != "85604"
        or score.get("held_out_85606_read") is not False
        or score.get("target_frames") != [498, 624]
        or forecast_metadata.get("arm") != "C5P-H1"
        or forecast_metadata.get("seed") != 1701
        or forecast_metadata.get("context_frames") != 1
        or score.get("forecast_artifact", {}).get("sha256")
        != parent["forecast_sha256"]
    ):
        raise ValueError("frozen deterministic H1 score contract differs")

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
    field = build_h1_field_comparator(
        catalog=catalog,
        eligible_mask=eligible_3d,
        forecast_path=forecast_path,
        forecast_sha256=parent["forecast_sha256"],
    )
    transport = deterministic_transport_comparator(score)
    transport["scope"] = (
        "frozen_gauge_invariant_deterministic_C5P_H1_transport_comparator"
    )
    result = {
        "schema_version": 1,
        "scope": "phase3_B3_frozen_matched_H1_comparators_85604",
        "status": "completed_before_B3_scientific_acceptance",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "B3_forecasts_or_scores_read": False,
        "deterministic_model_retrained": False,
        "deterministic_checkpoint_reselected": False,
        "uncompressed_reference_reselected": False,
        "scientific_acceptance_evaluated": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "seed": 1701,
        "arm": "C5P-H1",
        "forecast": {
            "path": str(forecast_path),
            "sha256": parent["forecast_sha256"],
        },
        "score": {"path": str(score_path), "sha256": parent["score_sha256"]},
        "field": field,
        "transport": transport,
        "best_uncompressed": best_uncompressed,
        "uncompressed_manifest_lock": uncompressed_lock,
        "source_b2_comparators": {
            "path": str(b2_path),
            "sha256": args.b2_comparators_sha256,
            "only_best_uncompressed_record_reused": True,
        },
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
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
