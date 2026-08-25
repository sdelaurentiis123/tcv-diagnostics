#!/usr/bin/env python3
"""Freeze the authorized old-85604 bounded rollout inputs and methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.bounded_rollout import FIELDS, method_schedule
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_INDEX_SHA256,
    OFFICIAL_DATASET_RESULT_SHA256,
    OFFICIAL_NORMALIZATION_SHA256,
)


SCOPE = "post_ecrd_old_85604_bounded_rollout"


def _lock(path: Path, digest: str, *, label: str) -> dict[str, str]:
    source = Path(path)
    assert_development_path(source)
    if sha256_path(source) != str(digest):
        raise ValueError(f"{label} SHA-256 differs")
    return {"path": str(source), "sha256": str(digest)}


def _model_lock(path: Path, digest: str, *, seed: int) -> dict[str, Any]:
    result_lock = _lock(path, digest, label=f"seed-{seed} result")
    result = load_strict_json(path)
    expected_scope = (
        "post_ecrd_old_85604_stage2_multilead_screen"
        if seed == 1701
        else "post_ecrd_old_85604_stage2_multilead_scaling"
    )
    confirmation = (
        result.get("advance_to_three_seed_scaling")
        if seed == 1701
        else result.get("seed_confirmation_passed")
    )
    if (
        result.get("scope") != expected_scope
        or result.get("status") != "passed"
        or result.get("development_run") != "85604"
        or result.get("held_out_85606_read") is not False
        or result.get("physics_derived_loss_used") is not False
        or int(result.get("seed", -1)) != seed
        or confirmation is not True
    ):
        raise ValueError(f"seed-{seed} result contract differs")
    checkpoint = result.get("best_checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    checkpoint_sha = str(checkpoint.get("sha256", ""))
    checkpoint_lock = _lock(
        checkpoint_path,
        checkpoint_sha,
        label=f"seed-{seed} selected checkpoint",
    )
    if int(checkpoint.get("epoch", -1)) != 4:
        raise ValueError(f"seed-{seed} selected epoch differs")
    return {
        "seed": seed,
        "result": result_lock,
        "training_commit": str(result["paper0_commit"]),
        "selected_checkpoint": {
            **checkpoint_lock,
            "epoch": 4,
            "selection_metric": float(checkpoint["selection_metric"]),
        },
    }


def freeze_manifest(
    *,
    reduction: Path,
    reduction_sha256: str,
    result_paths: dict[int, Path],
    result_sha256: dict[int, str],
    artifact_root: Path,
    native_truth_result: Path,
    native_truth_result_sha256: str,
    geometry_manifest: Path,
    geometry_manifest_sha256: str,
    geometry: Path,
    geometry_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    reduction_lock = _lock(reduction, reduction_sha256, label="three-seed reduction")
    reduction_record = load_strict_json(reduction)
    if (
        reduction_record.get("development_run") != "85604"
        or reduction_record.get("held_out_85606_read") is not False
        or reduction_record.get("new_nersc_data_read") is not False
        or reduction_record.get("three_seed_mechanism_confirmed") is not True
        or reduction_record.get("bounded_rollout_authorized") is not True
        or reduction_record.get("decision")
        != "freeze_bounded_direct_vs_autoregressive_validation"
    ):
        raise ValueError("three-seed reduction does not authorize rollout")
    models = {
        str(seed): _model_lock(result_paths[seed], result_sha256[seed], seed=seed)
        for seed in (1701, 1702, 1703)
    }

    root = Path(artifact_root)
    dataset_locks = {
        "manifest": _lock(
            root / "model_dataset_manifest.json",
            OFFICIAL_DATASET_RESULT_SHA256,
            label="official model dataset manifest",
        ),
        "normalization": _lock(
            root / "normalization.json",
            OFFICIAL_NORMALIZATION_SHA256,
            label="official model normalization",
        ),
        "artifact_index": _lock(
            root / "artifact_sha256.txt",
            OFFICIAL_ARTIFACT_INDEX_SHA256,
            label="official model dataset index",
        ),
    }
    native_lock = _lock(
        native_truth_result,
        native_truth_result_sha256,
        label="native truth result",
    )
    native_record = load_strict_json(native_truth_result)
    if (
        native_record.get("development_run") != "85604"
        or native_record.get("held_out_85606_read") is not False
    ):
        raise ValueError("native truth result scope differs")
    geometry_manifest_lock = _lock(
        geometry_manifest,
        geometry_manifest_sha256,
        label="geometry manifest",
    )
    geometry_record = load_strict_json(geometry_manifest)
    geometry_lock = _lock(geometry, geometry_sha256, label="geometry")
    if (
        geometry_record.get("development_run") != "85604"
        or geometry_record.get("sources", {}).get("geometry", {}).get("sha256")
        != geometry_sha256
    ):
        raise ValueError("geometry manifest scope differs")

    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "frozen_after_three_seed_confirmation_before_rollout_inference",
        "protocol": (
            "paper0/protocol/"
            "POST_ECRD_OLD_85604_STAGE2_SCALING_ROLLOUT_AMENDMENT_2026-08-25.md"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "paper0_commit_at_freeze": str(paper0_commit),
        "fields": list(FIELDS),
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "evidence": {
            "three_seed_reduction": reduction_lock,
            "models": models,
            "dataset_root": str(root),
            "dataset_locks": dataset_locks,
            "native_truth_result": native_lock,
            "geometry_manifest": geometry_manifest_lock,
            "geometry": geometry_lock,
        },
        "evaluation": {
            "validation_frames": [496, 624],
            "history_frames": 1,
            "horizons": {
                "4": {
                    "current_frames": [496, 620],
                    "target_frames": [500, 624],
                    "pair_count": 124,
                    "methods": method_schedule(4),
                },
                "8": {
                    "current_frames": [496, 616],
                    "target_frames": [504, 624],
                    "pair_count": 120,
                    "methods": method_schedule(8),
                },
            },
            "same_starts_and_targets_within_terminal_horizon": True,
            "intermediate_or_future_truth_used_as_model_input": False,
            "complete_predicted_five_field_state_fed_back": True,
            "state_metrics": [
                "standardized_per_field_RMSE",
                "standardized_per_field_persistence_relative_skill",
                "error_by_composition_depth",
                "seed_minimum_median_maximum",
            ],
            "physics_metrics_are_evaluation_only": True,
            "physics_metrics": [
                "directional_toroidal_spectra",
                "density_pressure_potential_cross_spectrum",
                "cross_field_coherence",
                "local_strict_face_transport",
                "integrated_separatrix_transport",
            ],
            "forecast_artifact": {
                "stored_value": "standardized_terminal_state_delta_from_current",
                "dtype": "float32",
                "compression": "gzip_level_4_shuffle_fletcher32",
                "truth_stored_in_forecast_artifact": False,
            },
            "example_start_frame": 560,
            "inference_batch_size": 4,
        },
        "wandb_required": True,
        "claims_not_authorized": [
            "long_autonomous_rollout",
            "stochastic_ensemble_calibration",
            "assimilation",
            "diagnostic_ranking",
            "steering",
            "85606_generalization",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduction", type=Path, required=True)
    parser.add_argument("--reduction-sha256", required=True)
    for seed in (1701, 1702, 1703):
        parser.add_argument(f"--seed{seed}-result", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-result-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    manifest = freeze_manifest(
        reduction=args.reduction,
        reduction_sha256=args.reduction_sha256,
        result_paths={
            seed: getattr(args, f"seed{seed}_result") for seed in (1701, 1702, 1703)
        },
        result_sha256={
            seed: getattr(args, f"seed{seed}_result_sha256")
            for seed in (1701, 1702, 1703)
        },
        artifact_root=args.artifact_root,
        native_truth_result=args.native_truth_result,
        native_truth_result_sha256=args.native_truth_result_sha256,
        geometry_manifest=args.geometry_manifest,
        geometry_manifest_sha256=args.geometry_manifest_sha256,
        geometry=args.geometry,
        geometry_sha256=args.geometry_sha256,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
