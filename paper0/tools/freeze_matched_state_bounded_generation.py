#!/usr/bin/env python3
"""Freeze causal bounded generation after the matched transition pair passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from paper0.tools.generate_matched_state_bounded_forecasts import (
    FAMILIES,
    REDUCTION_SCOPE,
    SCOPE,
)
from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.bounded_rollout import method_schedule
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


def locked_json(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return load_strict_json(path)


def freeze_manifest(
    *,
    paired_reduction: Path,
    paired_reduction_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    reduction_lock = {
        "path": str(paired_reduction),
        "sha256": paired_reduction_sha256,
    }
    reduction = locked_json(reduction_lock, label="paired transition reduction")
    if (
        reduction.get("scope") != REDUCTION_SCOPE
        or reduction.get("status") != "completed"
        or reduction.get("development_run") != "85604"
        or reduction.get("held_out_85606_read") is not False
        or reduction.get("new_nersc_data_read") is not False
        or reduction.get("paired_physics_evaluation_authorized") is not True
        or reduction.get("decision")
        != "run_causal_paired_derived_field_physics_evaluation"
    ):
        raise ValueError("paired reduction does not authorize generation")

    models: dict[str, Any] = {}
    result_locks = reduction.get("results", {})
    if set(result_locks) != set(FAMILIES):
        raise ValueError("paired reduction result set differs")
    for family in FAMILIES:
        result_lock = result_locks[family]
        result = locked_json(result_lock, label=f"{family} training result")
        selected = result.get("best_checkpoint", {})
        checkpoint_path = Path(str(selected.get("path", "")))
        checkpoint_sha = str(selected.get("sha256", ""))
        assert_development_path(checkpoint_path)
        if not checkpoint_sha or sha256_path(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"{family} selected checkpoint SHA-256 differs")
        if (
            result.get("status") != "passed"
            or result.get("family") != family
            or int(result.get("seed", -1)) != 1701
            or result.get("held_out_85606_read") is not False
            or result.get("new_nersc_data_read") is not False
            or result.get("training_gate", {}).get("passed") is not True
            or result.get("transition_gate", {}).get("passed") is not True
        ):
            raise ValueError(f"{family} training result no longer passes")
        models[family] = {
            "family": family,
            "seed": 1701,
            "result": dict(result_lock),
            "selected_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "epoch": int(selected["epoch"]),
                "selection_metric": float(selected["selection_metric"]),
            },
        }

    return {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "frozen_after_paired_transition_reduction_before_inference",
        "protocol": (
            "paper0/protocol/"
            "POST_ECRD_OLD_85604_MATCHED_STATE_MULTILEAD_PROTOCOL_2026-08-26.md"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "physics_derived_training_loss_allowed": False,
        "paper0_commit_at_freeze": paper0_commit,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "evidence": {
            "paired_reduction": reduction_lock,
            "models": models,
        },
        "evaluation": {
            "validation_frames": [496, 624],
            "history_frames": 1,
            "inference_batch_size": 4,
            "target_truth_used_during_generation": False,
            "complete_predicted_state_fed_back": True,
            "e6b_intermediate_phi_required": False,
            "e6b_boundary_source": "predicted_Bphi",
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
        },
        "downstream": {
            "e6b_exact_phi_required": True,
            "exact_phi_inputs": ["predicted_Ne", "predicted_Pi", "predicted_Vort", "predicted_Bphi"],
            "exact_phi_future_truth_allowed": False,
            "physics_scoring_allowed_only_after_exact_phi": True,
        },
        "wandb_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-reduction", type=Path, required=True)
    parser.add_argument("--paired-reduction-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = freeze_manifest(
        paired_reduction=args.paired_reduction,
        paired_reduction_sha256=args.paired_reduction_sha256,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
