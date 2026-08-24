#!/usr/bin/env python3
"""Freeze prerequisite artifacts and authorize the old-85604 phi repair screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from paper0.tools.train_exact_state_phi_repair_screen import (
    build_model,
    verify_prerequisites,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import FAMILY_FIELDS


def locked(path: Path, digest: str) -> dict[str, str]:
    assert_development_path(path)
    if sha256_path(path) != digest:
        raise ValueError(f"input SHA-256 differs for {path}")
    return {"path": str(path), "sha256": digest}


def freeze_manifest(
    *,
    stage1_reduction: Path,
    stage1_reduction_sha256: str,
    axial_smoke: Path,
    axial_smoke_sha256: str,
    baseline_e6b: Path,
    baseline_e6b_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    reduction_lock = locked(stage1_reduction, stage1_reduction_sha256)
    smoke_lock = locked(axial_smoke, axial_smoke_sha256)
    baseline_lock = locked(baseline_e6b, baseline_e6b_sha256)
    baseline = load_strict_json(baseline_e6b)
    if baseline.get("family") != "e6b" or int(baseline.get("seed", -1)) != 1701:
        raise ValueError("baseline identity differs")
    baseline_metric = float(baseline["best_checkpoint"]["selection_metric"])
    architectures = {
        "local_current_phi": {
            "base_channels": 24,
            "channel_multipliers": [1, 2, 4],
            "blocks_per_level": 2,
            "lead_embedding_channels": 128,
            "group_norm_maximum_groups": 8,
            "kernel_size": 3,
            "zero_initialize_output": True,
        },
        "axial_current_phi": {
            "width": 104,
            "blocks": 4,
            "attention_heads": 4,
            "feedforward_expansion": 2,
            "lead_embedding_channels": 128,
            "group_norm_maximum_groups": 8,
            "kernel_size": 3,
            "zero_initialize_output": True,
        },
    }
    parameter_counts = {}
    for name, record in architectures.items():
        model, _ = build_model(name, record)
        parameter_counts[name] = sum(
            parameter.numel() for parameter in model.parameters()
        )
    relative_gap = (
        abs(
            parameter_counts["local_current_phi"]
            - parameter_counts["axial_current_phi"]
        )
        / parameter_counts["local_current_phi"]
    )
    if relative_gap >= 0.03:
        raise ValueError("repair-screen parameter budgets are not matched")
    manifest = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_exact_state_phi_repair_screen",
        "protocol": (
            "paper0/protocol/"
            "POST_ECRD_OLD_85604_EXACT_STATE_PHI_REPAIR_SCREEN_2026-08-24.md"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "screen_training_authorized": True,
        "three_seed_scaling_authorized": False,
        "paper0_commit_at_freeze": paper0_commit,
        "prerequisites": {
            "stage1_reduction": reduction_lock,
            "axial_smoke": smoke_lock,
            "baseline_e6b_seed1701": {
                **baseline_lock,
                "selection_metric": baseline_metric,
            },
        },
        "state": {
            "predicted_volume_fields": list(FAMILY_FIELDS["e6b"]),
            "predicted_boundary": "Bphi",
            "auxiliary_context_fields": ["phi"],
            "future_auxiliary_context_allowed": False,
            "current_phi_rollout_ready_without_elliptic_closure": False,
        },
        "split": {
            "training_frames": [0, 432],
            "guard_frames": [432, 496],
            "validation_frames": [496, 624],
            "training_pair_count": 431,
            "validation_pair_count": 127,
            "lead_steps": [1],
            "history_frames": 1,
        },
        "architectures": architectures,
        "parameter_counts": parameter_counts,
        "parameter_count_relative_gap": relative_gap,
        "optimization": {
            "screen_seed": 1701,
            "epochs": 12,
            "sample_batch_size": 1,
            "gradient_accumulation_samples": 4,
            "optimizer": "AdamW",
            "peak_learning_rate": 0.0002,
            "minimum_learning_rate": 0.00001,
            "weight_decay": 0.0001,
            "warmup_fraction": 0.05,
            "gradient_clip_norm": 1.0,
            "autocast": "bfloat16",
            "cudnn_tf32": False,
            "matmul_tf32": False,
        },
        "loss": {
            "name": "train_derivative_rms_persistence_normalized_direct_state_mse",
            "scale_fit_split": "train",
            "physics_derived_quantities_used": False,
        },
        "screen_gates": {
            "minimum_shared_mse_improvement_fraction": 0.15,
            "every_e6b_field_positive_persistence_skill": True,
            "mechanical_and_equivariance_gates_required": True,
        },
        "wandb_required": True,
    }
    verify_prerequisites(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-reduction", type=Path, required=True)
    parser.add_argument("--stage1-reduction-sha256", required=True)
    parser.add_argument("--axial-smoke", type=Path, required=True)
    parser.add_argument("--axial-smoke-sha256", required=True)
    parser.add_argument("--baseline-e6b", type=Path, required=True)
    parser.add_argument("--baseline-e6b-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    manifest = freeze_manifest(
        stage1_reduction=args.stage1_reduction,
        stage1_reduction_sha256=args.stage1_reduction_sha256,
        axial_smoke=args.axial_smoke,
        axial_smoke_sha256=args.axial_smoke_sha256,
        baseline_e6b=args.baseline_e6b,
        baseline_e6b_sha256=args.baseline_e6b_sha256,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
