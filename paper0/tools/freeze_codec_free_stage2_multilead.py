#!/usr/bin/env python3
"""Freeze the old-85604 C5P multi-lead fine-tuning screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from paper0.tools.train_codec_free_stage2_multilead import (
    LEADS,
    build_model,
    verify_prerequisites,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


PARENT_ONE_STEP_SHARED_MSE = 0.005322341561633884
PARENT_CHECKPOINT_SHA256 = (
    "887dfcbe37e214f7217a5d4b900381cea370ca2e2c96687d2d6cd92c9e951c33"
)


def locked(path: Path, digest: str) -> dict[str, str]:
    """Validate and record an immutable development-only input."""

    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"input SHA-256 differs for {path}")
    return {"path": str(path), "sha256": digest}


def freeze_manifest(
    *,
    stage1_reduction: Path,
    stage1_reduction_sha256: str,
    parent_result: Path,
    parent_result_sha256: str,
    parent_checkpoint: Path,
    parent_checkpoint_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    """Create the prospective single-seed Stage-2 manifest."""

    reduction_lock = locked(stage1_reduction, stage1_reduction_sha256)
    reduction = load_strict_json(stage1_reduction)
    if reduction.get("development_run") != "85604":
        raise ValueError("Stage-1 reduction development run differs")
    if reduction.get("held_out_85606_read") is not False:
        raise ValueError("Stage-1 reduction held-out flag differs")
    if reduction.get("decision") != (
        "retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation"
    ):
        raise ValueError("Stage-1 reduction does not authorize C5P Stage 2")

    result_lock = locked(parent_result, parent_result_sha256)
    result = load_strict_json(parent_result)
    if result.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("parent result scope differs")
    if result.get("development_run") != "85604":
        raise ValueError("parent result development run differs")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("parent result held-out flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise ValueError("parent result physics-loss flag differs")
    if result.get("family") != "c5p" or int(result.get("seed", -1)) != 1701:
        raise ValueError("parent result identity differs")
    if result.get("status") != "passed":
        raise ValueError("parent result did not pass")
    if result.get("training_gate", {}).get("passed") is not True:
        raise ValueError("parent training gate did not pass")
    selected = result.get("best_checkpoint", {})
    if float(selected.get("selection_metric", float("nan"))) != (
        PARENT_ONE_STEP_SHARED_MSE
    ):
        raise ValueError("parent one-step selection metric differs")
    if Path(str(selected.get("path", ""))) != parent_checkpoint:
        raise ValueError("parent result selects another checkpoint")
    if selected.get("sha256") != parent_checkpoint_sha256:
        raise ValueError("parent result checkpoint hash differs")
    checkpoint_lock = locked(parent_checkpoint, parent_checkpoint_sha256)
    if parent_checkpoint_sha256 != PARENT_CHECKPOINT_SHA256:
        raise ValueError("parent checkpoint is not the prospectively selected artifact")

    architecture = {
        "base_channels": 24,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "lead_embedding_channels": 128,
        "group_norm_maximum_groups": 8,
        "kernel_size": 3,
        "zero_initialize_output": True,
    }
    model, config = build_model(architecture)
    payload = torch.load(parent_checkpoint, map_location="cpu", weights_only=True)
    if payload.get("family") != "c5p" or int(payload.get("seed", -1)) != 1701:
        raise ValueError("parent checkpoint identity differs")
    if payload.get("config") != config.to_record():
        raise ValueError("parent checkpoint architecture differs")
    if float(payload.get("selection_metric", float("nan"))) != (
        PARENT_ONE_STEP_SHARED_MSE
    ):
        raise ValueError("parent checkpoint metric differs")
    model.load_state_dict(payload["model"], strict=True)

    manifest = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_stage2_multilead_screen",
        "protocol": (
            "paper0/protocol/"
            "POST_ECRD_OLD_85604_STAGE2_MULTILEAD_PROTOCOL_2026-08-25.md"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "screen_training_authorized": True,
        "three_seed_scaling_authorized": False,
        "paper0_commit_at_freeze": paper0_commit,
        "state_family": "c5p",
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "prerequisites": {
            "stage1_reduction": reduction_lock,
            "parent_result": result_lock,
        },
        "parent": {
            "family": "c5p",
            "seed": 1701,
            "checkpoint": checkpoint_lock,
            "one_step_selection_metric": PARENT_ONE_STEP_SHARED_MSE,
            "load_state_dictionary_strictly": True,
            "require_bitwise_weight_equality": True,
        },
        "split": {
            "training_frames": [0, 432],
            "guard_frames": [432, 496],
            "validation_frames": [496, 624],
            "lead_steps": list(LEADS),
            "history_frames": 1,
            "training_pair_count": 2129,
            "validation_pair_count": 609,
            "training_pairs_by_lead": {
                "1": 431,
                "2": 430,
                "4": 428,
                "8": 424,
                "16": 416,
            },
            "validation_pairs_by_lead": {
                "1": 127,
                "2": 126,
                "4": 124,
                "8": 120,
                "16": 112,
            },
        },
        "architecture": architecture,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimization": {
            "screen_seed": 1701,
            "initialize_from_parent_model_only": True,
            "restore_parent_optimizer": False,
            "epochs": 4,
            "sample_batch_size": 1,
            "gradient_accumulation_samples": 4,
            "expected_optimizer_updates": 2132,
            "optimizer": "AdamW",
            "peak_learning_rate": 0.00005,
            "minimum_learning_rate": 0.000005,
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
            "scale_fit_leads": list(LEADS),
            "checkpoint_selection": "unweighted_mean_q_l",
            "q_l": "shared_model_derivative_mse/shared_persistence_derivative_mse",
            "physics_derived_quantities_used": False,
        },
        "screen_gates": {
            "mechanical_and_equivariance_gates_required": True,
            "every_c5p_field_positive_skill_at_every_lead": True,
            "maximum_lead1_shared_mse": PARENT_ONE_STEP_SHARED_MSE * 1.05,
            "minimum_parent_improvement_fraction": 0.10,
            "minimum_improved_longer_lead_count": 3,
        },
        "success_rule": (
            "A dated amendment may authorize seeds 1702 and 1703 plus bounded "
            "four/eight-frame autoregressive evaluation."
        ),
        "failure_rule": (
            "Do not tune this schedule against the result; proceed to a "
            "separately frozen operator-architecture experiment."
        ),
        "wandb_required": True,
    }
    verify_prerequisites(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-reduction", type=Path, required=True)
    parser.add_argument("--stage1-reduction-sha256", required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--parent-result-sha256", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    manifest = freeze_manifest(
        stage1_reduction=args.stage1_reduction,
        stage1_reduction_sha256=args.stage1_reduction_sha256,
        parent_result=args.parent_result,
        parent_result_sha256=args.parent_result_sha256,
        parent_checkpoint=args.parent_checkpoint,
        parent_checkpoint_sha256=args.parent_checkpoint_sha256,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
