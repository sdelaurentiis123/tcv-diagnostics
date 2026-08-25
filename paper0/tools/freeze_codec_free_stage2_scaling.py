#!/usr/bin/env python3
"""Freeze seed-1702/1703 confirmation of old-85604 multi-lead training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from paper0.tools.freeze_codec_free_stage2_multilead import locked
from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from paper0.tools.train_codec_free_stage2_multilead import (
    LEADS,
    SCALING_SCOPE,
    SCREEN_SCOPE,
    build_model,
    validate_parent_config,
    verify_prerequisites,
)
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


SCREEN_RESULT_SHA256 = (
    "6f4036ff6fd50a7090e60a351242f1a6ad00af6d3762322fe1075d22a9808c2a"
)
PARENT_METRICS = {
    1702: 0.005284142434913455,
    1703: 0.005365789313055087,
}
PARENT_CHECKPOINT_SHA256 = {
    1702: "59e9927ca88878a9d31a72789c6bbaf03248c507bc87f18ce2ac77e2026ea4a6",
    1703: "26e369f2114e56997a11a57e8233109aa501d82bf35f4f3ac632435ce2889b18",
}


def _architecture() -> dict[str, Any]:
    return {
        "base_channels": 24,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "lead_embedding_channels": 128,
        "group_norm_maximum_groups": 8,
        "kernel_size": 3,
        "zero_initialize_output": True,
    }


def _lock_parent(
    *,
    seed: int,
    result_path: Path,
    result_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    architecture: dict[str, Any],
) -> dict[str, Any]:
    result_lock = locked(result_path, result_sha256)
    result = load_strict_json(result_path)
    if result.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("parent result scope differs")
    if result.get("development_run") != "85604":
        raise ValueError("parent result development run differs")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("parent result held-out flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise ValueError("parent result physics-loss flag differs")
    if result.get("family") != "c5p" or int(result.get("seed", -1)) != seed:
        raise ValueError("parent result identity differs")
    if result.get("status") != "passed":
        raise ValueError("parent result did not pass")
    if result.get("training_gate", {}).get("passed") is not True:
        raise ValueError("parent training gate did not pass")
    selected = result.get("best_checkpoint", {})
    if float(selected.get("selection_metric", float("nan"))) != PARENT_METRICS[seed]:
        raise ValueError("parent one-step metric differs")
    if Path(str(selected.get("path", ""))) != checkpoint_path:
        raise ValueError("parent result selects another checkpoint")
    if selected.get("sha256") != checkpoint_sha256:
        raise ValueError("parent result checkpoint hash differs")
    checkpoint_lock = locked(checkpoint_path, checkpoint_sha256)
    if checkpoint_sha256 != PARENT_CHECKPOINT_SHA256[seed]:
        raise ValueError("parent checkpoint identity leaves frozen scaling set")

    model, config = build_model(architecture)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("family") != "c5p" or int(payload.get("seed", -1)) != seed:
        raise ValueError("parent checkpoint identity differs")
    config_validation = validate_parent_config(
        payload.get("config", {}), config.to_record()
    )
    if float(payload.get("selection_metric", float("nan"))) != PARENT_METRICS[seed]:
        raise ValueError("parent checkpoint metric differs")
    model.load_state_dict(payload["model"], strict=True)
    return {
        "family": "c5p",
        "seed": seed,
        "result": result_lock,
        "checkpoint": checkpoint_lock,
        "checkpoint_config_validation": config_validation,
        "one_step_selection_metric": PARENT_METRICS[seed],
        "load_state_dictionary_strictly": True,
        "require_bitwise_weight_equality": True,
    }


def freeze_scaling_manifest(
    *,
    stage1_reduction: Path,
    stage1_reduction_sha256: str,
    seed1701_screen_result: Path,
    seed1701_screen_result_sha256: str,
    seed1702_result: Path,
    seed1702_result_sha256: str,
    seed1702_checkpoint: Path,
    seed1702_checkpoint_sha256: str,
    seed1703_result: Path,
    seed1703_result_sha256: str,
    seed1703_checkpoint: Path,
    seed1703_checkpoint_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    """Create the prospective two-task Stage-2 scaling manifest."""

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

    screen_lock = locked(
        seed1701_screen_result, seed1701_screen_result_sha256
    )
    if seed1701_screen_result_sha256 != SCREEN_RESULT_SHA256:
        raise ValueError("seed-1701 result leaves frozen scaling evidence")
    screen = load_strict_json(seed1701_screen_result)
    if screen.get("scope") != SCREEN_SCOPE or int(screen.get("seed", -1)) != 1701:
        raise ValueError("seed-1701 screen identity differs")
    if screen.get("development_run") != "85604":
        raise ValueError("seed-1701 screen development run differs")
    if screen.get("held_out_85606_read") is not False:
        raise ValueError("seed-1701 screen held-out flag differs")
    if screen.get("advance_to_three_seed_scaling") is not True:
        raise ValueError("seed-1701 screen did not authorize scaling")

    architecture = _architecture()
    parents = {
        "1702": _lock_parent(
            seed=1702,
            result_path=seed1702_result,
            result_sha256=seed1702_result_sha256,
            checkpoint_path=seed1702_checkpoint,
            checkpoint_sha256=seed1702_checkpoint_sha256,
            architecture=architecture,
        ),
        "1703": _lock_parent(
            seed=1703,
            result_path=seed1703_result,
            result_sha256=seed1703_result_sha256,
            checkpoint_path=seed1703_checkpoint,
            checkpoint_sha256=seed1703_checkpoint_sha256,
            architecture=architecture,
        ),
    }
    model, _ = build_model(architecture)
    manifest = {
        "schema_version": 1,
        "scope": SCALING_SCOPE,
        "protocol": (
            "paper0/protocol/"
            "POST_ECRD_OLD_85604_STAGE2_SCALING_ROLLOUT_AMENDMENT_2026-08-25.md"
        ),
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "new_nersc_data_access_allowed": False,
        "screen_training_authorized": False,
        "three_seed_scaling_authorized": True,
        "seed_confirmation_training_authorized": True,
        "paper0_commit_at_freeze": paper0_commit,
        "state_family": "c5p",
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "prerequisites": {
            "stage1_reduction": reduction_lock,
            "seed1701_screen_result": screen_lock,
        },
        "parents": parents,
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
            "authorized_seeds": [1702, 1703],
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
            "maximum_lead1_shared_mse_by_seed": {
                str(seed): metric * 1.05 for seed, metric in PARENT_METRICS.items()
            },
            "minimum_parent_improvement_fraction": 0.10,
            "minimum_improved_longer_lead_count": 3,
        },
        "all_seed_confirmation_required": True,
        "conditional_bounded_rollout_authorized": True,
        "wandb_required": True,
    }
    for seed in (1702, 1703):
        verify_prerequisites(manifest, seed=seed)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-reduction", type=Path, required=True)
    parser.add_argument("--stage1-reduction-sha256", required=True)
    parser.add_argument("--seed1701-screen-result", type=Path, required=True)
    parser.add_argument("--seed1701-screen-result-sha256", required=True)
    for seed in (1702, 1703):
        parser.add_argument(f"--seed{seed}-result", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-result-sha256", required=True)
        parser.add_argument(f"--seed{seed}-checkpoint", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-checkpoint-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    manifest = freeze_scaling_manifest(
        stage1_reduction=args.stage1_reduction,
        stage1_reduction_sha256=args.stage1_reduction_sha256,
        seed1701_screen_result=args.seed1701_screen_result,
        seed1701_screen_result_sha256=args.seed1701_screen_result_sha256,
        seed1702_result=args.seed1702_result,
        seed1702_result_sha256=args.seed1702_result_sha256,
        seed1702_checkpoint=args.seed1702_checkpoint,
        seed1702_checkpoint_sha256=args.seed1702_checkpoint_sha256,
        seed1703_result=args.seed1703_result,
        seed1703_result_sha256=args.seed1703_result_sha256,
        seed1703_checkpoint=args.seed1703_checkpoint,
        seed1703_checkpoint_sha256=args.seed1703_checkpoint_sha256,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
