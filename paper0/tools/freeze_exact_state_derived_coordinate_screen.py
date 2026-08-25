#!/usr/bin/env python3
"""Freeze the old-85604 exact-state phi-plus-Vi coordinate screen."""

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


BASELINE_SHARED_MSE = 0.007772147896373167
LOCAL_PHI_SHARED_MSE = 0.007762225671440001
AXIAL_PHI_SHARED_MSE = 0.009164299673672594


def locked(path: Path, digest: str) -> dict[str, str]:
    assert_development_path(path)
    if sha256_path(path) != digest:
        raise ValueError(f"input SHA-256 differs for {path}")
    return {"path": str(path), "sha256": digest}


def _lock_result(
    path: Path,
    digest: str,
    *,
    architecture: str,
    expected_metric: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record_lock = locked(path, digest)
    result = load_strict_json(path)
    if result.get("architecture_kind") != architecture:
        raise ValueError(f"{architecture} result identity differs")
    metric = float(result["best_checkpoint"]["selection_metric"])
    if metric != expected_metric:
        raise ValueError(f"{architecture} selection metric differs")
    return {**record_lock, "selection_metric": metric}, result


def freeze_manifest(
    *,
    stage1_reduction: Path,
    stage1_reduction_sha256: str,
    baseline_e6b: Path,
    baseline_e6b_sha256: str,
    local_phi: Path,
    local_phi_sha256: str,
    axial_phi: Path,
    axial_phi_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    reduction_lock = locked(stage1_reduction, stage1_reduction_sha256)
    baseline_lock = locked(baseline_e6b, baseline_e6b_sha256)
    baseline = load_strict_json(baseline_e6b)
    if baseline.get("family") != "e6b" or int(baseline.get("seed", -1)) != 1701:
        raise ValueError("baseline identity differs")
    baseline_metric = float(baseline["best_checkpoint"]["selection_metric"])
    if baseline_metric != BASELINE_SHARED_MSE:
        raise ValueError("baseline selection metric differs")

    local_lock, local_result = _lock_result(
        local_phi,
        local_phi_sha256,
        architecture="local_current_phi",
        expected_metric=LOCAL_PHI_SHARED_MSE,
    )
    axial_lock, _ = _lock_result(
        axial_phi,
        axial_phi_sha256,
        architecture="axial_current_phi",
        expected_metric=AXIAL_PHI_SHARED_MSE,
    )

    architecture = {
        "base_channels": 24,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "lead_embedding_channels": 128,
        "group_norm_maximum_groups": 8,
        "kernel_size": 3,
        "zero_initialize_output": True,
    }
    model, _ = build_model("local_current_phi_vi", architecture)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    local_parameter_count = int(local_result["architecture"]["parameter_count"])
    relative_gap = abs(parameter_count - local_parameter_count) / local_parameter_count
    if relative_gap >= 0.03:
        raise ValueError("derived-coordinate parameter budget differs")

    manifest = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_exact_state_derived_coordinate_screen",
        "protocol": (
            "paper0/protocol/"
            "POST_ECRD_OLD_85604_DERIVED_COORDINATE_SCREEN_2026-08-25.md"
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
            "baseline_e6b_seed1701": {
                **baseline_lock,
                "selection_metric": baseline_metric,
            },
            "local_current_phi_seed1701": local_lock,
            "axial_current_phi_seed1701": axial_lock,
        },
        "state": {
            "predicted_volume_fields": list(FAMILY_FIELDS["e6b"]),
            "predicted_boundary": "Bphi",
            "auxiliary_context_fields": ["phi", "Vi"],
            "future_auxiliary_context_allowed": False,
            "current_phi_rollout_ready_without_elliptic_closure": False,
            "validated_elliptic_closure_available": True,
            "rollout_requires_external_elliptic_operator": True,
            "current_vi_rollout_ready_from_predicted_e6b": True,
            "vi_reconstruction": {
                "formula": "NVi / (2 * softFloor(Ne, 1e-7))",
                "source_fields": ["NVi", "Ne"],
                "future_truth_required": False,
            },
            "elliptic_closure_reference": (
                "paper0/PHASE2_POTENTIAL_VORTICITY_ALL_FRAME_READOUT.md"
            ),
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
        "architectures": {"local_current_phi_vi": architecture},
        "parameter_counts": {
            "completed_local_current_phi": local_parameter_count,
            "local_current_phi_vi": parameter_count,
        },
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
            "maximum_shared_mse": BASELINE_SHARED_MSE * 0.85,
            "every_e6b_field_positive_persistence_skill": True,
            "mechanical_and_equivariance_gates_required": True,
        },
        "failure_rule": (
            "Do not train additional derived-coordinate variants under this screen."
        ),
        "wandb_required": True,
    }
    verify_prerequisites(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-reduction", type=Path, required=True)
    parser.add_argument("--stage1-reduction-sha256", required=True)
    parser.add_argument("--baseline-e6b", type=Path, required=True)
    parser.add_argument("--baseline-e6b-sha256", required=True)
    parser.add_argument("--local-phi", type=Path, required=True)
    parser.add_argument("--local-phi-sha256", required=True)
    parser.add_argument("--axial-phi", type=Path, required=True)
    parser.add_argument("--axial-phi-sha256", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    manifest = freeze_manifest(
        stage1_reduction=args.stage1_reduction,
        stage1_reduction_sha256=args.stage1_reduction_sha256,
        baseline_e6b=args.baseline_e6b,
        baseline_e6b_sha256=args.baseline_e6b_sha256,
        local_phi=args.local_phi,
        local_phi_sha256=args.local_phi_sha256,
        axial_phi=args.axial_phi,
        axial_phi_sha256=args.axial_phi_sha256,
        paper0_commit=args.paper0_commit,
    )
    atomic_json(args.output, manifest)
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
