#!/usr/bin/env python3
"""Fine-tune one frozen old-85604 C5P parent on multiple lead times."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    evaluate,
    learning_rate,
    reload_and_equivariance_gate,
    repository_commit,
    tensor_batch,
    verify_finished_wandb_run,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import FAMILY_FIELDS, load_official_catalog
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
    persistence_normalized_state_derivative_loss,
)
from tcv_diagnostics.state_operator_data import (
    LeadTimeStateDataset,
    StateDerivativeRMS,
    fit_training_derivative_rms,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


LEADS = (1, 2, 4, 8, 16)
LONGER_LEADS = (2, 4, 8, 16)
FAMILY = "c5p"
FIELDS = FAMILY_FIELDS[FAMILY]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def authorize_manifest(manifest: Mapping[str, Any], *, seed: int) -> None:
    if manifest.get("scope") != "post_ecrd_old_85604_stage2_multilead_screen":
        raise ValueError("Stage-2 scope differs")
    if manifest.get("development_run") != "85604":
        raise ValueError("Stage-2 development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("Stage-2 held-out flag differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise ValueError("Stage-2 held-out access must be prohibited")
    if manifest.get("guard_frames_read_allowed") is not False:
        raise ValueError("Stage-2 guard reads must be prohibited")
    if manifest.get("screen_training_authorized") is not True:
        raise ValueError("Stage-2 screen training is not authorized")
    if manifest.get("three_seed_scaling_authorized") is not False:
        raise ValueError("Stage-2 three-seed scaling must remain unauthorized")
    if manifest.get("state_family") != FAMILY:
        raise ValueError("Stage-2 state family differs")
    split = manifest.get("split", {})
    expected_split = {
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
    }
    if any(split.get(key) != value for key, value in expected_split.items()):
        raise ValueError("Stage-2 split differs")
    optimization = manifest.get("optimization", {})
    if seed != int(optimization.get("screen_seed", -1)):
        raise ValueError("seed leaves frozen Stage-2 screen")
    if optimization.get("initialize_from_parent_model_only") is not True:
        raise ValueError("Stage-2 parent initialization rule differs")
    if optimization.get("restore_parent_optimizer") is not False:
        raise ValueError("Stage-2 parent optimizer must not be restored")


def load_locked_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{name} SHA-256 differs")
    return load_strict_json(path)


def verify_prerequisites(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = manifest.get("prerequisites", {})
    reduction_lock = records.get("stage1_reduction", {})
    reduction = load_locked_json(reduction_lock, name="Stage-1 reduction")
    if reduction.get("development_run") != "85604":
        raise ValueError("Stage-1 reduction development run differs")
    if reduction.get("held_out_85606_read") is not False:
        raise ValueError("Stage-1 reduction held-out flag differs")
    if reduction.get("decision") != (
        "retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation"
    ):
        raise ValueError("Stage-1 reduction does not authorize C5P Stage 2")

    parent_result_lock = records.get("parent_result", {})
    parent_result = load_locked_json(parent_result_lock, name="parent result")
    if parent_result.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("parent result scope differs")
    if parent_result.get("development_run") != "85604":
        raise ValueError("parent result development run differs")
    if parent_result.get("held_out_85606_read") is not False:
        raise ValueError("parent result held-out flag differs")
    if parent_result.get("physics_derived_loss_used") is not False:
        raise ValueError("parent result physics-loss flag differs")
    if parent_result.get("family") != FAMILY or int(parent_result.get("seed", -1)) != 1701:
        raise ValueError("parent result identity differs")
    if parent_result.get("status") != "passed":
        raise ValueError("parent result did not pass")
    if parent_result.get("training_gate", {}).get("passed") is not True:
        raise ValueError("parent training gate did not pass")

    parent = manifest.get("parent", {})
    result_metric = float(parent_result["best_checkpoint"]["selection_metric"])
    if result_metric != float(parent.get("one_step_selection_metric", float("nan"))):
        raise ValueError("parent one-step metric differs")
    checkpoint_lock = parent.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint_lock.get("path", "")))
    assert_development_path(checkpoint_path)
    if checkpoint_path != Path(parent_result["best_checkpoint"]["path"]):
        raise ValueError("parent checkpoint path differs from result")
    checkpoint_sha = str(checkpoint_lock.get("sha256", ""))
    if checkpoint_sha != parent_result["best_checkpoint"]["sha256"]:
        raise ValueError("parent checkpoint hash differs from result")
    if not checkpoint_sha or sha256_path(checkpoint_path) != checkpoint_sha:
        raise ValueError("parent checkpoint SHA-256 differs")
    return {
        "stage1_reduction": dict(reduction_lock),
        "parent_result": dict(parent_result_lock),
        "parent_checkpoint": dict(checkpoint_lock),
        "parent_one_step_selection_metric": result_metric,
    }


def build_model(architecture: Mapping[str, Any]) -> tuple[
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
]:
    config = CodecFreeOperatorConfig(
        state_family=FAMILY,
        history_frames=1,
        base_channels=int(architecture["base_channels"]),
        channel_multipliers=tuple(architecture["channel_multipliers"]),
        blocks_per_level=int(architecture["blocks_per_level"]),
        lead_embedding_channels=int(architecture["lead_embedding_channels"]),
        group_norm_maximum_groups=int(architecture["group_norm_maximum_groups"]),
        kernel_size=int(architecture["kernel_size"]),
        predict_boundary=False,
        zero_initialize_output=bool(architecture["zero_initialize_output"]),
    )
    return CodecFreeIncrementOperator3D(config), config


def validate_parent_config(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one legacy checkpoint without weakening architecture identity.

    The optional auxiliary-context channel count was added to config
    serialization after the frozen Stage-1 checkpoint was written.  Its
    historical absence has exactly one valid interpretation: the default of
    zero channels.  No other missing, extra, or changed field is accepted.
    """

    canonical = dict(stored)
    inserted: list[str] = []
    if "auxiliary_context_channels" not in canonical:
        canonical["auxiliary_context_channels"] = 0
        inserted.append("auxiliary_context_channels")
    if canonical != dict(expected):
        raise ValueError("parent checkpoint architecture differs")
    return {
        "stored_config_matches_after_explicit_legacy_default": True,
        "inserted_legacy_defaults": inserted,
        "legacy_default_values": {"auxiliary_context_channels": 0},
    }


def load_parent_model(
    *,
    manifest: Mapping[str, Any],
    device: torch.device,
) -> tuple[CodecFreeIncrementOperator3D, CodecFreeOperatorConfig, dict[str, Any]]:
    model, config = build_model(manifest["architecture"])
    checkpoint_path = Path(manifest["parent"]["checkpoint"]["path"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("family") != FAMILY or int(payload.get("seed", -1)) != 1701:
        raise ValueError("parent checkpoint identity differs")
    config_validation = validate_parent_config(
        payload.get("config", {}), config.to_record()
    )
    if float(payload.get("selection_metric", float("nan"))) != float(
        manifest["parent"]["one_step_selection_metric"]
    ):
        raise ValueError("parent checkpoint selection metric differs")
    model = model.to(device)
    model.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(payload["model"][name].to(device), model.state_dict()[name])
        for name in payload["model"]
    )
    if not bitwise:
        raise AssertionError("parent state dictionary did not load bitwise")
    return model, config, {
        "checkpoint_reload_bitwise": True,
        "checkpoint": dict(manifest["parent"]["checkpoint"]),
        "checkpoint_paper0_commit": payload.get("paper0_commit"),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_optimizer_updates": int(payload["optimizer_updates"]),
        "checkpoint_config_validation": config_validation,
    }


def evaluate_by_lead(
    model: CodecFreeIncrementOperator3D,
    datasets: Mapping[int, LeadTimeStateDataset],
    *,
    device: torch.device,
) -> dict[str, Any]:
    per_lead = {
        str(lead): evaluate(model, datasets[lead], family=FAMILY, device=device)
        for lead in LEADS
    }
    ratios = {
        lead: (
            float(per_lead[str(lead)]["shared_field_mean_model_derivative_mse"])
            / float(
                per_lead[str(lead)][
                    "shared_field_mean_zero_derivative_persistence_mse"
                ]
            )
        )
        for lead in LEADS
    }
    return {
        "per_lead": per_lead,
        "shared_persistence_normalized_mse_ratio_by_lead": {
            str(lead): ratios[lead] for lead in LEADS
        },
        "mean_shared_persistence_normalized_mse_ratio": float(
            np.mean(list(ratios.values()))
        ),
    }


def screen_decision(
    *,
    parent_evaluation: Mapping[str, Any],
    best_validation: Mapping[str, Any],
    training_gate: bool,
    frozen_gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the prospectively frozen scientific advancement gates."""

    parent_metric = float(
        parent_evaluation["mean_shared_persistence_normalized_mse_ratio"]
    )
    best_metric = float(
        best_validation["mean_shared_persistence_normalized_mse_ratio"]
    )
    improvement_fraction = 1.0 - best_metric / parent_metric
    every_field_positive = all(
        float(
            best_validation["per_lead"][str(lead)]["per_field"][field][
                "persistence_relative_skill"
            ]
        )
        > 0.0
        for lead in LEADS
        for field in FIELDS
    )
    lead1_mse = float(
        best_validation["per_lead"]["1"][
            "shared_field_mean_model_derivative_mse"
        ]
    )
    longer_improvements = {
        str(lead): (
            float(
                parent_evaluation["per_lead"][str(lead)][
                    "shared_field_mean_model_derivative_mse"
                ]
            )
            - float(
                best_validation["per_lead"][str(lead)][
                    "shared_field_mean_model_derivative_mse"
                ]
            )
        )
        for lead in LONGER_LEADS
    }
    longer_improved_count = sum(
        improvement > 0.0 for improvement in longer_improvements.values()
    )
    gates = {
        "training_gate_passed": training_gate,
        "every_c5p_field_positive_skill_at_every_lead": every_field_positive,
        "lead1_shared_mse_at_most_five_percent_above_parent": (
            lead1_mse <= float(frozen_gates["maximum_lead1_shared_mse"])
        ),
        "mean_multilead_ratio_improves_at_least_ten_percent": (
            improvement_fraction
            >= float(frozen_gates["minimum_parent_improvement_fraction"])
        ),
        "at_least_three_longer_leads_improve": (
            longer_improved_count
            >= int(frozen_gates["minimum_improved_longer_lead_count"])
        ),
    }
    return {
        "parent_selection_metric": parent_metric,
        "best_selection_metric": best_metric,
        "parent_improvement_fraction": improvement_fraction,
        "lead1_shared_mse": lead1_mse,
        "longer_lead_shared_mse_improvements": longer_improvements,
        "longer_lead_improved_count": longer_improved_count,
        "screen_gates": gates,
        "advance_to_three_seed_scaling": all(gates.values()),
    }


def save_checkpoint(
    path: Path,
    *,
    model: CodecFreeIncrementOperator3D,
    optimizer: torch.optim.Optimizer,
    config: CodecFreeOperatorConfig,
    derivative_rms: StateDerivativeRMS,
    seed: int,
    epoch: int,
    optimizer_updates: int,
    selection_metric: float,
    parent_checkpoint: Mapping[str, Any],
    paper0_commit: str,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_record(),
            "derivative_rms": derivative_rms.to_record(),
            "family": FAMILY,
            "stage": "stage2_multilead_finetune",
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "selection_metric": selection_metric,
            "parent_checkpoint": dict(parent_checkpoint),
            "paper0_commit": paper0_commit,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("Stage-2 manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(manifest, seed=args.seed)
    prerequisites = verify_prerequisites(manifest)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from launch lock")
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 training requires an allocated CUDA GPU")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)

    catalog = load_official_catalog(args.artifact_root)
    scale_dataset = LeadTimeStateDataset(
        catalog,
        family=FAMILY,
        split="train",
        lead_steps=LEADS,
        history_frames=1,
        augment=False,
        seed=args.seed,
    )
    try:
        derivative_rms = fit_training_derivative_rms(scale_dataset)
    finally:
        scale_dataset.close()
    atomic_json(args.output / "derivative_rms.json", derivative_rms.to_record())

    train = LeadTimeStateDataset(
        catalog,
        family=FAMILY,
        split="train",
        lead_steps=LEADS,
        history_frames=1,
        augment=True,
        seed=args.seed,
    )
    validations = {
        lead: LeadTimeStateDataset(
            catalog,
            family=FAMILY,
            split="validation",
            lead_steps=(lead,),
            history_frames=1,
            augment=False,
            seed=args.seed,
        )
        for lead in LEADS
    }
    split = manifest["split"]
    if len(train) != int(split["training_pair_count"]):
        raise ValueError("Stage-2 training pair count differs")
    for lead, dataset in validations.items():
        if len(dataset) != int(split["validation_pairs_by_lead"][str(lead)]):
            raise ValueError(f"Stage-2 validation pair count differs at lead {lead}")

    model, config, parent_load = load_parent_model(manifest=manifest, device=device)
    parent_evaluation = evaluate_by_lead(model, validations, device=device)
    parent_record = {
        **parent_load,
        "evaluation_before_optimizer_construction": True,
        "evaluation": parent_evaluation,
    }
    atomic_json(args.output / "parent_multilead_evaluation.json", parent_record)

    optimization = manifest["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["peak_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    volume_rms = torch.tensor(derivative_rms.volume, device=device)
    epochs = int(optimization["epochs"])
    accumulation = int(optimization["gradient_accumulation_samples"])
    updates_per_epoch = math.ceil(len(train) / accumulation)
    total_updates = updates_per_epoch * epochs
    if total_updates != int(optimization["expected_optimizer_updates"]):
        raise ValueError("Stage-2 expected optimizer-update count differs")
    warmup_updates = max(
        1,
        math.ceil(total_updates * float(optimization["warmup_fraction"])),
    )

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("online W&B is required") from error
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="old-85604-stage2-multilead-screen",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "codec-free",
            "c5p",
            "multilead",
            "finetune",
            "screen",
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key:
        raise RuntimeError("W&B API key is absent")
    viewer = api.viewer
    if str(getattr(viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B entity differs")
    tracking_directory = args.output / "wandb"
    tracking_directory.mkdir()
    run = wandb.init(
        entity=spec.entity,
        project=spec.project,
        group=spec.group,
        name=spec.run_name,
        id=spec.run_id,
        resume="never",
        job_type=spec.job_type,
        tags=list(spec.tags),
        config={
            **manifest["architecture"],
            **optimization,
            "scope": manifest["scope"],
            "state_family": FAMILY,
            "lead_steps": list(LEADS),
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "parent_checkpoint": manifest["parent"]["checkpoint"],
            "parent_multilead_selection_metric": parent_evaluation[
                "mean_shared_persistence_normalized_mse_ratio"
            ],
            "held_out_85606_read": False,
            "physics_derived_loss_used": False,
            "derivative_rms": derivative_rms.to_record(),
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    optimizer_updates = 0
    history: list[dict[str, Any]] = []
    best_metric = math.inf
    best_checkpoint: Path | None = None
    started = time.perf_counter()

    try:
        for epoch in range(epochs):
            train.set_epoch(epoch)
            order = np.random.default_rng(
                np.random.SeedSequence([args.seed, epoch, 0x4D554C54])
            ).permutation(len(train))
            raw_losses: list[float] = []
            gradient_norms: list[float] = []
            model.train()
            for group_start in range(0, len(order), accumulation):
                group = order[group_start : group_start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    values = tensor_batch(train[int(index)], device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = model(
                            values["context"],
                            values["lead_steps"],
                        )
                        loss, _ = persistence_normalized_state_derivative_loss(
                            prediction,
                            values["target_derivative"],
                            volume_rms,
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Stage-2 training loss is non-finite")
                    raw_losses.append(float(loss.detach().cpu()))
                    (loss / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(optimization["gradient_clip_norm"]),
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("Stage-2 gradient norm is non-finite")
                optimizer_updates += 1
                rate = learning_rate(
                    optimizer_updates,
                    total_updates=total_updates,
                    warmup_updates=warmup_updates,
                    peak=float(optimization["peak_learning_rate"]),
                    minimum=float(optimization["minimum_learning_rate"]),
                )
                for group_record in optimizer.param_groups:
                    group_record["lr"] = rate
                optimizer.step()
                gradient_norms.append(float(norm.detach().cpu()))
                if optimizer_updates % 40 == 0 or optimizer_updates == total_updates:
                    run.log(
                        {
                            "optimizer/update": optimizer_updates,
                            "optimizer/learning_rate": rate,
                            "train/recent_persistence_normalized_loss": float(
                                np.mean(raw_losses[-len(group) :])
                            ),
                            "train/preclip_gradient_norm": gradient_norms[-1],
                        },
                        step=optimizer_updates,
                    )

            validation = evaluate_by_lead(model, validations, device=device)
            selection_metric = float(
                validation["mean_shared_persistence_normalized_mse_ratio"]
            )
            checkpoint = args.output / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                derivative_rms=derivative_rms,
                seed=args.seed,
                epoch=epoch + 1,
                optimizer_updates=optimizer_updates,
                selection_metric=selection_metric,
                parent_checkpoint=manifest["parent"]["checkpoint"],
                paper0_commit=args.paper0_commit,
            )
            if selection_metric < best_metric:
                best_metric = selection_metric
                best_checkpoint = checkpoint
            epoch_record = {
                "epoch": epoch + 1,
                "training_sample_count": len(raw_losses),
                "training_persistence_normalized_loss_mean": float(
                    np.mean(raw_losses)
                ),
                "training_persistence_normalized_loss_first_quarter": float(
                    np.mean(raw_losses[: max(1, len(raw_losses) // 4)])
                ),
                "training_persistence_normalized_loss_last_quarter": float(
                    np.mean(raw_losses[-max(1, len(raw_losses) // 4) :])
                ),
                "preclip_gradient_norm_maximum": float(np.max(gradient_norms)),
                "optimizer_updates_cumulative": optimizer_updates,
                "validation": validation,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha256_path(checkpoint),
                },
            }
            history.append(epoch_record)
            metrics = {
                "epoch": epoch + 1,
                "train/epoch_persistence_normalized_loss": epoch_record[
                    "training_persistence_normalized_loss_mean"
                ],
                "validation/mean_shared_persistence_normalized_mse_ratio": (
                    selection_metric
                ),
            }
            for lead in LEADS:
                lead_record = validation["per_lead"][str(lead)]
                metrics[
                    f"validation/lead_{lead}_shared_derivative_mse"
                ] = lead_record["shared_field_mean_model_derivative_mse"]
                metrics[
                    f"validation/lead_{lead}_shared_persistence_skill"
                ] = lead_record["shared_field_persistence_relative_skill"]
            run.log(metrics, step=optimizer_updates)

        if best_checkpoint is None:
            raise AssertionError("no Stage-2 checkpoint was selected")
        numerical_gate = reload_and_equivariance_gate(
            checkpoint=best_checkpoint,
            config=config,
            validation=validations[1],
            device=device,
        )
        exact_updates = optimizer_updates == total_updates
        loss_decreased = (
            history[-1]["training_persistence_normalized_loss_mean"]
            < history[0]["training_persistence_normalized_loss_mean"]
        )
        finite_metrics = all(
            math.isfinite(
                float(
                    record["validation"][
                        "mean_shared_persistence_normalized_mse_ratio"
                    ]
                )
            )
            for record in history
        )
        training_gate = bool(
            exact_updates
            and loss_decreased
            and finite_metrics
            and numerical_gate["passed"]
        )
        best_record = min(
            history,
            key=lambda record: record["validation"][
                "mean_shared_persistence_normalized_mse_ratio"
            ],
        )
        best_validation = best_record["validation"]
        decision = screen_decision(
            parent_evaluation=parent_evaluation,
            best_validation=best_validation,
            training_gate=training_gate,
            frozen_gates=manifest["screen_gates"],
        )
        result = {
            "schema_version": 1,
            "scope": manifest["scope"],
            "status": "passed" if training_gate else "failed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "physics_derived_loss_used": False,
            "family": FAMILY,
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "prerequisites": prerequisites,
            "architecture": model.to_record(),
            "lead_steps": list(LEADS),
            "loss": manifest["loss"],
            "derivative_rms": derivative_rms.to_record(),
            "parent": parent_record,
            "training_pair_count": len(train),
            "validation_pair_count": sum(len(item) for item in validations.values()),
            "epochs": epochs,
            "optimizer_updates": optimizer_updates,
            "expected_optimizer_updates": total_updates,
            "history": history,
            "best_checkpoint": {
                "path": str(best_checkpoint),
                "sha256": sha256_path(best_checkpoint),
                "epoch": int(best_record["epoch"]),
                "selection_metric": best_metric,
            },
            "parent_selection_metric": decision["parent_selection_metric"],
            "parent_improvement_fraction": decision[
                "parent_improvement_fraction"
            ],
            "best_validation": best_validation,
            "lead1_shared_mse": decision["lead1_shared_mse"],
            "longer_lead_shared_mse_improvements": decision[
                "longer_lead_shared_mse_improvements"
            ],
            "longer_lead_improved_count": decision[
                "longer_lead_improved_count"
            ],
            "numerical_gate": numerical_gate,
            "training_gate": {
                "exact_optimizer_update_count": exact_updates,
                "epoch_mean_training_loss_decreased": loss_decreased,
                "finite_validation_metrics": finite_metrics,
                "numerical_gate_passed": numerical_gate["passed"],
                "passed": training_gate,
            },
            "screen_gates": decision["screen_gates"],
            "advance_to_three_seed_scaling": decision[
                "advance_to_three_seed_scaling"
            ],
            "peak_cuda_memory_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "wall_seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(device),
            "numeric_precision": {
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
        }
        run.summary.update(
            {
                "final/status": result["status"],
                "final/best_mean_multilead_ratio": best_metric,
                "final/parent_improvement_fraction": decision[
                    "parent_improvement_fraction"
                ],
                "final/lead1_shared_mse": decision["lead1_shared_mse"],
                "final/training_gate_passed": training_gate,
                "final/advance_to_three_seed_scaling": decision[
                    "advance_to_three_seed_scaling"
                ],
                "compute/peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
                "compute/wall_seconds": result["wall_seconds"],
                "scope/held_out_85606_read": False,
                "scope/physics_derived_loss_used": False,
            }
        )
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        train.close()
        for dataset in validations.values():
            dataset.close()

    atomic_json(args.output / "result.json", result)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb,
        remote_path=remote_path,
        expected_id=spec.run_id,
    )
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": spec.to_record(),
        "authenticated_username": str(getattr(viewer, "username", "")),
        "wandb_version": wandb.__version__,
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_state_after_finish": remote_state,
        "checkpoints_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    atomic_json(args.output / "wandb.json", tracking)
    index = args.output / "artifact_sha256.txt"
    index.write_text(
        "".join(
            f"{sha256_path(path)}  {path}\n"
            for path in sorted(args.output.iterdir())
            if path.is_file() and path != index
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
