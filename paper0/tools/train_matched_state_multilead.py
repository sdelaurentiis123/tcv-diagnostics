#!/usr/bin/env python3
"""Run one arm of the frozen old-85604 matched state-view multi-lead pilot."""

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
from paper0.tools.train_codec_free_stage2_multilead import validate_parent_config
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


SCOPE = "post_ecrd_old_85604_matched_state_multilead_pilot"
LEADS = (1, 2, 4, 8, 16)
FAMILIES = ("c5p", "e6b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def authorize_manifest(
    manifest: Mapping[str, Any], *, family: str, seed: int
) -> None:
    """Refuse every state, data, or budget change outside the frozen pilot."""

    expected_flags = {
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "paired_pilot_training_authorized": True,
        "three_seed_scaling_authorized": False,
        "physics_derived_training_loss_allowed": False,
    }
    if manifest.get("scope") != SCOPE:
        raise ValueError("matched state-view scope differs")
    if any(manifest.get(key) != value for key, value in expected_flags.items()):
        raise ValueError("matched state-view scope flags differ")
    if family not in FAMILIES or family not in manifest.get("state_views", {}):
        raise ValueError("state family leaves the frozen pair")

    split = manifest.get("split", {})
    expected_split = {
        "training_frames": [0, 432],
        "guard_frames": [432, 496],
        "validation_frames": [496, 624],
        "history_frames": 1,
        "lead_steps": list(LEADS),
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
        raise ValueError("matched state-view split differs")

    optimization = manifest.get("optimization", {})
    if seed != int(optimization.get("pilot_seed", -1)):
        raise ValueError("seed leaves the frozen matched pilot")
    expected_optimization = {
        "epochs": 12,
        "sample_batch_size": 1,
        "gradient_accumulation_samples": 4,
        "expected_optimizer_updates": 6396,
        "optimizer": "AdamW",
        "peak_learning_rate": 0.00005,
        "minimum_learning_rate": 0.000005,
        "weight_decay": 0.0001,
        "warmup_fraction": 0.05,
        "gradient_clip_norm": 1.0,
        "autocast": "bfloat16",
        "cudnn_tf32": False,
        "matmul_tf32": False,
        "initialize_from_parent_model_only": True,
        "restore_parent_optimizer": False,
    }
    if any(
        optimization.get(key) != value
        for key, value in expected_optimization.items()
    ):
        raise ValueError("matched state-view optimization differs")


def locked_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{name} SHA-256 differs")
    return load_strict_json(path)


def parent_specification(
    manifest: Mapping[str, Any], *, family: str, seed: int
) -> Mapping[str, Any]:
    parent = manifest.get("parents", {}).get(family, {})
    if parent.get("family") != family or int(parent.get("seed", -1)) != seed:
        raise ValueError("matched parent identity differs")
    return parent


def verify_prerequisites(
    manifest: Mapping[str, Any], *, family: str, seed: int
) -> dict[str, Any]:
    records = manifest.get("prerequisites", {})
    dataset_lock = records.get("model_dataset_result", {})
    dataset = locked_json(dataset_lock, name="model dataset result")
    if dataset.get("development_run") != "85604":
        raise ValueError("model dataset development run differs")
    if dataset.get("held_out_85606_read") is not False:
        raise ValueError("model dataset held-out flag differs")
    if dataset.get("dataset", {}).get("frame_count") != 624:
        raise ValueError("model dataset frame count differs")
    if dataset.get("gates", {}).get("all_passed") is not True:
        raise ValueError("model dataset did not pass its gates")

    stochastic_lock = records.get("persistent_global_local_physics_result", {})
    stochastic = locked_json(
        stochastic_lock, name="persistent global-local physics result"
    )
    if stochastic.get("development_run") != "85604":
        raise ValueError("stochastic prerequisite development run differs")
    if stochastic.get("held_out_85606_read") is not False:
        raise ValueError("stochastic prerequisite held-out flag differs")
    if stochastic.get("new_nersc_data_read") is not False:
        raise ValueError("stochastic prerequisite newer-data flag differs")
    if stochastic.get("status") != "completed_failed":
        raise ValueError("stochastic prerequisite status differs")
    if stochastic.get("gate", {}).get("passed") is not False:
        raise ValueError("stochastic prerequisite gate differs")

    parent = parent_specification(manifest, family=family, seed=seed)
    result_lock = parent.get("result", {})
    result = locked_json(result_lock, name=f"{family} Stage-1 result")
    if result.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("Stage-1 parent scope differs")
    if result.get("development_run") != "85604":
        raise ValueError("Stage-1 parent development run differs")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("Stage-1 parent held-out flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise ValueError("Stage-1 parent used a physics-derived loss")
    if result.get("family") != family or int(result.get("seed", -1)) != seed:
        raise ValueError("Stage-1 parent family or seed differs")
    if result.get("status") != "passed":
        raise ValueError("Stage-1 parent did not pass")
    if result.get("training_gate", {}).get("passed") is not True:
        raise ValueError("Stage-1 parent training gate did not pass")

    selected = result.get("best_checkpoint", {})
    expected_metric = float(parent.get("one_step_selection_metric", math.nan))
    if float(selected.get("selection_metric", math.nan)) != expected_metric:
        raise ValueError("Stage-1 parent metric differs")
    checkpoint = parent.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    checkpoint_digest = str(checkpoint.get("sha256", ""))
    assert_development_path(checkpoint_path)
    if checkpoint_path != Path(str(selected.get("path", ""))):
        raise ValueError("Stage-1 result selects another checkpoint")
    if checkpoint_digest != selected.get("sha256"):
        raise ValueError("Stage-1 result checkpoint hash differs")
    if not checkpoint_digest or sha256_path(checkpoint_path) != checkpoint_digest:
        raise ValueError("Stage-1 checkpoint SHA-256 differs")
    return {
        "model_dataset_result": dict(dataset_lock),
        "persistent_global_local_physics_result": dict(stochastic_lock),
        "parent_result": dict(result_lock),
        "parent_checkpoint": dict(checkpoint),
        "parent_one_step_selection_metric": expected_metric,
    }


def build_model(
    architecture: Mapping[str, Any], *, family: str
) -> tuple[CodecFreeIncrementOperator3D, CodecFreeOperatorConfig]:
    config = CodecFreeOperatorConfig(
        state_family=family,
        history_frames=1,
        base_channels=int(architecture["base_channels"]),
        channel_multipliers=tuple(architecture["channel_multipliers"]),
        blocks_per_level=int(architecture["blocks_per_level"]),
        lead_embedding_channels=int(architecture["lead_embedding_channels"]),
        group_norm_maximum_groups=int(architecture["group_norm_maximum_groups"]),
        kernel_size=int(architecture["kernel_size"]),
        predict_boundary=family == "e6b",
        zero_initialize_output=bool(architecture["zero_initialize_output"]),
    )
    return CodecFreeIncrementOperator3D(config), config


def load_parent_model(
    *,
    manifest: Mapping[str, Any],
    family: str,
    seed: int,
    device: torch.device,
) -> tuple[CodecFreeIncrementOperator3D, CodecFreeOperatorConfig, dict[str, Any]]:
    model, config = build_model(manifest["architecture"], family=family)
    parent = parent_specification(manifest, family=family, seed=seed)
    checkpoint_path = Path(str(parent["checkpoint"]["path"]))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("family") != family or int(payload.get("seed", -1)) != seed:
        raise ValueError("Stage-1 checkpoint identity differs")
    config_validation = validate_parent_config(
        payload.get("config", {}), config.to_record()
    )
    if float(payload.get("selection_metric", math.nan)) != float(
        parent["one_step_selection_metric"]
    ):
        raise ValueError("Stage-1 checkpoint metric differs")
    model = model.to(device)
    model.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(payload["model"][name].to(device), model.state_dict()[name])
        for name in payload["model"]
    )
    if not bitwise:
        raise AssertionError("Stage-1 state dictionary did not load bitwise")
    return model, config, {
        "checkpoint_reload_bitwise": True,
        "checkpoint": dict(parent["checkpoint"]),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_optimizer_updates": int(payload["optimizer_updates"]),
        "checkpoint_paper0_commit": payload.get("paper0_commit"),
        "checkpoint_config_validation": config_validation,
    }


def _add_boundary_skill(record: dict[str, Any]) -> None:
    for side_record in record.get("boundary_by_side", {}).values():
        model_mse = float(side_record["model_derivative_mse"])
        persistence_mse = float(side_record["zero_derivative_persistence_mse"])
        side_record["persistence_relative_skill"] = (
            1.0 - model_mse / persistence_mse if persistence_mse > 0.0 else None
        )


def evaluate_by_lead(
    model: CodecFreeIncrementOperator3D,
    datasets: Mapping[int, LeadTimeStateDataset],
    *,
    family: str,
    device: torch.device,
) -> dict[str, Any]:
    per_lead: dict[str, Any] = {}
    for lead in LEADS:
        record = evaluate(model, datasets[lead], family=family, device=device)
        if family == "e6b":
            _add_boundary_skill(record)
        per_lead[str(lead)] = record
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


def save_checkpoint(
    path: Path,
    *,
    model: CodecFreeIncrementOperator3D,
    optimizer: torch.optim.Optimizer,
    config: CodecFreeOperatorConfig,
    derivative_rms: StateDerivativeRMS,
    family: str,
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
            "family": family,
            "stage": "matched_state_multilead_pilot",
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "selection_metric": selection_metric,
            "parent_checkpoint": dict(parent_checkpoint),
            "paper0_commit": paper0_commit,
        },
        path,
    )


def transition_gate(
    validation: Mapping[str, Any], *, family: str
) -> dict[str, Any]:
    fields = FAMILY_FIELDS[family]
    volume_positive = all(
        float(
            validation["per_lead"][str(lead)]["per_field"][field][
                "persistence_relative_skill"
            ]
        )
        > 0.0
        for lead in LEADS
        for field in fields
    )
    boundary_positive = True
    if family == "e6b":
        boundary_positive = all(
            float(
                validation["per_lead"][str(lead)]["boundary_by_side"][side][
                    "persistence_relative_skill"
                ]
            )
            > 0.0
            for lead in LEADS
            for side in ("inner", "outer")
        )
    return {
        "every_volume_field_positive_skill_at_every_lead": volume_positive,
        "every_boundary_side_positive_skill_at_every_lead": boundary_positive,
        "passed": volume_positive and boundary_positive,
    }


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("matched state-view manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(manifest, family=args.family, seed=args.seed)
    prerequisites = verify_prerequisites(
        manifest, family=args.family, seed=args.seed
    )
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from launch lock")
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("matched state-view training requires an allocated GPU")
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
    rms_dataset = LeadTimeStateDataset(
        catalog,
        family=args.family,
        split="train",
        lead_steps=LEADS,
        history_frames=1,
        augment=False,
        seed=args.seed,
    )
    try:
        derivative_rms = fit_training_derivative_rms(rms_dataset)
    finally:
        rms_dataset.close()
    atomic_json(args.output / "derivative_rms.json", derivative_rms.to_record())

    train = LeadTimeStateDataset(
        catalog,
        family=args.family,
        split="train",
        lead_steps=LEADS,
        history_frames=1,
        augment=True,
        seed=args.seed,
    )
    validations = {
        lead: LeadTimeStateDataset(
            catalog,
            family=args.family,
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
        raise ValueError("matched training pair count differs")
    for lead, dataset in validations.items():
        if len(dataset) != int(split["validation_pairs_by_lead"][str(lead)]):
            raise ValueError(f"validation pair count differs at lead {lead}")

    parent_spec = parent_specification(
        manifest, family=args.family, seed=args.seed
    )
    model, config, parent_load = load_parent_model(
        manifest=manifest,
        family=args.family,
        seed=args.seed,
        device=device,
    )
    parent_evaluation = evaluate_by_lead(
        model, validations, family=args.family, device=device
    )
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
    boundary_rms = (
        None
        if derivative_rms.boundary is None
        else torch.tensor(derivative_rms.boundary, device=device)
    )
    epochs = int(optimization["epochs"])
    accumulation = int(optimization["gradient_accumulation_samples"])
    updates_per_epoch = math.ceil(len(train) / accumulation)
    total_updates = updates_per_epoch * epochs
    if total_updates != int(optimization["expected_optimizer_updates"]):
        raise ValueError("expected optimizer-update count differs")
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
        job_type="old-85604-matched-state-multilead-pilot",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "codec-free",
            args.family,
            "matched-state-view",
            "multilead",
            "pilot",
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
            "state_family": args.family,
            "lead_steps": list(LEADS),
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "parent_checkpoint": parent_spec["checkpoint"],
            "parent_multilead_selection_metric": parent_evaluation[
                "mean_shared_persistence_normalized_mse_ratio"
            ],
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
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
    result: dict[str, Any]

    try:
        for epoch in range(epochs):
            train.set_epoch(epoch)
            order = np.random.default_rng(
                np.random.SeedSequence([args.seed, epoch, 0x53544154])
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
                            values.get("context_boundary"),
                        )
                        loss, _ = persistence_normalized_state_derivative_loss(
                            prediction,
                            values["target_derivative"],
                            volume_rms,
                            values.get("target_boundary_derivative"),
                            boundary_rms,
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("training loss is non-finite")
                    raw_losses.append(float(loss.detach().cpu()))
                    (loss / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(optimization["gradient_clip_norm"]),
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("gradient norm is non-finite")
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

            validation = evaluate_by_lead(
                model, validations, family=args.family, device=device
            )
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
                family=args.family,
                seed=args.seed,
                epoch=epoch + 1,
                optimizer_updates=optimizer_updates,
                selection_metric=selection_metric,
                parent_checkpoint=parent_spec["checkpoint"],
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
                metrics[f"validation/lead_{lead}_shared_derivative_mse"] = (
                    lead_record["shared_field_mean_model_derivative_mse"]
                )
                metrics[f"validation/lead_{lead}_shared_persistence_skill"] = (
                    lead_record["shared_field_persistence_relative_skill"]
                )
            run.log(metrics, step=optimizer_updates)

        if best_checkpoint is None:
            raise AssertionError("no matched state-view checkpoint was selected")
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
        mechanical_passed = bool(
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
        state_gate = transition_gate(best_validation, family=args.family)
        result = {
            "schema_version": 1,
            "scope": manifest["scope"],
            "status": "passed" if mechanical_passed else "failed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "physics_derived_loss_used": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
            "family": args.family,
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
            "validation_pair_count": sum(
                len(item) for item in validations.values()
            ),
            "epochs": epochs,
            "optimizer_updates": optimizer_updates,
            "expected_optimizer_updates": total_updates,
            "history": history,
            "best_checkpoint": {
                "path": str(best_checkpoint),
                "sha256": sha256_path(best_checkpoint),
                "epoch": int(best_record["epoch"]),
                "selection_metric": best_metric,
                "selected_at_budget_boundary": int(best_record["epoch"]) == epochs,
            },
            "best_validation": best_validation,
            "parent_selection_metric": parent_evaluation[
                "mean_shared_persistence_normalized_mse_ratio"
            ],
            "parent_improvement_fraction": 1.0
            - best_metric
            / float(
                parent_evaluation[
                    "mean_shared_persistence_normalized_mse_ratio"
                ]
            ),
            "numerical_gate": numerical_gate,
            "training_gate": {
                "exact_optimizer_update_count": exact_updates,
                "epoch_mean_training_loss_decreased": loss_decreased,
                "finite_validation_metrics": finite_metrics,
                "numerical_gate_passed": numerical_gate["passed"],
                "passed": mechanical_passed,
            },
            "transition_gate": state_gate,
            "paired_physics_evaluation_eligible": (
                mechanical_passed and state_gate["passed"]
            ),
            "peak_cuda_memory_GiB": torch.cuda.max_memory_allocated(device)
            / 2**30,
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
                "final/parent_improvement_fraction": result[
                    "parent_improvement_fraction"
                ],
                "final/selected_epoch": result["best_checkpoint"]["epoch"],
                "final/selected_at_budget_boundary": result["best_checkpoint"][
                    "selected_at_budget_boundary"
                ],
                "final/training_gate_passed": mechanical_passed,
                "final/transition_gate_passed": state_gate["passed"],
                "final/paired_physics_evaluation_eligible": result[
                    "paired_physics_evaluation_eligible"
                ],
                "compute/peak_cuda_memory_GiB": result[
                    "peak_cuda_memory_GiB"
                ],
                "compute/wall_seconds": result["wall_seconds"],
                "scope/held_out_85606_read": False,
                "scope/new_nersc_data_read": False,
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
