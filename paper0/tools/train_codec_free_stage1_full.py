#!/usr/bin/env python3
"""Train one frozen full old-85604 codec-free Stage-1 arm."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

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
from tcv_diagnostics.model_training_data import load_official_catalog
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--family", choices=("c5p", "e6b"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def save_full_checkpoint(
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
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "selection_metric": selection_metric,
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
        raise ValueError("full Stage-1 manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    if manifest.get("development_run") != "85604":
        raise ValueError("full Stage-1 development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("full Stage-1 held-out flag differs")
    if manifest.get("pilot_training_authorized") is not False:
        raise ValueError("pilot authorization must be false")
    if manifest.get("full_training_authorized") is not True:
        raise ValueError("full Stage-1 training is not authorized")
    optimization = manifest["optimization"]
    if args.seed not in tuple(int(value) for value in optimization["seeds"]):
        raise ValueError("seed leaves the frozen full Stage-1 matrix")
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from launch lock")
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("full Stage-1 training requires an allocated CUDA GPU")
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
        family=args.family,
        split="train",
        lead_steps=(1,),
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
        family=args.family,
        split="train",
        lead_steps=(1,),
        history_frames=1,
        augment=True,
        seed=args.seed,
    )
    validation = LeadTimeStateDataset(
        catalog,
        family=args.family,
        split="validation",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=args.seed,
    )
    split = manifest["split"]
    if len(train) != int(split["training_pair_count"]):
        raise ValueError("training pair count differs")
    if len(validation) != int(split["validation_pair_count"]):
        raise ValueError("validation pair count differs")

    architecture = manifest["architecture"]
    config = CodecFreeOperatorConfig(
        state_family=args.family,
        history_frames=1,
        base_channels=int(architecture["base_channels"]),
        channel_multipliers=tuple(architecture["channel_multipliers"]),
        blocks_per_level=int(architecture["blocks_per_level"]),
        lead_embedding_channels=int(architecture["lead_embedding_channels"]),
        group_norm_maximum_groups=int(
            architecture["group_norm_maximum_groups"]
        ),
        kernel_size=int(architecture["kernel_size"]),
        predict_boundary=args.family == "e6b",
        zero_initialize_output=bool(architecture["zero_initialize_output"]),
    )
    model = CodecFreeIncrementOperator3D(config).to(device)
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
    warmup_updates = max(
        1, math.ceil(total_updates * float(optimization["warmup_fraction"]))
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
        job_type="codec-free-stage1-full",
        tags=("paper0", "85604", "old-data", "codec-free", args.family, "full"),
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
            **architecture,
            **optimization,
            "scope": manifest["scope"],
            "family": args.family,
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
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
                np.random.SeedSequence([args.seed, epoch, 0x53544147])
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
                        raise RuntimeError("full Stage-1 training loss is non-finite")
                    raw_losses.append(float(loss.detach().cpu()))
                    (loss / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimization["gradient_clip_norm"])
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("full Stage-1 gradient norm is non-finite")
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
                if optimizer_updates % 20 == 0 or optimizer_updates == total_updates:
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

            validation_record = evaluate(
                model, validation, family=args.family, device=device
            )
            selection_metric = float(
                validation_record["shared_field_mean_model_derivative_mse"]
            )
            checkpoint = args.output / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_full_checkpoint(
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
                "validation": validation_record,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": sha256_path(checkpoint),
                },
            }
            history.append(epoch_record)
            run.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_persistence_normalized_loss": epoch_record[
                        "training_persistence_normalized_loss_mean"
                    ],
                    "validation/shared_field_mean_derivative_mse": selection_metric,
                    "validation/shared_field_persistence_relative_skill": validation_record[
                        "shared_field_persistence_relative_skill"
                    ],
                },
                step=optimizer_updates,
            )

        if best_checkpoint is None:
            raise AssertionError("no full Stage-1 checkpoint was selected")
        numerical_gate = reload_and_equivariance_gate(
            checkpoint=best_checkpoint,
            config=config,
            validation=validation,
            device=device,
        )
        exact_updates = optimizer_updates == total_updates
        loss_decreased = (
            history[-1]["training_persistence_normalized_loss_mean"]
            < history[0]["training_persistence_normalized_loss_mean"]
        )
        finite_metrics = bool(
            all(
                math.isfinite(
                    float(
                        record["validation"][
                            "shared_field_mean_model_derivative_mse"
                        ]
                    )
                )
                for record in history
            )
        )
        training_gate = bool(
            exact_updates and loss_decreased and finite_metrics and numerical_gate["passed"]
        )
        result = {
            "schema_version": 1,
            "scope": manifest["scope"],
            "status": "passed" if training_gate else "failed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "family": args.family,
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "architecture": model.to_record(),
            "loss": manifest["loss"],
            "derivative_rms": derivative_rms.to_record(),
            "training_pair_count": len(train),
            "validation_pair_count": len(validation),
            "epochs": epochs,
            "optimizer_updates": optimizer_updates,
            "expected_optimizer_updates": total_updates,
            "history": history,
            "best_checkpoint": {
                "path": str(best_checkpoint),
                "sha256": sha256_path(best_checkpoint),
                "selection_metric": best_metric,
            },
            "numerical_gate": numerical_gate,
            "training_gate": {
                "exact_optimizer_update_count": exact_updates,
                "epoch_mean_training_loss_decreased": loss_decreased,
                "finite_validation_metrics": finite_metrics,
                "numerical_gate_passed": numerical_gate["passed"],
                "passed": training_gate,
            },
            "physics_derived_loss_used": False,
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
                "final/best_shared_field_derivative_mse": best_metric,
                "final/training_gate_passed": training_gate,
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
        validation.close()

    atomic_json(args.output / "result.json", result)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb, remote_path=remote_path, expected_id=spec.run_id
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
