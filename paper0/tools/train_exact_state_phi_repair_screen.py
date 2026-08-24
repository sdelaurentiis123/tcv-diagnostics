#!/usr/bin/env python3
"""Train one old-85604 exact-state plus current-phi repair screen arm."""

from __future__ import annotations

import argparse
from dataclasses import fields as dataclass_fields
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    learning_rate,
    repository_commit,
    tensor_batch,
    verify_finished_wandb_run,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import FAMILY_FIELDS, load_official_catalog
from tcv_diagnostics.models.axial_operator import (
    AxialIncrementOperator3D,
    AxialOperatorConfig,
)
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
    normalized_error_metrics,
    persistence_normalized_state_derivative_loss,
)
from tcv_diagnostics.state_operator_data import (
    LeadTimeStateDataset,
    StateDerivativeRMS,
    fit_training_derivative_rms,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ARCHITECTURES = ("local_current_phi", "axial_current_phi")
E6B_FIELDS = FAMILY_FIELDS["e6b"]
SHARED_FIELDS = ("Ne", "Pe", "Pi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def authorize_manifest(
    manifest: Mapping[str, Any], *, architecture: str, seed: int
) -> None:
    if manifest.get("scope") != "post_ecrd_old_85604_exact_state_phi_repair_screen":
        raise ValueError("repair-screen scope differs")
    if manifest.get("development_run") != "85604":
        raise ValueError("repair-screen development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("repair-screen held-out flag differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise ValueError("repair-screen held-out access must be prohibited")
    if manifest.get("guard_frames_read_allowed") is not False:
        raise ValueError("repair-screen guard reads must be prohibited")
    if manifest.get("screen_training_authorized") is not True:
        raise ValueError("repair-screen training is not authorized")
    if manifest.get("three_seed_scaling_authorized") is not False:
        raise ValueError("three-seed scaling must remain unauthorized")
    if architecture not in manifest.get("architectures", {}):
        raise ValueError("architecture leaves the frozen repair screen")
    optimization = manifest["optimization"]
    if seed != int(optimization["screen_seed"]):
        raise ValueError("seed leaves the frozen repair screen")
    split = manifest["split"]
    expected_split = {
        "training_frames": [0, 432],
        "guard_frames": [432, 496],
        "validation_frames": [496, 624],
        "training_pair_count": 431,
        "validation_pair_count": 127,
        "lead_steps": [1],
        "history_frames": 1,
    }
    if any(split.get(key) != value for key, value in expected_split.items()):
        raise ValueError("repair-screen split differs")
    state = manifest["state"]
    if state.get("predicted_volume_fields") != list(E6B_FIELDS):
        raise ValueError("repair-screen evolved fields differ")
    if state.get("predicted_boundary") != "Bphi":
        raise ValueError("repair-screen boundary state differs")
    if state.get("auxiliary_context_fields") != ["phi"]:
        raise ValueError("repair-screen auxiliary context differs")
    if state.get("future_auxiliary_context_allowed") is not False:
        raise ValueError("future auxiliary context must be prohibited")


def build_model(
    architecture: str, record: Mapping[str, Any]
) -> tuple[nn.Module, CodecFreeOperatorConfig | AxialOperatorConfig]:
    if architecture == "local_current_phi":
        allowed = {field.name for field in dataclass_fields(CodecFreeOperatorConfig)}
        locked = {
            "state_family",
            "history_frames",
            "auxiliary_context_channels",
            "predict_boundary",
        }
        values = {
            key: value
            for key, value in record.items()
            if key in allowed and key not in locked
        }
        if "channel_multipliers" in values:
            values["channel_multipliers"] = tuple(values["channel_multipliers"])
        config = CodecFreeOperatorConfig(
            state_family="e6b",
            history_frames=1,
            auxiliary_context_channels=1,
            predict_boundary=True,
            **values,
        )
        return CodecFreeIncrementOperator3D(config), config
    if architecture == "axial_current_phi":
        allowed = {field.name for field in dataclass_fields(AxialOperatorConfig)}
        locked = {
            "state_family",
            "history_frames",
            "auxiliary_context_channels",
            "static_context_channels",
            "predict_boundary",
        }
        values = {
            key: value
            for key, value in record.items()
            if key in allowed and key not in locked
        }
        config = AxialOperatorConfig(
            state_family="e6b",
            history_frames=1,
            auxiliary_context_channels=1,
            static_context_channels=0,
            predict_boundary=True,
            **values,
        )
        return AxialIncrementOperator3D(config), config
    raise ValueError(f"unsupported repair architecture {architecture!r}")


def load_locked_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{name} SHA-256 differs")
    return load_strict_json(path)


def verify_prerequisites(manifest: Mapping[str, Any]) -> dict[str, Any]:
    prerequisites = manifest.get("prerequisites", {})
    reduction_lock = prerequisites.get("stage1_reduction", {})
    reduction = load_locked_json(reduction_lock, name="Stage-1 reduction")
    if reduction.get("development_run") != "85604":
        raise ValueError("Stage-1 reduction development run differs")
    if reduction.get("held_out_85606_read") is not False:
        raise ValueError("Stage-1 reduction held-out flag differs")
    if reduction.get("decision") != (
        "retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation"
    ):
        raise ValueError("Stage-1 reduction does not authorize exact-state repair")

    smoke_lock = prerequisites.get("axial_smoke", {})
    smoke = load_locked_json(smoke_lock, name="axial smoke")
    if smoke.get("development_run") != "85604":
        raise ValueError("axial smoke development run differs")
    if smoke.get("held_out_85606_read") is not False:
        raise ValueError("axial smoke held-out flag differs")
    if smoke.get("status") != "passed":
        raise ValueError("axial smoke did not pass")
    gates = smoke.get("mechanical_gates", {})
    if not gates or not all(bool(value) for value in gates.values()):
        raise ValueError("axial smoke mechanical gates did not all pass")

    baseline_lock = prerequisites.get("baseline_e6b_seed1701", {})
    baseline = load_locked_json(baseline_lock, name="baseline E6B seed 1701")
    if baseline.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("baseline E6B scope differs")
    if baseline.get("development_run") != "85604":
        raise ValueError("baseline E6B development run differs")
    if baseline.get("held_out_85606_read") is not False:
        raise ValueError("baseline E6B held-out flag differs")
    if baseline.get("family") != "e6b" or int(baseline.get("seed", -1)) != 1701:
        raise ValueError("baseline E6B identity differs")
    metric = float(baseline["best_checkpoint"]["selection_metric"])
    expected_metric = float(baseline_lock.get("selection_metric", float("nan")))
    if not math.isfinite(expected_metric) or metric != expected_metric:
        raise ValueError("baseline E6B selection metric differs")
    return {
        "stage1_reduction": dict(reduction_lock),
        "axial_smoke": dict(smoke_lock),
        "baseline_e6b_seed1701": {
            **dict(baseline_lock),
            "selection_metric": metric,
        },
    }


def repair_tensor_batch(
    item: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    values = tensor_batch(item, device)
    if "auxiliary_context" not in item:
        raise ValueError("repair sample lacks current-phi context")
    auxiliary = np.asarray(item["auxiliary_context"])
    if auxiliary.ndim != 5 or auxiliary.shape[1] != 1:
        raise ValueError("current-phi context shape differs")
    values["auxiliary_context"] = torch.from_numpy(auxiliary).unsqueeze(0).to(device)
    return values


def evaluate_repair(
    model: nn.Module,
    dataset: LeadTimeStateDataset,
    *,
    device: torch.device,
) -> dict[str, Any]:
    squared_error = np.zeros(len(E6B_FIELDS), dtype=np.float64)
    persistence_error = np.zeros(len(E6B_FIELDS), dtype=np.float64)
    element_count = np.zeros(len(E6B_FIELDS), dtype=np.int64)
    boundary_error = np.zeros(2, dtype=np.float64)
    boundary_persistence = np.zeros(2, dtype=np.float64)
    boundary_count = np.zeros(2, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            values = repair_tensor_batch(dataset[index], device)
            prediction = model(
                values["context"],
                values["lead_steps"],
                values["context_boundary"],
                values["auxiliary_context"],
            )
            error = (
                prediction.volume.float() - values["target_derivative"].float()
            ).square()
            baseline = values["target_derivative"].float().square()
            squared_error += error.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            persistence_error += baseline.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            element_count += (
                error.shape[0] * error.shape[2] * error.shape[3] * error.shape[4]
            )
            if prediction.boundary is None:
                raise AssertionError("repair E6B boundary prediction is absent")
            target_boundary = values["target_boundary_derivative"].float()
            difference = (prediction.boundary.float() - target_boundary).square()
            boundary_error += difference.sum(dim=(0, 2)).cpu().numpy()
            boundary_persistence += (
                target_boundary.square().sum(dim=(0, 2)).cpu().numpy()
            )
            boundary_count += difference.shape[0] * difference.shape[2]

    mse = squared_error / element_count
    baseline_mse = persistence_error / element_count
    per_field = {
        field: {
            "model_derivative_mse": float(mse[position]),
            "zero_derivative_persistence_mse": float(baseline_mse[position]),
            "persistence_relative_skill": (
                float(1.0 - mse[position] / baseline_mse[position])
                if baseline_mse[position] > 0.0
                else None
            ),
        }
        for position, field in enumerate(E6B_FIELDS)
    }
    shared_positions = [E6B_FIELDS.index(field) for field in SHARED_FIELDS]
    shared_model = float(np.mean(mse[shared_positions]))
    shared_baseline = float(np.mean(baseline_mse[shared_positions]))
    return {
        "pair_count": len(dataset),
        "per_field": per_field,
        "shared_fields": list(SHARED_FIELDS),
        "shared_field_mean_model_derivative_mse": shared_model,
        "shared_field_mean_zero_derivative_persistence_mse": shared_baseline,
        "shared_field_persistence_relative_skill": (
            1.0 - shared_model / shared_baseline if shared_baseline > 0.0 else None
        ),
        "boundary_by_side": {
            side: {
                "model_derivative_mse": float(
                    boundary_error[position] / boundary_count[position]
                ),
                "zero_derivative_persistence_mse": float(
                    boundary_persistence[position] / boundary_count[position]
                ),
            }
            for position, side in enumerate(("inner", "outer"))
        },
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: CodecFreeOperatorConfig | AxialOperatorConfig,
    derivative_rms: StateDerivativeRMS,
    architecture: str,
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
            "family": "e6b",
            "architecture_kind": architecture,
            "auxiliary_context_fields": ["phi"],
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "selection_metric": selection_metric,
            "paper0_commit": paper0_commit,
        },
        path,
    )


def reload_and_equivariance_gate(
    *,
    checkpoint: Path,
    architecture: str,
    architecture_record: Mapping[str, Any],
    validation: LeadTimeStateDataset,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model, _ = build_model(architecture, architecture_record)
    model = model.to(device)
    model.load_state_dict(payload["model"], strict=True)
    reload_exact = all(
        torch.equal(payload["model"][name].to(device), model.state_dict()[name])
        for name in payload["model"]
    )
    values = repair_tensor_batch(validation[0], device)
    shift = 7
    model.eval()
    with torch.inference_mode():
        reference = model(
            values["context"],
            values["lead_steps"],
            values["context_boundary"],
            values["auxiliary_context"],
        )
        shifted = model(
            torch.roll(values["context"], shift, dims=-1),
            values["lead_steps"],
            values["context_boundary"],
            torch.roll(values["auxiliary_context"], shift, dims=-1),
        )
    if reference.boundary is None or shifted.boundary is None:
        raise AssertionError("repair E6B boundary prediction is absent")
    volume = normalized_error_metrics(
        shifted.volume, torch.roll(reference.volume, shift, dims=-1)
    )
    boundary = normalized_error_metrics(shifted.boundary, reference.boundary)
    convolution_strides = [
        tuple(module.stride)
        for module in model.modules()
        if isinstance(module, torch.nn.Conv3d)
    ]
    no_toroidal_stride = bool(convolution_strides) and all(
        stride[-1] == 1 for stride in convolution_strides
    )
    passed = bool(
        reload_exact
        and no_toroidal_stride
        and volume["normalized_maximum_absolute_error"] <= 1.0e-4
        and volume["normalized_root_mean_square_error"] <= 1.0e-5
        and boundary["normalized_maximum_absolute_error"] <= 1.0e-6
        and boundary["normalized_root_mean_square_error"] <= 1.0e-7
    )
    return {
        "checkpoint_reload_exact": reload_exact,
        "integer_toroidal_shift": shift,
        "no_toroidal_stride": no_toroidal_stride,
        "volume": volume,
        "boundary": boundary,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("repair-screen manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(manifest, architecture=args.architecture, seed=args.seed)
    prerequisites = verify_prerequisites(manifest)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from launch lock")
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("repair-screen training requires an allocated CUDA GPU")
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
        family="e6b",
        split="train",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=args.seed,
        auxiliary_context_fields=("phi",),
    )
    try:
        derivative_rms = fit_training_derivative_rms(scale_dataset)
    finally:
        scale_dataset.close()
    atomic_json(args.output / "derivative_rms.json", derivative_rms.to_record())

    train = LeadTimeStateDataset(
        catalog,
        family="e6b",
        split="train",
        lead_steps=(1,),
        history_frames=1,
        augment=True,
        seed=args.seed,
        auxiliary_context_fields=("phi",),
    )
    validation = LeadTimeStateDataset(
        catalog,
        family="e6b",
        split="validation",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=args.seed,
        auxiliary_context_fields=("phi",),
    )
    split = manifest["split"]
    if len(train) != int(split["training_pair_count"]):
        raise ValueError("repair-screen training pair count differs")
    if len(validation) != int(split["validation_pair_count"]):
        raise ValueError("repair-screen validation pair count differs")

    architecture_record = manifest["architectures"][args.architecture]
    model, config = build_model(args.architecture, architecture_record)
    model = model.to(device)
    optimization = manifest["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["peak_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    volume_rms = torch.tensor(derivative_rms.volume, device=device)
    if derivative_rms.boundary is None:
        raise AssertionError("repair-screen Bphi derivative scale is absent")
    boundary_rms = torch.tensor(derivative_rms.boundary, device=device)
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
        job_type="old-85604-exact-state-phi-repair-screen",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "exact-state",
            "current-phi",
            args.architecture,
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
            **architecture_record,
            **optimization,
            "scope": manifest["scope"],
            "architecture_kind": args.architecture,
            "state_family": "e6b",
            "auxiliary_context_fields": ["phi"],
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
                np.random.SeedSequence([args.seed, epoch, 0x504849])
            ).permutation(len(train))
            raw_losses: list[float] = []
            gradient_norms: list[float] = []
            model.train()
            for group_start in range(0, len(order), accumulation):
                group = order[group_start : group_start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    values = repair_tensor_batch(train[int(index)], device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        prediction = model(
                            values["context"],
                            values["lead_steps"],
                            values["context_boundary"],
                            values["auxiliary_context"],
                        )
                        loss, _ = persistence_normalized_state_derivative_loss(
                            prediction,
                            values["target_derivative"],
                            volume_rms,
                            values["target_boundary_derivative"],
                            boundary_rms,
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("repair-screen training loss is non-finite")
                    raw_losses.append(float(loss.detach().cpu()))
                    (loss / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(optimization["gradient_clip_norm"]),
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("repair-screen gradient norm is non-finite")
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

            validation_record = evaluate_repair(model, validation, device=device)
            selection_metric = float(
                validation_record["shared_field_mean_model_derivative_mse"]
            )
            checkpoint = args.output / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                derivative_rms=derivative_rms,
                architecture=args.architecture,
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
                "training_persistence_normalized_loss_mean": float(np.mean(raw_losses)),
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
            raise AssertionError("no repair-screen checkpoint was selected")
        numerical_gate = reload_and_equivariance_gate(
            checkpoint=best_checkpoint,
            architecture=args.architecture,
            architecture_record=architecture_record,
            validation=validation,
            device=device,
        )
        exact_updates = optimizer_updates == total_updates
        loss_decreased = (
            history[-1]["training_persistence_normalized_loss_mean"]
            < history[0]["training_persistence_normalized_loss_mean"]
        )
        finite_metrics = all(
            math.isfinite(
                float(record["validation"]["shared_field_mean_model_derivative_mse"])
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
                "shared_field_mean_model_derivative_mse"
            ],
        )
        best_validation = best_record["validation"]
        every_field_positive = all(
            float(best_validation["per_field"][field]["persistence_relative_skill"])
            > 0.0
            for field in E6B_FIELDS
        )
        baseline_metric = float(
            prerequisites["baseline_e6b_seed1701"]["selection_metric"]
        )
        improvement_fraction = 1.0 - best_metric / baseline_metric
        screen_gates = {
            "training_gate_passed": training_gate,
            "every_e6b_field_positive_persistence_skill": every_field_positive,
            "at_least_15_percent_shared_mse_improvement_over_seed1701_e6b": (
                improvement_fraction >= 0.15
            ),
        }
        advance = all(screen_gates.values())
        result = {
            "schema_version": 1,
            "scope": manifest["scope"],
            "status": "passed" if training_gate else "failed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "physics_derived_loss_used": False,
            "family": "e6b",
            "architecture_kind": args.architecture,
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "prerequisites": prerequisites,
            "state": {
                "predicted_volume_fields": list(E6B_FIELDS),
                "predicted_boundary": "Bphi",
                "auxiliary_context_fields": ["phi"],
                "future_auxiliary_context_read": False,
                "current_phi_rollout_ready_without_elliptic_closure": False,
            },
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
                "epoch": int(best_record["epoch"]),
                "selection_metric": best_metric,
            },
            "baseline_e6b_seed1701_shared_mse": baseline_metric,
            "shared_mse_improvement_fraction": improvement_fraction,
            "numerical_gate": numerical_gate,
            "training_gate": {
                "exact_optimizer_update_count": exact_updates,
                "epoch_mean_training_loss_decreased": loss_decreased,
                "finite_validation_metrics": bool(finite_metrics),
                "numerical_gate_passed": numerical_gate["passed"],
                "passed": training_gate,
            },
            "screen_gates": screen_gates,
            "advance_to_three_seed_scaling": advance,
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
                "final/shared_mse_improvement_fraction": improvement_fraction,
                "final/training_gate_passed": training_gate,
                "final/advance_to_three_seed_scaling": advance,
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
