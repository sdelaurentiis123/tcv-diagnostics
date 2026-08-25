#!/usr/bin/env python3
"""Fine-tune the old-85604 lead-one operator on four-step predicted feedback."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    learning_rate,
    reload_and_equivariance_gate,
    repository_commit,
    verify_finished_wandb_run,
)
from paper0.tools.train_codec_free_stage2_multilead import (
    build_model,
    validate_parent_config,
)
from tcv_diagnostics.autoregressive_training import (
    AutoregressiveStateWindowDataset,
    autoregressive_forecast_sequence,
    feedback_loss_weights,
    state_rms_normalized_mse,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import FAMILY_FIELDS, load_official_catalog
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
)
from tcv_diagnostics.state_operator_data import LeadTimeStateDataset
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SCOPE = "post_ecrd_old_85604_four_step_feedback_pilot"
FAMILY = "c5p"
FIELDS = FAMILY_FIELDS[FAMILY]
TRAINING_HORIZON = 4
SELECTION_HORIZONS = (1, 4, 8)
SELECTION_STARTS = tuple(range(496, 612, 5))


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
    """Reject any mutation of the prospective pilot contract."""

    flags = {
        "scope": SCOPE,
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "steering_allowed": False,
        "pilot_training_authorized": True,
        "confirmation_seed_training_authorized": False,
        "state_family": FAMILY,
        "wandb_required": True,
    }
    if any(manifest.get(key) != value for key, value in flags.items()):
        raise ValueError("four-step pilot scope or authorization differs")
    if tuple(manifest.get("fields", ())) != FIELDS:
        raise ValueError("four-step pilot fields differ")

    expected_split = {
        "training_frames": [0, 432],
        "guard_frames": [432, 496],
        "validation_frames": [496, 624],
        "history_frames": 1,
        "training_unroll_horizon": 4,
        "training_window_count": 428,
        "full_validation_window_count_by_horizon": {
            "1": 127,
            "4": 124,
            "8": 120,
        },
        "selection_current_frames": list(SELECTION_STARTS),
        "selection_horizons": list(SELECTION_HORIZONS),
    }
    if manifest.get("split") != expected_split:
        raise ValueError("four-step pilot split differs")

    expected_architecture = {
        "base_channels": 24,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "lead_embedding_channels": 128,
        "group_norm_maximum_groups": 8,
        "kernel_size": 3,
        "toroidal_stride": 1,
        "latent_codec": False,
        "zero_initialize_output": True,
        "parameter_count": 2174021,
    }
    if manifest.get("architecture") != expected_architecture:
        raise ValueError("four-step pilot architecture differs")

    expected_loss = {
        "name": "detached_four_step_pushforward_plus_retained_one_step_state_mse",
        "feedback_uses_predicted_intermediate_states": True,
        "teacher_forcing_after_initial_context": False,
        "backpropagate_through_previous_predicted_states": False,
        "direct_one_step_weight": 0.5,
        "equal_rollout_mean_weight": 0.5,
        "effective_step_weights": [0.625, 0.125, 0.125, 0.125],
        "field_scale": "frozen_stage2_training_derivative_rms",
        "physics_derived_quantities_used": False,
    }
    if manifest.get("loss") != expected_loss:
        raise ValueError("four-step pilot loss differs")
    computed_weights = feedback_loss_weights(
        horizon=TRAINING_HORIZON,
        direct_one_step_weight=float(expected_loss["direct_one_step_weight"]),
    )
    if tuple(expected_loss["effective_step_weights"]) != computed_weights:
        raise ValueError("four-step pilot effective weights differ")

    expected_optimization = {
        "seed": 1702,
        "initialize_from_parent_model_only": True,
        "restore_parent_optimizer": False,
        "epochs": 6,
        "sample_batch_size": 1,
        "gradient_accumulation_windows": 4,
        "expected_optimizer_updates": 642,
        "optimizer": "AdamW",
        "peak_learning_rate": 2.0e-5,
        "minimum_learning_rate": 2.0e-6,
        "weight_decay": 1.0e-4,
        "warmup_fraction": 0.05,
        "gradient_clip_norm": 1.0,
        "autocast": "bfloat16",
        "cudnn_tf32": False,
        "matmul_tf32": False,
    }
    if manifest.get("optimization") != expected_optimization:
        raise ValueError("four-step pilot optimization differs")
    if seed != expected_optimization["seed"]:
        raise ValueError("seed leaves the frozen four-step pilot")


def _locked_json(record: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{name} SHA-256 differs")
    return load_strict_json(path)


def verify_and_load_parent(
    manifest: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[CodecFreeIncrementOperator3D, CodecFreeOperatorConfig, dict[str, Any]]:
    parent = manifest.get("parent", {})
    if int(parent.get("seed", -1)) != 1702:
        raise ValueError("four-step parent seed differs")
    parent_result = _locked_json(parent.get("result", {}), name="parent result")
    required_result = {
        "scope": "post_ecrd_old_85604_stage2_multilead_scaling",
        "status": "passed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": FAMILY,
        "seed": 1702,
    }
    if any(parent_result.get(key) != value for key, value in required_result.items()):
        raise ValueError("four-step parent result contract differs")
    checkpoint_record = parent.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint_record.get("path", "")))
    checkpoint_sha = str(checkpoint_record.get("sha256", ""))
    assert_development_path(checkpoint_path)
    if parent_result.get("best_checkpoint", {}).get("path") != str(checkpoint_path):
        raise ValueError("parent result checkpoint path differs")
    if parent_result.get("best_checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("parent result checkpoint SHA-256 differs")
    if not checkpoint_sha or sha256_path(checkpoint_path) != checkpoint_sha:
        raise ValueError("parent checkpoint SHA-256 differs")

    model, config = build_model(manifest["architecture"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if payload.get("family") != FAMILY or int(payload.get("seed", -1)) != 1702:
        raise ValueError("parent checkpoint identity differs")
    config_validation = validate_parent_config(
        payload.get("config", {}), config.to_record()
    )
    if payload.get("derivative_rms") != parent_result.get("derivative_rms"):
        raise ValueError("parent checkpoint derivative RMS differs")
    model = model.to(device)
    model.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(payload["model"][name].to(device), model.state_dict()[name])
        for name in payload["model"]
    )
    if not bitwise:
        raise AssertionError("parent model did not load bitwise")
    return model, config, {
        "result": dict(parent["result"]),
        "checkpoint": dict(checkpoint_record),
        "checkpoint_reload_bitwise": True,
        "checkpoint_config_validation": config_validation,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_optimizer_updates": int(payload["optimizer_updates"]),
        "checkpoint_paper0_commit": payload.get("paper0_commit"),
    }


def derivative_rms_tensor(
    manifest: Mapping[str, Any], *, device: torch.device
) -> torch.Tensor:
    record = manifest.get("derivative_rms", {})
    if (
        record.get("family") != FAMILY
        or tuple(record.get("fields", ())) != FIELDS
        or record.get("fit_split") != "train"
        or int(record.get("source_pair_count", -1)) != 2129
        or record.get("physics_derived_quantity") is not False
    ):
        raise ValueError("four-step derivative RMS contract differs")
    values = tuple(float(record.get("volume", {}).get(field, float("nan"))) for field in FIELDS)
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if not torch.isfinite(tensor).all() or not torch.all(tensor > 0):
        raise ValueError("four-step derivative RMS values differ")
    return tensor


def tensor_window(item: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "context": torch.from_numpy(item["context"]).unsqueeze(0).to(device),
        "targets": torch.from_numpy(item["targets"]).unsqueeze(0).to(device),
    }


def _empty_accumulator() -> dict[str, np.ndarray]:
    return {
        "model": np.zeros(len(FIELDS), dtype=np.float64),
        "persistence": np.zeros(len(FIELDS), dtype=np.float64),
        "count": np.zeros(len(FIELDS), dtype=np.int64),
    }


def evaluate_state_rollouts(
    model: CodecFreeIncrementOperator3D,
    dataset: AutoregressiveStateWindowDataset,
    *,
    horizons: Sequence[int],
    device: torch.device,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate autonomous field state without physics-derived quantities."""

    requested = tuple(sorted({int(value) for value in horizons}))
    if not requested or requested[0] <= 0 or requested[-1] > dataset.horizon:
        raise ValueError("evaluation horizons leave the provided windows")
    selected = tuple(range(len(dataset))) if indices is None else tuple(map(int, indices))
    if not selected or min(selected) < 0 or max(selected) >= len(dataset):
        raise ValueError("evaluation indices leave the provided windows")
    accumulators = {horizon: _empty_accumulator() for horizon in requested}
    starts: list[int] = []
    model.eval()
    with torch.inference_mode():
        for index in selected:
            item = dataset[index]
            starts.append(int(item["current_frame_index"]))
            values = tensor_window(item, device)
            sequence = autoregressive_forecast_sequence(
                model, values["context"], steps=requested[-1]
            )
            persistence = values["context"][:, -1].float()
            for horizon in requested:
                candidate = sequence[horizon - 1].float()
                target = values["targets"][:, horizon - 1].float()
                error = (candidate - target).square()
                baseline = (persistence - target).square()
                record = accumulators[horizon]
                record["model"] += error.sum(dim=(0, 2, 3, 4)).cpu().numpy()
                record["persistence"] += baseline.sum(dim=(0, 2, 3, 4)).cpu().numpy()
                record["count"] += np.asarray(
                    [error.shape[0] * error.shape[2] * error.shape[3] * error.shape[4]]
                    * len(FIELDS),
                    dtype=np.int64,
                )

    per_horizon: dict[str, Any] = {}
    horizon_ratios: list[float] = []
    for horizon in requested:
        record = accumulators[horizon]
        model_mse = record["model"] / record["count"]
        persistence_mse = record["persistence"] / record["count"]
        ratios = model_mse / persistence_mse
        per_field = {
            field: {
                "model_state_mse": float(model_mse[position]),
                "persistence_state_mse": float(persistence_mse[position]),
                "model_over_persistence_mse": float(ratios[position]),
                "persistence_relative_skill": float(1.0 - ratios[position]),
            }
            for position, field in enumerate(FIELDS)
        }
        mean_ratio = float(np.mean(ratios))
        horizon_ratios.append(mean_ratio)
        per_horizon[str(horizon)] = {
            "window_count": len(selected),
            "per_field": per_field,
            "mean_field_model_state_mse": float(np.mean(model_mse)),
            "mean_field_persistence_state_mse": float(np.mean(persistence_mse)),
            "mean_field_model_over_persistence_mse": mean_ratio,
            "mean_field_persistence_relative_skill": 1.0 - mean_ratio,
        }
    return {
        "current_frames": starts,
        "horizons": per_horizon,
        "selection_metric": float(np.mean(horizon_ratios)),
        "selection_metric_definition": (
            "unweighted_mean_of_fieldwise_state_mse_over_persistence_mse"
        ),
        "future_truth_used_as_context": False,
        "physics_derived_metric": False,
    }


def selected_indices(
    dataset: AutoregressiveStateWindowDataset, starts: Sequence[int]
) -> tuple[int, ...]:
    by_current = {window.current: index for index, window in enumerate(dataset.windows)}
    requested = tuple(map(int, starts))
    if len(set(requested)) != len(requested) or any(value not in by_current for value in requested):
        raise ValueError("selection starts leave the frozen validation windows")
    return tuple(by_current[value] for value in requested)


def save_checkpoint(
    path: Path,
    *,
    model: CodecFreeIncrementOperator3D,
    optimizer: torch.optim.Optimizer,
    config: CodecFreeOperatorConfig,
    manifest: Mapping[str, Any],
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
            "derivative_rms": manifest["derivative_rms"],
            "family": FAMILY,
            "stage": "four_step_detached_feedback_finetune",
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "selection_metric": selection_metric,
            "parent_checkpoint": manifest["parent"]["checkpoint"],
            "loss": manifest["loss"],
            "paper0_commit": paper0_commit,
        },
        path,
    )


def state_gate(
    *,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    parent_error = {
        horizon: float(parent["horizons"][str(horizon)]["mean_field_model_state_mse"])
        for horizon in SELECTION_HORIZONS
    }
    candidate_error = {
        horizon: float(candidate["horizons"][str(horizon)]["mean_field_model_state_mse"])
        for horizon in SELECTION_HORIZONS
    }
    ratios = {
        horizon: candidate_error[horizon] / parent_error[horizon]
        for horizon in SELECTION_HORIZONS
    }
    mean_long_improvement = 1.0 - float(
        np.mean([candidate_error[4], candidate_error[8]])
        / np.mean([parent_error[4], parent_error[8]])
    )
    gates = {
        "one_step_error_retained": ratios[1]
        <= float(thresholds["maximum_one_step_error_ratio_to_parent"]),
        "four_step_error_nonincreasing": ratios[4]
        <= float(thresholds["maximum_four_step_error_ratio_to_parent"]),
        "eight_step_error_nonincreasing": ratios[8]
        <= float(thresholds["maximum_eight_step_error_ratio_to_parent"]),
        "mean_four_eight_improves_at_least_five_percent": mean_long_improvement
        >= float(thresholds["minimum_mean_four_eight_improvement_fraction"]),
    }
    return {
        "candidate_over_parent_mean_field_mse": {
            str(horizon): ratios[horizon] for horizon in SELECTION_HORIZONS
        },
        "mean_four_eight_improvement_fraction": mean_long_improvement,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("four-step pilot manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(manifest, seed=args.seed)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from launch lock")
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("four-step feedback training requires an allocated CUDA GPU")
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
    train = AutoregressiveStateWindowDataset(
        catalog,
        family=FAMILY,
        split="train",
        horizon=TRAINING_HORIZON,
        augment=True,
        seed=args.seed,
        current_interval=(0, 428),
    )
    selection = AutoregressiveStateWindowDataset(
        catalog,
        family=FAMILY,
        split="validation",
        horizon=8,
        augment=False,
        seed=args.seed,
        current_interval=(496, 616),
    )
    full_validations = {
        horizon: AutoregressiveStateWindowDataset(
            catalog,
            family=FAMILY,
            split="validation",
            horizon=horizon,
            augment=False,
            seed=args.seed,
        )
        for horizon in SELECTION_HORIZONS
    }
    lead1_validation = LeadTimeStateDataset(
        catalog,
        family=FAMILY,
        split="validation",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=args.seed,
    )
    if len(train) != int(manifest["split"]["training_window_count"]):
        raise ValueError("four-step training-window count differs")
    for horizon, dataset in full_validations.items():
        expected = manifest["split"]["full_validation_window_count_by_horizon"][str(horizon)]
        if len(dataset) != int(expected):
            raise ValueError(f"full validation count differs at horizon {horizon}")
    selection_subset = selected_indices(selection, SELECTION_STARTS)

    model, config, parent_load = verify_and_load_parent(manifest, device=device)
    volume_rms = derivative_rms_tensor(manifest, device=device)
    parent_selection = evaluate_state_rollouts(
        model,
        selection,
        horizons=SELECTION_HORIZONS,
        device=device,
        indices=selection_subset,
    )
    parent_full = {
        str(horizon): evaluate_state_rollouts(
            model,
            dataset,
            horizons=(horizon,),
            device=device,
        )["horizons"][str(horizon)]
        for horizon, dataset in full_validations.items()
    }
    parent_full_record = {
        "horizons": parent_full,
        "future_truth_used_as_context": False,
        "physics_derived_metric": False,
    }
    parent_record = {
        **parent_load,
        "evaluation_before_optimizer_construction": True,
        "selection": parent_selection,
        "full_validation": parent_full_record,
    }
    atomic_json(args.output / "parent_state_evaluation.json", parent_record)

    optimization = manifest["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["peak_learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    epochs = int(optimization["epochs"])
    accumulation = int(optimization["gradient_accumulation_windows"])
    updates_per_epoch = math.ceil(len(train) / accumulation)
    total_updates = updates_per_epoch * epochs
    if total_updates != int(optimization["expected_optimizer_updates"]):
        raise ValueError("four-step expected optimizer-update count differs")
    warmup_updates = max(
        1, math.ceil(total_updates * float(optimization["warmup_fraction"]))
    )
    step_weights = feedback_loss_weights(
        horizon=TRAINING_HORIZON,
        direct_one_step_weight=float(manifest["loss"]["direct_one_step_weight"]),
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
        job_type="old-85604-four-step-feedback-pilot",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "codec-free",
            "c5p",
            "autoregressive",
            "four-step",
            "detached-feedback",
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
            "scope": SCOPE,
            "state_family": FAMILY,
            "training_horizon": TRAINING_HORIZON,
            "selection_horizons": list(SELECTION_HORIZONS),
            "selection_current_frames": list(SELECTION_STARTS),
            "loss": manifest["loss"],
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "parent_checkpoint": manifest["parent"]["checkpoint"],
            "parent_selection_metric": parent_selection["selection_metric"],
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "physics_derived_loss_used": False,
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
                np.random.SeedSequence([args.seed, epoch, 0x50555348])
            ).permutation(len(train))
            raw_losses: list[float] = []
            per_step_losses: list[list[float]] = [list() for _ in step_weights]
            gradient_norms: list[float] = []
            model.train()
            for group_start in range(0, len(order), accumulation):
                group = order[group_start : group_start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                for index in group:
                    values = tensor_window(train[int(index)], device)
                    current = values["context"]
                    sample_total = 0.0
                    for step, weight in enumerate(step_weights):
                        lead = torch.ones(
                            current.shape[0], dtype=current.dtype, device=device
                        )
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            forecast = model.forecast(current, lead)
                            loss, _ = state_rms_normalized_mse(
                                forecast.volume,
                                values["targets"][:, step],
                                volume_rms,
                            )
                            weighted_loss = float(weight) * loss
                        if not torch.isfinite(loss):
                            raise RuntimeError("four-step training loss is non-finite")
                        per_step_losses[step].append(float(loss.detach().cpu()))
                        sample_total += float(weighted_loss.detach().cpu())
                        (weighted_loss / len(group)).backward()
                        current = forecast.volume.detach().unsqueeze(1)
                    raw_losses.append(sample_total)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(optimization["gradient_clip_norm"]),
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("four-step gradient norm is non-finite")
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
                            "train/recent_weighted_state_loss": float(
                                np.mean(raw_losses[-len(group) :])
                            ),
                            "train/preclip_gradient_norm": gradient_norms[-1],
                        },
                        step=optimizer_updates,
                    )

            validation = evaluate_state_rollouts(
                model,
                selection,
                horizons=SELECTION_HORIZONS,
                device=device,
                indices=selection_subset,
            )
            selection_metric = float(validation["selection_metric"])
            checkpoint = args.output / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                manifest=manifest,
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
                "training_window_count": len(raw_losses),
                "training_weighted_state_loss_mean": float(np.mean(raw_losses)),
                "training_weighted_state_loss_first_quarter": float(
                    np.mean(raw_losses[: max(1, len(raw_losses) // 4)])
                ),
                "training_weighted_state_loss_last_quarter": float(
                    np.mean(raw_losses[-max(1, len(raw_losses) // 4) :])
                ),
                "training_step_loss_means": {
                    str(step + 1): float(np.mean(values))
                    for step, values in enumerate(per_step_losses)
                },
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
                "train/epoch_weighted_state_loss": epoch_record[
                    "training_weighted_state_loss_mean"
                ],
                "validation/selection_metric": selection_metric,
            }
            for horizon in SELECTION_HORIZONS:
                metrics[f"validation/horizon_{horizon}_mean_field_skill"] = validation[
                    "horizons"
                ][str(horizon)]["mean_field_persistence_relative_skill"]
            run.log(metrics, step=optimizer_updates)

        if best_checkpoint is None:
            raise AssertionError("no four-step checkpoint was selected")
        best_record = min(history, key=lambda record: record["validation"]["selection_metric"])
        payload = torch.load(best_checkpoint, map_location=device, weights_only=True)
        selected_model, selected_config = build_model(manifest["architecture"])
        selected_model = selected_model.to(device)
        selected_model.load_state_dict(payload["model"], strict=True)
        numerical_gate = reload_and_equivariance_gate(
            checkpoint=best_checkpoint,
            config=selected_config,
            validation=lead1_validation,
            device=device,
        )
        candidate_full = {
            str(horizon): evaluate_state_rollouts(
                selected_model,
                dataset,
                horizons=(horizon,),
                device=device,
            )["horizons"][str(horizon)]
            for horizon, dataset in full_validations.items()
        }
        candidate_full_record = {
            "horizons": candidate_full,
            "future_truth_used_as_context": False,
            "physics_derived_metric": False,
        }
        exact_updates = optimizer_updates == total_updates
        loss_decreased = history[-1]["training_weighted_state_loss_mean"] < history[0][
            "training_weighted_state_loss_mean"
        ]
        finite_metrics = all(
            math.isfinite(float(record["validation"]["selection_metric"]))
            for record in history
        )
        mechanical_gate = {
            "exact_optimizer_update_count": exact_updates,
            "epoch_mean_training_loss_decreased": loss_decreased,
            "finite_validation_metrics": finite_metrics,
            "checkpoint_reload_and_equivariance_passed": numerical_gate["passed"],
        }
        mechanical_passed = all(mechanical_gate.values())
        state_decision = state_gate(
            parent=parent_full_record,
            candidate=candidate_full_record,
            thresholds=manifest["state_gates"],
        )
        state_pilot_passed = bool(mechanical_passed and state_decision["passed"])
        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": "completed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": True,
            "physics_derived_loss_used": False,
            "physics_diagnostics_scored": False,
            "family": FAMILY,
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "parent": parent_record,
            "architecture": selected_model.to_record(),
            "loss": manifest["loss"],
            "training_window_count": len(train),
            "selection_window_count": len(selection_subset),
            "full_validation_window_count_by_horizon": {
                str(horizon): len(dataset)
                for horizon, dataset in full_validations.items()
            },
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
            "best_selection_validation": best_record["validation"],
            "full_validation": candidate_full_record,
            "numerical_gate": numerical_gate,
            "mechanical_gate": {**mechanical_gate, "passed": mechanical_passed},
            "state_gate": state_decision,
            "state_pilot_passed": state_pilot_passed,
            "physics_evaluation_authorized": mechanical_passed,
            "confirmation_seed_training_authorized": False,
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
                "final/best_selection_metric": best_metric,
                "final/parent_selection_metric": parent_selection["selection_metric"],
                "final/mechanical_gate_passed": mechanical_passed,
                "final/state_pilot_passed": state_pilot_passed,
                "final/physics_evaluation_authorized": mechanical_passed,
                "compute/peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
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
        selection.close()
        for dataset in full_validations.values():
            dataset.close()
        lead1_validation.close()

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
