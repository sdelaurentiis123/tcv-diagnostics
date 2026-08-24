#!/usr/bin/env python3
"""Bounded non-scientific smoke for both codec-free state views on 85604."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
    StateDerivativePrediction,
    normalized_error_metrics,
    state_derivative_loss,
)
from tcv_diagnostics.state_operator_data import LeadTimeStateDataset
from tcv_diagnostics.wandb_tracking import WandbRunSpec


FAMILIES = ("c5p", "e6b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    partial.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def repository_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def batch(item: Mapping[str, Any], *, device: torch.device) -> dict[str, Tensor]:
    result = {
        "context": torch.from_numpy(item["context"]).unsqueeze(0).to(device),
        "target_derivative": torch.from_numpy(item["target_derivative"])
        .unsqueeze(0)
        .to(device),
        "lead_steps": torch.as_tensor(
            [item["lead_steps"]], dtype=torch.float32, device=device
        ),
    }
    if "context_boundary" in item:
        result["context_boundary"] = (
            torch.from_numpy(item["context_boundary"]).unsqueeze(0).to(device)
        )
        result["target_boundary_derivative"] = (
            torch.from_numpy(item["target_boundary_derivative"])
            .unsqueeze(0)
            .to(device)
        )
    return result


def reload_exact(
    *,
    model: CodecFreeIncrementOperator3D,
    config: CodecFreeOperatorConfig,
    checkpoint: Path,
    device: torch.device,
) -> bool:
    reloaded = CodecFreeIncrementOperator3D(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(payload["model"])
    source = model.state_dict()
    target = reloaded.state_dict()
    return source.keys() == target.keys() and all(
        torch.equal(source[name], target[name]) for name in source
    )


def run_family(
    *,
    family: str,
    catalog: Any,
    output: Path,
    seed: int,
    device: torch.device,
    wandb_run: Any,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    if family not in FAMILIES:
        raise ValueError(f"unsupported smoke family {family!r}")
    train = LeadTimeStateDataset(
        catalog,
        family=family,
        split="train",
        lead_steps=(1,),
        history_frames=1,
        augment=True,
        seed=seed,
        current_interval=(2, 4),
    )
    validation = LeadTimeStateDataset(
        catalog,
        family=family,
        split="validation",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=seed,
        current_interval=(496, 497),
    )
    config = CodecFreeOperatorConfig(
        state_family=family,
        history_frames=1,
        base_channels=8,
        channel_multipliers=(1, 2),
        blocks_per_level=1,
        lead_embedding_channels=16,
        group_norm_maximum_groups=4,
        predict_boundary=family == "e6b",
    )
    model = CodecFreeIncrementOperator3D(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    losses: list[float] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    model.train()
    train.set_epoch(0)
    for index in range(len(train)):
        values = batch(train[index], device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = model(
                values["context"],
                values["lead_steps"],
                values.get("context_boundary"),
            )
            loss = state_derivative_loss(
                prediction,
                values["target_derivative"],
                values.get("target_boundary_derivative"),
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"{family} smoke loss is non-finite")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"{family} smoke gradient norm is non-finite")
        optimizer.step()
        global_step += 1
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(norm.detach().cpu()))
        wandb_run.log(
            {
                "optimizer/global_step": global_step,
                f"{family}/train/state_derivative_mse": losses[-1],
                f"{family}/train/preclip_gradient_norm": gradient_norms[-1],
            },
            commit=True,
        )

    model.eval()
    values = batch(validation[0], device=device)
    with torch.inference_mode():
        prediction = model(
            values["context"],
            values["lead_steps"],
            values.get("context_boundary"),
        )
        validation_loss = state_derivative_loss(
            prediction,
            values["target_derivative"],
            values.get("target_boundary_derivative"),
        )
        shift = 7
        shifted = model(
            torch.roll(values["context"], shift, dims=-1),
            values["lead_steps"],
            values.get("context_boundary"),
        )
    volume_equivariance = normalized_error_metrics(
        shifted.volume, torch.roll(prediction.volume, shift, dims=-1)
    )
    boundary_equivariance = (
        {
            "normalized_maximum_absolute_error": 0.0,
            "normalized_root_mean_square_error": 0.0,
        }
        if prediction.boundary is None
        else normalized_error_metrics(shifted.boundary, prediction.boundary)
    )
    finite = bool(
        torch.isfinite(prediction.volume).all()
        and (
            prediction.boundary is None
            or torch.isfinite(prediction.boundary).all()
        )
        and torch.isfinite(validation_loss)
    )

    checkpoint = output / f"{family}_checkpoint.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_record(),
            "family": family,
            "seed": seed,
            "optimizer_steps": len(train),
        },
        checkpoint,
    )
    reload_bitwise_exact = reload_exact(
        model=model,
        config=config,
        checkpoint=checkpoint,
        device=device,
    )
    strides = [
        tuple(module.stride)
        for module in model.modules()
        if isinstance(module, torch.nn.Conv3d)
    ]
    no_toroidal_stride = bool(strides) and all(stride[-1] == 1 for stride in strides)
    all_mechanical_gates_passed = bool(
        finite
        and reload_bitwise_exact
        and no_toroidal_stride
        and volume_equivariance["normalized_maximum_absolute_error"] <= 1.0e-3
        and volume_equivariance["normalized_root_mean_square_error"] <= 1.0e-4
        and boundary_equivariance["normalized_maximum_absolute_error"] <= 1.0e-4
        and boundary_equivariance["normalized_root_mean_square_error"] <= 1.0e-5
        and len(losses) == 2
    )
    record = {
        "family": family,
        "architecture": model.to_record(),
        "train_pairs": [
            {"current": pair.current, "target": pair.target, "lead": pair.lead}
            for pair in train.pairs
        ],
        "validation_pair": {
            "current": validation.pairs[0].current,
            "target": validation.pairs[0].target,
            "lead": validation.pairs[0].lead,
        },
        "optimizer_steps": len(train),
        "train_losses": losses,
        "preclip_gradient_norms": gradient_norms,
        "validation_state_derivative_mse": float(validation_loss.cpu()),
        "finite": finite,
        "integer_toroidal_shift": shift,
        "volume_equivariance": volume_equivariance,
        "boundary_equivariance": boundary_equivariance,
        "equivariance_gate": {
            "volume_normalized_maximum_absolute_error_max": 1.0e-3,
            "volume_normalized_root_mean_square_error_max": 1.0e-4,
            "boundary_normalized_maximum_absolute_error_max": 1.0e-4,
            "boundary_normalized_root_mean_square_error_max": 1.0e-5,
            "amendment": "paper0/protocol/POST_ECRD_OPERATOR_SMOKE_NUMERICAL_AMENDMENT_2026-08-24.md"
        },
        "no_toroidal_stride": no_toroidal_stride,
        "checkpoint_reload_bitwise_exact": reload_bitwise_exact,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_path(checkpoint),
        },
        "wall_seconds": time.perf_counter() - started,
        "all_mechanical_gates_passed": all_mechanical_gates_passed,
        "scientific_result": False,
    }
    wandb_run.log(
        {
            f"{family}/smoke/validation_state_derivative_mse": record[
                "validation_state_derivative_mse"
            ],
            f"{family}/smoke/volume_equivariance_max_error": volume_equivariance[
                "normalized_maximum_absolute_error"
            ],
            f"{family}/smoke/volume_equivariance_rms_error": volume_equivariance[
                "normalized_root_mean_square_error"
            ],
            f"{family}/smoke/boundary_equivariance_max_error": boundary_equivariance[
                "normalized_maximum_absolute_error"
            ],
            f"{family}/smoke/boundary_equivariance_rms_error": boundary_equivariance[
                "normalized_root_mean_square_error"
            ],
            f"{family}/smoke/all_mechanical_gates_passed": int(
                all_mechanical_gates_passed
            ),
        },
        commit=True,
    )
    train.close()
    validation.close()
    return record, global_step


def verify_finished_wandb_run(
    *, module: Any, remote_path: str, expected_id: str
) -> str:
    last_state = "unknown"
    for delay in (0.0, 2.0, 4.0, 8.0, 16.0):
        if delay:
            time.sleep(delay)
        remote = module.Api(timeout=30).run(remote_path)
        if str(remote.id) != expected_id:
            raise RuntimeError("remote W&B run ID differs")
        last_state = str(remote.state)
        if last_state == "finished":
            return last_state
    raise RuntimeError(f"remote W&B run state is {last_state!r}, not finished")


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("state/data scaling manifest hash differs")
    manifest = load_strict_json(args.manifest)
    if manifest.get("development_run") != "85604":
        raise ValueError("smoke manifest development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("smoke manifest does not preserve held-out state")
    if manifest.get("engineering_smoke_authorized") is not True:
        raise ValueError("engineering smoke is not authorized")
    if manifest.get("training_authorized") is not False:
        raise ValueError("scientific training must remain unauthorized")
    actual_commit = repository_commit(args.paper0_root)
    if actual_commit != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from the launch lock")

    if not torch.cuda.is_available():
        raise RuntimeError("the codec-free smoke requires one allocated CUDA GPU")
    device = torch.device("cuda")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(device)

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("online W&B is required for the cluster smoke") from error
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="codec-free-operator-smoke",
        tags=("paper0", "85604", "codec-free", "exact-state", "smoke"),
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
            "scope": "bounded_non_scientific_85604_engineering_smoke",
            "paper0_commit": args.paper0_commit,
            "seed": args.seed,
            "families": list(FAMILIES),
            "lead_steps": [1],
            "optimizer_steps_per_family": 2,
            "physics_derived_loss_used": False,
            "held_out_85606_read": False,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    started = time.perf_counter()
    catalog = load_official_catalog(args.artifact_root)
    records: dict[str, Any] = {}
    global_step = 0
    try:
        for family in FAMILIES:
            records[family], global_step = run_family(
                family=family,
                catalog=catalog,
                output=args.output,
                seed=args.seed,
                device=device,
                wandb_run=run,
                global_step=global_step,
            )
        result = {
            "schema_version": 1,
            "scope": "bounded_non_scientific_codec_free_operator_smoke",
            "status": (
                "passed"
                if all(
                    record["all_mechanical_gates_passed"]
                    for record in records.values()
                )
                else "failed"
            ),
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "scientific_training_performed": False,
            "scientific_forecast_generated": False,
            "physics_derived_loss_used": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "seed": args.seed,
            "families": records,
            "optimizer_steps": global_step,
            "peak_cuda_memory_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "wall_seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        if result["status"] != "passed":
            raise RuntimeError("one or more codec-free smoke gates failed")
        run.summary.update(
            {
                "final/status": result["status"],
                "final/optimizer_steps": global_step,
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
    atomic_json(args.output / "result.json", result)
    atomic_json(args.output / "wandb.json", tracking)
    artifact_index = args.output / "artifact_sha256.txt"
    lines = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path != artifact_index:
            lines.append(f"{sha256_path(path)}  {path}\n")
    artifact_index.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
