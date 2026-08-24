#!/usr/bin/env python3
"""Bounded GPU engineering smoke for the old-85604 axial state operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from paper0.tools.smoke_codec_free_operator import verify_finished_wandb_run
from paper0.tools.train_codec_free_stage1_pilot import atomic_json, repository_commit
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.models.axial_operator import (
    AxialIncrementOperator3D,
    AxialOperatorConfig,
)
from tcv_diagnostics.models.codec_free_operator import (
    component_balanced_state_derivative_loss,
    normalized_error_metrics,
)
from tcv_diagnostics.state_operator_data import LeadTimeStateDataset
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
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def authorize_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("development_run") != "85604":
        raise ValueError("axial smoke development run differs")
    if manifest.get("held_out_85606_read") is not False:
        raise ValueError("axial smoke held-out flag differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise ValueError("axial smoke held-out access must be prohibited")
    if manifest.get("guard_frames_read_allowed") is not False:
        raise ValueError("axial smoke guard reads must be prohibited")
    if manifest.get("engineering_smoke_authorized") is not True:
        raise ValueError("axial engineering smoke is not authorized")
    if manifest.get("scientific_training_authorized") is not False:
        raise ValueError("axial scientific training must remain unauthorized")
    if manifest.get("scientific_model_selection_authorized") is not False:
        raise ValueError("axial model selection must remain unauthorized")
    if manifest["state"].get("future_auxiliary_context_allowed") is not False:
        raise ValueError("future auxiliary context must be prohibited")
    data = manifest["data"]
    expected = {
        "training_frames": [0, 432],
        "guard_frames": [432, 496],
        "validation_frames": [496, 624],
        "training_current_interval": [2, 4],
        "validation_current_interval": [496, 497],
        "lead_steps": [1],
        "history_frames": 1,
        "training_pair_count": 2,
        "validation_pair_count": 1,
    }
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError("axial smoke data contract differs")


def tensor_batch(
    item: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    required = (
        "context",
        "context_boundary",
        "auxiliary_context",
        "target_derivative",
        "target_boundary_derivative",
    )
    if any(name not in item for name in required):
        raise ValueError("axial smoke item lacks required exact-state tensors")
    return {
        "context": torch.from_numpy(item["context"]).unsqueeze(0).to(device),
        "context_boundary": torch.from_numpy(item["context_boundary"])
        .unsqueeze(0)
        .to(device),
        "auxiliary_context": torch.from_numpy(item["auxiliary_context"])
        .unsqueeze(0)
        .to(device),
        "target_derivative": torch.from_numpy(item["target_derivative"])
        .unsqueeze(0)
        .to(device),
        "target_boundary_derivative": torch.from_numpy(
            item["target_boundary_derivative"]
        )
        .unsqueeze(0)
        .to(device),
        "lead_steps": torch.as_tensor(
            [item["lead_steps"]],
            dtype=torch.float32,
            device=device,
        ),
    }


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("axial smoke manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(manifest)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from launch lock")
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("axial smoke requires one allocated CUDA GPU")
    device = torch.device("cuda")
    seed = int(manifest["optimization"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)

    architecture = manifest["architecture"]
    config = AxialOperatorConfig(
        state_family="e6b",
        history_frames=int(manifest["data"]["history_frames"]),
        auxiliary_context_channels=1,
        static_context_channels=0,
        width=int(architecture["width"]),
        blocks=int(architecture["blocks"]),
        attention_heads=int(architecture["attention_heads"]),
        feedforward_expansion=int(architecture["feedforward_expansion"]),
        lead_embedding_channels=int(architecture["lead_embedding_channels"]),
        group_norm_maximum_groups=int(
            architecture["group_norm_maximum_groups"]
        ),
        kernel_size=int(architecture["kernel_size"]),
        predict_boundary=True,
        zero_initialize_output=bool(architecture["zero_initialize_output"]),
    )
    model = AxialIncrementOperator3D(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(manifest["optimization"]["learning_rate"]),
        weight_decay=float(manifest["optimization"]["weight_decay"]),
    )

    catalog = load_official_catalog(args.artifact_root)
    train = LeadTimeStateDataset(
        catalog,
        family="e6b",
        split="train",
        lead_steps=(1,),
        history_frames=1,
        augment=True,
        seed=seed,
        current_interval=(2, 4),
        auxiliary_context_fields=("phi",),
    )
    validation = LeadTimeStateDataset(
        catalog,
        family="e6b",
        split="validation",
        lead_steps=(1,),
        history_frames=1,
        augment=False,
        seed=seed,
        current_interval=(496, 497),
        auxiliary_context_fields=("phi",),
    )
    if len(train) != 2 or len(validation) != 1:
        raise ValueError("axial smoke pair count differs")

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
        job_type="old-85604-axial-operator-smoke",
        tags=("paper0", "85604", "exact-state", "axial", "smoke"),
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
            **config.to_record(),
            "scope": manifest["scope"],
            "seed": seed,
            "paper0_commit": args.paper0_commit,
            "held_out_85606_read": False,
            "physics_derived_loss_used": False,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    losses: list[float] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    try:
        train.set_epoch(0)
        model.train()
        for index in range(len(train)):
            values = tensor_batch(train[index], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = model(
                    values["context"],
                    values["lead_steps"],
                    values["context_boundary"],
                    values["auxiliary_context"],
                )
                loss, _ = component_balanced_state_derivative_loss(
                    prediction,
                    values["target_derivative"],
                    values["target_boundary_derivative"],
                )
            if not torch.isfinite(loss):
                raise RuntimeError("axial smoke loss is non-finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(manifest["optimization"]["gradient_clip_norm"]),
            )
            if not torch.isfinite(norm):
                raise RuntimeError("axial smoke gradient norm is non-finite")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(norm.detach().cpu()))
            run.log(
                {
                    "optimizer/step": index + 1,
                    "train/direct_state_derivative_loss": losses[-1],
                    "train/preclip_gradient_norm": gradient_norms[-1],
                },
                step=index + 1,
            )

        model.eval()
        values = tensor_batch(validation[0], device=device)
        shift = 7
        with torch.inference_mode():
            reference = model(
                values["context"],
                values["lead_steps"],
                values["context_boundary"],
                values["auxiliary_context"],
            )
            shifted = model(
                torch.roll(values["context"], shift, -1),
                values["lead_steps"],
                values["context_boundary"],
                torch.roll(values["auxiliary_context"], shift, -1),
            )
        volume_equivariance = normalized_error_metrics(
            shifted.volume,
            torch.roll(reference.volume, shift, -1),
        )
        if reference.boundary is None or shifted.boundary is None:
            raise AssertionError("axial E6B boundary prediction is absent")
        boundary_equivariance = normalized_error_metrics(
            shifted.boundary,
            reference.boundary,
        )

        checkpoint = args.output / "checkpoint.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_record(),
                "seed": seed,
                "optimizer_steps": len(losses),
                "paper0_commit": args.paper0_commit,
            },
            checkpoint,
        )
        reloaded = AxialIncrementOperator3D(config).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        reloaded.load_state_dict(payload["model"])
        reload_exact = all(
            torch.equal(value, reloaded.state_dict()[name])
            for name, value in model.state_dict().items()
        )
        strides = [
            tuple(module.stride)
            for module in model.modules()
            if isinstance(module, torch.nn.Conv3d)
        ]
        no_toroidal_stride = bool(strides) and all(
            stride[-1] == 1 for stride in strides
        )
        finite = bool(
            torch.isfinite(reference.volume).all()
            and torch.isfinite(reference.boundary).all()
        )
        gates = {
            "exact_optimizer_steps": len(losses) == 2,
            "finite_predictions_losses_and_gradients": bool(
                finite
                and np.isfinite(losses).all()
                and np.isfinite(gradient_norms).all()
            ),
            "checkpoint_reload_exact": reload_exact,
            "no_toroidal_stride": no_toroidal_stride,
            "volume_toroidal_equivariance": bool(
                volume_equivariance["normalized_maximum_absolute_error"] <= 1e-4
                and volume_equivariance["normalized_root_mean_square_error"]
                <= 1e-5
            ),
            "boundary_toroidal_invariance": bool(
                boundary_equivariance["normalized_maximum_absolute_error"] <= 1e-5
                and boundary_equivariance["normalized_root_mean_square_error"]
                <= 1e-6
            ),
        }
        result = {
            "schema_version": 1,
            "scope": manifest["scope"],
            "status": "passed" if all(gates.values()) else "failed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "scientific_result": False,
            "scientific_training_authorized": False,
            "physics_derived_loss_used": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "seed": seed,
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
            "auxiliary_context_fields": ["phi"],
            "future_auxiliary_context_read": False,
            "optimizer_steps": len(losses),
            "training_losses": losses,
            "preclip_gradient_norms": gradient_norms,
            "integer_toroidal_shift": shift,
            "volume_equivariance": volume_equivariance,
            "boundary_equivariance": boundary_equivariance,
            "mechanical_gates": gates,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_path(checkpoint),
            },
            "peak_cuda_memory_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "wall_seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(device),
        }
        if result["status"] != "passed":
            raise RuntimeError("one or more axial smoke gates failed")
        run.summary.update(
            {
                "final/status": result["status"],
                "final/optimizer_steps": len(losses),
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
    }
    atomic_json(args.output / "wandb.json", tracking)
    result["wandb"] = tracking
    atomic_json(args.output / "result.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
