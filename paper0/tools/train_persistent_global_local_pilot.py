#!/usr/bin/env python3
"""Run the old-85604 persistent global--local smoke or one-seed pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    repository_commit,
    verify_finished_wandb_run,
)
from paper0.tools.train_codec_free_stage2_multilead import (
    build_model,
    validate_parent_config,
)
from tcv_diagnostics.autoregressive_training import AutoregressiveStateWindowDataset
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import FAMILY_FIELDS, load_official_catalog
from tcv_diagnostics.models.codec_free_operator import CodecFreeIncrementOperator3D
from tcv_diagnostics.models.persistent_global_local import (
    PersistentGlobalLocalConfig,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
)
from tcv_diagnostics.persistent_global_local_training import (
    PGL_SEED,
    PersistentPilotTrainingConfig,
    fit_parent_residual_scales,
    train_persistent_global_local,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SCOPE_BY_MODE = {
    "smoke": "post_ecrd_old_85604_persistent_global_local_smoke",
    "pilot": "post_ecrd_old_85604_persistent_global_local_pilot",
}
FIELDS = FAMILY_FIELDS["c5p"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--seed", type=int, default=PGL_SEED)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def _locked_json(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if len(digest) != 64 or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return load_strict_json(path)


def exact_model_config() -> PersistentGlobalLocalConfig:
    return PersistentGlobalLocalConfig(
        horizon=4,
        fields=5,
        base_channels=16,
        channel_multipliers=(1, 2, 4),
        residual_blocks_per_resolution=1,
        global_channels=24,
        global_pool_xy=(4, 4),
        low_mode_maximum=7,
        noise_embedding_features=128,
        group_norm_maximum_groups=8,
        kernel_size=3,
    )


def exact_noise_config() -> PersistentNoiseConfig:
    return PersistentNoiseConfig(
        global_weight=1.0,
        local_weight=1.0,
        global_pool_xy=(4, 4),
        low_mode_maximum=7,
    )


def authorize_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    artifact_root: Path,
) -> None:
    flags = {
        "scope": SCOPE_BY_MODE[mode],
        "mode": mode,
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_read": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "physics_derived_loss_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
        "steering_allowed": False,
        "wandb_required": True,
        "engineering_smoke_authorized": mode == "smoke",
        "pilot_training_authorized": mode == "pilot",
        "confirmation_seed_training_authorized": False,
        "state_family": "c5p",
    }
    if any(manifest.get(name) != value for name, value in flags.items()):
        raise ValueError("persistent global-local authorization differs")
    if tuple(manifest.get("fields", ())) != FIELDS or int(seed) != PGL_SEED:
        raise ValueError("persistent global-local field or seed identity differs")
    if Path(str(manifest.get("artifact_root", ""))) != artifact_root:
        raise ValueError("persistent global-local artifact root differs")

    split = manifest.get("split", {})
    expected_split = {
        "training_frames": [0, 432],
        "guard_frames": [432, 496],
        "validation_frames": [496, 624],
        "horizon_frames": 4,
        "full_training_window_count": 428,
        "full_validation_window_count": 124,
        "smoke_training_current_interval": [0, 8],
        "smoke_validation_current_interval": [496, 500],
        "validation_current_blocks": {
            "V00": [496, 537],
            "V01": [537, 578],
            "V02": [578, 620],
        },
        "cadence_microseconds": 3.131905426352636,
        "periodic_axis": "stored_toroidal_z",
        "zperiod": 5,
        "physical_mode_mapping": "n=5k",
    }
    if split != expected_split:
        raise ValueError("persistent global-local split differs")

    config = exact_model_config()
    noise = exact_noise_config()
    architecture = manifest.get("architecture", {})
    expected_architecture = {
        "name": "persistent_global_local_joint_residual_EDM",
        "future_frames": 4,
        "fields": 5,
        "base_channels": config.base_channels,
        "channel_multipliers": list(config.channel_multipliers),
        "residual_blocks_per_resolution": config.residual_blocks_per_resolution,
        "global_channels": config.global_channels,
        "global_pool_xy": list(config.global_pool_xy),
        "low_mode_maximum": config.low_mode_maximum,
        "noise_embedding_features": config.noise_embedding_features,
        "group_norm_maximum_groups": config.group_norm_maximum_groups,
        "kernel_size": config.kernel_size,
        "toroidal_downsampling": False,
        "global_recurrence": "ConvGRU_over_four_future_frames",
        "global_noise_shared_across_future_frames": True,
        "local_noise_independent_across_future_frames": True,
        "stochastic_parameter_count": 774234,
        "mean_parameter_count": 2174021,
        "total_parameter_count": 2948255,
    }
    if architecture != expected_architecture:
        raise ValueError("persistent global-local architecture differs")
    expected_noise = {
        "global_weight": noise.global_weight,
        "local_weight": noise.local_weight,
        "global_pool_xy": list(noise.global_pool_xy),
        "low_mode_maximum": noise.low_mode_maximum,
        "global_and_local_equal_rms": True,
        "posthoc_spread_multiplier": False,
    }
    if manifest.get("noise") != expected_noise:
        raise ValueError("persistent global-local noise law differs")

    training_config = PersistentPilotTrainingConfig(mode=mode, seed=seed)
    training = manifest.get("training", {})
    expected_training = {
        "seed": seed,
        "epochs": training_config.epochs,
        "training_windows": training_config.expected_training_windows,
        "validation_windows": training_config.expected_validation_windows,
        "gradient_accumulation_windows": training_config.accumulation_windows,
        "expected_optimizer_updates": training_config.total_optimizer_steps,
        "optimizer": "AdamW",
        "stochastic_peak_learning_rate": training_config.stochastic_peak_learning_rate,
        "mean_peak_learning_rate": training_config.mean_peak_learning_rate,
        "stochastic_minimum_learning_rate": training_config.stochastic_minimum_learning_rate,
        "mean_minimum_learning_rate": training_config.mean_minimum_learning_rate,
        "betas": list(training_config.betas),
        "weight_decay": training_config.weight_decay,
        "warmup_fraction": training_config.warmup_fraction,
        "gradient_clip_norm": training_config.gradient_clip,
        "ema_decay": training_config.ema_decay,
        "validation_probes": training_config.validation_probes,
        "autocast": "bfloat16",
        "cudnn_tf32": False,
        "matmul_tf32": False,
        "mean_step_weights": [0.625, 0.125, 0.125, 0.125],
        "mean_feedback_gradient": "detached_between_steps",
        "diffusion_gradient_to_mean": False,
        "physics_derived_loss_used": False,
    }
    if training != expected_training:
        raise ValueError("persistent global-local training budget differs")


def load_parent(
    manifest: Mapping[str, Any], *, device: torch.device
) -> tuple[CodecFreeIncrementOperator3D, dict[str, Any], torch.Tensor]:
    parent = manifest.get("parent", {})
    if (
        parent.get("kind") != "selected_seed1702_four_step_detached_feedback_mean"
        or int(parent.get("seed", -1)) != PGL_SEED
        or int(parent.get("selected_epoch", -1)) != 6
        or int(parent.get("parameter_count", -1)) != 2174021
    ):
        raise ValueError("persistent parent identity differs")
    source_manifest = _locked_json(parent.get("source_manifest", {}), label="parent manifest")
    result = _locked_json(parent.get("result", {}), label="parent result")
    required_result = {
        "scope": "post_ecrd_old_85604_four_step_feedback_pilot",
        "status": "completed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": "c5p",
        "seed": PGL_SEED,
    }
    if any(result.get(name) != value for name, value in required_result.items()):
        raise ValueError("persistent parent result contract differs")
    checkpoint_record = parent.get("checkpoint", {})
    checkpoint = Path(str(checkpoint_record.get("path", "")))
    digest = str(checkpoint_record.get("sha256", ""))
    assert_development_path(checkpoint)
    if (
        result.get("best_checkpoint", {}).get("path") != str(checkpoint)
        or result.get("best_checkpoint", {}).get("sha256") != digest
        or result.get("best_checkpoint", {}).get("epoch") != 6
        or sha256_path(checkpoint) != digest
    ):
        raise ValueError("persistent parent checkpoint record differs")
    model, config = build_model(source_manifest["architecture"])
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        payload.get("family") != "c5p"
        or payload.get("stage") != "four_step_detached_feedback_finetune"
        or int(payload.get("seed", -1)) != PGL_SEED
        or int(payload.get("epoch", -1)) != 6
    ):
        raise ValueError("persistent parent checkpoint identity differs")
    config_validation = validate_parent_config(payload.get("config", {}), config.to_record())
    if payload.get("derivative_rms") != source_manifest.get("derivative_rms"):
        raise ValueError("persistent parent derivative RMS differs")
    model = model.to(device, torch.float32)
    model.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(value.to(device), model.state_dict()[name])
        for name, value in payload["model"].items()
    )
    if not bitwise:
        raise AssertionError("persistent parent did not load bitwise")
    rms_record = source_manifest["derivative_rms"]
    derivative_rms = torch.tensor(
        [float(rms_record["volume"][field]) for field in FIELDS],
        device=device,
        dtype=torch.float32,
    )
    if not torch.all(torch.isfinite(derivative_rms) & (derivative_rms > 0.0)):
        raise ValueError("persistent parent derivative RMS values differ")
    return model, {
        "result": dict(parent["result"]),
        "checkpoint": dict(checkpoint_record),
        "source_manifest": dict(parent["source_manifest"]),
        "checkpoint_reload_bitwise": True,
        "config_validation": config_validation,
        "checkpoint_paper0_commit": payload.get("paper0_commit"),
    }, derivative_rms


def _load_or_fit_scales(
    *,
    mode: str,
    manifest: Mapping[str, Any],
    parent: CodecFreeIncrementOperator3D,
    catalog: Any,
    output: Path,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    record = manifest.get("residual_scales", {})
    if mode == "smoke":
        expected = {
            "action": "fit_once_in_smoke_from_all_428_training_windows",
            "fit_parent": "bitwise_selected_seed1702_feedback_mean",
            "fit_split": "training_only",
            "shape": [4, 5],
            "statistic": "RMS_parent_state_residual_by_future_step_and_field",
            "physics_derived_quantity": False,
        }
        if record != expected:
            raise ValueError("persistent smoke residual-scale action differs")
        dataset = AutoregressiveStateWindowDataset(
            catalog,
            family="c5p",
            split="train",
            horizon=4,
            augment=False,
            seed=PGL_SEED,
        )
        try:
            scales, scale_record = fit_parent_residual_scales(
                parent_mean=parent,
                dataset=dataset,
                device=device,
            )
        finally:
            dataset.close()
        scale_path = output / "residual_scales.json"
        atomic_json(scale_path, scale_record)
        scale_record["artifact"] = {
            "path": str(scale_path),
            "sha256": sha256_path(scale_path),
        }
        return scales.to(device), scale_record

    scale_record = _locked_json(record.get("artifact", {}), label="persistent scales")
    required = {
        "scope": "old_85604_persistent_global_local_parent_residual_scales",
        "development_run": "85604",
        "training_frames": [0, 432],
        "training_window_count": 428,
        "horizon": 4,
        "fields": list(FIELDS),
        "fit_split": "train_only",
        "physics_derived_quantity": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    if any(scale_record.get(name) != value for name, value in required.items()):
        raise ValueError("persistent pilot scale artifact contract differs")
    values = torch.tensor(scale_record["values"], device=device, dtype=torch.float32)
    if values.shape != (4, 5) or not torch.all(torch.isfinite(values) & (values > 0.0)):
        raise ValueError("persistent pilot scale values differ")
    return values, {**scale_record, "artifact": dict(record["artifact"])}


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("persistent manifest SHA-256 differs")
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("persistent Paper 0 commit differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(
        manifest,
        mode=args.mode,
        seed=args.seed,
        artifact_root=args.artifact_root,
    )
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("persistent pilot requires allocated CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parent, parent_record, derivative_rms = load_parent(manifest, device=device)
    catalog = load_official_catalog(args.artifact_root)

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
        job_type=f"old-85604-persistent-global-local-{args.mode}",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "persistent-global-local",
            "joint-four-frame",
            "residual-diffusion",
            args.mode,
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key:
        raise RuntimeError("W&B API key is absent")
    if str(getattr(api.viewer, "entity", "")) != spec.entity:
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
            "scope": SCOPE_BY_MODE[args.mode],
            "mode": args.mode,
            "seed": args.seed,
            "paper0_commit": args.paper0_commit,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "architecture": manifest["architecture"],
            "noise": manifest["noise"],
            "training": manifest["training"],
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

    train = None
    validation = None
    try:
        scales, scales_record = _load_or_fit_scales(
            mode=args.mode,
            manifest=manifest,
            parent=parent,
            catalog=catalog,
            output=args.output,
            device=device,
        )
        run.log(
            {
                "preflight/residual_scale_minimum": float(scales.min().cpu()),
                "preflight/residual_scale_maximum": float(scales.max().cpu()),
                "preflight/residual_scale_fit_windows": 428,
            },
            step=0,
        )
        noise_config = exact_noise_config()
        edm = PersistentGlobalLocalEDM(
            exact_model_config(),
            residual_scales=scales,
            noise_config=noise_config,
        )
        stochastic_parameters = sum(parameter.numel() for parameter in edm.parameters())
        total_parameters = stochastic_parameters + sum(
            parameter.numel() for parameter in parent.parameters()
        )
        if (
            stochastic_parameters != manifest["architecture"]["stochastic_parameter_count"]
            or total_parameters != manifest["architecture"]["total_parameter_count"]
        ):
            raise ValueError("persistent model parameter count differs")

        train = AutoregressiveStateWindowDataset(
            catalog,
            family="c5p",
            split="train",
            horizon=4,
            augment=True,
            seed=args.seed,
            current_interval=(0, 8) if args.mode == "smoke" else None,
        )
        validation = AutoregressiveStateWindowDataset(
            catalog,
            family="c5p",
            split="validation",
            horizon=4,
            augment=False,
            seed=args.seed,
            current_interval=(496, 500) if args.mode == "smoke" else None,
        )
        training_config = PersistentPilotTrainingConfig(mode=args.mode, seed=args.seed)

        def on_epoch(record: Mapping[str, Any]) -> None:
            metrics = {
                "epoch": int(record["completed_epoch"]),
                "optimizer/update": int(record["optimizer_updates"]),
                "optimizer/stochastic_learning_rate": float(
                    record["stochastic_learning_rate"]
                ),
                "optimizer/mean_learning_rate": float(record["mean_learning_rate"]),
                "train/objective": float(record["train_mean_objective"]),
                "train/mean_state_loss": float(record["train_mean_state_loss"]),
                "train/edm_loss": float(record["train_mean_edm_loss"]),
                "train/preclip_gradient_norm": float(
                    record["mean_preclip_gradient_norm"]
                ),
                "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
            }
            if record["validation"] is not None:
                metrics["validation/checkpoint_score"] = float(
                    record["validation"]["checkpoint_score"]
                )
                for block, values in record["validation"]["blocks"].items():
                    metrics[f"validation/{block}/objective"] = float(values["objective"])
                    metrics[f"validation/{block}/mean_state_loss"] = float(
                        values["mean_state_loss"]
                    )
                    metrics[f"validation/{block}/edm_loss"] = float(values["edm_loss"])
            run.log(metrics, step=int(record["optimizer_updates"]))

        training_result = train_persistent_global_local(
            mean_model=parent,
            edm=edm,
            training_dataset=train,
            validation_dataset=validation,
            derivative_rms=derivative_rms,
            output=args.output / "training",
            device=device,
            paper0_commit=args.paper0_commit,
            slurm_job_id=args.slurm_job_id,
            manifest=manifest,
            config=training_config,
            on_epoch=on_epoch,
        )
        result = {
            **training_result,
            "scope": SCOPE_BY_MODE[args.mode],
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "parent": parent_record,
            "residual_scale_record": scales_record,
            "residual_scale_artifact": (
                scales_record.get("artifact") if args.mode == "smoke" else manifest["residual_scales"]["artifact"]
            ),
        }
        atomic_json(args.output / "result.json", result)
        run.summary.update(
            {
                "final/status": result["status"],
                "final/mode": args.mode,
                "final/selected_epoch": result["selected_checkpoint"]["completed_epoch"],
                "final/selected_checkpoint_score": result["selected_checkpoint"][
                    "checkpoint_score"
                ],
                "final/mechanical_gate_passed": result["mechanical_gate"]["passed"],
                "final/state_gate_passed": result["state_gate"]["passed"],
                "final/physics_evaluation_authorized": result[
                    "physics_evaluation_authorized"
                ],
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
        if train is not None:
            train.close()
        if validation is not None:
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
        "wandb_version": wandb.__version__,
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_state_after_finish": remote_state,
        "checkpoints_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    atomic_json(args.output / "wandb.json", tracking)
    artifact_lines = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and "wandb" not in path.parts and path.name != "artifact_sha256.txt":
            artifact_lines.append(f"{sha256_path(path)}  {path}\n")
    (args.output / "artifact_sha256.txt").write_text("".join(artifact_lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "mode": args.mode,
        "selected_checkpoint": result["selected_checkpoint"],
        "mechanical_gate": result["mechanical_gate"],
        "state_gate": result["state_gate"],
        "physics_evaluation_authorized": result["physics_evaluation_authorized"],
        "peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
        "wandb": tracking,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
