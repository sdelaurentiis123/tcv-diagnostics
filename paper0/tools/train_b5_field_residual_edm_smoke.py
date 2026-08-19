#!/usr/bin/env python3
"""Run the one frozen, bounded B5 joint field-residual EDM smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.b5_residual_edm_training import (
    B5_EDM_SMOKE_TARGETS,
    B5EDMSmokeConfig,
    B5ResidualSmokeDataset,
    train_b5_edm_smoke,
)
from tcv_diagnostics.b5_residual_edm_wandb_tracking import (
    B5EDMOnlineWandbTracker,
)
from tcv_diagnostics.b5_residual_forecast import B5TrainingForecastArtifact
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.models.field_residual_edm import (
    B5_FIELD_ORDER,
    B5_RESIDUAL_SCALES,
    FieldResidualUNetConfig,
)
from tcv_diagnostics.o2_training_data import OneStepWindowDataset
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_SHA256 = (
    "2189c501071d245fbb87c0ddbb679f00fee73f8637e764b23d391421e8551283"
)
EXPECTED_PROTOCOL_SHA256 = (
    "e50a2a3f150f08f24c444f3b877fd887877cfe57a444c89d5c86b25054afde75"
)
EXPECTED_H1_FORECAST_SHA256 = (
    "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
)
EXPECTED_RESIDUAL_AUDIT_SHA256 = (
    "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke",), required=True)
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--h1-training-forecast", type=Path, required=True)
    parser.add_argument("--h1-training-forecast-sha256", required=True)
    parser.add_argument("--residual-audit", type=Path, required=True)
    parser.add_argument("--residual-audit-sha256", required=True)
    parser.add_argument("--b5-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit mismatch: {actual} != {expected_commit}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def require_rocky9_h100() -> dict[str, Any]:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    if (
        release.get("ID") != "rocky"
        or release.get("VERSION_ID", "").split(".")[0] != "9"
    ):
        raise RuntimeError("the B5 residual-EDM smoke requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the B5 residual-EDM smoke requires exactly one CUDA GPU")
    accelerator = torch.cuda.get_device_name(0)
    if "H100" not in accelerator:
        raise RuntimeError(
            f"the frozen B5 residual-EDM smoke requires H100, found {accelerator!r}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated H100 does not report bfloat16 support")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "accelerator": accelerator,
        "cuda_device_count": torch.cuda.device_count(),
        "bfloat16_supported": True,
    }


def authorize_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    mode: str,
    seed: int,
    forecast_sha256: str,
    residual_audit_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless every bounded B5 smoke authority remains exact."""

    if mode != "smoke" or int(seed) != 1701:
        raise RuntimeError("only the bounded seed-1701 B5 smoke is authorized")
    if manifest.get("status") != (
        "frozen_after_job_6901393_before_B5_model_implementation_or_optimization"
    ):
        raise RuntimeError("B5 smoke protocol status differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("B5 development run differs")
    if manifest.get("sequestered_run") != "85606":
        raise RuntimeError("B5 sequestered run identity differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("B5 manifest unexpectedly permits held-out access")
    expected_scope = {
        "B5_joint_field_residual_EDM_implementation",
        "B5_joint_field_residual_EDM_CPU_tests",
        "one_Rocky9_H100_bounded_full_field_mechanical_smoke",
    }
    if set(manifest.get("authorized_scope", ())) != expected_scope:
        raise RuntimeError("B5 authorized scope differs")
    required_forbidden = {
        "B5_full_training",
        "B5_validation_read_or_scoring",
        "scientific_checkpoint_selection",
        "O3",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    }
    if not required_forbidden.issubset(set(manifest.get("forbidden_scope", ()))):
        raise RuntimeError("B5 forbidden scope differs")

    protocol = manifest.get("protocol", {})
    if protocol.get("path") != (
        "paper0/protocol/PHASE3_B5_FIELD_RESIDUAL_EDM_SMOKE_PROTOCOL.md"
    ):
        raise RuntimeError("B5 protocol path differs")
    if protocol.get("status") != "frozen_before_implementation_or_execution":
        raise RuntimeError("B5 protocol freeze status differs")
    protocol_path = ROOT / str(protocol["path"])
    if sha256_path(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("B5 protocol bytes differ")

    evidence = manifest.get("evidence_locks", {})
    residual_lock = evidence.get("residual_audit", {})
    forecast_lock = evidence.get("H1_training_forecast", {})
    if residual_lock.get("job_id") != "6901393":
        raise RuntimeError("B5 residual-audit job differs")
    if (
        residual_lock.get("sha256") != residual_audit_sha256
        or residual_lock.get("sha256") != EXPECTED_RESIDUAL_AUDIT_SHA256
    ):
        raise RuntimeError("B5 residual-audit hash differs")
    if (
        forecast_lock.get("sha256") != forecast_sha256
        or forecast_lock.get("sha256") != EXPECTED_H1_FORECAST_SHA256
    ):
        raise RuntimeError("B5 H1 training-forecast hash differs")
    if forecast_lock.get("canonical_shape") != [430, 5, 64, 32, 88]:
        raise RuntimeError("B5 H1 training-forecast shape differs")
    if forecast_lock.get("forecast_closed_and_hashed_before_truth_read") is not True:
        raise RuntimeError("B5 H1 forecast truth-separation lock differs")
    parent = evidence.get("H1_checkpoint", {})
    if (
        parent.get("arm") != "C5P-H1"
        or parent.get("seed") != 1701
        or parent.get("sha256")
        != "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
        or parent.get("trainable") is not False
        or parent.get("reselection_allowed") is not False
    ):
        raise RuntimeError("B5 frozen H1 parent lock differs")

    data = manifest.get("data", {})
    if data.get("fields") != list(B5_FIELD_ORDER):
        raise RuntimeError("B5 field order differs")
    if data.get("input_channels") != (
        "physically_valid_complete_C5P_state_plus_frozen_H1_mean_condition"
    ):
        raise RuntimeError("B5 physical input state differs")
    if data.get("dynamic_condition") != [
        "x_t_minus_1_five_fields",
        "frozen_H1_mean_five_fields",
    ]:
        raise RuntimeError("B5 dynamic condition differs")
    if (
        data.get("smoke_target_frames") != [2, 10]
        or data.get("smoke_target_count") != 8
        or data.get("context_offset_frames") != -1
        or data.get("volume_shape") != [5, 64, 32, 88]
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
    ):
        raise RuntimeError("B5 data geometry or target budget differs")
    for flag in (
        "guard_frames_read_allowed",
        "validation_frames_read_allowed",
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "diagnostic_input_allowed",
        "region_mask_input_allowed",
        "future_truth_condition_allowed",
        "toroidal_augmentation_allowed",
    ):
        if data.get(flag) is not False:
            raise RuntimeError(f"B5 prohibited data flag {flag} differs")

    normalization = manifest.get("residual_normalization", {})
    if (
        normalization.get("operation") != "divide_without_centering"
        or normalization.get("field_order") != list(B5_FIELD_ORDER)
        or normalization.get("scale") != list(B5_RESIDUAL_SCALES)
        or normalization.get("nonzero_mean_preserved") is not True
        or normalization.get("pointwise_or_region_scaling_allowed") is not False
    ):
        raise RuntimeError("B5 residual-normalization contract differs")

    model = manifest.get("model", {})
    expected_model = FieldResidualUNetConfig().to_record()
    for key in (
        "name",
        "base_channels",
        "channel_multipliers",
        "noise_embedding_features",
        "kernel_size",
        "dropout",
    ):
        if model.get(key) != expected_model.get(key):
            raise RuntimeError(f"B5 model setting {key} differs")
    if (
        model.get("representation") != "full_decoded_standardized_field_coordinates"
        or model.get("joint_output_fields") != 5
        or model.get("dynamic_condition_channels") != 10
        or model.get("internal_position_channels") != 2
        or model.get("residual_blocks_per_encoder_resolution")
        != expected_model["residual_blocks_per_resolution"]
        or model.get("residual_blocks_per_decoder_resolution")
        != expected_model["residual_blocks_per_resolution"]
        or model.get("padding_by_axis") != ["zeros", "zeros", "circular"]
        or model.get("full_field_required") is not True
        or model.get("patch_fallback_allowed_in_this_protocol") is not False
        or model.get("DCAE_or_latent_representation_allowed") is not False
        or model.get("deterministic_parent_trainable") is not False
    ):
        raise RuntimeError("B5 model representation or boundary contract differs")

    edm = manifest.get("edm", {})
    if (
        edm.get("sigma_data") != 1.0
        or edm.get("training_sigma")
        != {"distribution": "log_normal", "P_mean": -1.2, "P_std": 1.2}
        or edm.get("training_noise") != "elementwise_standard_normal"
        or edm.get("loss") != "((sigma^2+1)/sigma^2)*mean((D_theta-z)^2)"
        or edm.get("equal_normalized_channel_weight") is not True
        or edm.get("physics_derived_training_loss_allowed") is not False
    ):
        raise RuntimeError("B5 EDM objective differs")

    config = B5EDMSmokeConfig(seed=int(seed))
    optimization = manifest.get("optimization", {})
    expected_optimization = {
        "seed": 1701,
        "optimizer": "AdamW",
        "optimizer_steps": config.optimizer_steps,
        "target_order": (
            "seed_67001_permutation_6_3_9_2_4_5_8_7_repeated_eight_times"
        ),
        "microbatch_targets": config.microbatch_targets,
        "gradient_accumulation_targets": config.gradient_accumulation_targets,
        "learning_rate": config.learning_rate,
        "betas": list(config.betas),
        "weight_decay": config.weight_decay,
        "gradient_clip": config.gradient_clip,
        "training_precision": config.training_precision,
        "training_order_seed": config.training_order_seed,
        "training_noise_seed": config.training_noise_seed,
        "sampler_seed": config.sampler_seed,
        "fixed_probe_seed": config.fixed_probe_seed,
        "fixed_probe_targets": [2, 6],
        "validation_used_for_selection": False,
        "scientific_checkpoint_selection": False,
    }
    if optimization != expected_optimization:
        raise RuntimeError("B5 bounded optimization contract differs")
    sampler = manifest.get("sampler_probe", {})
    if (
        sampler.get("algorithm") != "deterministic_EDM_probability_flow_ODE_Heun"
        or sampler.get("steps") != config.sampler_steps
        or sampler.get("sigma_max") != config.sampler_sigma_max
        or sampler.get("sigma_min") != config.sampler_sigma_min
        or sampler.get("rho") != config.sampler_rho
        or sampler.get("stochastic_churn") != 0.0
        or sampler.get("ensemble_members") != config.sampler_members
        or sampler.get("expected_shape") != [1, 2, 1, 5, 64, 32, 88]
        or sampler.get("scientific_calibration_result") is not False
    ):
        raise RuntimeError("B5 sampler-probe contract differs")

    post = manifest.get("post_smoke", {})
    for closed in (
        "B5_full_training_authorized",
        "validation_read_allowed",
        "O3_launch_allowed",
        "assimilation_allowed",
        "diagnostic_ranking_allowed",
        "held_out_85606_access_allowed",
    ):
        if post.get(closed) is not False:
            raise RuntimeError(f"B5 post-smoke closed scope {closed} differs")
    if sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("B5 manifest bytes differ from frozen authority")
    return {
        "authorized": True,
        "scope": "bounded_non_scientific_B5_joint_field_residual_EDM_smoke_85604",
        "development_run": "85604",
        "blind_test_read": False,
        "validation_read": False,
        "scientific_result": False,
        "full_training_authorized": False,
        "seed": int(seed),
        "evidence_hashes": {
            "H1_training_forecast": forecast_sha256,
            "residual_audit": residual_audit_sha256,
            "manifest": EXPECTED_MANIFEST_SHA256,
            "protocol": EXPECTED_PROTOCOL_SHA256,
        },
    }


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = sha256_path(source)
    if observed != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 differs: {observed} != {expected_sha256}")
    return source


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    environment = require_rocky9_h100()
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", 0)

    manifest_path = args.b5_manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("B5 manifest must be inside the Paper 0 repository") from error
    prohibited_paths = (
        args.artifact_root,
        args.h1_training_forecast,
        args.residual_audit,
        args.output,
        manifest_path,
    )
    if any("85606" in str(path).lower() for path in prohibited_paths):
        raise ValueError("held-out paths are prohibited during the B5 smoke")

    forecast_path = verify_input(
        args.h1_training_forecast,
        args.h1_training_forecast_sha256,
        "B5 H1 training forecast",
    )
    residual_audit_path = verify_input(
        args.residual_audit,
        args.residual_audit_sha256,
        "B5 residual audit",
    )
    manifest = load_strict_json(manifest_path)
    authorization = authorize_from_manifest(
        manifest,
        manifest_path=manifest_path,
        mode=args.mode,
        seed=args.seed,
        forecast_sha256=args.h1_training_forecast_sha256,
        residual_audit_sha256=args.residual_audit_sha256,
    )
    model_data_lock = manifest["evidence_locks"]["model_dataset"]
    verify_input(
        args.artifact_root / "model_dataset_manifest.json",
        model_data_lock["manifest_sha256"],
        "B5 model-data manifest",
    )
    verify_input(
        args.artifact_root / "normalization.json",
        model_data_lock["normalization_sha256"],
        "B5 model-data normalization",
    )
    verify_input(
        args.artifact_root / "artifact_sha256.txt",
        model_data_lock["artifact_index_sha256"],
        "B5 model-data artifact index",
    )
    residual_audit = load_strict_json(residual_audit_path)
    if (
        residual_audit.get("scope") != "B5_frozen_H1_training_residual_audit_85604"
        or residual_audit.get("canonical_shape") != [430, 5, 64, 32, 88]
        or residual_audit.get("scientific_boundaries", {}).get(
            "architecture_selected"
        )
        is not False
    ):
        raise RuntimeError("B5 residual-audit authority differs")

    catalog = load_official_catalog(args.artifact_root)
    config = B5EDMSmokeConfig(seed=args.seed)
    model_config = FieldResidualUNetConfig()
    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b5_joint_field_residual_edm_smoke",
        tags=(
            "paper0",
            "phase3",
            "b5",
            "joint-field-residual-edm",
            "smoke",
            "85604-only",
            "non-scientific",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_B5_joint_field_residual_EDM_smoke_85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "b5_manifest": {
            "path": str(manifest_relative),
            "sha256": sha256_path(manifest_path),
        },
        "inputs": {
            "model_data_root": str(args.artifact_root),
            "H1_training_forecast_sha256": sha256_path(forecast_path),
            "residual_audit_sha256": sha256_path(residual_audit_path),
            "development_run": "85604",
            "held_out_85606_read": False,
        },
        "training": config.to_record(),
        "model": model_config.to_record(),
        "environment": environment,
        "precision": {
            "training": config.training_precision,
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "model_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/field_residual_edm.py"
            ),
            "training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/b5_residual_edm_training.py"
            ),
            "wandb_tracking_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/b5_residual_edm_wandb_tracking.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
        "tracking_policy": {
            "mode": "online_required",
            "local_artifacts_are_scientific_authority": True,
            "large_artifact_upload": False,
        },
    }
    tracker = B5EDMOnlineWandbTracker.start(
        spec=wandb_spec,
        config=tracking_config,
        tracking_directory=args.output.parent / f".{args.output.name}.wandb",
    )
    print(
        json.dumps(
            {
                "authorization": authorization,
                "config": config.to_record(),
                "model": model_config.to_record(),
                "environment": environment,
                "artifact_root": str(args.artifact_root),
                "forecast": str(forecast_path),
                "output": str(args.output),
                "wandb": wandb_spec.to_record(),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )

    windows = OneStepWindowDataset(
        catalog,
        split="train",
        target_frames=B5_EDM_SMOKE_TARGETS,
        context_frames=1,
        augment=False,
        seed=args.seed,
        return_physical=False,
    )
    try:
        with B5TrainingForecastArtifact(
            forecast_path,
            expected_sha256=args.h1_training_forecast_sha256,
        ) as forecast:
            dataset = B5ResidualSmokeDataset(windows, forecast)
            try:
                result = train_b5_edm_smoke(
                    dataset=dataset,
                    output=args.output,
                    device=device,
                    paper0_commit=args.paper0_commit,
                    slurm_job_id=args.slurm_job_id,
                    authority=authorization,
                    config=config,
                    model_config=model_config,
                    on_step=tracker.log_step,
                )
                tracking_record = tracker.finish_success(result)
                write_strict_json_atomic(args.output / "wandb.json", tracking_record)
            except BaseException:
                tracker.finish_failure()
                raise
    finally:
        windows.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
