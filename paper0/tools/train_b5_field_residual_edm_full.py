#!/usr/bin/env python3
"""Run the one frozen seed-1701 full B5 training job on Rocky 9 H100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.b5_residual_edm_full_training import (
    B5EDMFullConfig,
    B5ResidualOneStepDataset,
    B5_FULL_TRAIN_TARGETS,
    B5_FULL_VALIDATION_TARGETS,
    train_b5_edm_full,
)
from tcv_diagnostics.b5_residual_edm_full_wandb_tracking import (
    B5EDMFullOnlineWandbTracker,
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
from tcv_diagnostics.o2_forecast import O2ForecastArtifact
from tcv_diagnostics.o2_training_data import OneStepWindowDataset
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_SHA256 = (
    "61f1fa565e2bcff008cbe72909daa97362dabe96d160a9beee4a3d5aa87d1334"
)
EXPECTED_PROTOCOL_SHA256 = (
    "faab336bf3ae1a49008eff0e6604d48d9c475aa83732184668c4c2e444c928b9"
)
EXPECTED_SMOKE_RESULT_SHA256 = (
    "1bb21a853d63ca66f16daaaa2a1521cec369fa0f3af64ec9d4d3f30bed73ddbd"
)
EXPECTED_TRAIN_FORECAST_SHA256 = (
    "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
)
EXPECTED_VALIDATION_FORECAST_SHA256 = (
    "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
)
EXPECTED_RESIDUAL_AUDIT_SHA256 = (
    "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), required=True)
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--h1-training-forecast", type=Path, required=True)
    parser.add_argument("--h1-training-forecast-sha256", required=True)
    parser.add_argument("--h1-validation-forecast", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast-sha256", required=True)
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
        raise RuntimeError("full B5 training requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full B5 training requires exactly one CUDA GPU")
    accelerator = torch.cuda.get_device_name(0)
    if "H100" not in accelerator:
        raise RuntimeError(f"full B5 training requires H100, found {accelerator!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated H100 does not report bfloat16 support")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "accelerator": accelerator,
        "cuda_device_count": torch.cuda.device_count(),
        "bfloat16_supported": True,
    }


def _locked_repo_json(record: Mapping[str, Any]) -> Mapping[str, Any]:
    relative = Path(str(record.get("tracked_result", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("B5 smoke evidence path is unsafe")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("B5 smoke evidence escapes the repository") from error
    expected = str(record.get("tracked_result_sha256", ""))
    if expected != EXPECTED_SMOKE_RESULT_SHA256 or sha256_path(path) != expected:
        raise RuntimeError("B5 smoke evidence bytes differ")
    return load_strict_json(path)


def authorize_full_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    mode: str,
    seed: int,
    train_forecast_sha256: str,
    validation_forecast_sha256: str,
    residual_audit_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless the exact post-smoke B5 full freeze is present."""

    if mode != "full" or int(seed) != 1701:
        raise RuntimeError("only the frozen seed-1701 full B5 run is authorized")
    if manifest.get("protocol_status") != (
        "frozen_after_passing_job_6901469_before_B5_full_training_validation_"
        "or_evaluation_implementation"
    ):
        raise RuntimeError("B5 full protocol status differs")
    if (
        manifest.get("development_run") != "85604"
        or manifest.get("sequestered_run") != "85606"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("B5 full development or sequestered scope differs")
    required_scope = {
        "B5_seed1701_full_training_on_85604_training_targets",
        "B5_seed1701_checkpoint_selection_on_85604_validation_denoising_loss",
    }
    if not required_scope.issubset(set(manifest.get("authorized_scope", ()))):
        raise RuntimeError("B5 full training scope is absent")
    for forbidden in (
        "architecture_or_schedule_sweep",
        "additional_model_seeds",
        "physics_metric_checkpoint_selection",
        "O3_fixed_block_forecast",
        "O4_autonomous_rollout",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    ):
        if forbidden not in manifest.get("forbidden_scope", ()):
            raise RuntimeError(f"required forbidden B5 scope is absent: {forbidden}")

    protocol = manifest.get("protocol", {})
    if (
        protocol.get("path")
        != "paper0/protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md"
        or protocol.get("status")
        != "frozen_before_full_training_validation_or_evaluation_implementation"
        or sha256_path(ROOT / str(protocol.get("path"))) != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("B5 full protocol bytes or identity differ")

    locks = manifest.get("evidence_locks", {})
    smoke = _locked_repo_json(locks.get("B5_smoke", {}))
    if not (
        smoke.get("status") == "passed"
        and smoke.get("slurm", {}).get("job_id") == "6901469"
        and smoke.get("mechanics", {}).get("parameter_count") == 11_604_709
        and smoke.get("mechanics", {}).get("checkpoint_reload_bitwise_exact") is True
        and smoke.get("mechanics", {}).get("toroidal_equivariance_passed") is True
        and smoke.get("sampler_probe", {}).get("nonzero_member_diversity") is True
        and smoke.get("scientific_result") is False
        and smoke.get("held_out_85606_read") is False
    ):
        raise RuntimeError("the hash-locked B5 smoke did not pass its gate")
    residual_lock = locks.get("residual_audit", {})
    train_lock = locks.get("H1_training_forecast", {})
    validation_lock = locks.get("H1_validation_forecast", {})
    if (
        residual_audit_sha256 != EXPECTED_RESIDUAL_AUDIT_SHA256
        or residual_lock.get("sha256") != residual_audit_sha256
    ):
        raise RuntimeError("B5 residual-audit hash differs")
    if (
        train_forecast_sha256 != EXPECTED_TRAIN_FORECAST_SHA256
        or train_lock.get("sha256") != train_forecast_sha256
        or train_lock.get("target_frames") != [2, 432]
        or train_lock.get("truth_separated") is not True
    ):
        raise RuntimeError("B5 H1 training-forecast lock differs")
    if (
        validation_forecast_sha256 != EXPECTED_VALIDATION_FORECAST_SHA256
        or validation_lock.get("sha256") != validation_forecast_sha256
        or validation_lock.get("target_frames") != [498, 624]
        or validation_lock.get("truth_separated") is not True
    ):
        raise RuntimeError("B5 H1 validation-forecast lock differs")

    data = manifest.get("data", {})
    expected_data = {
        "fields": list(B5_FIELD_ORDER),
        "context_frames": 1,
        "future_frames": 1,
        "training_targets": [2, 432],
        "guard_frames": [432, 496],
        "validation_targets": [498, 624],
        "volume_shape": [5, 64, 32, 88],
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "absolute_time_input_allowed": False,
        "future_truth_condition_allowed": False,
        "guard_frames_read_allowed": False,
        "toroidal_roll_augmentation": False,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise RuntimeError(f"B5 full data field {key!r} differs")
    residual = manifest.get("residual_target", {})
    if (
        residual.get("definition")
        != "standardized_truth_minus_frozen_H1_standardized_mean"
        or residual.get("normalization_operation") != "divide_without_centering"
        or residual.get("field_order") != list(B5_FIELD_ORDER)
        or residual.get("scale") != list(B5_RESIDUAL_SCALES)
        or residual.get("validation_statistics_used") is not False
    ):
        raise RuntimeError("B5 full residual target differs")

    model = manifest.get("model", {})
    model_config = FieldResidualUNetConfig().to_record()
    if (
        model.get("name") != model_config["name"]
        or model.get("initialization") != "fresh_seed1701_not_smoke_checkpoint"
        or model.get("parameter_count") != 11_604_709
        or model.get("joint_output_fields") != 5
        or model.get("dynamic_condition_channels") != 10
        or model.get("padding_by_axis") != ["zeros", "zeros", "circular"]
        or model.get("DCAE_or_latent_representation_used") is not False
        or model.get("physics_derived_training_loss_allowed") is not False
    ):
        raise RuntimeError("B5 full model identity differs")
    config = B5EDMFullConfig(seed=int(seed))
    training = manifest.get("full_training", {})
    expected_training = {
        "authorized": True,
        "seed": 1701,
        "epochs": 100,
        "target_presentations": 43_000,
        "training_order_seed": 67_501,
        "training_noise_seed": 67_502,
        "microbatch_targets": 1,
        "gradient_accumulation_targets": 4,
        "optimizer_steps_per_epoch": 108,
        "total_optimizer_steps": 10_800,
        "optimizer": "AdamW",
        "betas": [0.9, 0.99],
        "weight_decay": 0.0,
        "peak_learning_rate": 1.0e-4,
        "minimum_learning_rate": 1.0e-6,
        "gradient_clip": 1.0,
        "EMA_decay_per_optimizer_step": 0.999,
        "training_precision": "bfloat16_autocast_with_FP32_loss_optimizer_and_EMA",
        "TF32_allowed": False,
        "early_stopping": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError(f"B5 full training field {key!r} differs")
    if (
        training.get("training_order_raw_sha256")
        != "4eb79c67e03623ccb5e0b1735ff0d3a13c1202db833d0c42509af3ba7b0eafda"
        or training.get("training_order_npy_sha256")
        != "0e775e59e3596e63c2324a9a9fa5ff82df9dca1ff9d4923fbbef3a4126e97806"
    ):
        raise RuntimeError("B5 full training-order bytes differ")
    selection = manifest.get("checkpoint_selection", {})
    if (
        selection.get("candidate_completed_epochs") != list(range(5, 101, 5))
        or selection.get("weights") != "EMA"
        or selection.get("validation_targets") != [498, 624]
        or selection.get("validation_probe_draws_per_target") != 4
        or selection.get("validation_probe_count_per_candidate") != 504
        or selection.get("seed_bank_seed") != 67_503
        or selection.get("seed_bank_raw_sha256")
        != "f0e736a16be18289ef64fc190fac917eda284eac13ed5117fa7be2d7c2b7d411"
        or selection.get("rule")
        != "earliest_numerically_lowest_metric_after_complete_100_epoch_budget"
        or selection.get("physics_metric_allowed") is not False
        or selection.get("sampled_forecast_metric_allowed") is not False
        or selection.get("85606_value_allowed") is not False
    ):
        raise RuntimeError("B5 full checkpoint-selection contract differs")
    execution = manifest.get("execution", {})
    if execution != {
        "os": "Rocky_Linux_9",
        "accelerator": "one_H100",
        "submission": "sbatch_from_rusty9",
        "wandb_online_required": True,
        "local_Ceph_artifacts_are_scientific_authority": True,
        "large_artifact_upload_allowed": False,
    }:
        raise RuntimeError("B5 full execution contract differs")
    if sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("B5 full manifest bytes differ from frozen authority")
    if config != B5EDMFullConfig():
        raise RuntimeError("B5 full implementation config differs")
    return {
        "authorized": True,
        "scope": "B5_seed1701_full_training_and_data_only_selection_85604",
        "development_run": "85604",
        "blind_test_read": False,
        "scientific_result": False,
        "scientific_forecast_generated": False,
        "seed": int(seed),
        "evidence_hashes": {
            "H1_training_forecast": train_forecast_sha256,
            "H1_validation_forecast": validation_forecast_sha256,
            "residual_audit": residual_audit_sha256,
            "manifest": EXPECTED_MANIFEST_SHA256,
            "protocol": EXPECTED_PROTOCOL_SHA256,
            "smoke_result": EXPECTED_SMOKE_RESULT_SHA256,
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
        raise ValueError("B5 full manifest must be inside the repository") from error
    prohibited_paths = (
        args.artifact_root,
        args.h1_training_forecast,
        args.h1_validation_forecast,
        args.residual_audit,
        args.output,
        manifest_path,
    )
    if any("85606" in str(path).lower() for path in prohibited_paths):
        raise ValueError("held-out paths are prohibited during full B5 training")

    train_forecast_path = verify_input(
        args.h1_training_forecast,
        args.h1_training_forecast_sha256,
        "B5 H1 training forecast",
    )
    validation_forecast_path = verify_input(
        args.h1_validation_forecast,
        args.h1_validation_forecast_sha256,
        "B5 H1 validation forecast",
    )
    residual_audit_path = verify_input(
        args.residual_audit,
        args.residual_audit_sha256,
        "B5 residual audit",
    )
    manifest = load_strict_json(manifest_path)
    authorization = authorize_full_from_manifest(
        manifest,
        manifest_path=manifest_path,
        mode=args.mode,
        seed=args.seed,
        train_forecast_sha256=args.h1_training_forecast_sha256,
        validation_forecast_sha256=args.h1_validation_forecast_sha256,
        residual_audit_sha256=args.residual_audit_sha256,
    )
    model_data_lock = manifest["evidence_locks"]["model_dataset"]
    for filename, key in (
        ("model_dataset_manifest.json", "manifest_sha256"),
        ("normalization.json", "normalization_sha256"),
        ("artifact_sha256.txt", "artifact_index_sha256"),
    ):
        verify_input(
            args.artifact_root / filename,
            model_data_lock[key],
            f"B5 model-data {filename}",
        )
    residual_audit = load_strict_json(residual_audit_path)
    if (
        residual_audit.get("scope") != "B5_frozen_H1_training_residual_audit_85604"
        or residual_audit.get("target_frames") != [2, 432]
        or residual_audit.get("canonical_shape") != [430, 5, 64, 32, 88]
        or residual_audit.get("field_order") != list(B5_FIELD_ORDER)
        or residual_audit.get("validation_frames_read") is not False
        or residual_audit.get("scientific_boundaries", {}).get("architecture_selected")
        is not False
    ):
        raise RuntimeError("B5 residual-audit authority differs")
    observed_scales = [
        residual_audit["scale"]["global"][field]["population_standard_deviation"]
        for field in B5_FIELD_ORDER
    ]
    if observed_scales != list(B5_RESIDUAL_SCALES):
        raise RuntimeError("B5 training residual scales differ")

    catalog = load_official_catalog(args.artifact_root)
    config = B5EDMFullConfig(seed=args.seed)
    model_config = FieldResidualUNetConfig()
    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b5_joint_field_residual_edm_full",
        tags=(
            "paper0",
            "phase3",
            "b5",
            "joint-field-residual-edm",
            "full-training",
            "85604-only",
            "seed1701",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": authorization["scope"],
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "b5_manifest": {
            "path": str(manifest_relative),
            "sha256": sha256_path(manifest_path),
        },
        "inputs": {
            "model_data_root": str(args.artifact_root),
            "H1_training_forecast_sha256": sha256_path(train_forecast_path),
            "H1_validation_forecast_sha256": sha256_path(validation_forecast_path),
            "residual_audit_sha256": sha256_path(residual_audit_path),
            "development_run": "85604",
            "held_out_85606_read": False,
        },
        "training": config.to_record(),
        "model": model_config.to_record(),
        "environment": environment,
        "precision": {
            "training": config.training_precision,
            "validation": config.validation_precision,
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
                ROOT / "src/tcv_diagnostics/b5_residual_edm_full_training.py"
            ),
            "wandb_tracking_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/b5_residual_edm_full_wandb_tracking.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
        "tracking_policy": {
            "mode": "online_required",
            "local_artifacts_are_scientific_authority": True,
            "large_artifact_upload": False,
        },
    }
    tracker = B5EDMFullOnlineWandbTracker.start(
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
                "training_forecast": str(train_forecast_path),
                "validation_forecast": str(validation_forecast_path),
                "output": str(args.output),
                "wandb": wandb_spec.to_record(),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )

    training_windows = None
    validation_windows = None
    try:
        training_windows = OneStepWindowDataset(
            catalog,
            split="train",
            target_frames=B5_FULL_TRAIN_TARGETS,
            context_frames=1,
            augment=False,
            seed=args.seed,
            return_physical=False,
        )
        validation_windows = OneStepWindowDataset(
            catalog,
            split="validation",
            target_frames=B5_FULL_VALIDATION_TARGETS,
            context_frames=1,
            augment=False,
            seed=args.seed,
            return_physical=False,
        )
        with B5TrainingForecastArtifact(
            train_forecast_path,
            expected_sha256=args.h1_training_forecast_sha256,
        ) as train_forecast, O2ForecastArtifact(
            validation_forecast_path,
            expected_sha256=args.h1_validation_forecast_sha256,
            target_frames=B5_FULL_VALIDATION_TARGETS,
        ) as validation_forecast:
            training_dataset = B5ResidualOneStepDataset(
                training_windows, train_forecast, split="train"
            )
            validation_dataset = B5ResidualOneStepDataset(
                validation_windows, validation_forecast, split="validation"
            )
            result = train_b5_edm_full(
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                output=args.output,
                device=device,
                paper0_commit=args.paper0_commit,
                slurm_job_id=args.slurm_job_id,
                authority=authorization,
                config=config,
                model_config=model_config,
                on_epoch=tracker.log_epoch,
            )
            tracking_record = tracker.finish_success(result)
            write_strict_json_atomic(args.output / "wandb.json", tracking_record)
    except BaseException:
        tracker.finish_failure()
        raise
    finally:
        if training_windows is not None:
            training_windows.close()
        if validation_windows is not None:
            validation_windows.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
