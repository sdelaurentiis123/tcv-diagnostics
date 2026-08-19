#!/usr/bin/env python3
"""Run the frozen seed-1701 full B4 PDE-Refiner training on Rocky 9 CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.pde_refiner_full_training import (
    PDERefinerFullConfig,
    train_pde_refiner_full,
)
from tcv_diagnostics.pde_refiner_full_wandb_tracking import (
    PDERefinerFullOnlineWandbTracker,
)
from tcv_diagnostics.pde_refiner_training import RefinerParentArtifacts
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FULL_MANIFEST_SHA256 = (
    "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), required=True)
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--latent-normalization", type=Path, required=True)
    parser.add_argument("--latent-normalization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
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


def require_rocky9_hopper() -> dict[str, str]:
    os_release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')
    if (
        os_release.get("ID") != "rocky"
        or os_release.get("VERSION_ID", "").split(".")[0] != "9"
    ):
        raise RuntimeError("full B4 training requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full B4 training requires exactly one CUDA device")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and "H200" not in name:
        raise RuntimeError(f"full B4 training requires H100 or H200, found {name!r}")
    return {
        "os_id": os_release["ID"],
        "os_version": os_release["VERSION_ID"],
        "accelerator": name,
    }


def _locked_repo_json(record: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("B4 evidence path must be repository-relative")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("B4 evidence path escapes the repository") from error
    expected = str(record.get("sha256", ""))
    actual = sha256_path(path)
    if actual != expected:
        raise RuntimeError(f"B4 evidence hash differs for {relative}: {actual}")
    return path, load_strict_json(path)


def _locked_repo_file(record: Mapping[str, Any], label: str) -> Path:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"full B4 {label} path is unsafe")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"full B4 {label} path escapes repository") from error
    if sha256_path(path) != str(record.get("sha256", "")):
        raise RuntimeError(f"full B4 {label} bytes differ")
    return path


def authorize_full_from_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    artifacts: RefinerParentArtifacts,
    manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed unless the exact post-smoke B4 full freeze is present."""

    if mode != "full":
        raise RuntimeError("the full B4 entrypoint authorizes only mode='full'")
    if int(seed) != 1701:
        raise RuntimeError("the full B4 pilot authorizes only seed 1701")
    if manifest.get("protocol_status") != (
        "frozen_after_passing_B4_smoke_before_full_training_checkpoint_selection_"
        "or_scientific_evaluation_implementation"
    ):
        raise RuntimeError("full B4 protocol status differs")
    if manifest.get("decision_timing") != (
        "after_completed_B4_seed1701_bounded_GPU_smoke_before_full_B4_training_"
        "or_scientific_generation"
    ):
        raise RuntimeError("full B4 decision timing differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("full B4 development run is not 85604")
    if manifest.get("sequestered_run") != "85606":
        raise RuntimeError("full B4 sequestered run differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("full B4 manifest unexpectedly permits held-out access")
    if manifest.get("full_training_authorized") is not True:
        raise RuntimeError("full B4 training is not authorized")
    if "B4_PDE_Refiner_H1_seed1701_full_training_85604" not in manifest.get(
        "authorized_scope", []
    ):
        raise RuntimeError("full B4 training is absent from authorized scope")
    for forbidden in (
        "85606_access",
        "B4_seed1702_or_seed1703_training",
        "B4_architecture_schedule_noise_or_loss_ablation",
        "O3_or_longer_rollout_execution",
        "assimilation",
        "diagnostic_ranking",
        "physics_derived_training_loss",
    ):
        if forbidden not in manifest.get("forbidden_scope", []):
            raise RuntimeError(f"required forbidden B4 scope is absent: {forbidden}")

    data = manifest.get("data", {})
    expected_data = {
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "input_channels": "physically_valid_complete_C5P_state",
        "context_frames": 1,
        "future_frames": 1,
        "training_targets": [2, 432],
        "guard_frames": [432, 496],
        "validation_targets": [498, 624],
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "absolute_time_input_allowed": False,
        "normalized_frame_index_input_allowed": False,
        "future_truth_input_allowed_during_generation": False,
        "guard_frames_read_allowed": False,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise RuntimeError(f"full B4 data field {key!r} differs")

    model = manifest.get("model", {})
    if (
        model.get("arm") != "B4-PDE-Refiner-H1"
        or model.get("family")
        != "parent_initialized_explicit_latent_PDE_Refiner"
        or model.get("refinement_levels") != [0, 1, 2, 3]
        or model.get("refinement_steps") != 3
        or model.get("network_calls_per_unamortized_member") != 4
        or model.get("physics_derived_loss_allowed") is not False
    ):
        raise RuntimeError("full B4 model identity differs")
    schedule = manifest.get("noise_schedule", {})
    if schedule.get("standard_deviation_by_level") != {
        "1": 0.08583742189325572,
        "2": 0.007368062997280775,
        "3": 0.0006324555320336759,
    }:
        raise RuntimeError("full B4 noise schedule differs")

    training = manifest.get("training", {})
    expected_training = {
        "seed": 1701,
        "epochs": 100,
        "targets_per_epoch": 430,
        "microbatch_targets": 1,
        "gradient_accumulation_targets": 16,
        "final_partial_accumulation_targets": 14,
        "optimizer_steps_per_epoch": 27,
        "total_optimizer_steps": 2700,
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "weight_decay": 1e-5,
        "peak_learning_rate": 1e-4,
        "minimum_learning_rate": 1e-6,
        "warmup_optimizer_steps": 0,
        "gradient_clip": 1.0,
        "ema_decay": 0.995,
        "precision": "float32_no_autocast_TF32_disabled",
        "early_stopping": False,
        "objective": "uniform_level_explicit_standardized_latent_MSE",
        "physics_derived_loss_allowed": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError(f"full B4 training field {key!r} differs")
    levels = training.get("training_level_matrix", {})
    if (
        levels.get("shape") != [100, 430]
        or levels.get("raw_C_order_sha256")
        != "ac370fa17291d8bd4c36ac4d451f78e63250c19ad77cf70a3f8403465e339ff6"
        or levels.get("counts_by_level_0_1_2_3")
        != [10831, 10680, 10722, 10767]
    ):
        raise RuntimeError("full B4 level matrix differs")

    selection = manifest.get("checkpoint_selection", {})
    selection_bank = selection.get("selection_noise_bank", {})
    if (
        selection.get("completed_epoch_candidates") != list(range(5, 101, 5))
        or selection.get("ensemble_members") != 2
        or selection.get("metric")
        != "equal_channel_decoded_standardized_field_MAE_of_ensemble_mean_at_level3"
        or selection_bank.get("shape") != [126, 2, 3]
        or selection_bank.get("npy_sha256")
        != "127936e25054925f4b114d5b174cbe876847555ffd0963ca54ce0e6c72f29884"
        or selection.get(
            "physics_spectrum_transport_calibration_or_WandB_selection_allowed"
        )
        is not False
    ):
        raise RuntimeError("full B4 checkpoint-selection contract differs")
    scientific = manifest.get("scientific_ensemble", {})
    if (
        scientific.get("seed_bank_shape") != [126, 32, 3]
        or scientific.get("seed_bank_npy_sha256")
        != "a1871e069bce6244073bfe1aa835a53c1d7a59302b01f6a366b3dc88297b6205"
        or scientific.get("independent_of_checkpoint_selection_noise") is not True
        or scientific.get("regeneration_or_posthoc_calibration_allowed") is not False
    ):
        raise RuntimeError("full B4 scientific-ensemble contract differs")

    parent = manifest.get("deterministic_parent", {})
    codec = manifest.get("codec", {})
    expected_artifacts = (
        (artifacts.checkpoint_path, parent.get("checkpoint_path"), "parent path"),
        (artifacts.checkpoint_sha256, parent.get("checkpoint_sha256"), "parent hash"),
        (artifacts.codec_path, codec.get("checkpoint_path"), "codec path"),
        (artifacts.codec_sha256, codec.get("checkpoint_sha256"), "codec hash"),
        (
            artifacts.latent_normalization_path,
            codec.get("latent_normalization_path"),
            "latent-normalization path",
        ),
        (
            artifacts.latent_normalization_sha256,
            codec.get("latent_normalization_sha256"),
            "latent-normalization hash",
        ),
    )
    for actual, expected, label in expected_artifacts:
        if str(actual) != str(expected):
            raise RuntimeError(f"full B4 {label} differs")

    if sha256_path(manifest_path) != EXPECTED_FULL_MANIFEST_SHA256:
        raise RuntimeError("full B4 manifest bytes differ from frozen authority")
    for key in ("protocol", "implementation_protocol", "implementation_manifest"):
        _locked_repo_file(manifest.get(key, {}), key)

    _, smoke = _locked_repo_json(
        manifest.get("evidence_locks", {}).get("B4_smoke", {})
    )
    if not (
        smoke.get("status") == "passed"
        and smoke.get("slurm_job_id") == "6899469"
        and smoke.get("held_out_85606_read") is False
        and smoke.get("scientific_result") is False
        and smoke.get("full_B4_training_authorized") is False
        and smoke.get("preoptimization_parent_identity", {}).get("bitwise_exact")
        is True
        and smoke.get("checkpoint_reload_bitwise_exact") is True
        and smoke.get("codec_bitwise_unchanged") is True
        and smoke.get("member_and_stage_probe", {}).get(
            "nonzero_final_diversity_in_every_field"
        )
        is True
    ):
        raise RuntimeError("the hash-locked B4 smoke did not pass its gate")

    return {
        "authorized": True,
        "scope": "B4_PDE_Refiner_H1_seed1701_full_training_85604",
        "development_run": "85604",
        "held_out_85606_read": False,
        "full_B4_training_authorized": True,
        "scientific_result": False,
        "training_complete_is_scientific_acceptance": False,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
        "seed": int(seed),
        "passing_smoke": {
            "job_id": "6899469",
            "sha256": manifest["evidence_locks"]["B4_smoke"]["sha256"],
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": EXPECTED_FULL_MANIFEST_SHA256,
        },
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    environment = require_rocky9_hopper()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", 0)
    config = PDERefinerFullConfig.frozen(seed=args.seed)
    artifacts = RefinerParentArtifacts(
        checkpoint_path=args.parent_checkpoint,
        checkpoint_sha256=args.parent_sha256,
        codec_path=args.codec_checkpoint,
        codec_sha256=args.codec_sha256,
        latent_normalization_path=args.latent_normalization,
        latent_normalization_sha256=args.latent_normalization_sha256,
    )

    manifest_path = args.full_manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("full B4 manifest must be inside the repository") from error
    prohibited_paths = (
        manifest_path,
        args.output,
        args.artifact_root,
        artifacts.checkpoint_path,
        artifacts.codec_path,
        artifacts.latent_normalization_path,
    )
    if any("85606" in str(path).lower() for path in prohibited_paths):
        raise ValueError("held-out paths are prohibited during full B4 training")
    manifest = load_strict_json(manifest_path)
    authorization = authorize_full_from_manifest(
        manifest,
        mode=args.mode,
        seed=args.seed,
        artifacts=artifacts,
        manifest_path=manifest_path,
    )
    catalog = load_official_catalog(args.artifact_root)

    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b4_pde_refiner_full",
        tags=(
            "paper0",
            "phase3",
            "b4",
            "pde-refiner",
            "full-training",
            "85604-only",
            "seed1701",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": "B4_PDE_Refiner_H1_seed1701_full_training_85604",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "full_manifest": {
            "path": str(manifest_relative),
            "sha256": sha256_path(manifest_path),
        },
        "dataset": {
            "artifact_root": str(args.artifact_root),
            "manifest_sha256": sha256_path(
                args.artifact_root / "model_dataset_manifest.json"
            ),
            "normalization_sha256": sha256_path(
                args.artifact_root / "normalization.json"
            ),
            "artifact_index_sha256": sha256_path(
                args.artifact_root / "artifact_sha256.txt"
            ),
            "development_run": "85604",
            "held_out_85606_read": False,
        },
        "training": config.to_record(),
        "environment": environment,
        "precision": {
            "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "autocast": False,
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pde_refiner_full_training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/pde_refiner_full_training.py"
            ),
            "pde_refiner_model_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/pde_refiner.py"
            ),
            "pde_refiner_full_wandb_tracking_sha256": sha256_path(
                ROOT
                / "src/tcv_diagnostics/pde_refiner_full_wandb_tracking.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
        "tracking_policy": {
            "mode": "online_required",
            "local_artifacts_are_scientific_authority": True,
            "checkpoint_upload": False,
        },
    }
    tracker = PDERefinerFullOnlineWandbTracker.start(
        spec=wandb_spec,
        config=tracking_config,
        tracking_directory=args.output.parent / f".{args.output.name}.wandb",
    )
    print(
        json.dumps(
            {
                "authorization": authorization,
                "config": config.to_record(),
                "environment": environment,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "artifact_root": str(args.artifact_root),
                "output": str(args.output),
                "wandb": wandb_spec.to_record(),
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    try:
        result = train_pde_refiner_full(
            config=config,
            catalog=catalog,
            artifacts=artifacts,
            output_directory=args.output,
            paper0_commit=args.paper0_commit,
            slurm_job_id=args.slurm_job_id,
            device=device,
            epoch_callback=tracker.log_epoch,
        )
        tracking_record = tracker.finish_success(result)
        write_strict_json_atomic(args.output / "wandb.json", tracking_record)
    except BaseException:
        tracker.finish_failure()
        raise
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
