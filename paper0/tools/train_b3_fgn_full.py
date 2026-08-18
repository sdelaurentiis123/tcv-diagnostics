#!/usr/bin/env python3
"""Run the frozen seed-1701 full B3 FGN training budget on Rocky 9 CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.fgn_training import (
    FGNRunConfig,
    ParentArtifacts,
    train_fgn_full,
)
from tcv_diagnostics.fgn_wandb_tracking import FGNOnlineWandbTracker
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FULL_MANIFEST_SHA256 = (
    "2f1f83b3c4ce50a789d26ed6877142400b5f9f8e994b3e6bc92f997840832ad2"
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
        raise RuntimeError("full B3 training requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full B3 training requires exactly one allocated CUDA device")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and "H200" not in name:
        raise RuntimeError(f"full B3 training requires H100 or H200, found {name!r}")
    return {
        "os_id": os_release["ID"],
        "os_version": os_release["VERSION_ID"],
        "accelerator": name,
    }


def _locked_repo_json(record: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("B3 evidence path must be repository-relative")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("B3 evidence path escapes the repository") from error
    expected = str(record.get("sha256", ""))
    actual = sha256_path(path)
    if actual != expected:
        raise RuntimeError(f"B3 evidence hash differs for {relative}: {actual}")
    return path, load_strict_json(path)


def authorize_full_from_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    artifacts: ParentArtifacts,
    manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed unless the exact post-smoke seed-1701 freeze is present."""

    if mode != "full":
        raise RuntimeError("the full B3 entrypoint authorizes only mode='full'")
    if int(seed) != 1701:
        raise RuntimeError("the full B3 pilot authorizes only seed 1701")
    if manifest.get("protocol_status") != (
        "frozen_after_passing_B3_smoke_before_full_training_or_scientific_"
        "evaluation_implementation"
    ):
        raise RuntimeError("full B3 protocol status differs")
    if manifest.get("decision_timing") != (
        "after_completed_B3_seed1701_smoke_before_full_B3_training_checkpoint_"
        "selection_or_scientific_ensemble_generation"
    ):
        raise RuntimeError("full B3 decision timing differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("full B3 development run is not 85604")
    if manifest.get("sequestered_run") != "85606":
        raise RuntimeError("full B3 sequestered run differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("full B3 manifest unexpectedly permits held-out access")
    if manifest.get("full_training_authorized") is not True:
        raise RuntimeError("full B3 training is not authorized")
    if "B3_FGN_H1_seed1701_full_training_85604" not in manifest.get(
        "authorized_scope", []
    ):
        raise RuntimeError("full B3 training is absent from authorized scope")
    for forbidden in (
        "85606_access",
        "B3_seeds_1702_or_1703_training",
        "B3_architecture_or_noise_ablation",
        "O3_or_longer_rollout",
        "assimilation",
        "diagnostic_ranking",
        "physics_derived_training_loss",
    ):
        if forbidden not in manifest.get("forbidden_scope", []):
            raise RuntimeError(f"required forbidden B3 scope is absent: {forbidden}")

    data = manifest.get("data", {})
    expected_data = {
        "fields": ["Ne", "Pe", "Pi", "phi", "Vi"],
        "training_targets": [2, 432],
        "guard_frames": [432, 496],
        "validation_targets": [498, 624],
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "absolute_time_input_allowed": False,
        "normalized_frame_index_input_allowed": False,
        "future_truth_input_allowed": False,
        "guard_frames_read_allowed": False,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise RuntimeError(f"full B3 data field {key!r} differs")

    model = manifest.get("model", {})
    if (
        model.get("arm") != "B3-FGN-H1"
        or model.get("family") != "functional_generative_retrofit"
        or model.get("context_frames") != 1
        or model.get("future_frames") != 1
        or model.get("physical_time_input") is not False
    ):
        raise RuntimeError("full B3 model identity differs")
    noise = manifest.get("functional_noise", {})
    if (
        noise.get("raw_dimension") != 32
        or noise.get("embedded_dimension") != 256
        or noise.get("noise_layers") != "all"
        or noise.get("spatial_semantics")
        != "one_global_vector_shared_across_all_tokens"
        or noise.get("member_semantics")
        != "independent_raw_vector_per_ensemble_member"
    ):
        raise RuntimeError("full B3 functional-noise contract differs")

    training = manifest.get("training", {})
    expected_training = {
        "seed": 1701,
        "epochs": 100,
        "targets_per_epoch": 430,
        "validation_targets": 126,
        "microbatch": 1,
        "ensemble_members_per_target": 2,
        "gradient_accumulation": 16,
        "final_partial_accumulation": 14,
        "optimizer_steps_per_epoch": 27,
        "total_optimizer_steps": 2700,
        "optimizer": "AdamW",
        "betas": [0.9, 0.99],
        "weight_decay": 0.0,
        "common_parameter_peak_learning_rate": 3.0e-5,
        "new_parameter_peak_learning_rate": 1.0e-4,
        "warmup_epochs": 10,
        "warmup_optimizer_steps": 270,
        "gradient_clip": 1.0,
        "precision": "bfloat16_autocast",
        "early_stopping": False,
        "physics_derived_loss_allowed": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError(f"full B3 training field {key!r} differs")

    selection = manifest.get("selection_noise", {})
    evaluation = manifest.get("scientific_ensemble", {})
    if selection.get("seed") != 31003 or selection.get("shape") != [126, 2, 32]:
        raise RuntimeError("full B3 checkpoint-selection noise differs")
    if (
        evaluation.get("seed") != 31032
        or evaluation.get("noise_shape") != [126, 32, 32]
        or evaluation.get("independent_of_checkpoint_selection_noise") is not True
        or evaluation.get("forecast_shape")
        != [126, 32, 1, 5, 64, 32, 88]
        or evaluation.get("posthoc_calibration_allowed") is not False
    ):
        raise RuntimeError("full B3 scientific-ensemble contract differs")

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
            raise RuntimeError(f"full B3 {label} differs")

    if sha256_path(manifest_path) != EXPECTED_FULL_MANIFEST_SHA256:
        raise RuntimeError("full B3 manifest bytes differ from the frozen manifest")
    for key in ("protocol", "implementation_protocol"):
        lock = manifest.get(key, {})
        path = ROOT / str(lock.get("path", ""))
        if sha256_path(path) != str(lock.get("sha256", "")):
            raise RuntimeError(f"full B3 {key} file no longer matches its manifest")

    smoke_path, smoke = _locked_repo_json(
        manifest.get("evidence_locks", {}).get("B3_smoke", {})
    )
    if not (
        smoke.get("status") == "passed"
        and smoke.get("slurm_job_id") == "6898604"
        and smoke.get("held_out_85606_read") is False
        and smoke.get("scientific_result") is False
        and smoke.get("full_B3_training_authorized") is False
        and smoke.get("preoptimization_parent_identity", {}).get("bitwise_exact")
        is True
        and smoke.get("checkpoint_reload_bitwise_exact") is True
        and smoke.get("codec_bitwise_unchanged") is True
    ):
        raise RuntimeError("the hash-locked B3 smoke did not pass its bounded gate")
    for evidence_name in ("deterministic_O2_result", "amended_B2_result"):
        _locked_repo_json(manifest.get("evidence_locks", {}).get(evidence_name, {}))

    return {
        "authorized": True,
        "scope": "B3_FGN_H1_seed1701_full_training_85604",
        "development_run": "85604",
        "held_out_85606_read": False,
        "full_B3_training_authorized": True,
        "probabilistic_scientific_gate_evaluated": False,
        "seed": 1701,
        "parent": {
            "path": str(artifacts.checkpoint_path),
            "sha256": artifacts.checkpoint_sha256,
        },
        "codec": {
            "path": str(artifacts.codec_path),
            "sha256": artifacts.codec_sha256,
        },
        "latent_normalization": {
            "path": str(artifacts.latent_normalization_path),
            "sha256": artifacts.latent_normalization_sha256,
            "refit": False,
        },
        "full_manifest": {
            "path": str(manifest_path),
            "sha256": EXPECTED_FULL_MANIFEST_SHA256,
        },
        "passing_smoke": {
            "path": str(smoke_path),
            "sha256": sha256_path(smoke_path),
            "job_id": "6898604",
        },
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    environment = require_rocky9_hopper()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", 0)
    config = FGNRunConfig.frozen(mode=args.mode, seed=args.seed)
    artifacts = ParentArtifacts(
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
        raise ValueError("full B3 manifest must be inside the repository") from error
    prohibited_paths = (
        manifest_path,
        args.output,
        args.artifact_root,
        artifacts.checkpoint_path,
        artifacts.codec_path,
        artifacts.latent_normalization_path,
    )
    if any("85606" in str(path).lower() for path in prohibited_paths):
        raise ValueError("held-out paths are prohibited during full B3 training")
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
        job_type="phase3_b3_fgn_full",
        tags=(
            "paper0",
            "phase3",
            "b3",
            "fgn",
            "full-training",
            "85604-only",
            "seed-1701",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": "B3_FGN_H1_seed1701_full_training_85604",
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
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fgn_training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/fgn_training.py"
            ),
            "fgn_model_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/functional_noise.py"
            ),
            "fgn_wandb_tracking_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/fgn_wandb_tracking.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
        "tracking_policy": {
            "mode": "online_required",
            "local_artifacts_are_scientific_authority": True,
            "checkpoint_upload": False,
        },
    }
    tracker = FGNOnlineWandbTracker.start(
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
        result = train_fgn_full(
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
