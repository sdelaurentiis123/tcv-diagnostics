#!/usr/bin/env python3
"""Train one frozen B2-LDM-H2 seed on 85604 using Rocky 9 CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.b2_training import B2RunConfig, train_b2_full
from tcv_diagnostics.b2_wandb_tracking import B2OnlineWandbTracker
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), required=True)
    parser.add_argument(
        "--seed", type=int, choices=(1701, 1702, 1703), required=True
    )
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
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
        raise RuntimeError("full B2 training requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full B2 training requires exactly one allocated CUDA device")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and "H200" not in name:
        raise RuntimeError(f"full B2 training requires H100 or H200, found {name!r}")
    return {
        "os_id": os_release["ID"],
        "os_version": os_release["VERSION_ID"],
        "accelerator": name,
    }


def _locked_repo_record(record: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("evidence path must be repository-relative")
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("evidence path escapes the Paper 0 repository") from error
    actual = sha256_path(path)
    if actual != str(record.get("sha256", "")):
        raise RuntimeError(f"evidence hash differs for {relative}: {actual}")
    return path, load_strict_json(path)


def authorize_full_from_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    codec_checkpoint: Path,
    codec_sha256: str,
    manifest_path: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Fail closed unless the exact post-smoke full-training freeze is present."""

    if mode != "full":
        raise RuntimeError("the full B2 entrypoint authorizes only mode='full'")
    if manifest.get("protocol_status") != (
        "frozen_before_B2_full_training_or_scientific_metric_implementation"
    ):
        raise RuntimeError("full B2 protocol status differs")
    if manifest.get("decision_timing") != (
        "after_passing_B2_smoke_before_full_training_or_scientific_metric_implementation"
    ):
        raise RuntimeError("full B2 decision timing differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("full B2 development run is not 85604")
    if manifest.get("sequestered_run") != "85606":
        raise RuntimeError("full B2 sequestered run differs")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("full B2 manifest unexpectedly permits held-out access")
    if manifest.get("full_training_authorized") is not True:
        raise RuntimeError("full B2 training is not authorized")
    if "B2_LDM_H2_full_training_three_seeds_85604" not in manifest.get(
        "authorized_scope", []
    ):
        raise RuntimeError("full B2 training is absent from authorized scope")
    for forbidden in (
        "85606_access",
        "O3_or_longer_rollout",
        "assimilation",
        "diagnostic_ranking",
        "physics_derived_training_loss",
    ):
        if forbidden not in manifest.get("forbidden_scope", []):
            raise RuntimeError(f"required forbidden scope is absent: {forbidden}")

    model = manifest.get("model", {})
    if model != {
        "arm": "B2-LDM-H2",
        "context_frames": 2,
        "family": "LOLA_style_masked_latent_diffusion",
        "future_frames": 1,
        "representation": "C5P-dcae_l10",
        "trajectory_frames": 3,
    }:
        raise RuntimeError("full B2 model identity differs")
    data = manifest.get("data", {})
    if data.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]:
        raise RuntimeError("full B2 field set differs")
    if data.get("training_targets") != [2, 432]:
        raise RuntimeError("full B2 training target interval differs")
    if data.get("validation_targets") != [498, 624]:
        raise RuntimeError("full B2 validation target interval differs")
    if data.get("zperiod") != 5 or data.get("mode_mapping") != "n=5k":
        raise RuntimeError("full B2 toroidal convention differs")
    if data.get("future_truth_input_allowed") is not False:
        raise RuntimeError("full B2 unexpectedly permits future truth input")

    training = manifest.get("training", {})
    expected_training = {
        "seeds": [1701, 1702, 1703],
        "epochs": 200,
        "targets_per_epoch": 430,
        "validation_targets": 126,
        "gradient_accumulation": 16,
        "optimizer_steps_per_epoch": 27,
        "total_optimizer_steps": 5400,
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.99],
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "physics_derived_loss_allowed": False,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise RuntimeError(f"full B2 training field {key!r} differs")
    if int(seed) not in training["seeds"]:
        raise RuntimeError("full B2 seed is outside the frozen matrix")

    protocol = manifest.get("protocol", {})
    protocol_path = ROOT / str(protocol.get("path", ""))
    if sha256_path(protocol_path) != str(protocol.get("sha256", "")):
        raise RuntimeError("full B2 protocol file no longer matches its manifest")

    locks = manifest.get("evidence_locks", {})
    implementation_path, implementation = _locked_repo_record(
        locks.get("B2_implementation_manifest", {})
    )
    smoke_path, smoke = _locked_repo_record(locks.get("B2_smoke", {}))
    if implementation.get("protocol_status") != (
        "frozen_before_B2_implementation_smoke_or_training"
    ):
        raise RuntimeError("B2 implementation manifest identity differs")
    if not (
        smoke.get("one_gpu_smoke_passed") is True
        and smoke.get("rocky9_cpu_suite_passed") is True
        and smoke.get("scientific_result") is False
        and smoke.get("held_out_85606_read") is False
    ):
        raise RuntimeError("the hash-locked B2 smoke did not pass its bounded gate")

    checkpoints = {
        int(item["seed"]): item
        for item in implementation.get("codec", {}).get(
            "selected_checkpoints", []
        )
    }
    if set(checkpoints) != {1701, 1702, 1703}:
        raise RuntimeError("B2 codec checkpoint seed set differs")
    selected = checkpoints[int(seed)]
    if str(codec_checkpoint) != str(selected["path"]):
        raise RuntimeError("full B2 codec path differs from the frozen seed checkpoint")
    if str(codec_sha256) != str(selected["sha256"]):
        raise RuntimeError("full B2 codec hash differs from the frozen seed checkpoint")

    return (
        {
            "authorized": True,
            "scope": "B2_LDM_H2_full_training_85604",
            "development_run": "85604",
            "held_out_85606_read": False,
            "full_B2_training_authorized": True,
            "probabilistic_scientific_gate_evaluated": False,
            "seed": int(seed),
            "codec": "C5P-dcae_l10",
            "codec_checkpoint": {
                "path": str(codec_checkpoint),
                "sha256": str(codec_sha256),
            },
            "full_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_path(manifest_path),
            },
            "implementation_manifest": {
                "path": str(implementation_path),
                "sha256": sha256_path(implementation_path),
            },
            "passing_smoke": {
                "path": str(smoke_path),
                "sha256": sha256_path(smoke_path),
                "job_id": str(smoke["slurm_job_id"]),
            },
        },
        implementation,
    )


def verified_azula_provenance(
    implementation_manifest: Mapping[str, Any],
) -> dict[str, str]:
    expected = implementation_manifest.get("provenance", {}).get("azula", {})
    version = importlib.metadata.version("azula")
    if version != expected.get("version") or version != "0.3.1":
        raise RuntimeError(f"Azula version {version!r} differs from frozen 0.3.1")
    import azula.denoise
    import azula.noise
    import azula.sample

    paths = {
        "sample_py_sha256": Path(azula.sample.__file__).resolve(),
        "denoise_py_sha256": Path(azula.denoise.__file__).resolve(),
        "noise_py_sha256": Path(azula.noise.__file__).resolve(),
    }
    record = {"version": version}
    for label, path in paths.items():
        actual = sha256_path(path)
        if actual != expected.get(label):
            raise RuntimeError(f"installed Azula {label} differs: {actual}")
        record[label] = actual
    return record


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    environment = require_rocky9_hopper()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = B2RunConfig.frozen(mode=args.mode, seed=args.seed)

    manifest_path = args.full_manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("full B2 manifest must be inside the repository") from error
    if "85606" in str(args.artifact_root).lower() or "85606" in str(args.output).lower():
        raise ValueError("held-out paths are prohibited during full B2 training")
    manifest = load_strict_json(manifest_path)
    authorization, implementation_manifest = authorize_full_from_manifest(
        manifest,
        mode=args.mode,
        seed=args.seed,
        codec_checkpoint=args.codec_checkpoint,
        codec_sha256=args.codec_sha256,
        manifest_path=manifest_path,
    )
    azula = verified_azula_provenance(implementation_manifest)
    catalog = load_official_catalog(args.artifact_root)

    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b2_ldm_full",
        tags=(
            "paper0",
            "phase3",
            "b2",
            "ldm",
            "full-training",
            "85604-only",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": "B2_LDM_H2_full_training_85604",
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
            "azula": azula,
            "b2_training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/b2_training.py"
            ),
            "b2_wandb_tracking_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/b2_wandb_tracking.py"
            ),
            "latent_diffusion_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/latent_diffusion.py"
            ),
            "modulated_vit_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/modulated_vit.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
        "tracking_policy": {
            "mode": "online_required",
            "local_artifacts_are_scientific_authority": True,
            "checkpoint_upload": False,
        },
    }
    tracker = B2OnlineWandbTracker.start(
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
                "azula": azula,
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
        result = train_b2_full(
            config=config,
            catalog=catalog,
            codec_checkpoint=args.codec_checkpoint,
            codec_checkpoint_sha256=args.codec_sha256,
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
