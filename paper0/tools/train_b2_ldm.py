#!/usr/bin/env python3
"""Run the one frozen, bounded B2 latent-diffusion smoke on Rocky 9 CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.b2_training import B2RunConfig, train_b2_smoke
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
    parser.add_argument("--mode", choices=("smoke",), required=True)
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--b2-manifest", type=Path, required=True)
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
    if os_release.get("ID") != "rocky" or os_release.get("VERSION_ID", "").split(".")[0] != "9":
        raise RuntimeError("the B2 smoke requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the B2 smoke requires exactly one allocated CUDA device")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and "H200" not in name:
        raise RuntimeError(f"the B2 smoke requires H100 or H200, found {name!r}")
    return {
        "os_id": os_release["ID"],
        "os_version": os_release["VERSION_ID"],
        "accelerator": name,
    }


def authorize_from_manifest(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    codec_checkpoint: Path,
    codec_sha256: str,
    manifest_path: Path,
) -> dict[str, Any]:
    if mode != "smoke":
        raise RuntimeError("only the bounded B2 smoke is authorized")
    if manifest.get("protocol_status") != "frozen_before_B2_implementation_smoke_or_training":
        raise RuntimeError("B2 protocol status differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("B2 development run is not 85604")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("B2 manifest unexpectedly permits held-out access")
    if manifest.get("full_training_authorized") is not False:
        raise RuntimeError("B2 smoke manifest unexpectedly authorizes full training")
    if "B2_LDM_H2_single_seed_bounded_GPU_smoke_85604" not in manifest.get(
        "authorized_scope", []
    ):
        raise RuntimeError("bounded B2 smoke is absent from authorized scope")
    smoke = manifest.get("implementation_gate", {}).get("gpu_smoke", {})
    expected_smoke = {
        "accelerator": "one Rocky9 H100 or H200",
        "ensemble_members": 2,
        "epochs_max": 2,
        "seed": 1701,
        "training_targets_max": 16,
        "wandb_online_required": True,
    }
    if smoke != expected_smoke or int(seed) != 1701:
        raise RuntimeError("B2 smoke identity or budget differs")
    if manifest.get("model", {}).get("representation") != "C5P-dcae_l10":
        raise RuntimeError("B2 representation differs")
    if manifest.get("data", {}).get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]:
        raise RuntimeError("B2 field set differs")
    if manifest.get("data", {}).get("zperiod") != 5:
        raise RuntimeError("B2 toroidal period differs")
    if manifest.get("data", {}).get("mode_mapping") != "n=5k":
        raise RuntimeError("B2 toroidal mode mapping differs")

    checkpoints = {
        int(item["seed"]): item
        for item in manifest.get("codec", {}).get("selected_checkpoints", [])
    }
    if set(checkpoints) != {1701, 1702, 1703}:
        raise RuntimeError("B2 codec checkpoint seed set differs")
    selected = checkpoints[int(seed)]
    if str(codec_checkpoint) != str(selected["path"]):
        raise RuntimeError("B2 codec path differs from frozen seed checkpoint")
    if str(codec_sha256) != str(selected["sha256"]):
        raise RuntimeError("B2 codec hash differs from frozen seed checkpoint")

    protocol = manifest.get("protocol", {})
    protocol_path = ROOT / str(protocol.get("path", ""))
    if sha256_path(protocol_path) != str(protocol.get("sha256", "")):
        raise RuntimeError("B2 protocol file no longer matches the manifest")
    return {
        "authorized": True,
        "scope": "bounded_non_scientific_B2_LDM_H2_GPU_smoke",
        "development_run": "85604",
        "held_out_85606_read": False,
        "full_B2_training_authorized": False,
        "seed": int(seed),
        "codec": "C5P-dcae_l10",
        "codec_checkpoint": {
            "path": str(codec_checkpoint),
            "sha256": str(codec_sha256),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_path(manifest_path),
        },
    }


def verified_azula_provenance(manifest: Mapping[str, Any]) -> dict[str, str]:
    expected = manifest.get("provenance", {}).get("azula", {})
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

    manifest_path = args.b2_manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("B2 manifest must be inside the Paper 0 repository") from error
    if "85606" in str(manifest_path).lower() or "85606" in str(args.output).lower():
        raise ValueError("held-out paths are prohibited during the B2 smoke")
    manifest = load_strict_json(manifest_path)
    authorization = authorize_from_manifest(
        manifest,
        mode=args.mode,
        seed=args.seed,
        codec_checkpoint=args.codec_checkpoint,
        codec_sha256=args.codec_sha256,
        manifest_path=manifest_path,
    )
    azula = verified_azula_provenance(manifest)
    catalog = load_official_catalog(args.artifact_root)

    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b2_ldm_smoke",
        tags=(
            "paper0",
            "phase3",
            "b2",
            "ldm",
            "smoke",
            "85604-only",
            "non-scientific",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_B2_LDM_H2_GPU_smoke",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "b2_manifest": {
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
        result = train_b2_smoke(
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
