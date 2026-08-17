#!/usr/bin/env python3
"""Run one frozen O1 deterministic codec experiment on a CUDA worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from tcv_diagnostics.codec_training import CodecRunConfig, sha256_path, train_codec
from tcv_diagnostics.model_data import write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.wandb_tracking import OnlineWandbTracker, WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--codec", choices=("dcae_l20", "dcae_l10"), required=True)
    parser.add_argument("--family", choices=("c5p", "e6b"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
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


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA worker is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = CodecRunConfig.frozen(
        mode=args.mode,
        codec=args.codec,
        family=args.family,
        seed=args.seed,
    )
    catalog = load_official_catalog(args.artifact_root)
    run_manifest = args.run_manifest.resolve()
    try:
        run_manifest_relative = run_manifest.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("run manifest must be inside the Paper 0 checkout") from error
    if not run_manifest.is_file():
        raise FileNotFoundError(run_manifest)
    if "85606" in str(run_manifest).lower():
        raise ValueError("held-out run manifests are prohibited during codec training")
    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type=f"phase2_o1_codec_{args.mode}",
        tags=("paper0", "phase2", "o1", "codec", "85604-only", args.family),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": f"O1_codec_{args.mode}",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "run_manifest": {
            "path": str(run_manifest_relative),
            "sha256": sha256_path(run_manifest),
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
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "codec_training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/codec_training.py"
            ),
            "wandb_tracking_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/wandb_tracking.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
        "tracking_policy": {
            "mode": "online_required",
            "local_artifacts_are_scientific_authority": True,
            "checkpoint_upload": False,
        },
    }
    tracker = OnlineWandbTracker.start(
        spec=wandb_spec,
        config=tracking_config,
        tracking_directory=args.output.parent / f".{args.output.name}.wandb",
    )
    print(
        json.dumps(
            {
                "config": config.to_record(),
                "device": torch.cuda.get_device_name(device),
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
        result = train_codec(
            config=config,
            catalog=catalog,
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
