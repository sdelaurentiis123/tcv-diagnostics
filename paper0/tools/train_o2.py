#!/usr/bin/env python3
"""Run one frozen C5P deterministic O2 experiment on a CUDA worker."""

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
from tcv_diagnostics.o2_training import O2RunConfig, train_o2
from tcv_diagnostics.wandb_tracking import OnlineWandbTracker, WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--arm", choices=("C5P-H1", "C5P-H2"), required=True)
    parser.add_argument("--seed", type=int, choices=(1701, 1702, 1703), required=True)
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--continuation-manifest", type=Path, required=True)
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


def authorize_from_manifest(
    manifest: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    codec_checkpoint: Path,
    codec_sha256: str,
) -> dict[str, Any]:
    if manifest.get("protocol_status") != (
        "frozen_after_complete_R2_O1_and_before_O2_implementation_or_training"
    ):
        raise RuntimeError("O2 continuation manifest status differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("O2 continuation is not development run 85604")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("O2 continuation unexpectedly permits held-out access")
    if arm not in manifest.get("arms", {}):
        raise RuntimeError(f"O2 arm {arm!r} is not authorized")
    if arm not in ("C5P-H1", "C5P-H2"):
        raise RuntimeError("only C5P H1/H2 are authorized")
    selection = manifest.get("continuation_selection", {})
    if selection.get("selected_representation") != "C5P-dcae_l10":
        raise RuntimeError("selected O2 representation differs")
    if selection.get("selected_seeds") != [1701, 1702, 1703]:
        raise RuntimeError("selected O2 seeds differ")
    if selection.get("selected_pass_count") != 3:
        raise RuntimeError("C5P did not retain its 3/3 O1 pass record")
    if manifest.get("historical_matrix_decision", {}).get("R2_accepted") is not False:
        raise RuntimeError("historical R2 matrix decision was rewritten")
    checkpoints = {
        int(item["seed"]): item
        for item in manifest.get("codec", {}).get("selected_checkpoints", [])
    }
    if set(checkpoints) != {1701, 1702, 1703}:
        raise RuntimeError("continuation manifest checkpoint seeds differ")
    selected = checkpoints[int(seed)]
    if str(codec_checkpoint) != str(selected["path"]):
        raise RuntimeError("requested codec path differs from the frozen seed checkpoint")
    if str(codec_sha256) != str(selected["sha256"]):
        raise RuntimeError("requested codec hash differs from the frozen seed checkpoint")
    return {
        "development_run": "85604",
        "held_out_85606_read": False,
        "arm": arm,
        "seed": int(seed),
        "codec": "C5P-dcae_l10",
        "codec_checkpoint": {
            "path": str(codec_checkpoint),
            "sha256": str(codec_sha256),
        },
        "authorized": True,
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA worker is required")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = O2RunConfig.frozen(mode=args.mode, arm=args.arm, seed=args.seed)
    manifest_path = args.continuation_manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("continuation manifest must be inside Paper 0") from error
    if "85606" in str(manifest_path).lower():
        raise ValueError("held-out manifests are prohibited during O2")
    manifest = load_strict_json(manifest_path)
    authorization = authorize_from_manifest(
        manifest,
        arm=args.arm,
        seed=args.seed,
        codec_checkpoint=args.codec_checkpoint,
        codec_sha256=args.codec_sha256,
    )
    catalog = load_official_catalog(args.artifact_root)

    arm_slug = args.arm.lower()
    wandb_spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type=f"phase2_o2_{args.mode}",
        tags=("paper0", "phase2", "o2", "85604-only", "c5p", arm_slug),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": f"O2_teacher_forced_one_step_{args.mode}",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "continuation_manifest": {
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
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "o2_training_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/o2_training.py"
            ),
            "o2_model_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/o2.py"
            ),
            "vit_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/vit.py"
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
                "authorization": authorization,
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
        result = train_o2(
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
