#!/usr/bin/env python3
"""Run the one frozen, bounded B3 FGN smoke on Rocky 9 CUDA."""

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
    train_fgn_smoke,
)
from tcv_diagnostics.fgn_wandb_tracking import FGNOnlineWandbTracker
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_SHA256 = (
    "8789c1a922bbeb9817144344563107f6e72a7a7c549436ae698155d93daba900"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke",), required=True)
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
    parser.add_argument("--b3-manifest", type=Path, required=True)
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
        raise RuntimeError("the B3 smoke requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the B3 smoke requires exactly one allocated CUDA device")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and "H200" not in name:
        raise RuntimeError(f"the B3 smoke requires H100 or H200, found {name!r}")
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
    artifacts: ParentArtifacts,
    manifest_path: Path,
) -> dict[str, Any]:
    """Fail closed unless every bounded-smoke authority matches."""

    if mode != "smoke":
        raise RuntimeError("only the bounded B3 smoke is authorized")
    if manifest.get("protocol_status") != (
        "frozen_after_failed_B2_before_B3_implementation_smoke_or_training"
    ):
        raise RuntimeError("B3 protocol status differs")
    if manifest.get("development_run") != "85604":
        raise RuntimeError("B3 development run is not 85604")
    if manifest.get("held_out_85606_access_allowed") is not False:
        raise RuntimeError("B3 manifest unexpectedly permits held-out access")
    if manifest.get("full_training_authorized") is not False:
        raise RuntimeError("B3 smoke manifest unexpectedly authorizes full training")
    if "B3_FGN_H1_single_seed_bounded_GPU_smoke_85604" not in manifest.get(
        "authorized_scope", []
    ):
        raise RuntimeError("bounded B3 smoke is absent from authorized scope")
    smoke = manifest.get("implementation_gate", {}).get("gpu_smoke", {})
    expected_smoke = {
        "scientific_result": False,
        "accelerator": "one Rocky9 H100 or H200",
        "seed": 1701,
        "epochs": 2,
        "training_targets": [2, 18],
        "training_target_count": 16,
        "validation_targets": [498, 502],
        "validation_target_count": 4,
        "ensemble_members": 2,
        "optimizer_steps": 2,
        "wandb_online_required": True,
    }
    if smoke != expected_smoke or int(seed) != 1701:
        raise RuntimeError("B3 smoke identity or budget differs")

    data = manifest.get("data", {})
    if data.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]:
        raise RuntimeError("B3 field set differs")
    if data.get("input_channels") != "physically_valid_complete_C5P_state":
        raise RuntimeError("B3 input state differs")
    if data.get("zperiod") != 5 or data.get("mode_mapping") != "n=5k":
        raise RuntimeError("B3 toroidal-domain metadata differs")
    for flag in (
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "future_truth_input_allowed",
        "guard_frames_read_allowed",
    ):
        if data.get(flag) is not False:
            raise RuntimeError(f"B3 prohibited data flag {flag} differs")

    model = manifest.get("model", {})
    if model.get("arm") != "B3-FGN-H1" or model.get("context_frames") != 1:
        raise RuntimeError("B3 primary model arm differs")
    noise = manifest.get("functional_noise", {})
    if (
        noise.get("raw_dimension") != 32
        or noise.get("embedded_dimension") != 256
        or noise.get("noise_layers") != "all"
        or noise.get("spatial_semantics")
        != "one_global_vector_shared_across_all_tokens"
    ):
        raise RuntimeError("B3 functional-noise contract differs")

    parent = manifest.get("deterministic_parent", {})
    if str(artifacts.checkpoint_path) != str(parent.get("checkpoint_path")):
        raise RuntimeError("B3 deterministic parent path differs")
    if artifacts.checkpoint_sha256 != str(parent.get("checkpoint_sha256")):
        raise RuntimeError("B3 deterministic parent hash differs")
    codec = manifest.get("codec", {})
    if str(artifacts.codec_path) != str(codec.get("checkpoint_path")):
        raise RuntimeError("B3 codec path differs")
    if artifacts.codec_sha256 != str(codec.get("checkpoint_sha256")):
        raise RuntimeError("B3 codec hash differs")
    if str(artifacts.latent_normalization_path) != str(
        codec.get("latent_normalization_path")
    ):
        raise RuntimeError("B3 latent-normalization path differs")
    if artifacts.latent_normalization_sha256 != str(
        codec.get("latent_normalization_sha256")
    ):
        raise RuntimeError("B3 latent-normalization hash differs")

    if sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("B3 manifest bytes differ from the frozen manifest")
    protocol = manifest.get("protocol", {})
    protocol_path = ROOT / str(protocol.get("path", ""))
    if sha256_path(protocol_path) != str(protocol.get("sha256", "")):
        raise RuntimeError("B3 protocol file no longer matches the manifest")
    return {
        "authorized": True,
        "scope": "bounded_non_scientific_B3_FGN_H1_GPU_smoke",
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "full_B3_training_authorized": False,
        "seed": int(seed),
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
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": EXPECTED_MANIFEST_SHA256,
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

    manifest_path = args.b3_manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("B3 manifest must be inside the Paper 0 repository") from error
    prohibited_paths = (
        manifest_path,
        args.output,
        args.artifact_root,
        artifacts.checkpoint_path,
        artifacts.codec_path,
        artifacts.latent_normalization_path,
    )
    if any("85606" in str(path).lower() for path in prohibited_paths):
        raise ValueError("held-out paths are prohibited during the B3 smoke")
    manifest = load_strict_json(manifest_path)
    authorization = authorize_from_manifest(
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
        job_type="phase3_b3_fgn_smoke",
        tags=(
            "paper0",
            "phase3",
            "b3",
            "fgn",
            "smoke",
            "85604-only",
            "non-scientific",
        ),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_B3_FGN_H1_GPU_smoke",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "b3_manifest": {
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
        result = train_fgn_smoke(
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
