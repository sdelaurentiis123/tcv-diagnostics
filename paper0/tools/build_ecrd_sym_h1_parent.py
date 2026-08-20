#!/usr/bin/env python3
"""Generate truth-free four-phase H1 parent streams for ECRD on 85604."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch

from tcv_diagnostics.b5_residual_forecast import B5TrainingContextDataset
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_data import (
    ECRD_TRAIN_TARGETS,
    ECRD_VALIDATION_TARGETS,
    generate_symmetrized_h1_parent,
)
from tcv_diagnostics.ecrd_wandb_tracking import ECRDParentOnlineWandbTracker
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.model_training_data import (
    OFFICIAL_ARTIFACT_ROOT,
    load_official_catalog,
)
from tcv_diagnostics.o2_context_data import OneStepContextDataset
from tcv_diagnostics.o2_forecast import load_selected_o2_model
from tcv_diagnostics.wandb_tracking import WandbRunSpec


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MANIFEST_SHA256 = (
    "6fa7c02499cb94dac13d29d797b3f06693d2b07a922e0812a033079ea7185fa5"
)
EXPECTED_PROTOCOL_SHA256 = (
    "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4"
)
EXPECTED_H1_SHA256 = (
    "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
)
EXPECTED_CODEC_SHA256 = (
    "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
)
EXPECTED_H1_TRAINING_COMMIT = "9035bc3ce9d2351cd17586f4429af8116d43a47e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=OFFICIAL_ARTIFACT_ROOT)
    parser.add_argument("--h1-checkpoint", type=Path, required=True)
    parser.add_argument("--h1-checkpoint-sha256", required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint-sha256", required=True)
    parser.add_argument("--h1-training-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
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
    if release.get("ID") != "rocky" or release.get("VERSION_ID", "").split(".")[0] != "9":
        raise RuntimeError("ECRD parent generation requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ECRD parent generation requires exactly one CUDA GPU")
    accelerator = torch.cuda.get_device_name(0)
    if "H100" not in accelerator:
        raise RuntimeError(f"ECRD parent generation requires H100, found {accelerator!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the allocated H100 does not report bfloat16 support")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "accelerator": accelerator,
        "cuda_device_count": torch.cuda.device_count(),
        "bfloat16_supported": True,
    }


def verify_input(path: Path, expected_sha256: str, label: str) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = sha256_path(source)
    if observed != str(expected_sha256):
        raise RuntimeError(f"{label} SHA-256 differs: {observed} != {expected_sha256}")
    return source


def authorize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_root: Path,
    h1_checkpoint: Path,
    h1_sha256: str,
    codec_checkpoint: Path,
    codec_sha256: str,
    h1_training_commit: str,
) -> dict[str, Any]:
    if sha256_path(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("ECRD parent manifest bytes differ")
    if (
        manifest.get("status") != "frozen_before_truth_free_parent_generation"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("ECRD parent manifest scope differs")
    protocol = manifest.get("protocol", {})
    if (
        protocol.get("path") != "paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md"
        or protocol.get("sha256") != EXPECTED_PROTOCOL_SHA256
        or sha256_path(ROOT / protocol["path"]) != EXPECTED_PROTOCOL_SHA256
    ):
        raise RuntimeError("ECRD protocol identity differs")
    required_scope = {
        "load_frozen_C5P_H1_seed1701",
        "read_85604_training_and_validation_context_only",
        "generate_four_phase_symmetrized_H1_parent_means",
    }
    if set(manifest.get("authorized_scope", ())) != required_scope:
        raise RuntimeError("ECRD parent authorization differs")
    locks = manifest.get("evidence_locks", {})
    model_data = locks.get("model_dataset", {})
    if Path(model_data.get("root", "")) != Path(
        "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/"
        "phase2_model_dataset/job_6893525"
    ):
        raise RuntimeError("ECRD model-data source identity differs")
    h1 = locks.get("H1_checkpoint", {})
    codec = locks.get("codec_checkpoint", {})
    if (
        h1_sha256 != EXPECTED_H1_SHA256
        or h1.get("sha256") != h1_sha256
        or Path(h1.get("path", "")) != h1_checkpoint
        or h1.get("training_commit") != h1_training_commit
        or h1_training_commit != EXPECTED_H1_TRAINING_COMMIT
        or h1.get("arm") != "C5P-H1"
        or h1.get("seed") != 1701
        or h1.get("trainable") is not False
    ):
        raise RuntimeError("frozen H1 parent identity differs")
    if (
        codec_sha256 != EXPECTED_CODEC_SHA256
        or codec.get("sha256") != codec_sha256
        or Path(codec.get("path", "")) != codec_checkpoint
        or codec.get("trainable") is not False
    ):
        raise RuntimeError("frozen H1 codec identity differs")
    data = manifest.get("data", {})
    if (
        data.get("training_targets") != [2, 432]
        or data.get("validation_targets") != [498, 624]
        or data.get("guard_frames") != [432, 496]
        or data.get("target_truth_read_allowed") is not False
        or data.get("periodic_axes_xyz") != [False, False, True]
        or data.get("zperiod") != 5
        or data.get("mode_mapping") != "n=5k"
    ):
        raise RuntimeError("ECRD parent data boundary differs")
    sym = manifest.get("symmetrization", {})
    if (
        sym.get("phase_shifts") != [0, 1, 2, 3]
        or sym.get("definition") != "mean_q0_to_3_T_minus_q_H1_T_q"
        or sym.get("future_truth_used") is not False
    ):
        raise RuntimeError("ECRD H1 symmetrization differs")
    return {
        "authorized": True,
        "scope": "truth_free_four_phase_H1_parent_generation_85604",
        "development_run": "85604",
        "target_truth_read": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    environment = require_rocky9_h100()
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", 0)

    manifest_path = args.manifest.resolve()
    try:
        manifest_relative = manifest_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("ECRD parent manifest must be inside the repository") from error
    runtime_paths = (
        args.artifact_root,
        args.h1_checkpoint,
        args.codec_checkpoint,
        args.output,
    )
    if any("85606" in str(path).lower() for path in runtime_paths):
        raise ValueError("held-out paths are prohibited during ECRD parent generation")
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = load_strict_json(manifest_path)
    authorization = authorize_manifest(
        manifest,
        manifest_path=manifest_path,
        artifact_root=args.artifact_root,
        h1_checkpoint=args.h1_checkpoint,
        h1_sha256=args.h1_checkpoint_sha256,
        codec_checkpoint=args.codec_checkpoint,
        codec_sha256=args.codec_checkpoint_sha256,
        h1_training_commit=args.h1_training_commit,
    )
    locks = manifest["evidence_locks"]
    for filename, key in (
        ("model_dataset_manifest.json", "manifest_sha256"),
        ("normalization.json", "normalization_sha256"),
        ("artifact_sha256.txt", "artifact_index_sha256"),
    ):
        verify_input(
            args.artifact_root / filename,
            locks["model_dataset"][key],
            f"ECRD model-data {filename}",
        )
    h1_path = verify_input(
        args.h1_checkpoint, args.h1_checkpoint_sha256, "ECRD frozen H1"
    )
    codec_path = verify_input(
        args.codec_checkpoint, args.codec_checkpoint_sha256, "ECRD frozen codec"
    )
    catalog = load_official_catalog(args.artifact_root)
    model = load_selected_o2_model(
        checkpoint=h1_path,
        expected_checkpoint_sha256=args.h1_checkpoint_sha256,
        codec_checkpoint=codec_path,
        expected_codec_sha256=args.codec_checkpoint_sha256,
        arm="C5P-H1",
        seed=1701,
        training_commit=args.h1_training_commit,
        device=device,
    )
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="ecrd_symmetrized_h1_parent_generation",
        tags=("paper0", "ecrd", "h1-parent", "85604-only", "truth-free"),
    )
    tracking_config = {
        "schema_version": 1,
        "scope": authorization["scope"],
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "authorization": authorization,
        "manifest": {
            "path": str(manifest_relative),
            "sha256": sha256_path(manifest_path),
        },
        "inputs": {
            "model_data_root": str(args.artifact_root),
            "H1_checkpoint_sha256": sha256_path(h1_path),
            "codec_checkpoint_sha256": sha256_path(codec_path),
            "development_run": "85604",
            "target_truth_read": False,
            "held_out_85606_read": False,
        },
        "symmetrization": manifest["symmetrization"],
        "environment": environment,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ecrd_data_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/ecrd_data.py"
            ),
            "ecrd_model_sha256": sha256_path(
                ROOT / "src/tcv_diagnostics/models/ecrd.py"
            ),
            "entrypoint_sha256": sha256_path(Path(__file__).resolve()),
        },
    }
    tracker = ECRDParentOnlineWandbTracker.start(
        spec=spec,
        config=tracking_config,
        tracking_directory=args.output.parent / f".{args.output.name}.wandb",
    )
    train_context = None
    validation_context = None
    try:
        train_context = B5TrainingContextDataset(
            catalog, target_frames=ECRD_TRAIN_TARGETS, context_frames=1
        )
        validation_context = OneStepContextDataset(
            catalog,
            target_frames=ECRD_VALIDATION_TARGETS,
            context_frames=1,
            return_physical=False,
        )
        metadata = {
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": str(args.slurm_job_id),
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "H1_checkpoint": {"path": str(h1_path), "sha256": sha256_path(h1_path)},
            "codec_checkpoint": {
                "path": str(codec_path),
                "sha256": sha256_path(codec_path),
            },
            "target_truth_read": False,
            "guard_frames_read": False,
            "held_out_85606_read": False,
        }
        train = generate_symmetrized_h1_parent(
            model=model,
            dataset=train_context,
            output=args.output / "sym_h1_train.h5",
            metadata=metadata,
            device=device,
        )
        tracker.log_split(train)
        validation = generate_symmetrized_h1_parent(
            model=model,
            dataset=validation_context,
            output=args.output / "sym_h1_validation.h5",
            metadata=metadata,
            device=device,
        )
        tracker.log_split(validation)
        result = {
            "schema_version": 1,
            "scope": authorization["scope"],
            "status": "completed",
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": str(args.slurm_job_id),
            "development_run": "85604",
            "authorization": authorization,
            "environment": environment,
            "splits": {"train": train, "validation": validation},
            "target_truth_read": False,
            "guard_frames_read": False,
            "held_out_85606_read": False,
            "training_performed": False,
            "physics_metric_evaluated": False,
            "assimilation_performed": False,
        }
        write_strict_json_atomic(args.output / "result.json", result)
        tracking = tracker.finish_success(result)
        write_strict_json_atomic(args.output / "wandb.json", tracking)
    except BaseException:
        tracker.finish_failure()
        raise
    finally:
        if train_context is not None:
            train_context.close()
        if validation_context is not None:
            validation_context.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
