#!/usr/bin/env python3
"""Recover missing terminal metadata for one already-complete ECRD run.

This is deliberately narrower than training or evaluation.  It audits the
immutable ECRD-History seed-1701 artifacts produced by Slurm task 6913340_8,
verifies the corresponding finished online W&B run, and creates only the two
terminal metadata files that the original post-training propagation race did
not create.  It neither loads model-development data nor generates forecasts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from paper0.tools.summarize_ecrd_full_training import (
    audit_result_record,
    expected_artifact_relatives,
    verify_artifact_index,
)
from tcv_diagnostics.b5_residual_edm_full_training import (
    full_training_order,
    full_validation_seed_bank,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_training import (
    ECRDTrainingConfig,
    frozen_parameter_counts,
    model_config_record,
)
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ARRAY_JOB_ID = "6913340"
SOURCE_ARRAY_TASK_ID = 8
SOURCE_SLURM_JOB_ID = "6913340_8"
SOURCE_TRAINING_COMMIT = "d822ee2147a98713f1b2ecdfd0f5a4077eded062"
ARM = "ECRD-History"
SAFE_ARM = "ecrd_history"
SEED = 1701
EXPECTED_MODEL_ROOT = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_full_training/"
    "job_6913340/task_8_ecrd_history_seed_1701/model"
)
WANDB_ENTITY = "sdelaurentiis123-columbia-university"
WANDB_PROJECT = "tcv-diagnostics-paper0"
WANDB_GROUP = "ecrd-model-development"
WANDB_RUN_ID = "p0ecrdfull-6913340-8-s1701"
WANDB_RUN_NAME = "ecrd-full-ecrd_history-s1701-j6913340-8"
WANDB_JOB_TYPE = "ecrd_full_training"
WANDB_TAGS = (
    "paper0",
    "ecrd",
    "full",
    SAFE_ARM,
    "85604-only",
    f"seed{SEED}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
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


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_or_verify_json(path: Path, value: Mapping[str, Any]) -> str:
    """Create one immutable JSON file, or verify an identical prior recovery."""

    serialized = _canonical_json(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to replace differing metadata {path}")
    else:
        write_strict_json_atomic(path, value)
    return sha256_path(path)


def write_or_verify_text(path: Path, text: str) -> str:
    """Create one immutable text file atomically, or require byte identity."""

    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace differing metadata {path}")
        return sha256_path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_path(path)


def expected_source_paths(model_root: Path) -> tuple[Path, ...]:
    """The pre-recovery inventory; terminal metadata is intentionally absent."""

    return tuple(
        model_root / relative
        for relative in expected_artifact_relatives()
        if relative not in (Path("wandb.json"),)
    )


def _require_artifact(
    *,
    model_root: Path,
    record: Mapping[str, Any],
    relative: Path,
) -> str:
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    expected_path = (model_root / relative).resolve(strict=True)
    if path != expected_path:
        raise RuntimeError(f"artifact path differs for {relative}: {path}")
    expected_digest = str(record.get("sha256", ""))
    if len(expected_digest) != 64:
        raise RuntimeError(f"artifact digest is malformed for {relative}")
    observed = sha256_path(path)
    if observed != expected_digest:
        raise RuntimeError(f"artifact digest differs for {relative}")
    return observed


def audit_existing_training(model_root: Path) -> tuple[dict[str, Any], dict[Path, str]]:
    """Verify the complete source budget and every immutable training artifact."""

    result_path = (model_root / "result.json").resolve(strict=True)
    result = load_strict_json(result_path)
    placeholder_tracking = {
        "mode": "online",
        "remote_state_after_finish": "finished",
        "remote_presence_verified_after_finish": True,
    }
    audit_result_record(
        result,
        placeholder_tracking,
        arm=ARM,
        seed=SEED,
        training_commit=SOURCE_TRAINING_COMMIT,
        expected_slurm_job_id=SOURCE_SLURM_JOB_ID,
    )

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("source result has no artifact map")
    hashes: dict[Path, str] = {}
    for name, relative in (
        ("config", Path("config.json")),
        ("training_order", Path("training_order.npy")),
        ("validation_seed_bank", Path("validation_seed_bank.npy")),
        ("history", Path("history.jsonl")),
    ):
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"source result artifact {name!r} differs")
        hashes[relative] = _require_artifact(
            model_root=model_root,
            record=record,
            relative=relative,
        )

    candidates = artifacts.get("candidate_checkpoints")
    if not isinstance(candidates, list) or len(candidates) != 20:
        raise RuntimeError("source candidate inventory differs")
    expected_epochs = list(range(5, 101, 5))
    for record, epoch in zip(candidates, expected_epochs, strict=True):
        if not isinstance(record, Mapping) or record.get("completed_epoch") != epoch:
            raise RuntimeError("source candidate epoch inventory differs")
        relative = Path(f"candidates/ema_epoch_{epoch:03d}.pt")
        hashes[relative] = _require_artifact(
            model_root=model_root,
            record=record,
            relative=relative,
        )

    selected_record = artifacts.get("selected_checkpoint")
    if not isinstance(selected_record, Mapping):
        raise RuntimeError("source result selected-checkpoint artifact differs")
    hashes[Path("selected.pt")] = _require_artifact(
        model_root=model_root,
        record=selected_record,
        relative=Path("selected.pt"),
    )

    observed_order = np.load(model_root / "training_order.npy", allow_pickle=False)
    if not np.array_equal(observed_order, full_training_order()):
        raise RuntimeError("source paired training order differs from frozen order")
    observed_bank = np.load(model_root / "validation_seed_bank.npy", allow_pickle=False)
    if not np.array_equal(observed_bank, full_validation_seed_bank()):
        raise RuntimeError("source validation seed bank differs from frozen bank")

    history_lines = (model_root / "history.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if len(history_lines) != 100:
        raise RuntimeError("source history does not contain 100 completed epochs")
    history = [json.loads(line) for line in history_lines]
    if [item.get("completed_epoch") for item in history] != list(range(1, 101)):
        raise RuntimeError("source history epoch sequence differs")
    if history[-1].get("global_optimizer_step") != 10_800:
        raise RuntimeError("source history optimizer-step budget differs")
    if [
        item.get("completed_epoch")
        for item in history
        if item.get("validation_candidate") is True
    ] != expected_epochs:
        raise RuntimeError("source history validation-candidate schedule differs")

    try:
        selected = torch.load(
            model_root / "selected.pt",
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # Compatibility with the pinned pre-weights_only API.
        selected = torch.load(model_root / "selected.pt", map_location="cpu")
    selected_required = {
        "schema_version": 1,
        "kind": "ECRD_selected_EMA_checkpoint",
        "paper0_commit": SOURCE_TRAINING_COMMIT,
        "slurm_job_id": SOURCE_SLURM_JOB_ID,
        "training": ECRDTrainingConfig(arm=ARM, seed=SEED, mode="full").to_record(),
        "model": model_config_record(ARM),
        "parameter_count": frozen_parameter_counts()[ARM],
        "selected_completed_epoch": result["selected_completed_epoch"],
        "selected_validation": result["selected_validation"],
        "physics_metric_used_for_selection": False,
        "held_out_85606_read": False,
    }
    for name, expected in selected_required.items():
        if selected.get(name) != expected:
            raise RuntimeError(f"selected checkpoint metadata {name!r} differs")
    source_candidate = selected.get("source_candidate", {})
    selected_epoch = int(result["selected_completed_epoch"])
    candidate_relative = Path(f"candidates/ema_epoch_{selected_epoch:03d}.pt")
    if (
        Path(str(source_candidate.get("path", ""))).resolve(strict=True)
        != (model_root / candidate_relative).resolve(strict=True)
        or source_candidate.get("sha256") != hashes[candidate_relative]
    ):
        raise RuntimeError("selected checkpoint source-candidate link differs")

    hashes[Path("result.json")] = sha256_path(result_path)
    expected_without_tracking = tuple(
        relative
        for relative in expected_artifact_relatives()
        if relative != Path("wandb.json")
    )
    if tuple(hashes) != expected_without_tracking:
        raise RuntimeError("source training artifact inventory ordering differs")
    return result, hashes


def build_tracking_record(*, module: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the live run and reproduce the original terminal tracking schema."""

    api = module.Api(timeout=30)
    if not bool(getattr(api, "api_key", None)):
        raise RuntimeError("online W&B verification requires an API key")
    viewer = api.viewer
    if str(getattr(viewer, "entity", "")) != WANDB_ENTITY:
        raise RuntimeError("authenticated W&B entity differs")
    username = str(getattr(viewer, "username", ""))
    remote_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_RUN_ID}"
    remote = api.run(remote_path)
    if str(remote.id) != WANDB_RUN_ID or str(remote.state) != "finished":
        raise RuntimeError("source W&B run is not the required finished run")
    summary = dict(remote.summary)
    expected_summary = {
        "final/training_completed": True,
        "final/completed_epochs": 100,
        "final/completed_optimizer_steps": 10_800,
        "final/candidate_count": 20,
        "final/selected_completed_epoch": int(result["selected_completed_epoch"]),
        "final/selected_validation_objective": float(
            result["selected_validation"]["checkpoint_score"]
        ),
        "final/checkpoint_reload_bitwise_exact": True,
        "compute/parameter_count": frozen_parameter_counts()[ARM],
        "provenance/paper0_commit": SOURCE_TRAINING_COMMIT,
        "provenance/selected_checkpoint_sha256": result["artifacts"][
            "selected_checkpoint"
        ]["sha256"],
        "provenance/history_sha256": result["artifacts"]["history"]["sha256"],
        "scope/physics_derived_loss_used": False,
        "scope/held_out_85606_read": False,
        "scope/scientific_forecast_generated": False,
    }
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            raise RuntimeError(f"source W&B summary field {name!r} differs")
    run_url = str(remote.url)
    if not run_url.startswith("https://"):
        raise RuntimeError("source W&B run has no online URL")
    return {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "run_id": WANDB_RUN_ID,
            "run_name": WANDB_RUN_NAME,
            "job_type": WANDB_JOB_TYPE,
            "tags": list(WANDB_TAGS),
        },
        "authenticated_username": username,
        "wandb_version": str(module.__version__),
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": "finished",
        "checkpoints_uploaded": False,
        "samples_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
        "epochs_logged": 100,
    }


def artifact_index_text(model_root: Path) -> str:
    lines = []
    for relative in expected_artifact_relatives():
        path = (model_root / relative).resolve(strict=True)
        lines.append(f"{sha256_path(path)}  {path}\n")
    return "".join(lines)


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    if len(str(args.paper0_commit)) != 40:
        raise ValueError("Paper 0 recovery commit identity differs")
    assert_development_path(args.model_root)
    assert_development_path(args.output)
    model_root = Path(args.model_root).resolve(strict=True)
    if model_root != EXPECTED_MODEL_ROOT.resolve(strict=True):
        raise RuntimeError(f"recovery source differs: {model_root}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    result, pre_recovery_hashes = audit_existing_training(model_root)
    import wandb  # Imported only after the immutable local audit succeeds.

    tracking = build_tracking_record(module=wandb, result=result)
    audit_result_record(
        result,
        tracking,
        arm=ARM,
        seed=SEED,
        training_commit=SOURCE_TRAINING_COMMIT,
        expected_slurm_job_id=SOURCE_SLURM_JOB_ID,
    )
    wandb_path = model_root / "wandb.json"
    wandb_sha = write_or_verify_json(wandb_path, tracking)
    index_path = model_root / "artifact_sha256.txt"
    index_sha = write_or_verify_text(index_path, artifact_index_text(model_root))
    records = verify_artifact_index(index_path, model_root)
    if records[Path("wandb.json")] != wandb_sha:
        raise RuntimeError("recovered W&B metadata hash differs from final index")

    output.mkdir(parents=True)
    recovery = {
        "schema_version": 1,
        "scope": "ECRD_terminal_metadata_recovery_85604",
        "status": "completed_training_artifacts_and_remote_WandB_verified",
        "paper0_commit": str(args.paper0_commit),
        "slurm_job_id": str(args.slurm_job_id),
        "source_training_commit": SOURCE_TRAINING_COMMIT,
        "source_array_job_id": SOURCE_ARRAY_JOB_ID,
        "source_array_task_id": SOURCE_ARRAY_TASK_ID,
        "source_slurm_job_id": SOURCE_SLURM_JOB_ID,
        "source_arm": ARM,
        "source_seed": SEED,
        "source_model_root": str(model_root),
        "source_result": {
            "path": str((model_root / "result.json").resolve(strict=True)),
            "sha256": pre_recovery_hashes[Path("result.json")],
        },
        "recovered_metadata": {
            "wandb": {"path": str(wandb_path.resolve(strict=True)), "sha256": wandb_sha},
            "artifact_index": {
                "path": str(index_path.resolve(strict=True)),
                "sha256": index_sha,
                "verified_artifact_count": len(records),
            },
        },
        "verified_completed_epochs": int(result["completed_epochs"]),
        "verified_completed_optimizer_steps": int(
            result["completed_optimizer_steps"]
        ),
        "verified_candidate_count": int(result["candidate_count"]),
        "verified_selected_epoch": int(result["selected_completed_epoch"]),
        "remote_wandb_state": tracking["remote_state_after_finish"],
        "training_rerun": False,
        "checkpoint_bytes_modified": False,
        "training_data_loaded": False,
        "validation_data_loaded": False,
        "physics_metric_evaluated": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    recovery_path = output / "result.json"
    write_strict_json_atomic(recovery_path, recovery)
    recovery_index = output / "artifact_sha256.txt"
    write_or_verify_text(
        recovery_index,
        f"{sha256_path(recovery_path)}  {recovery_path.resolve(strict=True)}\n",
    )
    print(json.dumps(recovery, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
