#!/usr/bin/env python3
"""Verify and summarize the eleven new ECRD full-training tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

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
NEW_TASKS = (
    (0, "B5", "b5", 1702),
    (1, "B5", "b5", 1703),
    (2, "B5-Context", "b5_context", 1701),
    (3, "B5-Context", "b5_context", 1702),
    (4, "B5-Context", "b5_context", 1703),
    (5, "ECRD", "ecrd", 1701),
    (6, "ECRD", "ecrd", 1702),
    (7, "ECRD", "ecrd", 1703),
    (8, "ECRD-History", "ecrd_history", 1701),
    (9, "ECRD-History", "ecrd_history", 1702),
    (10, "ECRD-History", "ecrd_history", 1703),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--training-commit", required=True)
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
    if actual != str(expected_commit):
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


def expected_artifact_relatives() -> tuple[Path, ...]:
    candidates = tuple(
        Path(f"candidates/ema_epoch_{epoch:03d}.pt") for epoch in range(5, 101, 5)
    )
    return (
        Path("config.json"),
        Path("training_order.npy"),
        Path("validation_seed_bank.npy"),
        Path("history.jsonl"),
        *candidates,
        Path("selected.pt"),
        Path("result.json"),
        Path("wandb.json"),
    )


def verify_artifact_index(index_path: Path, run_root: Path) -> dict[Path, str]:
    index = Path(index_path).resolve(strict=True)
    root = Path(run_root).resolve(strict=True)
    records: dict[Path, str] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, raw_path = line.split(maxsplit=1)
        path = Path(raw_path.strip()).resolve(strict=True)
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("ECRD artifact index escapes its run root") from error
        if relative in records or len(digest) != 64:
            raise RuntimeError("ECRD artifact index entry differs")
        observed = sha256_path(path)
        if observed != digest:
            raise RuntimeError(f"ECRD artifact hash differs for {relative}")
        records[relative] = digest
    if tuple(records) != expected_artifact_relatives():
        raise RuntimeError("ECRD artifact index inventory differs")
    return records


def audit_result_record(
    result: Mapping[str, Any],
    tracking: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    training_commit: str,
    expected_slurm_job_id: str,
) -> None:
    config = ECRDTrainingConfig(arm=arm, seed=seed, mode="full")
    required = {
        "scope": "ECRD_matched_model_development_training_85604",
        "status": "training_completed_checkpoint_selected",
        "mode": "full",
        "arm": arm,
        "seed": seed,
        "paper0_commit": training_commit,
        "slurm_job_id": expected_slurm_job_id,
        "development_run": "85604",
        "training": json.loads(json.dumps(config.to_record())),
        "model": model_config_record(arm),
        "parameter_count": frozen_parameter_counts()[arm],
        "completed_epochs": 100,
        "completed_optimizer_steps": 10_800,
        "target_presentations": 43_000,
        "candidate_count": 20,
        "checkpoint_reload_bitwise_exact": True,
        "training_performed": True,
        "validation_frames_read": True,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    for name, expected in required.items():
        if result.get(name) != expected:
            raise RuntimeError(f"{arm} seed {seed} result field {name!r} differs")
    candidates = result.get("artifacts", {}).get("candidate_checkpoints", ())
    if [item.get("completed_epoch") for item in candidates] != list(range(5, 101, 5)):
        raise RuntimeError(f"{arm} seed {seed} candidate epochs differ")
    if (
        tracking.get("mode") != "online"
        or tracking.get("remote_state_after_finish") != "finished"
        or tracking.get("remote_presence_verified_after_finish") is not True
    ):
        raise RuntimeError(f"{arm} seed {seed} W&B completion differs")


def audit_task(
    *,
    array_root: Path,
    array_job_id: str,
    index: int,
    arm: str,
    safe_arm: str,
    seed: int,
    training_commit: str,
) -> dict[str, Any]:
    task_root = (
        Path(array_root) / f"task_{index}_{safe_arm}_seed_{seed}"
    ).resolve(strict=True)
    run_root = (task_root / "model").resolve(strict=True)
    records = verify_artifact_index(run_root / "artifact_sha256.txt", run_root)
    result = load_strict_json(run_root / "result.json")
    tracking = load_strict_json(run_root / "wandb.json")
    audit_result_record(
        result,
        tracking,
        arm=arm,
        seed=seed,
        training_commit=training_commit,
        expected_slurm_job_id=f"{array_job_id}_{index}",
    )
    selected = result["artifacts"]["selected_checkpoint"]
    selected_path = Path(selected["path"]).resolve(strict=True)
    if (
        selected_path != (run_root / "selected.pt").resolve(strict=True)
        or selected["sha256"] != records[Path("selected.pt")]
    ):
        raise RuntimeError(f"{arm} seed {seed} selected checkpoint differs")
    return {
        "array_index": index,
        "arm": arm,
        "seed": seed,
        "result": {
            "path": str((run_root / "result.json").resolve(strict=True)),
            "sha256": records[Path("result.json")],
        },
        "selected_checkpoint": {
            "path": str(selected_path),
            "sha256": selected["sha256"],
            "completed_epoch": result["selected_completed_epoch"],
            "validation": result["selected_validation"],
        },
        "artifact_index": {
            "path": str((run_root / "artifact_sha256.txt").resolve(strict=True)),
            "sha256": sha256_path(run_root / "artifact_sha256.txt"),
            "verified_artifact_count": len(records),
        },
        "training_order_sha256": records[Path("training_order.npy")],
        "validation_seed_bank_sha256": records[Path("validation_seed_bank.npy")],
        "wall_seconds": result["wall_seconds"],
        "peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
        "wandb": {
            "run_url": tracking["run_url"],
            "remote_state_after_finish": tracking["remote_state_after_finish"],
            "sha256": records[Path("wandb.json")],
        },
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    paths = (args.array_root, args.manifest, args.output)
    if any("85606" in str(path).lower() for path in paths):
        raise ValueError("held-out paths are prohibited during training finalization")
    if len(args.training_commit) != 40 or len(args.paper0_commit) != 40:
        raise ValueError("ECRD commit identity differs")
    array_root = Path(args.array_root).resolve(strict=True)
    output = Path(args.output)
    assert_development_path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_path = Path(args.manifest).resolve(strict=True)
    if sha256_path(manifest_path) != args.manifest_sha256:
        raise RuntimeError("ECRD full-training manifest SHA-256 differs")
    manifest = load_strict_json(manifest_path)
    expected_matrix = [
        {"array_index": index, "arm": arm, "seed": seed}
        for index, arm, _, seed in NEW_TASKS
    ]
    if (
        manifest.get("status")
        != "frozen_after_passing_ECRD_smoke_before_full_training"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("new_training_matrix") != expected_matrix
        or manifest.get("full_training_authorized") is not True
    ):
        raise RuntimeError("ECRD full-training manifest scope differs")
    runs = [
        audit_task(
            array_root=array_root,
            array_job_id=args.array_job_id,
            index=index,
            arm=arm,
            safe_arm=safe_arm,
            seed=seed,
            training_commit=args.training_commit,
        )
        for index, arm, safe_arm, seed in NEW_TASKS
    ]
    if len({run["training_order_sha256"] for run in runs}) != 1:
        raise RuntimeError("ECRD paired training-order hashes differ")
    if len({run["validation_seed_bank_sha256"] for run in runs}) != 1:
        raise RuntimeError("ECRD paired validation-bank hashes differ")
    output.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "scope": "ECRD_matched_full_training_finalization_85604",
        "status": "all_eleven_new_training_tasks_verified",
        "paper0_commit": args.paper0_commit,
        "training_commit": args.training_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "array_job_id": str(args.array_job_id),
        "development_run": "85604",
        "manifest": {
            "path": str(manifest_path),
            "sha256": args.manifest_sha256,
        },
        "new_runs": runs,
        "historical_B5_seed1701": manifest["historical_B5_seed1701"],
        "total_ladder_runs": 12,
        "all_training_artifact_indices_verified": True,
        "all_wandb_runs_finished": True,
        "paired_training_order_verified": True,
        "paired_validation_seed_bank_verified": True,
        "scientific_result": False,
        "physics_metric_evaluated": False,
        "target_truth_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    result_path = output / "result.json"
    write_strict_json_atomic(result_path, result)
    (output / "artifact_sha256.txt").write_text(
        f"{sha256_path(result_path)}  {result_path.resolve(strict=True)}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
