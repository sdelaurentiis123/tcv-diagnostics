#!/usr/bin/env python3
"""Freeze all six completed old-85604 Stage-1 arms for block evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


FAMILIES = ("c5p", "e6b")
TASKS = ((0, 1701), (1, 1702), (2, 1703))


def _single_task_root(array_root: Path, task: int, seed: int) -> Path:
    matches = tuple(
        sorted(array_root.glob(f"task_{task}_seed_{seed}_job_*"))
    )
    if len(matches) != 1 or not matches[0].is_dir():
        raise ValueError(f"task {task} seed {seed} root is not unique")
    return matches[0]


def _lock_arm(
    *,
    task_root: Path,
    family: str,
    seed: int,
    training_commit: str,
) -> dict[str, Any]:
    result_path = task_root / family / "result.json"
    assert_development_path(result_path)
    result = load_strict_json(result_path)
    if result.get("scope") != "post_ecrd_old_85604_stage1_codec_free_full":
        raise ValueError("Stage-1 result scope differs")
    if result.get("development_run") != "85604":
        raise ValueError("Stage-1 result development run differs")
    if result.get("held_out_85606_read") is not False:
        raise ValueError("Stage-1 result held-out flag differs")
    if result.get("physics_derived_loss_used") is not False:
        raise ValueError("Stage-1 result physics-loss flag differs")
    if result.get("family") != family or int(result.get("seed", -1)) != seed:
        raise ValueError("Stage-1 result arm identity differs")
    if result.get("paper0_commit") != training_commit:
        raise ValueError("Stage-1 result training commit differs")
    if result.get("status") != "passed":
        raise ValueError("Stage-1 result did not pass")
    if result.get("training_gate", {}).get("passed") is not True:
        raise ValueError("Stage-1 training gate did not pass")
    tracking_path = task_root / family / "wandb.json"
    assert_development_path(tracking_path)
    tracking = load_strict_json(tracking_path)
    if tracking.get("required") is not True or tracking.get("mode") != "online":
        raise ValueError("Stage-1 W&B tracking contract differs")
    if tracking.get("remote_state_after_finish") != "finished":
        raise ValueError("Stage-1 W&B run is not remotely finished")

    selected = result.get("best_checkpoint", {})
    checkpoint_path = Path(str(selected.get("path", "")))
    checkpoint_sha256 = str(selected.get("sha256", ""))
    assert_development_path(checkpoint_path)
    expected_parent = (task_root / family).resolve()
    if checkpoint_path.resolve().parent != expected_parent:
        raise ValueError("selected checkpoint leaves its arm directory")
    if not checkpoint_sha256 or sha256_path(checkpoint_path) != checkpoint_sha256:
        raise ValueError("selected checkpoint SHA-256 differs")

    history = result.get("history", [])
    best = min(
        history,
        key=lambda record: record["validation"][
            "shared_field_mean_model_derivative_mse"
        ],
    )
    metric = float(
        best["validation"]["shared_field_mean_model_derivative_mse"]
    )
    if metric != float(selected.get("selection_metric")):
        raise ValueError("selected checkpoint metric differs from history")
    return {
        "family": family,
        "seed": seed,
        "training_result": {
            "path": str(result_path),
            "sha256": sha256_path(result_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "epoch": int(best["epoch"]),
            "selection_metric": metric,
        },
        "wandb": {
            "path": str(tracking_path),
            "sha256": sha256_path(tracking_path),
            "run_url": tracking.get("run_url"),
            "remote_path": tracking.get("remote_path"),
            "remote_state_after_finish": "finished",
        },
    }


def freeze_matrix(
    *,
    array_root: Path,
    array_job_id: str,
    training_commit: str,
    evaluation_commit: str,
) -> dict[str, Any]:
    assert_development_path(array_root)
    arms: list[dict[str, Any]] = []
    task_records: list[dict[str, Any]] = []
    for task, seed in TASKS:
        task_root = _single_task_root(array_root, task, seed)
        task_manifest = task_root / "artifact_sha256.txt"
        if not task_manifest.is_file():
            raise ValueError("task artifact SHA-256 manifest is absent")
        task_records.append(
            {
                "task": task,
                "seed": seed,
                "path": str(task_root),
                "artifact_sha256_manifest": {
                    "path": str(task_manifest),
                    "sha256": sha256_path(task_manifest),
                },
            }
        )
        for family in FAMILIES:
            arms.append(
                _lock_arm(
                    task_root=task_root,
                    family=family,
                    seed=seed,
                    training_commit=training_commit,
                )
            )

    identities = {(arm["family"], arm["seed"]) for arm in arms}
    expected = {(family, seed) for family in FAMILIES for _, seed in TASKS}
    if identities != expected or len(arms) != 6:
        raise AssertionError("frozen matrix does not contain exactly six arms")
    return {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_stage1_block_evaluation_input_freeze",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "physics_derived_loss_used": False,
        "training_array_job_id": str(array_job_id),
        "training_commit": training_commit,
        "evaluation_commit": evaluation_commit,
        "array_root": str(array_root),
        "tasks": task_records,
        "blocks": {
            "V00": {"target_interval": [498, 540], "target_count": 42},
            "V01": {"target_interval": [540, 582], "target_count": 42},
            "V02": {"target_interval": [582, 624], "target_count": 42},
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    frozen = freeze_matrix(
        array_root=args.array_root,
        array_job_id=args.array_job_id,
        training_commit=args.training_commit,
        evaluation_commit=args.evaluation_commit,
    )
    atomic_json(args.output, frozen)
    print(json.dumps(frozen, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
