#!/usr/bin/env python3
"""Verify and reduce the three-seed old-85604 multi-lead result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from paper0.tools.train_codec_free_stage1_pilot import atomic_json
from paper0.tools.train_codec_free_stage2_multilead import (
    FIELDS,
    LEADS,
    SCALING_SCOPE,
    SCREEN_SCOPE,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


TASKS = ((1, 1702), (2, 1703))


def verify_artifact_index(index_path: Path, root: Path) -> dict[str, str]:
    """Verify every absolute path in one generated SHA-256 index."""

    index = index_path.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    records: dict[str, str] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, raw_path = line.split(maxsplit=1)
        path = Path(raw_path.strip()).resolve(strict=True)
        try:
            relative = str(path.relative_to(resolved_root))
        except ValueError as error:
            raise ValueError("artifact index leaves its task root") from error
        if relative in records or len(digest) != 64:
            raise ValueError("artifact index entry differs")
        if sha256_path(path) != digest:
            raise ValueError(f"artifact SHA-256 differs for {relative}")
        records[relative] = digest
    return records


def _task_root(array_root: Path, task: int, seed: int) -> Path:
    matches = tuple(
        sorted(array_root.glob(f"task_{task}_seed_{seed}_job_*"))
    )
    if len(matches) != 1 or not matches[0].is_dir():
        raise ValueError(f"seed {seed} task root is not unique")
    return matches[0]


def audit_confirmation_task(
    *,
    array_root: Path,
    task: int,
    seed: int,
    manifest_path: Path,
    manifest_sha256: str,
    training_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify one scaling task and return its immutable lock and result."""

    task_root = _task_root(array_root, task, seed).resolve(strict=True)
    run_root = (task_root / "run").resolve(strict=True)
    run_index = verify_artifact_index(
        run_root / "artifact_sha256.txt", run_root
    )
    expected_run_files = {
        *(f"checkpoint_epoch_{epoch:03d}.pt" for epoch in range(1, 5)),
        "derivative_rms.json",
        "parent_multilead_evaluation.json",
        "result.json",
        "wandb.json",
    }
    if set(run_index) != expected_run_files:
        raise ValueError("scaling run artifact inventory differs")
    task_index = verify_artifact_index(
        task_root / "artifact_sha256.txt", task_root
    )
    expected_task_files = {
        "command.sh",
        "environment.txt",
        "slurm_job.txt",
        "test_output.txt",
        "run/artifact_sha256.txt",
        "run/parent_multilead_evaluation.json",
        "run/result.json",
        "run/wandb.json",
    }
    if set(task_index) != expected_task_files:
        raise ValueError("scaling task artifact inventory differs")

    result = load_strict_json(run_root / "result.json")
    tracking = load_strict_json(run_root / "wandb.json")
    required = {
        "scope": SCALING_SCOPE,
        "status": "passed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": "c5p",
        "seed": seed,
        "paper0_commit": training_commit,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "training_pair_count": 2129,
        "validation_pair_count": 609,
        "epochs": 4,
        "optimizer_updates": 2132,
        "expected_optimizer_updates": 2132,
        "advance_to_three_seed_scaling": None,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"seed {seed} result field {key!r} differs")
    if result.get("training_gate", {}).get("passed") is not True:
        raise ValueError(f"seed {seed} training gate did not pass")
    if result.get("prospective_gate_passed") != result.get(
        "seed_confirmation_passed"
    ):
        raise ValueError(f"seed {seed} confirmation aliases differ")
    selected = result.get("best_checkpoint", {})
    selected_path = Path(str(selected.get("path", ""))).resolve(strict=True)
    if selected_path.parent != run_root:
        raise ValueError(f"seed {seed} checkpoint leaves run root")
    if selected.get("sha256") != run_index[selected_path.name]:
        raise ValueError(f"seed {seed} checkpoint hash differs")
    if (
        tracking.get("required") is not True
        or tracking.get("mode") != "online"
        or tracking.get("remote_state_after_finish") != "finished"
        or tracking.get("local_artifacts_are_scientific_authority") is not True
    ):
        raise ValueError(f"seed {seed} W&B completion differs")
    lock = {
        "task": task,
        "seed": seed,
        "task_root": str(task_root),
        "result": {
            "path": str(run_root / "result.json"),
            "sha256": run_index["result.json"],
        },
        "selected_checkpoint": {
            "path": str(selected_path),
            "sha256": selected["sha256"],
            "epoch": int(selected["epoch"]),
            "selection_metric": float(selected["selection_metric"]),
        },
        "wandb": {
            "path": str(run_root / "wandb.json"),
            "sha256": run_index["wandb.json"],
            "run_url": tracking["run_url"],
            "remote_state_after_finish": "finished",
        },
        "run_artifact_index": {
            "path": str(run_root / "artifact_sha256.txt"),
            "sha256": sha256_path(run_root / "artifact_sha256.txt"),
        },
        "task_artifact_index": {
            "path": str(task_root / "artifact_sha256.txt"),
            "sha256": sha256_path(task_root / "artifact_sha256.txt"),
        },
    }
    return lock, result


def summary_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("three finite seed values are required")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def reduce_scaling(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    seed1701_result_path: Path,
    seed1701_result_sha256: str,
    array_root: Path,
    training_commit: str,
) -> dict[str, Any]:
    """Apply the prospectively frozen all-seed confirmation rule."""

    for path in (manifest_path, seed1701_result_path, array_root):
        assert_development_path(path)
    if sha256_path(manifest_path) != manifest_sha256:
        raise ValueError("scaling manifest SHA-256 differs")
    manifest = load_strict_json(manifest_path)
    if (
        manifest.get("scope") != SCALING_SCOPE
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("new_nersc_data_access_allowed") is not False
        or manifest.get("all_seed_confirmation_required") is not True
        or manifest.get("conditional_bounded_rollout_authorized") is not True
        or manifest.get("paper0_commit_at_freeze") != training_commit
    ):
        raise ValueError("scaling manifest contract differs")
    if sha256_path(seed1701_result_path) != seed1701_result_sha256:
        raise ValueError("seed-1701 result SHA-256 differs")
    seed1701 = load_strict_json(seed1701_result_path)
    if (
        seed1701.get("scope") != SCREEN_SCOPE
        or int(seed1701.get("seed", -1)) != 1701
        or seed1701.get("advance_to_three_seed_scaling") is not True
        or seed1701.get("held_out_85606_read") is not False
    ):
        raise ValueError("seed-1701 screen result differs")

    locks: list[dict[str, Any]] = []
    results: dict[int, dict[str, Any]] = {1701: seed1701}
    for task, seed in TASKS:
        lock, result = audit_confirmation_task(
            array_root=array_root,
            task=task,
            seed=seed,
            manifest_path=manifest_path.resolve(strict=True),
            manifest_sha256=manifest_sha256,
            training_commit=training_commit,
        )
        locks.append(lock)
        results[seed] = result

    confirmations = {
        1701: bool(seed1701["advance_to_three_seed_scaling"]),
        1702: bool(results[1702]["seed_confirmation_passed"]),
        1703: bool(results[1703]["seed_confirmation_passed"]),
    }
    confirmed = all(confirmations.values())
    by_seed = {
        str(seed): {
            "selection_metric": float(
                result["best_checkpoint"]["selection_metric"]
            ),
            "lead1_shared_mse": float(result["lead1_shared_mse"]),
            "parent_improvement_fraction": float(
                result["parent_improvement_fraction"]
            ),
            "prospective_gate_passed": confirmations[seed],
            "per_lead": {
                str(lead): {
                    "shared_derivative_mse": float(
                        result["best_validation"]["per_lead"][str(lead)][
                            "shared_field_mean_model_derivative_mse"
                        ]
                    ),
                    "shared_persistence_relative_skill": float(
                        result["best_validation"]["per_lead"][str(lead)][
                            "shared_field_persistence_relative_skill"
                        ]
                    ),
                    "per_field_persistence_relative_skill": {
                        field: float(
                            result["best_validation"]["per_lead"][str(lead)][
                                "per_field"
                            ][field]["persistence_relative_skill"]
                        )
                        for field in FIELDS
                    },
                }
                for lead in LEADS
            },
        }
        for seed, result in sorted(results.items())
    }
    aggregates = {
        "selection_metric": summary_statistics(
            [by_seed[str(seed)]["selection_metric"] for seed in results]
        ),
        "lead1_shared_mse": summary_statistics(
            [by_seed[str(seed)]["lead1_shared_mse"] for seed in results]
        ),
        "shared_persistence_relative_skill_by_lead": {
            str(lead): summary_statistics(
                [
                    by_seed[str(seed)]["per_lead"][str(lead)][
                        "shared_persistence_relative_skill"
                    ]
                    for seed in results
                ]
            )
            for lead in LEADS
        },
    }
    return {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_stage2_multilead_scaling_reduction",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "physics_derived_loss_used": False,
        "training_commit": training_commit,
        "manifest": {
            "path": str(manifest_path.resolve(strict=True)),
            "sha256": manifest_sha256,
        },
        "seed1701_result": {
            "path": str(seed1701_result_path.resolve(strict=True)),
            "sha256": seed1701_result_sha256,
        },
        "new_run_locks": locks,
        "by_seed": by_seed,
        "aggregates": aggregates,
        "seed_confirmation_passed": {
            str(seed): value for seed, value in confirmations.items()
        },
        "three_seed_mechanism_confirmed": confirmed,
        "bounded_rollout_authorized": bool(
            confirmed and manifest["conditional_bounded_rollout_authorized"]
        ),
        "decision": (
            "freeze_bounded_direct_vs_autoregressive_validation"
            if confirmed
            else "stop_multilead_schedule_and_freeze_operator_experiment"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--seed1701-result", type=Path, required=True)
    parser.add_argument("--seed1701-result-sha256", required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_development_path(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = reduce_scaling(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        seed1701_result_path=args.seed1701_result,
        seed1701_result_sha256=args.seed1701_result_sha256,
        array_root=args.array_root,
        training_commit=args.training_commit,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
