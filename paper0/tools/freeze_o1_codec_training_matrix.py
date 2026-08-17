#!/usr/bin/env python3
"""Freeze six completed R1 codec runs before scientific O1 evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import CodecRunConfig, sha256_path  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


RUNS = (
    (0, "c5p", 1701),
    (1, "e6b", 1701),
    (2, "c5p", 1702),
    (3, "e6b", 1702),
    (4, "c5p", 1703),
    (5, "e6b", 1703),
)
ARTIFACTS = (
    "config.json",
    "history.jsonl",
    "result.json",
    "selected.pt",
    "final_training_state.pt",
    "wandb.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--training-slurm-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--audit-slurm-job-id", required=True)
    return parser.parse_args()


def _artifact_index(path: Path) -> dict[Path, str]:
    records: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        artifact = Path(name.strip()).resolve(strict=True)
        if artifact in records:
            raise ValueError(f"duplicate artifact-index path {artifact}")
        records[artifact] = digest
    return records


def _history(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 200 or [int(item["epoch"]) for item in records] != list(
        range(200)
    ):
        raise ValueError("codec history does not contain epochs 0..199 exactly once")
    for item in records:
        if int(item.get("examples", -1)) != 432:
            raise ValueError("codec history epoch does not contain 432 examples")
    return records


def freeze_run(
    run_dir: Path,
    *,
    run_index: int,
    family: str,
    seed: int,
    training_commit: str,
    training_slurm_job_id: str,
) -> dict[str, Any]:
    expected_name = f"task_{run_index}_{family}_seed_{seed}"
    directory = Path(run_dir).resolve(strict=True)
    assert_development_path(directory)
    if directory.name != expected_name:
        raise ValueError(f"codec run directory differs from {expected_name}")
    paths = {name: directory / name for name in (*ARTIFACTS, "artifact_sha256.txt")}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"codec run {run_index} artifacts are incomplete")
    expected_config = json.loads(
        json.dumps(
            CodecRunConfig.frozen(
                mode="full",
                codec="dcae_l20",
                family=family,
                seed=seed,
            ).to_record()
        )
    )
    config = load_strict_json(paths["config.json"])
    result = load_strict_json(paths["result.json"])
    tracking = load_strict_json(paths["wandb.json"])
    history = _history(paths["history.jsonl"])
    if config != expected_config or result.get("config") != expected_config:
        raise ValueError(f"codec run {run_index} configuration differs")
    if (
        result.get("scope") != "O1_codec_full"
        or result.get("paper0_commit") != training_commit
        or result.get("development_run") != "85604"
        or result.get("held_out_85606_read") is not False
        or result.get("completed_epochs") != 200
        or result.get("physics_derived_loss_used") is not False
        or result.get("checkpoint_reload_bitwise_exact") is not True
    ):
        raise ValueError(f"codec run {run_index} completion contract differs")
    selected_epoch = min(
        range(200),
        key=lambda epoch: float(history[epoch]["validation_equal_channel_mae"]),
    )
    selected_loss = float(history[selected_epoch]["validation_equal_channel_mae"])
    if (
        int(result.get("selected_epoch", -1)) != selected_epoch
        or float(result.get("selected_validation_equal_channel_mae")) != selected_loss
    ):
        raise ValueError(f"codec run {run_index} checkpoint selection differs")

    hashes = {name: sha256_path(path) for name, path in paths.items() if name != "artifact_sha256.txt"}
    if result.get("selected_checkpoint", {}).get("sha256") != hashes["selected.pt"]:
        raise ValueError(f"codec run {run_index} selected checkpoint hash differs")
    if result.get("final_training_state", {}).get("sha256") != hashes[
        "final_training_state.pt"
    ]:
        raise ValueError(f"codec run {run_index} final state hash differs")
    indexed = _artifact_index(paths["artifact_sha256.txt"])
    if set(indexed) != {paths[name].resolve(strict=True) for name in ARTIFACTS}:
        raise ValueError(f"codec run {run_index} artifact index inventory differs")
    for name in ARTIFACTS:
        if indexed[paths[name].resolve(strict=True)] != hashes[name]:
            raise ValueError(f"codec run {run_index} indexed {name} hash differs")

    expected_run_id = f"p0o1r1-{training_slurm_job_id}-{run_index}"
    if (
        tracking.get("required") is not True
        or tracking.get("mode") != "online"
        or tracking.get("epochs_logged") != 200
        or tracking.get("remote_presence_verified_after_finish") is not True
        or tracking.get("remote_state_after_finish") != "finished"
        or tracking.get("local_artifacts_are_scientific_authority") is not True
        or tracking.get("spec", {}).get("run_id") != expected_run_id
    ):
        raise ValueError(f"codec run {run_index} W&B completion differs")
    return {
        "run_index": run_index,
        "family": family,
        "seed": seed,
        "run_directory": str(directory),
        "training_result": {
            "path": str(paths["result.json"].resolve(strict=True)),
            "sha256": hashes["result.json"],
        },
        "selected_checkpoint": {
            "path": str(paths["selected.pt"].resolve(strict=True)),
            "sha256": hashes["selected.pt"],
        },
        "final_training_state": {
            "path": str(paths["final_training_state.pt"].resolve(strict=True)),
            "sha256": hashes["final_training_state.pt"],
        },
        "config_sha256": hashes["config.json"],
        "history_sha256": hashes["history.jsonl"],
        "wandb_record_sha256": hashes["wandb.json"],
        "artifact_index_sha256": sha256_path(paths["artifact_sha256.txt"]),
        "selected_epoch": selected_epoch,
        "selected_global_step": int(history[selected_epoch]["global_step"]),
        "selected_validation_equal_channel_mae": selected_loss,
        "wandb": {
            "run_id": expected_run_id,
            "run_url": tracking["run_url"],
            "remote_state": "finished",
        },
    }


def main() -> int:
    args = parse_args()
    for path in (args.job_root, args.output):
        assert_development_path(path)
    root = args.job_root.resolve(strict=True)
    if root.name != f"job_{args.training_slurm_job_id}":
        raise ValueError("training job root and Slurm ID differ")
    summary_path = root / "training_summary.json"
    summary = load_strict_json(summary_path)
    if (
        summary.get("scope") != "phase2_O1_R1_full_codec_training"
        or summary.get("paper0_commit") != args.training_commit
        or str(summary.get("slurm_job_id")) != args.training_slurm_job_id
        or summary.get("development_run") != "85604"
        or summary.get("held_out_85606_read") is not False
        or summary.get("completed_logical_runs") != 6
        or summary.get("training_result_accepted") is not False
        or summary.get("O1_scientific_evaluation_completed") is not False
    ):
        raise ValueError("training matrix summary contract differs")
    runs = [
        freeze_run(
            root / f"task_{index}_{family}_seed_{seed}",
            run_index=index,
            family=family,
            seed=seed,
            training_commit=args.training_commit,
            training_slurm_job_id=args.training_slurm_job_id,
        )
        for index, family, seed in RUNS
    ]
    result = {
        "schema_version": 1,
        "scope": "phase2_O1_R1_training_matrix_frozen",
        "status": "completed_pending_scientific_O1_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_commit": args.training_commit,
        "training_slurm_job_id": args.training_slurm_job_id,
        "audit_commit": args.audit_commit,
        "audit_slurm_job_id": args.audit_slurm_job_id,
        "training_summary": {
            "path": str(summary_path.resolve(strict=True)),
            "sha256": sha256_path(summary_path),
        },
        "runs": runs,
        "run_count": len(runs),
        "checkpoint_choice_frozen_before_physics_metrics": True,
        "O1_scientific_evaluation_completed": False,
        "R1_accepted": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output.resolve(strict=False)
    write_strict_json_atomic(output, result)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
