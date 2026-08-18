#!/usr/bin/env python3
"""Freeze the complete three-seed B2 training matrix before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper0.tools.evaluate_b2_checkpoint import (  # noqa: E402
    audit_full_training_result,
    audit_history,
)
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)


EXPECTED_SEEDS = (1701, 1702, 1703)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", action="append", type=Path, required=True)
    parser.add_argument("--training-result-sha256", action="append", required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--training-job-id", required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_checkout(expected_commit: str) -> None:
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(f"Paper 0 commit {actual} differs from {expected_commit}")
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


def verify_input(path: Path, expected_sha256: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    if sha256_path(resolved) != str(expected_sha256):
        raise ValueError(f"SHA-256 mismatch for {resolved}")
    return resolved


def freeze_training_seed(
    path: Path,
    expected_sha256: str,
    *,
    seed: int,
    training_commit: str,
) -> dict[str, Any]:
    result_path = verify_input(path, expected_sha256)
    record = load_strict_json(result_path)
    audited = audit_full_training_result(
        record,
        seed=seed,
        training_commit=training_commit,
    )
    history = audit_history(
        Path(audited["history"]["path"]),
        expected_sha256=audited["history"]["sha256"],
        selected_epoch=audited["selected_epoch"],
        selected_validation=record["selected_validation"],
        final_validation=record["final_validation"],
    )
    for name in (
        "selected_checkpoint",
        "final_training_state",
        "latent_normalization",
    ):
        verify_input(Path(record[name]["path"]), record[name]["sha256"])
    verify_input(
        Path(record["codec_checkpoint"]["path"]),
        record["codec_checkpoint"]["sha256"],
    )
    output = result_path.parent
    wandb_path = output / "wandb.json"
    config_path = output / "config.json"
    index_path = output / "artifact_sha256.txt"
    wandb = load_strict_json(verify_input(wandb_path, sha256_path(wandb_path)))
    verify_input(config_path, sha256_path(config_path))
    verify_input(index_path, sha256_path(index_path))
    if (
        wandb.get("mode") != "online"
        or wandb.get("remote_state_after_finish") != "finished"
        or wandb.get("epochs_logged") != 200
        or wandb.get("local_artifacts_are_scientific_authority") is not True
    ):
        raise ValueError("B2 training W&B completion record differs")
    return {
        "seed": int(seed),
        "training_result": {
            "path": str(result_path),
            "sha256": str(expected_sha256),
        },
        "artifact_index": {
            "path": str(index_path.resolve(strict=True)),
            "sha256": sha256_path(index_path),
        },
        "config": {
            "path": str(config_path.resolve(strict=True)),
            "sha256": sha256_path(config_path),
        },
        "selected_epoch": int(audited["selected_epoch"]),
        "selected_validation": dict(record["selected_validation"]),
        "final_validation": dict(record["final_validation"]),
        "history_audit": history,
        "selected_checkpoint": dict(record["selected_checkpoint"]),
        "final_training_state": dict(record["final_training_state"]),
        "latent_normalization": dict(record["latent_normalization"]),
        "codec_checkpoint": dict(record["codec_checkpoint"]),
        "parameter_count": int(record["parameter_count"]),
        "peak_cuda_bytes": int(record["peak_cuda_bytes"]),
        "wall_seconds": float(record["wall_seconds"]),
        "wandb": {
            "path": str(wandb_path.resolve(strict=True)),
            "sha256": sha256_path(wandb_path),
            "run_url": wandb["run_url"],
            "remote_state_after_finish": wandb["remote_state_after_finish"],
        },
        "training_complete": True,
        "scientific_acceptance_evaluated": False,
    }


def freeze_matrix(
    results: Sequence[Path],
    result_sha256: Sequence[str],
    *,
    training_commit: str,
    training_job_id: str,
    paper0_commit: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    if len(results) != 3 or len(result_sha256) != 3:
        raise ValueError("B2 training freeze requires exactly three results/hashes")
    runs = [
        freeze_training_seed(
            path,
            digest,
            seed=seed,
            training_commit=training_commit,
        )
        for seed, path, digest in zip(EXPECTED_SEEDS, results, result_sha256)
    ]
    if tuple(run["seed"] for run in runs) != EXPECTED_SEEDS:
        raise RuntimeError("B2 frozen training seed order differs")
    return {
        "schema_version": 1,
        "scope": "phase3_B2_LDM_H2_full_training_matrix_frozen",
        "status": "completed_pending_bounded_evaluator_smoke",
        "paper0_commit": paper0_commit,
        "training_commit": training_commit,
        "training_job_id": str(training_job_id),
        "slurm_job_id": str(slurm_job_id),
        "development_run": "85604",
        "held_out_85606_read": False,
        "seed_count": 3,
        "seeds": list(EXPECTED_SEEDS),
        "runs": runs,
        "all_training_histories_complete": True,
        "all_checkpoint_choices_frozen_before_probabilistic_metrics": True,
        "bounded_evaluator_smoke_required": True,
        "bounded_evaluator_smoke_completed": False,
        "full_probabilistic_evaluation_allowed": False,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }


def main() -> None:
    args = parse_args()
    assert_development_path(args.output)
    verify_checkout(args.paper0_commit)
    matrix = freeze_matrix(
        args.training_result,
        args.training_result_sha256,
        training_commit=args.training_commit,
        training_job_id=args.training_job_id,
        paper0_commit=args.paper0_commit,
        slurm_job_id=args.slurm_job_id,
    )
    write_strict_json_atomic(args.output, matrix)
    print(
        json.dumps(
            {
                "status": matrix["status"],
                "output": str(args.output.resolve(strict=True)),
                "sha256": sha256_path(args.output),
                "held_out_85606_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
