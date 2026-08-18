#!/usr/bin/env python3
"""Finalize six frozen O2 seed gates into the per-arm stop/go decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.o2_matrix import finalize_o2_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--result-sha256", action="append", required=True)
    parser.add_argument("--training-freeze-result", type=Path, required=True)
    parser.add_argument("--training-freeze-result-sha256", required=True)
    parser.add_argument("--references-result", type=Path, required=True)
    parser.add_argument("--references-result-sha256", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest-sha256", required=True)
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
    if sha256_path(resolved) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {resolved}")
    return resolved


def verify_task_artifacts(result_path: Path, result: dict) -> None:
    expected = {result_path.resolve(strict=True)}
    for name in ("generation", "forecast", "score"):
        record = result[name]
        path = verify_input(Path(record["path"]), record["sha256"])
        expected.add(path)
    index = result_path.parent / "artifact_sha256.txt"
    records = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        path = Path(name.strip()).resolve(strict=True)
        if path in records:
            raise ValueError("duplicate O2 task artifact-index path")
        records[path] = digest
    if set(records) != expected:
        raise ValueError("O2 task artifact-index inventory differs")
    for path, digest in records.items():
        if sha256_path(path) != digest:
            raise ValueError(f"O2 task indexed artifact hash differs: {path}")


def main() -> None:
    args = parse_args()
    if len(args.result) != 6 or len(args.result_sha256) != 6:
        raise ValueError("O2 finalization requires six result/hash pairs")
    for path in (
        *args.result,
        args.training_freeze_result,
        args.references_result,
        args.evaluation_manifest,
        args.output,
    ):
        assert_development_path(path)
    verify_checkout(args.paper0_commit)
    freeze_path = verify_input(
        args.training_freeze_result, args.training_freeze_result_sha256
    )
    references_path = verify_input(
        args.references_result, args.references_result_sha256
    )
    manifest_path = verify_input(
        args.evaluation_manifest, args.evaluation_manifest_sha256
    )
    manifest = load_strict_json(manifest_path)
    if (
        manifest.get("status") != "frozen_before_O2_scientific_evaluation"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
    ):
        raise RuntimeError("O2 evaluation manifest contract differs")
    freeze = load_strict_json(freeze_path)
    if (
        freeze.get("status") != "completed_pending_scientific_O2_evaluation"
        or freeze.get("O2_scientific_evaluation_completed") is not False
        or freeze.get("O3_launch_allowed") is not False
    ):
        raise RuntimeError("O2 training freeze is not awaiting evaluation")
    references = load_strict_json(references_path)
    if (
        references.get("scope") != "O2_frozen_uncompressed_references"
        or references.get("status") != "completed"
        or references.get("mode") != "full"
        or references.get("scientific_authority") is not True
        or references.get("O2_seed_gate_evaluated") is not False
        or references.get("O3_launch_allowed") is not False
    ):
        raise RuntimeError("full O2 references are not finalizable")

    run_records = []
    compact_runs = []
    for path, digest in zip(args.result, args.result_sha256):
        resolved = verify_input(path, digest)
        record = load_strict_json(resolved)
        verify_task_artifacts(resolved, record)
        run_records.append(record)
        compact_runs.append(
            {
                "training_run_index": int(record["training_run_index"]),
                "arm": record["arm"],
                "seed": int(record["seed"]),
                "selected_epoch": int(record["selected_epoch"]),
                "selected_checkpoint": dict(record["selected_checkpoint"]),
                "result": {"path": str(resolved), "sha256": digest},
                "score": dict(record["score"]),
                "forecast": dict(record["forecast"]),
                "passes": bool(record["O2_seed_accepted"]),
                "status": record["gate"]["status"],
            }
        )
    decision = finalize_o2_matrix(run_records)
    result = {
        "schema_version": 1,
        "scope": "phase2_C5P_O2_complete_scientific_matrix",
        "status": "completed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "training_freeze_result": {
            "path": str(freeze_path),
            "sha256": args.training_freeze_result_sha256,
        },
        "references_result": {
            "path": str(references_path),
            "sha256": args.references_result_sha256,
        },
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": args.evaluation_manifest_sha256,
        },
        "runs": compact_runs,
        "run_count": len(compact_runs),
        "decision": decision,
        "O2_scientific_evaluation_completed": True,
        "O2_accepted_arms": list(decision["accepted_arms"]),
        "new_O3_protocol_may_be_frozen": decision[
            "new_O3_protocol_may_be_frozen"
        ],
        "O3_launch_allowed": False,
        "stochastic_model_authorized": False,
        "held_out_85606_access_allowed": False,
    }
    output = Path(args.output).resolve(strict=False)
    write_strict_json_atomic(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_path(output),
                "accepted_arms": decision["accepted_arms"],
                "disposition": decision["disposition"],
                "O3_launch_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
