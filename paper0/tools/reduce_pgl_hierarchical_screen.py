#!/usr/bin/env python3
"""Reduce all six hierarchical M32 scores with the frozen stopping rule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.pgl_hierarchical_decision import evaluate_two_epoch_decision
from tcv_diagnostics.pgl_hierarchical_training import (
    PGL_HIERARCHICAL_ARMS,
    PGL_HIERARCHICAL_CHECKPOINT_UPDATES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def verify_checkout(root: Path, expected: str) -> None:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != expected or dirty:
        raise RuntimeError("hierarchical reduction requires the locked clean checkout")


def load_six_results(root: Path, *, commit: str) -> tuple[dict, list[dict]]:
    records = {}
    artifacts = []
    task = 0
    for arm in PGL_HIERARCHICAL_ARMS:
        for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
            path = (
                root
                / f"task_{task}_{arm}_u{update:04d}"
                / "scoring"
                / "result.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            result = load_strict_json(path)
            if (
                result.get("scope")
                != "old_85604_pgl_hierarchical_physics_evaluation"
                or result.get("status") not in ("completed_passed", "completed_failed")
                or result.get("development_run") != "85604"
                or result.get("arm") != arm
                or result.get("optimizer_update") != update
                or result.get("paper0_commit") != commit
                or result.get("training_performed") is not False
                or result.get("checkpoint_selection_performed") is not False
                or result.get("target_truth_used_during_generation") is not False
                or result.get("physics_derived_training_loss_used") is not (
                    arm == "TRANSPORT"
                )
                or result.get("held_out_85606_read") is not False
                or result.get("new_nersc_data_read") is not False
            ):
                raise ValueError(f"hierarchical score contract differs for task {task}")
            score_path = Path(str(result["score"]["path"]))
            assert_development_path(score_path)
            if sha256_path(score_path) != result["score"]["sha256"]:
                raise ValueError(f"hierarchical score SHA differs for task {task}")
            records[(arm, update)] = result
            artifacts.append(
                {
                    "task": task,
                    "arm": arm,
                    "optimizer_update": update,
                    "result": {"path": str(path), "sha256": sha256_path(path)},
                    "score": result["score"],
                }
            )
            task += 1
    return records, artifacts


def main() -> int:
    args = parse_args()
    for path in (args.score_root, args.output, args.paper0_root):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_checkout(args.paper0_root, args.paper0_commit)
    records, artifacts = load_six_results(args.score_root, commit=args.paper0_commit)
    decision = evaluate_two_epoch_decision(records)
    decision.update(
        {
            "status": "completed",
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "inputs": artifacts,
            "checkpoint_selection_performed": False,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
        }
    )
    args.output.mkdir(parents=True)
    decision_path = args.output / "decision.json"
    write_strict_json_atomic(decision_path, decision)
    csv_path = args.output / "checkpoint_metrics.csv"
    names = (
        "arm",
        "optimizer_update",
        "equivalent_epochs",
        "integrated_spread_skill",
        "spatial_covariance_error",
        "local_spread_skill_median",
        "mean_transport_relative_l2",
        "spectral_error",
        "cross_spectrum_error",
        "phase_error_degrees",
        "field_fair_crps_h4",
        "production_passed",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for arm in PGL_HIERARCHICAL_ARMS:
            for update in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
                metric = decision["metrics"][f"{arm}_update_{update}"]
                writer.writerow(
                    {
                        "arm": arm,
                        "optimizer_update": update,
                        "equivalent_epochs": update / 214,
                        **{name: metric[name] for name in names[3:]},
                    }
                )
    manifest = {
        "schema_version": 1,
        "scope": "old_85604_pgl_hierarchical_screen_reduction_manifest",
        "status": "completed",
        "paper0_commit": args.paper0_commit,
        "slurm_job_id": args.slurm_job_id,
        "score_root": str(args.score_root),
        "inputs": artifacts,
        "outputs": {
            "decision": {"path": str(decision_path), "sha256": sha256_path(decision_path)},
            "metrics_csv": {"path": str(csv_path), "sha256": sha256_path(csv_path)},
        },
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    write_strict_json_atomic(args.output / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "manifest": manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
