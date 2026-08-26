#!/usr/bin/env python3
"""Reduce the frozen paired old-85604 C5P/E6B multi-lead pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from paper0.tools.train_codec_free_stage1_pilot import atomic_json, repository_commit
from paper0.tools.train_matched_state_multilead import FAMILIES, LEADS, SCOPE
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json


REDUCTION_SCOPE = "post_ecrd_old_85604_matched_state_multilead_reduction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--c5p-result", type=Path, required=True)
    parser.add_argument("--c5p-result-sha256", required=True)
    parser.add_argument("--e6b-result", type=Path, required=True)
    parser.add_argument("--e6b-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def locked_result(
    path: Path,
    digest: str,
    *,
    family: str,
    manifest_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    assert_development_path(path)
    if sha256_path(path) != digest:
        raise ValueError(f"{family} result SHA-256 differs")
    result = load_strict_json(path)
    expected = {
        "scope": SCOPE,
        "status": "passed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "guard_frames_read": False,
        "physics_derived_loss_used": False,
        "family": family,
        "seed": 1701,
        "paper0_commit": paper0_commit,
        "manifest_sha256": manifest_sha256,
        "epochs": 12,
        "optimizer_updates": 6396,
        "expected_optimizer_updates": 6396,
        "lead_steps": list(LEADS),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{family} result identity differs")
    if result.get("training_gate", {}).get("passed") is not True:
        raise ValueError(f"{family} result training gate failed")
    checkpoint = result.get("best_checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    checkpoint_digest = str(checkpoint.get("sha256", ""))
    assert_development_path(checkpoint_path)
    if not checkpoint_digest or sha256_path(checkpoint_path) != checkpoint_digest:
        raise ValueError(f"{family} selected checkpoint SHA-256 differs")
    return result


def compare_results(
    c5p: Mapping[str, Any], e6b: Mapping[str, Any]
) -> dict[str, Any]:
    by_lead: dict[str, Any] = {}
    for lead in LEADS:
        key = str(lead)
        c5p_record = c5p["best_validation"]["per_lead"][key]
        e6b_record = e6b["best_validation"]["per_lead"][key]
        c5p_mse = float(c5p_record["shared_field_mean_model_derivative_mse"])
        e6b_mse = float(e6b_record["shared_field_mean_model_derivative_mse"])
        by_lead[key] = {
            "c5p_shared_Ne_Pe_Pi_derivative_mse": c5p_mse,
            "e6b_shared_Ne_Pe_Pi_derivative_mse": e6b_mse,
            "e6b_over_c5p": e6b_mse / c5p_mse,
            "c5p_shared_persistence_skill": float(
                c5p_record["shared_field_persistence_relative_skill"]
            ),
            "e6b_shared_persistence_skill": float(
                e6b_record["shared_field_persistence_relative_skill"]
            ),
        }
    ratios = np.asarray(
        [by_lead[str(lead)]["e6b_over_c5p"] for lead in LEADS],
        dtype=np.float64,
    )
    return {
        "by_lead": by_lead,
        "median_e6b_over_c5p_shared_mse": float(np.median(ratios)),
        "minimum_e6b_over_c5p_shared_mse": float(np.min(ratios)),
        "maximum_e6b_over_c5p_shared_mse": float(np.max(ratios)),
        "c5p_selection_metric": float(
            c5p["best_checkpoint"]["selection_metric"]
        ),
        "e6b_selection_metric": float(
            e6b["best_checkpoint"]["selection_metric"]
        ),
        "c5p_selected_epoch": int(c5p["best_checkpoint"]["epoch"]),
        "e6b_selected_epoch": int(e6b["best_checkpoint"]["epoch"]),
    }


def reduce_pair(
    *,
    c5p: Mapping[str, Any],
    e6b: Mapping[str, Any],
    c5p_lock: Mapping[str, str],
    e6b_lock: Mapping[str, str],
    manifest_lock: Mapping[str, str],
    paper0_commit: str,
    slurm_job_id: str,
) -> dict[str, Any]:
    mechanical = {
        family: bool(result.get("training_gate", {}).get("passed"))
        for family, result in (("c5p", c5p), ("e6b", e6b))
    }
    transition = {
        family: bool(result.get("transition_gate", {}).get("passed"))
        for family, result in (("c5p", c5p), ("e6b", e6b))
    }
    physics_authorized = all(mechanical.values()) and all(transition.values())
    selected_at_boundary = {
        family: bool(
            result.get("best_checkpoint", {}).get("selected_at_budget_boundary")
        )
        for family, result in (("c5p", c5p), ("e6b", e6b))
    }
    comparison = compare_results(c5p, e6b)
    numeric_values = [
        comparison["median_e6b_over_c5p_shared_mse"],
        comparison["minimum_e6b_over_c5p_shared_mse"],
        comparison["maximum_e6b_over_c5p_shared_mse"],
        comparison["c5p_selection_metric"],
        comparison["e6b_selection_metric"],
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("paired comparison contains non-finite values")
    return {
        "schema_version": 1,
        "scope": REDUCTION_SCOPE,
        "status": "completed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "guard_frames_read": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "physics_evaluation_performed": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
        "paper0_commit": paper0_commit,
        "slurm_job_id": slurm_job_id,
        "manifest": dict(manifest_lock),
        "results": {"c5p": dict(c5p_lock), "e6b": dict(e6b_lock)},
        "mechanical_gate_by_family": mechanical,
        "transition_gate_by_family": transition,
        "selected_at_budget_boundary": selected_at_boundary,
        "duration_censored_for_any_arm": any(selected_at_boundary.values()),
        "transition_comparison": comparison,
        "paired_physics_evaluation_authorized": physics_authorized,
        "three_seed_scaling_authorized": False,
        "decision": (
            "run_causal_paired_derived_field_physics_evaluation"
            if physics_authorized
            else "stop_before_paired_physics_and_record_transition_failure"
        ),
    }


def main() -> None:
    args = parse_args()
    for path in (
        args.manifest,
        args.c5p_result,
        args.e6b_result,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from reduction lock")
    if sha256_path(args.manifest) != args.manifest_sha256:
        raise ValueError("matched state-view manifest SHA-256 differs")
    manifest = load_strict_json(args.manifest)
    if (
        manifest.get("scope") != SCOPE
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("new_nersc_data_access_allowed") is not False
        or tuple(manifest.get("state_views", {})) != FAMILIES
    ):
        raise ValueError("matched state-view manifest scope differs")

    c5p = locked_result(
        args.c5p_result,
        args.c5p_result_sha256,
        family="c5p",
        manifest_sha256=args.manifest_sha256,
        paper0_commit=args.paper0_commit,
    )
    e6b = locked_result(
        args.e6b_result,
        args.e6b_result_sha256,
        family="e6b",
        manifest_sha256=args.manifest_sha256,
        paper0_commit=args.paper0_commit,
    )
    result = reduce_pair(
        c5p=c5p,
        e6b=e6b,
        c5p_lock={"path": str(args.c5p_result), "sha256": args.c5p_result_sha256},
        e6b_lock={"path": str(args.e6b_result), "sha256": args.e6b_result_sha256},
        manifest_lock={
            "path": str(args.manifest),
            "sha256": args.manifest_sha256,
        },
        paper0_commit=args.paper0_commit,
        slurm_job_id=args.slurm_job_id,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
