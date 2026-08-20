#!/usr/bin/env python3
"""Validate the four bounded ECRD smoke runs without scientific scoring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_training import (
    ECRD_ARMS,
    ECRDTrainingConfig,
    frozen_parameter_counts,
    model_config_record,
)
from tcv_diagnostics.model_data import load_strict_json, write_strict_json_atomic
from tcv_diagnostics.models.field_residual_edm import B5_RESIDUAL_SCALES


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PROTOCOL_SHA256 = (
    "74028e90568a4cfea0721c7fd7a28297a230672c538b3e7908784603c3b2fea4"
)
EXPECTED_MEMBER_SEEDS = [67_540, 67_541]
EXPECTED_EQUIVARIANCE_SHIFTS = [1, 2, 3, 7, 17]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="repeat exactly four times as ARM=/absolute/run/directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--smoke-paper0-commit", required=True)
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
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError(f"Paper 0 checkout is dirty:\n{dirty}")


def _mentions_held_out(value: Any) -> bool:
    if isinstance(value, str):
        return "85606" in value.lower()
    if isinstance(value, Mapping):
        return any(_mentions_held_out(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_mentions_held_out(item) for item in value)
    return False


def parse_run_arguments(values: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        arm, separator, raw_path = str(value).partition("=")
        if not separator or arm not in ECRD_ARMS or not raw_path:
            raise ValueError("each ECRD smoke run must be ARM=/absolute/path")
        if arm in runs:
            raise ValueError(f"duplicate ECRD smoke arm {arm!r}")
        path = Path(raw_path)
        if not path.is_absolute() or "85606" in str(path).lower():
            raise ValueError("ECRD smoke run path is unsafe")
        runs[arm] = path
    if tuple(runs) != ECRD_ARMS:
        raise ValueError("ECRD smoke runs must be supplied in frozen arm order")
    return runs


def _verify_file(record: Mapping[str, Any], *, label: str) -> dict[str, str]:
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if "85606" in str(path).lower() or not path.is_file() or len(expected) != 64:
        raise RuntimeError(f"{label} artifact record is incomplete")
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(f"{label} artifact SHA-256 differs")
    return {"path": str(path), "sha256": observed}


def _require_exact_fields(
    record: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(f"{label} field {name!r} differs")


def validate_smoke_run(
    *,
    arm: str,
    run: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    paper0_commit: str,
) -> dict[str, Any]:
    if not run.is_dir():
        raise FileNotFoundError(run)
    result_path = run / "result.json"
    probe_path = run / "smoke_probe.json"
    wandb_path = run / "wandb.json"
    result = load_strict_json(result_path)
    probe = load_strict_json(probe_path)
    tracking = load_strict_json(wandb_path)
    # Mapping keys such as ``held_out_85606_read`` are ignored by this
    # value-only traversal; a path or other string value mentioning it fails.
    if _mentions_held_out((result, probe, tracking)):
        raise RuntimeError(f"{arm} smoke artifacts mention the held-out run")
    config = ECRDTrainingConfig(arm=arm, seed=1701, mode="smoke")
    _require_exact_fields(
        result,
        {
            "scope": "ECRD_matched_model_development_training_85604",
            "status": "training_completed_checkpoint_selected",
            "mode": "smoke",
            "arm": arm,
            "seed": 1701,
            "paper0_commit": paper0_commit,
            "development_run": "85604",
            "training": config.to_record(),
            "model": model_config_record(arm),
            "parameter_count": frozen_parameter_counts()[arm],
            "completed_epochs": 1,
            "completed_optimizer_steps": 2,
            "target_presentations": 4,
            "candidate_count": 1,
            "selected_completed_epoch": 1,
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
        },
        label=f"{arm} result",
    )
    selected_validation = result.get("selected_validation", {})
    if (
        selected_validation.get("target_frames") != [498, 502]
        or selected_validation.get("target_count") != 4
        or selected_validation.get("probes_per_target") != 4
        or set(selected_validation.get("blocks", {})) != {"SMOKE"}
        or not math.isfinite(
            float(selected_validation.get("checkpoint_score", math.nan))
        )
    ):
        raise RuntimeError(f"{arm} data-only smoke validation differs")
    artifacts = result.get("artifacts", {})
    verified_artifacts = {
        name: _verify_file(artifacts.get(name, {}), label=f"{arm} {name}")
        for name in (
            "config",
            "training_order",
            "validation_seed_bank",
            "history",
            "selected_checkpoint",
        )
    }
    run_config = load_strict_json(Path(verified_artifacts["config"]["path"]))
    _require_exact_fields(
        run_config,
        {
            "scope": "ECRD_matched_model_development_85604",
            "paper0_commit": paper0_commit,
            "training": config.to_record(),
            "model": model_config_record(arm),
            "parameter_count": frozen_parameter_counts()[arm],
            "residual_scales": list(B5_RESIDUAL_SCALES),
        },
        label=f"{arm} run config",
    )
    _require_exact_fields(
        run_config.get("authority", {}),
        {
            "authorized": True,
            "scope": f"ECRD_smoke_{arm}_seed1701_85604",
            "mode": "smoke",
            "arm": arm,
            "seed": 1701,
            "development_run": "85604",
            "target_truth_used_as_condition": False,
            "guard_frames_read": False,
            "held_out_85606_read": False,
            "manifest_sha256": manifest_sha256,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        },
        label=f"{arm} run authority",
    )
    candidates = artifacts.get("candidate_checkpoints", ())
    if len(candidates) != 1:
        raise RuntimeError(f"{arm} smoke candidate count differs")
    verified_candidate = _verify_file(candidates[0], label=f"{arm} candidate")

    expected_equivariance = arm in ("ECRD", "ECRD-History")
    _require_exact_fields(
        probe,
        {
            "scope": "bounded_non_scientific_ECRD_full_volume_mechanical_probe",
            "arm": arm,
            "optimizer_steps": 2,
            "canonical_field_shape": [1, 2, 1, 5, 64, 32, 88],
            "ensemble_members": 2,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "member_seeds": EXPECTED_MEMBER_SEEDS,
            "finite": True,
            "equivariance_shifts": EXPECTED_EQUIVARIANCE_SHIFTS,
            "equivariance_required": expected_equivariance,
            "all_mechanical_gates_passed": True,
            "scientific_result": False,
            "physics_metric_evaluated": False,
            "held_out_85606_read": False,
        },
        label=f"{arm} mechanical probe",
    )
    gates = probe.get("gates", {})
    expected_gates = {
        "finite",
        "canonical_shape",
        "member_diversity",
        "network_evaluations",
        "peak_memory",
        "required_equivariance",
    }
    if set(gates) != expected_gates or not all(
        value is True for value in gates.values()
    ):
        raise RuntimeError(f"{arm} mechanical gates differ")
    scalar_names = (
        "member_diversity",
        "max_generator_equivariance_error",
        "max_mean_head_equivariance_error",
        "peak_cuda_GiB",
    )
    if any(
        not math.isfinite(float(probe.get(name, math.nan))) for name in scalar_names
    ):
        raise RuntimeError(f"{arm} mechanical scalar is non-finite")
    if float(probe["member_diversity"]) <= 1.0e-8:
        raise RuntimeError(f"{arm} members collapsed")
    if expected_equivariance and (
        float(probe["max_generator_equivariance_error"]) > 1.0e-4
        or float(probe["max_mean_head_equivariance_error"]) > 1.0e-4
    ):
        raise RuntimeError(f"{arm} exact equivariance tolerance failed")

    _require_exact_fields(
        tracking,
        {
            "required": True,
            "mode": "online",
            "remote_presence_verified_after_finish": True,
            "remote_state_after_finish": "finished",
            "checkpoints_uploaded": False,
            "samples_uploaded": False,
            "epochs_logged": 1,
        },
        label=f"{arm} W&B record",
    )
    spec = tracking.get("spec", {})
    execution = manifest.get("execution", {})
    if (
        spec.get("entity") != execution.get("wandb_entity")
        or spec.get("project") != execution.get("wandb_project")
        or spec.get("group") != execution.get("wandb_group")
        or spec.get("job_type") != "ecrd_smoke_training"
    ):
        raise RuntimeError(f"{arm} W&B run identity differs")
    if not str(tracking.get("run_url", "")).startswith("https://wandb.ai/"):
        raise RuntimeError(f"{arm} W&B URL differs")

    return {
        "arm": arm,
        "seed": 1701,
        "run_directory": str(run.resolve()),
        "result": {"path": str(result_path), "sha256": sha256_path(result_path)},
        "mechanical_probe": {
            "path": str(probe_path),
            "sha256": sha256_path(probe_path),
            "member_diversity": float(probe["member_diversity"]),
            "max_generator_equivariance_error": float(
                probe["max_generator_equivariance_error"]
            ),
            "max_mean_head_equivariance_error": float(
                probe["max_mean_head_equivariance_error"]
            ),
            "peak_cuda_GiB": float(probe["peak_cuda_GiB"]),
            "all_mechanical_gates_passed": True,
        },
        "wandb": {
            "path": str(wandb_path),
            "sha256": sha256_path(wandb_path),
            "run_url": str(tracking["run_url"]),
            "remote_state_after_finish": "finished",
        },
        "selected_checkpoint": verified_artifacts["selected_checkpoint"],
        "candidate_checkpoint": verified_candidate,
        "artifacts": verified_artifacts,
        "parameter_count": int(result["parameter_count"]),
        "completed_optimizer_steps": 2,
        "checkpoint_reload_bitwise_exact": True,
        "scientific_result": False,
    }


def main() -> int:
    args = parse_args()
    verify_checkout(args.paper0_commit)
    if "85606" in str(args.output).lower() or args.output.exists():
        raise ValueError("ECRD smoke summary output is unsafe or already exists")
    manifest_path = args.manifest.resolve()
    if sha256_path(manifest_path) != args.manifest_sha256:
        raise RuntimeError("ECRD smoke manifest SHA-256 differs")
    manifest = load_strict_json(manifest_path)
    if (
        manifest.get("status") != "frozen_before_ECRD_engineering_smoke"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256
        or manifest.get("full_training_authorized") is not False
    ):
        raise RuntimeError("ECRD smoke manifest scope differs")
    runs = parse_run_arguments(args.run)
    records = {
        arm: validate_smoke_run(
            arm=arm,
            run=path,
            manifest=manifest,
            manifest_sha256=args.manifest_sha256,
            paper0_commit=args.smoke_paper0_commit,
        )
        for arm, path in runs.items()
    }
    summary = {
        "schema_version": 1,
        "status": "all_four_arms_passed_mechanical_smoke",
        "scope": "bounded_non_scientific_ECRD_engineering_smoke_85604",
        "paper0_commit": args.smoke_paper0_commit,
        "verifier_paper0_commit": args.paper0_commit,
        "slurm_job_id": str(args.slurm_job_id),
        "development_run": "85604",
        "manifest": {
            "path": str(manifest_path),
            "sha256": args.manifest_sha256,
        },
        "arms": records,
        "all_four_arms_passed": True,
        "scientific_result": False,
        "physics_metric_evaluated": False,
        "full_training_authorized": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    write_strict_json_atomic(args.output, summary)
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
