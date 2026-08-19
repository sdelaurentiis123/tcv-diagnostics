#!/usr/bin/env python3
"""Run the B5 residual audit inside one required live online W&B run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


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
from tcv_diagnostics.wandb_tracking import WandbRunSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _one_argument(values: Sequence[str], flag: str) -> str:
    indices = [index for index, value in enumerate(values) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(values):
        raise ValueError(f"B5 audit command must contain exactly one {flag}")
    return str(values[indices[0] + 1])


def validate_audit_command(args: argparse.Namespace) -> list[str]:
    evaluator = Path(args.evaluator).resolve(strict=True)
    assert_development_path(evaluator)
    if sha256_path(evaluator) != args.evaluator_sha256:
        raise ValueError("B5 residual-audit evaluator SHA-256 differs")
    values = list(args.evaluation_args)
    if values and values[0] == "--":
        values = values[1:]
    if not values or any("85606" in value.lower() for value in values):
        raise ValueError("B5 audit command is empty or mentions held-out data")
    if Path(_one_argument(values, "--output-directory")) != args.output_directory:
        raise ValueError("B5 wrapper/evaluator output directories differ")
    if _one_argument(values, "--paper0-commit") != args.paper0_commit:
        raise ValueError("B5 wrapper/evaluator commits differ")
    return [sys.executable, "-u", str(evaluator), *values]


def audit_metrics(
    result: Mapping[str, Any], audit: Mapping[str, Any]
) -> dict[str, int | float | bool]:
    metrics: dict[str, int | float | bool] = {
        "audit/completed": 1,
        "audit/target_count": int(result["target_count"]),
        "audit/wall_seconds": float(result["wall_seconds"]),
        "forecast/wall_seconds": float(result["generation"]["wall_seconds"]),
        "forecast/peak_cuda_memory_bytes": int(
            result["generation"]["peak_cuda_memory_bytes"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/validation_frames_read": bool(result["validation_frames_read"]),
        "scope/training_performed": bool(result["training_performed"]),
        "scope/B5_training_authorized": bool(result["B5_training_authorized"]),
        "joint/global_entropy_effective_rank": float(
            audit["cross_field"]["global"]["entropy_effective_rank"]
        ),
    }
    for field in result["fields"]:
        scale = audit["scale"]["global"][field]
        metrics[f"residual/{field}/RMS"] = float(scale["RMS"])
        metrics[f"residual/{field}/MAE"] = float(scale["MAE"])
        metrics[f"residual/{field}/bias"] = float(scale["bias"])
        metrics[f"residual/{field}/variance_ratio"] = float(
            scale["residual_to_target_variance_ratio"]
        )
        hetero = audit["scale"]["heteroscedasticity"][field][
            "q95_to_q05_ratio"
        ]
        if hetero is not None:
            metrics[f"residual/{field}/pointwise_std_q95_to_q05"] = float(hetero)
        temporal = audit["temporal_autocorrelation"]["pattern"]["fields"][field][
            "length_summary"
        ]
        stable = temporal["first_stable_near_zero_lag"]
        if stable is not None:
            metrics[f"temporal/{field}/stable_near_zero_frames"] = int(stable)
        for band, values in audit["toroidal_support"]["fields"][field][
            "bands"
        ].items():
            metrics[f"toroidal/{field}/{band}/residual_power_fraction"] = float(
                values["residual_power_fraction"]
            )
    return metrics


def main() -> None:
    args = parse_args()
    assert_development_path(args.output_directory)
    command = validate_audit_command(args)
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b5_h1_training_residual_audit",
        tags=("paper0", "phase3", "b5", "residual-audit", "85604-only"),
    )
    import wandb

    api = wandb.Api(timeout=30)
    viewer = api.viewer
    if not api.api_key:
        raise RuntimeError("online W&B tracking is required but no API key exists")
    if str(getattr(viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B entity differs")
    tracking_directory = args.output_directory.parent / (
        f".{args.output_directory.name}.wandb"
    )
    if tracking_directory.exists():
        raise FileExistsError(tracking_directory)
    tracking_directory.mkdir(parents=True)
    run = wandb.init(
        entity=spec.entity,
        project=spec.project,
        group=spec.group,
        name=spec.run_name,
        id=spec.run_id,
        resume="never",
        job_type=spec.job_type,
        tags=list(spec.tags),
        config={
            "paper0_commit": args.paper0_commit,
            "development_run": "85604",
            "held_out_85606_access_allowed": False,
            "model_arm": "C5P-H1",
            "model_seed": 1701,
            "target_frames": [2, 432],
            "target_count": 430,
            "training_performed": False,
            "local_artifacts_are_scientific_authority": True,
            "command": command,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline) or str(run.id) != spec.run_id:
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError("W&B online B5 audit initialization differs")
    run.log({"audit/stage": 0, "audit/started": 1}, step=0)
    completed = subprocess.run(command, check=False, env=dict(os.environ))
    if completed.returncode != 0:
        run.summary["audit/exit_code"] = int(completed.returncode)
        run.finish(exit_code=completed.returncode)
        raise subprocess.CalledProcessError(completed.returncode, command)

    output = args.output_directory.resolve(strict=True)
    result = load_strict_json(output / "result.json")
    audit = load_strict_json(output / "residual_audit.json")
    if (
        result.get("scope") != "B5_frozen_H1_training_residual_audit_85604"
        or result.get("status") != "completed_architecture_sizing_audit_only"
        or result.get("paper0_commit") != args.paper0_commit
        or result.get("held_out_85606_read") is not False
        or result.get("validation_frames_read") is not False
        or result.get("forecast_closed_and_hashed_before_truth_read") is not True
        or result.get("training_performed") is not False
        or result.get("B5_training_authorized") is not False
    ):
        run.finish(exit_code=1)
        raise RuntimeError("completed B5 audit identity differs")
    metrics = audit_metrics(result, audit)
    run.log({"audit/stage": 1, **metrics}, step=1)
    run.summary.update(
        {
            **metrics,
            "provenance/result_sha256": sha256_path(output / "result.json"),
            "provenance/residual_audit_sha256": sha256_path(
                output / "residual_audit.json"
            ),
            "provenance/forecast_sha256": result["forecast"]["sha256"],
            "provenance/paper0_commit": args.paper0_commit,
            "scope/local_artifacts_are_scientific_authority": True,
        }
    )
    run_url = str(run.url)
    run.finish(exit_code=0)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote = wandb.Api(timeout=30).run(remote_path)
    if str(remote.state) != "finished":
        raise RuntimeError("remote W&B B5 audit run is not finished")
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": spec.to_record(),
        "authenticated_username": str(getattr(viewer, "username", "")),
        "wandb_version": str(wandb.__version__),
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": str(remote.state),
        "checkpoints_uploaded": False,
        "forecast_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    write_strict_json_atomic(output / "wandb.json", tracking)
    print(json.dumps(tracking, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
