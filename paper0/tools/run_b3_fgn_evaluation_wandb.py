#!/usr/bin/env python3
"""Run the B3 FGN evaluator inside a required live online W&B run."""

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
    parser.add_argument("--seed", type=int, choices=(1701,), required=True)
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
        raise ValueError(f"evaluation command must contain exactly one {flag}")
    return str(values[indices[0] + 1])


def validate_evaluation_command(args: argparse.Namespace) -> tuple[list[str], str]:
    evaluator = Path(args.evaluator).resolve(strict=True)
    assert_development_path(evaluator)
    if sha256_path(evaluator) != args.evaluator_sha256:
        raise ValueError("B3 FGN evaluator SHA-256 differs")
    values = list(args.evaluation_args)
    if values and values[0] == "--":
        values = values[1:]
    if not values or any("85606" in value.lower() for value in values):
        raise ValueError("B3 FGN evaluation command is empty or mentions held-out data")
    if Path(_one_argument(values, "--output-directory")) != args.output_directory:
        raise ValueError("B3 wrapper/evaluator output directories differ")
    if _one_argument(values, "--paper0-commit") != args.paper0_commit:
        raise ValueError("B3 wrapper/evaluator Paper 0 commits differ")
    if int(_one_argument(values, "--seed")) != args.seed:
        raise ValueError("B3 wrapper/evaluator seeds differ")
    mode = _one_argument(values, "--mode")
    if mode not in ("smoke", "full"):
        raise ValueError("B3 evaluator mode differs")
    return [sys.executable, "-u", str(evaluator), *values], mode


def evaluation_metrics(
    result: Mapping[str, Any],
    generation: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, int | float | bool]:
    aggregate = score["field_and_marginal_calibration"]["regions"][
        "eligible_union"
    ]["aggregate"]
    return {
        "evaluation/stage": 2,
        "evaluation/completed": 1,
        "forecast/target_count": int(generation["target_count"]),
        "forecast/ensemble_size": int(generation["forecast"]["shape"][1]),
        "forecast/wall_seconds": float(generation["wall_seconds"]),
        "forecast/peak_cuda_memory_bytes": int(
            generation["peak_cuda_memory_bytes"]
        ),
        "field/equal_channel_ensemble_mean_rmse": float(
            aggregate["equal_channel_ensemble_mean_rmse"]
        ),
        "field/equal_channel_ensemble_mean_mae": float(
            aggregate["equal_channel_ensemble_mean_mae"]
        ),
        "field/equal_channel_fair_crps": float(
            aggregate["equal_channel_fair_crps"]
        ),
        "field/equal_channel_corrected_spread_skill_ratio": float(
            aggregate["equal_channel_corrected_spread_skill_ratio"]
        ),
        "field/all_fields_nonzero_spread": bool(
            aggregate["all_fields_nonzero_spread"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
    }


def main() -> None:
    args = parse_args()
    assert_development_path(args.output_directory)
    command, mode = validate_evaluation_command(args)
    targets = [498, 502] if mode == "smoke" else [498, 624]
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b3_fgn_probabilistic_evaluation",
        tags=("paper0", "phase3", "b3", "fgn", "evaluation", "85604-only"),
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
            "seed": 1701,
            "model_arm": "B3-FGN-H1",
            "context_frames": 1,
            "ensemble_members": 32,
            "target_frames": targets,
            "evaluation_mode": mode,
            "evaluator_sha256": args.evaluator_sha256,
            "command": command,
            "local_artifacts_are_scientific_authority": True,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline) or str(run.id) != spec.run_id:
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError("W&B online run initialization differs")
    run.log({"evaluation/stage": 0, "evaluation/started": 1}, step=0)
    completed = subprocess.run(command, check=False, env=dict(os.environ))
    if completed.returncode != 0:
        run.summary["evaluation/exit_code"] = int(completed.returncode)
        run.finish(exit_code=completed.returncode)
        raise subprocess.CalledProcessError(completed.returncode, command)

    output = args.output_directory.resolve(strict=True)
    result = load_strict_json(output / "result.json")
    generation = load_strict_json(output / "generation.json")
    score = load_strict_json(output / "score.json")
    expected_scope = (
        "bounded_non_scientific_B3_FGN_H1_evaluator_smoke_85604"
        if mode == "smoke"
        else "B3_FGN_H1_full_probabilistic_evaluation_85604"
    )
    if (
        result.get("scope") != expected_scope
        or result.get("seed") != 1701
        or result.get("paper0_commit") != args.paper0_commit
        or result.get("held_out_85606_read") is not False
        or result.get("target_frames") != targets
    ):
        run.finish(exit_code=1)
        raise RuntimeError("completed B3 FGN evaluation identity differs")
    metrics = evaluation_metrics(result, generation, score)
    run.log(metrics, step=1)
    run.summary.update(
        {
            **metrics,
            "provenance/result_sha256": sha256_path(output / "result.json"),
            "provenance/forecast_sha256": result["forecast"]["sha256"],
            "provenance/score_sha256": result["score"]["sha256"],
            "provenance/scientific_noise_sha256": result["scientific_noise"][
                "sha256"
            ],
            "provenance/paper0_commit": args.paper0_commit,
            "scope/local_artifacts_are_scientific_authority": True,
        }
    )
    run_url = str(run.url)
    run.finish(exit_code=0)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote = wandb.Api(timeout=30).run(remote_path)
    if str(remote.state) != "finished":
        raise RuntimeError("remote W&B B3 evaluation run is not finished")
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "evaluation_mode": mode,
        "spec": spec.to_record(),
        "authenticated_username": str(getattr(viewer, "username", "")),
        "wandb_version": str(wandb.__version__),
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": str(remote.state),
        "checkpoints_uploaded": False,
        "forecasts_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    write_strict_json_atomic(output / "wandb.json", tracking)
    print(json.dumps(tracking, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
