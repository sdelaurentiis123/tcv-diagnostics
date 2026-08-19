#!/usr/bin/env python3
"""Run frozen B5 covariance localization inside one online W&B run.

Only compact scalar metrics, provenance hashes, command/run metadata, and the
W&B run record are sent to W&B.  Forecasts, simulation fields, raw
accumulators, figures, tables, and checkpoints remain local scientific
artifacts on Ceph.
"""

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
    parser.add_argument("--slurm-job-id", required=True)
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
        raise ValueError(f"localization command must contain exactly one {flag}")
    return str(values[indices[0] + 1])


def validate_localization_command(args: argparse.Namespace) -> list[str]:
    """Lock the wrapped command to the approved read-only localization."""

    evaluator = Path(args.evaluator).resolve(strict=True)
    assert_development_path(evaluator)
    if sha256_path(evaluator) != args.evaluator_sha256:
        raise ValueError("B5 covariance-localization evaluator SHA-256 differs")
    values = list(args.evaluation_args)
    if values and values[0] == "--":
        values = values[1:]
    lowered = [value.lower() for value in values]
    if not values or any("85606" in value for value in lowered):
        raise ValueError("localization command is empty or mentions held-out data")
    forbidden_flags = {
        "--checkpoint",
        "--mode",
        "--train",
        "--training",
        "--model-seed",
    }
    if forbidden_flags.intersection(values):
        raise ValueError("localization command contains a training or inference flag")
    if Path(_one_argument(values, "--output-directory")) != args.output_directory:
        raise ValueError("W&B wrapper/evaluator output directories differ")
    if _one_argument(values, "--paper0-commit") != args.paper0_commit:
        raise ValueError("W&B wrapper/evaluator commits differ")
    if _one_argument(values, "--slurm-job-id") != args.slurm_job_id:
        raise ValueError("W&B wrapper/evaluator Slurm job IDs differ")
    return [sys.executable, "-u", str(evaluator), *values]


def _add_optional_metric(
    metrics: dict[str, int | float | bool], key: str, value: Any
) -> None:
    if value is not None:
        metrics[key] = float(value)


def localization_metrics(
    result: Mapping[str, Any], localization: Mapping[str, Any]
) -> dict[str, int | float | bool]:
    """Project the local scientific result to compact scalar monitoring data."""

    anchor = localization["integrity_anchors"]["B5_marginal_recomputation"][
        "recomputed"
    ]
    history = localization["history_probe"]
    labels = localization["interpretation_labels"]
    l3 = localization["blockwise_L3"]
    metrics: dict[str, int | float | bool] = {
        "localization/completed": 1,
        "localization/target_count": int(result["target_count"]),
        "localization/ensemble_size": int(result["ensemble_size"]),
        "localization/wall_seconds": float(result["wall_seconds"]),
        "marginal/equal_channel_ensemble_mean_RMSE": float(
            anchor["equal_channel_ensemble_mean_RMSE"]
        ),
        "marginal/equal_channel_corrected_spread_skill_ratio": float(
            anchor["equal_channel_corrected_spread_skill_ratio"]
        ),
        "history/AR1_vs_H1_RMSE_improvement_fraction": float(
            history["AR1_vs_H1_equal_field_RMSE_improvement_fraction"]
        ),
        "history/AR1_improved_chronological_block_count": int(
            history["AR1_improved_chronological_comparison_count"]
        ),
        "dependence/systematic_identity_count": int(l3["systematic_identity_count"]),
        "label/L1_amplitude_limited": bool(
            labels["L1_predominantly_amplitude_limited"]["supported"]
        ),
        "label/L2_covariance_organization_limited": bool(
            labels["L2_covariance_organization_limited"]["supported"]
        ),
        "label/L3_dependence_mismatch_beyond_drift": bool(
            l3["L3_field_dependence_mismatch_beyond_within_run_drift_supported"]
        ),
        "label/L4_explicit_residual_history_signal": bool(
            labels["L4_explicit_residual_history_signal"]["supported"]
        ),
        "label/L5_unresolved_by_one_trajectory": bool(
            labels["L5_unresolved_by_one_realized_trajectory"]["supported"]
        ),
        "integrity/legacy_training_reconstruction_passed": bool(
            localization["integrity_anchors"]["legacy_training_reconstruction"][
                "passed"
            ]
        ),
        "scope/checkpoint_loaded": bool(result["checkpoint_loaded"]),
        "scope/model_inference_performed": bool(result["model_inference_performed"]),
        "scope/model_training_performed": bool(result["model_training_performed"]),
        "scope/forecast_mutated": bool(result["forecast_mutated"]),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
    }
    for region, value in localization["variogram_scores"]["field"][
        "aggregate_region_mean"
    ].items():
        metrics[f"field_variogram/{region}"] = float(value)
    transport_variogram = localization["variogram_scores"]["transport"][
        "aggregate_equal_lag_mean"
    ]
    associations = localization["spread_error_association"]["integrated_transport"]
    for quantity, record in localization["transport_covariance"]["quantities"].items():
        covariance = record["covariance_decomposition"]
        prefix = f"transport/{quantity}"
        metrics[f"{prefix}/local_spread_skill"] = float(
            covariance["local_corrected_spread_skill_ratio"]
        )
        metrics[f"{prefix}/integrated_spread_skill"] = float(
            covariance["integrated_corrected_spread_skill_ratio"]
        )
        metrics[f"{prefix}/coherence_multiplier_ratio"] = float(
            covariance["ensemble_to_error_coherence_multiplier_ratio"]
        )
        metrics[f"{prefix}/scalar_inflation_local_counterfactual"] = float(
            covariance["counterfactual_local_spread_skill_after_same_factor"]
        )
        metrics[f"{prefix}/variogram"] = float(transport_variogram[quantity])
        _add_optional_metric(
            metrics,
            f"{prefix}/spread_error_pearson",
            associations[quantity]["pearson"],
        )
        _add_optional_metric(
            metrics,
            f"{prefix}/spread_error_spearman",
            associations[quantity]["spearman"],
        )
    return metrics


def validate_completed_localization(
    *,
    result: Mapping[str, Any],
    localization: Mapping[str, Any],
    paper0_commit: str,
    slurm_job_id: str,
) -> None:
    if (
        result.get("scope") != "B5_read_only_covariance_localization_85604"
        or result.get("status") != "completed_without_retraining_or_downstream_opening"
        or result.get("paper0_commit") != paper0_commit
        or str(result.get("slurm_job_id")) != str(slurm_job_id)
        or localization.get("scope") != "B5_read_only_covariance_localization_85604"
        or localization.get("paper0_commit") != paper0_commit
        or str(localization.get("slurm_job_id")) != str(slurm_job_id)
    ):
        raise RuntimeError("completed B5 covariance-localization identity differs")
    false_scope = (
        "checkpoint_loaded",
        "model_inference_performed",
        "model_training_performed",
        "forecast_mutated",
        "additional_seed_trained",
        "O3_launched",
        "assimilation_performed",
        "diagnostic_ranking_performed",
        "held_out_85606_read",
    )
    if any(result.get(flag) is not False for flag in false_scope):
        raise RuntimeError("completed localization opened a forbidden scope")
    if (
        localization.get("variogram_scores", {}).get("used_as_training_loss")
        is not False
    ):
        raise RuntimeError("localization variogram was unexpectedly used for training")
    if (
        localization.get("integrity_anchors", {})
        .get("B5_marginal_recomputation", {})
        .get("passed")
        is not True
    ):
        raise RuntimeError("localization marginal integrity anchor failed")


def main() -> None:
    args = parse_args()
    assert_development_path(args.output_directory)
    command = validate_localization_command(args)
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_b5_covariance_localization",
        tags=(
            "paper0",
            "phase3",
            "b5",
            "covariance-localization",
            "85604-only",
            "CPU-only",
        ),
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
            "slurm_job_id": args.slurm_job_id,
            "development_run": "85604",
            "held_out_85606_access_allowed": False,
            "target_frames": [498, 624],
            "target_count": 126,
            "ensemble_size": 32,
            "mode_mapping": "n=5k",
            "training_performed": False,
            "model_inference_performed": False,
            "raw_artifacts_uploaded": False,
            "local_artifacts_are_scientific_authority": True,
            "approved_upload_scope": (
                "localization metrics, provenance hashes, command/run metadata, "
                "and W&B run metadata"
            ),
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
        raise RuntimeError("W&B online covariance-localization initialization differs")
    run.log({"localization/stage": 0, "localization/started": 1}, step=0)
    completed = subprocess.run(command, check=False, env=dict(os.environ))
    if completed.returncode != 0:
        run.summary["localization/exit_code"] = int(completed.returncode)
        run.finish(exit_code=completed.returncode)
        raise subprocess.CalledProcessError(completed.returncode, command)

    output = args.output_directory.resolve(strict=True)
    result = load_strict_json(output / "result.json")
    localization = load_strict_json(output / "covariance_localization.json")
    try:
        validate_completed_localization(
            result=result,
            localization=localization,
            paper0_commit=args.paper0_commit,
            slurm_job_id=args.slurm_job_id,
        )
        metrics = localization_metrics(result, localization)
        run.log({"localization/stage": 1, **metrics}, step=1)
        run.summary.update(
            {
                **metrics,
                "provenance/result_sha256": sha256_path(output / "result.json"),
                "provenance/localization_sha256": sha256_path(
                    output / "covariance_localization.json"
                ),
                "provenance/raw_accumulators_sha256": sha256_path(
                    output / "raw_accumulators.npz"
                ),
                "provenance/paper0_commit": args.paper0_commit,
                "scope/raw_artifacts_uploaded": False,
                "scope/local_artifacts_are_scientific_authority": True,
            }
        )
    except Exception:
        run.finish(exit_code=1)
        raise
    run_url = str(run.url)
    run.finish(exit_code=0)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote = wandb.Api(timeout=30).run(remote_path)
    if str(remote.id) != spec.run_id or str(remote.state) != "finished":
        raise RuntimeError("remote W&B covariance-localization run is not finished")
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "approved_upload_scope": (
            "localization metrics, provenance hashes, command/run metadata, "
            "and W&B run metadata"
        ),
        "spec": spec.to_record(),
        "authenticated_username": str(getattr(viewer, "username", "")),
        "wandb_version": str(wandb.__version__),
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_presence_verified_after_finish": True,
        "remote_state_after_finish": str(remote.state),
        "checkpoints_uploaded": False,
        "forecasts_uploaded": False,
        "simulation_fields_uploaded": False,
        "raw_accumulators_uploaded": False,
        "figures_uploaded": False,
        "tables_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    write_strict_json_atomic(output / "wandb.json", tracking)
    print(json.dumps(tracking, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
