#!/usr/bin/env python3
"""Run the closed residual-KL build and evaluation in one compact W&B run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcv_diagnostics.b5_residual_audit import B5_FIELDS  # noqa: E402
from tcv_diagnostics.codec_training import sha256_path  # noqa: E402
from tcv_diagnostics.codec_transport import TRANSPORT_QUANTITIES  # noqa: E402
from tcv_diagnostics.model_data import (  # noqa: E402
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--builder-sha256", required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--oracle-manifest-sha256", required=True)
    parser.add_argument("--oracle-protocol", type=Path, required=True)
    parser.add_argument("--oracle-protocol-sha256", required=True)
    parser.add_argument("--decision-memo", type=Path, required=True)
    parser.add_argument("--decision-memo-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--h1-training-forecast", type=Path, required=True)
    parser.add_argument("--h1-training-forecast-sha256", required=True)
    parser.add_argument("--h1-validation-forecast", type=Path, required=True)
    parser.add_argument("--h1-validation-forecast-sha256", required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--training-audit-sha256", required=True)
    parser.add_argument("--training-raw", type=Path, required=True)
    parser.add_argument("--training-raw-sha256", required=True)
    parser.add_argument("--native-truth-result", type=Path, required=True)
    parser.add_argument("--native-truth-result-sha256", required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-manifest-sha256", required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--b5-localization-result", type=Path, required=True)
    parser.add_argument("--b5-localization-result-sha256", required=True)
    parser.add_argument("--pretruth-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def _program(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    assert_development_path(resolved)
    if sha256_path(resolved) != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def validate_scope(args: argparse.Namespace) -> tuple[Path, Path]:
    """Reject held-out paths, mutable programs, or pre-existing outputs."""

    builder = _program(args.builder, args.builder_sha256, "residual-KL builder")
    evaluator = _program(
        args.evaluator, args.evaluator_sha256, "residual-KL evaluator"
    )
    path_values = (
        args.oracle_manifest,
        args.oracle_protocol,
        args.decision_memo,
        args.artifact_root,
        args.h1_training_forecast,
        args.h1_validation_forecast,
        args.training_audit,
        args.training_raw,
        args.native_truth_result,
        args.geometry_manifest,
        args.geometry,
        args.b5_localization_result,
        args.pretruth_output,
        args.evaluation_output,
        args.scratch_root,
    )
    for path in path_values:
        assert_development_path(Path(path))
    outputs = (
        Path(args.pretruth_output),
        Path(args.evaluation_output),
        Path(args.scratch_root) / "pretruth",
        Path(args.scratch_root) / "evaluation",
    )
    if len({str(path.resolve()) for path in outputs}) != len(outputs):
        raise ValueError("residual-KL output and scratch paths overlap")
    if any(path.exists() for path in outputs):
        raise FileExistsError("residual-KL output or scratch path already exists")
    if len(args.paper0_commit) != 40 or not args.slurm_job_id:
        raise ValueError("residual-KL commit or Slurm identity differs")
    return builder, evaluator


def _common(args: argparse.Namespace) -> list[str]:
    return [
        "--oracle-manifest", str(args.oracle_manifest),
        "--oracle-manifest-sha256", args.oracle_manifest_sha256,
        "--oracle-protocol", str(args.oracle_protocol),
        "--oracle-protocol-sha256", args.oracle_protocol_sha256,
        "--decision-memo", str(args.decision_memo),
        "--decision-memo-sha256", args.decision_memo_sha256,
        "--artifact-root", str(args.artifact_root),
        "--h1-validation-forecast", str(args.h1_validation_forecast),
        "--h1-validation-forecast-sha256", args.h1_validation_forecast_sha256,
        "--geometry-manifest", str(args.geometry_manifest),
        "--geometry-manifest-sha256", args.geometry_manifest_sha256,
        "--geometry", str(args.geometry),
        "--geometry-sha256", args.geometry_sha256,
        "--b5-localization-result", str(args.b5_localization_result),
        "--b5-localization-result-sha256", args.b5_localization_result_sha256,
        "--paper0-commit", args.paper0_commit,
        "--slurm-job-id", args.slurm_job_id,
    ]


def commands(
    args: argparse.Namespace,
    *,
    builder: Path,
    evaluator: Path,
    closure_sha256: str | None = None,
) -> tuple[list[str], list[str] | None]:
    builder_command = [
        sys.executable,
        "-u",
        str(builder),
        *_common(args),
        "--h1-training-forecast", str(args.h1_training_forecast),
        "--h1-training-forecast-sha256", args.h1_training_forecast_sha256,
        "--training-audit", str(args.training_audit),
        "--training-audit-sha256", args.training_audit_sha256,
        "--training-raw", str(args.training_raw),
        "--training-raw-sha256", args.training_raw_sha256,
        "--output-directory", str(args.pretruth_output),
        "--scratch-directory", str(Path(args.scratch_root) / "pretruth"),
    ]
    if closure_sha256 is None:
        return builder_command, None
    evaluator_command = [
        sys.executable,
        "-u",
        str(evaluator),
        *_common(args),
        "--pretruth-closure", str(Path(args.pretruth_output) / "pretruth_closure.json"),
        "--pretruth-closure-sha256", closure_sha256,
        "--native-truth-result", str(args.native_truth_result),
        "--native-truth-result-sha256", args.native_truth_result_sha256,
        "--output-directory", str(args.evaluation_output),
        "--scratch-directory", str(Path(args.scratch_root) / "evaluation"),
    ]
    return builder_command, evaluator_command


def validate_completed(
    *,
    pretruth: Mapping[str, Any],
    result: Mapping[str, Any],
    scientific: Mapping[str, Any],
    paper0_commit: str,
    slurm_job_id: str,
    closure_sha256: str,
) -> None:
    if (
        pretruth.get("status") != "completed_and_closed_before_validation_truth"
        or result.get("status")
        != "completed_without_model_training_or_downstream_opening"
        or scientific.get("status")
        != "completed_without_model_training_or_downstream_opening"
    ):
        raise RuntimeError("residual-KL completion status differs")
    for record in (pretruth, result, scientific):
        if (
            record.get("development_run") != "85604"
            or record.get("held_out_85606_read") is not False
            or record.get("guard_frames_read") is not False
            or record.get("paper0_commit") != paper0_commit
            or str(record.get("slurm_job_id")) != str(slurm_job_id)
        ):
            raise RuntimeError("residual-KL completed identity differs")
    if scientific.get("pretruth_closure", {}).get("sha256") != closure_sha256:
        raise RuntimeError("residual-KL pretruth/evaluation closure differs")
    boundaries = scientific["scientific_boundaries"]
    for flag in (
        "checkpoint_loaded",
        "model_inference_performed",
        "optimizer_or_trainable_parameter_created",
        "model_training_performed",
        "physics_metric_used_as_training_loss",
        "O3_launched",
        "O4_launched",
        "O5_launched",
        "assimilation_performed",
        "diagnostic_ranking_performed",
        "steering_performed",
        "held_out_85606_read",
    ):
        if boundaries.get(flag) is not False:
            raise RuntimeError(f"residual-KL forbidden scope opened: {flag}")
    if scientific.get("validation_truth_opened_only_after_pretruth_closure_verified") is not True:
        raise RuntimeError("residual-KL validation-truth barrier differs")
    ranks = scientific["tier_A_truth_projected_representation_oracle"]
    if not ranks or any(len(record.get("chronological_blocks", {})) != 6 for record in ranks.values()):
        raise RuntimeError("residual-KL Tier-A block inventory differs")
    static = scientific["tier_B_static_Gaussian_KL"]
    if len(
        static["field_and_marginal_calibration"][
            "chronological_blocks_eligible_union"
        ]
    ) != 6:
        raise RuntimeError("residual-KL Tier-B field blocks differ")


def compact_metrics(
    result: Mapping[str, Any], scientific: Mapping[str, Any]
) -> dict[str, int | float | bool]:
    """Return only scalar monitoring metrics; local files remain authoritative."""

    training = scientific["training_basis_summary"]
    static = scientific["tier_B_static_Gaussian_KL"]
    aggregate = static["field_and_marginal_calibration"]["regions"][
        "eligible_union"
    ]["aggregate"]
    metrics: dict[str, int | float | bool] = {
        "oracle/completed": 1,
        "oracle/positive_rank": int(result["positive_rank"]),
        "oracle/selected_static_rank": int(result["selected_static_rank"]),
        "oracle/tier_A_has_passing_rank": result["tier_A_minimum_passing_rank"] is not None,
        "oracle/tier_A_minimum_passing_rank": int(result["tier_A_minimum_passing_rank"] or 0),
        "oracle/tier_B_static_covariance_useful": bool(result["tier_B_static_covariance_useful"]),
        "oracle/wall_seconds": float(result["wall_seconds"]),
        "training/participation_ratio_rank": float(training["effective_rank"]["participation_ratio"]),
        "training/entropy_effective_rank": float(training["effective_rank"]["entropy_effective_rank"]),
        "static/aggregate_field_corrected_spread_skill": float(aggregate["equal_channel_corrected_spread_skill_ratio"]),
        "static/aggregate_field_RMSE": float(aggregate["equal_channel_ensemble_mean_rmse"]),
        "static/aggregate_field_fair_CRPS": float(aggregate["equal_channel_fair_crps"]),
        "scope/held_out_85606_read": False,
        "scope/model_training_performed": False,
        "scope/model_inference_performed": False,
    }
    for key, record in scientific["tier_A_truth_projected_representation_oracle"].items():
        prefix = f"tier_A/{key}"
        metrics[f"{prefix}/total_variance_capture"] = float(record["variance_capture"]["total"])
        metrics[f"{prefix}/gate_passed"] = bool(record["representation_gate"]["passes"])
        metrics[f"{prefix}/dependence_identity_pass_count"] = int(record["dependence"]["identity_pass_count"])
        metrics[f"{prefix}/material_power_pass_count"] = int(record["material_power"]["in_range_count"])
    for field in B5_FIELDS:
        record = static["field_and_marginal_calibration"]["regions"][
            "eligible_union"
        ]["fields"][field]
        metrics[f"static_field/{field}/RMSE"] = float(
            record["ensemble_mean"]["rmse"]
        )
        metrics[f"static_field/{field}/fair_CRPS"] = float(record["fair_crps"])
        metrics[f"static_field/{field}/corrected_spread_skill"] = float(
            record["corrected_spread_skill"]["ratio"]
        )
    for quantity in TRANSPORT_QUANTITIES:
        covariance = static["transport_covariance"]["quantities"][quantity][
            "covariance_decomposition"
        ]
        prefix = f"static_transport/{quantity}"
        metrics[f"{prefix}/local_spread_skill"] = float(covariance["local_corrected_spread_skill_ratio"])
        metrics[f"{prefix}/integrated_spread_skill"] = float(covariance["integrated_corrected_spread_skill_ratio"])
        metrics[f"{prefix}/coherence_multiplier_ratio"] = float(covariance["ensemble_to_error_coherence_multiplier_ratio"])
    return metrics


def main() -> None:
    args = parse_args()
    builder, evaluator = validate_scope(args)
    builder_command, _ = commands(args, builder=builder, evaluator=evaluator)
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="phase3_residual_KL_oracle",
        tags=("paper0", "phase3", "residual-KL", "oracle", "85604-only", "CPU-only"),
    )
    import wandb

    api = wandb.Api(timeout=30)
    viewer = api.viewer
    if not api.api_key or str(getattr(viewer, "entity", "")) != spec.entity:
        raise RuntimeError("online W&B authentication differs")
    tracking_directory = Path(args.pretruth_output).parent / ".residual-kl-wandb"
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
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "training_performed": False,
            "model_inference_performed": False,
            "raw_artifacts_uploaded": False,
            "builder_command": builder_command,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline) or str(run.id) != spec.run_id:
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError("W&B online residual-KL initialization differs")
    try:
        run.log({"oracle/stage": 0, "oracle/pretruth_started": 1}, step=0)
        subprocess.run(builder_command, check=True, env=dict(os.environ))
        pretruth_path = Path(args.pretruth_output) / "pretruth_result.json"
        closure_path = Path(args.pretruth_output) / "pretruth_closure.json"
        pretruth = load_strict_json(pretruth_path)
        closure_sha256 = sha256_path(closure_path)
        run.log(
            {
                "oracle/stage": 1,
                "oracle/pretruth_closed": 1,
                "oracle/positive_rank": int(pretruth["positive_rank"]),
                "oracle/selected_static_rank": int(pretruth["selected_static_rank"]),
                "oracle/pretruth_wall_seconds": float(pretruth["wall_seconds"]),
            },
            step=1,
        )
        _, evaluator_command = commands(
            args,
            builder=builder,
            evaluator=evaluator,
            closure_sha256=closure_sha256,
        )
        if evaluator_command is None:
            raise RuntimeError("residual-KL evaluator command was not constructed")
        run.config.update({"evaluator_command": evaluator_command}, allow_val_change=True)
        subprocess.run(evaluator_command, check=True, env=dict(os.environ))
        output = Path(args.evaluation_output).resolve(strict=True)
        result = load_strict_json(output / "result.json")
        scientific = load_strict_json(output / "residual_kl_oracle.json")
        validate_completed(
            pretruth=pretruth,
            result=result,
            scientific=scientific,
            paper0_commit=args.paper0_commit,
            slurm_job_id=args.slurm_job_id,
            closure_sha256=closure_sha256,
        )
        metrics = compact_metrics(result, scientific)
        run.log({"oracle/stage": 2, **metrics}, step=2)
        run.summary.update(
            {
                **metrics,
                "oracle/primary_outcome": result["primary_outcome"],
                "provenance/pretruth_closure_sha256": closure_sha256,
                "provenance/result_sha256": sha256_path(output / "result.json"),
                "provenance/scientific_sha256": sha256_path(output / "residual_kl_oracle.json"),
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
        raise RuntimeError("remote W&B residual-KL run is not finished")
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "approved_upload_scope": "compact scalar metrics, provenance hashes, and run metadata",
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
        "basis_arrays_uploaded": False,
        "raw_accumulators_uploaded": False,
        "figures_uploaded": False,
        "tables_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    write_strict_json_atomic(Path(args.evaluation_output) / "wandb.json", tracking)
    print(json.dumps(tracking, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
