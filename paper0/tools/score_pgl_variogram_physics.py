#!/usr/bin/env python3
"""Truth-separately score one closed PGL variogram M32 forecast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper0.tools.score_persistent_global_local_physics import (
    require_runtime,
    verify_checkout,
)
from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.matched_o1_transport import NativeTruthCatalog, load_transport_geometry
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.persistent_global_local_authority import (
    authorize_pgl_evaluation_manifest,
)
from tcv_diagnostics.persistent_global_local_forecast import (
    PGL_SCIENTIFIC_SEED_BANK_SHA256,
    PGLForecastArtifact,
)
from tcv_diagnostics.persistent_global_local_physics import (
    score_persistent_global_local_forecast,
)
from tcv_diagnostics.pgl_variogram_evaluation import (
    authorize_variogram_training_result,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--generation-result", type=Path, required=True)
    parser.add_argument("--generation-result-sha256", required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--forecast-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def verify_generation(args: argparse.Namespace) -> dict:
    if sha256_path(args.generation_result) != args.generation_result_sha256:
        raise ValueError("variogram generation-result SHA-256 differs")
    if sha256_path(args.forecast) != args.forecast_sha256:
        raise ValueError("variogram forecast SHA-256 differs")
    result = load_strict_json(args.generation_result)
    if (
        result.get("scope")
        != "old_85604_pgl_variogram_truth_free_forecast_generation"
        or result.get("status") != "truth_free_forecast_completed_and_hash_closed"
        or result.get("arm") != args.arm
        or result.get("paper0_commit") != args.paper0_commit
        or result.get("manifest", {}).get("sha256") != args.manifest_sha256
        or result.get("training_result", {}).get("sha256")
        != args.training_result_sha256
        or result.get("checkpoint", {}).get("sha256") != args.checkpoint_sha256
        or result.get("forecast", {}).get("path") != str(args.forecast)
        or result.get("forecast", {}).get("sha256") != args.forecast_sha256
        or result.get("start_count") != 36
        or result.get("ensemble_members") != 32
        or result.get("future_frames") != 4
        or result.get("target_truth_read") is not False
        or result.get("physics_diagnostics_scored") is not False
        or result.get("checkpoint_selection_performed") is not False
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
    ):
        raise ValueError("variogram truth-free generation contract differs")
    return result


def main() -> int:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.manifest,
        args.training_result,
        args.checkpoint,
        args.generation_result,
        args.forecast,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_checkout(args.paper0_root, args.paper0_commit)
    manifest = load_strict_json(args.manifest)
    paths = authorize_pgl_evaluation_manifest(
        manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        paper0_root=args.paper0_root,
    )
    if paths["model_dataset"] != args.artifact_root.resolve(strict=True):
        raise ValueError("variogram scoring model dataset differs")
    authorize_variogram_training_result(
        result_path=args.training_result,
        result_sha256=args.training_result_sha256,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        arm=args.arm,
    )
    generation = verify_generation(args)
    environment = require_runtime()
    args.output.mkdir(parents=True)
    catalog = load_official_catalog(args.artifact_root)
    native_truth = NativeTruthCatalog(load_strict_json(paths["native_truth_result"]))
    geometry = load_transport_geometry(
        geometry_path=paths["geometry"],
        geometry_manifest=load_strict_json(paths["geometry_manifest"]),
    )
    thresholds = load_strict_json(paths["event_threshold_result"])

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("online W&B is required") from error
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type="old-85604-pgl-variogram-physics-scoring",
        tags=("paper0", "85604", "pgl-variogram", f"arm-{args.arm}", "physics"),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key or str(getattr(api.viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B identity differs")
    tracking = args.output / "wandb"
    tracking.mkdir()
    physics_loss = args.arm in ("C", "D")
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
            "scope": "old_85604_pgl_variogram_truth_separated_scoring",
            "arm": args.arm,
            "paper0_commit": args.paper0_commit,
            "manifest_sha256": args.manifest_sha256,
            "forecast_sha256": args.forecast_sha256,
            "physics_derived_training_loss_used": physics_loss,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
        },
        mode="online",
        dir=str(tracking),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")
    try:
        with PGLForecastArtifact(
            args.forecast,
            expected_sha256=args.forecast_sha256,
            manifest_sha256=args.manifest_sha256,
            training_result_sha256=args.training_result_sha256,
            checkpoint_sha256=args.checkpoint_sha256,
            seed_bank_path=paths["scientific_seed_bank"],
            seed_bank_sha256=PGL_SCIENTIFIC_SEED_BANK_SHA256,
        ) as artifact:
            score = score_persistent_global_local_forecast(
                catalog=catalog,
                forecast_artifact=artifact,
                native_truth=native_truth,
                geometry=geometry,
                event_threshold_record=thresholds,
            )
            timing = artifact.timing_record()
        score["scope"] = "old_85604_pgl_variogram_truth_separated_physics_scoring"
        score["arm"] = args.arm
        score["physics_derived_training_loss_used"] = physics_loss
        score["variogram_training_interpretation"] = (
            "field_and_authoritative_transport_auxiliary_loss"
            if args.arm == "D"
            else "authoritative_transport_auxiliary_loss"
            if args.arm == "C"
            else "field_auxiliary_loss"
            if args.arm == "B"
            else "ordinary_loss_control"
        )
        score_path = args.output / "score.json"
        atomic_json(score_path, score)
        gate = score["gate"]
        result = {
            "schema_version": 1,
            "scope": "old_85604_pgl_variogram_physics_evaluation",
            "status": "completed_passed" if gate["passed"] else "completed_failed",
            "development_run": "85604",
            "arm": args.arm,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": {"path": str(args.manifest), "sha256": args.manifest_sha256},
            "training_result": {
                "path": str(args.training_result),
                "sha256": args.training_result_sha256,
            },
            "checkpoint": {"path": str(args.checkpoint), "sha256": args.checkpoint_sha256},
            "generation": {
                "path": str(args.generation_result),
                "sha256": args.generation_result_sha256,
                "forecast": {"path": str(args.forecast), "sha256": args.forecast_sha256},
                "timing": timing,
            },
            "score": {"path": str(score_path), "sha256": sha256_path(score_path)},
            "gate": gate,
            "environment": environment,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "target_truth_used_during_generation": False,
            "physics_derived_training_loss_used": physics_loss,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "confirmation_seeds_authorized": bool(gate["passed"]),
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
        }
        atomic_json(args.output / "result.json", result)
        summary = {
            "final/passed": bool(gate["passed"]),
            **{f"gate/{name}": bool(value) for name, value in gate["family_pass"].items()},
            "scope/physics_derived_training_loss_used": physics_loss,
            "scope/held_out_85606_read": False,
        }
        run.summary.update(summary)
        run.log(summary)
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb, remote_path=remote_path, expected_id=spec.run_id
    )
    wandb_record = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": spec.to_record(),
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_state_after_finish": remote_state,
        "local_artifacts_are_scientific_authority": True,
    }
    atomic_json(args.output / "wandb.json", wandb_record)
    lines = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and "wandb" not in path.parts and path.name != "artifact_sha256.txt":
            lines.append(f"{sha256_path(path)}  {path}\n")
    (args.output / "artifact_sha256.txt").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"result": result, "wandb": wandb_record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

