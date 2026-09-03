#!/usr/bin/env python3
"""Generate a truth-free M32 forecast for one fixed hierarchical checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from paper0.tools.generate_persistent_global_local_forecast import (
    load_selected_models,
    require_runtime,
    verify_checkout,
)
from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from tcv_diagnostics.b5_residual_edm_forecast import load_scientific_sampler_seed_bank
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.o2_context_data import OneStepContextDataset
from tcv_diagnostics.persistent_global_local_authority import (
    authorize_pgl_evaluation_manifest,
)
from tcv_diagnostics.persistent_global_local_forecast import (
    PGL_EVALUATION_STARTS,
    PGL_SCIENTIFIC_SEED_BANK_SHA256,
    PGLForecastSchema,
    PGLForecastWriter,
    SelectedContextDatasetAdapter,
    generate_pgl_forecast,
)
from tcv_diagnostics.pgl_hierarchical_evaluation import (
    authorize_hierarchical_training_result,
    load_hierarchical_checkpoint_state,
)
from tcv_diagnostics.pgl_hierarchical_training import PGL_HIERARCHICAL_ARMS
from tcv_diagnostics.wandb_tracking import WandbRunSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=PGL_HIERARCHICAL_ARMS, required=True)
    parser.add_argument("--optimizer-update", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
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


def main() -> int:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.manifest,
        args.training_result,
        args.checkpoint,
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
        raise ValueError("hierarchical forecast artifact root differs")
    training = authorize_hierarchical_training_result(
        result_path=args.training_result,
        result_sha256=args.training_result_sha256,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        arm=args.arm,
        optimizer_update=args.optimizer_update,
    )
    environment = require_runtime()
    args.output.mkdir(parents=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    selected_mean, parent_mean, model, base_provenance = load_selected_models(
        paths=paths, artifact_root=args.artifact_root, device=device
    )
    checkpoint_provenance = load_hierarchical_checkpoint_state(
        selected_mean=selected_mean,
        stochastic_model=model,
        training_result=training,
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        arm=args.arm,
        optimizer_update=args.optimizer_update,
        device=device,
    )
    catalog = load_official_catalog(args.artifact_root)
    targets = tuple(value + 1 for value in PGL_EVALUATION_STARTS)
    contiguous = OneStepContextDataset(
        catalog,
        target_frames=tuple(range(targets[0], targets[-1] + 1)),
        context_frames=1,
    )
    dataset = SelectedContextDatasetAdapter(contiguous, target_frames=targets)
    seed_bank = load_scientific_sampler_seed_bank(
        paths["scientific_seed_bank"], PGL_SCIENTIFIC_SEED_BANK_SHA256
    )

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
        job_type="old-85604-pgl-hierarchical-M32-generation",
        tags=(
            "paper0",
            "85604",
            "pgl-hierarchical",
            f"arm-{args.arm.lower()}",
            f"update-{args.optimizer_update}",
            "M32",
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key or str(getattr(api.viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B identity differs")
    tracking = args.output / "wandb"
    tracking.mkdir()
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
            "scope": "old_85604_pgl_hierarchical_truth_free_generation",
            "arm": args.arm,
            "optimizer_update": args.optimizer_update,
            "paper0_commit": args.paper0_commit,
            "manifest_sha256": args.manifest_sha256,
            "training_result_sha256": args.training_result_sha256,
            "checkpoint_sha256": args.checkpoint_sha256,
            "physics_derived_training_loss_used": args.arm == "TRANSPORT",
            "target_truth_read": False,
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
    schema = PGLForecastSchema.frozen()
    forecast_path = args.output / "forecast_M32_four_frame.h5"
    try:
        with PGLForecastWriter(
            forecast_path,
            paper0_commit=args.paper0_commit,
            manifest_sha256=args.manifest_sha256,
            training_result_sha256=args.training_result_sha256,
            checkpoint_sha256=args.checkpoint_sha256,
            seed_bank_path=paths["scientific_seed_bank"],
            seed_bank_sha256=PGL_SCIENTIFIC_SEED_BANK_SHA256,
            schema=schema,
        ) as writer:
            def on_start(record):
                run.log(
                    {
                        "generation/completed_starts": int(record["completed_starts"]),
                        "generation/current_frame": int(record["current_frame"]),
                        "timing/inference_seconds": float(record["inference_seconds"]),
                    },
                    step=int(record["completed_starts"]),
                )

            generation = generate_pgl_forecast(
                selected_mean=selected_mean,
                parent_mean=parent_mean,
                model=model,
                dataset=dataset,
                writer=writer,
                seed_bank=seed_bank,
                device=device,
                member_batch_size=8,
                on_start=on_start,
            )
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        result = {
            **generation,
            "scope": "old_85604_pgl_hierarchical_truth_free_forecast_generation",
            "status": "truth_free_forecast_completed_and_hash_closed",
            "arm": args.arm,
            "optimizer_update": args.optimizer_update,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": {"path": str(args.manifest), "sha256": args.manifest_sha256},
            "training_result": {
                "path": str(args.training_result),
                "sha256": args.training_result_sha256,
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": args.checkpoint_sha256,
            },
            "provenance": {
                "base_persistent_model": base_provenance,
                "hierarchical_warm_start": checkpoint_provenance,
            },
            "environment": environment,
            "peak_cuda_memory_bytes": peak_bytes,
            "peak_cuda_memory_GiB": peak_bytes / 2**30,
            "physics_derived_training_loss_used": args.arm == "TRANSPORT",
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_scoring_authorized_next": True,
        }
        atomic_json(args.output / "result.json", result)
        run.summary.update(
            {
                "final/status": result["status"],
                "final/forecast_sha256": result["forecast"]["sha256"],
                "final/start_count": result["start_count"],
                "compute/peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
                "scope/target_truth_read": False,
                "scope/held_out_85606_read": False,
            }
        )
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        contiguous.close()
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
