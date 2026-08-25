#!/usr/bin/env python3
"""Generate the frozen truth-free M32 persistent global--local forecast."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from paper0.tools.train_persistent_global_local_pilot import (
    authorize_manifest as authorize_training_manifest,
    exact_model_config,
    exact_noise_config,
    load_parent,
)
from tcv_diagnostics.b5_residual_edm_forecast import (
    load_scientific_sampler_seed_bank,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
)
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.models.persistent_global_local import PersistentGlobalLocalEDM
from tcv_diagnostics.o2_context_data import OneStepContextDataset
from tcv_diagnostics.persistent_global_local_authority import (
    PGL_EVALUATION_SCOPE,
    PGL_SELECTED_CHECKPOINT_SHA256,
    PGL_TRAINING_RESULT_SHA256,
    authorize_pgl_evaluation_manifest,
    load_authorized_training_result,
)
from tcv_diagnostics.persistent_global_local_forecast import (
    PGL_EVALUATION_STARTS,
    PGL_SCIENTIFIC_SEED_BANK_SHA256,
    PGLForecastSchema,
    PGLForecastWriter,
    SelectedContextDatasetAdapter,
    generate_pgl_forecast,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
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


def verify_checkout(root: Path, expected_commit: str) -> None:
    actual = subprocess.run(
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
    if actual != str(expected_commit) or dirty:
        raise RuntimeError("persistent evaluation checkout is not the locked clean commit")


def require_runtime() -> dict[str, Any]:
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            release[name] = value.strip().strip('"')
    if release.get("ID") != "rocky" or release.get("VERSION_ID", "").split(".")[0] != "9":
        raise RuntimeError("persistent scientific forecast requires Rocky Linux 9")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("persistent scientific forecast requires one allocated GPU")
    return {
        "os_id": release["ID"],
        "os_version": release["VERSION_ID"],
        "host": platform.node(),
        "accelerator": torch.cuda.get_device_name(0),
        "cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def load_selected_models(
    *,
    paths: Mapping[str, Path],
    artifact_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, PersistentGlobalLocalEDM, dict[str, Any]]:
    training_manifest = load_strict_json(paths["training_manifest"])
    authorize_training_manifest(
        training_manifest,
        mode="pilot",
        seed=1702,
        artifact_root=artifact_root,
    )
    result = load_authorized_training_result(paths["training_result"])
    result_checkpoint = Path(
        str(result["selected_checkpoint"]["path"])
    ).resolve(strict=True)
    if (
        result_checkpoint != paths["selected_checkpoint"]
        or result["selected_checkpoint"]["sha256"] != PGL_SELECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("persistent selected result/checkpoint identity differs")
    parent, parent_record, _ = load_parent(training_manifest, device=device)
    candidate_mean = copy.deepcopy(parent).to(device, torch.float32)
    scale_record = load_strict_json(paths["residual_scales"])
    if (
        scale_record.get("scope")
        != "old_85604_persistent_global_local_parent_residual_scales"
        or scale_record.get("development_run") != "85604"
        or scale_record.get("training_frames") != [0, 432]
        or scale_record.get("fields") != ["Ne", "Pe", "Pi", "phi", "Vi"]
        or scale_record.get("physics_derived_quantity") is not False
        or scale_record.get("held_out_85606_read") is not False
    ):
        raise ValueError("persistent residual-scale artifact differs")
    scales = torch.tensor(scale_record["values"], device=device, dtype=torch.float32)
    model = PersistentGlobalLocalEDM(
        exact_model_config(),
        residual_scales=scales,
        noise_config=exact_noise_config(),
    ).to(device, torch.float32)
    payload = torch.load(
        paths["selected_checkpoint"], map_location=device, weights_only=True
    )
    if (
        payload.get("kind") != "persistent_global_local_selected_EMA_checkpoint"
        or payload.get("completed_epoch") != 20
        or payload.get("paper0_commit") != result.get("paper0_commit")
        or payload.get("state_gate", {}).get("passed") is not True
        or payload.get("equivariance_gate", {}).get("passed") is not True
    ):
        raise ValueError("persistent selected checkpoint payload differs")
    candidate_mean.load_state_dict(payload["mean_model_state"], strict=True)
    model.load_state_dict(payload["stochastic_model_state"], strict=True)
    exact = all(
        torch.equal(value.to(device), candidate_mean.state_dict()[name])
        for name, value in payload["mean_model_state"].items()
    ) and all(
        torch.equal(value.to(device), model.state_dict()[name])
        for name, value in payload["stochastic_model_state"].items()
    )
    if not exact:
        raise AssertionError("persistent selected checkpoint did not reload bitwise")
    candidate_mean.eval()
    parent.eval()
    model.eval()
    return candidate_mean, parent, model, {
        "training_result": {
            "path": str(paths["training_result"]),
            "sha256": PGL_TRAINING_RESULT_SHA256,
        },
        "selected_checkpoint": {
            "path": str(paths["selected_checkpoint"]),
            "sha256": PGL_SELECTED_CHECKPOINT_SHA256,
            "completed_epoch": 20,
            "checkpoint_reload_bitwise": True,
        },
        "parent": parent_record,
        "residual_scales": {
            "path": str(paths["residual_scales"]),
            "sha256": sha256_path(paths["residual_scales"]),
        },
    }


def main() -> int:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
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
        raise ValueError("persistent artifact-root argument differs from manifest")
    environment = require_runtime()
    output = args.output
    output.mkdir(parents=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    selected_mean, parent_mean, model, provenance = load_selected_models(
        paths=paths,
        artifact_root=args.artifact_root,
        device=device,
    )
    catalog = load_official_catalog(args.artifact_root)
    selected_targets = tuple(value + 1 for value in PGL_EVALUATION_STARTS)
    contiguous_dataset = OneStepContextDataset(
        catalog,
        target_frames=tuple(range(selected_targets[0], selected_targets[-1] + 1)),
        context_frames=1,
    )
    dataset = SelectedContextDatasetAdapter(
        contiguous_dataset,
        target_frames=selected_targets,
    )
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
        job_type="old-85604-persistent-global-local-M32-generation",
        tags=(
            "paper0",
            "85604",
            "persistent-global-local",
            "M32",
            "four-frame",
            "truth-free-generation",
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key or str(getattr(api.viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B identity differs")
    tracking = output / "wandb"
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
            "scope": PGL_EVALUATION_SCOPE,
            "paper0_commit": args.paper0_commit,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "forecast_population": manifest["forecast_population"],
            "sampler": manifest["sampler"],
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "target_truth_read": False,
        },
        mode="online",
        dir=str(tracking),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")
    schema = PGLForecastSchema.frozen()
    forecast_path = output / "forecast_M32_four_frame.h5"
    try:
        with PGLForecastWriter(
            forecast_path,
            paper0_commit=args.paper0_commit,
            manifest_sha256=args.manifest_sha256,
            training_result_sha256=PGL_TRAINING_RESULT_SHA256,
            checkpoint_sha256=PGL_SELECTED_CHECKPOINT_SHA256,
            seed_bank_path=paths["scientific_seed_bank"],
            seed_bank_sha256=PGL_SCIENTIFIC_SEED_BANK_SHA256,
            schema=schema,
        ) as writer:
            def on_start(record: Mapping[str, Any]) -> None:
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
            "status": "truth_free_forecast_completed_and_hash_closed",
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "schema": schema.to_record(),
            "provenance": provenance,
            "environment": environment,
            "peak_cuda_memory_bytes": peak_bytes,
            "peak_cuda_memory_GiB": peak_bytes / 2**30,
            "physics_scoring_authorized_next": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
        }
        atomic_json(output / "result.json", result)
        run.summary.update(
            {
                "final/status": result["status"],
                "final/forecast_sha256": result["forecast"]["sha256"],
                "final/start_count": result["start_count"],
                "compute/peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
                "scope/target_truth_read": False,
                "scope/held_out_85606_read": False,
                "scope/new_nersc_data_read": False,
            }
        )
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise
    finally:
        contiguous_dataset.close()
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
    atomic_json(output / "wandb.json", wandb_record)
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and "wandb" not in path.parts and path.name != "artifact_sha256.txt":
            lines.append(f"{sha256_path(path)}  {path}\n")
    (output / "artifact_sha256.txt").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"result": result, "wandb": wandb_record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
