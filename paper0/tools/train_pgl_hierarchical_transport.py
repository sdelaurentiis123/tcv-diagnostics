#!/usr/bin/env python3
"""Train one matched PGL control or hierarchical transport-aware arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from paper0.tools.generate_persistent_global_local_forecast import load_selected_models
from paper0.tools.train_codec_free_stage1_pilot import verify_finished_wandb_run
from paper0.tools.train_persistent_global_local_pilot import load_parent
from tcv_diagnostics.autoregressive_training import AutoregressiveStateWindowDataset
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.matched_o1_transport import load_transport_geometry
from tcv_diagnostics.model_data import (
    assert_development_path,
    load_strict_json,
    write_strict_json_atomic,
)
from tcv_diagnostics.model_training_data import load_official_catalog
from tcv_diagnostics.persistent_global_local_authority import (
    authorize_pgl_evaluation_manifest,
)
from tcv_diagnostics.pgl_hierarchical_training import (
    PGL_HIERARCHICAL_ARMS,
    HierarchicalControlMagnitudes,
    HierarchicalTrainingConfig,
    train_hierarchical_arm,
)
from tcv_diagnostics.pgl_torch_transport import (
    TorchSeparatrixTransport,
    decoder_records_from_normalization,
)
from tcv_diagnostics.pgl_variogram_training import (
    load_pair_banks,
    load_training_transport_truth,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


PARENT_CHECKPOINT_SHA256 = (
    "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "screen"), required=True)
    parser.add_argument("--arm", choices=PGL_HIERARCHICAL_ARMS, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--smoke-root", type=Path)
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
    if commit != str(expected) or dirty:
        raise RuntimeError("hierarchical training requires the locked clean checkout")


def _locked_artifact(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    assert_development_path(path)
    digest = str(record.get("sha256", ""))
    if len(digest) != 64 or not path.is_file() or sha256_path(path) != digest:
        raise ValueError(f"{label} path or SHA-256 differs")
    return path.resolve(strict=True)


def authorize_preflight(
    path: Path, digest: str, *, commit: str
) -> tuple[dict[str, Any], dict[str, Path]]:
    if sha256_path(path) != str(digest):
        raise ValueError("hierarchical preflight SHA-256 differs")
    result = load_strict_json(path)
    if (
        result.get("scope")
        != "post_ecrd_old_85604_pgl_hierarchical_transport_preflight"
        or result.get("status") != "passed"
        or result.get("development_run") != "85604"
        or result.get("paper0_commit") != str(commit)
        or result.get("known_answer_gates", {}).get("passed") is not True
        or result.get("prior_preflight", {}).get("sampler_regression_passed") is not True
        or result.get("prior_preflight", {}).get("transport_equivalence_passed") is not True
        or result.get("screen_training_authorized") is not False
        or result.get("parent", {}).get("selected_checkpoint", {}).get("sha256")
        != PARENT_CHECKPOINT_SHA256
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
    ):
        raise ValueError("hierarchical preflight authorization differs")
    return result, {
        "pair_banks": _locked_artifact(result["pair_banks"], label="pair banks"),
        "native_transport_truth": _locked_artifact(
            result["native_transport_truth"], label="native transport truth"
        ),
        "controls": _locked_artifact(
            result["control_magnitudes"], label="control magnitudes"
        ),
        "gradient": _locked_artifact(
            result["gradient_calibration"], label="gradient calibration"
        ),
    }


def verify_smokes(root: Path, *, commit: str) -> dict[str, Any]:
    assert_development_path(root)
    records: dict[str, Any] = {}
    for index, arm in enumerate(PGL_HIERARCHICAL_ARMS):
        path = root / f"task_{index}_{arm}" / "run" / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = load_strict_json(path)
        physics_loss = arm == "TRANSPORT"
        if (
            result.get("scope")
            != "post_ecrd_old_85604_pgl_hierarchical_transport_training"
            or result.get("status") != "smoke_passed"
            or result.get("mode") != "smoke"
            or result.get("arm") != arm
            or result.get("paper0_commit") != str(commit)
            or result.get("completed_optimizer_updates") != 1
            or len(result.get("checkpoints", [])) != 1
            or result.get("full_sampler_compute_control_executed") is not True
            or result.get("checkpoint_selection_performed") is not False
            or result.get("physics_derived_training_loss_used") is not physics_loss
            or result.get("held_out_85606_read") is not False
            or result.get("new_nersc_data_read") is not False
        ):
            raise ValueError(f"hierarchical smoke gate differs for {arm}")
        records[arm] = {
            "path": str(path),
            "sha256": sha256_path(path),
            "peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
            "wall_seconds": result["wall_seconds"],
        }
    return records


def main() -> int:
    args = parse_args()
    for path in (
        args.artifact_root,
        args.evidence_manifest,
        args.preflight,
        args.output,
        args.paper0_root,
    ):
        assert_development_path(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_checkout(args.paper0_root, args.paper0_commit)
    preflight, artifacts = authorize_preflight(
        args.preflight, args.preflight_sha256, commit=args.paper0_commit
    )
    smoke_gate = None
    if args.mode == "screen":
        if args.smoke_root is None:
            raise ValueError("screen mode requires the two-arm smoke root")
        smoke_gate = verify_smokes(args.smoke_root, commit=args.paper0_commit)
    elif args.smoke_root is not None:
        raise ValueError("smoke mode may not consume a smoke root")

    manifest = load_strict_json(args.evidence_manifest)
    paths = authorize_pgl_evaluation_manifest(
        manifest,
        manifest_path=args.evidence_manifest,
        manifest_sha256=args.evidence_manifest_sha256,
        paper0_root=args.paper0_root,
    )
    if paths["model_dataset"] != args.artifact_root.resolve(strict=True):
        raise ValueError("hierarchical training model-dataset root differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("hierarchical training requires one allocated GPU")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected_mean, _, model, provenance = load_selected_models(
        paths=paths, artifact_root=args.artifact_root, device=device
    )
    if provenance["selected_checkpoint"]["sha256"] != PARENT_CHECKPOINT_SHA256:
        raise ValueError("hierarchical parent checkpoint differs")
    training_manifest = load_strict_json(paths["training_manifest"])
    _, _, derivative_rms = load_parent(training_manifest, device=device)
    catalog = load_official_catalog(args.artifact_root)
    geometry = load_transport_geometry(
        geometry_path=paths["geometry"],
        geometry_manifest=load_strict_json(paths["geometry_manifest"]),
    )
    transport = TorchSeparatrixTransport(
        geometry, decoder_records_from_normalization(catalog.normalization)
    )
    banks = load_pair_banks(
        artifacts["pair_banks"], expected_sha256=preflight["pair_banks"]["sha256"]
    )
    native_truth = load_training_transport_truth(
        artifacts["native_transport_truth"],
        expected_sha256=preflight["native_transport_truth"]["sha256"],
    )
    controls_record = load_strict_json(artifacts["controls"])
    controls = HierarchicalControlMagnitudes.from_record(controls_record)
    gradient = load_strict_json(artifacts["gradient"])
    if (
        gradient.get("status") != "passed"
        or gradient.get("target_ratio") != 0.25
        or gradient.get("held_out_85606_read") is not False
        or gradient.get("new_nersc_data_read") is not False
    ):
        raise ValueError("hierarchical gradient calibration artifact differs")
    config = HierarchicalTrainingConfig(mode=args.mode, arm=args.arm)

    args.output.mkdir(parents=True)
    tracking_directory = args.output / "wandb"
    tracking_directory.mkdir()
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("online W&B is required for hierarchical training") from error
    spec = WandbRunSpec(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        run_id=args.wandb_run_id,
        run_name=args.wandb_run_name,
        job_type=f"old-85604-pgl-hierarchical-{args.mode}-{args.arm.lower()}",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "pgl",
            "hierarchical-transport",
            args.mode,
            f"arm-{args.arm.lower()}",
            "physics-loss" if config.physics_derived_training_loss_used else "control",
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key or str(getattr(api.viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B identity differs")
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
            "scope": "post_ecrd_old_85604_pgl_hierarchical_transport_training",
            "paper0_commit": args.paper0_commit,
            "preflight_sha256": args.preflight_sha256,
            "mode": args.mode,
            "arm": args.arm,
            "training": config.to_record(),
            "auxiliary_lambda": gradient["auxiliary_lambda"],
            "gradient_target_ratio": gradient["target_ratio"],
            "physics_derived_training_loss_used": (
                config.physics_derived_training_loss_used
            ),
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    dataset = AutoregressiveStateWindowDataset(
        catalog,
        family="c5p",
        split="train",
        horizon=4,
        augment=True,
        seed=1702,
    )
    try:
        def on_update(record: Mapping[str, Any]) -> None:
            metrics = {
                "optimizer/update": int(record["optimizer_update"]),
                "optimizer/equivalent_epochs": float(record["equivalent_epochs"]),
                "optimizer/stochastic_learning_rate": float(
                    record["stochastic_learning_rate"]
                ),
                "optimizer/mean_learning_rate": float(record["mean_learning_rate"]),
                "train/objective": float(record["objective"]),
                "train/original": float(record["original"]),
                "train/mean": float(record["mean"]),
                "train/edm": float(record["edm"]),
                "train/local_normalized": float(record["local_normalized"]),
                "train/regional_normalized": float(record["regional_normalized"]),
                "train/global_normalized": float(record["global_normalized"]),
                "train/preclip_gradient_norm": float(record["preclip_gradient_norm"]),
                "timing/update_wall_seconds": float(record["wall_seconds"]),
            }
            for name, value in record.items():
                if any(
                    name.startswith(prefix)
                    for prefix in (
                        "local_spatial/",
                        "local_temporal/",
                        "regional/",
                        "fourier_low/",
                        "fourier_transport_band/",
                        "global_crps/",
                    )
                ):
                    metrics[f"train/{name}"] = float(value)
            audit = record.get("gradient_audit")
            if audit is not None:
                for loss_name, loss_record in audit["losses"].items():
                    metrics[f"gradient/{loss_name}/total"] = float(
                        loss_record["total_gradient_norm"]
                    )
                    for branch, branch_record in loss_record["branches"].items():
                        metrics[f"gradient/{loss_name}/{branch}"] = float(
                            branch_record["gradient_norm"]
                        )
            run.log(metrics, step=int(record["optimizer_update"]))

        training = train_hierarchical_arm(
            mean_model=selected_mean,
            edm=model,
            transport=transport,
            training_dataset=dataset,
            derivative_rms=derivative_rms,
            transport_truth_by_frame=native_truth,
            spatial_bank=banks["transport_spatial"],
            temporal_bank=banks["transport_temporal"],
            controls=controls,
            auxiliary_lambda=float(gradient["auxiliary_lambda"]),
            gradient_calibration=gradient,
            output=args.output / "training",
            device=device,
            paper0_commit=args.paper0_commit,
            slurm_job_id=args.slurm_job_id,
            parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
            config=config,
            on_update=on_update,
        )
        result = {
            **training,
            "preflight": {
                "path": str(args.preflight),
                "sha256": args.preflight_sha256,
            },
            "evidence_manifest": {
                "path": str(args.evidence_manifest),
                "sha256": args.evidence_manifest_sha256,
            },
            "smoke_gate": smoke_gate,
            "parent": provenance,
        }
        write_strict_json_atomic(args.output / "result.json", result)
        run.summary.update(
            {
                "final/status": result["status"],
                "final/arm": args.arm,
                "final/mode": args.mode,
                "final/optimizer_updates": result["completed_optimizer_updates"],
                "final/checkpoint_count": len(result["checkpoints"]),
                "compute/peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
                "compute/wall_seconds": result["wall_seconds"],
                "scope/physics_derived_training_loss_used": (
                    config.physics_derived_training_loss_used
                ),
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
        dataset.close()

    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb, remote_path=remote_path, expected_id=spec.run_id
    )
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": spec.to_record(),
        "wandb_version": wandb.__version__,
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_state_after_finish": remote_state,
        "checkpoints_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    write_strict_json_atomic(args.output / "wandb.json", tracking)
    artifact_lines = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and "wandb" not in path.parts and path.name != "artifact_sha256.txt":
            artifact_lines.append(f"{sha256_path(path)}  {path}\n")
    (args.output / "artifact_sha256.txt").write_text(
        "".join(artifact_lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": args.mode,
                "arm": args.arm,
                "optimizer_updates": result["completed_optimizer_updates"],
                "checkpoints": result["checkpoints"],
                "peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
                "wandb": tracking,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
