#!/usr/bin/env python3
"""Freeze six completed C5P O2 runs before reference or physics evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


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
from tcv_diagnostics.models.o2 import O2ViTConfig  # noqa: E402
from tcv_diagnostics.o2_training import O2RunConfig  # noqa: E402


RUNS = (
    (0, "C5P-H1", "c5p_h1", 1701, 1, 0),
    (1, "C5P-H1", "c5p_h1", 1702, 1, 2),
    (2, "C5P-H1", "c5p_h1", 1703, 1, 0),
    (3, "C5P-H2", "c5p_h2", 1701, 2, 1),
    (4, "C5P-H2", "c5p_h2", 1702, 2, 3),
    (5, "C5P-H2", "c5p_h2", 1703, 2, 1),
)
ARTIFACTS = (
    "config.json",
    "latent_normalization.json",
    "history.jsonl",
    "result.json",
    "selected.pt",
    "final_training_state.pt",
    "wandb.json",
)
FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--training-slurm-job-id", required=True)
    parser.add_argument("--full-run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--audit-slurm-job-id", required=True)
    return parser.parse_args()


def _artifact_index(path: Path) -> dict[Path, str]:
    records: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        artifact = Path(name.strip()).resolve(strict=True)
        if artifact in records:
            raise ValueError(f"duplicate artifact-index path {artifact}")
        records[artifact] = digest
    return records


def _history(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if len(records) != 200 or [int(item["epoch"]) for item in records] != list(
        range(200)
    ):
        raise ValueError("O2 history does not contain epochs 0..199 exactly once")
    for epoch, item in enumerate(records):
        if int(item.get("examples", -1)) != 430:
            raise ValueError("O2 history epoch does not contain 430 examples")
        if int(item.get("global_step", -1)) != (epoch + 1) * 27:
            raise ValueError("O2 history optimizer-step sequence differs")
        channels = item.get("validation_mae_by_channel", {})
        if set(channels) != set(FIELDS):
            raise ValueError("O2 history validation channel inventory differs")
        numeric = (
            item.get("train_equal_channel_mae"),
            item.get("validation_equal_channel_mae"),
            item.get("mean_preclip_gradient_norm"),
            item.get("maximum_preclip_gradient_norm"),
            item.get("epoch_wall_seconds"),
            *channels.values(),
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("O2 history contains a non-finite value")
    return records


def _expected_config(
    *,
    arm: str,
    seed: int,
    codec_checkpoint_path: str,
    codec_checkpoint_sha256: str,
) -> dict[str, Any]:
    record = {
        **O2RunConfig.frozen(mode="full", arm=arm, seed=seed).to_record(),
        "model": O2ViTConfig().to_record(),
        "codec_checkpoint": {
            "path": codec_checkpoint_path,
            "sha256": codec_checkpoint_sha256,
            "trainable": False,
        },
    }
    return json.loads(json.dumps(record))


def freeze_run(
    run_dir: Path,
    *,
    run_index: int,
    arm: str,
    slug: str,
    seed: int,
    context_frames: int,
    gpu_index: int,
    codec_checkpoint_path: str,
    codec_checkpoint_sha256: str,
    training_commit: str,
    training_slurm_job_id: str,
) -> dict[str, Any]:
    expected_name = f"task_{run_index}_{slug}_seed_{seed}"
    directory = Path(run_dir).resolve(strict=True)
    assert_development_path(directory)
    if directory.name != expected_name:
        raise ValueError(f"O2 run directory differs from {expected_name}")
    paths = {
        name: directory / name for name in (*ARTIFACTS, "artifact_sha256.txt")
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"O2 run {run_index} artifacts are incomplete")

    expected_config = _expected_config(
        arm=arm,
        seed=seed,
        codec_checkpoint_path=codec_checkpoint_path,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
    )
    config = load_strict_json(paths["config.json"])
    result = load_strict_json(paths["result.json"])
    tracking = load_strict_json(paths["wandb.json"])
    latent = load_strict_json(paths["latent_normalization.json"])
    history = _history(paths["history.jsonl"])
    if config != expected_config or result.get("config") != expected_config:
        raise ValueError(f"O2 run {run_index} configuration differs")
    expected_slurm_identity = (
        f"{training_slurm_job_id}:run{run_index}:gpu{gpu_index}"
    )
    completion_contract = (
        result.get("scope") == "O2_teacher_forced_one_step_full"
        and result.get("paper0_commit") == training_commit
        and result.get("slurm_job_id") == expected_slurm_identity
        and result.get("development_run") == "85604"
        and result.get("held_out_85606_read") is False
        and result.get("completed_epochs") == 200
        and result.get("completed_optimizer_steps") == 5400
        and result.get("physics_derived_loss_used") is False
        and result.get("target_truth_used_as_model_input") is False
        and result.get("absolute_time_used_as_model_input") is False
        and result.get("checkpoint_reload_bitwise_exact") is True
        and result.get("O2_scientific_gate_evaluated") is False
        and result.get("O3_launch_allowed") is False
    )
    if not completion_contract:
        raise ValueError(f"O2 run {run_index} completion contract differs")
    if int(config["context_frames"]) != context_frames:
        raise ValueError(f"O2 run {run_index} history length differs")

    selected_epoch = min(
        range(200),
        key=lambda epoch: float(history[epoch]["validation_equal_channel_mae"]),
    )
    selected_loss = float(
        history[selected_epoch]["validation_equal_channel_mae"]
    )
    if (
        int(result.get("selected_epoch", -1)) != selected_epoch
        or float(result.get("selected_validation_equal_channel_mae"))
        != selected_loss
    ):
        raise ValueError(f"O2 run {run_index} checkpoint selection differs")
    final = history[-1]
    if (
        float(result.get("final_validation_equal_channel_mae"))
        != float(final["validation_equal_channel_mae"])
        or result.get("final_validation_mae_by_channel")
        != final["validation_mae_by_channel"]
    ):
        raise ValueError(f"O2 run {run_index} final validation record differs")

    hashes = {
        name: sha256_path(path)
        for name, path in paths.items()
        if name != "artifact_sha256.txt"
    }
    if result.get("selected_checkpoint", {}).get("sha256") != hashes[
        "selected.pt"
    ]:
        raise ValueError(f"O2 run {run_index} selected checkpoint hash differs")
    if result.get("final_training_state", {}).get("sha256") != hashes[
        "final_training_state.pt"
    ]:
        raise ValueError(f"O2 run {run_index} final state hash differs")
    if result.get("history", {}).get("sha256") != hashes["history.jsonl"]:
        raise ValueError(f"O2 run {run_index} history hash differs")
    if result.get("latent_normalization", {}).get("sha256") != hashes[
        "latent_normalization.json"
    ]:
        raise ValueError(f"O2 run {run_index} latent-normalization hash differs")
    indexed = _artifact_index(paths["artifact_sha256.txt"])
    expected_inventory = {
        paths[name].resolve(strict=True) for name in ARTIFACTS
    }
    if set(indexed) != expected_inventory:
        raise ValueError(f"O2 run {run_index} artifact index inventory differs")
    for name in ARTIFACTS:
        if indexed[paths[name].resolve(strict=True)] != hashes[name]:
            raise ValueError(f"O2 run {run_index} indexed {name} hash differs")

    if (
        latent.get("kind")
        != "per_latent_channel_training_only_population_moments"
        or latent.get("fit_frames") != [0, 432]
        or latent.get("sample_count_per_channel") != 1_216_512
        or latent.get("codec_checkpoint_sha256")
        != codec_checkpoint_sha256
        or latent.get("scientific_authority") is not True
        or latent.get("held_out_85606_read") is not False
    ):
        raise ValueError(f"O2 run {run_index} latent normalization differs")

    expected_run_id = f"p0o2full-{training_slurm_job_id}-{run_index}"
    if (
        tracking.get("required") is not True
        or tracking.get("mode") != "online"
        or tracking.get("epochs_logged") != 200
        or tracking.get("remote_presence_verified_after_finish") is not True
        or tracking.get("remote_state_after_finish") != "finished"
        or tracking.get("local_artifacts_are_scientific_authority") is not True
        or tracking.get("spec", {}).get("run_id") != expected_run_id
        or tracking.get("spec", {}).get("group") != "o2-c5p-l10-full"
    ):
        raise ValueError(f"O2 run {run_index} W&B completion differs")

    return {
        "run_index": run_index,
        "arm": arm,
        "context_frames": context_frames,
        "seed": seed,
        "training_gpu_index": gpu_index,
        "run_directory": str(directory),
        "training_result": {
            "path": str(paths["result.json"].resolve(strict=True)),
            "sha256": hashes["result.json"],
        },
        "selected_checkpoint": {
            "path": str(paths["selected.pt"].resolve(strict=True)),
            "sha256": hashes["selected.pt"],
        },
        "final_training_state": {
            "path": str(paths["final_training_state.pt"].resolve(strict=True)),
            "sha256": hashes["final_training_state.pt"],
        },
        "codec_checkpoint": {
            "path": codec_checkpoint_path,
            "sha256": codec_checkpoint_sha256,
            "trainable_during_O2": False,
        },
        "config_sha256": hashes["config.json"],
        "latent_normalization_sha256": hashes["latent_normalization.json"],
        "history_sha256": hashes["history.jsonl"],
        "wandb_record_sha256": hashes["wandb.json"],
        "artifact_index_sha256": sha256_path(paths["artifact_sha256.txt"]),
        "selected_epoch": selected_epoch,
        "selected_global_step": int(history[selected_epoch]["global_step"]),
        "selected_validation_equal_channel_mae": selected_loss,
        "final_validation_mae_by_channel": dict(
            result["final_validation_mae_by_channel"]
        ),
        "parameter_count": int(result["parameter_count"]),
        "peak_cuda_bytes": int(result["peak_cuda_bytes"]),
        "wall_seconds": float(result["wall_seconds"]),
        "wandb": {
            "run_id": expected_run_id,
            "run_url": tracking["run_url"],
            "remote_state": "finished",
        },
    }


def _arm_training_summary(runs: list[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    values = [
        float(run["selected_validation_equal_channel_mae"])
        for run in runs
        if run["arm"] == arm
    ]
    if len(values) != 3:
        raise ValueError(f"O2 arm {arm} does not contain three seeds")
    return {
        "seed_count": len(values),
        "mean_selected_validation_equal_channel_mae": statistics.mean(values),
        "population_standard_deviation": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    args = parse_args()
    for path in (args.job_root, args.full_run_manifest, args.output):
        assert_development_path(path)
    root = args.job_root.resolve(strict=True)
    if root.name != f"job_{args.training_slurm_job_id}":
        raise ValueError("O2 training job root and Slurm ID differ")

    run_manifest_path = args.full_run_manifest.resolve(strict=True)
    run_manifest = load_strict_json(run_manifest_path)
    if (
        run_manifest.get("status")
        != "frozen_execution_revision_after_zero_compute_H100_hold_before_full_O2_training"
        or run_manifest.get("development_run") != "85604"
        or run_manifest.get("held_out_85606_access_allowed") is not False
        or run_manifest.get("model", {}).get("physics_derived_loss_allowed")
        is not False
        or len(run_manifest.get("tasks", [])) != 6
    ):
        raise ValueError("full O2 run manifest contract differs")
    manifest_tasks = {
        int(task["run_index"]): task for task in run_manifest["tasks"]
    }
    if set(manifest_tasks) != set(range(6)):
        raise ValueError("full O2 run manifest indices differ")

    summary_path = root / "training_summary.json"
    summary = load_strict_json(summary_path)
    if (
        summary.get("scope") != "phase2_C5P_O2_full_training"
        or summary.get("paper0_commit") != args.training_commit
        or str(summary.get("slurm_job_id")) != args.training_slurm_job_id
        or summary.get("development_run") != "85604"
        or summary.get("held_out_85606_read") is not False
        or summary.get("completed_logical_runs") != 6
        or summary.get("training_complete") is not True
        or summary.get("training_summary_is_scientific_acceptance") is not False
        or summary.get("O2_scientific_gate_evaluated") is not False
        or summary.get("O3_launch_allowed") is not False
    ):
        raise ValueError("full O2 training summary contract differs")

    runs: list[dict[str, Any]] = []
    for index, arm, slug, seed, context, gpu_index in RUNS:
        task = manifest_tasks[index]
        if (
            task["arm"] != arm
            or int(task["seed"]) != seed
            or int(task["context_frames"]) != context
        ):
            raise ValueError(f"full O2 manifest task {index} differs")
        runs.append(
            freeze_run(
                root / f"task_{index}_{slug}_seed_{seed}",
                run_index=index,
                arm=arm,
                slug=slug,
                seed=seed,
                context_frames=context,
                gpu_index=gpu_index,
                codec_checkpoint_path=str(task["codec_checkpoint"]),
                codec_checkpoint_sha256=str(task["codec_sha256"]),
                training_commit=args.training_commit,
                training_slurm_job_id=args.training_slurm_job_id,
            )
        )

    summary_runs = summary["runs"]
    if len(summary_runs) != 6:
        raise ValueError("full O2 training summary run count differs")
    for frozen, summarized in zip(runs, summary_runs):
        if (
            int(summarized["run_index"]) != frozen["run_index"]
            or summarized["arm"] != frozen["arm"]
            or int(summarized["seed"]) != frozen["seed"]
            or summarized["selected_checkpoint_sha256"]
            != frozen["selected_checkpoint"]["sha256"]
            or float(summarized["selected_validation_equal_channel_mae"])
            != frozen["selected_validation_equal_channel_mae"]
        ):
            raise ValueError("full O2 training summary and run artifacts differ")

    result = {
        "schema_version": 1,
        "scope": "phase2_C5P_O2_full_training_matrix_frozen",
        "status": "completed_pending_scientific_O2_evaluation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_commit": args.training_commit,
        "training_slurm_job_id": args.training_slurm_job_id,
        "audit_commit": args.audit_commit,
        "audit_slurm_job_id": args.audit_slurm_job_id,
        "full_run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256_path(run_manifest_path),
        },
        "training_summary": {
            "path": str(summary_path.resolve(strict=True)),
            "sha256": sha256_path(summary_path),
        },
        "training_artifact_index_sha256": sha256_path(
            root / "artifact_sha256.txt"
        ),
        "runs": runs,
        "run_count": len(runs),
        "training_loss_comparison_only": {
            "C5P-H1": _arm_training_summary(runs, "C5P-H1"),
            "C5P-H2": _arm_training_summary(runs, "C5P-H2"),
            "may_select_an_arm": False,
        },
        "checkpoint_choice_frozen_before_reference_or_physics_metrics": True,
        "O2_scientific_evaluation_completed": False,
        "O2_accepted_arms": [],
        "O3_launch_allowed": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = args.output.resolve(strict=False)
    write_strict_json_atomic(output, result)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
