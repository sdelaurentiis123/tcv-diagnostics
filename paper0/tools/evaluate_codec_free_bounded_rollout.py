#!/usr/bin/env python3
"""Generate and score the frozen old-85604 bounded state rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from paper0.tools.train_codec_free_stage1_pilot import (
    atomic_json,
    verify_finished_wandb_run,
)
from paper0.tools.freeze_codec_free_bounded_rollout import SCOPE
from tcv_diagnostics.bounded_rollout import (
    FIELDS,
    FieldErrorAccumulator,
    autoregressive_forecast_path,
    direct_forecast,
    method_schedule,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import (
    CodecFrameDataset,
    VOLUME_SHAPE,
    load_official_catalog,
)
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


HORIZONS = (4, 8)
SEEDS = (1701, 1702, 1703)
VALIDATION_START = 496
VALIDATION_STOP = 624
CUDA_DEVICE_INDEX = 0


def reset_peak_cuda_memory_stats() -> None:
    """Reset accounting on the sole Slurm-visible logical CUDA device."""

    torch.cuda.reset_peak_memory_stats(CUDA_DEVICE_INDEX)


def peak_cuda_memory_gib() -> float:
    """Return peak allocation on the sole Slurm-visible logical CUDA device."""

    return torch.cuda.max_memory_allocated(CUDA_DEVICE_INDEX) / 2**30


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


def repository_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_locked_file(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    assert_development_path(path)
    digest = str(record.get("sha256", ""))
    if len(digest) != 64 or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return path


def authorize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    paper0_commit: str,
) -> None:
    if sha256_path(manifest_path) != manifest_sha256:
        raise ValueError("bounded-rollout manifest SHA-256 differs")
    if (
        manifest.get("scope") != SCOPE
        or manifest.get("status")
        != "frozen_after_three_seed_confirmation_before_rollout_inference"
        or manifest.get("development_run") != "85604"
        or manifest.get("held_out_85606_read") is not False
        or manifest.get("held_out_85606_access_allowed") is not False
        or manifest.get("new_nersc_data_access_allowed") is not False
        or manifest.get("guard_frames_read_allowed") is not False
        or manifest.get("training_allowed") is not False
        or manifest.get("checkpoint_selection_allowed") is not False
        or manifest.get("paper0_commit_at_freeze") != paper0_commit
        or manifest.get("fields") != list(FIELDS)
        or manifest.get("zperiod") != 5
        or manifest.get("mode_mapping") != "n=5k"
        or manifest.get("wandb_required") is not True
    ):
        raise ValueError("bounded-rollout manifest scope differs")
    evidence = manifest.get("evidence", {})
    reduction_path = verify_locked_file(
        evidence.get("three_seed_reduction", {}),
        label="three-seed reduction",
    )
    reduction = load_strict_json(reduction_path)
    if (
        reduction.get("three_seed_mechanism_confirmed") is not True
        or reduction.get("bounded_rollout_authorized") is not True
    ):
        raise ValueError("three-seed reduction no longer authorizes rollout")
    models = evidence.get("models", {})
    if set(models) != {"1701", "1702", "1703"}:
        raise ValueError("bounded-rollout model set differs")
    for seed in SEEDS:
        record = models[str(seed)]
        if int(record.get("seed", -1)) != seed:
            raise ValueError("bounded-rollout model seed differs")
        result_path = verify_locked_file(
            record.get("result", {}), label=f"seed-{seed} result"
        )
        checkpoint_path = verify_locked_file(
            record.get("selected_checkpoint", {}),
            label=f"seed-{seed} checkpoint",
        )
        result = load_strict_json(result_path)
        selected = result.get("best_checkpoint", {})
        if (
            Path(str(selected.get("path", ""))) != checkpoint_path
            or selected.get("sha256") != record["selected_checkpoint"]["sha256"]
            or int(selected.get("epoch", -1)) != 4
        ):
            raise ValueError(f"seed-{seed} selected checkpoint differs")
    evaluation = manifest.get("evaluation", {})
    if (
        evaluation.get("validation_frames") != [496, 624]
        or evaluation.get("history_frames") != 1
        or evaluation.get("same_starts_and_targets_within_terminal_horizon") is not True
        or evaluation.get("intermediate_or_future_truth_used_as_model_input")
        is not False
        or evaluation.get("complete_predicted_five_field_state_fed_back") is not True
        or evaluation.get("inference_batch_size") != 4
    ):
        raise ValueError("bounded-rollout evaluation contract differs")
    expected = {
        "4": {
            "current_frames": [496, 620],
            "target_frames": [500, 624],
            "pair_count": 124,
            "methods": method_schedule(4),
        },
        "8": {
            "current_frames": [496, 616],
            "target_frames": [504, 624],
            "pair_count": 120,
            "methods": method_schedule(8),
        },
    }
    if evaluation.get("horizons") != expected:
        raise ValueError("bounded-rollout horizon definition differs")


def _config_from_record(record: Mapping[str, Any]) -> CodecFreeOperatorConfig:
    return CodecFreeOperatorConfig(
        state_family="c5p",
        history_frames=1,
        base_channels=int(record["base_channels"]),
        channel_multipliers=tuple(int(item) for item in record["channel_multipliers"]),
        blocks_per_level=int(record["blocks_per_level"]),
        lead_embedding_channels=int(record["lead_embedding_channels"]),
        group_norm_maximum_groups=int(record["group_norm_maximum_groups"]),
        kernel_size=int(record["kernel_size"]),
        predict_boundary=False,
        zero_initialize_output=bool(record["zero_initialize_output"]),
        auxiliary_context_channels=int(record.get("auxiliary_context_channels", 0)),
    )


def load_selected_model(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[CodecFreeIncrementOperator3D, dict[str, Any]]:
    lock = manifest["evidence"]["models"][str(seed)]
    result = load_strict_json(Path(lock["result"]["path"]))
    checkpoint_path = Path(lock["selected_checkpoint"]["path"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = result.get("architecture", {}).get("architecture", {})
    config = _config_from_record(architecture)
    if (
        payload.get("family") != "c5p"
        or payload.get("stage") != "stage2_multilead_finetune"
        or int(payload.get("seed", -1)) != seed
        or int(payload.get("epoch", -1)) != 4
        or float(payload.get("selection_metric", float("nan")))
        != float(lock["selected_checkpoint"]["selection_metric"])
        or payload.get("config") != config.to_record()
    ):
        raise ValueError(f"seed-{seed} checkpoint payload differs")
    model = CodecFreeIncrementOperator3D(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(payload["model"][name].to(device), model.state_dict()[name])
        for name in payload["model"]
    )
    if not bitwise:
        raise AssertionError(f"seed-{seed} checkpoint did not reload bitwise")
    model.eval()
    return model, {
        "seed": seed,
        "checkpoint": dict(lock["selected_checkpoint"]),
        "checkpoint_reload_bitwise": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_validation_states(catalog: Any) -> np.ndarray:
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="validation",
        frames=range(VALIDATION_START, VALIDATION_STOP),
        augment=False,
        seed=0,
        return_physical=False,
    )
    try:
        states = np.empty((len(dataset), len(FIELDS), *VOLUME_SHAPE), dtype=np.float32)
        for index in range(len(dataset)):
            item = dataset[index]
            if int(item["frame_index"]) != VALIDATION_START + index:
                raise ValueError("validation frame order differs")
            states[index] = item["volume"]
    finally:
        dataset.close()
    if not np.all(np.isfinite(states)):
        raise ValueError("validation state contains non-finite values")
    return states


def _frame_slice(states: np.ndarray, frames: np.ndarray) -> np.ndarray:
    indices = np.asarray(frames, dtype=np.int64) - VALIDATION_START
    if np.any(indices < 0) or np.any(indices >= states.shape[0]):
        raise ValueError("requested frame leaves frozen validation")
    return np.ascontiguousarray(states[indices], dtype=np.float32)


def _create_delta_dataset(
    group: h5py.Group,
    name: str,
    count: int,
) -> h5py.Dataset:
    return group.create_dataset(
        name,
        shape=(count, len(FIELDS), *VOLUME_SHAPE),
        dtype="f4",
        chunks=(1, 1, *VOLUME_SHAPE),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
        track_times=False,
    )


def _persistence_records(
    states: np.ndarray,
    *,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    current_frames = np.arange(496, 624 - horizon, dtype=np.int64)
    target_frames = current_frames + horizon
    current = _frame_slice(states, current_frames)
    truth = _frame_slice(states, target_frames)
    accumulator = FieldErrorAccumulator.empty()
    frame_rmse = accumulator.update(current, truth)
    result = accumulator.finalize()
    mse = {field: float(result["per_field"][field]["mse"]) for field in FIELDS}
    rows = []
    for offset, (start, target) in enumerate(zip(current_frames, target_frames)):
        for field_index, field in enumerate(FIELDS):
            rows.append(
                {
                    "seed": "baseline",
                    "horizon": horizon,
                    "method": "persistence",
                    "composition_step": 0,
                    "composition_depth": 0,
                    "elapsed_frames": horizon,
                    "current_frame": int(start),
                    "target_frame": int(target),
                    "field": field,
                    "standardized_rmse": float(frame_rmse[offset, field_index]),
                }
            )
    return result, mse, rows


def _score_path_state(
    accumulator: FieldErrorAccumulator,
    candidate: np.ndarray,
    truth: np.ndarray,
    *,
    rows: list[dict[str, Any]],
    seed: int,
    horizon: int,
    method: str,
    step: int,
    depth: int,
    current_frames: np.ndarray,
) -> None:
    frame_rmse = accumulator.update(candidate, truth)
    elapsed = step * depth
    for offset, current in enumerate(current_frames):
        for field_index, field in enumerate(FIELDS):
            rows.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "method": method,
                    "composition_step": step,
                    "composition_depth": depth,
                    "elapsed_frames": elapsed,
                    "current_frame": int(current),
                    "target_frame": int(current + elapsed),
                    "field": field,
                    "standardized_rmse": float(frame_rmse[offset, field_index]),
                }
            )


def evaluate_one_seed_horizon(
    *,
    model: CodecFreeIncrementOperator3D,
    seed: int,
    horizon: int,
    states: np.ndarray,
    device: torch.device,
    h5_group: h5py.Group,
    persistence_mse: Mapping[str, float],
    batch_size: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    current_frames = np.arange(496, 624 - horizon, dtype=np.int64)
    target_frames = current_frames + horizon
    count = current_frames.size
    methods = method_schedule(horizon)
    datasets = {
        method: _create_delta_dataset(h5_group, method, count) for method in methods
    }
    terminal = {method: FieldErrorAccumulator.empty() for method in methods}
    depth_accumulators = {
        method: {
            depth: FieldErrorAccumulator.empty()
            for depth in range(1, horizon // int(step) + 1)
        }
        for method, step in methods.items()
        if step is not None
    }
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            batch_current_frames = current_frames[start:stop]
            current_np = _frame_slice(states, batch_current_frames)
            target_np = _frame_slice(states, target_frames[start:stop])
            current = torch.from_numpy(current_np).to(device)

            direct = direct_forecast(model, current, horizon=horizon)
            direct_np = direct.float().cpu().numpy()
            if not np.all(np.isfinite(direct_np)):
                raise FloatingPointError("direct bounded forecast is non-finite")
            datasets["direct"][start:stop] = direct_np - current_np
            _score_path_state(
                terminal["direct"],
                direct_np,
                target_np,
                rows=rows,
                seed=seed,
                horizon=horizon,
                method="direct",
                step=horizon,
                depth=1,
                current_frames=batch_current_frames,
            )

            for method, step_value in methods.items():
                if step_value is None:
                    continue
                step = int(step_value)
                path = autoregressive_forecast_path(
                    model, current, step=step, horizon=horizon
                )
                for depth, state in enumerate(path, start=1):
                    state_np = state.float().cpu().numpy()
                    if not np.all(np.isfinite(state_np)):
                        raise FloatingPointError(
                            f"{method} bounded forecast is non-finite"
                        )
                    intermediate_truth = _frame_slice(
                        states, batch_current_frames + step * depth
                    )
                    _score_path_state(
                        depth_accumulators[method][depth],
                        state_np,
                        intermediate_truth,
                        rows=rows,
                        seed=seed,
                        horizon=horizon,
                        method=method,
                        step=step,
                        depth=depth,
                        current_frames=batch_current_frames,
                    )
                terminal_np = path[-1].float().cpu().numpy()
                datasets[method][start:stop] = terminal_np - current_np
                terminal[method].update(terminal_np, target_np)

    return {
        "seed": seed,
        "horizon": horizon,
        "pair_count": int(count),
        "terminal": {
            method: accumulator.finalize(persistence_mse=persistence_mse)
            for method, accumulator in terminal.items()
        },
        "composition_depth": {
            method: {
                str(depth): {
                    "elapsed_frames": int(depth * int(methods[method])),
                    **accumulator.finalize(),
                }
                for depth, accumulator in depths.items()
            }
            for method, depths in depth_accumulators.items()
        },
    }


def _aggregate_seed_terminal(by_seed: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        result[key] = {}
        for method in method_schedule(horizon):
            result[key][method] = {}
            for field in FIELDS:
                values = np.asarray(
                    [
                        by_seed[str(seed)][key]["terminal"][method]["per_field"][field][
                            "persistence_relative_skill"
                        ]
                        for seed in SEEDS
                    ],
                    dtype=np.float64,
                )
                result[key][method][field] = {
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    fields = (
        "seed",
        "horizon",
        "method",
        "composition_step",
        "composition_depth",
        "elapsed_frames",
        "current_frame",
        "target_frame",
        "field",
        "standardized_rmse",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runtime_paths = (
        args.artifact_root,
        args.manifest,
        args.output,
        args.paper0_root,
    )
    for path in runtime_paths:
        assert_development_path(path)
        if "85606" in str(path).lower():
            raise ValueError("held-out 85606 paths are prohibited")
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 checkout commit differs")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(
        manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        paper0_commit=args.paper0_commit,
    )
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bounded rollout requires one allocated CUDA GPU")
    device = torch.device("cuda", CUDA_DEVICE_INDEX)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    reset_peak_cuda_memory_stats()
    catalog = load_official_catalog(args.artifact_root)
    states = load_validation_states(catalog)

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
        job_type="old-85604-bounded-rollout-evaluation",
        tags=(
            "paper0",
            "85604",
            "old-data",
            "codec-free",
            "multilead",
            "bounded-rollout",
            "evaluation",
        ),
    )
    api = wandb.Api(timeout=30)
    if not api.api_key:
        raise RuntimeError("W&B API key is absent")
    viewer = api.viewer
    if str(getattr(viewer, "entity", "")) != spec.entity:
        raise RuntimeError("authenticated W&B entity differs")
    tracking_directory = args.output / "wandb"
    tracking_directory.mkdir()
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
            "scope": SCOPE,
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "horizons": list(HORIZONS),
            "seeds": list(SEEDS),
            "methods": {str(horizon): method_schedule(horizon) for horizon in HORIZONS},
            "paper0_commit": args.paper0_commit,
            "manifest_sha256": args.manifest_sha256,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    started = time.perf_counter()
    forecast_path = args.output / "terminal_forecast_deltas.h5"
    partial_forecast = forecast_path.with_name(f".{forecast_path.name}.partial")
    rows: list[dict[str, Any]] = []
    persistence: dict[str, Any] = {}
    persistence_mse: dict[int, dict[str, float]] = {}
    for horizon in HORIZONS:
        record, mse, baseline_rows = _persistence_records(states, horizon=horizon)
        persistence[str(horizon)] = record
        persistence_mse[horizon] = mse
        rows.extend(baseline_rows)

    by_seed: dict[str, Any] = {}
    model_audit: dict[str, Any] = {}
    try:
        with h5py.File(partial_forecast, "x") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["scope"] = SCOPE
            handle.attrs["development_run"] = "85604"
            handle.attrs["held_out_85606_read"] = False
            handle.attrs["new_nersc_data_read"] = False
            handle.attrs["guard_frames_read"] = False
            handle.attrs["fields"] = json.dumps(list(FIELDS))
            handle.attrs["zperiod"] = 5
            handle.attrs[
                "stored_value"
            ] = "standardized_terminal_state_delta_from_current"
            handle.attrs["paper0_commit"] = args.paper0_commit
            handle.attrs["manifest_sha256"] = args.manifest_sha256
            for horizon in HORIZONS:
                group = handle.create_group(f"horizon_{horizon}")
                count = 624 - horizon - 496
                current = np.arange(496, 624 - horizon, dtype=np.int64)
                group.create_dataset("current_frame", data=current, track_times=False)
                group.create_dataset(
                    "target_frame", data=current + horizon, track_times=False
                )
                group.attrs["horizon_saved_frames"] = horizon
                group.attrs["pair_count"] = count
            for seed in SEEDS:
                model, audit = load_selected_model(manifest, seed=seed, device=device)
                model_audit[str(seed)] = audit
                by_seed[str(seed)] = {}
                for horizon in HORIZONS:
                    seed_group = handle[f"horizon_{horizon}"].create_group(
                        f"seed_{seed}"
                    )
                    by_seed[str(seed)][str(horizon)] = evaluate_one_seed_horizon(
                        model=model,
                        seed=seed,
                        horizon=horizon,
                        states=states,
                        device=device,
                        h5_group=seed_group,
                        persistence_mse=persistence_mse[horizon],
                        batch_size=4,
                        rows=rows,
                    )
                    for method, record in by_seed[str(seed)][str(horizon)][
                        "terminal"
                    ].items():
                        run.log(
                            {
                                (
                                    f"state/h{horizon}/{method}/"
                                    "mean_field_persistence_skill"
                                ): record["mean_field_persistence_relative_skill"]
                            },
                            step=(seed - 1700) * 10 + horizon,
                        )
                del model
                torch.cuda.empty_cache()
            handle.flush()
        os.replace(partial_forecast, forecast_path)

        state_metrics = {
            "schema_version": 1,
            "scope": SCOPE,
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "physics_derived_metric": False,
            "metric_space": "training_only_standardized_C5P_state",
            "fields": list(FIELDS),
            "persistence": persistence,
            "by_seed": by_seed,
            "seed_range": _aggregate_seed_terminal(by_seed),
        }
        state_metrics_path = args.output / "state_metrics.json"
        atomic_json(state_metrics_path, state_metrics)
        per_target_path = args.output / "per_target_state_rmse.csv"
        _write_csv(per_target_path, rows)

        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": "bounded_state_forecast_generated_and_scored",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_derived_loss_used": False,
            "physics_diagnostics_scored": False,
            "physics_scoring_authorized_next": True,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": str(args.manifest),
            "manifest_sha256": args.manifest_sha256,
            "models": model_audit,
            "forecast_artifact": {
                "path": str(forecast_path),
                "sha256": sha256_path(forecast_path),
                "stored_value": "standardized_terminal_state_delta_from_current",
                "dtype": "float32",
            },
            "state_metrics": {
                "path": str(state_metrics_path),
                "sha256": sha256_path(state_metrics_path),
            },
            "per_target_state_rmse": {
                "path": str(per_target_path),
                "sha256": sha256_path(per_target_path),
                "row_count": len(rows),
            },
            "gpu": torch.cuda.get_device_name(device),
            "peak_cuda_memory_GiB": peak_cuda_memory_gib(),
            "wall_seconds_before_wandb_verification": time.perf_counter() - started,
            "numeric_precision": {
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
        }
        run.summary.update(
            {
                "final/status": result["status"],
                "scope/held_out_85606_read": False,
                "scope/new_nersc_data_read": False,
                "scope/training_performed": False,
                "compute/peak_cuda_memory_GiB": result["peak_cuda_memory_GiB"],
                "compute/wall_seconds": result[
                    "wall_seconds_before_wandb_verification"
                ],
            }
        )
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise

    result_path = args.output / "result.json"
    atomic_json(result_path, result)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb,
        remote_path=remote_path,
        expected_id=spec.run_id,
    )
    tracking = {
        "schema_version": 1,
        "required": True,
        "mode": "online",
        "spec": spec.to_record(),
        "authenticated_username": str(getattr(viewer, "username", "")),
        "wandb_version": wandb.__version__,
        "run_url": run_url,
        "remote_path": remote_path,
        "remote_state_after_finish": remote_state,
        "checkpoints_uploaded": False,
        "forecast_uploaded": False,
        "local_artifacts_are_scientific_authority": True,
    }
    atomic_json(args.output / "wandb.json", tracking)
    index = args.output / "artifact_sha256.txt"
    index.write_text(
        "".join(
            f"{sha256_path(path)}  {path.resolve(strict=True)}\n"
            for path in sorted(args.output.iterdir())
            if path.is_file() and path != index
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
