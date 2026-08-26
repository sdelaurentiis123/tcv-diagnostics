#!/usr/bin/env python3
"""Generate causal bounded forecasts for one frozen matched state-view arm."""

from __future__ import annotations

import argparse
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
from tcv_diagnostics.bounded_rollout import method_schedule
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.matched_codec_evaluation import (
    NATIVE81_FIELDS,
    decode_physical_batch,
    native81_candidate_fields,
)
from tcv_diagnostics.model_data import assert_development_path, load_strict_json
from tcv_diagnostics.model_training_data import (
    FAMILY_FIELDS,
    CodecFrameDataset,
    VOLUME_SHAPE,
    load_official_catalog,
)
from tcv_diagnostics.models.codec_free_operator import (
    CodecFreeIncrementOperator3D,
    CodecFreeOperatorConfig,
)
from tcv_diagnostics.state_view_rollout import (
    autoregressive_state_forecast_path,
    direct_state_forecast,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SCOPE = "post_ecrd_old_85604_matched_state_bounded_generation"
TRAINING_SCOPE = "post_ecrd_old_85604_matched_state_multilead_pilot"
REDUCTION_SCOPE = "post_ecrd_old_85604_matched_state_multilead_reduction"
FAMILIES = ("c5p", "e6b")
HORIZONS = (4, 8)
VALIDATION_START = 496
VALIDATION_STOP = 624
NATIVE_SHAPE = (64, 32, 81)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper0-root", type=Path, required=True)
    parser.add_argument("--paper0-commit", required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
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


def locked_json(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    assert_development_path(path)
    if not digest or sha256_path(path) != digest:
        raise ValueError(f"{label} SHA-256 differs")
    return load_strict_json(path)


def authorize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    paper0_commit: str,
    family: str,
) -> None:
    if sha256_path(manifest_path) != manifest_sha256:
        raise ValueError("bounded-generation manifest SHA-256 differs")
    expected = {
        "scope": SCOPE,
        "status": "frozen_after_paired_transition_reduction_before_inference",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "new_nersc_data_access_allowed": False,
        "guard_frames_read_allowed": False,
        "training_allowed": False,
        "checkpoint_selection_allowed": False,
        "paper0_commit_at_freeze": paper0_commit,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "wandb_required": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("bounded-generation manifest scope differs")
    if family not in FAMILIES:
        raise ValueError("bounded-generation family differs")

    reduction = locked_json(
        manifest.get("evidence", {}).get("paired_reduction", {}),
        label="paired transition reduction",
    )
    if (
        reduction.get("scope") != REDUCTION_SCOPE
        or reduction.get("development_run") != "85604"
        or reduction.get("held_out_85606_read") is not False
        or reduction.get("new_nersc_data_read") is not False
        or reduction.get("paired_physics_evaluation_authorized") is not True
    ):
        raise ValueError("paired reduction does not authorize bounded generation")

    models = manifest.get("evidence", {}).get("models", {})
    if set(models) != set(FAMILIES):
        raise ValueError("bounded-generation model pair differs")
    for model_family in FAMILIES:
        record = models[model_family]
        if record.get("family") != model_family or int(record.get("seed", -1)) != 1701:
            raise ValueError("bounded-generation model identity differs")
        result = locked_json(
            record.get("result", {}), label=f"{model_family} training result"
        )
        checkpoint_path = Path(str(record.get("selected_checkpoint", {}).get("path", "")))
        checkpoint_sha = str(
            record.get("selected_checkpoint", {}).get("sha256", "")
        )
        assert_development_path(checkpoint_path)
        if not checkpoint_sha or sha256_path(checkpoint_path) != checkpoint_sha:
            raise ValueError(f"{model_family} selected checkpoint SHA-256 differs")
        selected = result.get("best_checkpoint", {})
        if (
            result.get("scope") != TRAINING_SCOPE
            or result.get("family") != model_family
            or result.get("status") != "passed"
            or result.get("transition_gate", {}).get("passed") is not True
            or Path(str(selected.get("path", ""))) != checkpoint_path
            or selected.get("sha256") != checkpoint_sha
        ):
            raise ValueError(f"{model_family} selected model differs")

    evaluation = manifest.get("evaluation", {})
    expected_horizons = {
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
    if (
        evaluation.get("validation_frames") != [496, 624]
        or evaluation.get("history_frames") != 1
        or evaluation.get("inference_batch_size") != 4
        or evaluation.get("target_truth_used_during_generation") is not False
        or evaluation.get("complete_predicted_state_fed_back") is not True
        or evaluation.get("horizons") != expected_horizons
    ):
        raise ValueError("bounded-generation evaluation contract differs")


def _config_from_record(record: Mapping[str, Any]) -> CodecFreeOperatorConfig:
    return CodecFreeOperatorConfig(
        state_family=str(record["state_family"]),
        history_frames=int(record["history_frames"]),
        base_channels=int(record["base_channels"]),
        channel_multipliers=tuple(int(item) for item in record["channel_multipliers"]),
        blocks_per_level=int(record["blocks_per_level"]),
        lead_embedding_channels=int(record["lead_embedding_channels"]),
        group_norm_maximum_groups=int(record["group_norm_maximum_groups"]),
        kernel_size=int(record["kernel_size"]),
        predict_boundary=bool(record["predict_boundary"]),
        zero_initialize_output=bool(record["zero_initialize_output"]),
        auxiliary_context_channels=int(record.get("auxiliary_context_channels", 0)),
    )


def load_model(
    manifest: Mapping[str, Any],
    *,
    family: str,
    device: torch.device,
) -> tuple[CodecFreeIncrementOperator3D, dict[str, Any]]:
    record = manifest["evidence"]["models"][family]
    result = load_strict_json(Path(str(record["result"]["path"])))
    checkpoint_path = Path(str(record["selected_checkpoint"]["path"]))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = _config_from_record(
        result.get("architecture", {}).get("architecture", {})
    )
    if (
        config.state_family != family
        or payload.get("family") != family
        or payload.get("stage") != "matched_state_multilead_pilot"
        or int(payload.get("seed", -1)) != 1701
        or int(payload.get("epoch", -1)) != int(result["best_checkpoint"]["epoch"])
        or float(payload.get("selection_metric", float("nan")))
        != float(result["best_checkpoint"]["selection_metric"])
        or payload.get("config") != config.to_record()
    ):
        raise ValueError("selected bounded-generation checkpoint differs")
    model = CodecFreeIncrementOperator3D(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    bitwise = all(
        torch.equal(payload["model"][name].to(device), model.state_dict()[name])
        for name in payload["model"]
    )
    if not bitwise:
        raise AssertionError("selected bounded-generation model did not reload bitwise")
    model.eval()
    return model, {
        "family": family,
        "seed": 1701,
        "training_result": dict(record["result"]),
        "checkpoint": dict(record["selected_checkpoint"]),
        "selected_epoch": int(result["best_checkpoint"]["epoch"]),
        "checkpoint_reload_bitwise": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def load_current_states(
    catalog: Any,
    *,
    family: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray] | None]:
    frames = tuple(range(VALIDATION_START, VALIDATION_STOP - min(HORIZONS)))
    dataset = CodecFrameDataset(
        catalog,
        family=family,
        split="validation",
        frames=frames,
        augment=False,
        seed=0,
        return_physical=False,
    )
    volumes: dict[int, np.ndarray] = {}
    boundaries: dict[int, np.ndarray] | None = {} if family == "e6b" else None
    try:
        for index, frame in enumerate(frames):
            item = dataset[index]
            if int(item["frame_index"]) != frame:
                raise ValueError("validation current-frame order differs")
            volumes[frame] = np.ascontiguousarray(item["volume"], dtype=np.float32)
            if boundaries is not None:
                boundaries[frame] = np.ascontiguousarray(
                    item["boundary"], dtype=np.float32
                )
    finally:
        dataset.close()
    if not all(np.all(np.isfinite(value)) for value in volumes.values()):
        raise ValueError("current volume state is non-finite")
    if boundaries is not None and not all(
        np.all(np.isfinite(value)) for value in boundaries.values()
    ):
        raise ValueError("current boundary state is non-finite")
    return volumes, boundaries


class E6BCandidateWriter:
    """Atomically stream one predicted E6B method into the elliptic input."""

    def __init__(self, path: Path, *, target_frames: np.ndarray, method: str) -> None:
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        if path.exists() or self.temporary.exists():
            raise FileExistsError(path)
        self.handle = h5py.File(self.temporary, "x")
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["scope"] = SCOPE
        self.handle.attrs["development_run"] = "85604"
        self.handle.attrs["held_out_85606_read"] = False
        self.handle.attrs["new_nersc_data_read"] = False
        self.handle.attrs["family"] = "e6b"
        self.handle.attrs["method"] = method
        self.handle.attrs["zperiod"] = 5
        self.handle.attrs["boundary_policy"] = "predicted_Bphi_no_truth_bypass"
        self.handle.attrs["target_truth_used_during_generation"] = False
        coordinates = self.handle.create_group("coordinates")
        coordinates.create_dataset(
            "frame_index", data=np.asarray(target_frames, dtype=np.int64)
        )
        candidate = self.handle.create_group("candidate")
        self.fields = {
            field: candidate.create_dataset(
                field,
                shape=(target_frames.size, *NATIVE_SHAPE),
                dtype="f4",
                chunks=(1, *NATIVE_SHAPE),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            for field in NATIVE81_FIELDS["e6b"]
        }
        boundary = self.handle.create_group("boundary")
        self.boundary = boundary.create_dataset(
            "Bphi",
            shape=(target_frames.size, 2, 32),
            dtype="f4",
            chunks=(1, 2, 32),
        )
        self.written = np.zeros(target_frames.size, dtype=bool)

    def write(
        self,
        start: int,
        *,
        native_fields: Mapping[str, np.ndarray],
        boundary: np.ndarray,
    ) -> None:
        count = np.asarray(boundary).shape[0]
        stop = start + count
        if start < 0 or stop > self.written.size or np.any(self.written[start:stop]):
            raise ValueError("E6B candidate write overlaps or leaves its range")
        if set(native_fields) != set(self.fields):
            raise ValueError("E6B candidate field set differs")
        for field, values in native_fields.items():
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (count, *NATIVE_SHAPE) or not np.all(
                np.isfinite(array)
            ):
                raise ValueError(f"invalid E6B candidate field {field}")
            self.fields[field][start:stop] = array
        side_state = np.asarray(boundary, dtype=np.float32)
        if side_state.shape != (count, 2, 32) or not np.all(
            np.isfinite(side_state)
        ):
            raise ValueError("invalid predicted E6B boundary")
        self.boundary[start:stop] = side_state
        self.written[start:stop] = True

    def finish(self) -> None:
        if not np.all(self.written):
            raise RuntimeError("E6B candidate does not cover every target frame")
        self.handle.flush()
        self.handle.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        try:
            self.handle.close()
        finally:
            if self.temporary.exists():
                self.temporary.unlink()


def _create_forecast_datasets(
    group: h5py.Group,
    *,
    family: str,
    count: int,
) -> tuple[h5py.Dataset, h5py.Dataset | None]:
    volume = group.create_dataset(
        "volume",
        shape=(count, len(FAMILY_FIELDS[family]), *VOLUME_SHAPE),
        dtype="f4",
        chunks=(1, 1, *VOLUME_SHAPE),
        compression="gzip",
        compression_opts=1,
        shuffle=True,
        fletcher32=True,
        track_times=False,
    )
    boundary = None
    if family == "e6b":
        boundary = group.create_dataset(
            "boundary_Bphi",
            shape=(count, 2, 32),
            dtype="f4",
            chunks=(1, 2, 32),
            track_times=False,
        )
    return volume, boundary


def _physical_e6b_candidate(
    catalog: Any,
    volume: np.ndarray,
    boundary: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    physical = decode_physical_batch(
        catalog.normalization,
        FAMILY_FIELDS["e6b"],
        volume,
    )
    native = native81_candidate_fields("e6b", physical)
    physical_boundary = np.stack(
        [catalog.normalization.decode_boundary(frame) for frame in boundary],
        axis=0,
    )
    return native, np.ascontiguousarray(physical_boundary, dtype=np.float32)


def generate(
    *,
    model: CodecFreeIncrementOperator3D,
    catalog: Any,
    family: str,
    current_volumes: Mapping[int, np.ndarray],
    current_boundaries: Mapping[int, np.ndarray] | None,
    output: Path,
    batch_size: int,
) -> tuple[Path, list[Path], dict[str, Any]]:
    forecast_path = output / f"{family}_bounded_forecasts.h5"
    temporary = forecast_path.with_name(f".{forecast_path.name}.tmp.{os.getpid()}")
    if forecast_path.exists() or temporary.exists():
        raise FileExistsError(forecast_path)
    candidates_directory = output / "elliptic_candidates"
    candidates_directory.mkdir()
    candidate_paths: list[Path] = []
    records: dict[str, Any] = {}
    started = time.perf_counter()
    device = next(model.parameters()).device

    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs["schema_version"] = 1
            handle.attrs["scope"] = SCOPE
            handle.attrs["development_run"] = "85604"
            handle.attrs["held_out_85606_read"] = False
            handle.attrs["new_nersc_data_read"] = False
            handle.attrs["family"] = family
            handle.attrs["zperiod"] = 5
            handle.attrs["target_truth_used_during_generation"] = False

            with torch.inference_mode():
                for horizon in HORIZONS:
                    current_frames = np.arange(
                        VALIDATION_START,
                        VALIDATION_STOP - horizon,
                        dtype=np.int64,
                    )
                    target_frames = current_frames + horizon
                    horizon_group = handle.create_group(f"horizon_{horizon}")
                    coordinates = horizon_group.create_group("coordinates")
                    coordinates.create_dataset("current_frame", data=current_frames)
                    coordinates.create_dataset("target_frame", data=target_frames)
                    methods_group = horizon_group.create_group("methods")
                    method_datasets = {
                        method: _create_forecast_datasets(
                            methods_group.create_group(method),
                            family=family,
                            count=current_frames.size,
                        )
                        for method in method_schedule(horizon)
                    }
                    writers: dict[str, E6BCandidateWriter] = {}
                    if family == "e6b":
                        for method in method_schedule(horizon):
                            path = (
                                candidates_directory
                                / f"h{horizon}_{method}_predicted_e6b_native81.h5"
                            )
                            writers[method] = E6BCandidateWriter(
                                path,
                                target_frames=target_frames,
                                method=method,
                            )
                            candidate_paths.append(path)

                    try:
                        for start in range(0, current_frames.size, batch_size):
                            stop = min(current_frames.size, start + batch_size)
                            batch_frames = current_frames[start:stop]
                            volume_np = np.stack(
                                [current_volumes[int(frame)] for frame in batch_frames],
                                axis=0,
                            )
                            boundary_np = None
                            if family == "e6b":
                                if current_boundaries is None:
                                    raise AssertionError("E6B current boundary disappeared")
                                boundary_np = np.stack(
                                    [
                                        current_boundaries[int(frame)]
                                        for frame in batch_frames
                                    ],
                                    axis=0,
                                )
                            volume = torch.from_numpy(volume_np).to(device)
                            boundary = (
                                None
                                if boundary_np is None
                                else torch.from_numpy(boundary_np).to(device)
                            )
                            terminal: dict[
                                str, tuple[torch.Tensor, torch.Tensor | None]
                            ] = {}
                            terminal["direct"] = direct_state_forecast(
                                model,
                                volume,
                                boundary,
                                family=family,
                                horizon=horizon,
                            )
                            for method, step_value in method_schedule(horizon).items():
                                if step_value is None:
                                    continue
                                path = autoregressive_state_forecast_path(
                                    model,
                                    volume,
                                    boundary,
                                    family=family,
                                    step=int(step_value),
                                    horizon=horizon,
                                )
                                terminal[method] = path[-1]

                            for method, (state, side_state) in terminal.items():
                                state_np = np.ascontiguousarray(
                                    state.float().cpu().numpy(), dtype=np.float32
                                )
                                if not np.all(np.isfinite(state_np)):
                                    raise FloatingPointError(
                                        f"{family} {method} state is non-finite"
                                    )
                                volume_dataset, boundary_dataset = method_datasets[
                                    method
                                ]
                                volume_dataset[start:stop] = state_np
                                side_np = None
                                if family == "e6b":
                                    if side_state is None or boundary_dataset is None:
                                        raise AssertionError(
                                            "E6B forecast boundary disappeared"
                                        )
                                    side_np = np.ascontiguousarray(
                                        side_state.float().cpu().numpy(),
                                        dtype=np.float32,
                                    )
                                    if not np.all(np.isfinite(side_np)):
                                        raise FloatingPointError(
                                            f"E6B {method} boundary is non-finite"
                                        )
                                    boundary_dataset[start:stop] = side_np
                                    native, physical_boundary = _physical_e6b_candidate(
                                        catalog, state_np, side_np
                                    )
                                    writers[method].write(
                                        start,
                                        native_fields=native,
                                        boundary=physical_boundary,
                                    )
                    except Exception:
                        for writer in writers.values():
                            writer.abort()
                        raise
                    else:
                        for writer in writers.values():
                            writer.finish()

                    records[str(horizon)] = {
                        "current_frames": [
                            int(current_frames[0]),
                            int(current_frames[-1]) + 1,
                        ],
                        "target_frames": [
                            int(target_frames[0]),
                            int(target_frames[-1]) + 1,
                        ],
                        "pair_count": int(current_frames.size),
                        "methods": method_schedule(horizon),
                    }
            handle.flush()
        os.replace(temporary, forecast_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return forecast_path, candidate_paths, {
        "horizons": records,
        "batch_size": batch_size,
        "wall_seconds": time.perf_counter() - started,
        "target_truth_used_during_generation": False,
    }


def main() -> None:
    args = parse_args()
    for path in (args.artifact_root, args.manifest, args.output, args.paper0_root):
        assert_development_path(path)
        if "85606" in str(path).lower():
            raise ValueError("held-out 85606 paths are prohibited")
    if args.output.exists():
        raise FileExistsError(args.output)
    if repository_commit(args.paper0_root) != args.paper0_commit:
        raise ValueError("Paper 0 commit differs from generation lock")
    manifest = load_strict_json(args.manifest)
    authorize_manifest(
        manifest,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        paper0_commit=args.paper0_commit,
        family=args.family,
    )
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bounded generation requires one allocated CUDA GPU")
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)

    catalog = load_official_catalog(args.artifact_root)
    model, model_record = load_model(manifest, family=args.family, device=device)
    current_volumes, current_boundaries = load_current_states(
        catalog, family=args.family
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
        job_type="old-85604-matched-state-bounded-generation",
        tags=(
            "paper0",
            "85604",
            "old-data",
            args.family,
            "matched-state-view",
            "bounded-rollout",
            "inference-only",
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
            "family": args.family,
            "horizons": list(HORIZONS),
            "batch_size": 4,
            "paper0_commit": args.paper0_commit,
            "model": model_record,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "target_truth_used_during_generation": False,
        },
        mode="online",
        dir=str(tracking_directory),
        save_code=False,
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None or bool(run.offline):
        raise RuntimeError("W&B did not initialize online")

    try:
        forecast, candidate_paths, timing = generate(
            model=model,
            catalog=catalog,
            family=args.family,
            current_volumes=current_volumes,
            current_boundaries=current_boundaries,
            output=args.output,
            batch_size=4,
        )
        result = {
            "schema_version": 1,
            "scope": SCOPE,
            "status": "completed",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "physics_evaluation_performed": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
            "target_truth_used_during_generation": False,
            "family": args.family,
            "paper0_commit": args.paper0_commit,
            "slurm_job_id": args.slurm_job_id,
            "manifest": {
                "path": str(args.manifest),
                "sha256": args.manifest_sha256,
            },
            "model": model_record,
            "consumed_current_frames": [496, 620],
            "unread_terminal_truth_frames": [620, 624],
            "forecast": {"path": str(forecast), "sha256": sha256_path(forecast)},
            "elliptic_candidates": [
                {"path": str(path), "sha256": sha256_path(path)}
                for path in candidate_paths
            ],
            "timing": timing,
            "peak_cuda_memory_GiB": torch.cuda.max_memory_allocated(device) / 2**30,
            "gpu": torch.cuda.get_device_name(device),
        }
        run.summary.update(
            {
                "final/status": "completed",
                "final/family": args.family,
                "compute/wall_seconds": timing["wall_seconds"],
                "compute/peak_cuda_memory_GiB": result[
                    "peak_cuda_memory_GiB"
                ],
                "scope/held_out_85606_read": False,
                "scope/new_nersc_data_read": False,
                "scope/target_truth_used_during_generation": False,
            }
        )
        run_url = str(run.url)
        run.finish(exit_code=0)
    except Exception:
        run.finish(exit_code=1)
        raise

    atomic_json(args.output / "result.json", result)
    remote_path = f"{spec.entity}/{spec.project}/{spec.run_id}"
    remote_state = verify_finished_wandb_run(
        module=wandb,
        remote_path=remote_path,
        expected_id=spec.run_id,
    )
    atomic_json(
        args.output / "wandb.json",
        {
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
            "local_artifacts_are_scientific_authority": True,
        },
    )
    index = args.output / "artifact_sha256.txt"
    index.write_text(
        "".join(
            f"{sha256_path(path)}  {path}\n"
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path != index and "wandb" not in path.parts
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
