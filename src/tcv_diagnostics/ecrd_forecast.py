"""Truth-separated M32 one-step forecasts for matched ECRD checkpoints."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor

from .b5_residual_edm_forecast import (
    B5ForecastSchema,
    initial_noise_from_uint64,
    load_scientific_sampler_seed_bank,
)
from .b5_residual_edm_full_training import (
    B5_FULL_VALIDATION_TARGETS,
    B5_SCIENTIFIC_SAMPLER_BANK_SEED,
    scientific_sampler_seed_bank,
)
from .codec_training import sha256_path
from .ecrd_data import multiscale_noise_from_uint64
from .ecrd_training import (
    ECRD_ARMS,
    ECRDTrainingConfig,
    build_model,
    model_config_record,
)
from .model_data import assert_development_path
from .model_training_data import VOLUME_SHAPE
from .models.ecrd import ECRDTransition, MultiscaleNoiseConfig
from .models.field_residual_edm import B5_RESIDUAL_SCALES, JointFieldResidualEDM
from .o2_training_data import strict_o2_targets


ECRD_FORECAST_AXES = (
    "target_frame",
    "ensemble_member",
    "future_time",
    "channel",
    "x",
    "y",
    "stored_toroidal_z",
)


class _ContextDataset(Protocol):
    split: str
    context_frames: int
    target_frames: tuple[int, ...]
    target_truth_read: bool

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        ...


class _ParentArtifact(Protocol):
    split: str
    target_frames: tuple[int, ...]

    def read(self, start: int, stop: int) -> np.ndarray:
        ...


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _mentions_held_out(value: Any) -> bool:
    if isinstance(value, str):
        return "85606" in value.lower()
    if isinstance(value, Mapping):
        return any(_mentions_held_out(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_mentions_held_out(item) for item in value)
    return False


def _noise_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


def initial_noise_for_arm(
    seed: int | np.uint64,
    *,
    arm: str,
    spatial_shape: Sequence[int] = VOLUME_SHAPE,
    noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> np.ndarray:
    """Expand one paired seed using the arm's frozen innovation law."""

    if arm not in ECRD_ARMS:
        raise ValueError("ECRD forecast arm differs")
    if arm in ("ECRD", "ECRD-History"):
        return multiscale_noise_from_uint64(
            seed, spatial_shape=spatial_shape, config=noise_config
        )
    return initial_noise_from_uint64(seed, spatial_shape=spatial_shape)


class ECRDForecastWriter(AbstractContextManager["ECRDForecastWriter"]):
    """Write a canonical ECRD ensemble without replacement or truth access."""

    def __init__(
        self,
        path: Path,
        *,
        target_frames: Sequence[int],
        arm: str,
        model_seed: int,
        history_frames: int,
        parent_kind: str,
        metadata: Mapping[str, Any],
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: B5ForecastSchema = B5ForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(self.path)
        if arm not in ECRD_ARMS:
            raise ValueError("ECRD forecast arm differs")
        if int(model_seed) not in (1701, 1702, 1703):
            raise ValueError("ECRD forecast model seed differs")
        expected_history = 2 if arm == "ECRD-History" else 1
        if int(history_frames) != expected_history:
            raise ValueError("ECRD forecast history differs")
        expected_parent = (
            "four_phase_symmetrized_H1"
            if arm in ("ECRD", "ECRD-History")
            else "original_unsymmetrized_H1"
        )
        if parent_kind != expected_parent:
            raise ValueError("ECRD forecast parent identity differs")
        frames = tuple(int(value) for value in target_frames)
        strict_o2_targets(
            frames, split="validation", context_frames=int(history_frames)
        )
        if _mentions_held_out(dict(metadata)):
            raise ValueError("ECRD forecast metadata mentions held-out data")
        self.seed_bank_path = Path(seed_bank_path).resolve(strict=True)
        self.seed_bank_sha256 = str(seed_bank_sha256)
        self.seed_bank = load_scientific_sampler_seed_bank(
            self.seed_bank_path, self.seed_bank_sha256
        )
        if schema.members > self.seed_bank.shape[1]:
            raise ValueError("ECRD forecast exceeds the frozen member bank")
        self.target_frames = frames
        self.arm = arm
        self.model_seed = int(model_seed)
        self.history_frames = int(history_frames)
        self.parent_kind = parent_kind
        self.metadata = dict(metadata)
        self.schema = schema
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: h5py.File | None = h5py.File(self.partial_path, "x")
        attributes = {
            "schema_version": 1,
            "scope": "ECRD_truth_separated_M32_one_step_forecast",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": schema.members,
            "arm": arm,
            "model_seed": self.model_seed,
            "history_frames": self.history_frames,
            "parent_kind": parent_kind,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "member_interaction": False,
            "member_prefixes_regenerated": False,
            "posthoc_calibration": False,
            "initial_noise_device": "CPU",
            "initial_noise_algorithm": "NumPy_PCG64_float32",
            "innovation_noise": (
                "global_plus_xy_mesoscale_plus_local_Gaussian"
                if arm in ("ECRD", "ECRD-History")
                else "elementwise_standard_normal"
            ),
            "sampler_seed_bank_seed": B5_SCIENTIFIC_SAMPLER_BANK_SEED,
            "sampler_seed_bank_sha256": self.seed_bank_sha256,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
        }
        for name, value in attributes.items():
            self.handle.attrs[name] = value
        self.handle.attrs["forecast_axes_json"] = json.dumps(ECRD_FORECAST_AXES)
        self.handle.attrs["schema_json"] = json.dumps(
            schema.to_record(), sort_keys=True, allow_nan=False
        )
        self.handle.attrs["metadata_json"] = json.dumps(
            self.metadata, sort_keys=True, allow_nan=False
        )
        self.handle.create_dataset(
            "target_frame_index", data=np.asarray(frames, dtype=np.int64)
        )
        self.forecast = self.handle.create_dataset(
            "standardized_forecast",
            shape=(len(frames), *schema.per_target_shape),
            dtype="f4",
            chunks=(1, 1, 1, 1, *schema.volume_shape),
            shuffle=True,
            fletcher32=True,
        )
        self.inference_seconds = self.handle.create_dataset(
            "model_inference_seconds", shape=(len(frames),), dtype="f8"
        )
        self.sampler_seeds = self.handle.create_dataset(
            "sampler_seed_uint64", shape=(len(frames), schema.members), dtype="u8"
        )
        self.initial_noise_sha256 = self.handle.create_dataset(
            "initial_noise_sha256",
            shape=(len(frames), schema.members),
            dtype="S64",
        )

    def append(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        inference_seconds: float,
        sampler_seed_row: np.ndarray,
        initial_noise_sha256: Sequence[str],
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("ECRD forecast writer is closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("ECRD forecast writer received too many targets")
        target = self.target_frames[self.cursor]
        if int(target_frame) != target:
            raise ValueError("ECRD forecast target order differs")
        values = np.asarray(standardized_forecast)
        if values.shape != self.schema.per_target_shape or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("ECRD forecast tensor shape or values differ")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("ECRD forecast inference time differs")
        seeds = np.asarray(sampler_seed_row)
        expected_seeds = self.seed_bank[
            target - B5_FULL_VALIDATION_TARGETS[0], : self.schema.members
        ]
        if (
            seeds.shape != (self.schema.members,)
            or seeds.dtype != np.uint64
            or not np.array_equal(seeds, expected_seeds)
        ):
            raise ValueError("ECRD forecast member seed row differs")
        digests = tuple(str(value) for value in initial_noise_sha256)
        if len(digests) != self.schema.members or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digests
        ):
            raise ValueError("ECRD initial-noise hashes differ")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.sampler_seeds[self.cursor] = seeds
        self.initial_noise_sha256[self.cursor] = np.asarray(
            [value.encode("ascii") for value in digests], dtype="S64"
        )
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed or self.handle is None:
            raise RuntimeError("ECRD forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("ECRD forecast writer did not receive every target")
        self.handle.attrs["completed"] = True
        self.handle.flush()
        self.handle.close()
        self.handle = None
        os.replace(self.partial_path, self.path)
        self.completed = True
        return self.path

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        return None


class ECRDForecastArtifact(AbstractContextManager["ECRDForecastArtifact"]):
    """Hash-checked read access to a closed ECRD ensemble artifact."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        target_frames: Sequence[int],
        arm: str,
        model_seed: int,
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: B5ForecastSchema = B5ForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if sha256_path(self.path) != str(expected_sha256):
            raise ValueError("ECRD forecast SHA-256 differs")
        self.sha256 = str(expected_sha256)
        self.target_frames = tuple(int(value) for value in target_frames)
        self.arm = arm
        self.model_seed = int(model_seed)
        self.schema = schema
        self.seed_bank_path = Path(seed_bank_path)
        self.seed_bank_sha256 = str(seed_bank_sha256)
        self.seed_bank = load_scientific_sampler_seed_bank(
            self.seed_bank_path, self.seed_bank_sha256
        )
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("ECRD forecast artifact is closed")
        expected = {
            "schema_version": 1,
            "scope": "ECRD_truth_separated_M32_one_step_forecast",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": self.schema.members,
            "arm": self.arm,
            "model_seed": self.model_seed,
            "history_frames": 2 if self.arm == "ECRD-History" else 1,
            "parent_kind": (
                "four_phase_symmetrized_H1"
                if self.arm in ("ECRD", "ECRD-History")
                else "original_unsymmetrized_H1"
            ),
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "member_interaction": False,
            "member_prefixes_regenerated": False,
            "posthoc_calibration": False,
            "initial_noise_device": "CPU",
            "initial_noise_algorithm": "NumPy_PCG64_float32",
            "innovation_noise": (
                "global_plus_xy_mesoscale_plus_local_Gaussian"
                if self.arm in ("ECRD", "ECRD-History")
                else "elementwise_standard_normal"
            ),
            "sampler_seed_bank_seed": B5_SCIENTIFIC_SAMPLER_BANK_SEED,
            "sampler_seed_bank_sha256": self.seed_bank_sha256,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "completed": True,
        }
        for name, value in expected.items():
            if name not in self.handle.attrs:
                raise ValueError(f"ECRD forecast attribute {name!r} is absent")
            actual = self.handle.attrs[name]
            if isinstance(value, str):
                actual = _text(actual)
            elif isinstance(value, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != value:
                raise ValueError(f"ECRD forecast attribute {name!r} differs")
        if tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"]))) != (
            ECRD_FORECAST_AXES
        ):
            raise ValueError("ECRD forecast axes differ")
        if json.loads(_text(self.handle.attrs["schema_json"])) != self.schema.to_record():
            raise ValueError("ECRD forecast schema differs")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("ECRD forecast target frames differ")
        expected_shape = (len(self.target_frames), *self.schema.per_target_shape)
        forecast = self.handle["standardized_forecast"]
        if forecast.shape != expected_shape or forecast.dtype != np.dtype("f4"):
            raise ValueError("ECRD forecast tensor schema differs")
        expected_seeds = np.stack(
            [
                self.seed_bank[
                    target - B5_FULL_VALIDATION_TARGETS[0], : self.schema.members
                ]
                for target in self.target_frames
            ]
        )
        if not np.array_equal(self.handle["sampler_seed_uint64"][:], expected_seeds):
            raise ValueError("ECRD forecast seed values differ")
        digests = self.handle["initial_noise_sha256"][:]
        if digests.shape != (len(self.target_frames), self.schema.members) or any(
            len(_text(value)) != 64
            or any(character not in "0123456789abcdef" for character in _text(value))
            for value in digests.reshape(-1)
        ):
            raise ValueError("ECRD forecast initial-noise hashes differ")
        times = np.asarray(self.handle["model_inference_seconds"][:])
        if not np.all(np.isfinite(times) & (times >= 0.0)):
            raise ValueError("ECRD forecast timing differs")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if _mentions_held_out(self.metadata):
            raise ValueError("ECRD forecast metadata mentions held-out data")

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("ECRD forecast artifact is closed")
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("ECRD forecast read interval differs")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("ECRD forecast contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        if self.handle is None:
            raise RuntimeError("ECRD forecast artifact is closed")
        values = np.asarray(
            self.handle["model_inference_seconds"][:], dtype=np.float64
        )
        return {
            "target_count": int(values.size),
            "ensemble_members_per_target": self.schema.members,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "total_seconds": float(np.sum(values)),
            "mean_seconds_per_target": float(np.mean(values)),
            "median_seconds_per_target": float(np.median(values)),
        }

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        self.close()
        return None


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_selected_ecrd_model(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    arm: str,
    seed: int,
    training_commit: str,
    device: torch.device,
) -> JointFieldResidualEDM | ECRDTransition:
    """Load an exact selected checkpoint without result-dependent reselection."""

    selected = Path(checkpoint)
    assert_development_path(selected)
    if sha256_path(selected) != str(expected_checkpoint_sha256):
        raise ValueError("ECRD selected checkpoint SHA-256 differs")
    payload = _load_torch(selected)
    config = ECRDTrainingConfig(arm=arm, seed=int(seed), mode="full")
    if (
        payload.get("kind") != "ECRD_selected_EMA_checkpoint"
        or payload.get("paper0_commit") != str(training_commit)
        or payload.get("training") != config.to_record()
        or payload.get("model") != model_config_record(arm)
        or payload.get("residual_scales") != list(B5_RESIDUAL_SCALES)
        or payload.get("selected_completed_epoch") not in range(5, 101, 5)
        or payload.get("physics_metric_used_for_selection") is not False
        or payload.get("held_out_85606_read") is not False
    ):
        raise ValueError("ECRD selected checkpoint provenance differs")
    model = build_model(arm).to(device, torch.float32)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def sample_ecrd_target_from_seeds(
    *,
    model: JointFieldResidualEDM | ECRDTransition,
    arm: str,
    context: Tensor,
    parent_mean: Tensor,
    complete_member_seeds: np.ndarray,
    member_batch_size: int,
    noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> tuple[Tensor, tuple[str, ...]]:
    """Sample complete paired member prefixes without member interaction."""

    schema = B5ForecastSchema.frozen()
    history = 2 if arm == "ECRD-History" else 1
    if tuple(context.shape) != (1, history, 5, *VOLUME_SHAPE):
        raise ValueError("ECRD forecast context shape differs")
    if tuple(parent_mean.shape) != (1, 5, *VOLUME_SHAPE):
        raise ValueError("ECRD parent mean shape differs")
    seeds = np.asarray(complete_member_seeds)
    if seeds.shape != (schema.members,) or seeds.dtype != np.uint64:
        raise ValueError("ECRD complete member seed row differs")
    batch_size = int(member_batch_size)
    if not 0 < batch_size <= schema.members:
        raise ValueError("ECRD member batch size differs")
    condition = torch.cat(
        (context.reshape(1, history * 5, *VOLUME_SHAPE), parent_mean), dim=1
    )
    forecasts: list[Tensor] = []
    digests: list[str] = []
    for start in range(0, schema.members, batch_size):
        stop = min(start + batch_size, schema.members)
        noise_values = [
            initial_noise_for_arm(
                value, arm=arm, spatial_shape=VOLUME_SHAPE, noise_config=noise_config
            )
            for value in seeds[start:stop]
        ]
        digests.extend(_noise_sha256(value) for value in noise_values)
        initial = torch.from_numpy(np.stack(noise_values))[None].to(
            device=context.device, dtype=torch.float32, non_blocking=True
        )
        normalized = model.sample_normalized(
            condition,
            initial,
            steps=18,
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
        )
        if isinstance(model, ECRDTransition):
            composed = model.compose_fields(parent_mean, condition, normalized)[0]
        else:
            composed = model.compose_fields(parent_mean, normalized)[0]
        if tuple(composed.shape) != (stop - start, 1, 5, *VOLUME_SHAPE):
            raise RuntimeError("ECRD sampled member-batch shape differs")
        forecasts.append(composed.to("cpu", torch.float32))
    result = torch.cat(forecasts, dim=0)
    if tuple(result.shape) != schema.per_target_shape or not torch.isfinite(result).all():
        raise RuntimeError("ECRD canonical sampled forecast differs")
    return result, tuple(digests)


def generate_selected_ecrd_forecasts(
    *,
    model: JointFieldResidualEDM | ECRDTransition,
    arm: str,
    model_seed: int,
    dataset: _ContextDataset,
    parent_artifact: _ParentArtifact,
    target_frames: Sequence[int],
    seed_bank: np.ndarray,
    seed_bank_path: Path,
    seed_bank_sha256: str,
    output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
    member_batch_size: int = 8,
    bounded_smoke: bool = False,
) -> dict[str, Any]:
    """Generate M32 and close it before any target truth is opened."""

    history = 2 if arm == "ECRD-History" else 1
    targets = strict_o2_targets(
        target_frames, split="validation", context_frames=history
    )
    required = tuple(range(498, 502)) if bounded_smoke else B5_FULL_VALIDATION_TARGETS
    if targets != required:
        raise ValueError("ECRD forecast target interval differs")
    if (
        dataset.split != "validation"
        or dataset.target_frames != targets
        or dataset.context_frames != history
        or dataset.target_truth_read is not False
    ):
        raise ValueError("ECRD truth-free context dataset differs")
    if (
        parent_artifact.split != "validation"
        or parent_artifact.target_frames != B5_FULL_VALIDATION_TARGETS
    ):
        raise ValueError("ECRD validation parent coverage differs")
    bank = np.asarray(seed_bank)
    if bank.shape != (126, 32) or bank.dtype != np.uint64:
        raise ValueError("ECRD scientific seed bank schema differs")
    if not np.array_equal(bank, scientific_sampler_seed_bank()):
        raise ValueError("ECRD scientific seed bank values differ")
    if sha256_path(seed_bank_path) != str(seed_bank_sha256):
        raise ValueError("ECRD scientific seed bank persisted hash differs")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ECRD scientific forecast generation requires CUDA")
    parent_kind = (
        "four_phase_symmetrized_H1"
        if arm in ("ECRD", "ECRD-History")
        else "original_unsymmetrized_H1"
    )
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    started_wall = time.monotonic()
    with ECRDForecastWriter(
        output,
        target_frames=targets,
        arm=arm,
        model_seed=model_seed,
        history_frames=history,
        parent_kind=parent_kind,
        metadata=metadata,
        seed_bank_path=seed_bank_path,
        seed_bank_sha256=seed_bank_sha256,
    ) as writer:
        with torch.inference_mode():
            for position, target in enumerate(targets):
                item = dataset[position]
                if int(item["target_frame_index"]) != target:
                    raise RuntimeError("ECRD forecast target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError("ECRD forecast context contains target truth")
                context = torch.from_numpy(np.asarray(item["context"]))[None].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                parent_position = target - B5_FULL_VALIDATION_TARGETS[0]
                parent = torch.from_numpy(
                    parent_artifact.read(parent_position, parent_position + 1)
                ).to(device=device, dtype=torch.float32, non_blocking=True)
                row = bank[parent_position]
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                forecast, noise_hashes = sample_ecrd_target_from_seeds(
                    model=model,
                    arm=arm,
                    context=context,
                    parent_mean=parent,
                    complete_member_seeds=row,
                    member_batch_size=member_batch_size,
                )
                torch.cuda.synchronize(device)
                writer.append(
                    target_frame=target,
                    standardized_forecast=forecast.numpy(),
                    inference_seconds=time.perf_counter() - started,
                    sampler_seed_row=row,
                    initial_noise_sha256=noise_hashes,
                )
        writer.finalize()
    output_path = Path(output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_ECRD_M32_forecast_smoke_85604"
            if bounded_smoke
            else "ECRD_truth_separated_M32_one_step_forecast_85604"
        ),
        "bounded_non_scientific_smoke": bool(bounded_smoke),
        "development_run": "85604",
        "arm": arm,
        "model_seed": int(model_seed),
        "history_frames": history,
        "parent_kind": parent_kind,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "forecast": {
            "path": str(output_path),
            "sha256": sha256_path(output_path),
            "axes": list(ECRD_FORECAST_AXES),
            "shape": [len(targets), *B5ForecastSchema.frozen().per_target_shape],
        },
        "sampler_seed_bank": {
            "path": str(Path(seed_bank_path).resolve(strict=True)),
            "sha256": str(seed_bank_sha256),
            "seed": B5_SCIENTIFIC_SAMPLER_BANK_SEED,
        },
        "inference": {
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "member_batch_size": int(member_batch_size),
        },
        "wall_seconds": time.monotonic() - started_wall,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "posthoc_calibration": False,
        "metadata": dict(metadata),
    }
