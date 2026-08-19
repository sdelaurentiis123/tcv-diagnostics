"""Truth-separated M32 forecast artifacts for the selected B5 residual EDM."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor

from .b5_residual_edm_full_training import (
    B5EDMFullConfig,
    B5_FULL_VALIDATION_TARGETS,
    B5_SCIENTIFIC_BANK_NPY_SHA256,
    B5_SCIENTIFIC_SAMPLER_BANK_SEED,
    scientific_sampler_seed_bank,
)
from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import VOLUME_SHAPE
from .models.field_residual_edm import (
    B5_RESIDUAL_SCALES,
    FieldResidualUNet3D,
    FieldResidualUNetConfig,
    JointFieldResidualEDM,
)
from .o2_context_data import OneStepContextDataset
from .o2_training_data import strict_o2_targets


B5_FORECAST_AXES = (
    "target_frame",
    "ensemble_member",
    "future_time",
    "channel",
    "x",
    "y",
    "stored_toroidal_z",
)


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _metadata_mentions_held_out(value: Any) -> bool:
    if isinstance(value, str):
        return "85606" in value.lower()
    if isinstance(value, Mapping):
        return any(_metadata_mentions_held_out(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_metadata_mentions_held_out(item) for item in value)
    return False


@dataclass(frozen=True)
class B5ForecastSchema:
    """Canonical B5 schema; smaller alternatives are for unit tests only."""

    members: int = 32
    future_frames: int = 1
    channels: int = 5
    volume_shape: tuple[int, int, int] = VOLUME_SHAPE

    def __post_init__(self) -> None:
        dimensions = (
            self.members,
            self.future_frames,
            self.channels,
            *self.volume_shape,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("B5 forecast schema dimensions must be positive")
        if len(self.volume_shape) != 3:
            raise ValueError("B5 forecast grid must be three-dimensional")

    @classmethod
    def frozen(cls) -> "B5ForecastSchema":
        return cls()

    @property
    def per_target_shape(self) -> tuple[int, ...]:
        return (
            self.members,
            self.future_frames,
            self.channels,
            *self.volume_shape,
        )

    @property
    def scientific_seed_shape(self) -> tuple[int, int]:
        return (len(B5_FULL_VALIDATION_TARGETS), self.members)

    def to_record(self) -> dict[str, Any]:
        return {
            "members": self.members,
            "future_frames": self.future_frames,
            "channels": self.channels,
            "volume_shape": list(self.volume_shape),
        }


def save_scientific_sampler_seed_bank(path: Path, values: np.ndarray) -> str:
    """Persist the complete uint64 M32 bank without overwrite."""

    destination = Path(path)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    bank = np.asarray(values)
    if bank.shape != (126, 32) or bank.dtype != np.uint64:
        raise ValueError("B5 scientific sampler seed-bank schema differs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    if partial.exists():
        raise FileExistsError(partial)
    with partial.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(bank), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    digest = sha256_path(destination)
    if digest != B5_SCIENTIFIC_BANK_NPY_SHA256:
        raise RuntimeError("B5 persisted scientific sampler bank bytes differ")
    return digest


def load_scientific_sampler_seed_bank(path: Path, expected_sha256: str) -> np.ndarray:
    """Hash-check and load the complete uint64 M32 bank."""

    source = Path(path)
    assert_development_path(source)
    if sha256_path(source) != str(expected_sha256):
        raise ValueError("B5 scientific sampler seed-bank SHA-256 differs")
    if str(expected_sha256) != B5_SCIENTIFIC_BANK_NPY_SHA256:
        raise ValueError("B5 scientific sampler seed-bank identity differs")
    values = np.load(source, allow_pickle=False)
    if values.shape != (126, 32) or values.dtype != np.uint64:
        raise ValueError("B5 scientific sampler seed-bank schema differs")
    expected = scientific_sampler_seed_bank()
    if not np.array_equal(values, expected):
        raise ValueError("B5 scientific sampler seed-bank values differ")
    return np.ascontiguousarray(values, dtype=np.uint64)


def initial_noise_from_uint64(
    seed: int | np.uint64,
    *,
    spatial_shape: Sequence[int] = VOLUME_SHAPE,
) -> np.ndarray:
    """Expand one frozen member seed to one full normalized-residual field."""

    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("B5 sampler spatial shape differs")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    values = generator.standard_normal((5, *shape), dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("B5 initial sampler noise is non-finite")
    return np.ascontiguousarray(values, dtype=np.float32)


def _noise_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()


class B5ForecastWriter(AbstractContextManager["B5ForecastWriter"]):
    """Write one canonical B5 ensemble stream in target order."""

    def __init__(
        self,
        path: Path,
        *,
        target_frames: Sequence[int],
        metadata: Mapping[str, Any],
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: B5ForecastSchema = B5ForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite B5 forecast {self.path}")
        frames = tuple(int(frame) for frame in target_frames)
        if not frames or frames != tuple(range(frames[0], frames[-1] + 1)):
            raise ValueError("B5 forecast targets must be contiguous")
        if frames[0] < 498 or frames[-1] >= 624:
            raise ValueError("B5 forecast targets leave frozen 85604 validation")
        if _metadata_mentions_held_out(dict(metadata)):
            raise ValueError("B5 forecast metadata mentions held-out 85606")
        self.seed_bank_path = Path(seed_bank_path).resolve(strict=True)
        self.seed_bank_sha256 = str(seed_bank_sha256)
        self.seed_bank = load_scientific_sampler_seed_bank(
            self.seed_bank_path, self.seed_bank_sha256
        )
        if schema.members > self.seed_bank.shape[1]:
            raise ValueError("B5 forecast schema exceeds the sampler seed bank")
        self.target_frames = frames
        self.model_seed = 1701
        self.metadata = dict(metadata)
        self.schema = schema
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: h5py.File | None = h5py.File(self.partial_path, "x")
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["development_run"] = "85604"
        self.handle.attrs["held_out_85606_read"] = False
        self.handle.attrs["guard_frames_read"] = False
        self.handle.attrs["horizon_frames"] = 1
        self.handle.attrs["ensemble_size"] = schema.members
        self.handle.attrs["model_seed"] = 1701
        self.handle.attrs["zperiod"] = 5
        self.handle.attrs["mode_mapping"] = "n=5k"
        self.handle.attrs["target_truth_used_as_model_input"] = False
        self.handle.attrs["absolute_time_used_as_model_input"] = False
        self.handle.attrs["member_interaction"] = False
        self.handle.attrs["member_prefixes_regenerated"] = False
        self.handle.attrs["posthoc_calibration"] = False
        self.handle.attrs["initial_noise_device"] = "CPU"
        self.handle.attrs["initial_noise_algorithm"] = "NumPy_PCG64_float32"
        self.handle.attrs["sampler_seed_bank_seed"] = B5_SCIENTIFIC_SAMPLER_BANK_SEED
        self.handle.attrs["sampler_seed_bank_sha256"] = self.seed_bank_sha256
        self.handle.attrs["sampler_steps"] = 18
        self.handle.attrs["network_evaluations_per_member"] = 35
        self.handle.attrs["forecast_axes_json"] = json.dumps(B5_FORECAST_AXES)
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
            raise RuntimeError("B5 forecast writer is already closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("B5 forecast writer received too many targets")
        expected_target = self.target_frames[self.cursor]
        if int(target_frame) != expected_target:
            raise ValueError(
                f"B5 forecast target {target_frame} differs from {expected_target}"
            )
        values = np.asarray(standardized_forecast)
        if values.shape != self.schema.per_target_shape:
            raise ValueError("B5 standardized forecast shape differs")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("B5 standardized forecast must be finite floating point")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("B5 inference time must be finite and nonnegative")
        seeds = np.asarray(sampler_seed_row)
        expected_seeds = self.seed_bank[
            expected_target - B5_FULL_VALIDATION_TARGETS[0], : self.schema.members
        ]
        if (
            seeds.shape != (self.schema.members,)
            or seeds.dtype != np.uint64
            or not np.array_equal(seeds, expected_seeds)
        ):
            raise ValueError("B5 sampler seed row differs")
        digests = tuple(str(value) for value in initial_noise_sha256)
        if len(digests) != self.schema.members or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ValueError("B5 initial-noise hashes differ")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.sampler_seeds[self.cursor] = seeds
        self.initial_noise_sha256[self.cursor] = np.asarray(
            [digest.encode("ascii") for digest in digests], dtype="S64"
        )
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed:
            raise RuntimeError("B5 forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B5 forecast writer did not receive every target")
        if self.handle is None:
            raise RuntimeError("B5 forecast writer handle is closed")
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


class B5ForecastArtifact(AbstractContextManager["B5ForecastArtifact"]):
    """Hash-checked access to a closed B5 ensemble and exact seed bank."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        target_frames: Sequence[int],
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: B5ForecastSchema = B5ForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != str(expected_sha256):
            raise ValueError("B5 forecast artifact SHA-256 differs")
        self.target_frames = tuple(int(frame) for frame in target_frames)
        self.model_seed = 1701
        self.schema = schema
        self.seed_bank_path = Path(seed_bank_path)
        self.seed_bank_sha256 = str(seed_bank_sha256)
        self.seed_bank = load_scientific_sampler_seed_bank(
            self.seed_bank_path, self.seed_bank_sha256
        )
        if schema.members > self.seed_bank.shape[1]:
            raise ValueError("B5 forecast schema exceeds the sampler seed bank")
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("B5 forecast artifact is closed")
        expected_attributes = {
            "schema_version": 1,
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": self.schema.members,
            "model_seed": 1701,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "member_interaction": False,
            "member_prefixes_regenerated": False,
            "posthoc_calibration": False,
            "initial_noise_device": "CPU",
            "initial_noise_algorithm": "NumPy_PCG64_float32",
            "sampler_seed_bank_seed": B5_SCIENTIFIC_SAMPLER_BANK_SEED,
            "sampler_seed_bank_sha256": self.seed_bank_sha256,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "completed": True,
        }
        for name, expected in expected_attributes.items():
            if name not in self.handle.attrs:
                raise ValueError(f"B5 forecast artifact attribute {name} is missing")
            actual = self.handle.attrs[name]
            if isinstance(expected, str):
                actual = _text(actual)
            elif isinstance(expected, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise ValueError(f"B5 forecast artifact attribute {name} differs")
        axes = tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"])))
        if axes != B5_FORECAST_AXES:
            raise ValueError("B5 forecast artifact axes differ")
        stored_schema = json.loads(_text(self.handle.attrs["schema_json"]))
        if stored_schema != self.schema.to_record():
            raise ValueError("B5 forecast artifact schema differs")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("B5 forecast target frames differ")
        forecast = self.handle["standardized_forecast"]
        expected_shape = (len(self.target_frames), *self.schema.per_target_shape)
        if forecast.shape != expected_shape or forecast.dtype != np.dtype("f4"):
            raise ValueError("B5 forecast tensor schema differs")
        times = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        if times.shape != (len(self.target_frames),) or not np.all(
            np.isfinite(times) & (times >= 0.0)
        ):
            raise ValueError("B5 forecast inference-time record differs")
        seeds = np.asarray(self.handle["sampler_seed_uint64"][:], dtype=np.uint64)
        expected_seeds = np.stack(
            [
                self.seed_bank[
                    target - B5_FULL_VALIDATION_TARGETS[0], : self.schema.members
                ]
                for target in self.target_frames
            ],
            axis=0,
        )
        if not np.array_equal(seeds, expected_seeds):
            raise ValueError("B5 stored sampler seeds differ")
        digests = self.handle["initial_noise_sha256"][:]
        if digests.shape != (len(self.target_frames), self.schema.members) or any(
            len(_text(value)) != 64
            or any(character not in "0123456789abcdef" for character in _text(value))
            for value in digests.reshape(-1)
        ):
            raise ValueError("B5 stored initial-noise hashes are malformed")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if _metadata_mentions_held_out(self.metadata):
            raise ValueError("B5 stored metadata mentions held-out 85606")

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("B5 forecast artifact is closed")
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("B5 forecast read interval is invalid")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("B5 forecast artifact contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        if self.handle is None:
            raise RuntimeError("B5 forecast artifact is closed")
        values = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        return {
            "definition": (
                "device_synchronized_EDM_sampling_and_residual_composition_"
                "including_initial_noise_H2D_and_forecast_D2H_excluding_CPU_"
                "noise_generation_and_file_IO"
            ),
            "target_count": int(values.size),
            "ensemble_members_per_target": self.schema.members,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "total_seconds": float(np.sum(values, dtype=np.float64)),
            "mean_seconds_per_target": float(np.mean(values, dtype=np.float64)),
            "median_seconds_per_target": float(np.median(values)),
            "minimum_seconds_per_target": float(np.min(values)),
            "maximum_seconds_per_target": float(np.max(values)),
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


def load_selected_b5_model(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    device: torch.device,
    training_commit: str,
) -> JointFieldResidualEDM:
    """Load the exact selected EMA checkpoint without any reselection."""

    selected = Path(checkpoint)
    assert_development_path(selected)
    if sha256_path(selected) != str(expected_checkpoint_sha256):
        raise ValueError("B5 selected checkpoint SHA-256 differs")
    payload = _load_torch(selected)
    if (
        payload.get("kind") != "B5_selected_EMA_checkpoint"
        or payload.get("paper0_commit") != str(training_commit)
        or payload.get("run_config") != B5EDMFullConfig().to_record()
        or payload.get("model_config") != FieldResidualUNetConfig().to_record()
        or payload.get("residual_scales") != list(B5_RESIDUAL_SCALES)
        or payload.get("selected_completed_epoch") not in range(5, 101, 5)
        or payload.get("physics_metric_used_for_selection") is not False
        or payload.get("scientific_forecast_generated") is not False
        or payload.get("held_out_85606_read") is not False
    ):
        raise ValueError("B5 selected checkpoint provenance differs")
    model = JointFieldResidualEDM(FieldResidualUNet3D()).to(device, torch.float32)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def sample_b5_target_from_seeds(
    *,
    model: JointFieldResidualEDM,
    context: Tensor,
    deterministic_mean: Tensor,
    complete_member_seeds: np.ndarray,
    member_batch_size: int,
) -> tuple[Tensor, tuple[str, ...]]:
    """Sample complete M32 prefixes in fixed member order without interaction."""

    schema = B5ForecastSchema.frozen()
    expected_context = (1, 1, 5, *VOLUME_SHAPE)
    expected_mean = (1, 5, *VOLUME_SHAPE)
    if tuple(context.shape) != expected_context:
        raise ValueError("B5 generation context shape differs")
    if tuple(deterministic_mean.shape) != expected_mean:
        raise ValueError("B5 deterministic mean shape differs")
    seeds = np.asarray(complete_member_seeds)
    if seeds.shape != (schema.members,) or seeds.dtype != np.uint64:
        raise ValueError("B5 complete sampler seed row differs")
    batch_size = int(member_batch_size)
    if batch_size <= 0 or batch_size > schema.members:
        raise ValueError("B5 member batch size is invalid")
    condition = torch.cat((context[:, 0], deterministic_mean), dim=1)
    model.eval()
    forecasts: list[Tensor] = []
    digests: list[str] = []
    for start in range(0, schema.members, batch_size):
        stop = min(start + batch_size, schema.members)
        noise_values = [initial_noise_from_uint64(seed) for seed in seeds[start:stop]]
        digests.extend(_noise_sha256(value) for value in noise_values)
        initial_noise = torch.from_numpy(np.stack(noise_values, axis=0))[None].to(
            device=context.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        normalized = model.sample_normalized(
            condition,
            initial_noise,
            steps=18,
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
        )
        composed = model.compose_fields(deterministic_mean, normalized)[0]
        expected = (stop - start, 1, 5, *VOLUME_SHAPE)
        if tuple(composed.shape) != expected:
            raise RuntimeError("B5 sampled member-batch shape differs")
        forecasts.append(composed.to("cpu", torch.float32))
    result = torch.cat(forecasts, dim=0)
    if tuple(result.shape) != schema.per_target_shape:
        raise RuntimeError("B5 canonical per-target forecast shape differs")
    if not torch.all(torch.isfinite(result)):
        raise FloatingPointError("B5 sampled forecast is non-finite")
    return result, tuple(digests)


def generate_selected_b5_forecasts(
    *,
    model: JointFieldResidualEDM,
    dataset: OneStepContextDataset,
    deterministic_mean_artifact: Any,
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
    """Generate exact M32 fields and close them before target truth opens."""

    schema = B5ForecastSchema.frozen()
    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=1,
    )
    required_targets = (
        tuple(range(498, 502)) if bounded_smoke else B5_FULL_VALIDATION_TARGETS
    )
    if targets != required_targets:
        purpose = "bounded smoke" if bounded_smoke else "scientific"
        raise ValueError(f"{purpose} B5 generation target interval differs")
    if dataset.target_frames != targets or dataset.context_frames != 1:
        raise ValueError("B5 context dataset differs from frozen targets/history")
    if dataset.target_truth_read is not False:
        raise RuntimeError("B5 context dataset does not preserve the truth lock")
    if tuple(deterministic_mean_artifact.target_frames) != targets:
        raise ValueError("B5 deterministic mean targets differ")
    bank = np.asarray(seed_bank)
    if bank.shape != (126, 32) or bank.dtype != np.uint64:
        raise ValueError("B5 complete scientific sampler seed bank differs")
    if not np.array_equal(bank, scientific_sampler_seed_bank()):
        raise ValueError("B5 scientific sampler seed-bank values differ")
    if sha256_path(seed_bank_path) != str(seed_bank_sha256):
        raise ValueError("B5 scientific sampler seed-bank persisted hash differs")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B5 scientific forecast generation requires CUDA")
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    with B5ForecastWriter(
        output,
        target_frames=targets,
        metadata=metadata,
        seed_bank_path=seed_bank_path,
        seed_bank_sha256=seed_bank_sha256,
        schema=schema,
    ) as writer:
        with torch.inference_mode():
            for position, target_frame in enumerate(targets):
                item = dataset[position]
                if int(item["target_frame_index"]) != target_frame:
                    raise RuntimeError("B5 forecast dataset target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError(
                        "B5 forecast context unexpectedly contains truth"
                    )
                context = torch.from_numpy(item["context"])[None].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                mean_array = deterministic_mean_artifact.read(position, position + 1)
                deterministic_mean = torch.from_numpy(mean_array).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                row = bank[target_frame - B5_FULL_VALIDATION_TARGETS[0]]
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                forecast, noise_hashes = sample_b5_target_from_seeds(
                    model=model,
                    context=context,
                    deterministic_mean=deterministic_mean,
                    complete_member_seeds=row,
                    member_batch_size=member_batch_size,
                )
                torch.cuda.synchronize(device)
                writer.append(
                    target_frame=target_frame,
                    standardized_forecast=forecast.numpy(),
                    inference_seconds=time.perf_counter() - started,
                    sampler_seed_row=row,
                    initial_noise_sha256=noise_hashes,
                )
        writer.finalize()
    torch.cuda.synchronize(device)
    output_path = Path(output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B5_residual_EDM_M32_forecast_smoke_85604"
            if bounded_smoke
            else "B5_residual_EDM_one_step_M32_forecast_generation_85604"
        ),
        "bounded_non_scientific_smoke": bool(bounded_smoke),
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "member_interaction": False,
        "member_prefixes_regenerated": False,
        "posthoc_calibration": False,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "forecast": {
            "path": str(output_path),
            "sha256": sha256_path(output_path),
            "axes": list(B5_FORECAST_AXES),
            "shape": [len(targets), *schema.per_target_shape],
            "dtype": "float32",
            "uncompressed_payload_bytes": int(
                len(targets) * np.prod(schema.per_target_shape) * 4
            ),
        },
        "scientific_sampler_seed_bank": {
            "path": str(Path(seed_bank_path).resolve(strict=True)),
            "sha256": str(seed_bank_sha256),
            "seed": B5_SCIENTIFIC_SAMPLER_BANK_SEED,
            "shape": [126, 32],
            "dtype": "uint64",
            "independent_of_checkpoint_selection_noise": True,
            "complete_M32_generated_once": True,
        },
        "inference": {
            "kind": "EDM_probability_flow_ODE_Heun_residual_sampling",
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "member_batch_size": int(member_batch_size),
        },
        "wall_seconds": time.monotonic() - wall_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metadata": dict(metadata),
    }
