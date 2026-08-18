"""Immutable, truth-free ensemble forecasts for the frozen Paper 0 B2 arm.

This module deliberately lives beside, rather than inside, the hash-locked B2
training implementation.  Training selects a checkpoint by denoising loss;
this module reloads that checkpoint, generates one canonical M=32 ensemble
from a CPU-derived noise bank, and closes the artifact before any scorer is
allowed to read target truth.
"""

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

from .b2_training import B2RunConfig, build_b2_model
from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import FAMILY_FIELDS, VOLUME_SHAPE
from .models.latent_diffusion import (
    ConditionedMaskedDenoiser,
    C5PLatentDiffusionModel,
    LatentDiffusionViTConfig,
    build_azula_ab_sampler,
)
from .o2_context_data import OneStepContextDataset
from .o2_training import LatentNormalization, load_frozen_codec
from .o2_training_data import strict_o2_targets


B2_EVALUATION_SEED_TAG = 0x42324556
B2_TRAINING_COMMIT = "46e2ca07e15c7114aace18202b26a9756489a3f0"
B2_FORECAST_AXES = (
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
class B2ForecastSchema:
    """Tensor schema; non-production shapes exist only for bounded unit tests."""

    members: int = 32
    future_frames: int = 1
    channels: int = 5
    volume_shape: tuple[int, int, int] = VOLUME_SHAPE
    latent_channels: int = 32
    trajectory_frames: int = 3
    latent_shape: tuple[int, int, int] = (16, 8, 22)

    def __post_init__(self) -> None:
        counts = (
            self.members,
            self.future_frames,
            self.channels,
            self.latent_channels,
            self.trajectory_frames,
            *self.volume_shape,
            *self.latent_shape,
        )
        if any(int(item) <= 0 for item in counts):
            raise ValueError("B2 forecast schema dimensions must be positive")
        if len(self.volume_shape) != 3 or len(self.latent_shape) != 3:
            raise ValueError("B2 field and latent grids must each be three-dimensional")

    @classmethod
    def frozen(cls) -> "B2ForecastSchema":
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
    def initial_noise_shape(self) -> tuple[int, ...]:
        return (
            self.members,
            self.latent_channels,
            self.trajectory_frames,
            *self.latent_shape,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "members": self.members,
            "future_frames": self.future_frames,
            "channels": self.channels,
            "volume_shape": list(self.volume_shape),
            "latent_channels": self.latent_channels,
            "trajectory_frames": self.trajectory_frames,
            "latent_shape": list(self.latent_shape),
        }


def sampler_seed(model_seed: int, target_frame: int) -> int:
    """Stable uint64 label for the frozen SeedSequence key."""

    if int(model_seed) not in (1701, 1702, 1703):
        raise ValueError("B2 model seed is outside the frozen three-seed matrix")
    if int(target_frame) < 0:
        raise ValueError("target frame must be nonnegative")
    sequence = np.random.SeedSequence(
        [int(model_seed), int(target_frame), B2_EVALUATION_SEED_TAG]
    )
    words = sequence.generate_state(2, dtype=np.uint32)
    return (int(words[0]) << 32) | int(words[1])


def initial_standard_normal(
    *,
    model_seed: int,
    target_frame: int,
    shape: Sequence[int],
) -> tuple[np.ndarray, int, str]:
    """Generate the complete M-member CPU bank once using NumPy PCG64."""

    normalized_shape = tuple(int(item) for item in shape)
    if not normalized_shape or any(item <= 0 for item in normalized_shape):
        raise ValueError("initial-noise shape must contain positive dimensions")
    sequence = np.random.SeedSequence(
        [int(model_seed), int(target_frame), B2_EVALUATION_SEED_TAG]
    )
    generator = np.random.Generator(np.random.PCG64(sequence))
    values = generator.standard_normal(normalized_shape, dtype=np.float32)
    values = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256(memoryview(values)).hexdigest()
    return values, sampler_seed(model_seed, target_frame), digest


def load_selected_b2_model(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    codec_checkpoint: Path,
    expected_codec_sha256: str,
    seed: int,
    device: torch.device,
    training_commit: str = B2_TRAINING_COMMIT,
    expected_selected_epoch: int | None = None,
) -> C5PLatentDiffusionModel:
    """Reload the exact selected full B2 checkpoint and frozen seed codec."""

    selected_path = Path(checkpoint)
    codec_path = Path(codec_checkpoint)
    assert_development_path(selected_path)
    assert_development_path(codec_path)
    actual_sha256 = sha256_path(selected_path)
    if actual_sha256 != str(expected_checkpoint_sha256):
        raise ValueError("selected B2 checkpoint SHA-256 differs")
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    config = B2RunConfig.frozen(mode="full", seed=int(seed))
    model_config = LatentDiffusionViTConfig()
    if payload.get("kind") != "selected_B2_LDM":
        raise ValueError("B2 checkpoint is not a selected LDM")
    if payload.get("paper0_commit") != str(training_commit):
        raise ValueError("B2 checkpoint training commit differs")
    if payload.get("config") != config.to_record():
        raise ValueError("B2 checkpoint frozen run configuration differs")
    if payload.get("model_config") != model_config.to_record():
        raise ValueError("B2 checkpoint model configuration differs")
    if payload.get("physics_derived_loss_used") is not False:
        raise ValueError("B2 checkpoint unexpectedly used a physics-derived loss")
    if payload.get("held_out_85606_read") is not False:
        raise ValueError("B2 checkpoint does not preserve held-out status")
    if int(payload.get("epoch", -1)) not in range(config.epochs):
        raise ValueError("B2 selected epoch is outside the frozen training budget")
    if (
        expected_selected_epoch is not None
        and int(payload["epoch"]) != int(expected_selected_epoch)
    ):
        raise ValueError("B2 selected checkpoint epoch differs from training result")
    if int(payload.get("global_step", -1)) != (
        int(payload["epoch"]) + 1
    ) * config.optimizer_steps_per_epoch:
        raise ValueError("B2 selected checkpoint is not an epoch-end state")

    codec_record = payload.get("codec_checkpoint", {})
    if (
        Path(codec_record.get("path", "")) != codec_path
        or codec_record.get("sha256") != str(expected_codec_sha256)
        or codec_record.get("trainable") is not False
    ):
        raise ValueError("B2 checkpoint codec provenance differs")
    normalization = payload.get("latent_normalization", {})
    if (
        normalization.get("kind")
        != "per_latent_channel_training_only_population_moments"
        or normalization.get("fit_frames") != [0, 432]
        or normalization.get("codec_checkpoint_sha256")
        != str(expected_codec_sha256)
        or normalization.get("scientific_authority") is not True
        or normalization.get("held_out_85606_read") is not False
    ):
        raise ValueError("B2 checkpoint latent normalization differs")
    latent_normalization = LatentNormalization(
        mean=tuple(float(item) for item in normalization["mean"]),
        standard_deviation=tuple(
            float(item) for item in normalization["population_standard_deviation"]
        ),
        sample_count_per_channel=int(normalization["sample_count_per_channel"]),
        fit_frames=tuple(int(item) for item in normalization["fit_frames"]),
        codec_checkpoint_sha256=str(normalization["codec_checkpoint_sha256"]),
        scientific_authority=bool(normalization["scientific_authority"]),
    )
    codec = load_frozen_codec(
        checkpoint=codec_path,
        expected_sha256=str(expected_codec_sha256),
        expected_seed=int(seed),
        device=device,
    )
    model = build_b2_model(
        codec=codec,
        latent_normalization=latent_normalization,
        device=device,
        model_config=model_config,
        sampler_steps=config.sampler_steps,
        sampler_order=config.sampler_order,
    )
    model.denoiser.load_state_dict(payload["denoiser_state"], strict=True)
    model.eval()
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise RuntimeError("loaded B2 codec is unexpectedly trainable")
    return model


class B2ForecastWriter(AbstractContextManager["B2ForecastWriter"]):
    """Write one canonical ensemble stream in target order, without overwrite."""

    def __init__(
        self,
        path: Path,
        *,
        target_frames: Sequence[int],
        model_seed: int,
        metadata: Mapping[str, Any],
        schema: B2ForecastSchema = B2ForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite B2 forecast {self.path}")
        frames = tuple(int(frame) for frame in target_frames)
        if not frames or frames != tuple(range(frames[0], frames[-1] + 1)):
            raise ValueError("B2 forecast targets must be contiguous")
        if frames[0] < 498 or frames[-1] >= 624:
            raise ValueError("B2 forecast targets leave frozen 85604 validation")
        if int(model_seed) not in (1701, 1702, 1703):
            raise ValueError("B2 forecast model seed differs")
        if _metadata_mentions_held_out(dict(metadata)):
            raise ValueError("B2 forecast metadata mentions held-out 85606")
        self.target_frames = frames
        self.model_seed = int(model_seed)
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
        self.handle.attrs["model_seed"] = self.model_seed
        self.handle.attrs["zperiod"] = 5
        self.handle.attrs["mode_mapping"] = "n=5k"
        self.handle.attrs["target_truth_used_as_model_input"] = False
        self.handle.attrs["absolute_time_used_as_model_input"] = False
        self.handle.attrs["member_interaction"] = False
        self.handle.attrs["member_prefixes_regenerated"] = False
        self.handle.attrs["initial_noise_device"] = "CPU"
        self.handle.attrs["initial_noise_algorithm"] = "NumPy_PCG64_float32"
        self.handle.attrs["sampler_seed_key_json"] = json.dumps(
            ["model_seed", "target_frame", B2_EVALUATION_SEED_TAG]
        )
        self.handle.attrs["forecast_axes_json"] = json.dumps(B2_FORECAST_AXES)
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
            "sampler_seed_uint64", shape=(len(frames),), dtype="u8"
        )
        self.initial_noise_sha256 = self.handle.create_dataset(
            "initial_standard_normal_sha256", shape=(len(frames),), dtype="S64"
        )

    def append(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        inference_seconds: float,
        sampler_seed_uint64: int,
        initial_noise_sha256: str,
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("B2 forecast writer is already closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("B2 forecast writer received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(
                f"B2 forecast target {target_frame} differs from {expected}"
            )
        values = np.asarray(standardized_forecast)
        if values.shape != self.schema.per_target_shape:
            raise ValueError("B2 standardized forecast shape differs")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("B2 standardized forecast must be finite floating point")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("B2 inference time must be finite and nonnegative")
        expected_seed = sampler_seed(self.model_seed, expected)
        if int(sampler_seed_uint64) != expected_seed:
            raise ValueError("B2 sampler seed differs from the frozen key")
        digest = str(initial_noise_sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("B2 initial-noise SHA-256 is malformed")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.sampler_seeds[self.cursor] = np.uint64(expected_seed)
        self.initial_noise_sha256[self.cursor] = digest.encode("ascii")
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed:
            raise RuntimeError("B2 forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B2 forecast writer did not receive every target")
        if self.handle is None:
            raise RuntimeError("B2 forecast writer handle is closed")
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


class B2ForecastArtifact(AbstractContextManager["B2ForecastArtifact"]):
    """Hash-checked read access to a closed B2 ensemble artifact."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        target_frames: Sequence[int],
        model_seed: int,
        schema: B2ForecastSchema = B2ForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != str(expected_sha256):
            raise ValueError("B2 forecast artifact SHA-256 differs")
        self.target_frames = tuple(int(frame) for frame in target_frames)
        self.model_seed = int(model_seed)
        self.schema = schema
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("B2 forecast artifact is closed")
        expected_attributes = {
            "schema_version": 1,
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": self.schema.members,
            "model_seed": self.model_seed,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "member_interaction": False,
            "member_prefixes_regenerated": False,
            "initial_noise_device": "CPU",
            "initial_noise_algorithm": "NumPy_PCG64_float32",
            "completed": True,
        }
        for name, expected in expected_attributes.items():
            if name not in self.handle.attrs:
                raise ValueError(f"B2 forecast artifact attribute {name} is missing")
            actual = self.handle.attrs[name]
            if isinstance(expected, str):
                actual = _text(actual)
            elif isinstance(expected, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise ValueError(f"B2 forecast artifact attribute {name} differs")
        axes = tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"])))
        if axes != B2_FORECAST_AXES:
            raise ValueError("B2 forecast artifact axes differ")
        seed_key = json.loads(_text(self.handle.attrs["sampler_seed_key_json"]))
        if seed_key != ["model_seed", "target_frame", B2_EVALUATION_SEED_TAG]:
            raise ValueError("B2 sampler seed key differs")
        stored_schema = json.loads(_text(self.handle.attrs["schema_json"]))
        if stored_schema != self.schema.to_record():
            raise ValueError("B2 forecast artifact schema differs")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("B2 forecast target frames differ")
        forecast = self.handle["standardized_forecast"]
        expected_shape = (len(self.target_frames), *self.schema.per_target_shape)
        if forecast.shape != expected_shape or forecast.dtype != np.dtype("f4"):
            raise ValueError("B2 forecast tensor schema differs")
        times = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        if times.shape != (len(self.target_frames),) or not np.all(
            np.isfinite(times) & (times >= 0.0)
        ):
            raise ValueError("B2 forecast inference-time record differs")
        seeds = np.asarray(self.handle["sampler_seed_uint64"][:], dtype=np.uint64)
        expected_seeds = np.asarray(
            [sampler_seed(self.model_seed, frame) for frame in self.target_frames],
            dtype=np.uint64,
        )
        if not np.array_equal(seeds, expected_seeds):
            raise ValueError("B2 stored sampler seeds differ")
        digests = [
            _text(value)
            for value in self.handle["initial_standard_normal_sha256"][:]
        ]
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        ):
            raise ValueError("B2 stored initial-noise hashes are malformed")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if _metadata_mentions_held_out(self.metadata):
            raise ValueError("B2 stored metadata mentions held-out 85606")

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("B2 forecast artifact is closed")
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("B2 forecast read interval is invalid")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("B2 forecast artifact contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        if self.handle is None:
            raise RuntimeError("B2 forecast artifact is closed")
        values = np.asarray(
            self.handle["model_inference_seconds"][:], dtype=np.float64
        )
        return {
            "definition": (
                "device_synchronized_sampler_and_decode_including_initial_noise_"
                "H2D_and_forecast_D2H_excluding_CPU_noise_generation_and_file_IO"
            ),
            "target_count": int(values.size),
            "ensemble_members_per_target": self.schema.members,
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


def _initial_sample_from_standard_normal(
    *,
    standard_normal: Tensor,
    model: C5PLatentDiffusionModel,
) -> Tensor:
    """Reproduce Azula's N(0, I) prior initialization without GPU RNG."""

    if standard_normal.ndim != 6:
        raise ValueError("B2 sampler noise must be [member,C,time,x,y,z]")
    start = torch.ones((), dtype=standard_normal.dtype, device=standard_normal.device)
    alpha, sigma = model.denoiser.schedule(start)
    # Azula Sampler.init with its default prior mean=0 and variance=1:
    # alpha(start)*mean + sqrt(alpha(start)^2*variance + sigma(start)^2)*z.
    scale = torch.sqrt(alpha.square() + sigma.square())
    return scale * standard_normal


@torch.no_grad()
def sample_b2_target_from_noise(
    *,
    model: C5PLatentDiffusionModel,
    context: Tensor,
    complete_standard_normal: np.ndarray,
    member_batch_size: int,
) -> Tensor:
    """Sample and decode M members with no interaction between members."""

    schema = B2ForecastSchema.frozen()
    if context.shape != (1, 2, len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE):
        raise ValueError("B2 generation context shape differs")
    noise = np.asarray(complete_standard_normal)
    if noise.shape != schema.initial_noise_shape or noise.dtype != np.dtype("f4"):
        raise ValueError("B2 complete initial-noise bank differs")
    if not np.all(np.isfinite(noise)):
        raise ValueError("B2 initial-noise bank contains non-finite values")
    batch_size = int(member_batch_size)
    if batch_size <= 0 or batch_size > schema.members:
        raise ValueError("B2 member batch size is invalid")
    model.eval()
    standardized_context = model._encode_fields(context)
    if tuple(standardized_context.shape) != (1, 32, 2, 16, 8, 22):
        raise RuntimeError("B2 encoded context shape differs")
    target_slot = torch.zeros_like(standardized_context[:, :, :1])
    base_observed = torch.cat((standardized_context, target_slot), dim=2)
    base_mask = torch.zeros(
        (1, 1, 3, 16, 8, 22), dtype=torch.bool, device=context.device
    )
    base_mask[:, :, :2] = True
    decoded_members: list[Tensor] = []
    for start in range(0, schema.members, batch_size):
        stop = min(start + batch_size, schema.members)
        count = stop - start
        observed = base_observed.expand(count, *base_observed.shape[1:]).contiguous()
        mask = base_mask.expand(count, *base_mask.shape[1:]).contiguous()
        conditioned = ConditionedMaskedDenoiser(model.denoiser, observed, mask)
        sampler = build_azula_ab_sampler(
            conditioned,
            steps=model.sampler_steps,
            order=model.sampler_order,
        ).to(context.device)
        normal = torch.from_numpy(noise[start:stop]).to(
            device=context.device, dtype=torch.float32, non_blocking=True
        )
        initial = _initial_sample_from_standard_normal(
            standard_normal=normal,
            model=model,
        )
        sampled = sampler(initial)
        if tuple(sampled.shape) != (count, 32, 3, 16, 8, 22):
            raise RuntimeError("B2 sampled latent trajectory shape differs")
        decoded = model._decode_target(sampled[:, :, -1]).to(torch.float32)
        if tuple(decoded.shape) != (count, 5, *VOLUME_SHAPE):
            raise RuntimeError("B2 decoded member shape differs")
        decoded_members.append(decoded.cpu())
    result = torch.cat(decoded_members, dim=0)[:, None]
    if tuple(result.shape) != schema.per_target_shape:
        raise RuntimeError("B2 canonical per-target forecast shape differs")
    if not torch.all(torch.isfinite(result)):
        raise FloatingPointError("B2 forecast contains non-finite values")
    return result


def generate_selected_b2_forecasts(
    *,
    model: C5PLatentDiffusionModel,
    model_seed: int,
    dataset: OneStepContextDataset,
    target_frames: Sequence[int],
    output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
    member_batch_size: int = 4,
    bounded_smoke: bool = False,
) -> dict[str, Any]:
    """Generate frozen M32 forecasts while preserving the future-truth lock.

    ``bounded_smoke`` executes the exact same sampler on the first four
    validation targets, but labels the artifact non-scientific.  It exists
    solely to exercise the evaluator before the complete 126-target run.
    """

    schema = B2ForecastSchema.frozen()
    targets = strict_o2_targets(
        target_frames, split="validation", context_frames=model.context_frames
    )
    required_targets = (
        tuple(range(498, 502))
        if bounded_smoke
        else tuple(range(498, 624))
    )
    if targets != required_targets:
        purpose = "bounded smoke" if bounded_smoke else "scientific"
        raise ValueError(f"{purpose} B2 generation target interval differs")
    if dataset.target_frames != targets or dataset.context_frames != 2:
        raise ValueError("B2 context dataset differs from frozen targets/history")
    if dataset.target_truth_read is not False:
        raise RuntimeError("B2 context dataset does not preserve the future-truth lock")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scientific B2 forecast generation requires a CUDA worker")
    if int(model_seed) not in (1701, 1702, 1703):
        raise ValueError("B2 generation seed differs")
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    with B2ForecastWriter(
        output,
        target_frames=targets,
        model_seed=model_seed,
        metadata=metadata,
        schema=schema,
    ) as writer:
        with torch.inference_mode():
            for index, target_frame in enumerate(targets):
                item = dataset[index]
                if int(item["target_frame_index"]) != target_frame:
                    raise RuntimeError("B2 forecast dataset target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError(
                        "B2 forecast context unexpectedly contains truth"
                    )
                complete_noise, seed_value, noise_sha256 = initial_standard_normal(
                    model_seed=model_seed,
                    target_frame=target_frame,
                    shape=schema.initial_noise_shape,
                )
                context = torch.from_numpy(item["context"])[None].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                forecast = sample_b2_target_from_noise(
                    model=model,
                    context=context,
                    complete_standard_normal=complete_noise,
                    member_batch_size=member_batch_size,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                writer.append(
                    target_frame=target_frame,
                    standardized_forecast=forecast.numpy(),
                    inference_seconds=elapsed,
                    sampler_seed_uint64=seed_value,
                    initial_noise_sha256=noise_sha256,
                )
        writer.finalize()
    torch.cuda.synchronize(device)
    output_path = Path(output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B2_LDM_H2_M32_forecast_smoke_85604"
            if bounded_smoke
            else "B2_LDM_H2_one_step_M32_forecast_generation_85604"
        ),
        "bounded_non_scientific_smoke": bool(bounded_smoke),
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "member_interaction": False,
        "member_prefixes_regenerated": False,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "forecast": {
            "path": str(output_path),
            "sha256": sha256_path(output_path),
            "axes": list(B2_FORECAST_AXES),
            "shape": [len(targets), *schema.per_target_shape],
            "dtype": "float32",
        },
        "initial_noise": {
            "device": "CPU",
            "generator": "NumPy_PCG64",
            "dtype": "float32",
            "seed_key": ["model_seed", "target_frame", B2_EVALUATION_SEED_TAG],
            "complete_M32_generated_once_per_target": True,
        },
        "sampler": {
            "implementation": "Azula_0.3.1_ABSampler",
            "steps": model.sampler_steps,
            "order": model.sampler_order,
            "member_batch_size": int(member_batch_size),
        },
        "wall_seconds": time.monotonic() - wall_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metadata": dict(metadata),
    }
