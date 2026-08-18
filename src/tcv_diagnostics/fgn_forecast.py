"""Truth-free M32 forecast artifacts for the frozen Paper 0 B3 FGN arm."""

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

from .codec_training import sha256_path
from .fgn_training import (
    FGNRunConfig,
    ParentArtifacts,
    _reload_selected_model,
)
from .model_data import assert_development_path
from .model_training_data import FAMILY_FIELDS, VOLUME_SHAPE
from .models.functional_noise import (
    C5PFunctionalNoiseOneStepModel,
    FunctionalNoiseConfig,
)
from .models.o2 import O2ViTConfig
from .o2_context_data import OneStepContextDataset
from .o2_training_data import strict_o2_targets


FGN_SCIENTIFIC_NOISE_SEED = 31_032
FGN_SCIENTIFIC_TARGET_START = 498
FGN_SCIENTIFIC_TARGET_STOP = 624
FGN_FORECAST_AXES = (
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
class FGNForecastSchema:
    """Canonical B3 tensor schema; small alternatives are for unit tests only."""

    members: int = 32
    future_frames: int = 1
    channels: int = 5
    volume_shape: tuple[int, int, int] = VOLUME_SHAPE
    raw_noise_features: int = 32

    def __post_init__(self) -> None:
        dimensions = (
            self.members,
            self.future_frames,
            self.channels,
            self.raw_noise_features,
            *self.volume_shape,
        )
        if any(int(item) <= 0 for item in dimensions):
            raise ValueError("FGN forecast schema dimensions must be positive")
        if len(self.volume_shape) != 3:
            raise ValueError("FGN field grid must be three-dimensional")

    @classmethod
    def frozen(cls) -> "FGNForecastSchema":
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
    def scientific_noise_shape(self) -> tuple[int, ...]:
        return (
            FGN_SCIENTIFIC_TARGET_STOP - FGN_SCIENTIFIC_TARGET_START,
            self.members,
            self.raw_noise_features,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "members": self.members,
            "future_frames": self.future_frames,
            "channels": self.channels,
            "volume_shape": list(self.volume_shape),
            "raw_noise_features": self.raw_noise_features,
        }


def scientific_noise_bank() -> np.ndarray:
    """Return the independently frozen M32 evaluation bank in target order."""

    schema = FGNForecastSchema.frozen()
    generator = np.random.Generator(np.random.PCG64(FGN_SCIENTIFIC_NOISE_SEED))
    values = generator.standard_normal(
        schema.scientific_noise_shape,
        dtype=np.float32,
    )
    return np.ascontiguousarray(values, dtype=np.float32)


def save_scientific_noise_bank(path: Path, values: np.ndarray) -> str:
    """Persist the complete evaluation bank without overwrite."""

    destination = Path(path)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    expected = FGNForecastSchema.frozen().scientific_noise_shape
    bank = np.asarray(values)
    if bank.shape != expected or bank.dtype != np.dtype("f4"):
        raise ValueError("FGN scientific-noise bank schema differs")
    if not np.all(np.isfinite(bank)):
        raise ValueError("FGN scientific-noise bank contains non-finite values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(bank), allow_pickle=False)
    return sha256_path(destination)


def load_scientific_noise_bank(path: Path, expected_sha256: str) -> np.ndarray:
    """Hash-check and load the full independent evaluation bank."""

    source = Path(path)
    assert_development_path(source)
    if sha256_path(source) != str(expected_sha256):
        raise ValueError("FGN scientific-noise bank SHA-256 differs")
    values = np.load(source, allow_pickle=False)
    expected = FGNForecastSchema.frozen().scientific_noise_shape
    if values.shape != expected or values.dtype != np.dtype("f4"):
        raise ValueError("FGN scientific-noise bank schema differs")
    if not np.all(np.isfinite(values)):
        raise ValueError("FGN scientific-noise bank contains non-finite values")
    return np.ascontiguousarray(values, dtype=np.float32)


def _row_sha256(values: np.ndarray) -> str:
    row = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(memoryview(row)).hexdigest()


def load_selected_fgn_model(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    artifacts: ParentArtifacts,
    device: torch.device,
    training_commit: str,
    expected_selected_epoch: int | None = None,
    model_config: O2ViTConfig = O2ViTConfig(),
    noise_config: FunctionalNoiseConfig = FunctionalNoiseConfig(),
) -> C5PFunctionalNoiseOneStepModel:
    """Audit and reload the exact selected full B3 checkpoint."""

    selected_path = Path(checkpoint)
    assert_development_path(selected_path)
    if sha256_path(selected_path) != str(expected_checkpoint_sha256):
        raise ValueError("selected B3 checkpoint SHA-256 differs")
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    config = FGNRunConfig.frozen(mode="full", seed=1701)
    expected_config = json.loads(json.dumps(config.to_record()))
    observed_config = json.loads(json.dumps(payload.get("config", {})))
    if payload.get("kind") != "selected_B3_FGN_transition":
        raise ValueError("B3 checkpoint is not a selected FGN transition")
    if payload.get("paper0_commit") != str(training_commit):
        raise ValueError("B3 checkpoint training commit differs")
    if observed_config != expected_config:
        raise ValueError("B3 checkpoint frozen run configuration differs")
    if payload.get("model_config") != model_config.to_record():
        raise ValueError("B3 checkpoint deterministic model configuration differs")
    if payload.get("noise_config") != noise_config.to_record():
        raise ValueError("B3 checkpoint functional-noise configuration differs")
    epoch = int(payload.get("epoch", -1))
    if epoch not in range(config.epochs):
        raise ValueError("B3 selected epoch is outside the frozen training budget")
    if expected_selected_epoch is not None and epoch != int(expected_selected_epoch):
        raise ValueError("B3 selected checkpoint epoch differs from training result")
    if int(payload.get("global_step", -1)) != (
        epoch + 1
    ) * config.optimizer_steps_per_epoch:
        raise ValueError("B3 selected checkpoint is not an epoch-end state")

    parent = payload.get("deterministic_parent", {})
    codec = payload.get("codec_checkpoint", {})
    normalization_source = payload.get("latent_normalization_source", {})
    if (
        Path(parent.get("path", "")) != artifacts.checkpoint_path
        or parent.get("sha256") != artifacts.checkpoint_sha256
    ):
        raise ValueError("B3 checkpoint deterministic-parent provenance differs")
    if (
        Path(codec.get("path", "")) != artifacts.codec_path
        or codec.get("sha256") != artifacts.codec_sha256
        or codec.get("trainable") is not False
    ):
        raise ValueError("B3 checkpoint codec provenance differs")
    if (
        Path(normalization_source.get("path", ""))
        != artifacts.latent_normalization_path
        or normalization_source.get("sha256")
        != artifacts.latent_normalization_sha256
        or normalization_source.get("refit") is not False
    ):
        raise ValueError("B3 checkpoint latent-normalization provenance differs")
    validation_noise = payload.get("validation_noise_bank", {})
    if (
        validation_noise.get("seed") != 31_003
        or validation_noise.get("shape") != [126, 2, 32]
    ):
        raise ValueError("B3 checkpoint selection-noise provenance differs")

    model = _reload_selected_model(
        selected_checkpoint=selected_path,
        artifacts=artifacts,
        config=config,
        model_config=model_config,
        noise_config=noise_config,
        device=device,
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise RuntimeError("loaded B3 codec is unexpectedly trainable")
    if model.codec.training:
        raise RuntimeError("loaded B3 codec is not in evaluation mode")
    return model


class FGNForecastWriter(AbstractContextManager["FGNForecastWriter"]):
    """Write one canonical direct FGN ensemble in target order."""

    def __init__(
        self,
        path: Path,
        *,
        target_frames: Sequence[int],
        metadata: Mapping[str, Any],
        noise_bank_path: Path,
        noise_bank_sha256: str,
        schema: FGNForecastSchema = FGNForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite FGN forecast {self.path}")
        frames = tuple(int(frame) for frame in target_frames)
        if not frames or frames != tuple(range(frames[0], frames[-1] + 1)):
            raise ValueError("FGN forecast targets must be contiguous")
        if frames[0] < 498 or frames[-1] >= 624:
            raise ValueError("FGN forecast targets leave frozen 85604 validation")
        if _metadata_mentions_held_out(dict(metadata)):
            raise ValueError("FGN forecast metadata mentions held-out 85606")
        self.noise_bank_path = Path(noise_bank_path).resolve(strict=True)
        assert_development_path(self.noise_bank_path)
        if sha256_path(self.noise_bank_path) != str(noise_bank_sha256):
            raise ValueError("FGN writer scientific-noise bank SHA-256 differs")
        self.noise_bank_sha256 = str(noise_bank_sha256)
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
        self.handle.attrs["raw_noise_device"] = "CPU"
        self.handle.attrs["raw_noise_algorithm"] = "NumPy_PCG64_float32"
        self.handle.attrs["raw_noise_seed"] = FGN_SCIENTIFIC_NOISE_SEED
        self.handle.attrs["raw_noise_bank_sha256"] = self.noise_bank_sha256
        self.handle.attrs["network_evaluations_per_member"] = 1
        self.handle.attrs["forecast_axes_json"] = json.dumps(FGN_FORECAST_AXES)
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
        self.raw_noise_sha256 = self.handle.create_dataset(
            "raw_noise_row_sha256", shape=(len(frames),), dtype="S64"
        )

    def append(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        inference_seconds: float,
        raw_noise_row_sha256: str,
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("FGN forecast writer is already closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("FGN forecast writer received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(f"FGN forecast target {target_frame} differs from {expected}")
        values = np.asarray(standardized_forecast)
        if values.shape != self.schema.per_target_shape:
            raise ValueError("FGN standardized forecast shape differs")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("FGN standardized forecast must be finite floating point")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("FGN inference time must be finite and nonnegative")
        digest = str(raw_noise_row_sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("FGN raw-noise row SHA-256 is malformed")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.raw_noise_sha256[self.cursor] = digest.encode("ascii")
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed:
            raise RuntimeError("FGN forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("FGN forecast writer did not receive every target")
        if self.handle is None:
            raise RuntimeError("FGN forecast writer handle is closed")
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


class FGNForecastArtifact(AbstractContextManager["FGNForecastArtifact"]):
    """Hash-checked access to a closed FGN ensemble and its exact noise bank."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        target_frames: Sequence[int],
        noise_bank_path: Path,
        noise_bank_sha256: str,
        schema: FGNForecastSchema = FGNForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != str(expected_sha256):
            raise ValueError("FGN forecast artifact SHA-256 differs")
        self.target_frames = tuple(int(frame) for frame in target_frames)
        self.model_seed = 1701
        self.schema = schema
        self.noise_bank_path = Path(noise_bank_path)
        self.noise_bank_sha256 = str(noise_bank_sha256)
        self.noise_bank = load_scientific_noise_bank(
            self.noise_bank_path,
            self.noise_bank_sha256,
        )
        if (
            self.schema.members > self.noise_bank.shape[1]
            or self.schema.raw_noise_features > self.noise_bank.shape[2]
        ):
            raise ValueError("FGN forecast schema exceeds the frozen noise bank")
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("FGN forecast artifact is closed")
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
            "raw_noise_device": "CPU",
            "raw_noise_algorithm": "NumPy_PCG64_float32",
            "raw_noise_seed": FGN_SCIENTIFIC_NOISE_SEED,
            "raw_noise_bank_sha256": self.noise_bank_sha256,
            "network_evaluations_per_member": 1,
            "completed": True,
        }
        for name, expected in expected_attributes.items():
            if name not in self.handle.attrs:
                raise ValueError(f"FGN forecast artifact attribute {name} is missing")
            actual = self.handle.attrs[name]
            if isinstance(expected, str):
                actual = _text(actual)
            elif isinstance(expected, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise ValueError(f"FGN forecast artifact attribute {name} differs")
        axes = tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"])))
        if axes != FGN_FORECAST_AXES:
            raise ValueError("FGN forecast artifact axes differ")
        stored_schema = json.loads(_text(self.handle.attrs["schema_json"]))
        if stored_schema != self.schema.to_record():
            raise ValueError("FGN forecast artifact schema differs")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("FGN forecast target frames differ")
        forecast = self.handle["standardized_forecast"]
        expected_shape = (len(self.target_frames), *self.schema.per_target_shape)
        if forecast.shape != expected_shape or forecast.dtype != np.dtype("f4"):
            raise ValueError("FGN forecast tensor schema differs")
        times = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        if times.shape != (len(self.target_frames),) or not np.all(
            np.isfinite(times) & (times >= 0.0)
        ):
            raise ValueError("FGN forecast inference-time record differs")
        digests = [_text(value) for value in self.handle["raw_noise_row_sha256"][:]]
        expected_digests = [
            _row_sha256(
                self.noise_bank[
                    frame - FGN_SCIENTIFIC_TARGET_START,
                    : self.schema.members,
                    : self.schema.raw_noise_features,
                ]
            )
            for frame in self.target_frames
        ]
        if digests != expected_digests:
            raise ValueError("FGN stored raw-noise row hashes differ")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if _metadata_mentions_held_out(self.metadata):
            raise ValueError("FGN stored metadata mentions held-out 85606")

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("FGN forecast artifact is closed")
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("FGN forecast read interval is invalid")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("FGN forecast artifact contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        if self.handle is None:
            raise RuntimeError("FGN forecast artifact is closed")
        values = np.asarray(
            self.handle["model_inference_seconds"][:], dtype=np.float64
        )
        return {
            "definition": (
                "device_synchronized_direct_FGN_transition_and_decode_including_"
                "raw_noise_H2D_and_forecast_D2H_excluding_CPU_noise_generation_"
                "and_file_IO"
            ),
            "target_count": int(values.size),
            "ensemble_members_per_target": self.schema.members,
            "network_evaluations_per_member": 1,
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


@torch.no_grad()
def sample_fgn_target_from_noise(
    *,
    model: C5PFunctionalNoiseOneStepModel,
    context: Tensor,
    complete_raw_noise: np.ndarray,
    member_batch_size: int,
) -> Tensor:
    """Decode M32 direct members without interaction or GPU random draws."""

    schema = FGNForecastSchema.frozen()
    if context.shape != (1, 1, len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE):
        raise ValueError("FGN generation context shape differs")
    noise = np.asarray(complete_raw_noise)
    if noise.shape != (schema.members, schema.raw_noise_features):
        raise ValueError("FGN complete raw-noise row shape differs")
    if noise.dtype != np.dtype("f4") or not np.all(np.isfinite(noise)):
        raise ValueError("FGN raw-noise row must be finite float32")
    batch_size = int(member_batch_size)
    if batch_size <= 0 or batch_size > schema.members:
        raise ValueError("FGN member batch size is invalid")
    model.eval()
    decoded_members: list[Tensor] = []
    for start in range(0, schema.members, batch_size):
        stop = min(start + batch_size, schema.members)
        raw_noise = torch.from_numpy(noise[start:stop])[None].to(
            device=context.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        decoded = model.predict_with_noise(context, raw_noise, horizon=1)[0]
        expected = (stop - start, 1, 5, *VOLUME_SHAPE)
        if tuple(decoded.shape) != expected:
            raise RuntimeError("FGN decoded member batch shape differs")
        decoded_members.append(decoded.to("cpu", torch.float32))
    result = torch.cat(decoded_members, dim=0)
    if tuple(result.shape) != schema.per_target_shape:
        raise RuntimeError("FGN canonical per-target forecast shape differs")
    if not torch.all(torch.isfinite(result)):
        raise FloatingPointError("FGN forecast contains non-finite values")
    return result


def generate_selected_fgn_forecasts(
    *,
    model: C5PFunctionalNoiseOneStepModel,
    dataset: OneStepContextDataset,
    target_frames: Sequence[int],
    noise_bank: np.ndarray,
    noise_bank_path: Path,
    noise_bank_sha256: str,
    output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
    member_batch_size: int = 8,
    bounded_smoke: bool = False,
) -> dict[str, Any]:
    """Generate the exact direct M32 ensemble before any target truth opens."""

    schema = FGNForecastSchema.frozen()
    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=1,
    )
    required_targets = (
        tuple(range(498, 502))
        if bounded_smoke
        else tuple(range(FGN_SCIENTIFIC_TARGET_START, FGN_SCIENTIFIC_TARGET_STOP))
    )
    if targets != required_targets:
        purpose = "bounded smoke" if bounded_smoke else "scientific"
        raise ValueError(f"{purpose} FGN generation target interval differs")
    if dataset.target_frames != targets or dataset.context_frames != 1:
        raise ValueError("FGN context dataset differs from frozen targets/history")
    if dataset.target_truth_read is not False:
        raise RuntimeError("FGN context dataset does not preserve the future-truth lock")
    bank = np.asarray(noise_bank)
    if bank.shape != schema.scientific_noise_shape or bank.dtype != np.dtype("f4"):
        raise ValueError("FGN complete scientific-noise bank differs")
    if not np.all(np.isfinite(bank)):
        raise ValueError("FGN scientific-noise bank contains non-finite values")
    if sha256_path(noise_bank_path) != str(noise_bank_sha256):
        raise ValueError("FGN scientific-noise bank persisted hash differs")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scientific FGN forecast generation requires a CUDA worker")
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    with FGNForecastWriter(
        output,
        target_frames=targets,
        metadata=metadata,
        noise_bank_path=noise_bank_path,
        noise_bank_sha256=noise_bank_sha256,
        schema=schema,
    ) as writer:
        with torch.inference_mode():
            for index, target_frame in enumerate(targets):
                item = dataset[index]
                if int(item["target_frame_index"]) != target_frame:
                    raise RuntimeError("FGN forecast dataset target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError("FGN forecast context unexpectedly contains truth")
                row = bank[target_frame - FGN_SCIENTIFIC_TARGET_START]
                context = torch.from_numpy(item["context"])[None].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                forecast = sample_fgn_target_from_noise(
                    model=model,
                    context=context,
                    complete_raw_noise=row,
                    member_batch_size=member_batch_size,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                writer.append(
                    target_frame=target_frame,
                    standardized_forecast=forecast.numpy(),
                    inference_seconds=elapsed,
                    raw_noise_row_sha256=_row_sha256(row),
                )
        writer.finalize()
    torch.cuda.synchronize(device)
    output_path = Path(output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B3_FGN_H1_M32_forecast_smoke_85604"
            if bounded_smoke
            else "B3_FGN_H1_one_step_M32_forecast_generation_85604"
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
            "axes": list(FGN_FORECAST_AXES),
            "shape": [len(targets), *schema.per_target_shape],
            "dtype": "float32",
        },
        "raw_noise": {
            "path": str(Path(noise_bank_path).resolve(strict=True)),
            "sha256": str(noise_bank_sha256),
            "device": "CPU",
            "generator": "NumPy_PCG64",
            "seed": FGN_SCIENTIFIC_NOISE_SEED,
            "dtype": "float32",
            "shape": list(schema.scientific_noise_shape),
            "independent_of_checkpoint_selection_noise": True,
            "complete_M32_generated_once": True,
        },
        "inference": {
            "kind": "direct_functional_noise_single_pass",
            "network_evaluations_per_member": 1,
            "member_batch_size": int(member_batch_size),
        },
        "wall_seconds": time.monotonic() - wall_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metadata": dict(metadata),
    }
