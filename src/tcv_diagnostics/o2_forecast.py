"""Immutable forecast artifacts and selected-model loading for Paper 0 O2."""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import FAMILY_FIELDS, VOLUME_SHAPE
from .models.o2 import C5POneStepModel, MaskedLatentTransition, O2ViTConfig
from .o2_context_data import OneStepContextDataset
from .o2_training import O2RunConfig, load_frozen_codec
from .o2_training_data import strict_o2_targets


FORECAST_AXES = (
    "target_frame",
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


def load_selected_o2_model(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    codec_checkpoint: Path,
    expected_codec_sha256: str,
    arm: str,
    seed: int,
    training_commit: str,
    device: torch.device,
) -> C5POneStepModel:
    """Load one exact frozen selected transition plus its seed-matched codec."""

    selected_path = Path(checkpoint)
    codec_path = Path(codec_checkpoint)
    assert_development_path(selected_path)
    assert_development_path(codec_path)
    if sha256_path(selected_path) != expected_checkpoint_sha256:
        raise ValueError("selected O2 checkpoint SHA-256 differs")
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    expected_config = O2RunConfig.frozen(mode="full", arm=arm, seed=seed)
    if payload.get("kind") != "selected_O2_transition":
        raise ValueError("O2 checkpoint is not a selected transition")
    if payload.get("paper0_commit") != training_commit:
        raise ValueError("O2 checkpoint training commit differs")
    if payload.get("config") != expected_config.to_record():
        raise ValueError("O2 checkpoint frozen run configuration differs")
    if payload.get("model_config") != O2ViTConfig().to_record():
        raise ValueError("O2 checkpoint model configuration differs")
    codec_record = payload.get("codec_checkpoint", {})
    if (
        Path(codec_record.get("path", "")) != codec_path
        or codec_record.get("sha256") != expected_codec_sha256
        or codec_record.get("trainable") is not False
    ):
        raise ValueError("O2 checkpoint codec provenance differs")
    normalization = payload.get("latent_normalization", {})
    if (
        normalization.get("kind")
        != "per_latent_channel_training_only_population_moments"
        or normalization.get("fit_frames") != [0, 432]
        or normalization.get("codec_checkpoint_sha256") != expected_codec_sha256
        or normalization.get("scientific_authority") is not True
        or normalization.get("held_out_85606_read") is not False
    ):
        raise ValueError("O2 checkpoint latent normalization differs")

    codec = load_frozen_codec(
        checkpoint=codec_path,
        expected_sha256=expected_codec_sha256,
        expected_seed=seed,
        device=device,
    )
    transition = MaskedLatentTransition(
        context_frames=expected_config.context_frames,
        config=O2ViTConfig(),
    ).to(device)
    transition.load_state_dict(payload["transition_state"], strict=True)
    model = C5POneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor(normalization["mean"]),
        latent_standard_deviation=torch.tensor(
            normalization["population_standard_deviation"]
        ),
    ).to(device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise RuntimeError("loaded O2 codec is unexpectedly trainable")
    return model


class O2ForecastWriter(AbstractContextManager["O2ForecastWriter"]):
    """Write one ordered standardized forecast stream without replacement."""

    def __init__(
        self,
        path: Path,
        *,
        target_frames: Sequence[int],
        metadata: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite forecast artifact {self.path}")
        frames = tuple(int(frame) for frame in target_frames)
        if not frames or frames != tuple(range(frames[0], frames[-1] + 1)):
            raise ValueError("forecast target frames must be contiguous")
        if frames[0] < 498 or frames[-1] >= 624:
            raise ValueError("forecast target frames leave frozen 85604 validation")
        if _metadata_mentions_held_out(dict(metadata)):
            raise ValueError("forecast metadata mentions held-out 85606")
        self.target_frames = frames
        self.metadata = dict(metadata)
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial_path, "x")
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["development_run"] = "85604"
        self.handle.attrs["held_out_85606_read"] = False
        self.handle.attrs["guard_frames_read"] = False
        self.handle.attrs["horizon_frames"] = 1
        self.handle.attrs["ensemble_size"] = 1
        self.handle.attrs["zperiod"] = 5
        self.handle.attrs["target_truth_used_as_model_input"] = False
        self.handle.attrs["absolute_time_used_as_model_input"] = False
        self.handle.attrs["forecast_axes_json"] = json.dumps(FORECAST_AXES)
        self.handle.attrs["metadata_json"] = json.dumps(
            self.metadata, sort_keys=True, allow_nan=False
        )
        self.handle.create_dataset(
            "target_frame_index", data=np.asarray(frames, dtype=np.int64)
        )
        self.forecast = self.handle.create_dataset(
            "standardized_forecast",
            shape=(len(frames), len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE),
            dtype="f4",
            chunks=(1, 1, *VOLUME_SHAPE),
            shuffle=True,
            fletcher32=True,
        )
        self.inference_seconds = self.handle.create_dataset(
            "model_inference_seconds", shape=(len(frames),), dtype="f8"
        )

    def append(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        inference_seconds: float,
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("forecast writer is already closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("forecast writer received too many frames")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(f"forecast frame {target_frame} differs from {expected}")
        values = np.asarray(standardized_forecast)
        if values.shape != (len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE):
            raise ValueError("standardized forecast shape differs")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("standardized forecast must be finite floating point")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("inference time must be finite and nonnegative")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed:
            raise RuntimeError("forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("forecast writer did not receive every target")
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


class O2ForecastArtifact(AbstractContextManager["O2ForecastArtifact"]):
    """Hash-checked read access to a completed O2 forecast artifact."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        target_frames: Sequence[int],
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != expected_sha256:
            raise ValueError("forecast artifact SHA-256 differs")
        self.target_frames = tuple(int(frame) for frame in target_frames)
        self.handle = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        expected_attributes = {
            "schema_version": 1,
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": 1,
            "zperiod": 5,
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "completed": True,
        }
        for name, expected in expected_attributes.items():
            if name not in self.handle.attrs:
                raise ValueError(f"forecast artifact attribute {name} is missing")
            actual = self.handle.attrs[name]
            if isinstance(expected, str):
                actual = _text(actual)
            elif isinstance(expected, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise ValueError(f"forecast artifact attribute {name} differs")
        if tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"]))) != FORECAST_AXES:
            raise ValueError("forecast artifact axes differ")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("forecast artifact target frames differ")
        expected_shape = (
            len(self.target_frames),
            len(FAMILY_FIELDS["c5p"]),
            *VOLUME_SHAPE,
        )
        if (
            self.handle["standardized_forecast"].shape != expected_shape
            or self.handle["standardized_forecast"].dtype != np.dtype("f4")
        ):
            raise ValueError("forecast artifact tensor schema differs")
        times = np.asarray(self.handle["model_inference_seconds"][:])
        if times.shape != (len(self.target_frames),) or not np.all(
            np.isfinite(times) & (times >= 0.0)
        ):
            raise ValueError("forecast inference-time record differs")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("forecast artifact read interval is invalid")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("forecast artifact contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        values = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        return {
            "definition": "device_synchronized_model_predict_only_excluding_H2D_and_file_IO",
            "target_count": int(values.size),
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


def generate_selected_o2_forecasts(
    *,
    model: C5POneStepModel,
    dataset: OneStepContextDataset,
    target_frames: Sequence[int],
    output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Generate ordered one-step predictions without ever moving target truth."""

    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=model.context_frames,
    )
    if dataset.target_frames != targets or dataset.context_frames != model.context_frames:
        raise ValueError("forecast dataset differs from model history and targets")
    if dataset.target_truth_read is not False:
        raise RuntimeError("forecast dataset does not preserve the future-truth lock")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("selected O2 forecast generation requires a CUDA worker")
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    with O2ForecastWriter(
        output,
        target_frames=targets,
        metadata=metadata,
    ) as writer:
        with torch.inference_mode():
            for index, target_frame in enumerate(targets):
                item = dataset[index]
                if int(item["target_frame_index"]) != target_frame:
                    raise RuntimeError("forecast dataset target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError("forecast context unexpectedly contains target truth")
                context = torch.from_numpy(item["context"])[None].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                prediction = model.predict(context, horizon=1, ensemble_size=1)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                expected_shape = (1, 1, 1, len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE)
                if tuple(prediction.shape) != expected_shape:
                    raise RuntimeError("canonical O2 prediction axes differ")
                writer.append(
                    target_frame=target_frame,
                    standardized_forecast=prediction[0, 0, 0].to(
                        "cpu", torch.float32
                    ).numpy(),
                    inference_seconds=elapsed,
                )
        writer.finalize()
    torch.cuda.synchronize(device)
    return {
        "schema_version": 1,
        "scope": "O2_selected_checkpoint_forecast_generation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "forecast": {
            "path": str(Path(output).resolve(strict=True)),
            "sha256": sha256_path(Path(output)),
            "axes": list(FORECAST_AXES),
            "shape": [len(targets), len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE],
            "dtype": "float32",
        },
        "wall_seconds": time.monotonic() - wall_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metadata": dict(metadata),
    }
