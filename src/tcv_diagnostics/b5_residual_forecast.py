"""Truth-separated frozen-H1 forecasts for the Paper 0 B5 residual audit."""

from __future__ import annotations

from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch

from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog, VOLUME_SHAPE
from .models.o2 import C5POneStepModel
from .o2_training_data import strict_o2_targets


B5_TRAINING_TARGETS = tuple(range(2, 432))
B5_FORECAST_AXES = (
    "target_frame",
    "channel",
    "x",
    "y",
    "stored_toroidal_z",
)


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


class B5TrainingContextDataset:
    """Training contexts that verify shards but never read a target field."""

    def __init__(
        self,
        catalog: ModelDatasetCatalog,
        *,
        target_frames: Iterable[int] = B5_TRAINING_TARGETS,
        context_frames: int = 1,
    ) -> None:
        self.catalog = catalog
        self.split = "train"
        self.context_frames = int(context_frames)
        self.target_frames = strict_o2_targets(
            target_frames,
            split="train",
            context_frames=self.context_frames,
        )
        self.fields = FAMILY_FIELDS["c5p"]
        self.target_truth_read = False
        self._handles: dict[Path, h5py.File] = {}
        self.consumed_frames = tuple(
            sorted(
                {
                    frame
                    for target in self.target_frames
                    for frame in range(target - self.context_frames, target)
                }
            )
        )
        self.catalog.verify_consumed_frames(self.consumed_frames)

    def __len__(self) -> int:
        return len(self.target_frames)

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _frame(self, frame: int) -> np.ndarray:
        shard, local = self.catalog.locate(frame)
        if shard.path not in self.catalog._verified:
            raise RuntimeError("refusing to read a shard before integrity verification")
        handle = self._handle(shard.path)
        stored = int(handle["coordinates/frame_index"][local])
        if stored != frame:
            raise ValueError(f"stored frame {stored} differs from request {frame}")
        raw = [np.asarray(handle[f"fields/{field}"][local]) for field in self.fields]
        return self.catalog.normalization.encode_volume(self.fields, raw)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = self.target_frames[int(index)]
        context_indices = tuple(range(target - self.context_frames, target))
        if target in context_indices:
            raise RuntimeError("B5 forecast context unexpectedly contains truth")
        context = np.stack([self._frame(frame) for frame in context_indices], axis=0)
        return {
            "context": np.ascontiguousarray(context, dtype=np.float32),
            "context_frame_indices": np.asarray(context_indices, dtype=np.int64),
            "target_frame_index": np.int64(target),
            "target_truth_read": False,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        if hasattr(self, "_handles"):
            self.close()


class B5TrainingForecastWriter(AbstractContextManager["B5TrainingForecastWriter"]):
    """Write one complete ordered training forecast without replacement."""

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
            raise FileExistsError(f"refusing to overwrite B5 forecast {self.path}")
        frames = strict_o2_targets(
            target_frames,
            split="train",
            context_frames=1,
        )
        if _mentions_held_out(dict(metadata)):
            raise ValueError("B5 forecast metadata mentions held-out 85606")
        if metadata.get("target_truth_read") is not False:
            raise ValueError("B5 forecast metadata must lock target truth closed")
        self.target_frames = frames
        self.metadata = dict(metadata)
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial_path, "x")
        attributes = {
            "schema_version": 1,
            "scope": "B5_frozen_H1_training_context_only_forecast",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "validation_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": 1,
            "zperiod": 5,
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "training_performed": False,
        }
        for name, value in attributes.items():
            self.handle.attrs[name] = value
        self.handle.attrs["forecast_axes_json"] = json.dumps(B5_FORECAST_AXES)
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
            raise RuntimeError("B5 forecast writer is already closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("B5 forecast writer received too many frames")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(f"B5 forecast frame {target_frame} differs from {expected}")
        values = np.asarray(standardized_forecast)
        if values.shape != (len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE):
            raise ValueError("B5 standardized forecast shape differs")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
            raise ValueError("B5 standardized forecast must be finite floating point")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("B5 inference time must be finite and nonnegative")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed:
            raise RuntimeError("B5 forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B5 forecast writer did not receive every target")
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


class B5TrainingForecastArtifact(AbstractContextManager["B5TrainingForecastArtifact"]):
    """Hash-checked access to the closed H1 training forecast."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        target_frames: Sequence[int] = B5_TRAINING_TARGETS,
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != expected_sha256:
            raise ValueError("B5 forecast artifact SHA-256 differs")
        self.target_frames = strict_o2_targets(
            target_frames, split="train", context_frames=1
        )
        self.handle = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        expected = {
            "schema_version": 1,
            "scope": "B5_frozen_H1_training_context_only_forecast",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "validation_frames_read": False,
            "horizon_frames": 1,
            "ensemble_size": 1,
            "zperiod": 5,
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "training_performed": False,
            "completed": True,
        }
        for name, value in expected.items():
            if name not in self.handle.attrs:
                raise ValueError(f"B5 forecast attribute {name} is missing")
            actual = self.handle.attrs[name]
            if isinstance(value, str):
                actual = _text(actual)
            elif isinstance(value, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != value:
                raise ValueError(f"B5 forecast attribute {name} differs")
        axes = tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"])))
        if axes != B5_FORECAST_AXES:
            raise ValueError("B5 forecast axes differ")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("B5 forecast target frames differ")
        expected_shape = (
            len(self.target_frames), len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE
        )
        dataset = self.handle["standardized_forecast"]
        if dataset.shape != expected_shape or dataset.dtype != np.dtype("f4"):
            raise ValueError("B5 forecast tensor schema differs")
        times = np.asarray(self.handle["model_inference_seconds"][:])
        if times.shape != (len(self.target_frames),) or not np.all(
            np.isfinite(times) & (times >= 0.0)
        ):
            raise ValueError("B5 forecast inference times differ")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if self.metadata.get("target_truth_read") is not False:
            raise ValueError("B5 forecast metadata truth lock differs")

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("B5 forecast read interval is invalid")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("B5 forecast contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        values = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        return {
            "definition": "device_synchronized_H1_predict_only_excluding_H2D_and_file_IO",
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


def generate_frozen_h1_training_forecast(
    *,
    model: C5POneStepModel,
    dataset: B5TrainingContextDataset,
    output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Generate all 430 H1 forecasts before any target truth is opened."""

    targets = B5_TRAINING_TARGETS
    if dataset.target_frames != targets or dataset.context_frames != 1:
        raise ValueError("B5 forecast dataset differs from frozen targets/history")
    if model.context_frames != 1:
        raise ValueError("B5 deterministic parent must be H1")
    if dataset.target_truth_read is not False:
        raise RuntimeError("B5 forecast dataset does not preserve truth separation")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B5 H1 forecast generation requires a CUDA worker")
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    with B5TrainingForecastWriter(
        output, target_frames=targets, metadata=metadata
    ) as writer:
        with torch.inference_mode():
            for index, target_frame in enumerate(targets):
                item = dataset[index]
                if int(item["target_frame_index"]) != target_frame:
                    raise RuntimeError("B5 forecast target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError("B5 forecast context contains target truth")
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
                    raise RuntimeError("B5 H1 canonical prediction axes differ")
                writer.append(
                    target_frame=target_frame,
                    standardized_forecast=prediction[0, 0, 0].to(
                        "cpu", torch.float32
                    ).numpy(),
                    inference_seconds=elapsed,
                )
        writer.finalize()
    torch.cuda.synchronize(device)
    forecast_path = Path(output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": "B5_frozen_H1_training_context_only_forecast_generation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "validation_frames_read": False,
        "target_truth_read": False,
        "training_performed": False,
        "target_frames": [2, 432],
        "target_count": len(targets),
        "forecast": {
            "path": str(forecast_path),
            "sha256": sha256_path(forecast_path),
            "axes": list(B5_FORECAST_AXES),
            "shape": [len(targets), len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE],
            "dtype": "float32",
        },
        "wall_seconds": time.monotonic() - wall_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metadata": dict(metadata),
    }
