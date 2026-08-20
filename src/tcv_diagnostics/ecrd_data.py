"""Leakage-safe datasets, parent means, and noise expansion for ECRD."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

import h5py
import numpy as np
import torch

from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import FAMILY_FIELDS, VOLUME_SHAPE
from .models.ecrd import MultiscaleNoiseConfig, symmetrized_h1_mean
from .models.field_residual_edm import B5_FIELD_ORDER, B5_RESIDUAL_SCALES
from .o2_training_data import strict_o2_targets


ECRD_TRAIN_TARGETS = tuple(range(2, 432))
ECRD_VALIDATION_TARGETS = tuple(range(498, 624))
ECRD_PARENT_AXES = (
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


class _ContextDataset(Protocol):
    split: str
    context_frames: int
    target_frames: tuple[int, ...]
    fields: Sequence[str]
    target_truth_read: bool

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        ...


class _WindowDataset(Protocol):
    split: str
    context_frames: int
    target_frames: tuple[int, ...]
    fields: Sequence[str]
    augment: bool

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        ...


class _ParentArtifact(Protocol):
    split: str
    target_frames: tuple[int, ...]

    def read(self, start: int, stop: int) -> np.ndarray:
        ...


class FrozenH1ParentAdapter:
    """Expose an existing immutable H1 forecast through the ECRD parent API.

    B5-Context must retain the historical, unsymmetrized H1 parent.  The old
    training and validation forecast readers predate the explicit ``split``
    attribute used by ECRD, so this adapter supplies only that missing piece;
    it neither copies nor transforms forecast values.
    """

    def __init__(self, artifact: Any, *, split: str) -> None:
        expected = (
            ECRD_TRAIN_TARGETS
            if split == "train"
            else ECRD_VALIDATION_TARGETS
            if split == "validation"
            else ()
        )
        if not expected:
            raise ValueError("frozen H1 parent split differs")
        if tuple(int(value) for value in artifact.target_frames) != expected:
            raise ValueError("frozen H1 parent targets differ")
        if not isinstance(getattr(artifact, "sha256", None), str):
            raise ValueError("frozen H1 parent lacks an immutable hash")
        self.artifact = artifact
        self.split = split
        self.target_frames = expected
        self.sha256 = artifact.sha256

    def read(self, start: int, stop: int) -> np.ndarray:
        return self.artifact.read(start, stop)


class ECRDParentMeanWriter(AbstractContextManager["ECRDParentMeanWriter"]):
    """Write an immutable truth-free four-phase H1 parent stream."""

    def __init__(
        self,
        path: Path,
        *,
        split: str,
        target_frames: Iterable[int],
        metadata: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(self.path)
        context_frames = 1
        frames = strict_o2_targets(
            target_frames, split=split, context_frames=context_frames
        )
        expected = ECRD_TRAIN_TARGETS if split == "train" else ECRD_VALIDATION_TARGETS
        if frames != expected:
            raise ValueError("ECRD parent stream must cover its complete split")
        if _mentions_held_out(dict(metadata)):
            raise ValueError("ECRD parent metadata mentions the held-out run")
        if metadata.get("target_truth_read") is not False:
            raise ValueError("ECRD parent generation must keep target truth closed")
        self.split = split
        self.target_frames = frames
        self.metadata = dict(metadata)
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: h5py.File | None = h5py.File(self.partial_path, "x")
        attributes = {
            "schema_version": 1,
            "scope": "ECRD_four_phase_symmetrized_frozen_H1_parent_mean",
            "development_run": "85604",
            "split": split,
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "target_truth_read": False,
            "training_performed": False,
            "phase_shifts_json": json.dumps([0, 1, 2, 3]),
            "symmetrization": "mean_q0_to_3_T_minus_q_H1_T_q",
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "axes_json": json.dumps(ECRD_PARENT_AXES),
            "metadata_json": json.dumps(
                self.metadata, sort_keys=True, allow_nan=False
            ),
        }
        for name, value in attributes.items():
            self.handle.attrs[name] = value
        self.handle.create_dataset(
            "target_frame_index", data=np.asarray(frames, dtype=np.int64)
        )
        self.parent = self.handle.create_dataset(
            "standardized_parent_mean",
            shape=(len(frames), 5, *VOLUME_SHAPE),
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
        standardized_parent_mean: np.ndarray,
        inference_seconds: float,
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("ECRD parent writer is closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("ECRD parent writer received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError("ECRD parent target order differs")
        values = np.asarray(standardized_parent_mean)
        if values.shape != (5, *VOLUME_SHAPE) or not np.all(np.isfinite(values)):
            raise ValueError("ECRD parent mean shape or values differ")
        elapsed = float(inference_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("ECRD parent inference time differs")
        self.parent[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed or self.handle is None:
            raise RuntimeError("ECRD parent writer is already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("ECRD parent writer did not receive every target")
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


class ECRDParentMeanArtifact(AbstractContextManager["ECRDParentMeanArtifact"]):
    """Hash-checked access to a complete symmetrized H1 parent stream."""

    def __init__(
        self,
        path: Path,
        *,
        split: str,
        expected_sha256: str,
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if sha256_path(self.path) != str(expected_sha256):
            raise ValueError("ECRD parent artifact SHA-256 differs")
        self.sha256 = str(expected_sha256)
        self.split = split
        self.target_frames = (
            ECRD_TRAIN_TARGETS if split == "train" else ECRD_VALIDATION_TARGETS
            if split == "validation" else ()
        )
        if not self.target_frames:
            raise ValueError("ECRD parent artifact split differs")
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("ECRD parent artifact is closed")
        expected = {
            "schema_version": 1,
            "development_run": "85604",
            "split": self.split,
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "target_truth_read": False,
            "training_performed": False,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "completed": True,
        }
        for name, value in expected.items():
            if name not in self.handle.attrs:
                raise ValueError(f"ECRD parent attribute {name!r} is absent")
            actual = self.handle.attrs[name]
            if isinstance(value, str):
                actual = _text(actual)
            elif isinstance(value, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != value:
                raise ValueError(f"ECRD parent attribute {name!r} differs")
        if json.loads(_text(self.handle.attrs["phase_shifts_json"])) != [0, 1, 2, 3]:
            raise ValueError("ECRD parent phase shifts differ")
        if tuple(json.loads(_text(self.handle.attrs["axes_json"]))) != ECRD_PARENT_AXES:
            raise ValueError("ECRD parent axes differ")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("ECRD parent target frames differ")
        if self.handle["standardized_parent_mean"].shape != (
            len(self.target_frames), 5, *VOLUME_SHAPE
        ):
            raise ValueError("ECRD parent tensor shape differs")
        times = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        if not np.all(np.isfinite(times) & (times >= 0.0)):
            raise ValueError("ECRD parent timing record differs")
        metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if _mentions_held_out(metadata):
            raise ValueError("ECRD parent metadata mentions the held-out run")

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("ECRD parent artifact is closed")
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("ECRD parent read interval differs")
        values = np.asarray(
            self.handle["standardized_parent_mean"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("ECRD parent contains non-finite values")
        return values

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        self.close()
        return None


@torch.no_grad()
def generate_symmetrized_h1_parent(
    *,
    model: torch.nn.Module,
    dataset: _ContextDataset,
    output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Generate one complete split without reading any target truth."""

    split = dataset.split
    expected = ECRD_TRAIN_TARGETS if split == "train" else ECRD_VALIDATION_TARGETS
    if (
        tuple(dataset.target_frames) != expected
        or dataset.context_frames not in (1, 2)
        or tuple(dataset.fields) != B5_FIELD_ORDER
        or dataset.target_truth_read is not False
    ):
        raise ValueError("ECRD parent context dataset differs")
    if getattr(model, "context_frames", None) != 1:
        raise ValueError("ECRD parent must use frozen one-frame H1")
    if device.type not in ("cpu", "cuda"):
        raise RuntimeError("ECRD parent generation supports only CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ECRD CUDA parent generation requires an allocated GPU")
    execution_device = str(metadata.get("execution_device", ""))
    if device.type == "cuda" and execution_device != "h100":
        raise RuntimeError("ECRD CUDA parent metadata must identify H100 execution")
    if device.type == "cpu" and (
        execution_device != "cpu-smoke"
        or metadata.get("artifact_authority")
        != "bounded_non_scientific_engineering_smoke_only"
    ):
        raise RuntimeError("ECRD CPU parent lacks bounded-smoke authorization")
    model.eval()
    model.requires_grad_(False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    started_wall = time.monotonic()
    with ECRDParentMeanWriter(
        output,
        split=split,
        target_frames=expected,
        metadata=metadata,
    ) as writer:
        with torch.inference_mode():
            for position, target in enumerate(expected):
                item = dataset[position]
                if int(item["target_frame_index"]) != target:
                    raise RuntimeError("ECRD parent target order differs")
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError("ECRD parent context contains target truth")
                context = torch.from_numpy(np.asarray(item["context"]))[None].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=device.type == "cuda",
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                parent = symmetrized_h1_mean(model, context)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                writer.append(
                    target_frame=target,
                    standardized_parent_mean=parent[0].to("cpu", torch.float32).numpy(),
                    inference_seconds=time.perf_counter() - started,
                )
        writer.finalize()
    output_path = Path(output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": "ECRD_four_phase_symmetrized_frozen_H1_parent_generation",
        "development_run": "85604",
        "split": split,
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_read": False,
        "training_performed": False,
        "target_frames": [expected[0], expected[-1] + 1],
        "target_count": len(expected),
        "phase_shifts": [0, 1, 2, 3],
        "artifact": {
            "path": str(output_path),
            "sha256": sha256_path(output_path),
            "shape": [len(expected), 5, *VOLUME_SHAPE],
            "axes": list(ECRD_PARENT_AXES),
        },
        "wall_seconds": time.monotonic() - started_wall,
        "execution_device": execution_device,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "metadata": dict(metadata),
    }


class ECRDResidualDataset:
    """Join C5P windows to a truth-separated frozen parent mean."""

    def __init__(
        self,
        windows: _WindowDataset,
        parent: _ParentArtifact,
        *,
        split: str,
        history_frames: int,
        augment: bool,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError("ECRD residual split differs")
        expected = strict_o2_targets(
            windows.target_frames,
            split=split,
            context_frames=int(history_frames),
        )
        parent_targets = tuple(int(value) for value in parent.target_frames)
        if (
            expected[0] < parent_targets[0]
            or expected[-1] > parent_targets[-1]
            or parent_targets != tuple(range(parent_targets[0], parent_targets[-1] + 1))
        ):
            raise ValueError("ECRD parent does not cover residual targets")
        if (
            windows.split != split
            or windows.context_frames != int(history_frames)
            or tuple(windows.target_frames) != expected
            or bool(windows.augment) != bool(augment)
            or tuple(windows.fields) != B5_FIELD_ORDER
            or parent.split != split
        ):
            raise ValueError("ECRD residual dataset contract differs")
        if split != "train" and augment:
            raise ValueError("ECRD validation augmentation is forbidden")
        self.windows = windows
        self.parent = parent
        self.split = split
        self.history_frames = int(history_frames)
        self.augment = bool(augment)
        self.target_frames = expected
        self.scales = np.asarray(B5_RESIDUAL_SCALES, dtype=np.float32).reshape(
            5, 1, 1, 1
        )

    def __len__(self) -> int:
        return len(self.target_frames)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.windows, "set_epoch"):
            self.windows.set_epoch(int(epoch))

    def index_for_target(self, target_frame: int) -> int:
        target = int(target_frame)
        position = target - self.target_frames[0]
        if not 0 <= position < len(self) or self.target_frames[position] != target:
            raise IndexError(target)
        return position

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError(position)
        item = self.windows[position]
        target = int(item["target_frame_index"])
        if target != self.target_frames[position]:
            raise RuntimeError("ECRD residual target order differs")
        expected_context = tuple(range(target - self.history_frames, target))
        observed_context = tuple(int(value) for value in item["context_frame_indices"])
        if observed_context != expected_context:
            raise RuntimeError("ECRD residual context indices differ")
        context = np.asarray(item["context"], dtype=np.float32)
        truth = np.asarray(item["target"], dtype=np.float32)
        parent_position = target - int(self.parent.target_frames[0])
        parent = self.parent.read(parent_position, parent_position + 1)[0]
        roll = int(item.get("toroidal_roll", 0))
        if self.augment:
            parent = np.ascontiguousarray(np.roll(parent, roll, axis=-1))
        elif roll != 0:
            raise RuntimeError("ECRD non-augmented example has a nonzero roll")
        if (
            context.shape != (self.history_frames, 5, *VOLUME_SHAPE)
            or truth.shape != (5, *VOLUME_SHAPE)
            or parent.shape != (5, *VOLUME_SHAPE)
        ):
            raise ValueError("ECRD residual example shape differs")
        normalized = (truth - parent) / self.scales
        condition = np.concatenate(
            (context.reshape(5 * self.history_frames, *VOLUME_SHAPE), parent), axis=0
        )
        if not np.all(np.isfinite(normalized)) or not np.all(np.isfinite(condition)):
            raise FloatingPointError("ECRD residual example is non-finite")
        return {
            "target_frame_index": np.int64(target),
            "context_frame_indices": np.asarray(expected_context, dtype=np.int64),
            "condition": np.ascontiguousarray(condition, dtype=np.float32),
            "normalized_parent_residual": np.ascontiguousarray(
                normalized, dtype=np.float32
            ),
            "normalized_residual": np.ascontiguousarray(
                normalized, dtype=np.float32
            ),
            "parent_mean": np.ascontiguousarray(parent, dtype=np.float32),
            "toroidal_roll": np.int64(roll),
            "target_truth_used_as_condition": False,
            "absolute_time_used_as_condition": False,
        }


def multiscale_noise_from_uint64(
    seed: int | np.uint64,
    *,
    spatial_shape: Sequence[int] = VOLUME_SHAPE,
    config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> np.ndarray:
    """Expand one seed into the frozen global/mesoscale/local innovation."""

    n_x, n_y, n_z = (int(value) for value in spatial_shape)
    factor_x, factor_y = config.mesoscale_xy
    if n_x % factor_x or n_y % factor_y:
        raise ValueError("ECRD multiscale noise shape differs")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    global_noise = generator.standard_normal((5, 1, 1, 1), dtype=np.float32)
    meso = generator.standard_normal(
        (5, n_x // factor_x, n_y // factor_y, n_z), dtype=np.float32
    )
    local = generator.standard_normal((5, n_x, n_y, n_z), dtype=np.float32)
    meso = np.repeat(np.repeat(meso, factor_x, axis=-3), factor_y, axis=-2)
    values = (
        config.global_weight * global_noise
        + config.mesoscale_weight * meso
        + config.local_weight * local
    ) / config.normalization
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("ECRD multiscale noise is non-finite")
    return np.ascontiguousarray(values, dtype=np.float32)


def keyed_ecrd_sigma_and_noise(
    *,
    base_seed: int,
    epoch_zero_based: int,
    target_frame: int,
    multiscale: bool,
    spatial_shape: Sequence[int] = VOLUME_SHAPE,
    config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> tuple[np.float32, np.ndarray]:
    """Common-random-number training corruption keyed independently of arm."""

    if not 0 <= int(epoch_zero_based) < 100:
        raise ValueError("ECRD training epoch differs")
    if int(target_frame) not in ECRD_TRAIN_TARGETS:
        raise ValueError("ECRD training target differs")
    sequence = np.random.SeedSequence(
        [int(base_seed), int(epoch_zero_based), int(target_frame), 0xB5ED_0003]
    )
    generator = np.random.Generator(np.random.PCG64(sequence))
    sigma = np.float32(math.exp(-1.2 + 1.2 * float(generator.standard_normal())))
    shape = tuple(int(value) for value in spatial_shape)
    if multiscale:
        n_x, n_y, n_z = shape
        factor_x, factor_y = config.mesoscale_xy
        if n_x % factor_x or n_y % factor_y:
            raise ValueError("ECRD multiscale training shape differs")
        global_noise = generator.standard_normal((5, 1, 1, 1), dtype=np.float32)
        meso = generator.standard_normal(
            (5, n_x // factor_x, n_y // factor_y, n_z), dtype=np.float32
        )
        local = generator.standard_normal((5, n_x, n_y, n_z), dtype=np.float32)
        meso = np.repeat(np.repeat(meso, factor_x, axis=-3), factor_y, axis=-2)
        noise = (
            config.global_weight * global_noise
            + config.mesoscale_weight * meso
            + config.local_weight * local
        ) / config.normalization
    else:
        noise = generator.standard_normal((5, *shape), dtype=np.float32)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.all(np.isfinite(noise)):
        raise FloatingPointError("ECRD keyed corruption is non-finite")
    return sigma, np.ascontiguousarray(noise, dtype=np.float32)


def validation_sigma_and_noise_from_uint64(
    seed: int | np.uint64,
    *,
    multiscale: bool,
    spatial_shape: Sequence[int] = VOLUME_SHAPE,
    config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> tuple[np.float32, np.ndarray]:
    """Fixed validation corruption with paired sigma across all arms."""

    generator = np.random.Generator(np.random.PCG64(int(seed)))
    sigma = np.float32(math.exp(-1.2 + 1.2 * float(generator.standard_normal())))
    shape = tuple(int(value) for value in spatial_shape)
    if multiscale:
        n_x, n_y, n_z = shape
        factor_x, factor_y = config.mesoscale_xy
        if n_x % factor_x or n_y % factor_y:
            raise ValueError("ECRD multiscale validation shape differs")
        global_noise = generator.standard_normal((5, 1, 1, 1), dtype=np.float32)
        meso = generator.standard_normal(
            (5, n_x // factor_x, n_y // factor_y, n_z), dtype=np.float32
        )
        local = generator.standard_normal((5, n_x, n_y, n_z), dtype=np.float32)
        meso = np.repeat(np.repeat(meso, factor_x, axis=-3), factor_y, axis=-2)
        noise = (
            config.global_weight * global_noise
            + config.mesoscale_weight * meso
            + config.local_weight * local
        ) / config.normalization
    else:
        noise = generator.standard_normal((5, *shape), dtype=np.float32)
    return sigma, np.ascontiguousarray(noise, dtype=np.float32)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes(order="C")).hexdigest()
