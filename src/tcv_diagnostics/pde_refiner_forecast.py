"""Truth-free scientific forecast artifacts for Paper 0 B4 PDE-Refiner.

The B4 generator is deliberately separate from training.  It reads only the
preceding context frame, expands the prospectively frozen uint64 seed bank on
CPU, and closes and hashes both forecast artifacts before any target truth is
opened by a scorer.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor

from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import FAMILY_FIELDS, VOLUME_SHAPE
from .models.o2 import O2ViTConfig
from .models.pde_refiner import (
    C5PPDERefinerOneStepModel,
    PDERefinerConfig,
    explicit_denoising_update,
)
from .o2_context_data import OneStepContextDataset
from .o2_training_data import strict_o2_targets
from .pde_refiner_full_training import (
    B4_FULL_SEED,
    B4_VALIDATION_BANK_NPY_SHA256,
    PDERefinerFullConfig,
)
from .pde_refiner_training import (
    B4_LATENT_SHAPE,
    RefinerParentArtifacts,
    _reload_selected_model,
)


B4_SCIENTIFIC_SEED_BANK_SEED = 41_032
B4_SCIENTIFIC_TARGET_START = 498
B4_SCIENTIFIC_TARGET_STOP = 624
B4_SCIENTIFIC_SEED_RAW_SHA256 = (
    "f6990201934ae1d2c215458e875b9b7950965a73645a7bd28f5b034121f0a892"
)
B4_SCIENTIFIC_SEED_NPY_SHA256 = (
    "a1871e069bce6244073bfe1aa835a53c1d7a59302b01f6a366b3dc88297b6205"
)
B4_FINAL_FORECAST_AXES = (
    "target_frame",
    "ensemble_member",
    "future_time",
    "channel",
    "x",
    "y",
    "stored_toroidal_z",
)
B4_STAGE_FORECAST_AXES = (
    "target_frame",
    "ensemble_member",
    "refinement_stage",
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


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(array)).hexdigest()


@dataclass(frozen=True)
class PDERefinerForecastSchema:
    """Canonical final and stage tensor schemas; small values support tests."""

    final_members: int = 32
    stage_members: int = 4
    stages: int = 4
    future_frames: int = 1
    channels: int = 5
    volume_shape: tuple[int, int, int] = VOLUME_SHAPE

    def __post_init__(self) -> None:
        values = (
            self.final_members,
            self.stage_members,
            self.stages,
            self.future_frames,
            self.channels,
            *self.volume_shape,
        )
        if any(int(item) <= 0 for item in values):
            raise ValueError("B4 forecast schema dimensions must be positive")
        if self.stage_members > self.final_members:
            raise ValueError("B4 stage prefix exceeds the final ensemble")
        if len(self.volume_shape) != 3:
            raise ValueError("B4 field grid must be three-dimensional")

    @classmethod
    def frozen(cls) -> "PDERefinerForecastSchema":
        return cls()

    @property
    def final_per_target_shape(self) -> tuple[int, ...]:
        return (
            self.final_members,
            self.future_frames,
            self.channels,
            *self.volume_shape,
        )

    @property
    def stage_per_target_shape(self) -> tuple[int, ...]:
        return (
            self.stage_members,
            self.stages,
            self.channels,
            *self.volume_shape,
        )

    @property
    def scientific_seed_shape(self) -> tuple[int, ...]:
        return (
            B4_SCIENTIFIC_TARGET_STOP - B4_SCIENTIFIC_TARGET_START,
            self.final_members,
            3,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "final_members": self.final_members,
            "stage_members": self.stage_members,
            "stages": self.stages,
            "future_frames": self.future_frames,
            "channels": self.channels,
            "volume_shape": list(self.volume_shape),
        }


def scientific_refiner_seed_bank() -> np.ndarray:
    """Materialize the exact independent `[126,32,3]` uint64 seed bank."""

    schema = PDERefinerForecastSchema.frozen()
    generator = np.random.Generator(
        np.random.PCG64(B4_SCIENTIFIC_SEED_BANK_SEED)
    )
    values = generator.integers(
        0,
        np.iinfo(np.uint64).max,
        size=schema.scientific_seed_shape,
        dtype=np.uint64,
    )
    values = np.ascontiguousarray(values, dtype=np.uint64)
    if _array_sha256(values) != B4_SCIENTIFIC_SEED_RAW_SHA256:
        raise RuntimeError("B4 scientific seed-bank raw bytes differ")
    return values


def save_scientific_refiner_seed_bank(path: Path, values: np.ndarray) -> str:
    """Persist the complete bank without overwrite and verify frozen bytes."""

    destination = Path(path)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    bank = np.asarray(values)
    expected = PDERefinerForecastSchema.frozen().scientific_seed_shape
    if bank.shape != expected or bank.dtype != np.dtype("u8"):
        raise ValueError("B4 scientific seed-bank schema differs")
    if _array_sha256(bank) != B4_SCIENTIFIC_SEED_RAW_SHA256:
        raise ValueError("B4 scientific seed-bank raw bytes differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(bank), allow_pickle=False)
    digest = sha256_path(destination)
    if digest != B4_SCIENTIFIC_SEED_NPY_SHA256:
        raise RuntimeError("B4 scientific seed-bank NPY bytes differ")
    return digest


def load_scientific_refiner_seed_bank(
    path: Path,
    expected_sha256: str,
) -> np.ndarray:
    """Hash-check and load the complete frozen seed bank."""

    source = Path(path)
    assert_development_path(source)
    if sha256_path(source) != str(expected_sha256):
        raise ValueError("B4 scientific seed-bank SHA-256 differs")
    if str(expected_sha256) != B4_SCIENTIFIC_SEED_NPY_SHA256:
        raise ValueError("B4 scientific seed-bank is not the frozen NPY artifact")
    values = np.load(source, allow_pickle=False)
    expected = PDERefinerForecastSchema.frozen().scientific_seed_shape
    if values.shape != expected or values.dtype != np.dtype("u8"):
        raise ValueError("B4 scientific seed-bank schema differs")
    values = np.ascontiguousarray(values, dtype=np.uint64)
    if _array_sha256(values) != B4_SCIENTIFIC_SEED_RAW_SHA256:
        raise ValueError("B4 scientific seed-bank raw bytes differ")
    return values


def refinement_noise_from_scientific_seeds(seeds: np.ndarray) -> np.ndarray:
    """Expand one `[member,3]` row into canonical full-latent float32 noise."""

    values = np.asarray(seeds)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not 1 <= values.shape[0] <= 32
        or values.dtype != np.dtype("u8")
    ):
        raise ValueError("B4 scientific seeds must be uint64 [1..32,3]")
    noise = np.empty((values.shape[0], 3, *B4_LATENT_SHAPE), dtype=np.float32)
    for member in range(values.shape[0]):
        for level in range(3):
            generator = np.random.Generator(
                np.random.PCG64(values[member, level])
            )
            noise[member, level] = generator.standard_normal(
                B4_LATENT_SHAPE,
                dtype=np.float32,
            )
    if not np.all(np.isfinite(noise)):
        raise FloatingPointError("B4 expanded scientific noise is non-finite")
    return np.ascontiguousarray(noise, dtype=np.float32)


def load_selected_pde_refiner_model(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    artifacts: RefinerParentArtifacts,
    device: torch.device,
    training_commit: str,
    expected_selected_epoch: int | None = None,
    model_config: O2ViTConfig = O2ViTConfig(),
    refiner_config: PDERefinerConfig = PDERefinerConfig(),
) -> C5PPDERefinerOneStepModel:
    """Audit and reload exactly one selected full B4 EMA checkpoint."""

    selected_path = Path(checkpoint)
    assert_development_path(selected_path)
    if sha256_path(selected_path) != str(expected_checkpoint_sha256):
        raise ValueError("selected B4 checkpoint SHA-256 differs")
    payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    config = PDERefinerFullConfig.frozen(seed=B4_FULL_SEED)
    expected_config = json.loads(json.dumps(config.to_record()))
    observed_config = json.loads(json.dumps(payload.get("config", {})))
    if payload.get("kind") != "selected_B4_PDE_Refiner_transition":
        raise ValueError("B4 checkpoint is not a selected PDE-Refiner transition")
    if payload.get("paper0_commit") != str(training_commit):
        raise ValueError("B4 checkpoint training commit differs")
    if observed_config != expected_config:
        raise ValueError("B4 checkpoint frozen run configuration differs")
    if payload.get("model_config") != model_config.to_record():
        raise ValueError("B4 checkpoint deterministic model configuration differs")
    if payload.get("refiner_config") != refiner_config.to_record():
        raise ValueError("B4 checkpoint refiner configuration differs")
    epoch = int(payload.get("epoch", -1))
    if epoch + 1 not in config.validation_completed_epochs:
        raise ValueError("B4 selected epoch is not a frozen validation candidate")
    if expected_selected_epoch is not None and epoch != int(expected_selected_epoch):
        raise ValueError("B4 selected checkpoint epoch differs from training result")
    if int(payload.get("global_step", -1)) != (
        epoch + 1
    ) * config.optimizer_steps_per_epoch:
        raise ValueError("B4 selected checkpoint is not an epoch-end state")

    parent = payload.get("deterministic_parent", {})
    codec = payload.get("codec_checkpoint", {})
    normalization = payload.get("latent_normalization_source", {})
    validation_bank = payload.get("validation_seed_bank", {})
    if (
        Path(parent.get("path", "")) != artifacts.checkpoint_path
        or parent.get("sha256") != artifacts.checkpoint_sha256
    ):
        raise ValueError("B4 deterministic-parent provenance differs")
    if (
        Path(codec.get("path", "")) != artifacts.codec_path
        or codec.get("sha256") != artifacts.codec_sha256
        or codec.get("trainable") is not False
    ):
        raise ValueError("B4 codec provenance differs")
    if (
        Path(normalization.get("path", "")) != artifacts.latent_normalization_path
        or normalization.get("sha256") != artifacts.latent_normalization_sha256
        or normalization.get("refit") is not False
    ):
        raise ValueError("B4 latent-normalization provenance differs")
    if (
        validation_bank.get("seed") != 41_003
        or validation_bank.get("shape") != [126, 2, 3]
        or validation_bank.get("dtype") != "uint64"
        or validation_bank.get("sha256") != B4_VALIDATION_BANK_NPY_SHA256
    ):
        raise ValueError("B4 checkpoint selection seed-bank provenance differs")

    model = _reload_selected_model(
        selected_checkpoint=selected_path,
        artifacts=artifacts,
        config=config,  # type: ignore[arg-type]
        model_config=model_config,
        refiner_config=refiner_config,
        device=device,
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise RuntimeError("loaded B4 codec is unexpectedly trainable")
    if model.codec.training:
        raise RuntimeError("loaded B4 codec is not in evaluation mode")
    return model


class _PDERefinerForecastWriter(AbstractContextManager["_PDERefinerForecastWriter"]):
    """Shared fail-closed writer for one final or all-stage artifact."""

    def __init__(
        self,
        path: Path,
        *,
        kind: str,
        target_frames: Sequence[int],
        metadata: Mapping[str, Any],
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: PDERefinerForecastSchema,
    ) -> None:
        if kind not in {"final", "stages"}:
            raise ValueError("B4 forecast writer kind differs")
        self.kind = kind
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite B4 forecast {self.path}")
        frames = tuple(int(frame) for frame in target_frames)
        if not frames or frames != tuple(range(frames[0], frames[-1] + 1)):
            raise ValueError("B4 forecast targets must be contiguous")
        if frames[0] < 498 or frames[-1] >= 624:
            raise ValueError("B4 forecast targets leave frozen 85604 validation")
        if _metadata_mentions_held_out(dict(metadata)):
            raise ValueError("B4 forecast metadata mentions held-out 85606")
        self.seed_bank_path = Path(seed_bank_path).resolve(strict=True)
        assert_development_path(self.seed_bank_path)
        if sha256_path(self.seed_bank_path) != str(seed_bank_sha256):
            raise ValueError("B4 writer scientific seed-bank SHA-256 differs")
        if str(seed_bank_sha256) != B4_SCIENTIFIC_SEED_NPY_SHA256:
            raise ValueError("B4 writer seed bank is not the frozen artifact")
        self.seed_bank_sha256 = str(seed_bank_sha256)
        self.target_frames = frames
        self.model_seed = 1701
        self.metadata = dict(metadata)
        self.schema = schema
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: h5py.File | None = h5py.File(self.partial_path, "x")
        self.handle.attrs["schema_version"] = 1
        self.handle.attrs["artifact_kind"] = kind
        self.handle.attrs["development_run"] = "85604"
        self.handle.attrs["held_out_85606_read"] = False
        self.handle.attrs["guard_frames_read"] = False
        self.handle.attrs["horizon_frames"] = 1
        self.handle.attrs["model_seed"] = 1701
        self.handle.attrs["zperiod"] = 5
        self.handle.attrs["mode_mapping"] = "n=5k"
        self.handle.attrs["target_truth_used_as_model_input"] = False
        self.handle.attrs["absolute_time_used_as_model_input"] = False
        self.handle.attrs["member_interaction"] = False
        self.handle.attrs["member_prefixes_regenerated"] = False
        self.handle.attrs["posthoc_calibration"] = False
        self.handle.attrs["seed_bank_seed"] = B4_SCIENTIFIC_SEED_BANK_SEED
        self.handle.attrs["seed_bank_sha256"] = self.seed_bank_sha256
        self.handle.attrs["network_evaluations_per_unamortized_member"] = 4
        axes = B4_FINAL_FORECAST_AXES if kind == "final" else B4_STAGE_FORECAST_AXES
        self.handle.attrs["forecast_axes_json"] = json.dumps(axes)
        self.handle.attrs["schema_json"] = json.dumps(
            schema.to_record(), sort_keys=True, allow_nan=False
        )
        self.handle.attrs["metadata_json"] = json.dumps(
            self.metadata, sort_keys=True, allow_nan=False
        )
        self.handle.create_dataset(
            "target_frame_index", data=np.asarray(frames, dtype=np.int64)
        )
        per_target = (
            schema.final_per_target_shape
            if kind == "final"
            else schema.stage_per_target_shape
        )
        chunks = (1, 1, 1, *per_target[2:])
        self.forecast = self.handle.create_dataset(
            "standardized_forecast",
            shape=(len(frames), *per_target),
            dtype="f4",
            chunks=chunks,
            shuffle=True,
            fletcher32=True,
        )
        self.inference_seconds = self.handle.create_dataset(
            "model_inference_seconds", shape=(len(frames),), dtype="f8"
        )
        self.seed_row_sha256 = self.handle.create_dataset(
            "seed_row_sha256", shape=(len(frames),), dtype="S64"
        )

    def append(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        inference_seconds: float,
        seed_row_sha256: str,
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("B4 forecast writer is already closed")
        if self.cursor >= len(self.target_frames):
            raise ValueError("B4 forecast writer received too many targets")
        expected_frame = self.target_frames[self.cursor]
        if int(target_frame) != expected_frame:
            raise ValueError(
                f"B4 forecast target {target_frame} differs from {expected_frame}"
            )
        expected_shape = (
            self.schema.final_per_target_shape
            if self.kind == "final"
            else self.schema.stage_per_target_shape
        )
        values = np.asarray(standardized_forecast)
        if values.shape != expected_shape:
            raise ValueError("B4 standardized forecast shape differs")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("B4 standardized forecast must be finite floating point")
        elapsed = float(inference_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("B4 inference time must be finite and nonnegative")
        digest = str(seed_row_sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("B4 seed-row SHA-256 is malformed")
        self.forecast[self.cursor] = np.asarray(values, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.seed_row_sha256[self.cursor] = digest.encode("ascii")
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed:
            raise RuntimeError("B4 forecast writer was already finalized")
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B4 forecast writer did not receive every target")
        if self.handle is None:
            raise RuntimeError("B4 forecast writer handle is closed")
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


class PDERefinerFinalForecastWriter(_PDERefinerForecastWriter):
    def __init__(self, path: Path, **kwargs: Any) -> None:
        super().__init__(path, kind="final", **kwargs)


class PDERefinerStageForecastWriter(_PDERefinerForecastWriter):
    def __init__(self, path: Path, **kwargs: Any) -> None:
        super().__init__(path, kind="stages", **kwargs)


class _PDERefinerForecastArtifact(
    AbstractContextManager["_PDERefinerForecastArtifact"]
):
    """Hash-checked access to one closed B4 forecast artifact."""

    def __init__(
        self,
        path: Path,
        *,
        kind: str,
        expected_sha256: str,
        target_frames: Sequence[int],
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: PDERefinerForecastSchema = PDERefinerForecastSchema.frozen(),
    ) -> None:
        if kind not in {"final", "stages"}:
            raise ValueError("B4 forecast artifact kind differs")
        self.kind = kind
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != str(expected_sha256):
            raise ValueError("B4 forecast artifact SHA-256 differs")
        self.target_frames = tuple(int(frame) for frame in target_frames)
        self.model_seed = 1701
        self.schema = schema
        self.seed_bank_path = Path(seed_bank_path)
        self.seed_bank_sha256 = str(seed_bank_sha256)
        self.seed_bank = load_scientific_refiner_seed_bank(
            self.seed_bank_path, self.seed_bank_sha256
        )
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify()

    def _verify(self) -> None:
        if self.handle is None:
            raise RuntimeError("B4 forecast artifact is closed")
        expected_attributes: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": self.kind,
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "horizon_frames": 1,
            "model_seed": 1701,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "target_truth_used_as_model_input": False,
            "absolute_time_used_as_model_input": False,
            "member_interaction": False,
            "member_prefixes_regenerated": False,
            "posthoc_calibration": False,
            "seed_bank_seed": B4_SCIENTIFIC_SEED_BANK_SEED,
            "seed_bank_sha256": self.seed_bank_sha256,
            "network_evaluations_per_unamortized_member": 4,
            "completed": True,
        }
        for name, expected in expected_attributes.items():
            if name not in self.handle.attrs:
                raise ValueError(f"B4 forecast artifact attribute {name} is missing")
            actual = self.handle.attrs[name]
            if isinstance(expected, str):
                actual = _text(actual)
            elif isinstance(expected, bool):
                actual = bool(actual)
            else:
                actual = int(actual)
            if actual != expected:
                raise ValueError(f"B4 forecast artifact attribute {name} differs")
        expected_axes = (
            B4_FINAL_FORECAST_AXES
            if self.kind == "final"
            else B4_STAGE_FORECAST_AXES
        )
        axes = tuple(json.loads(_text(self.handle.attrs["forecast_axes_json"])))
        if axes != expected_axes:
            raise ValueError("B4 forecast artifact axes differ")
        stored_schema = json.loads(_text(self.handle.attrs["schema_json"]))
        if stored_schema != self.schema.to_record():
            raise ValueError("B4 forecast artifact schema differs")
        frames = np.asarray(self.handle["target_frame_index"][:], dtype=np.int64)
        if not np.array_equal(frames, self.target_frames):
            raise ValueError("B4 forecast target frames differ")
        per_target = (
            self.schema.final_per_target_shape
            if self.kind == "final"
            else self.schema.stage_per_target_shape
        )
        forecast = self.handle["standardized_forecast"]
        if forecast.shape != (len(self.target_frames), *per_target):
            raise ValueError("B4 forecast tensor shape differs")
        if forecast.dtype != np.dtype("f4"):
            raise ValueError("B4 forecast tensor dtype differs")
        times = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        if times.shape != (len(self.target_frames),) or not np.all(
            np.isfinite(times) & (times >= 0.0)
        ):
            raise ValueError("B4 forecast inference-time record differs")
        observed = [_text(item) for item in self.handle["seed_row_sha256"][:]]
        expected = [
            _array_sha256(
                self.seed_bank[frame - B4_SCIENTIFIC_TARGET_START]
            )
            for frame in self.target_frames
        ]
        if observed != expected:
            raise ValueError("B4 stored seed-row hashes differ")
        self.metadata = json.loads(_text(self.handle.attrs["metadata_json"]))
        if _metadata_mentions_held_out(self.metadata):
            raise ValueError("B4 stored metadata mentions held-out 85606")

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("B4 forecast artifact is closed")
        if start < 0 or stop > len(self.target_frames) or stop <= start:
            raise ValueError("B4 forecast read interval is invalid")
        values = np.asarray(
            self.handle["standardized_forecast"][start:stop], dtype=np.float32
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("B4 forecast artifact contains non-finite values")
        return values

    def timing_record(self) -> dict[str, Any]:
        if self.handle is None:
            raise RuntimeError("B4 forecast artifact is closed")
        values = np.asarray(
            self.handle["model_inference_seconds"][:], dtype=np.float64
        )
        return {
            "definition": (
                "device_synchronized_B4_transition_and_decode_including_seed_"
                "expanded_noise_H2D_and_forecast_D2H_excluding_CPU_seed_expansion_"
                "and_file_IO"
            ),
            "target_count": int(values.size),
            "ensemble_members_per_target": (
                self.schema.final_members
                if self.kind == "final"
                else self.schema.stage_members
            ),
            "network_evaluations_per_unamortized_member": 4,
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


class PDERefinerFinalForecastArtifact(_PDERefinerForecastArtifact):
    def __init__(self, path: Path, **kwargs: Any) -> None:
        super().__init__(path, kind="final", **kwargs)


class PDERefinerStageForecastArtifact(_PDERefinerForecastArtifact):
    def __init__(self, path: Path, **kwargs: Any) -> None:
        super().__init__(path, kind="stages", **kwargs)


@torch.no_grad()
def sample_pde_refiner_target_from_seeds(
    *,
    model: C5PPDERefinerOneStepModel,
    context: Tensor,
    complete_seed_row: np.ndarray,
    member_batch_size: int,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    """Decode the frozen M32 final ensemble and its exact M4 stage prefix."""

    schema = PDERefinerForecastSchema.frozen()
    if context.shape != (1, 1, len(FAMILY_FIELDS["c5p"]), *VOLUME_SHAPE):
        raise ValueError("B4 generation context shape differs")
    seeds = np.asarray(complete_seed_row)
    if seeds.shape != (schema.final_members, 3) or seeds.dtype != np.dtype("u8"):
        raise ValueError("B4 complete scientific seed row differs")
    batch_size = int(member_batch_size)
    if batch_size <= 0 or batch_size > schema.final_members:
        raise ValueError("B4 member batch size is invalid")

    model.eval()
    standardized_context = model.encode_context(context)
    zero = torch.zeros_like(standardized_context[:, -1])
    initial = standardized_context[:, -1] + model.transition(
        standardized_context, zero, 0
    )
    final_members: list[Tensor] = []
    stage_prefix: list[Tensor] = []
    batched_transition_calls = 1
    for start in range(0, schema.final_members, batch_size):
        stop = min(start + batch_size, schema.final_members)
        noise_cpu = refinement_noise_from_scientific_seeds(seeds[start:stop])
        noise = torch.from_numpy(noise_cpu)[None].to(
            device=context.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        members = stop - start
        expanded_context = standardized_context[:, None].expand(
            1, members, *standardized_context.shape[1:]
        )
        flattened_context = expanded_context.reshape(
            members, *standardized_context.shape[1:]
        )
        current = initial[:, None].expand(1, members, *initial.shape[1:])
        latent_stages = [current]
        for level, sigma in enumerate(
            model.refinement_standard_deviations, start=1
        ):
            noisy = current + sigma * noise[:, :, level - 1]
            predicted = model.transition(
                flattened_context,
                noisy.reshape(members, *noisy.shape[2:]),
                level,
            ).reshape_as(noisy)
            current = explicit_denoising_update(noisy, predicted, sigma)
            latent_stages.append(current)
            batched_transition_calls += 1
        stacked = torch.stack(latent_stages, dim=2)[0]
        flattened = stacked.reshape(
            members * schema.stages, *stacked.shape[2:]
        )
        mean = model.latent_mean[:, 0]
        standard_deviation = model.latent_standard_deviation[:, 0]
        decoded = model.codec.decode(flattened * standard_deviation + mean)
        decoded = decoded.reshape(members, schema.stages, *decoded.shape[1:])
        final_members.append(decoded[:, -1][:, None].to("cpu", torch.float32))
        if start < schema.stage_members:
            prefix_stop = min(stop, schema.stage_members) - start
            stage_prefix.append(decoded[:prefix_stop].to("cpu", torch.float32))

    final = torch.cat(final_members, dim=0)
    stages = torch.cat(stage_prefix, dim=0)
    if tuple(final.shape) != schema.final_per_target_shape:
        raise RuntimeError("B4 canonical final per-target shape differs")
    if tuple(stages.shape) != schema.stage_per_target_shape:
        raise RuntimeError("B4 canonical stage per-target shape differs")
    if not torch.equal(final[: schema.stage_members, 0], stages[:, -1]):
        raise RuntimeError("B4 M4 stage-three prefix differs from M32 final")
    if any(
        not torch.equal(stages[0, 0], stages[member, 0])
        for member in range(1, schema.stage_members)
    ):
        raise RuntimeError("B4 level zero is not shared across members")
    if not torch.all(torch.isfinite(final)) or not torch.all(torch.isfinite(stages)):
        raise FloatingPointError("B4 forecast contains non-finite values")
    return final, stages, {
        "unamortized_member_equivalent_transition_evaluations": 4
        * schema.final_members,
        "shared_level0_member_equivalent_transition_evaluations": 1
        + 3 * schema.final_members,
        "actual_batched_transition_forward_calls": batched_transition_calls,
    }


def generate_selected_pde_refiner_forecasts(
    *,
    model: C5PPDERefinerOneStepModel,
    dataset: OneStepContextDataset,
    target_frames: Sequence[int],
    seed_bank: np.ndarray,
    seed_bank_path: Path,
    seed_bank_sha256: str,
    final_output: Path,
    stage_output: Path,
    metadata: Mapping[str, Any],
    device: torch.device,
    member_batch_size: int = 8,
    bounded_smoke: bool = False,
) -> dict[str, Any]:
    """Generate both B4 artifacts before any target truth can open."""

    schema = PDERefinerForecastSchema.frozen()
    targets = strict_o2_targets(
        target_frames, split="validation", context_frames=1
    )
    required = (
        tuple(range(498, 502))
        if bounded_smoke
        else tuple(
            range(B4_SCIENTIFIC_TARGET_START, B4_SCIENTIFIC_TARGET_STOP)
        )
    )
    if targets != required:
        purpose = "bounded smoke" if bounded_smoke else "scientific"
        raise ValueError(f"{purpose} B4 generation target interval differs")
    if dataset.target_frames != targets or dataset.context_frames != 1:
        raise ValueError("B4 context dataset differs from frozen targets/history")
    if dataset.target_truth_read is not False:
        raise RuntimeError("B4 context dataset does not preserve future-truth lock")
    bank = np.asarray(seed_bank)
    if bank.shape != schema.scientific_seed_shape or bank.dtype != np.dtype("u8"):
        raise ValueError("B4 complete scientific seed bank differs")
    if _array_sha256(bank) != B4_SCIENTIFIC_SEED_RAW_SHA256:
        raise ValueError("B4 complete scientific seed-bank bytes differ")
    if sha256_path(seed_bank_path) != str(seed_bank_sha256):
        raise ValueError("B4 scientific seed-bank persisted hash differs")
    if str(seed_bank_sha256) != B4_SCIENTIFIC_SEED_NPY_SHA256:
        raise ValueError("B4 scientific seed bank is not the frozen NPY artifact")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scientific B4 forecast generation requires a CUDA worker")

    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    accounting: dict[str, int] | None = None
    writer_kwargs = {
        "target_frames": targets,
        "metadata": metadata,
        "seed_bank_path": seed_bank_path,
        "seed_bank_sha256": seed_bank_sha256,
        "schema": schema,
    }
    with PDERefinerFinalForecastWriter(final_output, **writer_kwargs) as final_writer:
        with PDERefinerStageForecastWriter(stage_output, **writer_kwargs) as stage_writer:
            with torch.inference_mode():
                for index, target_frame in enumerate(targets):
                    item = dataset[index]
                    if int(item["target_frame_index"]) != target_frame:
                        raise RuntimeError("B4 forecast dataset target order differs")
                    if item.get("target_truth_read") is not False or "target" in item:
                        raise RuntimeError("B4 forecast context unexpectedly contains truth")
                    row = bank[target_frame - B4_SCIENTIFIC_TARGET_START]
                    context = torch.from_numpy(item["context"])[None].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    final, stages, target_accounting = (
                        sample_pde_refiner_target_from_seeds(
                            model=model,
                            context=context,
                            complete_seed_row=row,
                            member_batch_size=member_batch_size,
                        )
                    )
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    if accounting is None:
                        accounting = target_accounting
                    elif accounting != target_accounting:
                        raise RuntimeError("B4 transition accounting changed by target")
                    digest = _array_sha256(row)
                    final_writer.append(
                        target_frame=target_frame,
                        standardized_forecast=final.numpy(),
                        inference_seconds=elapsed,
                        seed_row_sha256=digest,
                    )
                    stage_writer.append(
                        target_frame=target_frame,
                        standardized_forecast=stages.numpy(),
                        inference_seconds=elapsed,
                        seed_row_sha256=digest,
                    )
            stage_writer.finalize()
            final_writer.finalize()
    if accounting is None:
        raise RuntimeError("B4 forecast generation consumed no targets")
    torch.cuda.synchronize(device)
    final_path = Path(final_output).resolve(strict=True)
    stage_path = Path(stage_output).resolve(strict=True)
    return {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B4_PDE_Refiner_H1_forecast_smoke_85604"
            if bounded_smoke
            else "B4_PDE_Refiner_H1_one_step_forecast_generation_85604"
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
        "final_forecast": {
            "path": str(final_path),
            "sha256": sha256_path(final_path),
            "axes": list(B4_FINAL_FORECAST_AXES),
            "shape": [len(targets), *schema.final_per_target_shape],
            "dtype": "float32",
        },
        "stage_forecast": {
            "path": str(stage_path),
            "sha256": sha256_path(stage_path),
            "axes": list(B4_STAGE_FORECAST_AXES),
            "shape": [len(targets), *schema.stage_per_target_shape],
            "dtype": "float32",
            "M4_stage3_bitwise_prefix_of_M32": True,
            "level0_bitwise_shared_across_members": True,
        },
        "scientific_seed_bank": {
            "path": str(Path(seed_bank_path).resolve(strict=True)),
            "sha256": str(seed_bank_sha256),
            "raw_C_order_sha256": B4_SCIENTIFIC_SEED_RAW_SHA256,
            "generator": "NumPy_Generator_PCG64_uint64",
            "seed": B4_SCIENTIFIC_SEED_BANK_SEED,
            "shape": list(schema.scientific_seed_shape),
            "independent_of_checkpoint_selection_noise": True,
            "complete_M32_generated_once": True,
        },
        "inference": {
            "kind": "three_stage_explicit_latent_PDE_Refiner",
            "member_batch_size": int(member_batch_size),
            **accounting,
        },
        "wall_seconds": time.monotonic() - wall_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "metadata": dict(metadata),
    }
