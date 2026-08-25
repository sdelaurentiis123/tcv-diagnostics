"""Truth-separated forecast generation for the persistent global--local pilot."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn

from .b5_residual_edm_forecast import load_scientific_sampler_seed_bank
from .codec_training import sha256_path
from .model_data import assert_development_path
from .model_training_data import VOLUME_SHAPE
from .models.persistent_global_local import (
    PGL_FIELD_ORDER,
    PersistentGlobalLocalEDM,
    sample_persistent_global_local_noise,
)
from .persistent_global_local_training import mean_forecast_trajectory


PGL_EVALUATION_BLOCKS = {
    "V00": (497, 501, 504, 508, 511, 515, 518, 522, 525, 529, 532, 536),
    "V01": (537, 541, 544, 548, 552, 555, 559, 562, 566, 570, 573, 577),
    "V02": (578, 582, 585, 589, 593, 597, 600, 604, 608, 612, 615, 619),
}
PGL_EVALUATION_STARTS = tuple(
    frame for block in PGL_EVALUATION_BLOCKS.values() for frame in block
)
PGL_SCIENTIFIC_SEED_BANK_SHA256 = (
    "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
)
PGL_FORECAST_AXES = (
    "start",
    "ensemble_member",
    "future_time",
    "channel",
    "x",
    "y",
    "stored_toroidal_z",
)
PGL_MEAN_AXES = (
    "start",
    "future_time",
    "channel",
    "x",
    "y",
    "stored_toroidal_z",
)


class _ContextDataset(Protocol):
    target_frames: tuple[int, ...]
    context_frames: int
    target_truth_read: bool

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


class _MeanOperator(Protocol):
    def forecast(self, history: Tensor, lead_frames: Tensor) -> Any: ...


@dataclass(frozen=True)
class PGLForecastSchema:
    starts: tuple[int, ...] = PGL_EVALUATION_STARTS
    members: int = 32
    horizon: int = 4
    fields: int = 5
    volume_shape: tuple[int, int, int] = VOLUME_SHAPE

    def __post_init__(self) -> None:
        if not self.starts or tuple(sorted(set(self.starts))) != self.starts:
            raise ValueError("persistent evaluation starts must be increasing")
        if min(self.members, self.horizon, self.fields, *self.volume_shape) <= 0:
            raise ValueError("persistent forecast schema dimensions must be positive")

    @classmethod
    def frozen(cls) -> "PGLForecastSchema":
        return cls()

    @property
    def forecast_shape(self) -> tuple[int, ...]:
        return (
            len(self.starts),
            self.members,
            self.horizon,
            self.fields,
            *self.volume_shape,
        )

    @property
    def mean_shape(self) -> tuple[int, ...]:
        return (len(self.starts), self.horizon, self.fields, *self.volume_shape)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["starts"] = list(self.starts)
        record["volume_shape"] = list(self.volume_shape)
        return record


def evaluation_seed_rows(seed_bank: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    """Select the frozen row associated with each one-frame target ``t+1``."""

    bank = np.asarray(seed_bank)
    if bank.shape != (126, 32) or bank.dtype != np.dtype("u8"):
        raise ValueError("persistent scientific seed bank differs")
    current = tuple(int(value) for value in starts)
    if not current or any(not 497 <= value <= 619 for value in current):
        raise ValueError("persistent forecast start leaves the validation region")
    rows = np.stack([bank[value + 1 - 498] for value in current], axis=0)
    if rows.shape != (len(current), 32):
        raise RuntimeError("persistent scientific seed rows differ")
    return np.ascontiguousarray(rows, dtype=np.uint64)


def initial_noise_from_uint64(
    seed: int | np.uint64,
    *,
    reference: Tensor,
    model: PersistentGlobalLocalEDM,
) -> Tensor:
    """Expand one member seed through the frozen structured innovation law."""

    if reference.ndim != 6 or reference.shape[0] != 1:
        raise ValueError("persistent noise reference must be one trajectory")
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(np.uint64(seed)))
    return sample_persistent_global_local_noise(
        reference,
        config=model.noise_config,
        generator=generator,
    ).total[0]


def tensor_sha256(values: Tensor) -> str:
    array = np.ascontiguousarray(values.detach().float().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


class PGLForecastWriter(AbstractContextManager["PGLForecastWriter"]):
    """Atomically write the canonical forecast without target-truth access."""

    def __init__(
        self,
        path: Path,
        *,
        paper0_commit: str,
        manifest_sha256: str,
        training_result_sha256: str,
        checkpoint_sha256: str,
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: PGLForecastSchema = PGLForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        self.partial_path = self.path.with_name(f".{self.path.name}.partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(self.path)
        if len(str(paper0_commit)) != 40 or any(
            len(str(value)) != 64
            for value in (
                manifest_sha256,
                training_result_sha256,
                checkpoint_sha256,
                seed_bank_sha256,
            )
        ):
            raise ValueError("persistent forecast provenance lock differs")
        if str(seed_bank_sha256) != PGL_SCIENTIFIC_SEED_BANK_SHA256:
            raise ValueError("persistent forecast seed-bank identity differs")
        self.schema = schema
        self.cursor = 0
        self.completed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle: h5py.File | None = h5py.File(self.partial_path, "x")
        attributes = {
            "schema_version": 1,
            "scope": "old_85604_persistent_global_local_M32_four_frame_forecast",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "target_truth_read_during_generation": False,
            "paper0_commit": str(paper0_commit),
            "evaluation_manifest_sha256": str(manifest_sha256),
            "training_result_sha256": str(training_result_sha256),
            "selected_checkpoint_sha256": str(checkpoint_sha256),
            "scientific_seed_bank_path": str(Path(seed_bank_path)),
            "scientific_seed_bank_sha256": str(seed_bank_sha256),
            "ensemble_size": schema.members,
            "future_frames": schema.horizon,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "member_batch_size": 8,
            "posthoc_spread_multiplier": False,
            "physics_diagnostic_used_during_generation": False,
        }
        for name, value in attributes.items():
            self.handle.attrs[name] = value
        self.handle.attrs["forecast_axes_json"] = json.dumps(PGL_FORECAST_AXES)
        self.handle.attrs["mean_axes_json"] = json.dumps(PGL_MEAN_AXES)
        self.handle.attrs["schema_json"] = json.dumps(
            schema.to_record(), sort_keys=True, allow_nan=False
        )
        starts = np.asarray(schema.starts, dtype=np.int64)
        targets = starts[:, None] + np.arange(1, schema.horizon + 1)[None]
        self.handle.create_dataset("current_frame", data=starts, track_times=False)
        self.handle.create_dataset("target_frame", data=targets, track_times=False)
        self.forecast = self.handle.create_dataset(
            "standardized_forecast",
            shape=schema.forecast_shape,
            dtype="f4",
            chunks=(1, 1, 1, 1, *schema.volume_shape),
            shuffle=True,
            fletcher32=True,
            track_times=False,
        )
        self.selected_mean = self.handle.create_dataset(
            "standardized_selected_mean",
            shape=schema.mean_shape,
            dtype="f4",
            chunks=(1, 1, 1, *schema.volume_shape),
            shuffle=True,
            fletcher32=True,
            track_times=False,
        )
        self.parent_mean = self.handle.create_dataset(
            "standardized_parent_mean",
            shape=schema.mean_shape,
            dtype="f4",
            chunks=(1, 1, 1, *schema.volume_shape),
            shuffle=True,
            fletcher32=True,
            track_times=False,
        )
        self.inference_seconds = self.handle.create_dataset(
            "model_inference_seconds", shape=(len(schema.starts),), dtype="f8"
        )
        self.sampler_seeds = self.handle.create_dataset(
            "sampler_seed_uint64",
            shape=(len(schema.starts), schema.members),
            dtype="u8",
        )
        self.initial_noise_sha256 = self.handle.create_dataset(
            "initial_noise_sha256",
            shape=(len(schema.starts), schema.members),
            dtype="S64",
        )

    def append(
        self,
        *,
        current_frame: int,
        standardized_forecast: np.ndarray,
        selected_mean: np.ndarray,
        parent_mean: np.ndarray,
        inference_seconds: float,
        sampler_seed_row: np.ndarray,
        initial_noise_sha256: Sequence[str],
    ) -> None:
        if self.completed or self.handle is None:
            raise RuntimeError("persistent forecast writer is closed")
        if self.cursor >= len(self.schema.starts):
            raise ValueError("persistent forecast writer received too many starts")
        expected = self.schema.starts[self.cursor]
        if int(current_frame) != expected:
            raise ValueError("persistent forecast start order differs")
        forecast = np.asarray(standardized_forecast)
        selected = np.asarray(selected_mean)
        parent = np.asarray(parent_mean)
        expected_forecast = self.schema.forecast_shape[1:]
        expected_mean = self.schema.mean_shape[1:]
        if forecast.shape != expected_forecast or selected.shape != expected_mean:
            raise ValueError("persistent forecast tensor shape differs")
        if parent.shape != expected_mean or not all(
            np.all(np.isfinite(value)) for value in (forecast, selected, parent)
        ):
            raise ValueError("persistent forecast tensors are invalid")
        seeds = np.asarray(sampler_seed_row)
        if seeds.shape != (self.schema.members,) or seeds.dtype != np.dtype("u8"):
            raise ValueError("persistent sampler seed row differs")
        hashes = tuple(str(value) for value in initial_noise_sha256)
        if len(hashes) != self.schema.members or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError("persistent initial-noise hashes differ")
        elapsed = float(inference_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("persistent forecast timing differs")
        self.forecast[self.cursor] = np.asarray(forecast, dtype=np.float32)
        self.selected_mean[self.cursor] = np.asarray(selected, dtype=np.float32)
        self.parent_mean[self.cursor] = np.asarray(parent, dtype=np.float32)
        self.inference_seconds[self.cursor] = elapsed
        self.sampler_seeds[self.cursor] = seeds
        self.initial_noise_sha256[self.cursor] = np.asarray(
            [value.encode("ascii") for value in hashes], dtype="S64"
        )
        self.handle.flush()
        self.cursor += 1

    def finalize(self) -> Path:
        if self.completed or self.handle is None:
            raise RuntimeError("persistent forecast writer was already finalized")
        if self.cursor != len(self.schema.starts):
            raise RuntimeError("persistent forecast writer did not receive every start")
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


class PGLForecastArtifact(AbstractContextManager["PGLForecastArtifact"]):
    """Hash-checked read access to one completed canonical forecast."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        manifest_sha256: str,
        training_result_sha256: str,
        checkpoint_sha256: str,
        seed_bank_path: Path,
        seed_bank_sha256: str,
        schema: PGLForecastSchema = PGLForecastSchema.frozen(),
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if sha256_path(self.path) != str(expected_sha256):
            raise ValueError("persistent forecast SHA-256 differs")
        self.sha256 = str(expected_sha256)
        self.schema = schema
        self.seed_bank = load_scientific_sampler_seed_bank(
            Path(seed_bank_path), str(seed_bank_sha256)
        )
        self.seed_rows = evaluation_seed_rows(
            self.seed_bank, schema.starts
        )[:, : schema.members]
        self.handle: h5py.File | None = h5py.File(self.path, "r")
        self._verify(
            manifest_sha256=str(manifest_sha256),
            training_result_sha256=str(training_result_sha256),
            checkpoint_sha256=str(checkpoint_sha256),
            seed_bank_sha256=str(seed_bank_sha256),
        )

    def _verify(
        self,
        *,
        manifest_sha256: str,
        training_result_sha256: str,
        checkpoint_sha256: str,
        seed_bank_sha256: str,
    ) -> None:
        if self.handle is None:
            raise RuntimeError("persistent forecast artifact is closed")
        expected = {
            "schema_version": 1,
            "scope": "old_85604_persistent_global_local_M32_four_frame_forecast",
            "development_run": "85604",
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "guard_frames_read": False,
            "target_truth_read_during_generation": False,
            "evaluation_manifest_sha256": manifest_sha256,
            "training_result_sha256": training_result_sha256,
            "selected_checkpoint_sha256": checkpoint_sha256,
            "scientific_seed_bank_sha256": seed_bank_sha256,
            "ensemble_size": self.schema.members,
            "future_frames": self.schema.horizon,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "member_batch_size": 8,
            "posthoc_spread_multiplier": False,
            "physics_diagnostic_used_during_generation": False,
            "completed": True,
        }
        for name, value in expected.items():
            if name not in self.handle.attrs:
                raise ValueError(f"persistent forecast attribute {name!r} is absent")
            actual = self.handle.attrs[name]
            if isinstance(value, bool):
                actual = bool(actual)
            elif isinstance(value, int):
                actual = int(actual)
            else:
                actual = actual.decode() if isinstance(actual, bytes) else str(actual)
            if actual != value:
                raise ValueError(f"persistent forecast attribute {name!r} differs")
        if json.loads(str(self.handle.attrs["schema_json"])) != self.schema.to_record():
            raise ValueError("persistent forecast schema differs")
        if tuple(json.loads(str(self.handle.attrs["forecast_axes_json"]))) != PGL_FORECAST_AXES:
            raise ValueError("persistent forecast axes differ")
        if tuple(json.loads(str(self.handle.attrs["mean_axes_json"]))) != PGL_MEAN_AXES:
            raise ValueError("persistent mean axes differ")
        starts = np.asarray(self.handle["current_frame"][:], dtype=np.int64)
        targets = np.asarray(self.handle["target_frame"][:], dtype=np.int64)
        expected_starts = np.asarray(self.schema.starts, dtype=np.int64)
        expected_targets = expected_starts[:, None] + np.arange(1, 5)[None]
        if not np.array_equal(starts, expected_starts) or not np.array_equal(
            targets, expected_targets
        ):
            raise ValueError("persistent forecast coordinates differ")
        if self.handle["standardized_forecast"].shape != self.schema.forecast_shape:
            raise ValueError("persistent forecast dataset shape differs")
        for name in ("standardized_selected_mean", "standardized_parent_mean"):
            if self.handle[name].shape != self.schema.mean_shape:
                raise ValueError("persistent mean dataset shape differs")
        if not np.array_equal(self.handle["sampler_seed_uint64"][:], self.seed_rows):
            raise ValueError("persistent forecast seed rows differ")
        hashes = self.handle["initial_noise_sha256"][:].reshape(-1)
        if any(
            len(value.decode("ascii")) != 64
            for value in hashes
        ):
            raise ValueError("persistent forecast noise hashes differ")
        times = np.asarray(self.handle["model_inference_seconds"][:])
        if not np.all(np.isfinite(times) & (times >= 0.0)):
            raise ValueError("persistent forecast timing differs")

    def read_forecast(self, index: int) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("persistent forecast artifact is closed")
        values = np.asarray(self.handle["standardized_forecast"][int(index)])
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("persistent forecast contains non-finite values")
        return np.ascontiguousarray(values, dtype=np.float32)

    def read_mean(self, index: int, *, parent: bool) -> np.ndarray:
        if self.handle is None:
            raise RuntimeError("persistent forecast artifact is closed")
        name = "standardized_parent_mean" if parent else "standardized_selected_mean"
        values = np.asarray(self.handle[name][int(index)])
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("persistent mean contains non-finite values")
        return np.ascontiguousarray(values, dtype=np.float32)

    def timing_record(self) -> dict[str, Any]:
        if self.handle is None:
            raise RuntimeError("persistent forecast artifact is closed")
        values = np.asarray(self.handle["model_inference_seconds"][:], dtype=np.float64)
        return {
            "start_count": int(values.size),
            "ensemble_members": self.schema.members,
            "future_frames": self.schema.horizon,
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "total_seconds": float(np.sum(values)),
            "mean_seconds_per_start": float(np.mean(values)),
            "median_seconds_per_start": float(np.median(values)),
        }

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        self.close()
        return None


@torch.no_grad()
def generate_pgl_forecast(
    *,
    selected_mean: _MeanOperator,
    parent_mean: _MeanOperator,
    model: PersistentGlobalLocalEDM,
    dataset: _ContextDataset,
    writer: PGLForecastWriter,
    seed_bank: np.ndarray,
    device: torch.device,
    member_batch_size: int = 8,
) -> dict[str, Any]:
    """Generate a complete truth-free M32 four-frame forecast."""

    schema = writer.schema
    if schema != PGLForecastSchema.frozen():
        raise ValueError("scientific persistent forecast schema differs")
    if (
        dataset.target_truth_read is not False
        or dataset.context_frames != 1
        or tuple(dataset.target_frames) != tuple(value + 1 for value in schema.starts)
        or len(dataset) != len(schema.starts)
    ):
        raise ValueError("persistent context-only dataset contract differs")
    if int(member_batch_size) != 8 or schema.members % int(member_batch_size):
        raise ValueError("persistent member batch differs from frozen execution")
    rows = evaluation_seed_rows(seed_bank, schema.starts)
    selected_mean.eval()
    parent_mean.eval()
    model.eval()
    started_wall = time.perf_counter()
    for position, current_frame in enumerate(schema.starts):
        item = dataset[position]
        if (
            int(item["target_frame_index"]) != current_frame + 1
            or tuple(int(value) for value in item["context_frame_indices"])
            != (current_frame,)
            or item.get("target_truth_read") is not False
            or "target" in item
        ):
            raise RuntimeError("persistent forecast context accessed future truth")
        context = torch.from_numpy(np.asarray(item["context"]))[None].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        candidate = mean_forecast_trajectory(selected_mean, context)
        parent = mean_forecast_trajectory(parent_mean, context)
        member_outputs: list[np.ndarray] = []
        noise_hashes: list[str] = []
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for begin in range(0, schema.members, member_batch_size):
            seeds = rows[position, begin : begin + member_batch_size]
            noise_members = [
                initial_noise_from_uint64(seed, reference=candidate, model=model)
                for seed in seeds
            ]
            noise_hashes.extend(tensor_sha256(value) for value in noise_members)
            initial_noise = torch.stack(noise_members, dim=0).unsqueeze(0)
            normalized = model.sample_normalized(
                context[:, -1],
                candidate,
                initial_noise,
                steps=18,
            )
            composed = model.compose_fields(candidate, normalized)[0]
            member_outputs.append(
                np.ascontiguousarray(composed.float().cpu().numpy(), dtype=np.float32)
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        forecast = np.concatenate(member_outputs, axis=0)
        writer.append(
            current_frame=current_frame,
            standardized_forecast=forecast,
            selected_mean=candidate[0].float().cpu().numpy(),
            parent_mean=parent[0].float().cpu().numpy(),
            inference_seconds=elapsed,
            sampler_seed_row=rows[position],
            initial_noise_sha256=noise_hashes,
        )
    writer.finalize()
    return {
        "schema_version": 1,
        "scope": "old_85604_persistent_global_local_truth_free_forecast_generation",
        "development_run": "85604",
        "start_count": len(schema.starts),
        "ensemble_members": schema.members,
        "future_frames": schema.horizon,
        "forecast": {"path": str(writer.path), "sha256": sha256_path(writer.path)},
        "timing": {
            "total_wall_seconds": float(time.perf_counter() - started_wall),
            "sampler_steps": 18,
            "network_evaluations_per_member": 35,
            "member_batch_size": member_batch_size,
        },
        "target_truth_read": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "physics_diagnostics_scored": False,
    }
