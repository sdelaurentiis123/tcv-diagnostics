"""Frozen full-training mechanics for the Paper 0 B5 residual EDM.

This module implements only the prospectively fixed seed-1701 development
training run and its data-only checkpoint selection.  It does not generate or
score the scientific M32 forecast, evaluate physics diagnostics, access the
held-out run, or authorize downstream work.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from .b5_residual_edm_training import module_state_sha256, parameter_count
from .codec_training import save_torch_atomic, sha256_path
from .model_data import assert_development_path, write_strict_json_atomic
from .model_training_data import VOLUME_SHAPE
from .models.field_residual_edm import (
    B5_FIELD_ORDER,
    B5_RESIDUAL_SCALES,
    B5_SPATIAL_SHAPE,
    FieldResidualUNet3D,
    FieldResidualUNetConfig,
    JointFieldResidualEDM,
)


B5_FULL_SEED = 1701
B5_FULL_EPOCHS = 100
B5_FULL_TRAIN_TARGETS = tuple(range(2, 432))
B5_FULL_VALIDATION_TARGETS = tuple(range(498, 624))
B5_FULL_TRAINING_ORDER_SEED = 67_501
B5_FULL_TRAINING_NOISE_SEED = 67_502
B5_FULL_VALIDATION_BANK_SEED = 67_503
B5_FULL_RELOAD_PROBE_SEED = 67_504
B5_SCIENTIFIC_SAMPLER_BANK_SEED = 67_532
B5_FULL_ORDER_RAW_SHA256 = (
    "4eb79c67e03623ccb5e0b1735ff0d3a13c1202db833d0c42509af3ba7b0eafda"
)
B5_FULL_ORDER_NPY_SHA256 = (
    "0e775e59e3596e63c2324a9a9fa5ff82df9dca1ff9d4923fbbef3a4126e97806"
)
B5_VALIDATION_BANK_RAW_SHA256 = (
    "f0e736a16be18289ef64fc190fac917eda284eac13ed5117fa7be2d7c2b7d411"
)
B5_VALIDATION_BANK_NPY_SHA256 = (
    "fca7f1254b28fda0a1dad91aea4e1e8ce2faef5dbe7484e13c86ee885e5a5e12"
)
B5_SCIENTIFIC_BANK_RAW_SHA256 = (
    "dcd4eb49682e5783508e423951108a4b47afeb103e1c3d9fcdcb1bae88b8ec19"
)
B5_SCIENTIFIC_BANK_NPY_SHA256 = (
    "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
)


class _WindowDataset(Protocol):
    split: str
    context_frames: int
    target_frames: tuple[int, ...]
    augment: bool
    fields: Sequence[str]

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        ...


class _ForecastArtifact(Protocol):
    target_frames: tuple[int, ...]
    sha256: str

    def read(self, start: int, stop: int) -> np.ndarray:
        ...


@dataclass(frozen=True)
class B5EDMFullConfig:
    """Exact full B5 training and validation-selection configuration."""

    seed: int = B5_FULL_SEED
    epochs: int = B5_FULL_EPOCHS
    microbatch_targets: int = 1
    gradient_accumulation_targets: int = 4
    peak_learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-6
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    training_precision: str = "bfloat16_autocast_with_FP32_loss_optimizer_and_EMA"
    validation_precision: str = "float32_no_autocast_TF32_disabled"
    training_order_seed: int = B5_FULL_TRAINING_ORDER_SEED
    training_noise_seed: int = B5_FULL_TRAINING_NOISE_SEED
    validation_seed_bank_seed: int = B5_FULL_VALIDATION_BANK_SEED
    validation_probe_draws_per_target: int = 4
    validation_every_completed_epochs: int = 5

    def __post_init__(self) -> None:
        if self.seed != B5_FULL_SEED or self.epochs != B5_FULL_EPOCHS:
            raise ValueError("full B5 identity is fixed to seed 1701 and 100 epochs")
        if self.microbatch_targets != 1 or self.gradient_accumulation_targets != 4:
            raise ValueError("full B5 microbatch/accumulation contract differs")
        if self.training_precision != (
            "bfloat16_autocast_with_FP32_loss_optimizer_and_EMA"
        ):
            raise ValueError("full B5 training precision differs")
        if self.validation_precision != "float32_no_autocast_TF32_disabled":
            raise ValueError("full B5 validation precision differs")
        if self.validation_probe_draws_per_target != 4:
            raise ValueError("full B5 validation probe count differs")
        if self.validation_every_completed_epochs != 5:
            raise ValueError("full B5 validation cadence differs")

    @property
    def train_targets(self) -> tuple[int, ...]:
        return B5_FULL_TRAIN_TARGETS

    @property
    def validation_targets(self) -> tuple[int, ...]:
        return B5_FULL_VALIDATION_TARGETS

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_targets) / self.gradient_accumulation_targets)

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    @property
    def target_presentations(self) -> int:
        return self.epochs * len(self.train_targets)

    @property
    def validation_completed_epochs(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.validation_every_completed_epochs,
                self.epochs + 1,
                self.validation_every_completed_epochs,
            )
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["betas"] = list(self.betas)
        record.update(
            {
                "mode": "full",
                "arm": "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI",
                "train_targets": [2, 432],
                "validation_targets": [498, 624],
                "train_target_count": len(self.train_targets),
                "validation_target_count": len(self.validation_targets),
                "target_presentations": self.target_presentations,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "validation_completed_epochs": list(self.validation_completed_epochs),
                "checkpoint_selection": (
                    "earliest_lowest_fixed_seed_validation_EDM_loss_EMA_after_"
                    "complete_100_epoch_budget"
                ),
                "early_stopping": False,
                "toroidal_roll_augmentation": False,
                "physics_derived_loss_allowed": False,
                "absolute_time_input_allowed": False,
                "scientific_forecast_generated": False,
                "held_out_85606_access_allowed": False,
            }
        )
        return record


def _array_npy_sha256(values: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def full_training_order(
    config: B5EDMFullConfig = B5EDMFullConfig(),
) -> np.ndarray:
    """Materialize and byte-verify all 100 target permutations."""

    if config != B5EDMFullConfig():
        raise ValueError("B5 order requires the exact frozen full config")
    generator = np.random.Generator(np.random.PCG64(config.training_order_seed))
    values = np.ascontiguousarray(
        np.stack(
            [
                generator.permutation(np.asarray(config.train_targets, dtype=np.int64))
                for _ in range(config.epochs)
            ]
        ),
        dtype=np.int64,
    )
    if values.shape != (100, 430):
        raise RuntimeError("B5 full target-order shape differs")
    if any(set(map(int, row)) != set(config.train_targets) for row in values):
        raise RuntimeError("a B5 full epoch does not contain every target once")
    if hashlib.sha256(values.tobytes(order="C")).hexdigest() != (
        B5_FULL_ORDER_RAW_SHA256
    ):
        raise RuntimeError("B5 full target-order raw bytes differ")
    if _array_npy_sha256(values) != B5_FULL_ORDER_NPY_SHA256:
        raise RuntimeError("B5 full target-order NPY bytes differ")
    return values


def _uint64_seed_bank(*, seed: int, shape: tuple[int, int]) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    return np.ascontiguousarray(
        generator.integers(
            0,
            np.iinfo(np.uint64).max,
            size=shape,
            dtype=np.uint64,
        ),
        dtype=np.uint64,
    )


def full_validation_seed_bank(
    *, seed: int = B5_FULL_VALIDATION_BANK_SEED
) -> np.ndarray:
    if int(seed) != B5_FULL_VALIDATION_BANK_SEED:
        raise ValueError("B5 validation seed-bank seed differs")
    values = _uint64_seed_bank(seed=seed, shape=(126, 4))
    if hashlib.sha256(values.tobytes(order="C")).hexdigest() != (
        B5_VALIDATION_BANK_RAW_SHA256
    ):
        raise RuntimeError("B5 validation seed-bank raw bytes differ")
    if _array_npy_sha256(values) != B5_VALIDATION_BANK_NPY_SHA256:
        raise RuntimeError("B5 validation seed-bank NPY bytes differ")
    return values


def scientific_sampler_seed_bank(
    *, seed: int = B5_SCIENTIFIC_SAMPLER_BANK_SEED
) -> np.ndarray:
    """Materialize the independent future M32 bank without using it here."""

    if int(seed) != B5_SCIENTIFIC_SAMPLER_BANK_SEED:
        raise ValueError("B5 scientific sampler-bank seed differs")
    values = _uint64_seed_bank(seed=seed, shape=(126, 32))
    if hashlib.sha256(values.tobytes(order="C")).hexdigest() != (
        B5_SCIENTIFIC_BANK_RAW_SHA256
    ):
        raise RuntimeError("B5 scientific sampler-bank raw bytes differ")
    if _array_npy_sha256(values) != B5_SCIENTIFIC_BANK_NPY_SHA256:
        raise RuntimeError("B5 scientific sampler-bank NPY bytes differ")
    return values


def keyed_full_sigma_and_noise(
    *,
    seed: int,
    epoch_zero_based: int,
    target_frame: int,
    spatial_shape: Sequence[int] = B5_SPATIAL_SHAPE,
) -> tuple[np.float32, np.ndarray]:
    """Return one exact fresh full-training corruption."""

    if int(seed) != B5_FULL_TRAINING_NOISE_SEED:
        raise ValueError("B5 full training-noise seed differs")
    if not 0 <= int(epoch_zero_based) < B5_FULL_EPOCHS:
        raise ValueError("B5 full training epoch differs")
    if int(target_frame) not in B5_FULL_TRAIN_TARGETS:
        raise ValueError("B5 full training target differs")
    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("B5 full training spatial shape differs")
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(epoch_zero_based),
            int(target_frame),
            0xB5ED_0003,
        ]
    )
    generator = np.random.Generator(np.random.PCG64(sequence))
    sigma = np.float32(math.exp(-1.2 + 1.2 * float(generator.standard_normal())))
    noise = generator.standard_normal((5, *shape), dtype=np.float32)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.all(np.isfinite(noise)):
        raise FloatingPointError("B5 full training corruption is non-finite")
    return sigma, np.ascontiguousarray(noise, dtype=np.float32)


def sigma_and_noise_from_uint64(
    seed: int | np.uint64,
    *,
    spatial_shape: Sequence[int] = B5_SPATIAL_SHAPE,
) -> tuple[np.float32, np.ndarray]:
    """Expand one immutable validation seed to sigma and full noise."""

    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("B5 validation spatial shape differs")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    sigma = np.float32(math.exp(-1.2 + 1.2 * float(generator.standard_normal())))
    noise = generator.standard_normal((5, *shape), dtype=np.float32)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.all(np.isfinite(noise)):
        raise FloatingPointError("B5 validation corruption is non-finite")
    return sigma, np.ascontiguousarray(noise, dtype=np.float32)


def full_learning_rate(
    config: B5EDMFullConfig,
    zero_based_update: int,
) -> float:
    if config != B5EDMFullConfig():
        raise ValueError("B5 learning rate requires exact frozen config")
    index = int(zero_based_update)
    if not 0 <= index < config.total_optimizer_steps:
        raise ValueError("B5 optimizer-update index differs")
    progress = index / (config.total_optimizer_steps - 1)
    return float(
        config.minimum_learning_rate
        + 0.5
        * (config.peak_learning_rate - config.minimum_learning_rate)
        * (1.0 + math.cos(math.pi * progress))
    )


def accumulation_groups(
    epoch_order: Sequence[int] | np.ndarray,
    config: B5EDMFullConfig = B5EDMFullConfig(),
) -> tuple[np.ndarray, ...]:
    """Partition one exact epoch into 107 groups of four and one of two."""

    if config != B5EDMFullConfig():
        raise ValueError("B5 accumulation requires the exact frozen full config")
    values = np.asarray(epoch_order, dtype=np.int64)
    if values.shape != (len(config.train_targets),):
        raise ValueError("B5 accumulation epoch-order shape differs")
    if set(map(int, values)) != set(config.train_targets):
        raise ValueError("B5 accumulation epoch order is not a target permutation")
    groups = tuple(
        np.ascontiguousarray(
            values[cursor : cursor + config.gradient_accumulation_targets],
            dtype=np.int64,
        )
        for cursor in range(0, len(values), config.gradient_accumulation_targets)
    )
    if len(groups) != config.optimizer_steps_per_epoch:
        raise RuntimeError("B5 accumulation optimizer-step count differs")
    if [len(group) for group in groups] != [4] * 107 + [2]:
        raise RuntimeError("B5 accumulation group sizes differ")
    if not np.array_equal(np.concatenate(groups), values):
        raise RuntimeError("B5 accumulation changed target order")
    return groups


def _save_npy_atomic(path: Path, values: np.ndarray) -> Path:
    destination = Path(path)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    with partial.open("xb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


class B5ResidualOneStepDataset:
    """Join exact C5P windows to one immutable H1 mean artifact."""

    def __init__(
        self,
        windows: _WindowDataset,
        forecast: _ForecastArtifact,
        *,
        split: str,
    ) -> None:
        expected = (
            B5_FULL_TRAIN_TARGETS
            if split == "train"
            else B5_FULL_VALIDATION_TARGETS
            if split == "validation"
            else ()
        )
        if not expected:
            raise ValueError("B5 residual dataset split differs")
        if (
            windows.split != split
            or windows.context_frames != 1
            or tuple(windows.target_frames) != expected
            or windows.augment
            or tuple(windows.fields) != B5_FIELD_ORDER
        ):
            raise ValueError("B5 residual one-step window contract differs")
        if tuple(forecast.target_frames) != expected:
            raise ValueError("B5 residual H1 mean target index differs")
        self.windows = windows
        self.forecast = forecast
        self.split = split
        self.target_frames = expected
        self.scales = np.asarray(B5_RESIDUAL_SCALES, dtype=np.float32).reshape(
            5, 1, 1, 1
        )

    def __len__(self) -> int:
        return len(self.target_frames)

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
            raise RuntimeError("B5 residual target order differs")
        context_indices = tuple(int(value) for value in item["context_frame_indices"])
        if context_indices != (target - 1,):
            raise RuntimeError("B5 residual context index differs")
        if int(item.get("toroidal_roll", 0)) != 0:
            raise RuntimeError("B5 full residual augmentation is prohibited")
        context = np.asarray(item["context"], dtype=np.float32)
        truth = np.asarray(item["target"], dtype=np.float32)
        mean = self.forecast.read(position, position + 1)[0]
        if (
            context.shape != (1, 5, *VOLUME_SHAPE)
            or truth.shape != (5, *VOLUME_SHAPE)
            or mean.shape != (5, *VOLUME_SHAPE)
        ):
            raise ValueError("B5 full residual example shape differs")
        normalized = (truth - mean) / self.scales
        condition = np.concatenate((context[0], mean), axis=0)
        if not np.all(np.isfinite(normalized)) or not np.all(np.isfinite(condition)):
            raise FloatingPointError("B5 full residual example is non-finite")
        return {
            "target_frame_index": np.int64(target),
            "context_frame_index": np.int64(target - 1),
            "condition": np.ascontiguousarray(condition, dtype=np.float32),
            "normalized_residual": np.ascontiguousarray(normalized, dtype=np.float32),
            "deterministic_mean": np.ascontiguousarray(mean, dtype=np.float32),
            "target_truth_used_as_condition": False,
            "absolute_time_used_as_condition": False,
        }


def _example_tensors(
    dataset: B5ResidualOneStepDataset,
    target_frame: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    item = dataset[dataset.index_for_target(target_frame)]
    condition = torch.from_numpy(item["condition"])[None].to(device, torch.float32)
    clean = torch.from_numpy(item["normalized_residual"])[None].to(
        device, torch.float32
    )
    return condition, clean


@torch.no_grad()
def update_ema_model(
    ema_model: nn.Module,
    raw_model: nn.Module,
    *,
    decay: float,
) -> None:
    """Apply the exact post-optimizer B5 EMA update in place."""

    if float(decay) != 0.999:
        raise ValueError("B5 full EMA decay differs")
    ema_parameters = dict(ema_model.named_parameters())
    raw_parameters = dict(raw_model.named_parameters())
    if ema_parameters.keys() != raw_parameters.keys():
        raise RuntimeError("B5 EMA/raw parameter names differ")
    for name, ema_parameter in ema_parameters.items():
        raw_parameter = raw_parameters[name]
        ema_parameter.mul_(decay).add_(raw_parameter.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    raw_buffers = dict(raw_model.named_buffers())
    if ema_buffers.keys() != raw_buffers.keys():
        raise RuntimeError("B5 EMA/raw buffer names differ")
    for name, ema_buffer in ema_buffers.items():
        if not torch.equal(ema_buffer, raw_buffers[name]):
            raise RuntimeError(f"B5 fixed buffer {name!r} changed")


def validation_edm_loss(
    *,
    model: JointFieldResidualEDM,
    dataset: B5ResidualOneStepDataset,
    seed_bank: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Compute the exact FP32 126-target, four-probe selection metric."""

    seeds = np.asarray(seed_bank)
    if dataset.split != "validation" or dataset.target_frames != (
        B5_FULL_VALIDATION_TARGETS
    ):
        raise ValueError("B5 validation residual dataset differs")
    if seeds.shape != (126, 4) or seeds.dtype != np.uint64:
        raise ValueError("B5 validation seed-bank schema differs")
    if hashlib.sha256(seeds.tobytes(order="C")).hexdigest() != (
        B5_VALIDATION_BANK_RAW_SHA256
    ):
        raise ValueError("B5 validation seed-bank bytes differ")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    unweighted: list[float] = []
    sigma_minimum = math.inf
    sigma_maximum = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for position, target in enumerate(dataset.target_frames):
            condition, clean = _example_tensors(dataset, target, device=device)
            sigma_values: list[float] = []
            noise_values: list[np.ndarray] = []
            for seed in seeds[position]:
                sigma, noise = sigma_and_noise_from_uint64(seed)
                sigma_values.append(float(sigma))
                noise_values.append(noise)
            sigma_tensor = torch.tensor(
                sigma_values, device=device, dtype=torch.float32
            )
            noise_tensor = torch.from_numpy(np.stack(noise_values, axis=0)).to(
                device, torch.float32
            )
            result = model.training_loss(
                clean.expand(4, *clean.shape[1:]).contiguous(),
                condition.expand(4, *condition.shape[1:]).contiguous(),
                sigma=sigma_tensor,
                noise=noise_tensor,
            )
            value = float(result.loss.detach().cpu())
            raw = float(result.unweighted_mse.detach().cpu())
            if not math.isfinite(value) or not math.isfinite(raw):
                raise FloatingPointError("B5 validation denoising loss is non-finite")
            losses.append(value)
            unweighted.append(raw)
            sigma_minimum = min(sigma_minimum, min(sigma_values))
            sigma_maximum = max(sigma_maximum, max(sigma_values))
    model.train(was_training)
    return {
        "target_frames": [498, 624],
        "target_count": 126,
        "probes_per_target": 4,
        "probe_count": 504,
        "precision": "float32_no_autocast_TF32_disabled",
        "mean_EDM_loss": float(np.mean(losses, dtype=np.float64)),
        "mean_unweighted_MSE": float(np.mean(unweighted, dtype=np.float64)),
        "minimum_sigma": float(sigma_minimum),
        "maximum_sigma": float(sigma_maximum),
        "wall_seconds": float(time.perf_counter() - started),
    }


def select_earliest_lowest_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the earliest numerically lowest validation-loss candidate."""

    records = list(candidates)
    if len(records) != 20:
        raise ValueError("B5 selection requires exactly 20 candidates")
    expected_epochs = list(range(5, 101, 5))
    observed_epochs = [int(record["completed_epoch"]) for record in records]
    if observed_epochs != expected_epochs:
        raise ValueError("B5 selection candidate epochs differ")
    for record in records:
        value = float(record["validation"]["mean_EDM_loss"])
        if not math.isfinite(value):
            raise FloatingPointError("B5 candidate validation loss is non-finite")
    return min(
        records,
        key=lambda record: (
            float(record["validation"]["mean_EDM_loss"]),
            int(record["completed_epoch"]),
        ),
    )


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fixed_reload_output(
    *,
    model: JointFieldResidualEDM,
    dataset: B5ResidualOneStepDataset,
    device: torch.device,
) -> Tensor:
    condition, clean = _example_tensors(dataset, 498, device=device)
    sigma, noise = sigma_and_noise_from_uint64(B5_FULL_RELOAD_PROBE_SEED)
    sigma_tensor = torch.tensor([float(sigma)], device=device)
    noisy = clean + sigma_tensor.reshape(1, 1, 1, 1, 1) * torch.from_numpy(noise)[
        None
    ].to(device)
    model.eval()
    with torch.inference_mode():
        return model.denoise(noisy, condition, sigma_tensor).to("cpu", torch.float32)


def train_b5_edm_full(
    *,
    training_dataset: B5ResidualOneStepDataset,
    validation_dataset: B5ResidualOneStepDataset,
    output: Path,
    device: torch.device,
    paper0_commit: str,
    slurm_job_id: str,
    authority: Mapping[str, Any],
    config: B5EDMFullConfig = B5EDMFullConfig(),
    model_config: FieldResidualUNetConfig = FieldResidualUNetConfig(),
    on_epoch: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute the one prospectively frozen B5 seed-1701 full training run."""

    destination = Path(output)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if config != B5EDMFullConfig() or model_config != FieldResidualUNetConfig():
        raise ValueError("B5 full training configuration differs")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B5 full training requires an allocated CUDA device")
    if (
        training_dataset.split != "train"
        or training_dataset.target_frames != B5_FULL_TRAIN_TARGETS
        or validation_dataset.split != "validation"
        or validation_dataset.target_frames != B5_FULL_VALIDATION_TARGETS
    ):
        raise ValueError("B5 full residual datasets differ")
    if any("85606" in str(value).lower() for value in authority.values()):
        raise ValueError("B5 full authority mentions held-out 85606")
    destination.mkdir(parents=True)
    candidate_directory = destination / "candidates"
    candidate_directory.mkdir()
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats(device)

    order = full_training_order(config)
    order_path = _save_npy_atomic(destination / "training_order.npy", order)
    validation_bank = full_validation_seed_bank(seed=config.validation_seed_bank_seed)
    validation_bank_path = _save_npy_atomic(
        destination / "validation_seed_bank.npy", validation_bank
    )
    run_config_path = destination / "config.json"
    write_strict_json_atomic(
        run_config_path,
        {
            "schema_version": 1,
            "scope": "B5_seed1701_full_training_and_data_only_selection_85604",
            "paper0_commit": str(paper0_commit),
            "slurm_job_id": str(slurm_job_id),
            "run_config": config.to_record(),
            "model_config": model_config.to_record(),
            "residual_scales": {
                "field_order": list(B5_FIELD_ORDER),
                "values": list(B5_RESIDUAL_SCALES),
                "operation": "divide_without_centering",
            },
            "authority": dict(authority),
        },
    )

    raw = JointFieldResidualEDM(FieldResidualUNet3D(model_config)).to(
        device, torch.float32
    )
    raw.train()
    count = parameter_count(raw)
    if count != 11_604_709:
        raise RuntimeError(f"B5 full parameter count differs: {count}")
    initial_state_sha = module_state_sha256(raw)
    ema = copy.deepcopy(raw).to(device, torch.float32)
    ema.eval()
    ema.requires_grad_(False)
    optimizer = AdamW(
        raw.parameters(),
        lr=config.peak_learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    history_path = destination / "history.jsonl"
    records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    global_update = 0
    for epoch_zero, epoch_order in enumerate(order):
        completed_epoch = epoch_zero + 1
        epoch_started = time.perf_counter()
        raw.train()
        epoch_losses: list[float] = []
        epoch_unweighted: list[float] = []
        gradient_norms: list[float] = []
        epoch_sigmas: list[float] = []
        learning_rates: list[float] = []
        for group in accumulation_groups(epoch_order, config):
            group_size = len(group)
            optimizer.zero_grad(set_to_none=True)
            for target_value in group:
                target = int(target_value)
                condition, clean = _example_tensors(
                    training_dataset, target, device=device
                )
                sigma, noise = keyed_full_sigma_and_noise(
                    seed=config.training_noise_seed,
                    epoch_zero_based=epoch_zero,
                    target_frame=target,
                )
                sigma_tensor = torch.tensor([float(sigma)], device=device)
                noise_tensor = torch.from_numpy(noise)[None].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = raw.training_loss(
                        clean,
                        condition,
                        sigma=sigma_tensor,
                        noise=noise_tensor,
                    )
                if not torch.isfinite(losses.loss):
                    raise FloatingPointError(
                        f"non-finite B5 full loss at epoch {completed_epoch} target {target}"
                    )
                (losses.loss / group_size).backward()
                epoch_losses.append(float(losses.loss.detach().cpu()))
                epoch_unweighted.append(float(losses.unweighted_mse.detach().cpu()))
                epoch_sigmas.append(float(sigma))
            preclip = torch.nn.utils.clip_grad_norm_(
                raw.parameters(), config.gradient_clip
            )
            if not torch.isfinite(preclip):
                raise FloatingPointError("B5 full gradient norm is non-finite")
            learning_rate = full_learning_rate(config, global_update)
            for group_record in optimizer.param_groups:
                group_record["lr"] = learning_rate
            optimizer.step()
            update_ema_model(ema, raw, decay=config.ema_decay)
            gradient_norms.append(float(preclip.detach().cpu()))
            learning_rates.append(learning_rate)
            global_update += 1

        if global_update != completed_epoch * config.optimizer_steps_per_epoch:
            raise RuntimeError("B5 full optimizer-update count drifted")
        if any(not torch.isfinite(parameter).all() for parameter in raw.parameters()):
            raise FloatingPointError("B5 full raw parameter is non-finite")
        if any(not torch.isfinite(parameter).all() for parameter in ema.parameters()):
            raise FloatingPointError("B5 full EMA parameter is non-finite")
        torch.cuda.synchronize(device)
        validation: dict[str, Any] | None = None
        candidate: dict[str, Any] | None = None
        if completed_epoch in config.validation_completed_epochs:
            validation = validation_edm_loss(
                model=ema,
                dataset=validation_dataset,
                seed_bank=validation_bank,
                device=device,
            )
            candidate_path = candidate_directory / f"ema_epoch_{completed_epoch:03d}.pt"
            candidate_payload = {
                "schema_version": 1,
                "kind": "B5_EMA_validation_candidate",
                "paper0_commit": str(paper0_commit),
                "slurm_job_id": str(slurm_job_id),
                "completed_epoch": completed_epoch,
                "global_optimizer_step": global_update,
                "validation": validation,
                "run_config": config.to_record(),
                "model_config": model_config.to_record(),
                "residual_scales": list(B5_RESIDUAL_SCALES),
                "EMA_model_state": {
                    name: value.detach().to("cpu")
                    for name, value in ema.state_dict().items()
                },
                "scientific_forecast_generated": False,
                "physics_metric_used_for_selection": False,
                "held_out_85606_read": False,
            }
            save_torch_atomic(candidate_path, candidate_payload)
            candidate = {
                "completed_epoch": completed_epoch,
                "global_optimizer_step": global_update,
                "validation": validation,
                "path": str(candidate_path),
                "sha256": sha256_path(candidate_path),
            }
            candidates.append(candidate)
        record = {
            "completed_epoch": completed_epoch,
            "global_optimizer_step": global_update,
            "train_target_count": len(epoch_losses),
            "train_mean_EDM_loss": float(np.mean(epoch_losses, dtype=np.float64)),
            "train_mean_unweighted_MSE": float(
                np.mean(epoch_unweighted, dtype=np.float64)
            ),
            "train_minimum_sigma": float(min(epoch_sigmas)),
            "train_maximum_sigma": float(max(epoch_sigmas)),
            "mean_preclip_gradient_norm": float(
                np.mean(gradient_norms, dtype=np.float64)
            ),
            "maximum_preclip_gradient_norm": float(max(gradient_norms)),
            "first_learning_rate": float(learning_rates[0]),
            "last_learning_rate": float(learning_rates[-1]),
            "EMA_updates": global_update,
            "validation_candidate": validation is not None,
            "validation": validation,
            "candidate": candidate,
            "epoch_wall_seconds": float(time.perf_counter() - epoch_started),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records.append(record)
        if on_epoch is not None:
            on_epoch(record)

    if global_update != config.total_optimizer_steps or len(records) != 100:
        raise RuntimeError("B5 full training budget did not complete")
    selected_record = dict(select_earliest_lowest_candidate(candidates))
    selected_candidate = _load_torch(Path(selected_record["path"]))
    selected_state = selected_candidate["EMA_model_state"]
    selected_model = JointFieldResidualEDM(FieldResidualUNet3D(model_config)).to(
        device, torch.float32
    )
    selected_model.load_state_dict(selected_state, strict=True)
    selected_model.eval()
    expected_reload = _fixed_reload_output(
        model=selected_model,
        dataset=validation_dataset,
        device=device,
    )
    selected_path = destination / "selected.pt"
    selected_payload = {
        "schema_version": 1,
        "kind": "B5_selected_EMA_checkpoint",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "run_config": config.to_record(),
        "model_config": model_config.to_record(),
        "residual_scales": list(B5_RESIDUAL_SCALES),
        "selected_completed_epoch": int(selected_record["completed_epoch"]),
        "selected_optimizer_step": int(selected_record["global_optimizer_step"]),
        "selection_metric": "fixed_seed_validation_EDM_loss",
        "selected_validation": dict(selected_record["validation"]),
        "source_candidate": {
            "path": selected_record["path"],
            "sha256": selected_record["sha256"],
        },
        "model_state": selected_state,
        "physics_metric_used_for_selection": False,
        "scientific_forecast_generated": False,
        "held_out_85606_read": False,
    }
    save_torch_atomic(selected_path, selected_payload)
    reloaded_payload = _load_torch(selected_path)
    restored = JointFieldResidualEDM(FieldResidualUNet3D(model_config)).to(
        device, torch.float32
    )
    restored.load_state_dict(reloaded_payload["model_state"], strict=True)
    restored.eval()
    observed_reload = _fixed_reload_output(
        model=restored,
        dataset=validation_dataset,
        device=device,
    )
    reload_exact = bool(torch.equal(expected_reload, observed_reload))

    final_state_path = destination / "final_training_state.pt"
    final_state_payload = {
        "schema_version": 1,
        "kind": "B5_final_resumable_training_state",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "run_config": config.to_record(),
        "model_config": model_config.to_record(),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_update,
        "raw_model_state": {
            name: value.detach().to("cpu") for name, value in raw.state_dict().items()
        },
        "EMA_model_state": {
            name: value.detach().to("cpu") for name, value in ema.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "training_order_sha256": sha256_path(order_path),
        "validation_seed_bank_sha256": sha256_path(validation_bank_path),
        "scientific_forecast_generated": False,
        "held_out_85606_read": False,
    }
    save_torch_atomic(final_state_path, final_state_payload)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    wall_seconds = time.perf_counter() - started
    all_finite = all(
        math.isfinite(float(record["train_mean_EDM_loss"]))
        and math.isfinite(float(record["mean_preclip_gradient_norm"]))
        and (
            record["validation"] is None
            or math.isfinite(float(record["validation"]["mean_EDM_loss"]))
        )
        for record in records
    )
    result = {
        "schema_version": 1,
        "scope": "B5_seed1701_full_training_and_data_only_selection_85604",
        "status": "training_completed_checkpoint_selected",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "development_run": "85604",
        "sequestered_run": "85606",
        "config": config.to_record(),
        "model_config": model_config.to_record(),
        "parameter_count": count,
        "initial_model_state_sha256": initial_state_sha,
        "completed_epochs": len(records),
        "target_presentations": config.target_presentations,
        "completed_optimizer_steps": global_update,
        "EMA_updates": global_update,
        "candidate_count": len(candidates),
        "candidate_completed_epochs": [
            int(candidate["completed_epoch"]) for candidate in candidates
        ],
        "selected_completed_epoch": int(selected_record["completed_epoch"]),
        "selected_optimizer_step": int(selected_record["global_optimizer_step"]),
        "selected_validation": dict(selected_record["validation"]),
        "final_candidate_validation": dict(candidates[-1]["validation"]),
        "checkpoint_reload_bitwise_exact": reload_exact,
        "selected_model_state_sha256": module_state_sha256(restored),
        "all_losses_and_gradients_finite": all_finite,
        "peak_cuda_bytes": peak_bytes,
        "peak_cuda_GiB": float(peak_bytes / 1024**3),
        "wall_seconds": float(wall_seconds),
        "artifacts": {
            "config": {
                "path": str(run_config_path),
                "sha256": sha256_path(run_config_path),
            },
            "training_order": {
                "path": str(order_path),
                "sha256": sha256_path(order_path),
                "raw_sha256": B5_FULL_ORDER_RAW_SHA256,
            },
            "validation_seed_bank": {
                "path": str(validation_bank_path),
                "sha256": sha256_path(validation_bank_path),
                "raw_sha256": B5_VALIDATION_BANK_RAW_SHA256,
            },
            "history": {
                "path": str(history_path),
                "sha256": sha256_path(history_path),
                "records": len(records),
            },
            "selected_checkpoint": {
                "path": str(selected_path),
                "sha256": sha256_path(selected_path),
            },
            "selected_source_candidate": {
                "path": selected_record["path"],
                "sha256": selected_record["sha256"],
            },
            "final_training_state": {
                "path": str(final_state_path),
                "sha256": sha256_path(final_state_path),
            },
            "candidate_checkpoints": candidates,
        },
        "training_performed": True,
        "validation_frames_read": True,
        "checkpoint_selection_used_validation": True,
        "checkpoint_selection_metric": "fixed_seed_validation_EDM_loss",
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "sampled_forecast_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "scientific_evaluation_completed": False,
        "scientific_acceptance_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    write_strict_json_atomic(destination / "result.json", result)
    return result
