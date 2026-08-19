"""Frozen 85604-only mechanics for the bounded B4 PDE-Refiner smoke.

This module intentionally exposes no full-training entrypoint.  A successful
smoke may authorize writing another protocol; it does not authorize the
prospective 100-epoch run specified in the B4 design document.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .codec_training import save_torch_atomic, seed_everything, sha256_path
from .model_data import load_strict_json, write_strict_json_atomic
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog, epoch_order
from .models.o2 import MaskedLatentTransition, O2ViTConfig
from .models.pde_refiner import (
    C5PPDERefinerOneStepModel,
    PDERefinerConfig,
    PDERefinerMaskedLatentTransition,
    RefinerLoadAudit,
)
from .o2_training import load_frozen_codec, scale_accumulated_gradients
from .o2_training_data import OneStepWindowDataset


B4_SMOKE_SEED = 1701
B4_TRAINING_LEVEL_SEED = 41_001
B4_TRAINING_NOISE_SEED = 41_002
B4_VALIDATION_SEED_BANK_SEED = 41_003
B4_SCIENTIFIC_EVALUATION_SEED_BANK_SEED = 41_032
B4_VALIDATION_TARGET_START = 498
B4_VALIDATION_TARGET_STOP = 624
B4_VALIDATION_MEMBERS = 2
B4_LATENT_SHAPE = (32, 16, 8, 22)


@dataclass(frozen=True)
class PDERefinerSmokeConfig:
    """The only B4 optimization budget authorized by the current protocol."""

    seed: int
    epochs: int
    train_target_start: int
    train_target_stop: int
    validation_target_start: int
    validation_target_stop: int
    microbatch_targets: int
    gradient_accumulation_targets: int
    validation_members: int
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    ema_decay: float
    training_precision: str
    training_level_seed: int
    training_noise_seed: int
    validation_seed_bank_seed: int

    @classmethod
    def frozen(cls, *, seed: int) -> "PDERefinerSmokeConfig":
        if int(seed) != B4_SMOKE_SEED:
            raise ValueError("the bounded B4 smoke is fixed to seed 1701")
        return cls(
            seed=int(seed),
            epochs=2,
            train_target_start=2,
            train_target_stop=18,
            validation_target_start=498,
            validation_target_stop=502,
            microbatch_targets=1,
            gradient_accumulation_targets=16,
            validation_members=2,
            learning_rate=1.0e-4,
            betas=(0.9, 0.999),
            weight_decay=1.0e-5,
            gradient_clip=1.0,
            ema_decay=0.995,
            training_precision="float32_no_autocast_TF32_disabled",
            training_level_seed=B4_TRAINING_LEVEL_SEED,
            training_noise_seed=B4_TRAINING_NOISE_SEED,
            validation_seed_bank_seed=B4_VALIDATION_SEED_BANK_SEED,
        )

    @property
    def context_frames(self) -> int:
        return 1

    @property
    def train_targets(self) -> tuple[int, ...]:
        return tuple(range(self.train_target_start, self.train_target_stop))

    @property
    def validation_targets(self) -> tuple[int, ...]:
        return tuple(range(self.validation_target_start, self.validation_target_stop))

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(
            len(self.train_targets) / self.gradient_accumulation_targets
        )

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "mode": "smoke",
                "arm": "B4-PDE-Refiner-H1",
                "context_frames": self.context_frames,
                "future_frames": 1,
                "fields": list(FAMILY_FIELDS["c5p"]),
                "train_targets": [self.train_target_start, self.train_target_stop],
                "validation_targets": [
                    self.validation_target_start,
                    self.validation_target_stop,
                ],
                "optimizer": "AdamW",
                "scheduler": "constant_bounded_smoke",
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "training_loss": "uniform_level_explicit_standardized_latent_MSE",
                "checkpoint_selection": (
                    "earliest_lowest_fixed_seed_M2_ensemble_mean_equal_channel_"
                    "decoded_standardized_field_MAE_final_refinement_EMA"
                ),
                "physics_derived_loss_allowed": False,
                "absolute_time_input_allowed": False,
                "early_stopping": False,
                "scientific_result": False,
                "full_training_authorized": False,
            }
        )
        return record


@dataclass(frozen=True)
class RefinerParentArtifacts:
    checkpoint_path: Path
    checkpoint_sha256: str
    codec_path: Path
    codec_sha256: str
    latent_normalization_path: Path
    latent_normalization_sha256: str


@dataclass(frozen=True)
class RefinerParameterGroups:
    parent: tuple[nn.Parameter, ...]
    refinement: tuple[nn.Parameter, ...]
    parent_names: tuple[str, ...]
    refinement_names: tuple[str, ...]

    @property
    def all_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.parent + self.refinement

    def to_record(self) -> dict[str, Any]:
        return {
            "parent_parameter_tensor_count": len(self.parent),
            "refinement_parameter_tensor_count": len(self.refinement),
            "parent_parameter_count": sum(item.numel() for item in self.parent),
            "refinement_parameter_count": sum(
                item.numel() for item in self.refinement
            ),
            "total_parameter_count": sum(
                item.numel() for item in self.all_parameters
            ),
            "parent_parameter_names": list(self.parent_names),
            "refinement_parameter_names": list(self.refinement_names),
        }


@dataclass(frozen=True)
class RefinerObjective:
    loss: Tensor
    prediction: Tensor
    target: Tensor
    provisional_target: Tensor
    levels: Tensor


def smoke_training_levels(
    config: PDERefinerSmokeConfig,
) -> np.ndarray:
    """Materialize the exact PCG64 level draws for the bounded smoke."""

    if config != PDERefinerSmokeConfig.frozen(seed=B4_SMOKE_SEED):
        raise ValueError("B4 level sequence requires the frozen smoke config")
    generator = np.random.Generator(np.random.PCG64(config.training_level_seed))
    levels = generator.integers(
        0,
        4,
        size=(config.epochs, len(config.train_targets)),
        dtype=np.int64,
    )
    if set(map(int, np.unique(levels))) != {0, 1, 2, 3}:
        raise RuntimeError("bounded B4 level draws do not exercise every level")
    if any(
        set(map(int, np.unique(epoch_levels))) != {0, 1, 2, 3}
        for epoch_levels in levels
    ):
        raise RuntimeError("a B4 smoke epoch omits a refinement level")
    return np.ascontiguousarray(levels, dtype=np.int64)


def validation_seed_bank(
    *,
    seed: int = B4_VALIDATION_SEED_BANK_SEED,
) -> np.ndarray:
    """Return the complete immutable `[target,member,refinement]` seed bank."""

    if int(seed) != B4_VALIDATION_SEED_BANK_SEED:
        raise ValueError("B4 validation seed-bank seed differs")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    values = generator.integers(
        0,
        np.iinfo(np.uint64).max,
        size=(
            B4_VALIDATION_TARGET_STOP - B4_VALIDATION_TARGET_START,
            B4_VALIDATION_MEMBERS,
            3,
        ),
        dtype=np.uint64,
    )
    return np.ascontiguousarray(values, dtype=np.uint64)


def refinement_noise_from_seeds(
    seeds: np.ndarray,
    *,
    latent_shape: Sequence[int] = B4_LATENT_SHAPE,
) -> np.ndarray:
    """Expand one `[member,level]` seed row into full float32 latent fields."""

    values = np.asarray(seeds)
    if values.shape != (B4_VALIDATION_MEMBERS, 3) or values.dtype != np.uint64:
        raise ValueError("B4 validation seeds must be uint64 [2,3]")
    shape = tuple(int(item) for item in latent_shape)
    if shape != B4_LATENT_SHAPE:
        raise ValueError("B4 latent-noise shape differs from [32,16,8,22]")
    noise = np.empty((B4_VALIDATION_MEMBERS, 3, *shape), dtype=np.float32)
    for member in range(B4_VALIDATION_MEMBERS):
        for level in range(3):
            generator = np.random.Generator(
                np.random.PCG64(values[member, level])
            )
            noise[member, level] = generator.standard_normal(
                shape,
                dtype=np.float32,
            )
    return np.ascontiguousarray(noise, dtype=np.float32)


def save_numpy_exclusive(path: Path, values: np.ndarray) -> str:
    """Save one NPY artifact without overwriting and return its SHA-256."""

    destination = Path(path)
    with destination.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    return sha256_path(destination)


def refinement_training_pair(
    *,
    previous: Tensor,
    target: Tensor,
    levels: Tensor,
    noise: Tensor,
    standard_deviations: Sequence[float],
) -> tuple[Tensor, Tensor]:
    """Construct provisional inputs and denoising targets for mixed levels."""

    if previous.shape != target.shape or target.shape != noise.shape:
        raise ValueError("previous, target, and noise latent shapes differ")
    if levels.shape != (target.shape[0],):
        raise ValueError("refinement levels must have shape [batch]")
    if levels.dtype == torch.bool:
        raise ValueError("refinement levels must be integer-valued")
    if levels.is_floating_point() and not torch.equal(levels, levels.round()):
        raise ValueError("refinement levels must be integer-valued")
    levels = levels.to(device=target.device, dtype=torch.int64)
    if torch.any(levels < 0) or torch.any(levels > 3):
        raise ValueError("refinement level leaves 0..3")
    if len(tuple(standard_deviations)) != 3:
        raise ValueError("B4 requires three refinement standard deviations")
    sigma_table = target.new_tensor((0.0, *map(float, standard_deviations)))
    sigma = sigma_table[levels].reshape(
        target.shape[0],
        *([1] * (target.ndim - 1)),
    )
    refining = (levels > 0).reshape(
        target.shape[0],
        *([1] * (target.ndim - 1)),
    )
    provisional = torch.where(
        refining,
        target + sigma * noise,
        torch.zeros_like(target),
    )
    objective_target = torch.where(refining, noise, target - previous)
    return provisional, objective_target


def refinement_objective(
    *,
    model: C5PPDERefinerOneStepModel,
    context: Tensor,
    target: Tensor,
    levels: Tensor,
    noise: Tensor,
) -> RefinerObjective:
    """Compute the frozen data-only latent MSE for one mixed-level batch."""

    standardized_context = model.encode_context(context)
    standardized_target = model.encode_target(target)
    if noise.shape != standardized_target.shape:
        raise ValueError("B4 training noise shape differs from target latent")
    noise = noise.to(
        device=standardized_target.device,
        dtype=standardized_target.dtype,
    )
    if not torch.isfinite(noise).all():
        raise ValueError("B4 training noise must be finite")
    provisional, objective_target = refinement_training_pair(
        previous=standardized_context[:, -1],
        target=standardized_target,
        levels=levels,
        noise=noise,
        standard_deviations=model.refinement_standard_deviations,
    )
    prediction = model.transition(
        standardized_context,
        provisional,
        levels,
    )
    loss = (prediction - objective_target).square().mean()
    return RefinerObjective(
        loss=loss,
        prediction=prediction,
        target=objective_target,
        provisional_target=provisional,
        levels=levels.to(torch.int64),
    )


class TransitionEMA:
    """Exact named-state exponential moving average for one transition."""

    def __init__(self, module: nn.Module, *, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }
        self.updates = 0

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        observed = module.state_dict()
        if tuple(observed) != tuple(self.shadow):
            raise ValueError("EMA module state keys differ")
        for name, value in observed.items():
            shadow = self.shadow[name]
            if shadow.shape != value.shape or shadow.dtype != value.dtype:
                raise ValueError(f"EMA state tensor {name} schema differs")
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value.detach())
        self.updates += 1

    def cpu_state_dict(self) -> dict[str, Tensor]:
        return {
            name: value.detach().to("cpu").clone()
            for name, value in self.shadow.items()
        }

    @contextmanager
    def applied_to(self, module: nn.Module) -> Iterator[None]:
        current = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }
        module.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            module.load_state_dict(current, strict=True)


def _window_loader(
    dataset: OneStepWindowDataset,
    ordered_targets: Sequence[int],
) -> DataLoader:
    start = dataset.target_frames[0]
    indices = [int(target) - start for target in ordered_targets]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("B4 batch order contains a target outside the dataset")
    return DataLoader(dataset, batch_size=1, sampler=indices, num_workers=0)


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _parameter_groups(
    transition: PDERefinerMaskedLatentTransition,
) -> RefinerParameterGroups:
    parent: list[nn.Parameter] = []
    refinement: list[nn.Parameter] = []
    parent_names: list[str] = []
    refinement_names: list[str] = []
    for name, parameter in transition.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"B4 transition parameter {name} is frozen")
        if transition._is_refinement_key(name):
            refinement.append(parameter)
            refinement_names.append(name)
        else:
            parent.append(parameter)
            parent_names.append(name)
    if not parent or not refinement:
        raise RuntimeError("B4 parameter accounting has an empty family")
    if {id(item) for item in parent} & {id(item) for item in refinement}:
        raise RuntimeError("B4 parameter families overlap")
    if {id(item) for item in parent + refinement} != {
        id(item) for item in transition.parameters()
    }:
        raise RuntimeError("B4 parameter accounting omits transition tensors")
    return RefinerParameterGroups(
        parent=tuple(parent),
        refinement=tuple(refinement),
        parent_names=tuple(parent_names),
        refinement_names=tuple(refinement_names),
    )


def _verify_parent_payload(
    *,
    artifacts: RefinerParentArtifacts,
    model_config: O2ViTConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (
        artifacts.checkpoint_path,
        artifacts.codec_path,
        artifacts.latent_normalization_path,
    ):
        if "85606" in str(path).lower():
            raise ValueError("held-out paths are prohibited")
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_path(artifacts.checkpoint_path) != artifacts.checkpoint_sha256:
        raise ValueError("B4 parent checkpoint SHA-256 mismatch")
    if sha256_path(artifacts.codec_path) != artifacts.codec_sha256:
        raise ValueError("B4 codec checkpoint SHA-256 mismatch")
    if (
        sha256_path(artifacts.latent_normalization_path)
        != artifacts.latent_normalization_sha256
    ):
        raise ValueError("B4 latent-normalization SHA-256 mismatch")

    payload = torch.load(
        artifacts.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("kind") != "selected_O2_transition":
        raise ValueError("B4 parent is not a selected O2 transition")
    config = payload.get("config", {})
    if config.get("arm") != "C5P-H1" or int(config.get("seed", -1)) != 1701:
        raise ValueError("B4 parent is not frozen C5P-H1 seed 1701")
    if int(payload.get("epoch", -1)) != 193:
        raise ValueError("B4 parent selected epoch differs")
    if int(payload.get("global_step", -1)) != 5238:
        raise ValueError("B4 parent selected optimizer step differs")
    if float(payload.get("validation_loss", math.inf)) != 0.04558250684515488:
        raise ValueError("B4 parent selected validation loss differs")
    if payload.get("model_config") != model_config.to_record():
        raise ValueError("B4 parent model configuration differs")
    codec_record = payload.get("codec_checkpoint", {})
    if str(codec_record.get("path")) != str(artifacts.codec_path):
        raise ValueError("B4 parent codec path differs")
    if str(codec_record.get("sha256")) != artifacts.codec_sha256:
        raise ValueError("B4 parent codec hash differs")

    normalization = load_strict_json(artifacts.latent_normalization_path)
    if payload.get("latent_normalization") != normalization:
        raise ValueError("B4 embedded and external latent normalization differ")
    if normalization.get("fit_frames") != [0, 432]:
        raise ValueError("B4 latent normalization was not fit on training only")
    if normalization.get("codec_checkpoint_sha256") != artifacts.codec_sha256:
        raise ValueError("B4 latent normalization belongs to another codec")
    if normalization.get("held_out_85606_read") is not False:
        raise ValueError("B4 normalization reports held-out access")
    return payload, normalization


def _build_model(
    *,
    config: PDERefinerSmokeConfig,
    artifacts: RefinerParentArtifacts,
    device: torch.device,
    model_config: O2ViTConfig,
    refiner_config: PDERefinerConfig,
) -> tuple[
    C5PPDERefinerOneStepModel,
    RefinerLoadAudit,
    dict[str, Any],
    dict[str, Any],
]:
    payload, normalization = _verify_parent_payload(
        artifacts=artifacts,
        model_config=model_config,
    )
    codec = load_frozen_codec(
        checkpoint=artifacts.codec_path,
        expected_sha256=artifacts.codec_sha256,
        expected_seed=config.seed,
        device=device,
    )
    transition = PDERefinerMaskedLatentTransition(
        context_frames=1,
        config=model_config,
        refiner_config=refiner_config,
    ).to(device)
    audit = transition.load_deterministic_state(payload["transition_state"])
    model = C5PPDERefinerOneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor(normalization["mean"]),
        latent_standard_deviation=torch.tensor(
            normalization["population_standard_deviation"]
        ),
    ).to(device)
    return model, audit, payload, normalization


def _preoptimization_parent_identity(
    *,
    model: C5PPDERefinerOneStepModel,
    parent_state: Mapping[str, Tensor],
    dataset: OneStepWindowDataset,
    device: torch.device,
    model_config: O2ViTConfig,
) -> dict[str, Any]:
    parent = MaskedLatentTransition(context_frames=1, config=model_config).to(device)
    parent.load_state_dict(parent_state, strict=True)
    parent.eval()
    model.eval()
    item = dataset[0]
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    with torch.inference_mode():
        standardized = model.encode_context(context)
        expected = parent(standardized).to("cpu", torch.float32)
        observed = model.transition(
            standardized,
            torch.zeros_like(standardized[:, -1]),
            0,
        ).to("cpu", torch.float32)
    difference = (expected - observed).abs()
    record = {
        "target_frame_index": int(item["target_frame_index"]),
        "bitwise_exact": bool(torch.equal(expected, observed)),
        "maximum_absolute_difference": float(difference.max()),
    }
    if not record["bitwise_exact"]:
        raise RuntimeError("B4 level-0 path differs from the deterministic parent")
    return record


def _seed_rows_for_targets(
    targets: Sequence[int],
    bank: np.ndarray,
) -> np.ndarray:
    expected = (126, 2, 3)
    if bank.shape != expected or bank.dtype != np.uint64:
        raise ValueError("B4 validation seed bank must be uint64 [126,2,3]")
    rows = [int(target) - B4_VALIDATION_TARGET_START for target in targets]
    if any(row < 0 or row >= bank.shape[0] for row in rows):
        raise ValueError("B4 validation target leaves seed-bank interval")
    return np.ascontiguousarray(bank[np.asarray(rows, dtype=np.int64)])


def validation_decoded_mae(
    *,
    model: C5PPDERefinerOneStepModel,
    dataset: OneStepWindowDataset,
    targets: Sequence[int],
    seed_bank: np.ndarray,
    device: torch.device,
    collect_stages: bool = False,
) -> tuple[dict[str, Any], Tensor | None]:
    """Chronological M2 EMA validation; no physics metric is computed."""

    ordered_targets = tuple(int(item) for item in targets)
    rows = _seed_rows_for_targets(ordered_targets, seed_bank)
    loader = _window_loader(dataset, ordered_targets)
    fields = FAMILY_FIELDS["c5p"]
    stage_sums = torch.zeros(4, len(fields), dtype=torch.float64)
    observed: list[int] = []
    saved: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            target_frame = int(batch["target_frame_index"][0])
            if target_frame != ordered_targets[index]:
                raise RuntimeError("B4 validation target order differs")
            observed.append(target_frame)
            context = batch["context"].to(device, torch.float32)
            truth = batch["target"].to(device, torch.float32)
            noise = torch.from_numpy(refinement_noise_from_seeds(rows[index]))[
                None
            ].to(device, torch.float32)
            decoded = model.decoded_stages_with_noise(context, noise)
            if decoded.shape != (1, 2, 4, 5, 64, 32, 88):
                raise RuntimeError("B4 validation stage tensor axes differ")
            ensemble_mean = decoded.mean(dim=1)
            per_stage_channel = (ensemble_mean - truth[:, None]).abs().mean(
                dim=(0, 3, 4, 5)
            )
            stage_sums += per_stage_channel.to("cpu", torch.float64)
            if collect_stages:
                saved.append(decoded[0].to("cpu", torch.float32))
    if tuple(observed) != ordered_targets:
        raise RuntimeError("B4 validation did not consume all targets")
    averages = stage_sums / len(ordered_targets)
    if not torch.isfinite(averages).all():
        raise FloatingPointError("B4 validation decoded MAE is non-finite")
    final = averages[-1]
    record = {
        "ensemble_mean_equal_channel_decoded_standardized_field_MAE": float(
            final.mean()
        ),
        "final_MAE_by_channel": dict(zip(fields, map(float, final))),
        "equal_channel_MAE_by_level": [
            float(values.mean()) for values in averages
        ],
        "MAE_by_level_and_channel": [
            dict(zip(fields, map(float, values))) for values in averages
        ],
        "target_count": len(ordered_targets),
        "ensemble_members": B4_VALIDATION_MEMBERS,
        "refinement_levels": [0, 1, 2, 3],
        "checkpoint_weights": "EMA",
        "physics_metrics_used": False,
    }
    stages = torch.stack(saved, dim=0) if collect_stages else None
    return record, stages


def _group_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    squares = [
        parameter.grad.detach().to(torch.float32).square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return torch.tensor(float("nan"))
    return torch.stack(squares).sum().sqrt()


def _fixed_latent_probe(
    *,
    model: C5PPDERefinerOneStepModel,
    dataset: OneStepWindowDataset,
    seed_bank: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    item = dataset[0]
    target_frame = int(item["target_frame_index"])
    row = target_frame - B4_VALIDATION_TARGET_START
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    noise = torch.from_numpy(refinement_noise_from_seeds(seed_bank[row]))[
        None
    ].to(device, torch.float32)
    model.eval()
    with torch.inference_mode():
        latent = model.standardized_latent_stages(context, noise).to(
            "cpu", torch.float32
        )
        final = model.predict_with_noise(context, noise).to("cpu", torch.float32)
    return {
        "target_frame_index": target_frame,
        "refinement_noise": noise.to("cpu", torch.float32),
        "standardized_latent_stages": latent,
        "final_forecast": final,
    }


def _checkpoint_payload(
    *,
    transition_state: Mapping[str, Tensor],
    config: PDERefinerSmokeConfig,
    model_config: O2ViTConfig,
    refiner_config: PDERefinerConfig,
    artifacts: RefinerParentArtifacts,
    normalization: Mapping[str, Any],
    load_audit: RefinerLoadAudit,
    parameter_groups: RefinerParameterGroups,
    validation_seed_bank_path: Path,
    validation_seed_bank_sha256: str,
    training_levels_path: Path,
    training_levels_sha256: str,
    epoch: int,
    global_step: int,
    validation: Mapping[str, Any],
    ema_updates: int,
    paper0_commit: str,
    selected: bool,
    reload_probe: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    raw_transition_state: Mapping[str, Tensor] | None = None,
    training_noise_generator_state: Tensor | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": (
            "selected_B4_PDE_Refiner_transition"
            if selected
            else "final_B4_PDE_Refiner_smoke_state"
        ),
        "paper0_commit": str(paper0_commit),
        "config": config.to_record(),
        "model_config": model_config.to_record(),
        "refiner_config": refiner_config.to_record(),
        "parameter_groups": parameter_groups.to_record(),
        "deterministic_parent": {
            "path": str(artifacts.checkpoint_path),
            "sha256": artifacts.checkpoint_sha256,
            "load_audit": load_audit.to_record(),
        },
        "codec_checkpoint": {
            "path": str(artifacts.codec_path),
            "sha256": artifacts.codec_sha256,
            "trainable": False,
        },
        "latent_normalization": dict(normalization),
        "latent_normalization_source": {
            "path": str(artifacts.latent_normalization_path),
            "sha256": artifacts.latent_normalization_sha256,
            "refit": False,
        },
        "validation_seed_bank": {
            "path": str(validation_seed_bank_path),
            "sha256": validation_seed_bank_sha256,
            "seed": B4_VALIDATION_SEED_BANK_SEED,
            "shape": [126, 2, 3],
            "dtype": "uint64",
        },
        "training_levels": {
            "path": str(training_levels_path),
            "sha256": training_levels_sha256,
            "seed": B4_TRAINING_LEVEL_SEED,
            "shape": [2, 16],
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation": dict(validation),
        "EMA_decay": config.ema_decay,
        "EMA_updates": int(ema_updates),
        "transition_state": dict(transition_state),
    }
    if reload_probe is not None:
        payload["reload_probe"] = dict(reload_probe)
    if optimizer_state is not None:
        payload["optimizer_state"] = dict(optimizer_state)
    if raw_transition_state is not None:
        payload["raw_transition_state"] = dict(raw_transition_state)
    if training_noise_generator_state is not None:
        payload["training_noise_generator_state"] = (
            training_noise_generator_state
        )
    return payload


def _reload_selected_model(
    *,
    selected_checkpoint: Path,
    artifacts: RefinerParentArtifacts,
    config: PDERefinerSmokeConfig,
    model_config: O2ViTConfig,
    refiner_config: PDERefinerConfig,
    device: torch.device,
) -> C5PPDERefinerOneStepModel:
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "selected_B4_PDE_Refiner_transition":
        raise ValueError("B4 selected checkpoint kind differs")
    codec = load_frozen_codec(
        checkpoint=artifacts.codec_path,
        expected_sha256=artifacts.codec_sha256,
        expected_seed=config.seed,
        device=device,
    )
    transition = PDERefinerMaskedLatentTransition(
        context_frames=1,
        config=model_config,
        refiner_config=refiner_config,
    ).to(device)
    transition.load_state_dict(payload["transition_state"], strict=True)
    normalization = payload["latent_normalization"]
    return C5PPDERefinerOneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor(normalization["mean"]),
        latent_standard_deviation=torch.tensor(
            normalization["population_standard_deviation"]
        ),
    ).to(device).eval()


def _reload_probe_and_stage_artifact(
    *,
    selected_checkpoint: Path,
    artifacts: RefinerParentArtifacts,
    config: PDERefinerSmokeConfig,
    model_config: O2ViTConfig,
    refiner_config: PDERefinerConfig,
    validation_dataset: OneStepWindowDataset,
    seed_bank: np.ndarray,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    restored = _reload_selected_model(
        selected_checkpoint=selected_checkpoint,
        artifacts=artifacts,
        config=config,
        model_config=model_config,
        refiner_config=refiner_config,
        device=device,
    )
    expected_probe = payload["reload_probe"]
    observed_probe = _fixed_latent_probe(
        model=restored,
        dataset=validation_dataset,
        seed_bank=seed_bank,
        device=device,
    )
    latent_exact = bool(
        torch.equal(
            expected_probe["standardized_latent_stages"],
            observed_probe["standardized_latent_stages"],
        )
    )
    forecast_exact = bool(
        torch.equal(
            expected_probe["final_forecast"],
            observed_probe["final_forecast"],
        )
    )
    validation, stages = validation_decoded_mae(
        model=restored,
        dataset=validation_dataset,
        targets=config.validation_targets,
        seed_bank=seed_bank,
        device=device,
        collect_stages=True,
    )
    if stages is None or stages.shape != (4, 2, 4, 5, 64, 32, 88):
        raise RuntimeError("B4 collected stage artifact schema differs")
    level0_shared = bool(torch.equal(stages[:, 0, 0], stages[:, 1, 0]))
    final_rms_by_field = {
        field: float(
            (stages[:, 0, -1, index] - stages[:, 1, -1, index])
            .square()
            .mean()
            .sqrt()
        )
        for index, field in enumerate(FAMILY_FIELDS["c5p"])
    }
    nonzero_every_field = all(value > 0.0 for value in final_rms_by_field.values())
    finite = bool(torch.isfinite(stages).all())
    stage_path = output / "validation_decoded_stages.pt"
    save_torch_atomic(
        stage_path,
        {
            "schema_version": 1,
            "kind": "bounded_non_scientific_B4_validation_decoded_stages",
            "target_frames": list(config.validation_targets),
            "axes": [
                "target",
                "ensemble_member",
                "refinement_level",
                "channel",
                "x",
                "y",
                "z",
            ],
            "fields": list(FAMILY_FIELDS["c5p"]),
            "stages": stages,
            "scientific_result": False,
            "held_out_85606_read": False,
        },
    )
    probe = {
        "target_frame_index": int(observed_probe["target_frame_index"]),
        "canonical_stage_shape": list(stages.shape),
        "canonical_final_forecast_shape": [4, 2, 1, 5, 64, 32, 88],
        "reload_latent_bitwise_exact": latent_exact,
        "reload_forecast_bitwise_exact": forecast_exact,
        "level0_shared_bitwise_across_members": level0_shared,
        "finite": finite,
        "final_member_RMS_difference_by_field": final_rms_by_field,
        "nonzero_final_diversity_in_every_field": nonzero_every_field,
    }
    if not all(
        (
            latent_exact,
            forecast_exact,
            level0_shared,
            finite,
            nonzero_every_field,
        )
    ):
        raise RuntimeError(f"B4 selected-checkpoint smoke probe failed: {probe}")
    return probe, {
        "path": str(stage_path),
        "sha256": sha256_path(stage_path),
        "validation": validation,
    }


def _train_pde_refiner_smoke(
    *,
    config: PDERefinerSmokeConfig,
    catalog: ModelDatasetCatalog,
    artifacts: RefinerParentArtifacts,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
    model_config: O2ViTConfig = O2ViTConfig(),
    refiner_config: PDERefinerConfig = PDERefinerConfig(),
) -> dict[str, Any]:
    """Private engine reached only through the frozen smoke wrapper."""

    frozen = PDERefinerSmokeConfig.frozen(seed=B4_SMOKE_SEED)
    if config != frozen:
        raise ValueError("B4 config differs from the frozen bounded smoke")
    output = Path(output_directory)
    if "85606" in str(output).lower():
        raise ValueError("held-out paths are prohibited")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B4 run {output}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the bounded B4 smoke requires one CUDA worker")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("B4 requires highest float32 matmul precision")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("B4 prohibits TF32")
    output.mkdir(parents=True)

    seed_everything(config.seed)
    torch.set_float32_matmul_precision("highest")
    model, load_audit, parent_payload, normalization = _build_model(
        config=config,
        artifacts=artifacts,
        device=device,
        model_config=model_config,
        refiner_config=refiner_config,
    )
    codec_initial_digest = _module_state_sha256(model.codec)
    shutil.copyfile(
        artifacts.latent_normalization_path,
        output / "latent_normalization.json",
    )
    if sha256_path(output / "latent_normalization.json") != (
        artifacts.latent_normalization_sha256
    ):
        raise RuntimeError("copied B4 latent normalization changed bytes")

    level_values = smoke_training_levels(config)
    level_path = output / "training_levels.npy"
    level_sha256 = save_numpy_exclusive(level_path, level_values)
    seed_values = validation_seed_bank(seed=config.validation_seed_bank_seed)
    seed_path = output / "validation_seed_bank.npy"
    seed_sha256 = save_numpy_exclusive(seed_path, seed_values)
    if not np.array_equal(np.load(level_path, allow_pickle=False), level_values):
        raise RuntimeError("saved B4 training levels changed values")
    if not np.array_equal(np.load(seed_path, allow_pickle=False), seed_values):
        raise RuntimeError("saved B4 validation seeds changed values")

    train_dataset = OneStepWindowDataset(
        catalog,
        split="train",
        target_frames=config.train_targets,
        context_frames=1,
        augment=True,
        seed=config.seed,
    )
    validation_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=config.validation_targets,
        context_frames=1,
        augment=False,
        seed=config.seed,
    )
    parent_identity = _preoptimization_parent_identity(
        model=model,
        parent_state=parent_payload["transition_state"],
        dataset=validation_dataset,
        device=device,
        model_config=model_config,
    )

    groups = _parameter_groups(model.transition)
    optimizer = AdamW(
        groups.all_parameters,
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    ema = TransitionEMA(model.transition, decay=config.ema_decay)
    training_noise_generator = torch.Generator(device=device)
    training_noise_generator.manual_seed(config.training_noise_seed)

    run_record = {
        **config.to_record(),
        "model": model_config.to_record(),
        "pde_refiner": refiner_config.to_record(),
        "parameter_groups": groups.to_record(),
        "deterministic_parent": {
            "path": str(artifacts.checkpoint_path),
            "sha256": artifacts.checkpoint_sha256,
            "load_audit": load_audit.to_record(),
            "preoptimization_identity": parent_identity,
        },
        "codec_checkpoint": {
            "path": str(artifacts.codec_path),
            "sha256": artifacts.codec_sha256,
            "trainable": False,
        },
        "latent_normalization": {
            "path": str(artifacts.latent_normalization_path),
            "sha256": artifacts.latent_normalization_sha256,
            "refit": False,
        },
        "training_levels": {
            "seed": config.training_level_seed,
            "shape": list(level_values.shape),
            "counts": np.bincount(level_values.reshape(-1), minlength=4).tolist(),
            "sha256": level_sha256,
        },
        "validation_seed_bank": {
            "seed": config.validation_seed_bank_seed,
            "shape": list(seed_values.shape),
            "dtype": str(seed_values.dtype),
            "sha256": seed_sha256,
        },
    }
    write_strict_json_atomic(output / "config.json", run_record)

    history_path = output / "history.jsonl"
    history_handle = history_path.open("x", encoding="utf-8", buffering=1)
    selected_path = output / "selected.pt"
    final_path = output / "final_training_state.pt"
    selected_epoch: int | None = None
    selected_step: int | None = None
    selected_validation: dict[str, Any] | None = None
    selected_state: dict[str, Tensor] | None = None
    selected_probe: dict[str, Any] | None = None
    global_step = 0
    parent_gradient_seen = False
    refinement_gradient_seen = False
    observed_level_counts = np.zeros(4, dtype=np.int64)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(config.epochs):
            epoch_started = time.monotonic()
            train_dataset.set_epoch(epoch)
            order = epoch_order(config.train_targets, seed=config.seed, epoch=epoch)
            loader = _window_loader(train_dataset, order)
            model.train()
            train_loss_sum = 0.0
            train_loss_by_level = np.zeros(4, dtype=np.float64)
            train_count_by_level = np.zeros(4, dtype=np.int64)
            examples = 0
            accumulation_count = 0
            total_gradient_norms: list[float] = []
            parent_gradient_norms: list[float] = []
            refinement_gradient_norms: list[float] = []

            for microstep, batch in enumerate(loader, start=1):
                level = int(level_values[epoch, microstep - 1])
                context = batch["context"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                target = batch["target"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                latent_shape = (context.shape[0], *B4_LATENT_SHAPE)
                if latent_shape != (1, *B4_LATENT_SHAPE):
                    raise RuntimeError("B4 training latent shape differs")
                if level == 0:
                    noise = torch.zeros(latent_shape, device=device, dtype=torch.float32)
                else:
                    noise = torch.randn(
                        latent_shape,
                        generator=training_noise_generator,
                        device=device,
                        dtype=torch.float32,
                    )
                levels = torch.full(
                    (context.shape[0],),
                    level,
                    device=device,
                    dtype=torch.int64,
                )
                objective = refinement_objective(
                    model=model,
                    context=context,
                    target=target,
                    levels=levels,
                    noise=noise,
                )
                loss = objective.loss
                if loss.dtype != torch.float32 or not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"invalid B4 loss at epoch {epoch}, microstep {microstep}"
                    )
                loss.backward()
                value = float(loss.detach())
                train_loss_sum += value
                train_loss_by_level[level] += value
                train_count_by_level[level] += 1
                observed_level_counts[level] += 1
                accumulation_count += 1
                examples += 1

                step_due = (
                    accumulation_count == config.gradient_accumulation_targets
                )
                final_microstep = microstep == len(config.train_targets)
                if step_due or final_microstep:
                    scale_accumulated_gradients(
                        groups.all_parameters,
                        accumulation_count,
                    )
                    parent_norm = _group_gradient_norm(groups.parent).to(device)
                    refinement_norm = _group_gradient_norm(groups.refinement).to(
                        device
                    )
                    if not torch.isfinite(parent_norm) or parent_norm <= 0:
                        raise FloatingPointError("invalid B4 parent gradient")
                    if not torch.isfinite(refinement_norm) or refinement_norm <= 0:
                        raise FloatingPointError("invalid B4 refinement gradient")
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        groups.all_parameters,
                        config.gradient_clip,
                    )
                    if not torch.isfinite(total_norm):
                        raise FloatingPointError("non-finite B4 total gradient norm")
                    optimizer.step()
                    ema.update(model.transition)
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    parent_gradient_seen = True
                    refinement_gradient_seen = True
                    parent_gradient_norms.append(float(parent_norm))
                    refinement_gradient_norms.append(float(refinement_norm))
                    total_gradient_norms.append(float(total_norm))
                    accumulation_count = 0

            if accumulation_count != 0:
                raise RuntimeError("B4 epoch left unstepped accumulated gradients")
            if examples != len(config.train_targets):
                raise RuntimeError("B4 epoch did not consume every target once")
            if np.any(train_count_by_level == 0):
                raise RuntimeError("B4 smoke epoch omitted a refinement level")
            if global_step != (epoch + 1) * config.optimizer_steps_per_epoch:
                raise RuntimeError("B4 optimizer-step count differs")
            if ema.updates != global_step:
                raise RuntimeError("B4 EMA update count differs")

            with ema.applied_to(model.transition):
                validation, _ = validation_decoded_mae(
                    model=model,
                    dataset=validation_dataset,
                    targets=config.validation_targets,
                    seed_bank=seed_values,
                    device=device,
                )
                candidate = validation[
                    "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
                ]
                incumbent = (
                    math.inf
                    if selected_validation is None
                    else selected_validation[
                        "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
                    ]
                )
                if candidate < incumbent:
                    selected_epoch = epoch
                    selected_step = global_step
                    selected_validation = dict(validation)
                    selected_state = ema.cpu_state_dict()
                    selected_probe = _fixed_latent_probe(
                        model=model,
                        dataset=validation_dataset,
                        seed_bank=seed_values,
                        device=device,
                    )

            epoch_record = {
                "epoch": epoch,
                "examples": examples,
                "global_step": global_step,
                "learning_rate": config.learning_rate,
                "EMA_decay": config.ema_decay,
                "EMA_updates": ema.updates,
                "train_standardized_latent_MSE": train_loss_sum / examples,
                "train_MSE_by_level": {
                    str(level): float(train_loss_by_level[level] / train_count_by_level[level])
                    for level in range(4)
                },
                "train_count_by_level": {
                    str(level): int(train_count_by_level[level])
                    for level in range(4)
                },
                "validation_ensemble_mean_equal_channel_decoded_standardized_field_MAE": (
                    validation[
                        "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
                    ]
                ),
                "validation_final_MAE_by_channel": validation[
                    "final_MAE_by_channel"
                ],
                "validation_equal_channel_MAE_by_level": validation[
                    "equal_channel_MAE_by_level"
                ],
                "mean_preclip_total_gradient_norm": float(
                    np.mean(total_gradient_norms)
                ),
                "maximum_preclip_total_gradient_norm": float(
                    np.max(total_gradient_norms)
                ),
                "mean_preclip_parent_gradient_norm": float(
                    np.mean(parent_gradient_norms)
                ),
                "mean_preclip_refinement_gradient_norm": float(
                    np.mean(refinement_gradient_norms)
                ),
                "selected_so_far": selected_epoch,
                "epoch_wall_seconds": time.monotonic() - epoch_started,
            }
            history_handle.write(
                json.dumps(epoch_record, sort_keys=True, allow_nan=False) + "\n"
            )
            print(json.dumps(epoch_record, sort_keys=True, allow_nan=False), flush=True)
            if epoch_callback is not None:
                epoch_callback(epoch_record)
    finally:
        history_handle.close()
        train_dataset.close()

    if (
        selected_epoch is None
        or selected_step is None
        or selected_validation is None
        or selected_state is None
        or selected_probe is None
    ):
        validation_dataset.close()
        raise RuntimeError("B4 smoke completed without a selected checkpoint")
    with ema.applied_to(model.transition):
        final_validation, _ = validation_decoded_mae(
            model=model,
            dataset=validation_dataset,
            targets=config.validation_targets,
            seed_bank=seed_values,
            device=device,
        )

    save_torch_atomic(
        selected_path,
        _checkpoint_payload(
            transition_state=selected_state,
            config=config,
            model_config=model_config,
            refiner_config=refiner_config,
            artifacts=artifacts,
            normalization=normalization,
            load_audit=load_audit,
            parameter_groups=groups,
            validation_seed_bank_path=seed_path,
            validation_seed_bank_sha256=seed_sha256,
            training_levels_path=level_path,
            training_levels_sha256=level_sha256,
            epoch=selected_epoch,
            global_step=selected_step,
            validation=selected_validation,
            ema_updates=selected_step,
            paper0_commit=paper0_commit,
            selected=True,
            reload_probe=selected_probe,
        ),
    )
    save_torch_atomic(
        final_path,
        _checkpoint_payload(
            transition_state=ema.cpu_state_dict(),
            raw_transition_state=model.transition.state_dict(),
            config=config,
            model_config=model_config,
            refiner_config=refiner_config,
            artifacts=artifacts,
            normalization=normalization,
            load_audit=load_audit,
            parameter_groups=groups,
            validation_seed_bank_path=seed_path,
            validation_seed_bank_sha256=seed_sha256,
            training_levels_path=level_path,
            training_levels_sha256=level_sha256,
            epoch=config.epochs - 1,
            global_step=global_step,
            validation=final_validation,
            ema_updates=ema.updates,
            paper0_commit=paper0_commit,
            selected=False,
            optimizer_state=optimizer.state_dict(),
            training_noise_generator_state=training_noise_generator.get_state(),
        ),
    )
    del selected_state, selected_probe

    reload_probe, stages_artifact = _reload_probe_and_stage_artifact(
        selected_checkpoint=selected_path,
        artifacts=artifacts,
        config=config,
        model_config=model_config,
        refiner_config=refiner_config,
        validation_dataset=validation_dataset,
        seed_bank=seed_values,
        output=output,
        device=device,
    )
    validation_dataset.close()
    codec_final_digest = _module_state_sha256(model.codec)
    codec_unchanged = codec_initial_digest == codec_final_digest
    if not codec_unchanged:
        raise RuntimeError("frozen B4 codec changed during optimization")
    if not parent_gradient_seen or not refinement_gradient_seen:
        raise RuntimeError("B4 smoke did not exercise both parameter families")
    expected_counts = np.bincount(level_values.reshape(-1), minlength=4)
    if not np.array_equal(observed_level_counts, expected_counts):
        raise RuntimeError("B4 observed training-level counts differ")
    torch.cuda.synchronize(device)

    result = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_B4_PDE_Refiner_H1_GPU_smoke_85604",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "config": run_record,
        "parameter_count": groups.to_record()["total_parameter_count"],
        "parameter_groups": groups.to_record(),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_step,
        "EMA_updates": ema.updates,
        "selected_epoch": selected_epoch,
        "selected_optimizer_step": selected_step,
        "selected_validation": selected_validation,
        "final_validation": final_validation,
        "preoptimization_parent_identity": parent_identity,
        "deterministic_parent_load_audit": load_audit.to_record(),
        "checkpoint_reload_bitwise_exact": bool(
            reload_probe["reload_latent_bitwise_exact"]
            and reload_probe["reload_forecast_bitwise_exact"]
        ),
        "member_and_stage_probe": reload_probe,
        "parent_parameter_gradient_seen": parent_gradient_seen,
        "refinement_parameter_gradient_seen": refinement_gradient_seen,
        "training_level_counts": observed_level_counts.tolist(),
        "all_four_training_levels_exercised": bool(
            np.all(observed_level_counts > 0)
        ),
        "codec_state_sha256_before": codec_initial_digest,
        "codec_state_sha256_after": codec_final_digest,
        "codec_bitwise_unchanged": codec_unchanged,
        "selected_checkpoint": {
            "path": str(selected_path),
            "sha256": sha256_path(selected_path),
        },
        "final_training_state": {
            "path": str(final_path),
            "sha256": sha256_path(final_path),
        },
        "history": {
            "path": str(history_path),
            "sha256": sha256_path(history_path),
        },
        "validation_seed_bank": {
            "path": str(seed_path),
            "sha256": seed_sha256,
            "seed": config.validation_seed_bank_seed,
            "shape": list(seed_values.shape),
        },
        "training_levels": {
            "path": str(level_path),
            "sha256": level_sha256,
            "seed": config.training_level_seed,
            "shape": list(level_values.shape),
        },
        "validation_decoded_stages": stages_artifact,
        "latent_normalization": {
            "path": str(output / "latent_normalization.json"),
            "sha256": sha256_path(output / "latent_normalization.json"),
            "refit": False,
        },
        "deterministic_parent": {
            "path": str(artifacts.checkpoint_path),
            "sha256": artifacts.checkpoint_sha256,
        },
        "codec_checkpoint": {
            "path": str(artifacts.codec_path),
            "sha256": artifacts.codec_sha256,
            "trainable": False,
        },
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "network_calls_per_member": 4,
        "training_dtype": "float32",
        "validation_dtype": "float32",
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "strict_cuda_bitwise_determinism_claimed": False,
        "cudnn_deterministic_requested": True,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "simulation_data_read": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "full_B4_training_authorized": False,
        "scientific_B4_evaluation_authorized": False,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    write_strict_json_atomic(output / "result.json", result)
    return result


def train_pde_refiner_smoke(
    *,
    config: PDERefinerSmokeConfig,
    catalog: ModelDatasetCatalog,
    artifacts: RefinerParentArtifacts,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
    model_config: O2ViTConfig = O2ViTConfig(),
    refiner_config: PDERefinerConfig = PDERefinerConfig(),
) -> dict[str, Any]:
    """Execute only the bounded, explicitly non-scientific B4 GPU smoke."""

    frozen = PDERefinerSmokeConfig.frozen(seed=B4_SMOKE_SEED)
    if config != frozen:
        raise ValueError("B4 smoke config differs from the frozen bounded budget")
    return _train_pde_refiner_smoke(
        config=config,
        catalog=catalog,
        artifacts=artifacts,
        output_directory=output_directory,
        paper0_commit=paper0_commit,
        slurm_job_id=slurm_job_id,
        device=device,
        epoch_callback=epoch_callback,
        model_config=model_config,
        refiner_config=refiner_config,
    )
