"""Frozen 85604-only mechanics for the bounded B3 FGN implementation smoke.

This module deliberately exposes no full-training public entrypoint. The
prospective full budget is represented for regression testing, but remains
unauthorized until the bounded smoke passes and a later protocol says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .codec_training import save_torch_atomic, seed_everything, sha256_path
from .model_data import load_strict_json, write_strict_json_atomic
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog, epoch_order
from .models.functional_noise import (
    C5PFunctionalNoiseOneStepModel,
    DeterministicLoadAudit,
    FairCRPSResult,
    FunctionalNoiseConfig,
    FunctionalNoiseMaskedLatentTransition,
    fair_crps,
)
from .models.o2 import MaskedLatentTransition, O2ViTConfig
from .o2_training import load_frozen_codec, scale_accumulated_gradients
from .o2_training_data import OneStepWindowDataset


FGN_SMOKE_SEED = 1701
FGN_VALIDATION_NOISE_SEED = 31_003
FGN_VALIDATION_TARGET_START = 498
FGN_VALIDATION_TARGET_STOP = 624
FGN_VALIDATION_MEMBER_COUNT = 2


@dataclass(frozen=True)
class FGNRunConfig:
    """Immutable bounded-smoke or prospective full B3 budget."""

    mode: str
    seed: int
    epochs: int
    train_target_start: int
    train_target_stop: int
    validation_target_start: int
    validation_target_stop: int
    microbatch_targets: int
    gradient_accumulation_targets: int
    ensemble_members: int
    common_peak_learning_rate: float
    new_peak_learning_rate: float
    warmup_epochs: int
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    training_precision: str
    validation_noise_seed: int

    @classmethod
    def frozen(cls, *, mode: str, seed: int) -> "FGNRunConfig":
        if int(seed) != FGN_SMOKE_SEED:
            raise ValueError("the first B3 pilot is prospectively fixed to seed 1701")
        common = {
            "mode": str(mode),
            "seed": int(seed),
            "train_target_start": 2,
            "validation_target_start": FGN_VALIDATION_TARGET_START,
            "microbatch_targets": 1,
            "gradient_accumulation_targets": 16,
            "ensemble_members": 2,
            "common_peak_learning_rate": 3.0e-5,
            "new_peak_learning_rate": 1.0e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "training_precision": "bfloat16_autocast",
            "validation_noise_seed": FGN_VALIDATION_NOISE_SEED,
        }
        if mode == "smoke":
            return cls(
                **common,
                epochs=2,
                train_target_stop=18,
                validation_target_stop=502,
                warmup_epochs=0,
            )
        if mode == "full":
            return cls(
                **common,
                epochs=100,
                train_target_stop=432,
                validation_target_stop=FGN_VALIDATION_TARGET_STOP,
                warmup_epochs=10,
            )
        raise ValueError(f"unsupported B3 mode {mode!r}")

    @property
    def context_frames(self) -> int:
        return 1

    @property
    def train_targets(self) -> tuple[int, ...]:
        return tuple(range(self.train_target_start, self.train_target_stop))

    @property
    def validation_targets(self) -> tuple[int, ...]:
        return tuple(
            range(self.validation_target_start, self.validation_target_stop)
        )

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(
            len(self.train_targets) / self.gradient_accumulation_targets
        )

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    @property
    def warmup_optimizer_steps(self) -> int:
        return self.warmup_epochs * self.optimizer_steps_per_epoch

    @property
    def final_accumulation_count(self) -> int:
        remainder = len(self.train_targets) % self.gradient_accumulation_targets
        return remainder if remainder else self.gradient_accumulation_targets

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "context_frames": self.context_frames,
                "fields": list(FAMILY_FIELDS["c5p"]),
                "train_targets": [self.train_target_start, self.train_target_stop],
                "validation_targets": [
                    self.validation_target_start,
                    self.validation_target_stop,
                ],
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "warmup_optimizer_steps": self.warmup_optimizer_steps,
                "final_accumulation_count": self.final_accumulation_count,
                "training_loss": (
                    "equal_channel_decoded_standardized_field_fair_CRPS"
                ),
                "scheduler": (
                    "constant_bounded_smoke"
                    if self.mode == "smoke"
                    else "independent_linear_warmup_cosine_to_zero"
                ),
                "physics_derived_loss_allowed": False,
                "absolute_time_input_allowed": False,
                "early_stopping": False,
                "scientific_result": False,
                "prospective_full_budget": self.mode == "full",
                "full_training_authorized": False,
            }
        )
        return record


@dataclass(frozen=True)
class ParentArtifacts:
    checkpoint_path: Path
    checkpoint_sha256: str
    codec_path: Path
    codec_sha256: str
    latent_normalization_path: Path
    latent_normalization_sha256: str


@dataclass(frozen=True)
class ParameterGroups:
    common: tuple[nn.Parameter, ...]
    new: tuple[nn.Parameter, ...]
    common_names: tuple[str, ...]
    new_names: tuple[str, ...]

    @property
    def all_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.common + self.new

    def to_record(self) -> dict[str, Any]:
        return {
            "common_parameter_tensor_count": len(self.common),
            "new_parameter_tensor_count": len(self.new),
            "common_parameter_count": sum(item.numel() for item in self.common),
            "new_parameter_count": sum(item.numel() for item in self.new),
            "total_parameter_count": sum(
                item.numel() for item in self.all_parameters
            ),
            "common_parameter_names": list(self.common_names),
            "new_parameter_names": list(self.new_names),
        }


def learning_rate_at_step(
    config: FGNRunConfig,
    step: int,
    *,
    group: str,
) -> float:
    """One-indexed staged learning-rate schedule."""

    index = int(step)
    if not 1 <= index <= config.total_optimizer_steps:
        raise ValueError(
            f"optimizer step {index} is outside 1..{config.total_optimizer_steps}"
        )
    if group == "common":
        peak = config.common_peak_learning_rate
    elif group == "new":
        peak = config.new_peak_learning_rate
    else:
        raise ValueError(f"unknown B3 parameter group {group!r}")
    if config.mode == "smoke":
        return float(peak)
    warmup = config.warmup_optimizer_steps
    if warmup and index <= warmup:
        return float(peak * index / warmup)
    if config.total_optimizer_steps == warmup:
        return float(peak)
    progress = (index - warmup) / (config.total_optimizer_steps - warmup)
    return float(peak * 0.5 * (1.0 + math.cos(math.pi * progress)))


def validation_noise_bank(
    *,
    seed: int = FGN_VALIDATION_NOISE_SEED,
    members: int = FGN_VALIDATION_MEMBER_COUNT,
    raw_features: int = 32,
) -> np.ndarray:
    """Generate the complete immutable chronological validation bank."""

    if int(seed) != FGN_VALIDATION_NOISE_SEED:
        raise ValueError("B3 validation-noise seed differs from the frozen seed")
    if int(members) != FGN_VALIDATION_MEMBER_COUNT:
        raise ValueError("B3 validation uses exactly two members during training")
    if int(raw_features) != 32:
        raise ValueError("B3 raw validation noise must have 32 features")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    values = generator.standard_normal(
        (
            FGN_VALIDATION_TARGET_STOP - FGN_VALIDATION_TARGET_START,
            int(members),
            int(raw_features),
        ),
        dtype=np.float32,
    )
    return np.ascontiguousarray(values, dtype=np.float32)


def save_validation_noise_bank(path: Path, values: np.ndarray) -> str:
    """Persist one immutable NPY bank and return its SHA-256."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    with destination.open("xb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
    return sha256_path(destination)


def _window_loader(
    dataset: OneStepWindowDataset,
    ordered_targets: Sequence[int],
) -> DataLoader:
    start = dataset.target_frames[0]
    indices = [int(target) - start for target in ordered_targets]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("B3 batch order contains a target outside the dataset")
    return DataLoader(dataset, batch_size=1, sampler=indices, num_workers=0)


def _module_state_sha256(module: nn.Module) -> str:
    """Hash tensor names, shapes, dtypes, and values for an in-memory module."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _verify_parent_payload(
    *,
    artifacts: ParentArtifacts,
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
        raise ValueError("deterministic parent checkpoint SHA-256 mismatch")
    if sha256_path(artifacts.codec_path) != artifacts.codec_sha256:
        raise ValueError("B3 codec checkpoint SHA-256 mismatch")
    if (
        sha256_path(artifacts.latent_normalization_path)
        != artifacts.latent_normalization_sha256
    ):
        raise ValueError("B3 latent-normalization SHA-256 mismatch")

    payload = torch.load(
        artifacts.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("kind") != "selected_O2_transition":
        raise ValueError("B3 parent is not a selected O2 transition")
    config = payload.get("config", {})
    if config.get("arm") != "C5P-H1" or int(config.get("seed", -1)) != 1701:
        raise ValueError("B3 parent is not the frozen C5P-H1 seed-1701 model")
    if int(payload.get("epoch", -1)) != 193:
        raise ValueError("B3 parent selected epoch differs")
    if int(payload.get("global_step", -1)) != 5238:
        raise ValueError("B3 parent selected optimizer step differs")
    if float(payload.get("validation_loss", math.inf)) != 0.04558250684515488:
        raise ValueError("B3 parent selected validation loss differs")
    if payload.get("model_config") != model_config.to_record():
        raise ValueError("B3 parent model configuration differs")
    codec_record = payload.get("codec_checkpoint", {})
    if str(codec_record.get("path")) != str(artifacts.codec_path):
        raise ValueError("B3 parent codec path differs")
    if str(codec_record.get("sha256")) != artifacts.codec_sha256:
        raise ValueError("B3 parent codec hash differs")

    normalization = load_strict_json(artifacts.latent_normalization_path)
    if payload.get("latent_normalization") != normalization:
        raise ValueError("external and embedded latent normalization differ")
    if normalization.get("fit_frames") != [0, 432]:
        raise ValueError("B3 latent normalization was not fit on training only")
    if normalization.get("codec_checkpoint_sha256") != artifacts.codec_sha256:
        raise ValueError("B3 latent normalization belongs to another codec")
    if normalization.get("held_out_85606_read") is not False:
        raise ValueError("B3 latent normalization reports held-out access")
    return payload, normalization


def _parameter_groups(
    transition: FunctionalNoiseMaskedLatentTransition,
) -> ParameterGroups:
    common: list[nn.Parameter] = []
    new: list[nn.Parameter] = []
    common_names: list[str] = []
    new_names: list[str] = []
    for name, parameter in transition.named_parameters():
        if not parameter.requires_grad:
            raise ValueError(f"transition parameter {name} is unexpectedly frozen")
        if transition._is_noise_key(name):
            new.append(parameter)
            new_names.append(name)
        else:
            common.append(parameter)
            common_names.append(name)
    if not common or not new:
        raise RuntimeError("B3 staged optimizer has an empty parameter group")
    if {id(item) for item in common} & {id(item) for item in new}:
        raise RuntimeError("B3 staged optimizer parameter groups overlap")
    if {id(item) for item in common + new} != {
        id(item) for item in transition.parameters()
    }:
        raise RuntimeError("B3 staged optimizer omits transition parameters")
    return ParameterGroups(
        common=tuple(common),
        new=tuple(new),
        common_names=tuple(common_names),
        new_names=tuple(new_names),
    )


def _build_model(
    *,
    config: FGNRunConfig,
    artifacts: ParentArtifacts,
    device: torch.device,
    model_config: O2ViTConfig,
    noise_config: FunctionalNoiseConfig,
) -> tuple[
    C5PFunctionalNoiseOneStepModel,
    DeterministicLoadAudit,
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
    transition = FunctionalNoiseMaskedLatentTransition(
        context_frames=config.context_frames,
        config=model_config,
        noise_config=noise_config,
    ).to(device)
    audit = transition.load_deterministic_state(payload["transition_state"])
    model = C5PFunctionalNoiseOneStepModel(
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
    model: C5PFunctionalNoiseOneStepModel,
    parent_state: Mapping[str, Tensor],
    dataset: OneStepWindowDataset,
    device: torch.device,
    model_config: O2ViTConfig,
) -> dict[str, Any]:
    parent = MaskedLatentTransition(
        context_frames=1,
        config=model_config,
    ).to(device)
    parent.load_state_dict(parent_state, strict=True)
    parent.eval()
    model.eval()
    item = dataset[0]
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    with torch.inference_mode():
        standardized = model.encode_context(context)
        expected = parent(standardized).to("cpu", torch.float32)
        observed = model.transition(standardized, raw_noise=None).to(
            "cpu", torch.float32
        )
    difference = (expected - observed).abs()
    record = {
        "target_frame_index": int(item["target_frame_index"]),
        "bitwise_exact": bool(torch.equal(expected, observed)),
        "maximum_absolute_difference": float(difference.max()),
    }
    if not record["bitwise_exact"]:
        raise RuntimeError("noise-disabled B3 transition differs from H1 parent")
    return record


def _bank_rows_for_targets(
    targets: Sequence[int],
    bank: np.ndarray,
) -> np.ndarray:
    expected = (
        FGN_VALIDATION_TARGET_STOP - FGN_VALIDATION_TARGET_START,
        FGN_VALIDATION_MEMBER_COUNT,
        32,
    )
    if bank.shape != expected or bank.dtype != np.float32:
        raise ValueError(f"validation noise bank must be float32 {expected}")
    rows = [int(target) - FGN_VALIDATION_TARGET_START for target in targets]
    if any(row < 0 or row >= bank.shape[0] for row in rows):
        raise ValueError("validation target leaves frozen noise-bank interval")
    return bank[np.asarray(rows, dtype=np.int64)]


def validation_fair_crps(
    *,
    model: C5PFunctionalNoiseOneStepModel,
    dataset: OneStepWindowDataset,
    targets: Sequence[int],
    bank: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Full chronological validation with the immutable CPU noise bank."""

    ordered_targets = tuple(int(item) for item in targets)
    rows = _bank_rows_for_targets(ordered_targets, bank)
    loader = _window_loader(dataset, ordered_targets)
    fields = FAMILY_FIELDS["c5p"]
    score_sum = torch.zeros(len(fields), dtype=torch.float64)
    accuracy_sum = torch.zeros(len(fields), dtype=torch.float64)
    spread_sum = torch.zeros(len(fields), dtype=torch.float64)
    observed_targets: list[int] = []
    model.eval()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            target_frame = int(batch["target_frame_index"][0])
            if target_frame != ordered_targets[index]:
                raise RuntimeError("B3 validation target order differs")
            observed_targets.append(target_frame)
            context = batch["context"].to(device, torch.float32)
            target = batch["target"].to(device, torch.float32)
            raw_noise = torch.from_numpy(rows[index])[None].to(
                device=device,
                dtype=torch.float32,
            )
            predictions = model.predict_with_noise(
                context,
                raw_noise,
                horizon=1,
            )[:, :, 0].to(torch.float32)
            result = fair_crps(predictions, target)
            score_sum += result.per_channel.to("cpu", torch.float64)
            accuracy_sum += result.accuracy_per_channel.to("cpu", torch.float64)
            spread_sum += result.spread_per_channel.to("cpu", torch.float64)
    if tuple(observed_targets) != ordered_targets:
        raise RuntimeError("B3 validation did not consume every target chronologically")
    count = len(ordered_targets)
    score = score_sum / count
    accuracy = accuracy_sum / count
    spread = spread_sum / count
    aggregate = score.mean()
    if not torch.isfinite(aggregate):
        raise FloatingPointError("B3 validation fair CRPS is non-finite")
    return {
        "equal_channel_fair_crps": float(aggregate),
        "fair_crps_by_channel": dict(zip(fields, map(float, score))),
        "accuracy_by_channel": dict(zip(fields, map(float, accuracy))),
        "spread_by_channel": dict(zip(fields, map(float, spread))),
        "target_count": count,
        "ensemble_members": FGN_VALIDATION_MEMBER_COUNT,
    }


def _group_gradient_norm(parameters: Iterable[nn.Parameter]) -> Tensor:
    squares: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().to(torch.float32).square().sum())
    if not squares:
        return torch.tensor(float("nan"))
    return torch.stack(squares).sum().sqrt()


def _fixed_probe(
    *,
    model: C5PFunctionalNoiseOneStepModel,
    dataset: OneStepWindowDataset,
    bank: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    item = dataset[0]
    target_frame = int(item["target_frame_index"])
    row = target_frame - FGN_VALIDATION_TARGET_START
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    raw_noise = torch.from_numpy(bank[row])[None].to(device, torch.float32)
    model.eval()
    with torch.inference_mode():
        latent = model.standardized_latent_members(context, raw_noise).to(
            "cpu", torch.float32
        )
        forecast = model.predict_with_noise(context, raw_noise).to(
            "cpu", torch.float32
        )
    return {
        "target_frame_index": target_frame,
        "raw_noise": raw_noise.to("cpu", torch.float32),
        "standardized_latent": latent,
        "forecast": forecast,
    }


def _checkpoint_payload(
    *,
    model: C5PFunctionalNoiseOneStepModel,
    optimizer: AdamW,
    parameter_groups: ParameterGroups,
    config: FGNRunConfig,
    model_config: O2ViTConfig,
    noise_config: FunctionalNoiseConfig,
    artifacts: ParentArtifacts,
    normalization: Mapping[str, Any],
    load_audit: DeterministicLoadAudit,
    validation_noise_path: Path,
    validation_noise_sha256: str,
    train_noise_generator: torch.Generator,
    epoch: int,
    global_step: int,
    validation: Mapping[str, Any],
    paper0_commit: str,
    selected: bool,
    include_optimizer: bool,
    transition_state: Mapping[str, Tensor] | None = None,
    reload_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": (
            "selected_B3_FGN_transition"
            if selected
            else "final_B3_FGN_training_state"
        ),
        "paper0_commit": str(paper0_commit),
        "config": config.to_record(),
        "model_config": model_config.to_record(),
        "noise_config": noise_config.to_record(),
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
        "validation_noise_bank": {
            "path": str(validation_noise_path),
            "sha256": str(validation_noise_sha256),
            "seed": FGN_VALIDATION_NOISE_SEED,
            "shape": [126, 2, 32],
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation": dict(validation),
        "transition_state": (
            model.transition.state_dict()
            if transition_state is None
            else dict(transition_state)
        ),
        "train_noise_generator_state": train_noise_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    if reload_probe is not None:
        payload["reload_probe"] = dict(reload_probe)
    return payload


def _reload_selected_model(
    *,
    selected_checkpoint: Path,
    artifacts: ParentArtifacts,
    config: FGNRunConfig,
    model_config: O2ViTConfig,
    noise_config: FunctionalNoiseConfig,
    device: torch.device,
) -> C5PFunctionalNoiseOneStepModel:
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "selected_B3_FGN_transition":
        raise ValueError("B3 selected checkpoint kind differs")
    codec = load_frozen_codec(
        checkpoint=artifacts.codec_path,
        expected_sha256=artifacts.codec_sha256,
        expected_seed=config.seed,
        device=device,
    )
    transition = FunctionalNoiseMaskedLatentTransition(
        context_frames=1,
        config=model_config,
        noise_config=noise_config,
    ).to(device)
    transition.load_state_dict(payload["transition_state"], strict=True)
    normalization = payload["latent_normalization"]
    return C5PFunctionalNoiseOneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor(normalization["mean"]),
        latent_standard_deviation=torch.tensor(
            normalization["population_standard_deviation"]
        ),
    ).to(device).eval()


def _reload_identity_and_diversity(
    *,
    selected_checkpoint: Path,
    artifacts: ParentArtifacts,
    config: FGNRunConfig,
    model_config: O2ViTConfig,
    noise_config: FunctionalNoiseConfig,
    validation_dataset: OneStepWindowDataset,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    restored = _reload_selected_model(
        selected_checkpoint=selected_checkpoint,
        artifacts=artifacts,
        config=config,
        model_config=model_config,
        noise_config=noise_config,
        device=device,
    )
    item = validation_dataset[0]
    probe = payload["reload_probe"]
    if int(item["target_frame_index"]) != int(probe["target_frame_index"]):
        raise RuntimeError("B3 checkpoint reload probe target differs")
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    raw_noise = probe["raw_noise"].to(device, torch.float32)
    with torch.inference_mode():
        latent = restored.standardized_latent_members(context, raw_noise).to(
            "cpu", torch.float32
        )
        forecast = restored.predict_with_noise(context, raw_noise).to(
            "cpu", torch.float32
        )
    latent_exact = bool(torch.equal(latent, probe["standardized_latent"]))
    forecast_exact = bool(torch.equal(forecast, probe["forecast"]))
    latent_rms = float((latent[:, 0] - latent[:, 1]).square().mean().sqrt())
    field_rms = float((forecast[:, 0] - forecast[:, 1]).square().mean().sqrt())
    record = {
        "target_frame_index": int(item["target_frame_index"]),
        "ensemble_size": 2,
        "canonical_forecast_shape": list(forecast.shape),
        "finite": bool(torch.isfinite(latent).all() and torch.isfinite(forecast).all()),
        "reload_latent_bitwise_exact": latent_exact,
        "reload_forecast_bitwise_exact": forecast_exact,
        "latent_member_rms_difference": latent_rms,
        "field_member_rms_difference": field_rms,
        "nonzero_latent_diversity": latent_rms > 0.0,
        "nonzero_field_diversity": field_rms > 0.0,
    }
    if not all(
        (
            record["finite"],
            latent_exact,
            forecast_exact,
            record["nonzero_latent_diversity"],
            record["nonzero_field_diversity"],
        )
    ):
        raise RuntimeError(f"B3 reload/diversity probe failed: {record}")
    return record


def train_fgn_smoke(
    *,
    config: FGNRunConfig,
    catalog: ModelDatasetCatalog,
    artifacts: ParentArtifacts,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
    model_config: O2ViTConfig = O2ViTConfig(),
    noise_config: FunctionalNoiseConfig = FunctionalNoiseConfig(),
) -> dict[str, Any]:
    """Execute only the bounded, explicitly non-scientific B3 smoke."""

    frozen = FGNRunConfig.frozen(mode="smoke", seed=FGN_SMOKE_SEED)
    if config != frozen:
        raise ValueError("B3 smoke config differs from the frozen bounded budget")
    if config.mode != "smoke":
        raise ValueError("this entrypoint cannot execute full B3 training")
    output = Path(output_directory)
    if "85606" in str(output).lower():
        raise ValueError("held-out paths are prohibited")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B3 run {output}")
    output.mkdir(parents=True)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B3 smoke execution requires one CUDA worker")

    seed_everything(config.seed)
    model, load_audit, parent_payload, normalization = _build_model(
        config=config,
        artifacts=artifacts,
        device=device,
        model_config=model_config,
        noise_config=noise_config,
    )
    codec_initial_digest = _module_state_sha256(model.codec)
    shutil.copyfile(
        artifacts.latent_normalization_path,
        output / "latent_normalization.json",
    )
    if sha256_path(output / "latent_normalization.json") != (
        artifacts.latent_normalization_sha256
    ):
        raise RuntimeError("copied B3 latent normalization changed bytes")

    bank = validation_noise_bank(
        seed=config.validation_noise_seed,
        members=config.ensemble_members,
        raw_features=noise_config.raw_noise_features,
    )
    bank_path = output / "validation_noise.npy"
    bank_sha256 = save_validation_noise_bank(bank_path, bank)
    reloaded_bank = np.load(bank_path, allow_pickle=False)
    if not np.array_equal(bank, reloaded_bank):
        raise RuntimeError("saved B3 validation noise changed values")

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
        [
            {
                "params": groups.common,
                "lr": config.common_peak_learning_rate,
                "name": "common",
            },
            {
                "params": groups.new,
                "lr": config.new_peak_learning_rate,
                "name": "new",
            },
        ],
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    train_noise_generator = torch.Generator(device="cpu")
    train_noise_generator.manual_seed(config.seed + FGN_VALIDATION_NOISE_SEED)

    run_record = {
        **config.to_record(),
        "model": model_config.to_record(),
        "functional_noise": noise_config.to_record(),
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
        "validation_noise_bank": {
            "seed": config.validation_noise_seed,
            "shape": list(bank.shape),
            "dtype": str(bank.dtype),
            "sha256": bank_sha256,
        },
    }
    write_strict_json_atomic(output / "config.json", run_record)

    history_path = output / "history.jsonl"
    history_handle = history_path.open("x", encoding="utf-8", buffering=1)
    selected_path = output / "selected.pt"
    final_path = output / "final_training_state.pt"
    selected_epoch: int | None = None
    selected_global_step: int | None = None
    selected_validation: dict[str, Any] | None = None
    selected_state: dict[str, Tensor] | None = None
    selected_probe: dict[str, Any] | None = None
    global_step = 0
    common_gradient_seen = False
    new_gradient_seen = False
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(config.epochs):
            epoch_started = time.monotonic()
            train_dataset.set_epoch(epoch)
            order = epoch_order(config.train_targets, seed=config.seed, epoch=epoch)
            loader = _window_loader(train_dataset, order)
            model.train()
            fields = FAMILY_FIELDS["c5p"]
            train_score = torch.zeros(len(fields), dtype=torch.float64)
            train_accuracy = torch.zeros(len(fields), dtype=torch.float64)
            train_spread = torch.zeros(len(fields), dtype=torch.float64)
            examples = 0
            accumulation_count = 0
            total_gradient_norms: list[float] = []
            common_gradient_norms: list[float] = []
            new_gradient_norms: list[float] = []

            for microstep, batch in enumerate(loader, start=1):
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
                raw_noise = torch.randn(
                    context.shape[0],
                    config.ensemble_members,
                    noise_config.raw_noise_features,
                    generator=train_noise_generator,
                    dtype=torch.float32,
                    device="cpu",
                ).to(device=device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    predictions = model.predict_with_noise(
                        context,
                        raw_noise,
                        horizon=1,
                    )[:, :, 0]
                loss_result: FairCRPSResult = fair_crps(
                    predictions.to(torch.float32),
                    target,
                )
                loss = loss_result.total
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite B3 loss at epoch {epoch}, microstep {microstep}"
                    )
                loss.backward()
                accumulation_count += 1
                examples += 1
                train_score += loss_result.per_channel.detach().to(
                    "cpu", torch.float64
                )
                train_accuracy += loss_result.accuracy_per_channel.detach().to(
                    "cpu", torch.float64
                )
                train_spread += loss_result.spread_per_channel.detach().to(
                    "cpu", torch.float64
                )

                step_due = (
                    accumulation_count == config.gradient_accumulation_targets
                )
                final_microstep = microstep == len(config.train_targets)
                if step_due or final_microstep:
                    scale_accumulated_gradients(
                        groups.all_parameters,
                        accumulation_count,
                    )
                    common_norm = _group_gradient_norm(groups.common).to(device)
                    new_norm = _group_gradient_norm(groups.new).to(device)
                    if not torch.isfinite(common_norm) or common_norm <= 0:
                        raise FloatingPointError("invalid B3 common-parameter gradient")
                    if not torch.isfinite(new_norm) or new_norm <= 0:
                        raise FloatingPointError("invalid B3 new-parameter gradient")
                    common_gradient_seen = True
                    new_gradient_seen = True
                    next_step = global_step + 1
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate_at_step(
                            config,
                            next_step,
                            group=str(group["name"]),
                        )
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        groups.all_parameters,
                        config.gradient_clip,
                    )
                    if not torch.isfinite(total_norm):
                        raise FloatingPointError("non-finite B3 total gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = next_step
                    common_gradient_norms.append(float(common_norm))
                    new_gradient_norms.append(float(new_norm))
                    total_gradient_norms.append(float(total_norm))
                    accumulation_count = 0

            if accumulation_count != 0:
                raise RuntimeError("B3 epoch left unstepped accumulated gradients")
            if examples != len(config.train_targets):
                raise RuntimeError("B3 epoch did not consume every frozen target once")
            if global_step != (epoch + 1) * config.optimizer_steps_per_epoch:
                raise RuntimeError("B3 optimizer-step count differs from schedule")

            validation = validation_fair_crps(
                model=model,
                dataset=validation_dataset,
                targets=config.validation_targets,
                bank=bank,
                device=device,
            )
            if (
                selected_validation is None
                or validation["equal_channel_fair_crps"]
                < selected_validation["equal_channel_fair_crps"]
            ):
                selected_epoch = epoch
                selected_global_step = global_step
                selected_validation = dict(validation)
                selected_state = {
                    name: value.detach().to("cpu").clone()
                    for name, value in model.transition.state_dict().items()
                }
                selected_probe = _fixed_probe(
                    model=model,
                    dataset=validation_dataset,
                    bank=bank,
                    device=device,
                )

            epoch_record = {
                "epoch": epoch,
                "examples": examples,
                "ensemble_members": config.ensemble_members,
                "global_step": global_step,
                "common_learning_rate": learning_rate_at_step(
                    config, global_step, group="common"
                ),
                "new_learning_rate": learning_rate_at_step(
                    config, global_step, group="new"
                ),
                "train_equal_channel_fair_crps": float(
                    (train_score / examples).mean()
                ),
                "train_fair_crps_by_channel": dict(
                    zip(fields, map(float, train_score / examples))
                ),
                "train_accuracy_by_channel": dict(
                    zip(fields, map(float, train_accuracy / examples))
                ),
                "train_spread_by_channel": dict(
                    zip(fields, map(float, train_spread / examples))
                ),
                "validation_equal_channel_fair_crps": validation[
                    "equal_channel_fair_crps"
                ],
                "validation_fair_crps_by_channel": validation[
                    "fair_crps_by_channel"
                ],
                "validation_accuracy_by_channel": validation[
                    "accuracy_by_channel"
                ],
                "validation_spread_by_channel": validation["spread_by_channel"],
                "mean_preclip_total_gradient_norm": float(
                    np.mean(total_gradient_norms)
                ),
                "maximum_preclip_total_gradient_norm": float(
                    np.max(total_gradient_norms)
                ),
                "mean_preclip_common_gradient_norm": float(
                    np.mean(common_gradient_norms)
                ),
                "mean_preclip_new_gradient_norm": float(
                    np.mean(new_gradient_norms)
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
        or selected_global_step is None
        or selected_validation is None
        or selected_state is None
        or selected_probe is None
    ):
        validation_dataset.close()
        raise RuntimeError("B3 smoke completed without a selected checkpoint")
    final_validation = validation_fair_crps(
        model=model,
        dataset=validation_dataset,
        targets=config.validation_targets,
        bank=bank,
        device=device,
    )
    save_torch_atomic(
        final_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            parameter_groups=groups,
            config=config,
            model_config=model_config,
            noise_config=noise_config,
            artifacts=artifacts,
            normalization=normalization,
            load_audit=load_audit,
            validation_noise_path=bank_path,
            validation_noise_sha256=bank_sha256,
            train_noise_generator=train_noise_generator,
            epoch=config.epochs - 1,
            global_step=global_step,
            validation=final_validation,
            paper0_commit=paper0_commit,
            selected=False,
            include_optimizer=True,
        ),
    )
    save_torch_atomic(
        selected_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            parameter_groups=groups,
            config=config,
            model_config=model_config,
            noise_config=noise_config,
            artifacts=artifacts,
            normalization=normalization,
            load_audit=load_audit,
            validation_noise_path=bank_path,
            validation_noise_sha256=bank_sha256,
            train_noise_generator=train_noise_generator,
            epoch=selected_epoch,
            global_step=selected_global_step,
            validation=selected_validation,
            paper0_commit=paper0_commit,
            selected=True,
            include_optimizer=False,
            transition_state=selected_state,
            reload_probe=selected_probe,
        ),
    )
    del selected_state, selected_probe

    probe = _reload_identity_and_diversity(
        selected_checkpoint=selected_path,
        artifacts=artifacts,
        config=config,
        model_config=model_config,
        noise_config=noise_config,
        validation_dataset=validation_dataset,
        device=device,
    )
    validation_dataset.close()
    codec_final_digest = _module_state_sha256(model.codec)
    codec_unchanged = codec_initial_digest == codec_final_digest
    if not codec_unchanged:
        raise RuntimeError("frozen B3 codec changed during optimization")
    if not common_gradient_seen or not new_gradient_seen:
        raise RuntimeError("B3 smoke did not exercise both staged parameter groups")
    torch.cuda.synchronize(device)

    result = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_B3_FGN_H1_GPU_smoke",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "config": run_record,
        "parameter_count": groups.to_record()["total_parameter_count"],
        "parameter_groups": groups.to_record(),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_step,
        "selected_epoch": selected_epoch,
        "selected_validation": selected_validation,
        "final_validation": final_validation,
        "preoptimization_parent_identity": parent_identity,
        "deterministic_parent_load_audit": load_audit.to_record(),
        "checkpoint_reload_bitwise_exact": bool(
            probe["reload_latent_bitwise_exact"]
            and probe["reload_forecast_bitwise_exact"]
        ),
        "member_probe": probe,
        "common_parameter_gradient_seen": common_gradient_seen,
        "new_parameter_gradient_seen": new_gradient_seen,
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
        "validation_noise_bank": {
            "path": str(bank_path),
            "sha256": bank_sha256,
            "seed": config.validation_noise_seed,
            "shape": list(bank.shape),
        },
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
        "strict_cuda_bitwise_determinism_claimed": False,
        "cudnn_deterministic_requested": True,
        "tf32_allowed": False,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "simulation_data_read": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "full_B3_training_authorized": False,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    write_strict_json_atomic(output / "result.json", result)
    return result
