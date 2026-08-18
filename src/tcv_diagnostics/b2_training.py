"""Frozen 85604-only training mechanics for the Paper 0 B2 LDM arm.

The bounded smoke and the full three-seed training run through separate,
fail-closed public functions.  Neither path computes physics-derived losses;
scientific forecast acceptance remains the responsibility of the separately
frozen probabilistic evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .codec_training import save_torch_atomic, seed_everything, sha256_path
from .model_data import write_strict_json_atomic
from .model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    ModelDatasetCatalog,
    epoch_order,
)
from .models.latent_diffusion import (
    C5PLatentDiffusionModel,
    LatentDiffusionViTConfig,
    MaskedEDMDenoiser,
    NoiseConditionedBackbone,
)
from .o2_training import (
    LatentNormalization,
    fit_latent_normalization,
    load_frozen_codec,
    scale_accumulated_gradients,
)
from .o2_training_data import OneStepWindowDataset


B2_LATENT_GRID = (16, 8, 22)
B2_VALIDATION_NOISE_SEED = 2_031_905_426
B2_SAMPLER_PROBE_SEED = 3_131_905
B2_SMOKE_SEED = 1701


@dataclass(frozen=True)
class B2RunConfig:
    """Immutable bounded-smoke or full-training budget."""

    mode: str
    seed: int
    epochs: int
    train_target_start: int
    train_target_stop: int
    validation_target_start: int
    validation_target_stop: int
    latent_fit_start: int
    latent_fit_stop: int
    microbatch: int
    gradient_accumulation: int
    validation_microbatch: int
    latent_fit_microbatch: int
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    training_precision: str
    validation_noise_seed: int
    sampler_probe_seed: int
    sampler_steps: int
    sampler_order: int

    @classmethod
    def frozen(cls, *, mode: str, seed: int) -> "B2RunConfig":
        if int(seed) not in (1701, 1702, 1703):
            raise ValueError("B2 seed must be one of 1701, 1702, or 1703")
        common = {
            "mode": str(mode),
            "seed": int(seed),
            "microbatch": 1,
            "gradient_accumulation": 16,
            "validation_microbatch": 1,
            "latent_fit_microbatch": 4,
            "learning_rate": 1.0e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "training_precision": "bfloat16_autocast",
            "validation_noise_seed": B2_VALIDATION_NOISE_SEED,
            "sampler_probe_seed": B2_SAMPLER_PROBE_SEED,
            "sampler_steps": 16,
            "sampler_order": 3,
        }
        if mode == "smoke":
            if int(seed) != B2_SMOKE_SEED:
                raise ValueError("the single authorized B2 smoke uses seed 1701")
            return cls(
                **common,
                epochs=2,
                train_target_start=2,
                train_target_stop=18,
                validation_target_start=498,
                validation_target_stop=502,
                latent_fit_start=0,
                latent_fit_stop=16,
            )
        if mode == "full":
            return cls(
                **common,
                epochs=200,
                train_target_start=2,
                train_target_stop=432,
                validation_target_start=498,
                validation_target_stop=624,
                latent_fit_start=0,
                latent_fit_stop=432,
            )
        raise ValueError(f"unsupported B2 mode {mode!r}")

    @property
    def context_frames(self) -> int:
        return 2

    @property
    def trajectory_frames(self) -> int:
        return self.context_frames + 1

    @property
    def train_targets(self) -> tuple[int, ...]:
        return tuple(range(self.train_target_start, self.train_target_stop))

    @property
    def validation_targets(self) -> tuple[int, ...]:
        return tuple(range(self.validation_target_start, self.validation_target_stop))

    @property
    def latent_fit_frames(self) -> tuple[int, ...]:
        return tuple(range(self.latent_fit_start, self.latent_fit_stop))

    @property
    def latent_trajectory_shape(self) -> tuple[int, ...]:
        return (32, self.trajectory_frames, *B2_LATENT_GRID)

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_targets) / self.gradient_accumulation)

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    @property
    def final_accumulation_count(self) -> int:
        remainder = len(self.train_targets) % self.gradient_accumulation
        return remainder if remainder else self.gradient_accumulation

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "arm": "B2-LDM-H2",
                "context_frames": self.context_frames,
                "future_frames": 1,
                "fields": list(FAMILY_FIELDS["c5p"]),
                "train_targets": [self.train_target_start, self.train_target_stop],
                "validation_targets": [
                    self.validation_target_start,
                    self.validation_target_stop,
                ],
                "latent_fit_frames": [self.latent_fit_start, self.latent_fit_stop],
                "latent_trajectory_shape": list(self.latent_trajectory_shape),
                "optimizer": "AdamW",
                "scheduler": "cosine_to_zero_per_optimizer_step",
                "warmup_steps": 0,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "final_accumulation_count": self.final_accumulation_count,
                "training_loss": "LOLA_EDM_complete_trajectory_denoising_MSE",
                "checkpoint_selection": (
                    "earliest_lowest_fixed_noise_validation_complete_loss"
                ),
                "physics_derived_loss_allowed": False,
                "full_training_authorized": self.mode == "full",
                "scientific_result": False,
            }
        )
        return record


def learning_rate_at_step(config: B2RunConfig, step: int) -> float:
    """One-indexed cosine schedule with exact frozen endpoints."""

    index = int(step)
    total = config.total_optimizer_steps
    if not 1 <= index <= total:
        raise ValueError(f"optimizer step {index} is outside 1..{total}")
    if total == 1:
        return config.learning_rate
    progress = (index - 1) / (total - 1)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _validation_generator_seed(base_seed: int, target_frame: int) -> int:
    if int(base_seed) < 0 or int(target_frame) < 0:
        raise ValueError("validation seeds and target frames must be nonnegative")
    state = np.random.SeedSequence(
        [int(base_seed), int(target_frame), 0x42324C44]
    ).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(2**63 - 1))


def fixed_validation_perturbation(
    *,
    target_frame: int,
    latent_trajectory_shape: Sequence[int],
    base_seed: int = B2_VALIDATION_NOISE_SEED,
) -> tuple[Tensor, Tensor]:
    """Create one immutable CPU noise time/tensor independent of model seed."""

    shape = tuple(int(item) for item in latent_trajectory_shape)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("latent trajectory shape must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_validation_generator_seed(base_seed, target_frame))
    noise_time = torch.rand((1,), generator=generator, dtype=torch.float32)
    noise = torch.randn((1, *shape), generator=generator, dtype=torch.float32)
    return noise_time, noise


class FixedValidationNoiseBank:
    """Materialized CPU bank shared across epochs and model seeds."""

    def __init__(
        self,
        targets: Sequence[int],
        latent_trajectory_shape: Sequence[int],
        *,
        base_seed: int,
    ) -> None:
        values = tuple(int(target) for target in targets)
        if not values or len(values) != len(set(values)):
            raise ValueError("validation-noise targets must be nonempty and unique")
        self.targets = values
        self.latent_trajectory_shape = tuple(
            int(item) for item in latent_trajectory_shape
        )
        self.base_seed = int(base_seed)
        self._bank = {
            target: fixed_validation_perturbation(
                target_frame=target,
                latent_trajectory_shape=self.latent_trajectory_shape,
                base_seed=self.base_seed,
            )
            for target in self.targets
        }

    def get(self, target_frame: int) -> tuple[Tensor, Tensor]:
        try:
            return self._bank[int(target_frame)]
        except KeyError as error:
            raise KeyError(f"target {target_frame} is absent from noise bank") from error


@dataclass(frozen=True)
class LossAverages:
    complete: float
    context: float
    target: float
    examples: int

    def to_record(self, prefix: str) -> dict[str, float | int]:
        return {
            f"{prefix}_complete_denoising_loss": float(self.complete),
            f"{prefix}_context_denoising_loss": float(self.context),
            f"{prefix}_target_denoising_loss": float(self.target),
            f"{prefix}_examples": int(self.examples),
        }


def _loss_averages(sums: Mapping[str, float], examples: int) -> LossAverages:
    if int(examples) <= 0:
        raise ValueError("cannot average zero losses")
    values = LossAverages(
        complete=float(sums["complete"]) / examples,
        context=float(sums["context"]) / examples,
        target=float(sums["target"]) / examples,
        examples=int(examples),
    )
    if not all(
        math.isfinite(value) for value in (values.complete, values.context, values.target)
    ):
        raise FloatingPointError("non-finite B2 loss average")
    return values


def _window_loader(
    dataset: OneStepWindowDataset,
    ordered_targets: Sequence[int],
) -> DataLoader:
    start = dataset.target_frames[0]
    indices = [int(target) - start for target in ordered_targets]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("B2 order contains a target outside the dataset")
    return DataLoader(dataset, batch_size=1, sampler=indices, num_workers=0)


def build_b2_model(
    *,
    codec: nn.Module,
    latent_normalization: LatentNormalization,
    device: torch.device,
    model_config: LatentDiffusionViTConfig = LatentDiffusionViTConfig(),
    sampler_steps: int = 16,
    sampler_order: int = 3,
) -> C5PLatentDiffusionModel:
    """Construct the exact B2 wrapper without mutating package exports."""

    backbone = NoiseConditionedBackbone(model_config)
    denoiser = MaskedEDMDenoiser(backbone)
    return C5PLatentDiffusionModel(
        codec=codec,
        denoiser=denoiser,
        latent_mean=torch.tensor(latent_normalization.mean),
        latent_standard_deviation=torch.tensor(
            latent_normalization.standard_deviation
        ),
        context_frames=2,
        sampler_steps=sampler_steps,
        sampler_order=sampler_order,
    ).to(device)


def validation_loss(
    *,
    model: C5PLatentDiffusionModel,
    dataset: OneStepWindowDataset,
    config: B2RunConfig,
    noise_bank: FixedValidationNoiseBank,
    device: torch.device,
) -> LossAverages:
    """Full chronological validation against one frozen perturbation bank."""

    model.eval()
    sums = {"complete": 0.0, "context": 0.0, "target": 0.0}
    examples = 0
    loader = _window_loader(dataset, config.validation_targets)
    with torch.inference_mode():
        for batch in loader:
            target_frame = int(batch["target_frame_index"].item())
            noise_time, noise = noise_bank.get(target_frame)
            context = batch["context"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            losses = model.training_loss(
                context,
                target,
                noise_time=noise_time.to(device),
                noise=noise.to(device),
            )
            for name in sums:
                value = getattr(losses, name).detach().to(torch.float64)
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite B2 validation {name} loss at {target_frame}"
                    )
                sums[name] += float(value)
            examples += 1
    if examples != len(config.validation_targets):
        raise RuntimeError("validation did not consume every frozen target")
    return _loss_averages(sums, examples)


def _checkpoint_payload(
    *,
    model: C5PLatentDiffusionModel,
    optimizer: AdamW,
    config: B2RunConfig,
    model_config: LatentDiffusionViTConfig,
    latent_normalization: LatentNormalization,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    epoch: int,
    global_step: int,
    validation: LossAverages,
    paper0_commit: str,
    selected: bool,
    include_optimizer: bool,
    denoiser_state: Mapping[str, Tensor] | None = None,
    reload_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "selected_B2_LDM" if selected else "final_B2_LDM_training_state",
        "paper0_commit": str(paper0_commit),
        "config": config.to_record(),
        "model_config": model_config.to_record(),
        "latent_normalization": latent_normalization.to_record(),
        "codec_checkpoint": {
            "path": str(codec_checkpoint),
            "sha256": str(codec_checkpoint_sha256),
            "trainable": False,
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation": asdict(validation),
        "denoiser_state": (
            model.denoiser.state_dict()
            if denoiser_state is None
            else dict(denoiser_state)
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "physics_derived_loss_used": False,
        "held_out_85606_read": False,
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    if reload_probe is not None:
        payload["reload_probe"] = dict(reload_probe)
    return payload


def _seed_sampler(seed: int) -> None:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _sampler_probe(
    *,
    model: C5PLatentDiffusionModel,
    validation_dataset: OneStepWindowDataset,
    config: B2RunConfig,
    device: torch.device,
) -> dict[str, Any]:
    item = validation_dataset[0]
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    model.eval()
    _seed_sampler(config.sampler_probe_seed)
    standardized_latent = model._sample_standardized_target(
        context,
        ensemble_size=2,
    )
    batch, members = standardized_latent.shape[:2]
    flattened = standardized_latent.reshape(
        batch * members,
        *standardized_latent.shape[2:],
    )
    decoded = model._decode_target(flattened)
    forecast = decoded.reshape(batch, members, *decoded.shape[1:])[:, :, None]
    if forecast.shape != (1, 2, 1, 5, 64, 32, 88):
        raise RuntimeError(f"B2 probe has noncanonical axes {forecast.shape}")
    if not torch.isfinite(standardized_latent).all() or not torch.isfinite(forecast).all():
        raise FloatingPointError("B2 sampler probe is non-finite")
    latent_difference = (
        (standardized_latent[:, 0] - standardized_latent[:, 1])
        .to(torch.float64)
        .square()
        .mean()
        .sqrt()
    )
    field_difference = (
        (forecast[:, 0] - forecast[:, 1])
        .to(torch.float64)
        .square()
        .mean()
        .sqrt()
    )
    if not float(latent_difference) > 0.0:
        raise RuntimeError("B2 sampler produced zero latent member diversity")
    if not float(field_difference) > 0.0:
        raise RuntimeError("B2 sampler produced zero decoded member diversity")
    return {
        "target_frame_index": int(item["target_frame_index"]),
        "sampler_seed": int(config.sampler_probe_seed),
        "ensemble_size": 2,
        "standardized_latent": standardized_latent.detach().to("cpu", torch.float32),
        "forecast": forecast.detach().to("cpu", torch.float32),
        "latent_member_rms_difference": float(latent_difference),
        "field_member_rms_difference": float(field_difference),
    }


def _reload_identity(
    *,
    selected_checkpoint: Path,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    validation_dataset: OneStepWindowDataset,
    config: B2RunConfig,
    model_config: LatentDiffusionViTConfig,
    device: torch.device,
) -> tuple[bool, dict[str, float]]:
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    codec = load_frozen_codec(
        checkpoint=codec_checkpoint,
        expected_sha256=codec_checkpoint_sha256,
        expected_seed=config.seed,
        device=device,
    )
    normalization = payload["latent_normalization"]
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
    restored = build_b2_model(
        codec=codec,
        latent_normalization=latent_normalization,
        device=device,
        model_config=model_config,
        sampler_steps=config.sampler_steps,
        sampler_order=config.sampler_order,
    )
    restored.denoiser.load_state_dict(payload["denoiser_state"], strict=True)
    actual = _sampler_probe(
        model=restored,
        validation_dataset=validation_dataset,
        config=config,
        device=device,
    )
    expected = payload["reload_probe"]
    identity = bool(
        torch.equal(actual["standardized_latent"], expected["standardized_latent"])
        and torch.equal(actual["forecast"], expected["forecast"])
    )
    metrics = {
        "latent_member_rms_difference": float(
            actual["latent_member_rms_difference"]
        ),
        "field_member_rms_difference": float(actual["field_member_rms_difference"]),
    }
    return identity, metrics


def _train_b2_authorized(
    *,
    config: B2RunConfig,
    catalog: ModelDatasetCatalog,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one already-authorized B2 run without making a science claim."""

    if config.mode not in {"smoke", "full"}:
        raise RuntimeError(f"unsupported authorized B2 mode {config.mode!r}")
    if config.mode == "smoke" and config.seed != B2_SMOKE_SEED:
        raise RuntimeError("the authorized B2 smoke seed is 1701")
    output = Path(output_directory)
    if "85606" in str(output).lower():
        raise ValueError("held-out paths are prohibited")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite B2 run {output}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B2 execution requires a CUDA worker")
    output.mkdir(parents=True)

    model_config = LatentDiffusionViTConfig()
    codec = load_frozen_codec(
        checkpoint=codec_checkpoint,
        expected_sha256=codec_checkpoint_sha256,
        expected_seed=config.seed,
        device=device,
    )
    latent_dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=config.latent_fit_frames,
        augment=False,
        seed=config.seed,
    )
    latent_normalization = fit_latent_normalization(
        codec=codec,
        dataset=latent_dataset,
        frames=config.latent_fit_frames,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
        device=device,
        batch_size=config.latent_fit_microbatch,
        expected_channels=model_config.latent_channels,
        scientific_authority=config.mode == "full",
    )
    latent_dataset.close()
    write_strict_json_atomic(
        output / "latent_normalization.json",
        latent_normalization.to_record(),
    )

    train_dataset = OneStepWindowDataset(
        catalog,
        split="train",
        target_frames=config.train_targets,
        context_frames=2,
        augment=True,
        seed=config.seed,
    )
    validation_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=config.validation_targets,
        context_frames=2,
        augment=False,
        seed=config.seed,
    )
    noise_bank = FixedValidationNoiseBank(
        config.validation_targets,
        config.latent_trajectory_shape,
        base_seed=config.validation_noise_seed,
    )

    seed_everything(config.seed)
    model = build_b2_model(
        codec=codec,
        latent_normalization=latent_normalization,
        device=device,
        model_config=model_config,
        sampler_steps=config.sampler_steps,
        sampler_order=config.sampler_order,
    )
    trainable_parameters = tuple(model.denoiser.parameters())
    if any(parameter.requires_grad for parameter in model.codec.parameters()):
        raise RuntimeError("the B2 codec unexpectedly has trainable parameters")
    parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    optimizer = AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    run_record = {
        **config.to_record(),
        "model": model_config.to_record(),
        "codec_checkpoint": {
            "path": str(codec_checkpoint),
            "sha256": str(codec_checkpoint_sha256),
            "trainable": False,
        },
    }
    write_strict_json_atomic(output / "config.json", run_record)
    history_path = output / "history.jsonl"
    history_handle = history_path.open("x", encoding="utf-8", buffering=1)
    selected_path = output / "selected.pt"
    final_path = output / "final_training_state.pt"
    selected_epoch: int | None = None
    selected_global_step: int | None = None
    selected_validation: LossAverages | None = None
    selected_state: dict[str, Tensor] | None = None
    global_step = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(config.epochs):
            epoch_started = time.monotonic()
            train_dataset.set_epoch(epoch)
            order = epoch_order(config.train_targets, seed=config.seed, epoch=epoch)
            loader = _window_loader(train_dataset, order)
            model.train()
            sums = {"complete": 0.0, "context": 0.0, "target": 0.0}
            examples = 0
            accumulation_count = 0
            gradient_norms: list[float] = []
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
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    losses = model.training_loss(context, target)
                if not all(
                    torch.isfinite(value)
                    for value in (losses.complete, losses.context, losses.target)
                ):
                    raise FloatingPointError(
                        f"non-finite B2 loss at epoch {epoch}, microstep {microstep}"
                    )
                losses.complete.backward()
                accumulation_count += 1
                examples += 1
                for name in sums:
                    sums[name] += float(
                        getattr(losses, name).detach().to(torch.float64)
                    )

                step_due = accumulation_count == config.gradient_accumulation
                final_microstep = microstep == len(config.train_targets)
                if step_due or final_microstep:
                    scale_accumulated_gradients(
                        trainable_parameters,
                        accumulation_count,
                    )
                    next_step = global_step + 1
                    learning_rate = learning_rate_at_step(config, next_step)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        config.gradient_clip,
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("non-finite B2 gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = next_step
                    gradient_norms.append(float(gradient_norm))
                    accumulation_count = 0

            if accumulation_count != 0:
                raise RuntimeError("B2 epoch left unstepped accumulated gradients")
            if examples != len(config.train_targets):
                raise RuntimeError("B2 epoch did not consume each target exactly once")
            if global_step != (epoch + 1) * config.optimizer_steps_per_epoch:
                raise RuntimeError("B2 optimizer-step count differs from schedule")

            train_average = _loss_averages(sums, examples)
            validation_average = validation_loss(
                model=model,
                dataset=validation_dataset,
                config=config,
                noise_bank=noise_bank,
                device=device,
            )
            if (
                selected_validation is None
                or validation_average.complete < selected_validation.complete
            ):
                selected_epoch = epoch
                selected_global_step = global_step
                selected_validation = validation_average
                selected_state = {
                    name: value.detach().to("cpu").clone()
                    for name, value in model.denoiser.state_dict().items()
                }

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": learning_rate_at_step(config, global_step),
                **train_average.to_record("train"),
                **validation_average.to_record("validation"),
                "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
                "maximum_preclip_gradient_norm": float(np.max(gradient_norms)),
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
    ):
        raise RuntimeError("B2 run completed without a selected checkpoint")

    final_validation = validation_loss(
        model=model,
        dataset=validation_dataset,
        config=config,
        noise_bank=noise_bank,
        device=device,
    )
    save_torch_atomic(
        final_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            model_config=model_config,
            latent_normalization=latent_normalization,
            codec_checkpoint=codec_checkpoint,
            codec_checkpoint_sha256=codec_checkpoint_sha256,
            epoch=config.epochs - 1,
            global_step=global_step,
            validation=final_validation,
            paper0_commit=paper0_commit,
            selected=False,
            include_optimizer=True,
        ),
    )

    model.denoiser.load_state_dict(selected_state, strict=True)
    selected_probe = _sampler_probe(
        model=model,
        validation_dataset=validation_dataset,
        config=config,
        device=device,
    )
    save_torch_atomic(
        selected_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            model_config=model_config,
            latent_normalization=latent_normalization,
            codec_checkpoint=codec_checkpoint,
            codec_checkpoint_sha256=codec_checkpoint_sha256,
            epoch=selected_epoch,
            global_step=selected_global_step,
            validation=selected_validation,
            paper0_commit=paper0_commit,
            selected=True,
            include_optimizer=False,
            denoiser_state=selected_state,
            reload_probe=selected_probe,
        ),
    )
    del selected_state, selected_probe, model, optimizer, codec
    torch.cuda.empty_cache()

    reload_exact, sampler_metrics = _reload_identity(
        selected_checkpoint=selected_path,
        codec_checkpoint=codec_checkpoint,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
        validation_dataset=validation_dataset,
        config=config,
        model_config=model_config,
        device=device,
    )
    validation_dataset.close()
    if not reload_exact:
        raise RuntimeError("selected B2 checkpoint reload changed sampler output")
    torch.cuda.synchronize(device)

    result = {
        "schema_version": 1,
        "scope": (
            "bounded_non_scientific_B2_LDM_H2_GPU_smoke"
            if config.mode == "smoke"
            else "B2_LDM_H2_full_training_85604"
        ),
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "config": run_record,
        "parameter_count": int(parameter_count),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_step,
        "selected_epoch": selected_epoch,
        "selected_validation": asdict(selected_validation),
        "final_validation": asdict(final_validation),
        "checkpoint_reload_bitwise_exact": reload_exact,
        "sampler_probe": {
            "target_frame_index": config.validation_targets[0],
            "sampler_seed": config.sampler_probe_seed,
            "ensemble_size": 2,
            "canonical_forecast_shape": [1, 2, 1, 5, 64, 32, 88],
            **sampler_metrics,
            "finite": True,
            "nonzero_latent_diversity": True,
            "nonzero_decoded_diversity": True,
        },
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
        "latent_normalization": {
            "path": str(output / "latent_normalization.json"),
            "sha256": sha256_path(output / "latent_normalization.json"),
        },
        "codec_checkpoint": {
            "path": str(codec_checkpoint),
            "sha256": str(codec_checkpoint_sha256),
            "trainable": False,
        },
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "strict_cuda_bitwise_determinism_claimed": False,
        "reload_identity_same_process_same_device": True,
        "cudnn_deterministic_requested": True,
        "tf32_allowed": False,
        "physics_derived_loss_used": False,
        "target_truth_used_as_model_input": False,
        "absolute_time_used_as_model_input": False,
        "simulation_data_read": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "scientific_result": False,
        "full_B2_training_authorized": config.mode == "full",
        "training_complete_is_scientific_acceptance": False,
        "probabilistic_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    write_strict_json_atomic(output / "result.json", result)
    return result


def train_b2_smoke(
    *,
    config: B2RunConfig,
    catalog: ModelDatasetCatalog,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute only the historical bounded smoke authorization."""

    if config.mode != "smoke":
        raise RuntimeError("full B2 training is not authorized by the smoke path")
    if config.seed != B2_SMOKE_SEED:
        raise RuntimeError("the authorized B2 smoke seed is 1701")
    if config != B2RunConfig.frozen(mode="smoke", seed=config.seed):
        raise RuntimeError("the bounded B2 smoke config differs from its freeze")
    return _train_b2_authorized(
        config=config,
        catalog=catalog,
        codec_checkpoint=codec_checkpoint,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
        output_directory=output_directory,
        paper0_commit=paper0_commit,
        slurm_job_id=slurm_job_id,
        device=device,
        epoch_callback=epoch_callback,
    )


def train_b2_full(
    *,
    config: B2RunConfig,
    catalog: ModelDatasetCatalog,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one seed of the separately frozen full B2 training matrix."""

    if config.mode != "full":
        raise RuntimeError("the full B2 path requires mode='full'")
    if config.seed not in (1701, 1702, 1703):
        raise RuntimeError("the full B2 seed is outside the frozen three-seed matrix")
    if config != B2RunConfig.frozen(mode="full", seed=config.seed):
        raise RuntimeError("the full B2 config differs from its frozen budget")
    return _train_b2_authorized(
        config=config,
        catalog=catalog,
        codec_checkpoint=codec_checkpoint,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
        output_directory=output_directory,
        paper0_commit=paper0_commit,
        slurm_job_id=slurm_job_id,
        device=device,
        epoch_callback=epoch_callback,
    )
