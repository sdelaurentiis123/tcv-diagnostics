"""Frozen deterministic C5P O2 optimization and checkpoint mechanics.

Physics-derived quantities are absent from this module by construction.  They
remain downstream evaluation gates.
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

from .codec_training import (
    ALLOWED_SEEDS,
    save_torch_atomic,
    seed_everything,
    sha256_path,
)
from .model_data import write_strict_json_atomic
from .model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    ModelDatasetCatalog,
    epoch_order,
)
from .models import build_codec, equal_channel_mae
from .models.o2 import C5POneStepModel, MaskedLatentTransition, O2ViTConfig
from .o2_training_data import OneStepWindowDataset


O2_ARMS = {"C5P-H1": 1, "C5P-H2": 2}


@dataclass(frozen=True)
class O2RunConfig:
    mode: str
    arm: str
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
    minimum_learning_rate: float
    warmup_epochs: int
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    training_precision: str

    @classmethod
    def frozen(cls, *, mode: str, arm: str, seed: int) -> "O2RunConfig":
        if arm not in O2_ARMS:
            raise ValueError(f"unsupported O2 arm {arm!r}")
        if seed not in ALLOWED_SEEDS:
            raise ValueError(f"seed {seed} is outside {ALLOWED_SEEDS}")
        common = {
            "mode": mode,
            "arm": arm,
            "seed": int(seed),
            "microbatch": 1,
            "gradient_accumulation": 16,
            "validation_microbatch": 1,
            "latent_fit_microbatch": 4,
            "learning_rate": 2.0e-4,
            "minimum_learning_rate": 2.0e-6,
            "betas": (0.9, 0.95),
            "weight_decay": 1.0e-4,
            "gradient_clip": 1.0,
            "training_precision": "bfloat16_autocast",
        }
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
                warmup_epochs=10,
            )
        if mode == "smoke":
            return cls(
                **common,
                epochs=2,
                train_target_start=2,
                train_target_stop=18,
                validation_target_start=498,
                validation_target_stop=502,
                latent_fit_start=0,
                latent_fit_stop=16,
                warmup_epochs=1,
            )
        raise ValueError(f"unsupported O2 run mode {mode!r}")

    @property
    def context_frames(self) -> int:
        return O2_ARMS[self.arm]

    @property
    def train_targets(self) -> tuple[int, ...]:
        return tuple(range(self.train_target_start, self.train_target_stop))

    @property
    def validation_targets(self) -> tuple[int, ...]:
        return tuple(
            range(self.validation_target_start, self.validation_target_stop)
        )

    @property
    def latent_fit_frames(self) -> tuple[int, ...]:
        return tuple(range(self.latent_fit_start, self.latent_fit_stop))

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_targets) / self.gradient_accumulation)

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    @property
    def warmup_optimizer_steps(self) -> int:
        return self.warmup_epochs * self.optimizer_steps_per_epoch

    @property
    def final_accumulation_count(self) -> int:
        remainder = len(self.train_targets) % self.gradient_accumulation
        return remainder if remainder else self.gradient_accumulation

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
                "latent_fit_frames": [self.latent_fit_start, self.latent_fit_stop],
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "warmup_optimizer_steps": self.warmup_optimizer_steps,
                "final_accumulation_count": self.final_accumulation_count,
                "training_loss": (
                    "equal_channel_standardized_field_MAE_after_frozen_decode"
                ),
                "physics_derived_loss_allowed": False,
                "early_stopping": False,
            }
        )
        return record


def learning_rate_at_step(config: O2RunConfig, step: int) -> float:
    """One-indexed frozen linear-warmup/cosine schedule."""

    index = int(step)
    total = config.total_optimizer_steps
    warmup = config.warmup_optimizer_steps
    if not 1 <= index <= total:
        raise ValueError(f"optimizer step {index} is outside 1..{total}")
    if warmup > 0 and index <= warmup:
        return config.learning_rate * index / warmup
    if total == warmup:
        return config.learning_rate
    progress = (index - warmup) / (total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def scale_accumulated_gradients(parameters: Iterable[nn.Parameter], count: int) -> None:
    """Average accumulated sample gradients by the actual microbatch count."""

    divisor = int(count)
    if divisor <= 0:
        raise ValueError("gradient accumulation divisor must be positive")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


@dataclass(frozen=True)
class LatentNormalization:
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    sample_count_per_channel: int
    fit_frames: tuple[int, int]
    codec_checkpoint_sha256: str
    scientific_authority: bool

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        standard_deviation = np.asarray(self.standard_deviation, dtype=np.float64)
        if mean.ndim != 1 or standard_deviation.shape != mean.shape or mean.size == 0:
            raise ValueError("latent normalization arrays must be matched vectors")
        if not np.all(np.isfinite(mean)):
            raise ValueError("latent means must be finite")
        if not np.all(np.isfinite(standard_deviation)) or np.any(
            standard_deviation <= 0.0
        ):
            raise ValueError("latent standard deviations must be finite and positive")
        if self.sample_count_per_channel <= 0:
            raise ValueError("latent normalization count must be positive")
        if self.fit_frames[1] <= self.fit_frames[0]:
            raise ValueError("latent fit interval is empty")
        if len(self.codec_checkpoint_sha256) != 64:
            raise ValueError("codec checkpoint SHA-256 is malformed")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "per_latent_channel_training_only_population_moments",
            "mean": list(self.mean),
            "population_standard_deviation": list(self.standard_deviation),
            "sample_count_per_channel": int(self.sample_count_per_channel),
            "fit_frames": list(self.fit_frames),
            "codec_checkpoint_sha256": self.codec_checkpoint_sha256,
            "scientific_authority": bool(self.scientific_authority),
            "held_out_85606_read": False,
        }


def fit_latent_normalization(
    *,
    codec: nn.Module,
    dataset: Sequence[Mapping[str, Any]],
    frames: Sequence[int],
    codec_checkpoint_sha256: str,
    device: torch.device,
    batch_size: int,
    expected_channels: int,
    scientific_authority: bool,
) -> LatentNormalization:
    """Fit per-channel codec-latent population moments with batch Welford merges."""

    expected_frames = tuple(int(frame) for frame in frames)
    if not expected_frames or expected_frames != tuple(
        range(expected_frames[0], expected_frames[-1] + 1)
    ):
        raise ValueError("latent-fit frames must be ordered and contiguous")
    if len(dataset) != len(expected_frames):
        raise ValueError("latent-fit dataset length differs from frozen frames")
    if batch_size <= 0 or expected_channels <= 0:
        raise ValueError("latent-fit batch/channel counts must be positive")

    codec.eval()
    running_count = 0
    running_mean = np.zeros(expected_channels, dtype=np.float64)
    running_m2 = np.zeros(expected_channels, dtype=np.float64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    observed_frames: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            observed_frames.extend(int(item) for item in batch["frame_index"])
            values = batch["volume"].to(device=device, dtype=torch.float32)
            latent = codec.encode(values).to(torch.float64)
            if latent.shape[1] != expected_channels:
                raise ValueError("encoded latent channel count differs")
            axes = (0, *range(2, latent.ndim))
            batch_count = int(latent.numel() // latent.shape[1])
            batch_mean = latent.mean(dim=axes).cpu().numpy()
            centered = latent - torch.as_tensor(
                batch_mean,
                device=latent.device,
                dtype=latent.dtype,
            ).reshape(1, -1, *([1] * (latent.ndim - 2)))
            batch_m2 = centered.square().sum(dim=axes).cpu().numpy()
            if running_count == 0:
                running_count = batch_count
                running_mean = batch_mean
                running_m2 = batch_m2
            else:
                total = running_count + batch_count
                delta = batch_mean - running_mean
                running_m2 = (
                    running_m2
                    + batch_m2
                    + delta * delta * running_count * batch_count / total
                )
                running_mean = running_mean + delta * batch_count / total
                running_count = total
    if tuple(observed_frames) != expected_frames:
        raise RuntimeError("latent fit did not consume the frozen frames chronologically")
    variance = running_m2 / running_count
    standard_deviation = np.sqrt(variance)
    return LatentNormalization(
        mean=tuple(float(item) for item in running_mean),
        standard_deviation=tuple(float(item) for item in standard_deviation),
        sample_count_per_channel=running_count,
        fit_frames=(expected_frames[0], expected_frames[-1] + 1),
        codec_checkpoint_sha256=str(codec_checkpoint_sha256),
        scientific_authority=bool(scientific_authority),
    )


def load_frozen_codec(
    *,
    checkpoint: Path,
    expected_sha256: str,
    expected_seed: int,
    device: torch.device,
) -> nn.Module:
    """Load one exact selected C5P DCAE-L10 checkpoint and freeze every parameter."""

    path = Path(checkpoint)
    if "85606" in str(path).lower():
        raise ValueError("held-out paths are prohibited")
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise ValueError(f"codec checkpoint SHA-256 mismatch: {actual} != {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if config.get("codec") != "dcae_l10" or config.get("family") != "c5p":
        raise ValueError("codec checkpoint is not frozen C5P DCAE-L10")
    if int(config.get("seed", -1)) != int(expected_seed):
        raise ValueError("codec checkpoint seed differs")
    if payload.get("kind") != "selected_model":
        raise ValueError("codec checkpoint is not a selected model")
    codec = build_codec("dcae_l10", len(FAMILY_FIELDS["c5p"]))
    codec.load_state_dict(payload["model_state"], strict=True)
    codec.to(device).eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    return codec


def _window_loader(
    dataset: OneStepWindowDataset,
    ordered_targets: Sequence[int],
) -> DataLoader:
    start = dataset.target_frames[0]
    indices = [int(target) - start for target in ordered_targets]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("O2 batch order contains a target outside the dataset")
    return DataLoader(dataset, batch_size=1, sampler=indices, num_workers=0)


def validation_loss(
    model: C5POneStepModel,
    dataset: OneStepWindowDataset,
    config: O2RunConfig,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Full chronological float32 validation with float64 metric accumulators."""

    model.eval()
    channel_sum = torch.zeros(len(FAMILY_FIELDS["c5p"]), dtype=torch.float64)
    examples = 0
    loader = _window_loader(dataset, config.validation_targets)
    with torch.inference_mode():
        for batch in loader:
            context = batch["context"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            forecast = model(context).to(torch.float32)
            per_example_channel = (
                (forecast - target)
                .abs()
                .to(torch.float64)
                .flatten(start_dim=2)
                .mean(dim=2)
                .cpu()
            )
            channel_sum += per_example_channel.sum(dim=0)
            examples += int(target.shape[0])
    if examples != len(config.validation_targets):
        raise RuntimeError("validation did not consume every frozen target")
    per_channel = channel_sum / examples
    aggregate = per_channel.mean()
    if not torch.isfinite(aggregate):
        raise FloatingPointError("O2 validation loss is non-finite")
    return float(aggregate), [float(item) for item in per_channel]


def _probe_forecast(
    model: C5POneStepModel,
    dataset: OneStepWindowDataset,
    device: torch.device,
) -> dict[str, Any]:
    item = dataset[0]
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    model.eval()
    with torch.inference_mode():
        forecast = model(context).to("cpu", torch.float32)
    return {
        "target_frame_index": int(item["target_frame_index"]),
        "forecast": forecast,
    }


def _checkpoint_payload(
    *,
    model: C5POneStepModel,
    optimizer: AdamW,
    config: O2RunConfig,
    model_config: O2ViTConfig,
    latent_normalization: LatentNormalization,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    epoch: int,
    global_step: int,
    validation_loss_value: float,
    paper0_commit: str,
    selected: bool,
    include_optimizer: bool,
    transition_state: Mapping[str, Tensor] | None = None,
    reload_probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "selected_O2_transition" if selected else "final_O2_training_state",
        "paper0_commit": paper0_commit,
        "config": config.to_record(),
        "model_config": model_config.to_record(),
        "latent_normalization": latent_normalization.to_record(),
        "codec_checkpoint": {
            "path": str(codec_checkpoint),
            "sha256": codec_checkpoint_sha256,
            "trainable": False,
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_loss": float(validation_loss_value),
        "transition_state": (
            model.transition.state_dict()
            if transition_state is None
            else dict(transition_state)
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    if reload_probe is not None:
        payload["reload_probe"] = dict(reload_probe)
    return payload


def _reload_identity(
    *,
    selected_checkpoint: Path,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    validation_dataset: OneStepWindowDataset,
    config: O2RunConfig,
    model_config: O2ViTConfig,
    device: torch.device,
) -> bool:
    payload = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    codec = load_frozen_codec(
        checkpoint=codec_checkpoint,
        expected_sha256=codec_checkpoint_sha256,
        expected_seed=config.seed,
        device=device,
    )
    transition = MaskedLatentTransition(
        context_frames=config.context_frames,
        config=model_config,
    ).to(device)
    transition.load_state_dict(payload["transition_state"], strict=True)
    normalization = payload["latent_normalization"]
    restored = C5POneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor(normalization["mean"]),
        latent_standard_deviation=torch.tensor(
            normalization["population_standard_deviation"]
        ),
    ).to(device).eval()
    item = validation_dataset[0]
    probe = payload["reload_probe"]
    if int(item["target_frame_index"]) != int(probe["target_frame_index"]):
        raise RuntimeError("O2 checkpoint reload probe target differs")
    context = torch.from_numpy(item["context"])[None].to(device, torch.float32)
    with torch.inference_mode():
        actual = restored(context).to("cpu", torch.float32)
    return bool(torch.equal(actual, probe["forecast"]))


def train_o2(
    *,
    config: O2RunConfig,
    catalog: ModelDatasetCatalog,
    codec_checkpoint: Path,
    codec_checkpoint_sha256: str,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
    epoch_callback: Callable[[Mapping[str, Any]], None] | None = None,
    model_config: O2ViTConfig = O2ViTConfig(),
) -> dict[str, Any]:
    """Execute one frozen full or bounded-smoke deterministic O2 run."""

    output = Path(output_directory)
    if "85606" in str(output).lower():
        raise ValueError("held-out paths are prohibited")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite O2 run {output}")
    output.mkdir(parents=True)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("O2 training and smoke execution require a CUDA worker")

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
        context_frames=config.context_frames,
        augment=True,
        seed=config.seed,
    )
    validation_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=config.validation_targets,
        context_frames=config.context_frames,
        augment=False,
        seed=config.seed,
    )

    seed_everything(config.seed)
    transition = MaskedLatentTransition(
        context_frames=config.context_frames,
        config=model_config,
    ).to(device)
    model = C5POneStepModel(
        codec=codec,
        transition=transition,
        latent_mean=torch.tensor(latent_normalization.mean),
        latent_standard_deviation=torch.tensor(
            latent_normalization.standard_deviation
        ),
    ).to(device)
    trainable_parameters = tuple(model.transition.parameters())
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
            "sha256": codec_checkpoint_sha256,
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
    selected_loss = math.inf
    selected_state: dict[str, Tensor] | None = None
    selected_probe: dict[str, Any] | None = None
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
            loss_sum = 0.0
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
                    forecast = model(context)
                    loss = equal_channel_mae(target, forecast)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite O2 loss at epoch {epoch}, microstep {microstep}"
                    )
                loss.backward()
                accumulation_count += 1
                loss_sum += float(loss.detach().to(torch.float64))
                examples += 1

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
                        raise FloatingPointError("non-finite O2 gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = next_step
                    gradient_norms.append(float(gradient_norm))
                    accumulation_count = 0

            if accumulation_count != 0:
                raise RuntimeError("O2 epoch left unstepped accumulated gradients")
            if examples != len(config.train_targets):
                raise RuntimeError("O2 epoch did not consume every frozen target once")
            if global_step != (epoch + 1) * config.optimizer_steps_per_epoch:
                raise RuntimeError("O2 optimizer-step count differs from schedule")

            validation_value, validation_by_channel = validation_loss(
                model,
                validation_dataset,
                config,
                device,
            )
            if validation_value < selected_loss:
                selected_loss = validation_value
                selected_epoch = epoch
                selected_global_step = global_step
                selected_state = {
                    name: value.detach().to("cpu").clone()
                    for name, value in model.transition.state_dict().items()
                }
                selected_probe = _probe_forecast(
                    model,
                    validation_dataset,
                    device,
                )

            epoch_record = {
                "epoch": epoch,
                "examples": examples,
                "global_step": global_step,
                "learning_rate": learning_rate_at_step(config, global_step),
                "train_equal_channel_mae": loss_sum / examples,
                "validation_equal_channel_mae": validation_value,
                "validation_mae_by_channel": dict(
                    zip(FAMILY_FIELDS["c5p"], validation_by_channel)
                ),
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
        or selected_state is None
        or selected_probe is None
    ):
        raise RuntimeError("O2 training completed without a selected checkpoint")
    final_validation_loss, final_validation_by_channel = validation_loss(
        model,
        validation_dataset,
        config,
        device,
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
            validation_loss_value=final_validation_loss,
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
            config=config,
            model_config=model_config,
            latent_normalization=latent_normalization,
            codec_checkpoint=codec_checkpoint,
            codec_checkpoint_sha256=codec_checkpoint_sha256,
            epoch=selected_epoch,
            global_step=selected_global_step,
            validation_loss_value=selected_loss,
            paper0_commit=paper0_commit,
            selected=True,
            include_optimizer=False,
            transition_state=selected_state,
            reload_probe=selected_probe,
        ),
    )
    del selected_state, selected_probe
    reload_exact = _reload_identity(
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
        raise RuntimeError("selected O2 checkpoint reload changed model output")
    torch.cuda.synchronize(device)

    result = {
        "schema_version": 1,
        "scope": f"O2_teacher_forced_one_step_{config.mode}",
        "paper0_commit": paper0_commit,
        "slurm_job_id": str(slurm_job_id),
        "config": run_record,
        "parameter_count": int(parameter_count),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_step,
        "selected_epoch": selected_epoch,
        "selected_validation_equal_channel_mae": selected_loss,
        "final_validation_equal_channel_mae": final_validation_loss,
        "final_validation_mae_by_channel": dict(
            zip(FAMILY_FIELDS["c5p"], final_validation_by_channel)
        ),
        "checkpoint_reload_bitwise_exact": reload_exact,
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
            "sha256": codec_checkpoint_sha256,
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
        "O2_scientific_gate_evaluated": False,
        "O3_launch_allowed": False,
    }
    write_strict_json_atomic(output / "result.json", result)
    return result
