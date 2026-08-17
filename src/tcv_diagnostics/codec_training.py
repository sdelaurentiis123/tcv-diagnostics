"""Frozen O1 codec optimization and checkpoint mechanics.

This module contains data-only optimization.  Physics quantities are absent by
construction and remain downstream evaluation metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model_data import write_strict_json_atomic
from .model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    ModelDatasetCatalog,
    epoch_order,
)
from .models import build_codec, equal_channel_mae, latent_shape, CODEC_CONFIGS


ALLOWED_SEEDS = (1701, 1702, 1703)


@dataclass(frozen=True)
class CodecRunConfig:
    mode: str
    codec: str
    family: str
    seed: int
    epochs: int
    train_start: int
    train_stop: int
    validation_start: int
    validation_stop: int
    microbatch: int
    gradient_accumulation: int
    validation_microbatch: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_epochs: int
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    training_precision: str

    @classmethod
    def frozen(
        cls,
        *,
        mode: str,
        codec: str,
        family: str,
        seed: int,
    ) -> "CodecRunConfig":
        if codec not in CODEC_CONFIGS:
            raise ValueError(f"unsupported codec {codec!r}")
        if family not in FAMILY_FIELDS:
            raise ValueError(f"unsupported family {family!r}")
        if seed not in ALLOWED_SEEDS:
            raise ValueError(f"seed {seed} is outside {ALLOWED_SEEDS}")
        common = {
            "mode": mode,
            "codec": codec,
            "family": family,
            "seed": seed,
            "microbatch": 4,
            "gradient_accumulation": 4,
            "validation_microbatch": 1,
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
                train_start=0,
                train_stop=432,
                validation_start=496,
                validation_stop=624,
                warmup_epochs=10,
            )
        if mode == "smoke":
            return cls(
                **common,
                epochs=2,
                train_start=0,
                train_stop=16,
                validation_start=496,
                validation_stop=500,
                warmup_epochs=1,
            )
        raise ValueError(f"unsupported run mode {mode!r}")

    @property
    def channels(self) -> int:
        return len(FAMILY_FIELDS[self.family])

    @property
    def train_frames(self) -> tuple[int, ...]:
        return tuple(range(self.train_start, self.train_stop))

    @property
    def validation_frames(self) -> tuple[int, ...]:
        return tuple(range(self.validation_start, self.validation_stop))

    @property
    def effective_batch(self) -> int:
        return self.microbatch * self.gradient_accumulation

    @property
    def optimizer_steps_per_epoch(self) -> int:
        examples = len(self.train_frames)
        if examples % self.effective_batch:
            raise ValueError("training frames are not divisible by effective batch")
        return examples // self.effective_batch

    @property
    def total_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_epoch * self.epochs

    @property
    def warmup_optimizer_steps(self) -> int:
        return self.optimizer_steps_per_epoch * self.warmup_epochs

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "channels": self.channels,
                "fields": list(FAMILY_FIELDS[self.family]),
                "effective_batch": self.effective_batch,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "warmup_optimizer_steps": self.warmup_optimizer_steps,
                "training_loss": "equal_channel_standardized_MAE",
                "physics_derived_loss_allowed": False,
            }
        )
        return record


def learning_rate_at_step(config: CodecRunConfig, step: int) -> float:
    """One-indexed linear-warmup/cosine learning rate."""

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


def seed_everything(seed: int) -> None:
    """Seed model initialization and dropout without claiming bitwise CUDA runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_path(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def save_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a job-owned checkpoint without leaving partial finals."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _loader(
    dataset: CodecFrameDataset,
    ordered_frames: Iterable[int],
    *,
    batch_size: int,
) -> DataLoader:
    start = dataset.frames[0]
    indices = [int(frame) - start for frame in ordered_frames]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("batch order contains a frame outside the dataset")
    batches = [
        indices[offset : offset + batch_size]
        for offset in range(0, len(indices), batch_size)
    ]
    if any(len(batch) != batch_size for batch in batches):
        raise ValueError("partial microbatches are prohibited")
    return DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _validation_loss(
    model: nn.Module,
    dataset: CodecFrameDataset,
    config: CodecRunConfig,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Full chronological validation loss with float64 accumulators."""

    model.eval()
    channel_sum = torch.zeros(config.channels, dtype=torch.float64)
    frames = 0
    loader = _loader(
        dataset,
        config.validation_frames,
        batch_size=config.validation_microbatch,
    )
    with torch.inference_mode():
        for batch in loader:
            target = batch["volume"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            reconstruction, _ = model(target)
            per_frame_channel = (
                (target - reconstruction)
                .abs()
                .to(torch.float64)
                .flatten(start_dim=2)
                .mean(dim=2)
                .cpu()
            )
            channel_sum += per_frame_channel.sum(dim=0)
            frames += int(target.shape[0])
    if frames != len(config.validation_frames):
        raise RuntimeError("validation did not consume every frozen frame")
    per_channel = channel_sum / frames
    aggregate = per_channel.mean()
    if not torch.isfinite(aggregate):
        raise FloatingPointError("validation loss is non-finite")
    return float(aggregate), [float(item) for item in per_channel]


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: AdamW,
    config: CodecRunConfig,
    epoch: int,
    global_step: int,
    validation_loss: float,
    paper0_commit: str,
    selected: bool,
    include_optimizer: bool,
    reload_probe: Mapping[str, Any] | None = None,
    model_state: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "selected_model" if selected else "final_training_state",
        "paper0_commit": paper0_commit,
        "config": config.to_record(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "validation_loss": float(validation_loss),
        "model_state": model.state_dict() if model_state is None else dict(model_state),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    if include_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    if reload_probe is not None:
        payload["reload_probe"] = dict(reload_probe)
    return payload


def _capture_reload_probe(
    model: nn.Module,
    dataset: CodecFrameDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Capture one float32 validation output at checkpoint-save time."""

    item = dataset[0]
    target = torch.from_numpy(item["volume"])[None].to(device, torch.float32)
    model.eval()
    with torch.inference_mode():
        reconstruction, latent = model(target)
    return {
        "frame_index": int(item["frame_index"]),
        "reconstruction": reconstruction.detach().to("cpu", torch.float32),
        "latent": latent.detach().to("cpu", torch.float32),
    }


def _reload_identity(
    checkpoint: Path,
    dataset: CodecFrameDataset,
    config: CodecRunConfig,
    device: torch.device,
) -> bool:
    """Require exact output equality after rebuilding from selected weights."""

    restored = build_codec(config.codec, config.channels).to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(payload["model_state"], strict=True)
    item = dataset[0]
    probe = payload["reload_probe"]
    if int(item["frame_index"]) != int(probe["frame_index"]):
        raise RuntimeError("checkpoint reload probe frame differs")
    target = torch.from_numpy(item["volume"])[None].to(device, torch.float32)
    with torch.inference_mode():
        actual = restored(target)
    exact = torch.equal(
        probe["reconstruction"].to(device), actual[0]
    ) and torch.equal(probe["latent"].to(device), actual[1])
    del restored, payload, target, actual
    return bool(exact)


def train_codec(
    *,
    config: CodecRunConfig,
    catalog: ModelDatasetCatalog,
    output_directory: Path,
    paper0_commit: str,
    slurm_job_id: str,
    device: torch.device,
) -> dict[str, Any]:
    """Execute one frozen full or bounded-smoke codec run."""

    output = Path(output_directory)
    if "85606" in str(output).lower():
        raise ValueError("held-out paths are prohibited")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite model run {output}")
    output.mkdir(parents=True)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("codec training and smoke execution require a CUDA worker")

    seed_everything(config.seed)
    train_dataset = CodecFrameDataset(
        catalog,
        family=config.family,
        split="train",
        frames=config.train_frames,
        augment=True,
        seed=config.seed,
    )
    validation_dataset = CodecFrameDataset(
        catalog,
        family=config.family,
        split="validation",
        frames=config.validation_frames,
        augment=False,
        seed=config.seed,
    )
    model = build_codec(config.codec, config.channels).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)

    write_strict_json_atomic(output / "config.json", config.to_record())
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
            order = epoch_order(config.train_frames, seed=config.seed, epoch=epoch)
            loader = _loader(
                train_dataset,
                order,
                batch_size=config.microbatch,
            )
            model.train()
            loss_sum = 0.0
            examples = 0
            gradient_norms: list[float] = []
            for microstep, batch in enumerate(loader, start=1):
                target = batch["volume"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=True,
                ):
                    reconstruction, _ = model(target)
                    loss = equal_channel_mae(target, reconstruction)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}, microstep {microstep}"
                    )
                (loss / config.gradient_accumulation).backward()
                batch_examples = int(target.shape[0])
                loss_sum += float(loss.detach().to(torch.float64)) * batch_examples
                examples += batch_examples

                if microstep % config.gradient_accumulation == 0:
                    next_step = global_step + 1
                    learning_rate = learning_rate_at_step(config, next_step)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("non-finite gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = next_step
                    gradient_norms.append(float(gradient_norm))

            if examples != len(config.train_frames):
                raise RuntimeError("training epoch did not consume every frame once")
            if global_step != (epoch + 1) * config.optimizer_steps_per_epoch:
                raise RuntimeError("optimizer-step count differs from frozen schedule")

            validation_loss, validation_by_channel = _validation_loss(
                model,
                validation_dataset,
                config,
                device,
            )
            if validation_loss < selected_loss:
                selected_loss = validation_loss
                selected_epoch = epoch
                selected_global_step = global_step
                selected_state = {
                    name: value.detach().to("cpu").clone()
                    for name, value in model.state_dict().items()
                }
                selected_probe = _capture_reload_probe(
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
                "validation_equal_channel_mae": validation_loss,
                "validation_mae_by_channel": dict(
                    zip(FAMILY_FIELDS[config.family], validation_by_channel)
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
    finally:
        history_handle.close()
        train_dataset.close()

    if (
        selected_epoch is None
        or selected_global_step is None
        or selected_state is None
        or selected_probe is None
    ):
        raise RuntimeError("training completed without a selected checkpoint")
    final_validation_loss, final_validation_by_channel = _validation_loss(
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
            epoch=config.epochs - 1,
            global_step=global_step,
            validation_loss=final_validation_loss,
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
            epoch=selected_epoch,
            global_step=selected_global_step,
            validation_loss=selected_loss,
            paper0_commit=paper0_commit,
            selected=True,
            include_optimizer=False,
            reload_probe=selected_probe,
            model_state=selected_state,
        ),
    )
    del selected_state, selected_probe
    reload_exact = _reload_identity(
        selected_path,
        validation_dataset,
        config,
        device,
    )
    validation_dataset.close()
    if not reload_exact:
        raise RuntimeError("selected checkpoint reload changed model output")
    torch.cuda.synchronize(device)

    result = {
        "schema_version": 1,
        "scope": f"O1_codec_{config.mode}",
        "paper0_commit": paper0_commit,
        "slurm_job_id": str(slurm_job_id),
        "config": config.to_record(),
        "parameter_count": int(parameter_count),
        "latent_shape": list(latent_shape(CODEC_CONFIGS[config.codec])),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_step,
        "selected_epoch": selected_epoch,
        "selected_validation_equal_channel_mae": selected_loss,
        "final_validation_equal_channel_mae": final_validation_loss,
        "final_validation_mae_by_channel": dict(
            zip(FAMILY_FIELDS[config.family], final_validation_by_channel)
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
        "wall_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "strict_cuda_bitwise_determinism_claimed": False,
        "cudnn_deterministic_requested": True,
        "tf32_allowed": False,
        "physics_derived_loss_used": False,
        "simulation_data_read": True,
        "development_run": "85604",
        "held_out_85606_read": False,
    }
    write_strict_json_atomic(output / "result.json", result)
    return result
