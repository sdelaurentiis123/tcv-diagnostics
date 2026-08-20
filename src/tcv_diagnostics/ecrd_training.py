"""Matched training and data-only selection for the ECRD model ladder."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
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

from .b5_residual_edm_full_training import (
    B5EDMFullConfig,
    B5_FULL_TRAINING_NOISE_SEED,
    accumulation_groups,
    full_learning_rate,
    full_training_order,
    full_validation_seed_bank,
    keyed_full_sigma_and_noise,
    update_ema_model,
)
from .b5_residual_edm_training import module_state_sha256, parameter_count
from .codec_training import save_torch_atomic, sha256_path
from .ecrd_data import (
    ECRD_TRAIN_TARGETS,
    ECRD_VALIDATION_TARGETS,
    keyed_ecrd_sigma_and_noise,
    validation_sigma_and_noise_from_uint64,
)
from .model_data import assert_development_path, write_strict_json_atomic
from .models.ecrd import (
    ECRDTransition,
    ECRDUNetConfig,
    MultiscaleNoiseConfig,
)
from .models.field_residual_edm import (
    B5_RESIDUAL_SCALES,
    FieldResidualUNet3D,
    FieldResidualUNetConfig,
    JointFieldResidualEDM,
)


ECRD_MODEL_SEEDS = (1701, 1702, 1703)
ECRD_ARMS = ("B5", "B5-Context", "ECRD", "ECRD-History")
ECRD_AUGMENTATION_SEED = 67_505
ECRD_VALIDATION_BLOCKS = {
    "V00": (498, 540),
    "V01": (540, 582),
    "V02": (582, 624),
}
ECRD_REFERENCE_PARAMETER_COUNT = 11_604_709


class _ResidualDataset(Protocol):
    split: str
    target_frames: tuple[int, ...]

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ECRDTrainingConfig:
    """Frozen full budget plus a bounded engineering-smoke variant."""

    arm: str
    seed: int
    mode: str = "full"
    peak_learning_rate: float = 1.0e-4
    minimum_learning_rate: float = 1.0e-6
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    mean_loss_weight: float = 1.0
    training_noise_seed: int = B5_FULL_TRAINING_NOISE_SEED
    validation_seed_bank_seed: int = 67_503

    def __post_init__(self) -> None:
        if self.arm not in ECRD_ARMS:
            raise ValueError(f"unsupported ECRD training arm {self.arm!r}")
        if int(self.seed) not in ECRD_MODEL_SEEDS:
            raise ValueError("ECRD model seed must be 1701, 1702, or 1703")
        if self.mode not in ("smoke", "full"):
            raise ValueError("ECRD mode must be smoke or full")
        if (
            self.peak_learning_rate != 1.0e-4
            or self.minimum_learning_rate != 1.0e-6
            or self.betas != (0.9, 0.99)
            or self.weight_decay != 0.0
            or self.gradient_clip != 1.0
            or self.ema_decay != 0.999
            or self.mean_loss_weight != 1.0
            or self.training_noise_seed != 67_502
            or self.validation_seed_bank_seed != 67_503
        ):
            raise ValueError("ECRD frozen optimizer/noise contract differs")

    @property
    def epochs(self) -> int:
        return 1 if self.mode == "smoke" else 100

    @property
    def train_targets(self) -> tuple[int, ...]:
        return tuple(range(2, 6)) if self.mode == "smoke" else ECRD_TRAIN_TARGETS

    @property
    def validation_targets(self) -> tuple[int, ...]:
        return tuple(range(498, 502)) if self.mode == "smoke" else ECRD_VALIDATION_TARGETS

    @property
    def accumulation_targets(self) -> int:
        return 2 if self.mode == "smoke" else 4

    @property
    def validation_epochs(self) -> tuple[int, ...]:
        return (1,) if self.mode == "smoke" else tuple(range(5, 101, 5))

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_targets) / self.accumulation_targets)

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    @property
    def target_presentations(self) -> int:
        return self.epochs * len(self.train_targets)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["betas"] = list(self.betas)
        record.update(
            {
                "epochs": self.epochs,
                "train_targets": [self.train_targets[0], self.train_targets[-1] + 1],
                "validation_targets": [
                    self.validation_targets[0],
                    self.validation_targets[-1] + 1,
                ],
                "accumulation_targets": self.accumulation_targets,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "target_presentations": self.target_presentations,
                "validation_epochs": list(self.validation_epochs),
                "physics_derived_loss_allowed": False,
                "early_stopping": False,
                "held_out_85606_access_allowed": False,
            }
        )
        return record


def exact_model_config(arm: str) -> ECRDUNetConfig | FieldResidualUNetConfig:
    """Parameter-only width choice; no result-dependent configuration."""

    if arm == "B5":
        return FieldResidualUNetConfig()
    equivariant = arm != "B5-Context"
    return ECRDUNetConfig(
        arm=arm,
        history_frames=2 if arm == "ECRD-History" else 1,
        base_channels=28,
        preserve_toroidal_resolution=equivariant,
        mean_head=equivariant,
        multiscale_noise=equivariant,
    )


def build_model(
    arm: str,
    *,
    noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> JointFieldResidualEDM | ECRDTransition:
    config = exact_model_config(arm)
    if arm == "B5":
        if not isinstance(config, FieldResidualUNetConfig):
            raise AssertionError("B5 model configuration type drifted")
        return JointFieldResidualEDM(FieldResidualUNet3D(config))
    if not isinstance(config, ECRDUNetConfig):
        raise AssertionError("ECRD model configuration type drifted")
    return ECRDTransition(config, noise_config=noise_config)


def model_config_record(
    arm: str,
    *,
    noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> dict[str, Any]:
    config = exact_model_config(arm)
    if arm == "B5":
        return {
            "arm": "B5",
            "architecture": config.to_record(),
            "noise": {"distribution": "elementwise_standard_normal"},
            "mean_head": False,
        }
    model = build_model(arm, noise_config=noise_config)
    if not isinstance(model, ECRDTransition):
        raise AssertionError("new ECRD arm built a legacy model")
    return model.to_record()


def frozen_parameter_counts() -> dict[str, int]:
    return {arm: parameter_count(build_model(arm)) for arm in ECRD_ARMS}


def validate_parameter_matching(counts: Mapping[str, int]) -> None:
    if set(counts) != set(ECRD_ARMS):
        raise ValueError("ECRD parameter-count arm set differs")
    for arm, count in counts.items():
        relative = abs(int(count) / ECRD_REFERENCE_PARAMETER_COUNT - 1.0)
        if relative > 0.10:
            raise ValueError(f"{arm} parameter count is not matched: {count}")


def _training_order(config: ECRDTrainingConfig) -> np.ndarray:
    if config.mode == "full":
        return full_training_order()
    return np.asarray([config.train_targets], dtype=np.int64)


def _accumulation_groups(
    order: np.ndarray,
    config: ECRDTrainingConfig,
) -> tuple[np.ndarray, ...]:
    values = np.asarray(order, dtype=np.int64)
    if values.shape != (len(config.train_targets),) or set(map(int, values)) != set(
        config.train_targets
    ):
        raise ValueError("ECRD epoch target order differs")
    if config.mode == "full":
        return accumulation_groups(values)
    return tuple(
        values[start : start + config.accumulation_targets]
        for start in range(0, len(values), config.accumulation_targets)
    )


def _learning_rate(config: ECRDTrainingConfig, update: int) -> float:
    if config.mode == "full":
        return full_learning_rate(B5EDMFullConfig(), update)
    if config.total_optimizer_steps == 1:
        return config.peak_learning_rate
    progress = int(update) / (config.total_optimizer_steps - 1)
    return float(
        config.minimum_learning_rate
        + 0.5
        * (config.peak_learning_rate - config.minimum_learning_rate)
        * (1.0 + math.cos(math.pi * progress))
    )


def _dataset_item_tensors(
    dataset: _ResidualDataset,
    target: int,
    *,
    arm: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    position = int(target) - dataset.target_frames[0]
    item = dataset[position]
    if int(item["target_frame_index"]) != int(target):
        raise RuntimeError("ECRD tensor target order differs")
    condition = torch.from_numpy(np.asarray(item["condition"]))[None].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    key = "normalized_residual" if arm == "B5" else "normalized_parent_residual"
    target_tensor = torch.from_numpy(np.asarray(item[key]))[None].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return condition, target_tensor


def _loss(
    model: JointFieldResidualEDM | ECRDTransition,
    *,
    arm: str,
    target: Tensor,
    condition: Tensor,
    sigma: Tensor,
    noise: Tensor,
) -> dict[str, Tensor]:
    if arm == "B5":
        if not isinstance(model, JointFieldResidualEDM):
            raise TypeError("B5 loss received a non-B5 model")
        result = model.training_loss(target, condition, sigma=sigma, noise=noise)
        zero = result.loss.new_zeros(())
        return {
            "objective": result.loss,
            "edm_loss": result.loss,
            "unweighted_edm_mse": result.unweighted_mse,
            "mean_mse": zero,
        }
    if not isinstance(model, ECRDTransition):
        raise TypeError("ECRD loss received a legacy model")
    result = model.training_loss(target, condition, sigma=sigma, noise=noise)
    return {
        "objective": result.loss,
        "edm_loss": result.edm_loss,
        "unweighted_edm_mse": result.unweighted_edm_mse,
        "mean_mse": result.mean_mse,
    }


def _validation_blocks(targets: Sequence[int]) -> dict[str, tuple[int, ...]]:
    values = tuple(int(value) for value in targets)
    if values == ECRD_VALIDATION_TARGETS:
        return {
            name: tuple(range(start, stop))
            for name, (start, stop) in ECRD_VALIDATION_BLOCKS.items()
        }
    return {"SMOKE": values}


@torch.no_grad()
def validation_objective(
    *,
    model: JointFieldResidualEDM | ECRDTransition,
    dataset: _ResidualDataset,
    config: ECRDTrainingConfig,
    seed_bank: np.ndarray,
    device: torch.device,
    noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> dict[str, Any]:
    """Evaluate identical fixed corruptions, preserving the three blocks."""

    targets = config.validation_targets
    if dataset.split != "validation" or tuple(dataset.target_frames) != targets:
        raise ValueError("ECRD validation dataset differs")
    bank = np.asarray(seed_bank)
    if bank.shape != (len(targets), 4) or bank.dtype != np.uint64:
        raise ValueError("ECRD validation seed bank differs")
    was_training = model.training
    model.eval()
    per_target: dict[int, dict[str, float]] = {}
    started = time.perf_counter()
    for position, target_frame in enumerate(targets):
        condition, target = _dataset_item_tensors(
            dataset, target_frame, arm=config.arm, device=device
        )
        sigmas: list[float] = []
        noises: list[np.ndarray] = []
        for seed in bank[position]:
            sigma, noise = validation_sigma_and_noise_from_uint64(
                seed,
                multiscale=config.arm in ("ECRD", "ECRD-History"),
                config=noise_config,
            )
            sigmas.append(float(sigma))
            noises.append(noise)
        sigma_tensor = torch.tensor(sigmas, device=device, dtype=torch.float32)
        noise_tensor = torch.from_numpy(np.stack(noises)).to(
            device=device, dtype=torch.float32
        )
        values = _loss(
            model,
            arm=config.arm,
            target=target.expand(4, *target.shape[1:]).contiguous(),
            condition=condition.expand(4, *condition.shape[1:]).contiguous(),
            sigma=sigma_tensor,
            noise=noise_tensor,
        )
        per_target[target_frame] = {
            name: float(value.detach().cpu()) for name, value in values.items()
        }
    model.train(was_training)
    blocks: dict[str, dict[str, float | int | list[int]]] = {}
    for name, block_targets in _validation_blocks(targets).items():
        records = [per_target[target] for target in block_targets]
        blocks[name] = {
            "target_frames": [block_targets[0], block_targets[-1] + 1],
            "target_count": len(block_targets),
            **{
                metric: float(np.mean([record[metric] for record in records]))
                for metric in (
                    "objective",
                    "edm_loss",
                    "unweighted_edm_mse",
                    "mean_mse",
                )
            },
        }
    aggregate = {
        metric: float(np.mean([record[metric] for record in per_target.values()]))
        for metric in ("objective", "edm_loss", "unweighted_edm_mse", "mean_mse")
    }
    # The checkpoint score is explicitly the unweighted mean of block means.
    checkpoint_score = float(
        np.mean([float(record["objective"]) for record in blocks.values()])
    )
    return {
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "probes_per_target": 4,
        "blocks": blocks,
        "aggregate": aggregate,
        "checkpoint_score": checkpoint_score,
        "checkpoint_score_definition": "unweighted_mean_of_block_mean_objectives",
        "wall_seconds": float(time.perf_counter() - started),
    }


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


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def _reload_probe(
    *,
    model: JointFieldResidualEDM | ECRDTransition,
    dataset: _ResidualDataset,
    config: ECRDTrainingConfig,
    device: torch.device,
    noise_config: MultiscaleNoiseConfig,
) -> Tensor:
    target_frame = config.validation_targets[0]
    condition, target = _dataset_item_tensors(
        dataset, target_frame, arm=config.arm, device=device
    )
    sigma, noise = validation_sigma_and_noise_from_uint64(
        67_504,
        multiscale=config.arm in ("ECRD", "ECRD-History"),
        config=noise_config,
    )
    sigma_tensor = torch.tensor([float(sigma)], device=device)
    noise_tensor = torch.from_numpy(noise)[None].to(device)
    values = _loss(
        model,
        arm=config.arm,
        target=target,
        condition=condition,
        sigma=sigma_tensor,
        noise=noise_tensor,
    )
    parts = [values[name].reshape(1) for name in sorted(values)]
    if isinstance(model, ECRDTransition):
        with torch.inference_mode():
            parts.append(model.mean_correction_normalized(condition).flatten().to(torch.float32))
    return torch.cat(parts).detach().to("cpu", torch.float32)


def train_ecrd_arm(
    *,
    training_dataset: _ResidualDataset,
    validation_dataset: _ResidualDataset,
    output: Path,
    device: torch.device,
    paper0_commit: str,
    slurm_job_id: str,
    authority: Mapping[str, Any],
    config: ECRDTrainingConfig,
    noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
    on_epoch: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train exactly one arm/seed under the frozen matched budget."""

    destination = Path(output)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if tuple(training_dataset.target_frames) != config.train_targets:
        raise ValueError("ECRD training dataset targets differ")
    if tuple(validation_dataset.target_frames) != config.validation_targets:
        raise ValueError("ECRD validation dataset targets differ")
    if training_dataset.split != "train" or validation_dataset.split != "validation":
        raise ValueError("ECRD dataset split identity differs")
    if any("85606" in str(value).lower() for value in authority.values()):
        raise ValueError("ECRD authority mentions the held-out run")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ECRD training requires an allocated CUDA device")
    destination.mkdir(parents=True)
    candidates_directory = destination / "candidates"
    candidates_directory.mkdir()
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    order = _training_order(config)
    order_path = _save_npy_atomic(destination / "training_order.npy", order)
    full_bank = full_validation_seed_bank()
    bank_start = config.validation_targets[0] - ECRD_VALIDATION_TARGETS[0]
    validation_bank = np.ascontiguousarray(
        full_bank[bank_start : bank_start + len(config.validation_targets)],
        dtype=np.uint64,
    )
    validation_bank_path = _save_npy_atomic(
        destination / "validation_seed_bank.npy", validation_bank
    )

    model_record = model_config_record(config.arm, noise_config=noise_config)
    raw = build_model(config.arm, noise_config=noise_config).to(device, torch.float32)
    count = parameter_count(raw)
    validate_parameter_matching({**frozen_parameter_counts(), config.arm: count})
    initial_sha = module_state_sha256(raw)
    ema = copy.deepcopy(raw).to(device, torch.float32)
    ema.eval()
    ema.requires_grad_(False)
    raw.train()
    optimizer = AdamW(
        raw.parameters(),
        lr=config.peak_learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    run_config_path = destination / "config.json"
    write_strict_json_atomic(
        run_config_path,
        {
            "schema_version": 1,
            "scope": "ECRD_matched_model_development_85604",
            "paper0_commit": str(paper0_commit),
            "slurm_job_id": str(slurm_job_id),
            "training": config.to_record(),
            "model": model_record,
            "parameter_count": count,
            "residual_scales": list(B5_RESIDUAL_SCALES),
            "authority": dict(authority),
        },
    )

    history_path = destination / "history.jsonl"
    candidates: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    global_update = 0
    for epoch_zero, epoch_order in enumerate(order):
        completed_epoch = epoch_zero + 1
        if hasattr(training_dataset, "set_epoch"):
            training_dataset.set_epoch(epoch_zero)
        epoch_started = time.perf_counter()
        raw.train()
        metrics = {
            "objective": [],
            "edm_loss": [],
            "unweighted_edm_mse": [],
            "mean_mse": [],
        }
        gradient_norms: list[float] = []
        learning_rates: list[float] = []
        for group in _accumulation_groups(epoch_order, config):
            optimizer.zero_grad(set_to_none=True)
            group_size = len(group)
            for target_frame_value in group:
                target_frame = int(target_frame_value)
                condition, target = _dataset_item_tensors(
                    training_dataset,
                    target_frame,
                    arm=config.arm,
                    device=device,
                )
                if config.arm == "B5":
                    sigma, noise = keyed_full_sigma_and_noise(
                        seed=config.training_noise_seed,
                        epoch_zero_based=epoch_zero,
                        target_frame=target_frame,
                    )
                else:
                    sigma, noise = keyed_ecrd_sigma_and_noise(
                        base_seed=config.training_noise_seed,
                        epoch_zero_based=epoch_zero,
                        target_frame=target_frame,
                        multiscale=config.arm in ("ECRD", "ECRD-History"),
                        config=noise_config,
                    )
                sigma_tensor = torch.tensor([float(sigma)], device=device)
                noise_tensor = torch.from_numpy(noise)[None].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = _loss(
                        raw,
                        arm=config.arm,
                        target=target,
                        condition=condition,
                        sigma=sigma_tensor,
                        noise=noise_tensor,
                    )
                if not torch.isfinite(losses["objective"]):
                    raise FloatingPointError(
                        f"non-finite {config.arm} loss at epoch {completed_epoch} target {target_frame}"
                    )
                (losses["objective"] / group_size).backward()
                for name, value in losses.items():
                    metrics[name].append(float(value.detach().cpu()))
            preclip = torch.nn.utils.clip_grad_norm_(raw.parameters(), config.gradient_clip)
            if not torch.isfinite(preclip):
                raise FloatingPointError("ECRD gradient norm is non-finite")
            learning_rate = _learning_rate(config, global_update)
            for group_record in optimizer.param_groups:
                group_record["lr"] = learning_rate
            optimizer.step()
            update_ema_model(ema, raw, decay=config.ema_decay)
            gradient_norms.append(float(preclip.detach().cpu()))
            learning_rates.append(float(learning_rate))
            global_update += 1

        torch.cuda.synchronize(device)
        validation: dict[str, Any] | None = None
        candidate: dict[str, Any] | None = None
        if completed_epoch in config.validation_epochs:
            validation = validation_objective(
                model=ema,
                dataset=validation_dataset,
                config=config,
                seed_bank=validation_bank,
                device=device,
                noise_config=noise_config,
            )
            candidate_path = candidates_directory / f"ema_epoch_{completed_epoch:03d}.pt"
            payload = {
                "schema_version": 1,
                "kind": "ECRD_EMA_validation_candidate",
                "paper0_commit": str(paper0_commit),
                "slurm_job_id": str(slurm_job_id),
                "completed_epoch": completed_epoch,
                "global_optimizer_step": global_update,
                "training": config.to_record(),
                "model": model_record,
                "validation": validation,
                "model_state": {
                    name: value.detach().to("cpu")
                    for name, value in ema.state_dict().items()
                },
                "physics_metric_used_for_selection": False,
                "held_out_85606_read": False,
            }
            save_torch_atomic(candidate_path, payload)
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
            "train_target_count": len(metrics["objective"]),
            **{
                f"train_mean_{name}": float(np.mean(values, dtype=np.float64))
                for name, values in metrics.items()
            },
            "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
            "maximum_preclip_gradient_norm": float(max(gradient_norms)),
            "first_learning_rate": learning_rates[0],
            "last_learning_rate": learning_rates[-1],
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

    if global_update != config.total_optimizer_steps or len(records) != config.epochs:
        raise RuntimeError("ECRD training budget did not complete")
    if len(candidates) != len(config.validation_epochs):
        raise RuntimeError("ECRD validation candidate count differs")
    selected = min(
        candidates,
        key=lambda item: (
            float(item["validation"]["checkpoint_score"]),
            int(item["completed_epoch"]),
        ),
    )
    selected_payload = _load_torch(Path(selected["path"]))
    restored = build_model(config.arm, noise_config=noise_config).to(device, torch.float32)
    restored.load_state_dict(selected_payload["model_state"], strict=True)
    restored.eval()
    expected_reload = _reload_probe(
        model=restored,
        dataset=validation_dataset,
        config=config,
        device=device,
        noise_config=noise_config,
    )
    selected_path = destination / "selected.pt"
    final_payload = {
        "schema_version": 1,
        "kind": "ECRD_selected_EMA_checkpoint",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "training": config.to_record(),
        "model": model_record,
        "parameter_count": count,
        "residual_scales": list(B5_RESIDUAL_SCALES),
        "selected_completed_epoch": int(selected["completed_epoch"]),
        "selected_optimizer_step": int(selected["global_optimizer_step"]),
        "selected_validation": selected["validation"],
        "source_candidate": {"path": selected["path"], "sha256": selected["sha256"]},
        "model_state": selected_payload["model_state"],
        "physics_metric_used_for_selection": False,
        "held_out_85606_read": False,
    }
    save_torch_atomic(selected_path, final_payload)
    reloaded_payload = _load_torch(selected_path)
    reloaded = build_model(config.arm, noise_config=noise_config).to(device, torch.float32)
    reloaded.load_state_dict(reloaded_payload["model_state"], strict=True)
    reloaded.eval()
    observed_reload = _reload_probe(
        model=reloaded,
        dataset=validation_dataset,
        config=config,
        device=device,
        noise_config=noise_config,
    )
    reload_exact = bool(torch.equal(expected_reload, observed_reload))
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "schema_version": 1,
        "scope": "ECRD_matched_model_development_training_85604",
        "status": "training_completed_checkpoint_selected",
        "mode": config.mode,
        "arm": config.arm,
        "seed": config.seed,
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "development_run": "85604",
        "training": config.to_record(),
        "model": model_record,
        "parameter_count": count,
        "initial_model_state_sha256": initial_sha,
        "completed_epochs": len(records),
        "completed_optimizer_steps": global_update,
        "target_presentations": config.target_presentations,
        "candidate_count": len(candidates),
        "selected_completed_epoch": int(selected["completed_epoch"]),
        "selected_validation": selected["validation"],
        "checkpoint_reload_bitwise_exact": reload_exact,
        "selected_model_state_sha256": module_state_sha256(reloaded),
        "wall_seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_bytes": peak_bytes,
        "peak_cuda_memory_GiB": float(peak_bytes / 1024**3),
        "artifacts": {
            "config": {"path": str(run_config_path), "sha256": sha256_path(run_config_path)},
            "training_order": {"path": str(order_path), "sha256": sha256_path(order_path)},
            "validation_seed_bank": {
                "path": str(validation_bank_path),
                "sha256": sha256_path(validation_bank_path),
            },
            "history": {"path": str(history_path), "sha256": sha256_path(history_path)},
            "selected_checkpoint": {
                "path": str(selected_path),
                "sha256": sha256_path(selected_path),
            },
            "candidate_checkpoints": candidates,
        },
        "training_performed": True,
        "validation_frames_read": True,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    write_strict_json_atomic(destination / "result.json", result)
    return result
