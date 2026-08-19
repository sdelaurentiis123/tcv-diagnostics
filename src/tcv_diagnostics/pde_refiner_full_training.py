"""Full-only 85604 training for the frozen B4 PDE-Refiner experiment.

The bounded smoke implementation remains unchanged in
``pde_refiner_training``.  This module imports its verified mechanics but
exposes only the separately authorized 100-epoch seed-1701 budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW

from .codec_training import save_torch_atomic, seed_everything, sha256_path
from .model_data import write_strict_json_atomic
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog, epoch_order
from .models.o2 import O2ViTConfig
from .models.pde_refiner import PDERefinerConfig
from .o2_training import scale_accumulated_gradients
from .o2_training_data import OneStepWindowDataset
from .pde_refiner_training import (
    B4_LATENT_SHAPE,
    B4_TRAINING_LEVEL_SEED,
    B4_TRAINING_NOISE_SEED,
    B4_VALIDATION_SEED_BANK_SEED,
    RefinerParentArtifacts,
    TransitionEMA,
    _build_model,
    _fixed_latent_probe,
    _group_gradient_norm,
    _module_state_sha256,
    _parameter_groups,
    _preoptimization_parent_identity,
    _reload_selected_model,
    _window_loader,
    refinement_objective,
    save_numpy_exclusive,
    validation_decoded_mae,
    validation_seed_bank,
)


B4_FULL_SEED = 1701
B4_FULL_EPOCHS = 100
B4_FULL_TRAIN_TARGETS = tuple(range(2, 432))
B4_FULL_VALIDATION_TARGETS = tuple(range(498, 624))
B4_FULL_LEVEL_RAW_SHA256 = (
    "ac370fa17291d8bd4c36ac4d451f78e63250c19ad77cf70a3f8403465e339ff6"
)
B4_FULL_LEVEL_COUNTS = (10_831, 10_680, 10_722, 10_767)
B4_VALIDATION_BANK_NPY_SHA256 = (
    "127936e25054925f4b114d5b174cbe876847555ffd0963ca54ce0e6c72f29884"
)


@dataclass(frozen=True)
class PDERefinerFullConfig:
    """The one immutable full B4 training budget."""

    seed: int
    epochs: int
    train_target_start: int
    train_target_stop: int
    validation_target_start: int
    validation_target_stop: int
    microbatch_targets: int
    gradient_accumulation_targets: int
    validation_members: int
    peak_learning_rate: float
    minimum_learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    gradient_clip: float
    ema_decay: float
    training_precision: str
    training_level_seed: int
    training_noise_seed: int
    validation_seed_bank_seed: int
    validation_every_completed_epochs: int

    @classmethod
    def frozen(cls, *, seed: int) -> "PDERefinerFullConfig":
        if int(seed) != B4_FULL_SEED:
            raise ValueError("the full B4 pilot is fixed to seed 1701")
        return cls(
            seed=int(seed),
            epochs=B4_FULL_EPOCHS,
            train_target_start=2,
            train_target_stop=432,
            validation_target_start=498,
            validation_target_stop=624,
            microbatch_targets=1,
            gradient_accumulation_targets=16,
            validation_members=2,
            peak_learning_rate=1.0e-4,
            minimum_learning_rate=1.0e-6,
            betas=(0.9, 0.999),
            weight_decay=1.0e-5,
            gradient_clip=1.0,
            ema_decay=0.995,
            training_precision="float32_no_autocast_TF32_disabled",
            training_level_seed=B4_TRAINING_LEVEL_SEED,
            training_noise_seed=B4_TRAINING_NOISE_SEED,
            validation_seed_bank_seed=B4_VALIDATION_SEED_BANK_SEED,
            validation_every_completed_epochs=5,
        )

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
        record.update(
            {
                "mode": "full",
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
                "scheduler": "cosine_inclusive_endpoints_per_optimizer_update",
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "validation_completed_epochs": list(
                    self.validation_completed_epochs
                ),
                "training_loss": (
                    "uniform_level_explicit_standardized_latent_MSE"
                ),
                "checkpoint_selection": (
                    "earliest_lowest_fixed_seed_M2_ensemble_mean_equal_channel_"
                    "decoded_standardized_field_MAE_level3_EMA_after_full_budget"
                ),
                "physics_derived_loss_allowed": False,
                "absolute_time_input_allowed": False,
                "early_stopping": False,
                "scientific_result": False,
                "full_training_authorized": True,
                "training_complete_is_scientific_acceptance": False,
            }
        )
        return record


def full_training_levels(config: PDERefinerFullConfig) -> np.ndarray:
    """Materialize and verify the prospectively frozen full level matrix."""

    if config != PDERefinerFullConfig.frozen(seed=B4_FULL_SEED):
        raise ValueError("B4 levels require the frozen full config")
    generator = np.random.Generator(np.random.PCG64(config.training_level_seed))
    values = generator.integers(
        0,
        4,
        size=(config.epochs, len(config.train_targets)),
        dtype=np.int64,
    )
    values = np.ascontiguousarray(values, dtype=np.int64)
    observed_sha = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    observed_counts = tuple(
        map(int, np.bincount(values.reshape(-1), minlength=4))
    )
    if observed_sha != B4_FULL_LEVEL_RAW_SHA256:
        raise RuntimeError("full B4 level-matrix bytes differ")
    if observed_counts != B4_FULL_LEVEL_COUNTS:
        raise RuntimeError("full B4 level-matrix counts differ")
    if any(set(map(int, np.unique(row))) != {0, 1, 2, 3} for row in values):
        raise RuntimeError("a full B4 epoch omits a refinement level")
    return values


def full_learning_rate(
    config: PDERefinerFullConfig,
    zero_based_update: int,
) -> float:
    """Inclusive-endpoint cosine schedule for update index 0 through 2699."""

    if config != PDERefinerFullConfig.frozen(seed=B4_FULL_SEED):
        raise ValueError("B4 learning rate requires the frozen full config")
    update = int(zero_based_update)
    if update < 0 or update >= config.total_optimizer_steps:
        raise ValueError("B4 optimizer update index is outside 0..2699")
    progress = update / (config.total_optimizer_steps - 1)
    return config.minimum_learning_rate + 0.5 * (
        config.peak_learning_rate - config.minimum_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _tensor_bytes_sha256(value: Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _checkpoint_payload(
    *,
    kind: str,
    transition_state: Mapping[str, Tensor],
    config: PDERefinerFullConfig,
    model_config: O2ViTConfig,
    refiner_config: PDERefinerConfig,
    artifacts: RefinerParentArtifacts,
    normalization: Mapping[str, Any],
    load_audit: Any,
    parameter_groups: Any,
    validation_seed_bank_path: Path,
    validation_seed_bank_sha256: str,
    training_levels_path: Path,
    training_levels_sha256: str,
    epoch: int,
    global_step: int,
    validation: Mapping[str, Any],
    ema_updates: int,
    paper0_commit: str,
    reload_probe: Mapping[str, Any] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    raw_transition_state: Mapping[str, Tensor] | None = None,
    training_noise_generator_state: Tensor | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
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
            "raw_C_order_sha256": B4_FULL_LEVEL_RAW_SHA256,
            "seed": B4_TRAINING_LEVEL_SEED,
            "shape": [100, 430],
            "counts": list(B4_FULL_LEVEL_COUNTS),
        },
        "epoch": int(epoch),
        "completed_epoch": int(epoch) + 1,
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
        payload["training_noise_generator_state_sha256"] = _tensor_bytes_sha256(
            training_noise_generator_state
        )
    return payload


def _train_pde_refiner_full(
    *,
    config: PDERefinerFullConfig,
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
    frozen = PDERefinerFullConfig.frozen(seed=B4_FULL_SEED)
    if config != frozen:
        raise ValueError("B4 config differs from the frozen 100-epoch budget")
    output = Path(output_directory)
    if "85606" in str(output).lower():
        raise ValueError("held-out paths are prohibited")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full B4 run {output}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full B4 training requires one CUDA worker")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("full B4 training requires highest float32 precision")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("full B4 training prohibits TF32")
    output.mkdir(parents=True)

    seed_everything(config.seed)
    model, load_audit, parent_payload, normalization = _build_model(
        config=config,  # type: ignore[arg-type]
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
        raise RuntimeError("copied full B4 latent normalization changed bytes")

    level_values = full_training_levels(config)
    level_path = output / "training_levels.npy"
    level_sha256 = save_numpy_exclusive(level_path, level_values)
    seed_values = validation_seed_bank(seed=config.validation_seed_bank_seed)
    seed_path = output / "validation_seed_bank.npy"
    seed_sha256 = save_numpy_exclusive(seed_path, seed_values)
    if seed_sha256 != B4_VALIDATION_BANK_NPY_SHA256:
        raise RuntimeError("full B4 selection seed-bank artifact differs")

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
        lr=config.peak_learning_rate,
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
            "counts": list(B4_FULL_LEVEL_COUNTS),
            "raw_C_order_sha256": B4_FULL_LEVEL_RAW_SHA256,
            "npy_sha256": level_sha256,
        },
        "validation_seed_bank": {
            "seed": config.validation_seed_bank_seed,
            "shape": list(seed_values.shape),
            "dtype": str(seed_values.dtype),
            "npy_sha256": seed_sha256,
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
    final_validation: dict[str, Any] | None = None
    global_step = 0
    parent_gradient_seen = False
    refinement_gradient_seen = False
    observed_level_counts = np.zeros(4, dtype=np.int64)
    validation_count = 0
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
            last_learning_rate = math.nan

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
                    raise RuntimeError("full B4 training latent shape differs")
                if level == 0:
                    noise = torch.zeros(
                        latent_shape,
                        device=device,
                        dtype=torch.float32,
                    )
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
                        f"invalid full B4 loss at epoch {epoch}, microstep {microstep}"
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
                        raise FloatingPointError("invalid full B4 parent gradient")
                    if not torch.isfinite(refinement_norm) or refinement_norm <= 0:
                        raise FloatingPointError("invalid full B4 refinement gradient")
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        groups.all_parameters,
                        config.gradient_clip,
                    )
                    if not torch.isfinite(total_norm):
                        raise FloatingPointError(
                            "non-finite full B4 total gradient norm"
                        )
                    last_learning_rate = full_learning_rate(config, global_step)
                    for group in optimizer.param_groups:
                        group["lr"] = last_learning_rate
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
                raise RuntimeError("full B4 epoch left unstepped gradients")
            if examples != len(config.train_targets):
                raise RuntimeError("full B4 epoch did not consume all targets")
            if np.any(train_count_by_level == 0):
                raise RuntimeError("full B4 epoch omitted a refinement level")
            if global_step != (epoch + 1) * config.optimizer_steps_per_epoch:
                raise RuntimeError("full B4 optimizer-step count differs")
            if ema.updates != global_step:
                raise RuntimeError("full B4 EMA update count differs")

            validation: dict[str, Any] | None = None
            completed_epoch = epoch + 1
            if completed_epoch in config.validation_completed_epochs:
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
                validation_count += 1
                final_validation = dict(validation)

            epoch_record: dict[str, Any] = {
                "epoch": epoch,
                "completed_epoch": completed_epoch,
                "examples": examples,
                "global_step": global_step,
                "learning_rate": last_learning_rate,
                "EMA_decay": config.ema_decay,
                "EMA_updates": ema.updates,
                "train_standardized_latent_MSE": train_loss_sum / examples,
                "train_MSE_by_level": {
                    str(level): float(
                        train_loss_by_level[level] / train_count_by_level[level]
                    )
                    for level in range(4)
                },
                "train_count_by_level": {
                    str(level): int(train_count_by_level[level])
                    for level in range(4)
                },
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
                "validation_performed": validation is not None,
                "validation": validation,
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
        or final_validation is None
    ):
        validation_dataset.close()
        raise RuntimeError("full B4 completed without a selected checkpoint")
    if validation_count != len(config.validation_completed_epochs):
        validation_dataset.close()
        raise RuntimeError("full B4 validation-candidate count differs")
    if global_step != config.total_optimizer_steps or ema.updates != global_step:
        validation_dataset.close()
        raise RuntimeError("full B4 final optimizer accounting differs")

    save_torch_atomic(
        selected_path,
        _checkpoint_payload(
            kind="selected_B4_PDE_Refiner_transition",
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
            reload_probe=selected_probe,
        ),
    )
    final_noise_state = training_noise_generator.get_state()
    save_torch_atomic(
        final_path,
        _checkpoint_payload(
            kind="final_B4_PDE_Refiner_full_training_state",
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
            optimizer_state=optimizer.state_dict(),
            training_noise_generator_state=final_noise_state,
        ),
    )

    restored = _reload_selected_model(
        selected_checkpoint=selected_path,
        artifacts=artifacts,
        config=config,  # type: ignore[arg-type]
        model_config=model_config,
        refiner_config=refiner_config,
        device=device,
    )
    observed_probe = _fixed_latent_probe(
        model=restored,
        dataset=validation_dataset,
        seed_bank=seed_values,
        device=device,
    )
    reload_latent_exact = bool(
        torch.equal(
            selected_probe["standardized_latent_stages"],
            observed_probe["standardized_latent_stages"],
        )
    )
    reload_forecast_exact = bool(
        torch.equal(
            selected_probe["final_forecast"],
            observed_probe["final_forecast"],
        )
    )
    validation_dataset.close()
    if not reload_latent_exact or not reload_forecast_exact:
        raise RuntimeError("full B4 selected checkpoint reload differs")

    codec_final_digest = _module_state_sha256(model.codec)
    codec_unchanged = codec_initial_digest == codec_final_digest
    if not codec_unchanged:
        raise RuntimeError("frozen codec changed during full B4 training")
    if not parent_gradient_seen or not refinement_gradient_seen:
        raise RuntimeError("full B4 did not exercise both parameter families")
    if tuple(map(int, observed_level_counts)) != B4_FULL_LEVEL_COUNTS:
        raise RuntimeError("full B4 observed training-level counts differ")
    torch.cuda.synchronize(device)

    result = {
        "schema_version": 1,
        "scope": "B4_PDE_Refiner_H1_seed1701_full_training_85604",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "config": run_record,
        "parameter_count": groups.to_record()["total_parameter_count"],
        "parameter_groups": groups.to_record(),
        "completed_epochs": config.epochs,
        "completed_optimizer_steps": global_step,
        "EMA_updates": ema.updates,
        "validation_candidates_evaluated": validation_count,
        "validation_completed_epochs": list(config.validation_completed_epochs),
        "selected_epoch": selected_epoch,
        "selected_completed_epoch": selected_epoch + 1,
        "selected_optimizer_step": selected_step,
        "selected_validation": selected_validation,
        "final_validation": final_validation,
        "preoptimization_parent_identity": parent_identity,
        "deterministic_parent_load_audit": load_audit.to_record(),
        "checkpoint_reload": {
            "latent_bitwise_exact": reload_latent_exact,
            "forecast_bitwise_exact": reload_forecast_exact,
        },
        "checkpoint_reload_bitwise_exact": bool(
            reload_latent_exact and reload_forecast_exact
        ),
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
            "epochs": config.epochs,
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
            "raw_C_order_sha256": B4_FULL_LEVEL_RAW_SHA256,
            "seed": config.training_level_seed,
            "shape": list(level_values.shape),
        },
        "training_noise_generator": {
            "seed": config.training_noise_seed,
            "final_state_sha256": _tensor_bytes_sha256(final_noise_state),
            "saved_in_final_training_state": True,
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
        "network_calls_per_unamortized_member": 4,
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
        "training_complete_is_scientific_acceptance": False,
        "full_B4_training_authorized": True,
        "scientific_B4_evaluation_performed": False,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    write_strict_json_atomic(output / "result.json", result)
    return result


def train_pde_refiner_full(
    *,
    config: PDERefinerFullConfig,
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
    """Execute only the frozen seed-1701 100-epoch B4 training run."""

    frozen = PDERefinerFullConfig.frozen(seed=B4_FULL_SEED)
    if config != frozen:
        raise ValueError("B4 full config differs from the frozen 100-epoch budget")
    return _train_pde_refiner_full(
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
