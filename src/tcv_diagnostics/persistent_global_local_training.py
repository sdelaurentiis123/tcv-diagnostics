"""Training utilities for the bounded old-85604 persistent global--local pilot."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
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

from .autoregressive_training import feedback_loss_weights, state_rms_normalized_mse
from .b5_residual_edm_full_training import update_ema_model
from .b5_residual_edm_training import module_state_sha256, parameter_count
from .codec_training import save_torch_atomic, sha256_path
from .model_data import assert_development_path, write_strict_json_atomic
from .models.persistent_global_local import (
    PGL_FIELD_ORDER,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
    sample_persistent_global_local_noise,
)


PGL_SEED = 1702
PGL_HORIZON = 4
PGL_MEAN_STEP_WEIGHTS = feedback_loss_weights(
    horizon=PGL_HORIZON, direct_one_step_weight=0.5
)
PGL_VALIDATION_BLOCKS = {
    "V00": (496, 537),
    "V01": (537, 578),
    "V02": (578, 620),
}
PGL_NOISE_BASE_SEED = 856_041_702


class _WindowDataset(Protocol):
    split: str
    horizon: int
    windows: Sequence[Any]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    def set_epoch(self, epoch: int) -> None: ...


class _Forecast(Protocol):
    volume: Tensor


class _MeanOperator(Protocol):
    def forecast(self, context: Tensor, lead_steps: Tensor) -> _Forecast: ...


@dataclass(frozen=True)
class PersistentPilotTrainingConfig:
    """Frozen smoke or one-seed pilot budget."""

    mode: str
    seed: int = PGL_SEED
    stochastic_peak_learning_rate: float = 1.0e-4
    mean_peak_learning_rate: float = 1.0e-5
    stochastic_minimum_learning_rate: float = 1.0e-6
    mean_minimum_learning_rate: float = 1.0e-6
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1.0e-4
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    accumulation_windows: int = 2
    validation_probes: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("smoke", "pilot"):
            raise ValueError("persistent pilot mode must be smoke or pilot")
        if int(self.seed) != PGL_SEED:
            raise ValueError("the bounded persistent pilot uses seed 1702")
        expected = {
            "stochastic_peak_learning_rate": 1.0e-4,
            "mean_peak_learning_rate": 1.0e-5,
            "stochastic_minimum_learning_rate": 1.0e-6,
            "mean_minimum_learning_rate": 1.0e-6,
            "betas": (0.9, 0.99),
            "weight_decay": 1.0e-4,
            "warmup_fraction": 0.05,
            "gradient_clip": 1.0,
            "ema_decay": 0.999,
            "accumulation_windows": 2,
            "validation_probes": 2,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("persistent pilot optimization contract differs")

    @property
    def epochs(self) -> int:
        return 1 if self.mode == "smoke" else 20

    @property
    def expected_training_windows(self) -> int:
        return 8 if self.mode == "smoke" else 428

    @property
    def expected_validation_windows(self) -> int:
        return 4 if self.mode == "smoke" else 124

    @property
    def validation_epochs(self) -> tuple[int, ...]:
        return (1,) if self.mode == "smoke" else tuple(range(2, 21, 2))

    @property
    def optimizer_steps_per_epoch(self) -> int:
        return math.ceil(self.expected_training_windows / self.accumulation_windows)

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.optimizer_steps_per_epoch

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["betas"] = list(self.betas)
        record.update(
            {
                "epochs": self.epochs,
                "training_windows": self.expected_training_windows,
                "validation_windows": self.expected_validation_windows,
                "optimizer_steps_per_epoch": self.optimizer_steps_per_epoch,
                "total_optimizer_steps": self.total_optimizer_steps,
                "validation_epochs": list(self.validation_epochs),
                "mean_step_weights": list(PGL_MEAN_STEP_WEIGHTS),
                "mean_feedback_gradient": "detached_between_steps",
                "diffusion_gradient_to_mean": False,
                "physics_derived_loss_used": False,
                "held_out_85606_access_allowed": False,
                "new_nersc_data_access_allowed": False,
            }
        )
        return record


def tensor_window(
    item: Mapping[str, Any], device: torch.device
) -> tuple[Tensor, Tensor, int]:
    context = torch.from_numpy(np.asarray(item["context"]))[None].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    targets = torch.from_numpy(np.asarray(item["targets"]))[None].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    current_frame = int(item["current_frame_index"])
    if context.ndim != 6 or context.shape[1:3] != (1, 5):
        raise ValueError("persistent context tensor shape differs")
    if targets.shape != (
        context.shape[0], PGL_HORIZON, 5, *context.shape[-3:]
    ):
        raise ValueError("persistent target trajectory shape differs")
    expected = tuple(range(current_frame + 1, current_frame + 5))
    if tuple(int(value) for value in item["target_frame_indices"]) != expected:
        raise ValueError("persistent target-frame sequence differs")
    return context, targets, current_frame


def mean_forecast_trajectory(
    model: _MeanOperator,
    context: Tensor,
    *,
    horizon: int = PGL_HORIZON,
    detach_feedback: bool = True,
) -> Tensor:
    """Roll the trainable deterministic mean with no future-truth input."""

    if context.ndim != 6 or context.shape[1:3] != (1, 5):
        raise ValueError("persistent mean context must be [B,1,5,x,y,z]")
    current = context
    outputs: list[Tensor] = []
    for _ in range(int(horizon)):
        lead = torch.ones(
            current.shape[0], device=current.device, dtype=current.dtype
        )
        forecast = model.forecast(current, lead)
        if forecast.volume.shape != current[:, -1].shape:
            raise ValueError("persistent mean forecast shape differs")
        outputs.append(forecast.volume)
        feedback = forecast.volume.detach() if detach_feedback else forecast.volume
        current = feedback.unsqueeze(1)
    return torch.stack(outputs, dim=1)


def weighted_mean_state_loss(
    mean: Tensor,
    targets: Tensor,
    derivative_rms: Tensor,
) -> tuple[Tensor, Tensor]:
    if mean.shape != targets.shape or mean.ndim != 6:
        raise ValueError("persistent mean and target trajectories differ")
    losses: list[Tensor] = []
    per_field: list[Tensor] = []
    for step, weight in enumerate(PGL_MEAN_STEP_WEIGHTS):
        value, fields = state_rms_normalized_mse(
            mean[:, step], targets[:, step], derivative_rms
        )
        losses.append(float(weight) * value)
        per_field.append(fields)
    return torch.stack(losses).sum(), torch.stack(per_field, dim=0)


@torch.no_grad()
def fit_parent_residual_scales(
    *,
    parent_mean: _MeanOperator,
    dataset: _WindowDataset,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    """Fit one nonphysics RMS per future step and field on all train windows."""

    if dataset.split != "train" or dataset.horizon != PGL_HORIZON:
        raise ValueError("persistent scale-fit dataset differs")
    if len(dataset) != 428:
        raise ValueError("persistent scales require all 428 training windows")
    sums = torch.zeros((PGL_HORIZON, 5), dtype=torch.float64)
    count = 0
    parent_mean.eval()
    started = time.perf_counter()
    for index in range(len(dataset)):
        context, targets, _ = tensor_window(dataset[index], device)
        mean = mean_forecast_trajectory(parent_mean, context)
        residual = (targets - mean).double()
        sums += residual.square().sum(dim=(0, 3, 4, 5)).cpu()
        count += int(residual.shape[0] * residual.shape[3] * residual.shape[4] * residual.shape[5])
    scales = torch.sqrt(sums / count).float()
    if scales.shape != (4, 5) or not torch.all(torch.isfinite(scales) & (scales > 0.0)):
        raise FloatingPointError("persistent residual scales are invalid")
    return scales, {
        "schema_version": 1,
        "scope": "old_85604_persistent_global_local_parent_residual_scales",
        "development_run": "85604",
        "training_frames": [0, 432],
        "training_window_count": len(dataset),
        "horizon": PGL_HORIZON,
        "fields": list(PGL_FIELD_ORDER),
        "values": scales.tolist(),
        "statistic": "root_mean_square_parent_state_residual_by_future_step_and_field",
        "fit_split": "train_only",
        "parent_feedback": "detached_between_steps",
        "physics_derived_quantity": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "wall_seconds": float(time.perf_counter() - started),
    }


def _keyed_uint64(*values: int) -> int:
    payload = ":".join(str(int(value)) for value in values).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def keyed_sigma_and_noise(
    *,
    base_seed: int,
    epoch_zero_based: int,
    current_frame: int,
    probe: int,
    reference: Tensor,
    noise_config: PersistentNoiseConfig,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> tuple[Tensor, Tensor, int]:
    """Expand a stable key into one joint sigma and structured trajectory noise."""

    sigma_seed = _keyed_uint64(
        base_seed, epoch_zero_based, current_frame, probe, 0x51_6D_61
    )
    noise_seed = _keyed_uint64(
        base_seed, epoch_zero_based, current_frame, probe, 0x6E_6F_69
    )
    rng = np.random.Generator(np.random.PCG64(sigma_seed))
    sigma_value = float(np.exp(p_mean + p_std * rng.standard_normal()))
    generator = torch.Generator(device=reference.device).manual_seed(noise_seed)
    noise = sample_persistent_global_local_noise(
        reference, config=noise_config, generator=generator
    ).total
    sigma = torch.full(
        (reference.shape[0],), sigma_value, device=reference.device, dtype=reference.dtype
    )
    return sigma, noise, noise_seed


def _block_for_current(current: int, *, smoke: bool) -> str:
    if smoke:
        return "SMOKE"
    for name, (start, stop) in PGL_VALIDATION_BLOCKS.items():
        if start <= int(current) < stop:
            return name
    raise ValueError("validation current frame leaves frozen chronological blocks")


@torch.no_grad()
def validation_objective(
    *,
    mean_model: _MeanOperator,
    edm: PersistentGlobalLocalEDM,
    dataset: _WindowDataset,
    derivative_rms: Tensor,
    config: PersistentPilotTrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    if dataset.split != "validation" or dataset.horizon != PGL_HORIZON:
        raise ValueError("persistent validation dataset differs")
    if len(dataset) != config.expected_validation_windows:
        raise ValueError("persistent validation window count differs")
    mean_model.eval()
    edm.eval()
    records: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for index in range(len(dataset)):
        context, targets, current_frame = tensor_window(dataset[index], device)
        mean = mean_forecast_trajectory(mean_model, context)
        mean_loss, mean_by_step_field = weighted_mean_state_loss(
            mean, targets, derivative_rms
        )
        clean = edm.normalize_residual(targets - mean.detach())
        edm_losses: list[float] = []
        unweighted: list[float] = []
        step_field: list[np.ndarray] = []
        for probe in range(config.validation_probes):
            sigma, noise, _ = keyed_sigma_and_noise(
                base_seed=PGL_NOISE_BASE_SEED + 1,
                epoch_zero_based=0,
                current_frame=current_frame,
                probe=probe,
                reference=clean,
                noise_config=edm.noise_config,
            )
            loss = edm.training_loss(
                clean, context[:, -1], mean.detach(), sigma=sigma, noise=noise
            )
            edm_losses.append(float(loss.loss.cpu()))
            unweighted.append(float(loss.unweighted_mse.cpu()))
            step_field.append(loss.per_step_field_mse.cpu().numpy())
        block = _block_for_current(current_frame, smoke=config.mode == "smoke")
        records.setdefault(block, []).append(
            {
                "current_frame": current_frame,
                "mean_loss": float(mean_loss.cpu()),
                "mean_step_field_mse": mean_by_step_field.cpu().numpy(),
                "edm_loss": float(np.mean(edm_losses)),
                "unweighted_edm_mse": float(np.mean(unweighted)),
                "edm_step_field_mse": np.mean(step_field, axis=0),
            }
        )

    blocks: dict[str, Any] = {}
    for name, values in records.items():
        blocks[name] = {
            "window_count": len(values),
            "current_frames": [int(value["current_frame"]) for value in values],
            "mean_state_loss": float(np.mean([value["mean_loss"] for value in values])),
            "edm_loss": float(np.mean([value["edm_loss"] for value in values])),
            "unweighted_edm_mse": float(
                np.mean([value["unweighted_edm_mse"] for value in values])
            ),
            "mean_step_field_mse": np.mean(
                [value["mean_step_field_mse"] for value in values], axis=0
            ).tolist(),
            "edm_step_field_mse": np.mean(
                [value["edm_step_field_mse"] for value in values], axis=0
            ).tolist(),
        }
        blocks[name]["objective"] = (
            blocks[name]["mean_state_loss"] + blocks[name]["edm_loss"]
        )
    expected_blocks = ("SMOKE",) if config.mode == "smoke" else tuple(PGL_VALIDATION_BLOCKS)
    if tuple(blocks) != expected_blocks:
        raise ValueError("persistent validation block identity differs")
    score = float(np.mean([blocks[name]["objective"] for name in expected_blocks]))
    return {
        "checkpoint_score": score,
        "checkpoint_score_definition": (
            "equal_block_mean_of_EDM_loss_plus_weighted_normalized_mean_state_MSE"
        ),
        "blocks": blocks,
        "validation_probes_per_window": config.validation_probes,
        "physics_metric_used": False,
        "future_truth_used_as_condition": False,
        "wall_seconds": float(time.perf_counter() - started),
    }


@torch.no_grad()
def evaluate_mean_state(
    *,
    mean_model: _MeanOperator,
    dataset: _WindowDataset,
    device: torch.device,
) -> dict[str, Any]:
    if dataset.split != "validation" or dataset.horizon != PGL_HORIZON:
        raise ValueError("persistent mean-state evaluation dataset differs")
    sums = {step: np.zeros(5, dtype=np.float64) for step in (1, 4)}
    counts = {step: 0 for step in (1, 4)}
    mean_model.eval()
    for index in range(len(dataset)):
        context, targets, _ = tensor_window(dataset[index], device)
        mean = mean_forecast_trajectory(mean_model, context)
        for step in (1, 4):
            error = (mean[:, step - 1].float() - targets[:, step - 1].float()).square()
            sums[step] += error.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            counts[step] += int(error.shape[0] * error.shape[2] * error.shape[3] * error.shape[4])
    return {
        "horizons": {
            str(step): {
                "per_field_mse": {
                    field: float(value)
                    for field, value in zip(PGL_FIELD_ORDER, sums[step] / counts[step])
                },
                "mean_field_mse": float(np.mean(sums[step] / counts[step])),
            }
            for step in (1, 4)
        },
        "window_count": len(dataset),
        "physics_derived_metric": False,
        "future_truth_used_as_context": False,
    }


def _learning_rate(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    peak: float,
    minimum: float,
) -> float:
    if not 1 <= int(update) <= int(total_updates):
        raise ValueError("persistent learning-rate update leaves budget")
    if update <= warmup_updates:
        return float(peak * update / max(1, warmup_updates))
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
    return float(minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress)))


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@torch.no_grad()
def toroidal_equivariance_gate(
    *,
    edm: PersistentGlobalLocalEDM,
    current: Tensor,
    mean: Tensor,
    clean_reference: Tensor,
    tolerance: float = 5.0e-5,
) -> dict[str, Any]:
    """Audit every integer shift on one full-resolution denoiser input."""

    if clean_reference.shape[-1] != 88:
        raise ValueError("production equivariance gate requires all 88 toroidal cells")
    sigma = torch.full((current.shape[0],), 0.7, device=current.device)
    _, noise, _ = keyed_sigma_and_noise(
        base_seed=PGL_NOISE_BASE_SEED + 2,
        epoch_zero_based=0,
        current_frame=496,
        probe=0,
        reference=clean_reference,
        noise_config=edm.noise_config,
    )
    edm.eval()
    reference = edm.denoise(noise, current, mean, sigma)
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1.0e-8)
    errors: list[float] = []
    for shift in range(88):
        observed = edm.denoise(
            torch.roll(noise, shift, -1),
            torch.roll(current, shift, -1),
            torch.roll(mean, shift, -1),
            sigma,
        )
        expected = torch.roll(reference, shift, -1)
        error = float(
            (torch.linalg.vector_norm((observed - expected).float()) / denominator).cpu()
        )
        errors.append(error)
    maximum = max(errors)
    return {
        "all_integer_shifts_checked": list(range(88)),
        "relative_L2_error_by_shift": errors,
        "maximum_relative_L2_error": maximum,
        "tolerance": float(tolerance),
        "passed": bool(maximum <= tolerance),
        "nonperiodic_axis_shifted": False,
    }


def train_persistent_global_local(
    *,
    mean_model: nn.Module,
    edm: PersistentGlobalLocalEDM,
    training_dataset: _WindowDataset,
    validation_dataset: _WindowDataset,
    derivative_rms: Tensor,
    output: Path,
    device: torch.device,
    paper0_commit: str,
    slurm_job_id: str,
    manifest: Mapping[str, Any],
    config: PersistentPilotTrainingConfig,
    on_epoch: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one bounded seed with state-only checkpoint selection."""

    destination = Path(output)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("persistent global-local training requires allocated CUDA")
    if (
        training_dataset.split != "train"
        or validation_dataset.split != "validation"
        or training_dataset.horizon != PGL_HORIZON
        or validation_dataset.horizon != PGL_HORIZON
        or len(training_dataset) != config.expected_training_windows
        or len(validation_dataset) != config.expected_validation_windows
    ):
        raise ValueError("persistent pilot dataset contract differs")
    if any("85606" in str(value).lower() for value in manifest.values() if isinstance(value, str)):
        raise ValueError("persistent pilot manifest string mentions held-out path")
    scales = edm.residual_scales.reshape(PGL_HORIZON, 5)
    if derivative_rms.shape != (5,) or not torch.all(
        torch.isfinite(derivative_rms) & (derivative_rms > 0.0)
    ):
        raise ValueError("persistent derivative RMS differs")

    destination.mkdir(parents=True)
    candidates_directory = destination / "candidates"
    candidates_directory.mkdir()
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    mean_model = mean_model.to(device, torch.float32)
    edm = edm.to(device, torch.float32)
    raw_mean = mean_model
    raw_edm = edm
    ema_mean = copy.deepcopy(raw_mean).to(device, torch.float32).eval()
    ema_edm = copy.deepcopy(raw_edm).to(device, torch.float32).eval()
    ema_mean.requires_grad_(False)
    ema_edm.requires_grad_(False)
    raw_mean.train()
    raw_edm.train()
    optimizer = AdamW(
        [
            {
                "params": list(raw_edm.parameters()),
                "lr": config.stochastic_peak_learning_rate,
                "name": "stochastic",
            },
            {
                "params": list(raw_mean.parameters()),
                "lr": config.mean_peak_learning_rate,
                "name": "mean",
            },
        ],
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    initial_states = {
        "mean": module_state_sha256(raw_mean),
        "stochastic": module_state_sha256(raw_edm),
    }
    parent_state = evaluate_mean_state(
        mean_model=ema_mean, dataset=validation_dataset, device=device
    )
    run_config = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_persistent_global_local_training",
        "mode": config.mode,
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "training": config.to_record(),
        "stochastic_model": raw_edm.to_record(),
        "parameter_count": {
            "mean": parameter_count(raw_mean),
            "stochastic": parameter_count(raw_edm),
            "total": parameter_count(raw_mean) + parameter_count(raw_edm),
        },
        "derivative_rms": derivative_rms.detach().cpu().tolist(),
        "residual_scales": scales.detach().cpu().tolist(),
        "manifest": dict(manifest),
        "physics_derived_loss_used": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    write_strict_json_atomic(destination / "config.json", run_config)

    total_updates = config.total_optimizer_steps
    warmup_updates = max(1, round(config.warmup_fraction * total_updates))
    global_update = 0
    candidates: list[dict[str, Any]] = []
    history_path = destination / "history.jsonl"
    history: list[dict[str, Any]] = []
    for epoch_zero in range(config.epochs):
        completed_epoch = epoch_zero + 1
        training_dataset.set_epoch(epoch_zero)
        order = np.random.default_rng(
            np.random.SeedSequence([config.seed, epoch_zero, 0x50474C])
        ).permutation(len(training_dataset))
        raw_mean.train()
        raw_edm.train()
        epoch_started = time.perf_counter()
        objectives: list[float] = []
        mean_losses: list[float] = []
        edm_losses: list[float] = []
        gradients: list[float] = []
        for group_start in range(0, len(order), config.accumulation_windows):
            group = order[group_start : group_start + config.accumulation_windows]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                context, targets, current_frame = tensor_window(
                    training_dataset[int(index)], device
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mean = mean_forecast_trajectory(raw_mean, context)
                    mean_loss, _ = weighted_mean_state_loss(
                        mean, targets, derivative_rms
                    )
                    clean = raw_edm.normalize_residual(targets - mean.detach())
                    sigma, noise, _ = keyed_sigma_and_noise(
                        base_seed=PGL_NOISE_BASE_SEED,
                        epoch_zero_based=epoch_zero,
                        current_frame=current_frame,
                        probe=0,
                        reference=clean,
                        noise_config=raw_edm.noise_config,
                    )
                    edm_loss = raw_edm.training_loss(
                        clean,
                        context[:, -1],
                        mean.detach(),
                        sigma=sigma,
                        noise=noise,
                    )
                    objective = mean_loss + edm_loss.loss
                if not torch.isfinite(objective):
                    raise FloatingPointError("persistent pilot objective is non-finite")
                (objective / len(group)).backward()
                objectives.append(float(objective.detach().cpu()))
                mean_losses.append(float(mean_loss.detach().cpu()))
                edm_losses.append(float(edm_loss.loss.detach().cpu()))
            norm = torch.nn.utils.clip_grad_norm_(
                list(raw_mean.parameters()) + list(raw_edm.parameters()),
                config.gradient_clip,
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("persistent pilot gradient norm is non-finite")
            global_update += 1
            stochastic_rate = _learning_rate(
                global_update,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                peak=config.stochastic_peak_learning_rate,
                minimum=config.stochastic_minimum_learning_rate,
            )
            mean_rate = _learning_rate(
                global_update,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                peak=config.mean_peak_learning_rate,
                minimum=config.mean_minimum_learning_rate,
            )
            for group_record in optimizer.param_groups:
                group_record["lr"] = (
                    stochastic_rate
                    if group_record["name"] == "stochastic"
                    else mean_rate
                )
            optimizer.step()
            update_ema_model(ema_mean, raw_mean, decay=config.ema_decay)
            update_ema_model(ema_edm, raw_edm, decay=config.ema_decay)
            gradients.append(float(norm.detach().cpu()))

        validation: dict[str, Any] | None = None
        candidate: dict[str, Any] | None = None
        if completed_epoch in config.validation_epochs:
            validation = validation_objective(
                mean_model=ema_mean,
                edm=ema_edm,
                dataset=validation_dataset,
                derivative_rms=derivative_rms,
                config=config,
                device=device,
            )
            checkpoint = candidates_directory / f"ema_epoch_{completed_epoch:03d}.pt"
            payload = {
                "schema_version": 1,
                "kind": "persistent_global_local_EMA_candidate",
                "paper0_commit": str(paper0_commit),
                "slurm_job_id": str(slurm_job_id),
                "completed_epoch": completed_epoch,
                "optimizer_updates": global_update,
                "training": config.to_record(),
                "stochastic_model": ema_edm.to_record(),
                "mean_model_state": {
                    name: value.detach().cpu() for name, value in ema_mean.state_dict().items()
                },
                "stochastic_model_state": {
                    name: value.detach().cpu() for name, value in ema_edm.state_dict().items()
                },
                "validation": validation,
                "physics_metric_used_for_selection": False,
                "held_out_85606_read": False,
                "new_nersc_data_read": False,
            }
            save_torch_atomic(checkpoint, payload)
            candidate = {
                "completed_epoch": completed_epoch,
                "optimizer_updates": global_update,
                "checkpoint_score": validation["checkpoint_score"],
                "path": str(checkpoint),
                "sha256": sha256_path(checkpoint),
            }
            candidates.append(candidate)
        record = {
            "completed_epoch": completed_epoch,
            "optimizer_updates": global_update,
            "training_window_count": len(objectives),
            "train_mean_objective": float(np.mean(objectives)),
            "train_mean_state_loss": float(np.mean(mean_losses)),
            "train_mean_edm_loss": float(np.mean(edm_losses)),
            "mean_preclip_gradient_norm": float(np.mean(gradients)),
            "maximum_preclip_gradient_norm": float(max(gradients)),
            "stochastic_learning_rate": optimizer.param_groups[0]["lr"],
            "mean_learning_rate": optimizer.param_groups[1]["lr"],
            "validation_candidate": validation is not None,
            "validation": validation,
            "candidate": candidate,
            "epoch_wall_seconds": float(time.perf_counter() - epoch_started),
        }
        _append_jsonl(history_path, record)
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)

    if global_update != total_updates:
        raise RuntimeError("persistent pilot optimizer update count differs")
    if len(candidates) != len(config.validation_epochs):
        raise RuntimeError("persistent pilot checkpoint count differs")
    selected = min(
        candidates,
        key=lambda value: (float(value["checkpoint_score"]), int(value["completed_epoch"])),
    )
    selected_payload = torch.load(
        selected["path"], map_location=device, weights_only=True
    )
    selected_mean = copy.deepcopy(raw_mean).to(device, torch.float32)
    selected_edm = copy.deepcopy(raw_edm).to(device, torch.float32)
    selected_mean.load_state_dict(selected_payload["mean_model_state"], strict=True)
    selected_edm.load_state_dict(selected_payload["stochastic_model_state"], strict=True)
    selected_mean.eval()
    selected_edm.eval()
    reload_exact = all(
        torch.equal(value.to(device), selected_mean.state_dict()[name])
        for name, value in selected_payload["mean_model_state"].items()
    ) and all(
        torch.equal(value.to(device), selected_edm.state_dict()[name])
        for name, value in selected_payload["stochastic_model_state"].items()
    )
    selected_state = evaluate_mean_state(
        mean_model=selected_mean, dataset=validation_dataset, device=device
    )
    state_ratios = {
        horizon: selected_state["horizons"][horizon]["mean_field_mse"]
        / parent_state["horizons"][horizon]["mean_field_mse"]
        for horizon in ("1", "4")
    }
    state_gate = {
        "candidate_over_parent_mean_field_MSE": state_ratios,
        "threshold_maximum": 1.05,
        "passed": bool(all(value <= 1.05 for value in state_ratios.values())),
    }
    first_context, first_targets, _ = tensor_window(validation_dataset[0], device)
    first_mean = mean_forecast_trajectory(selected_mean, first_context)
    first_clean = selected_edm.normalize_residual(first_targets - first_mean)
    equivariance = toroidal_equivariance_gate(
        edm=selected_edm,
        current=first_context[:, -1],
        mean=first_mean,
        clean_reference=first_clean,
    )
    mechanical = {
        "exact_optimizer_update_count": global_update == total_updates,
        "checkpoint_reload_bitwise_exact": reload_exact,
        "finite_training_history": all(
            math.isfinite(float(record["train_mean_objective"])) for record in history
        ),
        "toroidal_equivariance_passed": equivariance["passed"],
    }
    mechanical["passed"] = bool(all(mechanical.values()))
    selected_checkpoint = destination / "selected.pt"
    final_payload = {
        **selected_payload,
        "kind": "persistent_global_local_selected_EMA_checkpoint",
        "source_candidate": {
            "path": selected["path"],
            "sha256": selected["sha256"],
        },
        "parent_mean_state": parent_state,
        "selected_mean_state": selected_state,
        "state_gate": state_gate,
        "equivariance_gate": equivariance,
    }
    save_torch_atomic(selected_checkpoint, final_payload)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    engineering_passed = bool(mechanical["passed"] and state_gate["passed"])
    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_persistent_global_local_training",
        "status": "completed",
        "mode": config.mode,
        "development_run": "85604",
        "seed": config.seed,
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "training": config.to_record(),
        "stochastic_model": selected_edm.to_record(),
        "parameter_count": run_config["parameter_count"],
        "initial_model_state_sha256": initial_states,
        "selected_model_state_sha256": {
            "mean": module_state_sha256(selected_mean),
            "stochastic": module_state_sha256(selected_edm),
        },
        "completed_epochs": len(history),
        "completed_optimizer_steps": global_update,
        "history": history,
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "sha256": sha256_path(selected_checkpoint),
            "completed_epoch": int(selected["completed_epoch"]),
            "checkpoint_score": float(selected["checkpoint_score"]),
        },
        "parent_mean_state": parent_state,
        "selected_mean_state": selected_state,
        "state_gate": state_gate,
        "equivariance_gate": equivariance,
        "mechanical_gate": mechanical,
        "engineering_or_state_pilot_passed": engineering_passed,
        "physics_evaluation_authorized": bool(
            config.mode == "pilot" and engineering_passed
        ),
        "confirmation_seed_training_authorized": False,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_bytes": peak_bytes,
        "peak_cuda_memory_GiB": float(peak_bytes / 2**30),
        "gpu": torch.cuda.get_device_name(device),
        "artifacts": {
            "config": {
                "path": str(destination / "config.json"),
                "sha256": sha256_path(destination / "config.json"),
            },
            "history": {"path": str(history_path), "sha256": sha256_path(history_path)},
            "selected_checkpoint": {
                "path": str(selected_checkpoint),
                "sha256": sha256_path(selected_checkpoint),
            },
            "candidate_checkpoints": candidates,
        },
        "training_performed": True,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "physics_diagnostics_scored": False,
        "future_truth_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
    }
    write_strict_json_atomic(destination / "result.json", result)
    return result
