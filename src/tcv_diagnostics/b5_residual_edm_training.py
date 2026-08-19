"""Bounded 85604-only mechanics for the frozen B5 field-residual EDM smoke."""

from __future__ import annotations

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

from .codec_training import save_torch_atomic, sha256_path
from .model_data import assert_development_path, write_strict_json_atomic
from .model_training_data import FAMILY_FIELDS, VOLUME_SHAPE
from .models.field_residual_edm import (
    B5_FIELD_ORDER,
    B5_RESIDUAL_SCALES,
    B5_SPATIAL_SHAPE,
    FieldResidualUNet3D,
    FieldResidualUNetConfig,
    JointFieldResidualEDM,
)


B5_EDM_SMOKE_TARGETS = tuple(range(2, 10))
B5_EDM_FIXED_PROBE_TARGETS = tuple(range(2, 6))
B5_EDM_TRAINING_ORDER_SEED = 67_001
B5_EDM_TRAINING_NOISE_SEED = 67_002
B5_EDM_SAMPLER_SEED = 67_003
B5_EDM_FIXED_PROBE_SEED = 67_004


class _WindowDataset(Protocol):
    split: str
    context_frames: int
    target_frames: tuple[int, ...]
    augment: bool
    fields: Sequence[str]

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


class _ForecastArtifact(Protocol):
    target_frames: tuple[int, ...]
    sha256: str

    def read(self, start: int, stop: int) -> np.ndarray: ...


@dataclass(frozen=True)
class B5EDMSmokeConfig:
    """Exact non-scientific optimization and sampler budget."""

    seed: int = 1701
    optimizer_steps: int = 64
    microbatch_targets: int = 1
    gradient_accumulation_targets: int = 1
    learning_rate: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    training_precision: str = "bfloat16_autocast_with_FP32_loss"
    training_order_seed: int = B5_EDM_TRAINING_ORDER_SEED
    training_noise_seed: int = B5_EDM_TRAINING_NOISE_SEED
    sampler_seed: int = B5_EDM_SAMPLER_SEED
    fixed_probe_seed: int = B5_EDM_FIXED_PROBE_SEED
    sampler_steps: int = 18
    sampler_sigma_max: float = 80.0
    sampler_sigma_min: float = 0.002
    sampler_rho: float = 7.0
    sampler_members: int = 2
    peak_cuda_GiB_limit: float = 75.0
    toroidal_equivariance_shift: int = 8
    toroidal_equivariance_rtol: float = 2.0e-5
    toroidal_equivariance_atol: float = 2.0e-5

    def __post_init__(self) -> None:
        if self.seed != 1701 or self.optimizer_steps != 64:
            raise ValueError("B5 smoke identity is fixed to seed 1701 and 64 steps")
        if self.microbatch_targets != 1 or self.gradient_accumulation_targets != 1:
            raise ValueError("B5 smoke uses one target per optimizer step")
        if self.training_precision != "bfloat16_autocast_with_FP32_loss":
            raise ValueError("B5 smoke precision differs")
        if self.sampler_members != 2 or self.sampler_steps != 18:
            raise ValueError("B5 smoke sampler budget differs")
        if self.toroidal_equivariance_shift != 8:
            raise ValueError("B5 equivariance shift must match total U-Net stride")

    @property
    def target_sequence(self) -> tuple[int, ...]:
        return smoke_target_sequence(
            seed=self.training_order_seed,
            steps=self.optimizer_steps,
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["betas"] = list(self.betas)
        record.update(
            {
                "mode": "bounded_mechanical_smoke",
                "training_targets": [2, 10],
                "training_target_count": 8,
                "fixed_probe_targets": [2, 6],
                "target_sequence": list(self.target_sequence),
                "scientific_result": False,
                "validation_read_allowed": False,
                "full_training_authorized": False,
            }
        )
        return record


def smoke_target_sequence(
    *,
    seed: int = B5_EDM_TRAINING_ORDER_SEED,
    steps: int = 64,
) -> tuple[int, ...]:
    """Repeat one prospectively seeded permutation of the eight targets."""

    if int(seed) != B5_EDM_TRAINING_ORDER_SEED or int(steps) != 64:
        raise ValueError("B5 smoke target-order identity differs")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    cycle = tuple(int(value) for value in generator.permutation(B5_EDM_SMOKE_TARGETS))
    if set(cycle) != set(B5_EDM_SMOKE_TARGETS):
        raise AssertionError("B5 target permutation is incomplete")
    return tuple(cycle[index % len(cycle)] for index in range(int(steps)))


def keyed_sigma_and_noise(
    *,
    seed: int,
    ordinal: int,
    target_frame: int,
    spatial_shape: Sequence[int] = B5_SPATIAL_SHAPE,
) -> tuple[np.float32, np.ndarray]:
    """Generate one immutable EDM sigma/noise pair without a stored bank."""

    shape = tuple(int(value) for value in spatial_shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("B5 keyed noise spatial shape differs")
    if int(ordinal) < 0 or int(target_frame) not in B5_EDM_SMOKE_TARGETS:
        raise ValueError("B5 keyed noise identity differs")
    sequence = np.random.SeedSequence(
        [int(seed), int(ordinal), int(target_frame), 0xB5ED_0001]
    )
    generator = np.random.Generator(np.random.PCG64(sequence))
    log_sigma = -1.2 + 1.2 * float(generator.standard_normal())
    sigma = np.float32(math.exp(log_sigma))
    noise = generator.standard_normal((5, *shape), dtype=np.float32)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.all(np.isfinite(noise)):
        raise FloatingPointError("B5 keyed noise is non-finite")
    return sigma, np.ascontiguousarray(noise, dtype=np.float32)


def sampler_initial_noise(
    *,
    seed: int = B5_EDM_SAMPLER_SEED,
    target_frame: int = 2,
    members: int = 2,
    spatial_shape: Sequence[int] = B5_SPATIAL_SHAPE,
) -> np.ndarray:
    """Return independent member noise from member-keyed PCG64 streams."""

    if int(seed) != B5_EDM_SAMPLER_SEED or int(target_frame) != 2:
        raise ValueError("B5 sampler-noise identity differs")
    if int(members) != 2:
        raise ValueError("B5 smoke sampler requires two members")
    shape = tuple(int(value) for value in spatial_shape)
    values = []
    for member in range(int(members)):
        generator = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    [int(seed), int(target_frame), member, 0xB5ED_0002]
                )
            )
        )
        values.append(generator.standard_normal((5, *shape), dtype=np.float32))
    result = np.ascontiguousarray(np.stack(values, axis=0), dtype=np.float32)
    if not np.all(np.isfinite(result)) or np.array_equal(result[0], result[1]):
        raise FloatingPointError("B5 sampler initial members are invalid")
    return result


class B5ResidualSmokeDataset:
    """Join exact training windows to the immutable H1 mean artifact."""

    def __init__(
        self,
        windows: _WindowDataset,
        forecast: _ForecastArtifact,
    ) -> None:
        if (
            windows.split != "train"
            or windows.context_frames != 1
            or tuple(windows.target_frames) != B5_EDM_SMOKE_TARGETS
            or windows.augment
            or tuple(windows.fields) != B5_FIELD_ORDER
        ):
            raise ValueError("B5 smoke window dataset differs from frozen contract")
        if tuple(forecast.target_frames) != tuple(range(2, 432)):
            raise ValueError("B5 H1 forecast artifact target index differs")
        self.windows = windows
        self.forecast = forecast
        self.target_frames = B5_EDM_SMOKE_TARGETS
        self.scales = np.asarray(B5_RESIDUAL_SCALES, dtype=np.float32).reshape(
            5, 1, 1, 1
        )

    def __len__(self) -> int:
        return len(self.target_frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError(position)
        item = self.windows[position]
        target_frame = int(item["target_frame_index"])
        if target_frame != self.target_frames[position]:
            raise RuntimeError("B5 smoke target order differs")
        context_indices = tuple(int(value) for value in item["context_frame_indices"])
        if context_indices != (target_frame - 1,):
            raise RuntimeError("B5 smoke context index differs")
        context = np.asarray(item["context"], dtype=np.float32)
        truth = np.asarray(item["target"], dtype=np.float32)
        forecast_index = target_frame - 2
        mean = self.forecast.read(forecast_index, forecast_index + 1)[0]
        if (
            context.shape != (1, 5, *VOLUME_SHAPE)
            or truth.shape != (5, *VOLUME_SHAPE)
            or mean.shape != (5, *VOLUME_SHAPE)
        ):
            raise ValueError("B5 smoke context/truth/mean shape differs")
        residual = truth - mean
        normalized = residual / self.scales
        condition = np.concatenate((context[0], mean), axis=0)
        if not (
            np.all(np.isfinite(condition)) and np.all(np.isfinite(normalized))
        ):
            raise FloatingPointError("B5 smoke example is non-finite")
        return {
            "target_frame_index": np.int64(target_frame),
            "context_frame_index": np.int64(target_frame - 1),
            "condition": np.ascontiguousarray(condition, dtype=np.float32),
            "normalized_residual": np.ascontiguousarray(
                normalized, dtype=np.float32
            ),
            "deterministic_mean": np.ascontiguousarray(mean, dtype=np.float32),
            "target_truth_used_as_condition": False,
            "absolute_time_used_as_condition": False,
        }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def module_state_sha256(module: nn.Module) -> str:
    """Hash state names, dtypes, shapes, and canonical CPU tensor bytes."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _example_tensors(
    dataset: B5ResidualSmokeDataset,
    target_frame: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    index = int(target_frame) - B5_EDM_SMOKE_TARGETS[0]
    item = dataset[index]
    if int(item["target_frame_index"]) != int(target_frame):
        raise RuntimeError("B5 tensor example target differs")
    condition = torch.from_numpy(item["condition"])[None].to(device, torch.float32)
    clean = torch.from_numpy(item["normalized_residual"])[None].to(
        device, torch.float32
    )
    mean = torch.from_numpy(item["deterministic_mean"])[None].to(
        device, torch.float32
    )
    return condition, clean, mean


def fixed_probe_loss(
    *,
    model: JointFieldResidualEDM,
    dataset: B5ResidualSmokeDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate four immutable FP32 denoising examples."""

    was_training = model.training
    model.eval()
    values: list[float] = []
    by_target: dict[str, float] = {}
    with torch.inference_mode():
        for ordinal, target in enumerate(B5_EDM_FIXED_PROBE_TARGETS):
            condition, clean, _ = _example_tensors(dataset, target, device=device)
            sigma, noise = keyed_sigma_and_noise(
                seed=B5_EDM_FIXED_PROBE_SEED,
                ordinal=ordinal,
                target_frame=target,
            )
            result = model.training_loss(
                clean,
                condition,
                sigma=torch.tensor([float(sigma)], device=device),
                noise=torch.from_numpy(noise)[None].to(device),
            )
            value = float(result.loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError("B5 fixed-probe loss is non-finite")
            values.append(value)
            by_target[str(target)] = value
    model.train(was_training)
    return {
        "seed": B5_EDM_FIXED_PROBE_SEED,
        "target_frames": [2, 6],
        "mean_EDM_loss": float(np.mean(values, dtype=np.float64)),
        "EDM_loss_by_target": by_target,
    }


def _fixed_denoiser_output(
    *,
    model: JointFieldResidualEDM,
    dataset: B5ResidualSmokeDataset,
    device: torch.device,
) -> Tensor:
    condition, clean, _ = _example_tensors(dataset, 2, device=device)
    sigma, noise = keyed_sigma_and_noise(
        seed=B5_EDM_FIXED_PROBE_SEED,
        ordinal=0,
        target_frame=2,
    )
    sigma_tensor = torch.tensor([float(sigma)], device=device)
    noise_tensor = torch.from_numpy(noise)[None].to(device)
    noisy = clean + sigma_tensor.reshape(1, 1, 1, 1, 1) * noise_tensor
    model.eval()
    with torch.inference_mode():
        return model.denoise(noisy, condition, sigma_tensor).to("cpu", torch.float32)


def toroidal_equivariance_probe(
    *,
    model: JointFieldResidualEDM,
    dataset: B5ResidualSmokeDataset,
    config: B5EDMSmokeConfig,
    device: torch.device,
) -> dict[str, Any]:
    condition, clean, _ = _example_tensors(dataset, 2, device=device)
    sigma, noise = keyed_sigma_and_noise(
        seed=B5_EDM_FIXED_PROBE_SEED,
        ordinal=0,
        target_frame=2,
    )
    sigma_tensor = torch.tensor([float(sigma)], device=device)
    noisy = clean + sigma_tensor.reshape(1, 1, 1, 1, 1) * torch.from_numpy(
        noise
    )[None].to(device)
    model.eval()
    with torch.inference_mode():
        reference = model.denoise(noisy, condition, sigma_tensor)
        shifted = model.denoise(
            torch.roll(noisy, config.toroidal_equivariance_shift, dims=-1),
            torch.roll(condition, config.toroidal_equivariance_shift, dims=-1),
            sigma_tensor,
        )
    expected = torch.roll(reference, config.toroidal_equivariance_shift, dims=-1)
    difference = (shifted - expected).abs()
    passed = bool(
        torch.allclose(
            shifted,
            expected,
            rtol=config.toroidal_equivariance_rtol,
            atol=config.toroidal_equivariance_atol,
        )
    )
    return {
        "shift_cells": config.toroidal_equivariance_shift,
        "rtol": config.toroidal_equivariance_rtol,
        "atol": config.toroidal_equivariance_atol,
        "maximum_absolute_difference": float(difference.max().cpu()),
        "passed": passed,
    }


def _write_npz_atomic(path: Path, **values: np.ndarray) -> Path:
    destination = Path(path)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def train_b5_edm_smoke(
    *,
    dataset: B5ResidualSmokeDataset,
    output: Path,
    device: torch.device,
    paper0_commit: str,
    slurm_job_id: str,
    authority: Mapping[str, Any],
    config: B5EDMSmokeConfig = B5EDMSmokeConfig(),
    model_config: FieldResidualUNetConfig = FieldResidualUNetConfig(),
    on_step: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute the exact 64-step smoke and archive every mechanical gate."""

    destination = Path(output)
    assert_development_path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B5 EDM smoke requires an allocated CUDA device")
    if len(dataset) != 8 or dataset.target_frames != B5_EDM_SMOKE_TARGETS:
        raise ValueError("B5 EDM smoke dataset budget differs")
    if tuple(FAMILY_FIELDS["c5p"]) != B5_FIELD_ORDER or VOLUME_SHAPE != B5_SPATIAL_SHAPE:
        raise RuntimeError("B5 field or spatial contract differs")
    if any("85606" in str(value).lower() for value in authority.values()):
        raise ValueError("B5 smoke authority mentions held-out 85606")
    destination.mkdir(parents=True)
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats(device)

    backbone = FieldResidualUNet3D(model_config).to(device, torch.float32)
    model = JointFieldResidualEDM(backbone).to(device, torch.float32)
    model.train()
    count = parameter_count(model)
    initial_state_sha = module_state_sha256(model)
    config_path = destination / "config.json"
    write_strict_json_atomic(
        config_path,
        {
            "schema_version": 1,
            "scope": "bounded_non_scientific_B5_joint_field_residual_EDM_smoke",
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

    initial_probe = fixed_probe_loss(model=model, dataset=dataset, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    history_path = destination / "history.jsonl"
    records: list[dict[str, Any]] = []
    for zero_step, target_frame in enumerate(config.target_sequence):
        global_step = zero_step + 1
        condition, clean, _ = _example_tensors(
            dataset, target_frame, device=device
        )
        sigma, noise = keyed_sigma_and_noise(
            seed=config.training_noise_seed,
            ordinal=zero_step,
            target_frame=target_frame,
        )
        sigma_tensor = torch.tensor([float(sigma)], device=device)
        noise_tensor = torch.from_numpy(noise)[None].to(device)
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = model.training_loss(
                clean,
                condition,
                sigma=sigma_tensor,
                noise=noise_tensor,
            )
        loss = losses.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite B5 loss at step {global_step}")
        loss.backward()
        preclip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        if not torch.isfinite(preclip):
            raise FloatingPointError(
                f"non-finite B5 gradient norm at step {global_step}"
            )
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(
                f"non-finite B5 parameter at step {global_step}"
            )
        torch.cuda.synchronize(device)
        record = {
            "global_step": global_step,
            "target_frame": int(target_frame),
            "context_frame": int(target_frame) - 1,
            "sigma": float(sigma),
            "EDM_loss": float(loss.detach().cpu()),
            "unweighted_MSE": float(losses.unweighted_mse.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "preclip_gradient_norm": float(preclip.detach().cpu()),
            "step_wall_seconds": float(time.perf_counter() - step_started),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in record.items()
            if key not in {"global_step", "target_frame", "context_frame"}
        ):
            raise FloatingPointError("B5 history record is non-finite")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        records.append(record)
        if on_step is not None:
            on_step(record)

    final_probe = fixed_probe_loss(model=model, dataset=dataset, device=device)
    probe_decreased = bool(
        final_probe["mean_EDM_loss"] < initial_probe["mean_EDM_loss"]
    )
    final_state_sha = module_state_sha256(model)
    expected_reload = _fixed_denoiser_output(
        model=model, dataset=dataset, device=device
    )
    checkpoint_path = destination / "smoke_checkpoint.pt"
    checkpoint_payload = {
        "schema_version": 1,
        "kind": "B5_joint_field_residual_EDM_bounded_smoke_non_scientific",
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "run_config": config.to_record(),
        "model_config": model_config.to_record(),
        "residual_scales": list(B5_RESIDUAL_SCALES),
        "model_state": {
            name: value.detach().to("cpu")
            for name, value in model.state_dict().items()
        },
        "completed_optimizer_steps": len(records),
        "initial_fixed_probe": initial_probe,
        "final_fixed_probe": final_probe,
        "scientific_result": False,
        "validation_frames_read": False,
        "held_out_85606_read": False,
    }
    save_torch_atomic(checkpoint_path, checkpoint_payload)
    restored_backbone = FieldResidualUNet3D(model_config).to(device, torch.float32)
    restored = JointFieldResidualEDM(restored_backbone).to(device, torch.float32)
    restored.load_state_dict(checkpoint_payload["model_state"], strict=True)
    restored.eval()
    observed_reload = _fixed_denoiser_output(
        model=restored, dataset=dataset, device=device
    )
    reload_exact = bool(torch.equal(expected_reload, observed_reload))

    equivariance = toroidal_equivariance_probe(
        model=restored,
        dataset=dataset,
        config=config,
        device=device,
    )
    condition, _, deterministic_mean = _example_tensors(
        dataset, 2, device=device
    )
    initial_noise = sampler_initial_noise(
        seed=config.sampler_seed,
        target_frame=2,
        members=config.sampler_members,
    )
    initial_noise_tensor = torch.from_numpy(initial_noise)[None].to(
        device, torch.float32
    )
    restored.eval()
    sample_started = time.perf_counter()
    normalized_samples = restored.sample_normalized(
        condition,
        initial_noise_tensor,
        steps=config.sampler_steps,
        sigma_max=config.sampler_sigma_max,
        sigma_min=config.sampler_sigma_min,
        rho=config.sampler_rho,
    )
    composed = restored.compose_fields(deterministic_mean, normalized_samples)
    torch.cuda.synchronize(device)
    sampler_seconds = time.perf_counter() - sample_started
    normalized_cpu = normalized_samples.to("cpu", torch.float32)
    composed_cpu = composed.to("cpu", torch.float32)
    residual_member_difference = float(
        (normalized_cpu[:, 0] - normalized_cpu[:, 1]).square().mean().sqrt()
    )
    field_member_difference = float(
        (composed_cpu[:, 0] - composed_cpu[:, 1]).square().mean().sqrt()
    )
    sample_finite = bool(
        torch.isfinite(normalized_cpu).all() and torch.isfinite(composed_cpu).all()
    )
    sample_path = _write_npz_atomic(
        destination / "sampler_probe.npz",
        initial_noise=initial_noise,
        normalized_residual=normalized_cpu.numpy(),
        standardized_fields=composed_cpu.numpy(),
    )
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_limit_bytes = int(config.peak_cuda_GiB_limit * 1024**3)
    sampler_probe = {
        "target_frame": 2,
        "condition_context_frame": 1,
        "members": config.sampler_members,
        "steps": config.sampler_steps,
        "network_evaluations_per_member": 2 * config.sampler_steps - 1,
        "sigma_max": config.sampler_sigma_max,
        "sigma_min": config.sampler_sigma_min,
        "rho": config.sampler_rho,
        "stochastic_churn": 0.0,
        "normalized_residual_shape": list(normalized_cpu.shape),
        "canonical_field_shape": list(composed_cpu.shape),
        "normalized_residual_member_RMS_difference": residual_member_difference,
        "standardized_field_member_RMS_difference": field_member_difference,
        "finite": sample_finite,
        "nonzero_member_diversity": residual_member_difference > 0.0
        and field_member_difference > 0.0,
        "wall_seconds": float(sampler_seconds),
        "fresh_noise_per_future_rollout_step_required": True,
        "scientific_calibration_result": False,
    }
    gates = {
        "optimizer_steps_exact": len(records) == config.optimizer_steps,
        "all_history_values_finite": all(
            math.isfinite(record["EDM_loss"])
            and math.isfinite(record["preclip_gradient_norm"])
            for record in records
        ),
        "fixed_probe_loss_decreased": probe_decreased,
        "checkpoint_reload_bitwise_exact": reload_exact,
        "toroidal_equivariance_passed": bool(equivariance["passed"]),
        "sampler_finite": sample_finite,
        "sampler_member_diversity_nonzero": bool(
            sampler_probe["nonzero_member_diversity"]
        ),
        "sampler_canonical_axes_exact": sampler_probe["canonical_field_shape"]
        == [1, 2, 1, 5, 64, 32, 88],
        "peak_cuda_memory_below_75_GiB": peak_bytes < peak_limit_bytes,
        "validation_frames_read": False,
        "held_out_85606_read": False,
    }
    all_passed = all(
        bool(value)
        for key, value in gates.items()
        if key not in {"validation_frames_read", "held_out_85606_read"}
    ) and gates["validation_frames_read"] is False and gates[
        "held_out_85606_read"
    ] is False
    wall_seconds = time.perf_counter() - started
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "bounded_non_scientific_B5_joint_field_residual_EDM_smoke_85604",
        "status": "passed" if all_passed else "failed",
        "scientific_result": False,
        "paper0_commit": str(paper0_commit),
        "slurm_job_id": str(slurm_job_id),
        "development_run": "85604",
        "sequestered_run": "85606",
        "config": config.to_record(),
        "model_config": model_config.to_record(),
        "parameter_count": count,
        "initial_model_state_sha256": initial_state_sha,
        "final_model_state_sha256": final_state_sha,
        "completed_optimizer_steps": len(records),
        "initial_fixed_probe": initial_probe,
        "final_fixed_probe": final_probe,
        "fixed_probe_relative_change": float(
            final_probe["mean_EDM_loss"] / initial_probe["mean_EDM_loss"] - 1.0
        ),
        "checkpoint_reload_bitwise_exact": reload_exact,
        "toroidal_equivariance": equivariance,
        "sampler_probe": sampler_probe,
        "peak_cuda_bytes": peak_bytes,
        "peak_cuda_GiB": float(peak_bytes / 1024**3),
        "peak_cuda_GiB_limit": config.peak_cuda_GiB_limit,
        "wall_seconds": float(wall_seconds),
        "gates": gates,
        "all_mechanical_gates_passed": all_passed,
        "artifacts": {
            "config": {
                "path": str(config_path),
                "sha256": sha256_path(config_path),
            },
            "history": {
                "path": str(history_path),
                "sha256": sha256_path(history_path),
                "records": len(records),
            },
            "smoke_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_path(checkpoint_path),
            },
            "sampler_probe": {
                "path": str(sample_path),
                "sha256": sha256_path(sample_path),
            },
        },
        "training_performed": True,
        "physics_derived_loss_used": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "full_B5_training_authorized": False,
        "scientific_B5_evaluation_authorized": False,
        "O3_launch_allowed": False,
        "assimilation_allowed": False,
        "diagnostic_ranking_allowed": False,
    }
    write_strict_json_atomic(destination / "result.json", result)
    return result
