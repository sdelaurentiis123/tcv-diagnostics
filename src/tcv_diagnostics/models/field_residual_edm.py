"""Joint full-field residual EDM for the prospective Paper 0 B5 smoke.

This module contains only data-space denoising mechanics.  It never reads a
dataset, validation split, physics diagnostic, or held-out simulation.  The
architecture and equations are frozen in
``paper0/protocol/PHASE3_B5_FIELD_RESIDUAL_EDM_SMOKE_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import make_conv
from .modulated_vit import NoiseTimeEmbedding


B5_FIELD_ORDER = ("Ne", "Pe", "Pi", "phi", "Vi")
B5_RESIDUAL_SCALES = (
    0.05503048051260375,
    0.04825854004472835,
    0.06096460194410047,
    0.04632595196855943,
    0.10251610501339582,
)
B5_SPATIAL_SHAPE = (64, 32, 88)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("every positive channel count has group count one")


def _mixed_conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int = 1,
    padding: int | None = None,
    bias: bool = True,
) -> nn.Module:
    amount = kernel_size // 2 if padding is None else int(padding)
    return make_conv(
        in_channels,
        out_channels,
        spatial=3,
        kernel_size=kernel_size,
        stride=stride,
        padding=amount,
        padding_mode=("zeros", "zeros", "circular"),
        bias=bias,
    )


@dataclass(frozen=True)
class FieldResidualUNetConfig:
    """Exact B5 smoke architecture, with smaller values allowed in tests."""

    residual_channels: int = 5
    condition_channels: int = 10
    position_channels: int = 2
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    residual_blocks_per_resolution: int = 2
    noise_embedding_features: int = 256
    kernel_size: int = 3
    group_norm_maximum_groups: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        counts = (
            self.residual_channels,
            self.condition_channels,
            self.position_channels,
            self.base_channels,
            self.residual_blocks_per_resolution,
            self.noise_embedding_features,
            self.kernel_size,
            self.group_norm_maximum_groups,
        )
        if min(counts) <= 0:
            raise ValueError("B5 U-Net counts must be positive")
        if self.residual_channels != 5 or self.condition_channels != 10:
            raise ValueError("B5 uses five residual and ten dynamic condition channels")
        if self.position_channels != 2:
            raise ValueError("B5 uses only static x/y position channels")
        if not self.channel_multipliers or any(
            int(value) <= 0 for value in self.channel_multipliers
        ):
            raise ValueError("B5 channel multipliers must be positive")
        if self.noise_embedding_features % 2:
            raise ValueError("B5 noise embedding features must be even")
        if self.kernel_size != 3:
            raise ValueError("B5 uses kernel size three")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("B5 dropout must lie in [0,1)")

    @property
    def level_channels(self) -> tuple[int, ...]:
        return tuple(
            self.base_channels * int(multiplier)
            for multiplier in self.channel_multipliers
        )

    @property
    def downsample_count(self) -> int:
        return len(self.channel_multipliers) - 1

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["channel_multipliers"] = list(self.channel_multipliers)
        record.update(
            {
                "name": "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI",
                "field_order": list(B5_FIELD_ORDER),
                "padding_by_axis": ["zeros", "zeros", "circular"],
                "full_field": True,
                "physics_derived_loss_allowed": False,
                "absolute_time_input_allowed": False,
                "absolute_z_coordinate_input_allowed": False,
            }
        )
        return record


def normalized_xy_coordinates(reference: Tensor) -> Tensor:
    """Return two static coordinate channels, constant along periodic z."""

    if reference.ndim != 5:
        raise ValueError("coordinate reference must be [batch,channel,x,y,z]")
    batch, _, n_x, n_y, n_z = reference.shape
    x = torch.linspace(-1.0, 1.0, n_x, dtype=reference.dtype, device=reference.device)
    y = torch.linspace(-1.0, 1.0, n_y, dtype=reference.dtype, device=reference.device)
    x_grid = x.reshape(1, 1, n_x, 1, 1).expand(batch, 1, n_x, n_y, n_z)
    y_grid = y.reshape(1, 1, 1, n_y, 1).expand(batch, 1, n_x, n_y, n_z)
    return torch.cat((x_grid, y_grid), dim=1)


def periodic_trilinear_upsample(inputs: Tensor, size: Sequence[int]) -> Tensor:
    """Upsample x/y bilinearly and z linearly across the periodic seam.

    PyTorch's ordinary trilinear interpolation clamps the end points of every
    axis.  That is correct for the two nonperiodic axes but silently breaks
    toroidal shift equivariance.  The frozen B5 path doubles every dimension,
    so this routine applies the ``align_corners=False`` weights explicitly at
    the circular z seam.
    """

    if inputs.ndim != 5:
        raise ValueError("B5 upsampling expects [batch,channel,x,y,z]")
    target = tuple(int(value) for value in size)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("B5 upsampling target must contain three positive sizes")
    batch, channels, n_x, n_y, n_z = inputs.shape
    if target[2] != 2 * n_z:
        raise ValueError("B5 periodic z upsampling must be exactly twofold")

    # Treat z as an independent batch axis so x/y interpolation cannot apply
    # any nonperiodic boundary convention to the toroidal coordinate.
    slices = inputs.permute(0, 4, 1, 2, 3).reshape(
        batch * n_z, channels, n_x, n_y
    )
    slices = F.interpolate(
        slices,
        size=target[:2],
        mode="bilinear",
        align_corners=False,
    )
    resized = slices.reshape(
        batch, n_z, channels, target[0], target[1]
    ).permute(0, 2, 3, 4, 1)

    previous = torch.roll(resized, 1, dims=-1)
    following = torch.roll(resized, -1, dims=-1)
    even = 0.75 * resized + 0.25 * previous
    odd = 0.75 * resized + 0.25 * following
    return torch.stack((even, odd), dim=-1).flatten(-2)


class NoiseFiLMResidualBlock3D(nn.Module):
    """Mixed-boundary 3-D residual block with noise-level FiLM."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        noise_features: int,
        kernel_size: int,
        maximum_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, noise_features) <= 0:
            raise ValueError("residual-block channels must be positive")
        self.normalization1 = nn.GroupNorm(
            _group_count(in_channels, maximum_groups), in_channels
        )
        self.convolution1 = _mixed_conv(
            in_channels, out_channels, kernel_size=kernel_size
        )
        self.normalization2 = nn.GroupNorm(
            _group_count(out_channels, maximum_groups), out_channels
        )
        self.noise_projection = nn.Linear(noise_features, 2 * out_channels)
        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        self.convolution2 = _mixed_conv(
            out_channels, out_channels, kernel_size=kernel_size
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else _mixed_conv(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            )
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.noise_features = int(noise_features)

    def forward(self, inputs: Tensor, noise_embedding: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.in_channels:
            raise ValueError("B5 residual block input shape differs")
        if noise_embedding.shape != (inputs.shape[0], self.noise_features):
            raise ValueError("B5 residual block noise embedding shape differs")
        hidden = self.convolution1(F.silu(self.normalization1(inputs)))
        scale, shift = self.noise_projection(noise_embedding).chunk(2, dim=1)
        scale = scale.reshape(scale.shape[0], scale.shape[1], 1, 1, 1)
        shift = shift.reshape(shift.shape[0], shift.shape[1], 1, 1, 1)
        hidden = self.normalization2(hidden) * (1.0 + scale) + shift
        hidden = self.convolution2(self.dropout(F.silu(hidden)))
        return (self.skip(inputs) + hidden) * math.sqrt(0.5)


class FieldResidualUNet3D(nn.Module):
    """Full-volume joint residual U-Net with periodic toroidal operations."""

    def __init__(
        self,
        config: FieldResidualUNetConfig = FieldResidualUNetConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        level_channels = config.level_channels
        total_input = (
            config.residual_channels
            + config.condition_channels
            + config.position_channels
        )
        self.noise_embedding = NoiseTimeEmbedding(config.noise_embedding_features)
        self.input_convolution = _mixed_conv(
            total_input,
            level_channels[0],
            kernel_size=config.kernel_size,
        )

        encoders: list[nn.ModuleList] = []
        downsamples: list[nn.Module] = []
        current = level_channels[0]
        for level, channels in enumerate(level_channels):
            blocks: list[nn.Module] = []
            for _ in range(config.residual_blocks_per_resolution):
                blocks.append(
                    NoiseFiLMResidualBlock3D(
                        current,
                        channels,
                        noise_features=config.noise_embedding_features,
                        kernel_size=config.kernel_size,
                        maximum_groups=config.group_norm_maximum_groups,
                        dropout=config.dropout,
                    )
                )
                current = channels
            encoders.append(nn.ModuleList(blocks))
            if level < len(level_channels) - 1:
                next_channels = level_channels[level + 1]
                downsamples.append(
                    _mixed_conv(
                        current,
                        next_channels,
                        kernel_size=config.kernel_size,
                        stride=2,
                    )
                )
                current = next_channels
        self.encoders = nn.ModuleList(encoders)
        self.downsamples = nn.ModuleList(downsamples)

        decoders: list[nn.ModuleList] = []
        up_convolutions: list[nn.Module] = []
        for reverse_index, level in enumerate(
            range(len(level_channels) - 1, -1, -1)
        ):
            channels = level_channels[level]
            if reverse_index > 0:
                up_convolutions.append(
                    _mixed_conv(
                        current,
                        channels,
                        kernel_size=config.kernel_size,
                    )
                )
                current = channels
            blocks = []
            for block_index in range(config.residual_blocks_per_resolution):
                block_input = current + channels if block_index == 0 else channels
                blocks.append(
                    NoiseFiLMResidualBlock3D(
                        block_input,
                        channels,
                        noise_features=config.noise_embedding_features,
                        kernel_size=config.kernel_size,
                        maximum_groups=config.group_norm_maximum_groups,
                        dropout=config.dropout,
                    )
                )
                current = channels
            decoders.append(nn.ModuleList(blocks))
        self.decoders = nn.ModuleList(decoders)
        self.up_convolutions = nn.ModuleList(up_convolutions)
        self.output_normalization = nn.GroupNorm(
            _group_count(current, config.group_norm_maximum_groups), current
        )
        self.output_convolution = _mixed_conv(
            current,
            config.residual_channels,
            kernel_size=config.kernel_size,
        )
        with torch.no_grad():
            self.output_convolution.weight.zero_()
            if self.output_convolution.bias is not None:
                self.output_convolution.bias.zero_()

    def forward(
        self,
        noisy_residual: Tensor,
        condition: Tensor,
        noise_coordinate: Tensor,
    ) -> Tensor:
        if noisy_residual.ndim != 5 or noisy_residual.shape[1] != 5:
            raise ValueError("noisy residual must be [batch,5,x,y,z]")
        expected_condition = (noisy_residual.shape[0], 10, *noisy_residual.shape[2:])
        if condition.shape != expected_condition:
            raise ValueError(f"B5 condition shape must be {expected_condition}")
        if noise_coordinate.shape != (noisy_residual.shape[0],):
            raise ValueError("B5 noise coordinate must contain one scalar per sample")
        if not (
            torch.isfinite(noisy_residual).all()
            and torch.isfinite(condition).all()
            and torch.isfinite(noise_coordinate).all()
        ):
            raise ValueError("B5 network inputs must be finite")

        position = normalized_xy_coordinates(noisy_residual)
        hidden = self.input_convolution(
            torch.cat((noisy_residual, condition, position), dim=1)
        )
        embedded = self.noise_embedding(noise_coordinate)
        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoders):
            for block in blocks:
                hidden = block(hidden, embedded)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        up_index = 0
        for reverse_index, blocks in enumerate(self.decoders):
            skip = skips[-1 - reverse_index]
            if reverse_index > 0:
                hidden = periodic_trilinear_upsample(hidden, skip.shape[2:])
                hidden = self.up_convolutions[up_index](hidden)
                up_index += 1
            if hidden.shape[2:] != skip.shape[2:]:
                raise RuntimeError("B5 decoder and skip spatial shapes differ")
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, embedded)
        return self.output_convolution(F.silu(self.output_normalization(hidden)))


def _expand_sample_scalar(values: Tensor, reference: Tensor) -> Tensor:
    result = values
    while result.ndim < reference.ndim:
        result = result[..., None]
    return result


@dataclass(frozen=True)
class EDMLossResult:
    loss: Tensor
    unweighted_mse: Tensor
    sigma_minimum: float
    sigma_maximum: float


class JointFieldResidualEDM(nn.Module):
    """EDM preconditioning, objective, and deterministic Heun sampler."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        residual_scales: Sequence[float] = B5_RESIDUAL_SCALES,
        sigma_data: float = 1.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ) -> None:
        super().__init__()
        scales = torch.as_tensor(tuple(residual_scales), dtype=torch.float32)
        if scales.shape != (5,) or not torch.all(torch.isfinite(scales) & (scales > 0)):
            raise ValueError("B5 residual scaling must contain five positive values")
        if sigma_data != 1.0 or p_std <= 0.0:
            raise ValueError("B5 smoke requires sigma_data=1 and positive P_std")
        self.backbone = backbone
        self.register_buffer("residual_scales", scales.reshape(1, 5, 1, 1, 1))
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    def normalize_residual(self, residual: Tensor) -> Tensor:
        if residual.ndim != 5 or residual.shape[1] != 5:
            raise ValueError("B5 residual must be [batch,5,x,y,z]")
        return residual / self.residual_scales.to(residual)

    def denormalize_residual(self, normalized: Tensor) -> Tensor:
        if normalized.ndim < 5 or normalized.shape[-4] != 5:
            raise ValueError("B5 normalized residual channel axis differs")
        scales = self.residual_scales.to(normalized)
        while scales.ndim < normalized.ndim:
            scales = scales.unsqueeze(1)
        return normalized * scales

    def denoise(self, noisy: Tensor, condition: Tensor, sigma: Tensor) -> Tensor:
        if noisy.ndim != 5 or noisy.shape[1] != 5:
            raise ValueError("B5 noisy residual must be [batch,5,x,y,z]")
        if condition.shape != (noisy.shape[0], 10, *noisy.shape[2:]):
            raise ValueError("B5 EDM condition shape differs")
        sigma_values = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype)
        if sigma_values.ndim == 0:
            sigma_values = sigma_values.expand(noisy.shape[0])
        if sigma_values.shape != (noisy.shape[0],):
            raise ValueError("B5 sigma must be scalar or one value per sample")
        if not torch.all(torch.isfinite(sigma_values) & (sigma_values > 0.0)):
            raise ValueError("B5 sigma values must be finite and positive")
        sigma_x = _expand_sample_scalar(sigma_values, noisy)
        denominator = torch.sqrt(sigma_x.square() + 1.0)
        c_in = torch.reciprocal(denominator)
        c_skip = torch.reciprocal(sigma_x.square() + 1.0)
        c_out = sigma_x / denominator
        c_noise = torch.log(sigma_values) * 0.25
        network = self.backbone(c_in * noisy, condition, c_noise)
        if network.shape != noisy.shape:
            raise RuntimeError("B5 backbone output shape differs")
        return c_skip * noisy + c_out * network

    def training_loss(
        self,
        clean: Tensor,
        condition: Tensor,
        *,
        sigma: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> EDMLossResult:
        if clean.ndim != 5 or clean.shape[1] != 5:
            raise ValueError("B5 clean target must be [batch,5,x,y,z]")
        if condition.shape != (clean.shape[0], 10, *clean.shape[2:]):
            raise ValueError("B5 loss condition shape differs")
        if sigma is None:
            gaussian = torch.randn(
                (clean.shape[0],),
                generator=generator,
                device=clean.device,
                dtype=torch.float32,
            )
            sigma_values = torch.exp(self.p_mean + self.p_std * gaussian).to(clean)
        else:
            sigma_values = torch.as_tensor(sigma, device=clean.device, dtype=clean.dtype)
            if sigma_values.ndim == 0:
                sigma_values = sigma_values.expand(clean.shape[0])
        if sigma_values.shape != (clean.shape[0],) or not torch.all(
            torch.isfinite(sigma_values) & (sigma_values > 0.0)
        ):
            raise ValueError("B5 training sigma values differ")
        if noise is None:
            noise_values = torch.randn(
                clean.shape,
                generator=generator,
                device=clean.device,
                dtype=clean.dtype,
            )
        else:
            noise_values = torch.as_tensor(noise, device=clean.device, dtype=clean.dtype)
        if noise_values.shape != clean.shape or not torch.isfinite(noise_values).all():
            raise ValueError("B5 training noise shape or values differ")
        sigma_x = _expand_sample_scalar(sigma_values, clean)
        noisy = clean + sigma_x * noise_values
        prediction = self.denoise(noisy, condition, sigma_values)
        squared = (prediction.float() - clean.float()).square()
        per_sample_mse = squared.flatten(1).mean(dim=1)
        weights = (sigma_values.float().square() + 1.0) / sigma_values.float().square()
        loss = torch.mean(weights * per_sample_mse)
        return EDMLossResult(
            loss=loss,
            unweighted_mse=torch.mean(per_sample_mse),
            sigma_minimum=float(torch.min(sigma_values.detach().float()).cpu()),
            sigma_maximum=float(torch.max(sigma_values.detach().float()).cpu()),
        )

    @staticmethod
    def sampling_schedule(
        *,
        steps: int,
        sigma_max: float,
        sigma_min: float,
        rho: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if steps < 2 or not 0.0 < sigma_min < sigma_max or rho <= 0.0:
            raise ValueError("invalid B5 EDM sampling schedule")
        ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float64)
        maximum = float(sigma_max) ** (1.0 / float(rho))
        minimum = float(sigma_min) ** (1.0 / float(rho))
        sigma = (maximum + ramp * (minimum - maximum)).pow(float(rho))
        sigma = torch.cat((sigma, sigma.new_zeros(1)))
        return sigma.to(dtype=dtype)

    @torch.no_grad()
    def sample_normalized(
        self,
        condition: Tensor,
        initial_noise: Tensor,
        *,
        steps: int = 18,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
    ) -> Tensor:
        """Return `[batch,member,5,x,y,z]` normalized residual samples."""

        if condition.ndim != 5 or condition.shape[1] != 10:
            raise ValueError("B5 sampling condition must be [batch,10,x,y,z]")
        if initial_noise.ndim != 6 or initial_noise.shape[0] != condition.shape[0]:
            raise ValueError("B5 initial noise must be [batch,member,5,x,y,z]")
        if initial_noise.shape[2:] != (5, *condition.shape[2:]):
            raise ValueError("B5 initial noise field shape differs")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("B5 initial noise must be finite")
        batch, members = initial_noise.shape[:2]
        expanded_condition = (
            condition[:, None]
            .expand(batch, members, *condition.shape[1:])
            .reshape(batch * members, *condition.shape[1:])
            .contiguous()
        )
        sample = initial_noise.reshape(batch * members, *initial_noise.shape[2:])
        schedule = self.sampling_schedule(
            steps=steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            device=sample.device,
            dtype=sample.dtype,
        )
        sample = sample * schedule[0]
        for index in range(len(schedule) - 1):
            current_sigma = schedule[index]
            next_sigma = schedule[index + 1]
            sigma_batch = current_sigma.expand(sample.shape[0])
            denoised = self.denoise(sample, expanded_condition, sigma_batch)
            derivative = (sample - denoised) / current_sigma
            proposed = sample + (next_sigma - current_sigma) * derivative
            if float(next_sigma) != 0.0:
                next_batch = next_sigma.expand(sample.shape[0])
                next_denoised = self.denoise(
                    proposed, expanded_condition, next_batch
                )
                next_derivative = (proposed - next_denoised) / next_sigma
                sample = sample + (next_sigma - current_sigma) * 0.5 * (
                    derivative + next_derivative
                )
            else:
                sample = proposed
        return sample.reshape(batch, members, *sample.shape[1:])

    def compose_fields(
        self,
        deterministic_mean: Tensor,
        normalized_residual: Tensor,
    ) -> Tensor:
        """Compose canonical one-step fields `[B,M,1,5,x,y,z]`."""

        if deterministic_mean.ndim != 5 or deterministic_mean.shape[1] != 5:
            raise ValueError("B5 deterministic mean must be [batch,5,x,y,z]")
        if normalized_residual.ndim != 6 or normalized_residual.shape[0] != (
            deterministic_mean.shape[0]
        ):
            raise ValueError("B5 sampled residual must be [batch,member,5,x,y,z]")
        if normalized_residual.shape[2:] != deterministic_mean.shape[1:]:
            raise ValueError("B5 residual and deterministic mean shapes differ")
        residual = self.denormalize_residual(normalized_residual)
        fields = deterministic_mean[:, None] + residual
        return fields[:, :, None]
