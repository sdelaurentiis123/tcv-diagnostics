"""Equivariant context-conditioned residual diffusion components.

The module is deliberately data agnostic.  It implements the ECRD field-space
mean/innovation decomposition frozen in
``paper0/protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md`` and never reads a
simulation, split, diagnostic, or held-out artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .field_residual_edm import (
    B5_FIELD_ORDER,
    B5_RESIDUAL_SCALES,
    B5_SPATIAL_SHAPE,
    EDMLossResult,
    _expand_sample_scalar,
    _group_count,
    normalized_xy_coordinates,
    periodic_trilinear_upsample,
)
from .layers import make_conv
from .modulated_vit import NoiseTimeEmbedding


ECRD_ARMS = ("B5-Context", "ECRD", "ECRD-History")


def _mixed_conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] | None = None,
    bias: bool = True,
) -> nn.Module:
    amount: int | Sequence[int]
    amount = kernel_size // 2 if padding is None else padding
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


def xy_bilinear_upsample(inputs: Tensor, size: Sequence[int]) -> Tensor:
    """Resize only nonperiodic x/y while preserving every toroidal cell."""

    if inputs.ndim != 5:
        raise ValueError("ECRD x/y upsampling expects [batch,channel,x,y,z]")
    target = tuple(int(value) for value in size)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("ECRD upsampling target must have three positive sizes")
    batch, channels, n_x, n_y, n_z = inputs.shape
    if target[2] != n_z:
        raise ValueError("ECRD is forbidden to resize the toroidal axis")
    slices = inputs.permute(0, 4, 1, 2, 3).reshape(
        batch * n_z, channels, n_x, n_y
    )
    slices = F.interpolate(
        slices,
        size=target[:2],
        mode="bilinear",
        align_corners=False,
    )
    return slices.reshape(
        batch, n_z, channels, target[0], target[1]
    ).permute(0, 2, 3, 4, 1).contiguous()


@dataclass(frozen=True)
class MultiscaleNoiseConfig:
    """Fixed full-rank global/mesoscale/local Gaussian construction."""

    global_weight: float = 0.35
    mesoscale_weight: float = 0.55
    local_weight: float = 1.0
    mesoscale_xy: tuple[int, int] = (4, 4)

    def __post_init__(self) -> None:
        weights = (self.global_weight, self.mesoscale_weight, self.local_weight)
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in weights):
            raise ValueError("multiscale noise weights must be finite and nonnegative")
        if self.local_weight <= 0.0 or sum(value * value for value in weights) <= 0.0:
            raise ValueError("multiscale noise must retain a positive local component")
        if len(self.mesoscale_xy) != 2 or any(
            int(value) <= 0 for value in self.mesoscale_xy
        ):
            raise ValueError("mesoscale x/y factors must be positive")

    @property
    def normalization(self) -> float:
        return math.sqrt(
            self.global_weight**2
            + self.mesoscale_weight**2
            + self.local_weight**2
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["mesoscale_xy"] = list(self.mesoscale_xy)
        record.update(
            {
                "distribution": "full_rank_global_plus_xy_mesoscale_plus_local_Gaussian",
                "toroidal_cells_retained_at_all_scales": True,
                "posthoc_spread_multiplier": False,
            }
        )
        return record


def compose_multiscale_noise(
    *,
    global_noise: Tensor,
    mesoscale_noise: Tensor,
    local_noise: Tensor,
    config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
) -> Tensor:
    """Combine independent scale components with unit pointwise variance."""

    if local_noise.ndim != 5:
        raise ValueError("local noise must be [batch,channel,x,y,z]")
    batch, channels, n_x, n_y, n_z = local_noise.shape
    if global_noise.shape != (batch, channels, 1, 1, 1):
        raise ValueError("global noise shape differs")
    factor_x, factor_y = config.mesoscale_xy
    if n_x % factor_x or n_y % factor_y:
        raise ValueError("full x/y sizes must be divisible by mesoscale factors")
    expected_meso = (batch, channels, n_x // factor_x, n_y // factor_y, n_z)
    if mesoscale_noise.shape != expected_meso:
        raise ValueError(f"mesoscale noise shape must be {expected_meso}")
    if not (
        torch.isfinite(global_noise).all()
        and torch.isfinite(mesoscale_noise).all()
        and torch.isfinite(local_noise).all()
    ):
        raise ValueError("multiscale noise components must be finite")
    meso = mesoscale_noise.repeat_interleave(factor_x, dim=-3).repeat_interleave(
        factor_y, dim=-2
    )
    global_full = global_noise.expand_as(local_noise)
    return (
        config.global_weight * global_full
        + config.mesoscale_weight * meso
        + config.local_weight * local_noise
    ) / config.normalization


def sample_multiscale_noise(
    reference: Tensor,
    *,
    config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ECRD corruption noise without downsampling periodic z."""

    if reference.ndim != 5:
        raise ValueError("multiscale noise reference must be five-dimensional")
    batch, channels, n_x, n_y, n_z = reference.shape
    factor_x, factor_y = config.mesoscale_xy
    if n_x % factor_x or n_y % factor_y:
        raise ValueError("reference x/y sizes are incompatible with mesoscale noise")
    kwargs = {
        "device": reference.device,
        "dtype": reference.dtype,
        "generator": generator,
    }
    return compose_multiscale_noise(
        global_noise=torch.randn((batch, channels, 1, 1, 1), **kwargs),
        mesoscale_noise=torch.randn(
            (batch, channels, n_x // factor_x, n_y // factor_y, n_z), **kwargs
        ),
        local_noise=torch.randn(reference.shape, **kwargs),
        config=config,
    )


@dataclass(frozen=True)
class ECRDUNetConfig:
    """Deep-conditioning U-Net configuration for B5-Context and ECRD."""

    arm: str = "ECRD"
    history_frames: int = 1
    residual_channels: int = 5
    parent_mean_channels: int = 5
    position_channels: int = 2
    base_channels: int = 28
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    residual_blocks_per_resolution: int = 2
    conditioner_blocks_per_resolution: int = 1
    noise_embedding_features: int = 256
    kernel_size: int = 3
    group_norm_maximum_groups: int = 8
    dropout: float = 0.0
    preserve_toroidal_resolution: bool = True
    mean_head: bool = True
    multiscale_noise: bool = True
    mean_head_channels: int = 24
    mean_head_blocks: int = 3

    def __post_init__(self) -> None:
        if self.arm not in ECRD_ARMS:
            raise ValueError(f"unsupported ECRD arm {self.arm!r}")
        if self.history_frames not in (1, 2):
            raise ValueError("ECRD supports one or two C5P history frames")
        expected_history = 2 if self.arm == "ECRD-History" else 1
        if self.history_frames != expected_history:
            raise ValueError("ECRD arm/history identity differs")
        expected_equivariant = self.arm != "B5-Context"
        if self.preserve_toroidal_resolution != expected_equivariant:
            raise ValueError("arm toroidal-resolution contract differs")
        if self.mean_head != expected_equivariant:
            raise ValueError("arm mean-head contract differs")
        if self.multiscale_noise != expected_equivariant:
            raise ValueError("arm multiscale-noise contract differs")
        counts = (
            self.residual_channels,
            self.parent_mean_channels,
            self.position_channels,
            self.base_channels,
            self.residual_blocks_per_resolution,
            self.conditioner_blocks_per_resolution,
            self.noise_embedding_features,
            self.kernel_size,
            self.group_norm_maximum_groups,
            self.mean_head_channels,
            self.mean_head_blocks,
        )
        if min(counts) <= 0:
            raise ValueError("ECRD architecture counts must be positive")
        if self.residual_channels != 5 or self.parent_mean_channels != 5:
            raise ValueError("ECRD jointly predicts the five C5P fields")
        if self.position_channels != 2 or self.kernel_size != 3:
            raise ValueError("ECRD uses x/y coordinates and kernel size three")
        if not self.channel_multipliers or any(
            int(value) <= 0 for value in self.channel_multipliers
        ):
            raise ValueError("ECRD channel multipliers must be positive")
        if self.noise_embedding_features % 2:
            raise ValueError("ECRD noise embedding width must be even")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ECRD dropout must lie in [0,1)")

    @property
    def condition_channels(self) -> int:
        return 5 * self.history_frames + self.parent_mean_channels

    @property
    def level_channels(self) -> tuple[int, ...]:
        return tuple(
            self.base_channels * int(multiplier)
            for multiplier in self.channel_multipliers
        )

    @property
    def downsample_stride(self) -> tuple[int, int, int]:
        return (2, 2, 1) if self.preserve_toroidal_resolution else (2, 2, 2)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["channel_multipliers"] = list(self.channel_multipliers)
        record.update(
            {
                "condition_channels": self.condition_channels,
                "field_order": list(B5_FIELD_ORDER),
                "padding_by_axis": ["zeros", "zeros", "circular"],
                "downsample_stride_xyz": list(self.downsample_stride),
                "absolute_time_input_allowed": False,
                "absolute_z_coordinate_input_allowed": False,
                "physics_derived_loss_allowed": False,
                "joint_field_generation": True,
            }
        )
        return record


class _ContextBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        maximum_groups: int,
    ) -> None:
        super().__init__()
        self.normalization = nn.GroupNorm(_group_count(channels, maximum_groups), channels)
        self.convolution1 = _mixed_conv(channels, channels, kernel_size=kernel_size)
        self.convolution2 = _mixed_conv(channels, channels, kernel_size=kernel_size)
        self.channels = int(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError("context block input shape differs")
        hidden = self.convolution1(F.silu(self.normalization(inputs)))
        hidden = self.convolution2(F.silu(hidden))
        return (inputs + hidden) * math.sqrt(0.5)


class ContextFeaturePyramid3D(nn.Module):
    """Raw-field conditioner evaluated at every U-Net resolution."""

    def __init__(self, config: ECRDUNetConfig) -> None:
        super().__init__()
        levels = config.level_channels
        self.input_convolution = _mixed_conv(
            config.condition_channels,
            levels[0],
            kernel_size=config.kernel_size,
        )
        blocks: list[nn.ModuleList] = []
        downsample: list[nn.Module] = []
        current = levels[0]
        for level, channels in enumerate(levels):
            if current != channels:
                raise RuntimeError("context pyramid channel transition drifted")
            blocks.append(
                nn.ModuleList(
                    [
                        _ContextBlock3D(
                            channels,
                            kernel_size=config.kernel_size,
                            maximum_groups=config.group_norm_maximum_groups,
                        )
                        for _ in range(config.conditioner_blocks_per_resolution)
                    ]
                )
            )
            if level < len(levels) - 1:
                next_channels = levels[level + 1]
                downsample.append(
                    _mixed_conv(
                        channels,
                        next_channels,
                        kernel_size=config.kernel_size,
                        stride=config.downsample_stride,
                    )
                )
                current = next_channels
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList(downsample)
        self.condition_channels = config.condition_channels

    def forward(self, condition: Tensor) -> tuple[Tensor, ...]:
        if condition.ndim != 5 or condition.shape[1] != self.condition_channels:
            raise ValueError("context pyramid input shape differs")
        hidden = self.input_convolution(condition)
        features: list[Tensor] = []
        for level, blocks in enumerate(self.blocks):
            for block in blocks:
                hidden = block(hidden)
            features.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        return tuple(features)


class SpatialContextFiLMResidualBlock3D(nn.Module):
    """Residual block modulated by noise time and spatial raw-state context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        context_channels: int,
        *,
        noise_features: int,
        kernel_size: int,
        maximum_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
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
        self.context_projection = _mixed_conv(
            context_channels,
            2 * out_channels,
            kernel_size=1,
            padding=0,
        )
        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        self.convolution2 = _mixed_conv(
            out_channels, out_channels, kernel_size=kernel_size
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else _mixed_conv(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.context_channels = int(context_channels)
        self.noise_features = int(noise_features)

    def forward(
        self,
        inputs: Tensor,
        noise_embedding: Tensor,
        context: Tensor,
    ) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.in_channels:
            raise ValueError("ECRD residual block input shape differs")
        if noise_embedding.shape != (inputs.shape[0], self.noise_features):
            raise ValueError("ECRD noise embedding shape differs")
        if context.shape != (
            inputs.shape[0],
            self.context_channels,
            *inputs.shape[2:],
        ):
            raise ValueError("ECRD spatial context shape differs")
        hidden = self.convolution1(F.silu(self.normalization1(inputs)))
        noise_scale, noise_shift = self.noise_projection(noise_embedding).chunk(2, dim=1)
        noise_scale = noise_scale.reshape(noise_scale.shape[0], -1, 1, 1, 1)
        noise_shift = noise_shift.reshape(noise_shift.shape[0], -1, 1, 1, 1)
        context_scale, context_shift = self.context_projection(context).chunk(2, dim=1)
        hidden = self.normalization2(hidden) * (
            1.0 + noise_scale + context_scale
        ) + noise_shift + context_shift
        hidden = self.convolution2(self.dropout(F.silu(hidden)))
        return (self.skip(inputs) + hidden) * math.sqrt(0.5)


class DeepConditionalResidualUNet3D(nn.Module):
    """Joint field denoiser with spatial conditioning at every resolution."""

    def __init__(self, config: ECRDUNetConfig = ECRDUNetConfig()) -> None:
        super().__init__()
        self.config = config
        levels = config.level_channels
        self.conditioner = ContextFeaturePyramid3D(config)
        self.noise_embedding = NoiseTimeEmbedding(config.noise_embedding_features)
        self.input_convolution = _mixed_conv(
            config.residual_channels + config.position_channels,
            levels[0],
            kernel_size=config.kernel_size,
        )

        encoders: list[nn.ModuleList] = []
        downsamples: list[nn.Module] = []
        current = levels[0]
        for level, channels in enumerate(levels):
            level_blocks: list[nn.Module] = []
            for _ in range(config.residual_blocks_per_resolution):
                level_blocks.append(
                    SpatialContextFiLMResidualBlock3D(
                        current,
                        channels,
                        channels,
                        noise_features=config.noise_embedding_features,
                        kernel_size=config.kernel_size,
                        maximum_groups=config.group_norm_maximum_groups,
                        dropout=config.dropout,
                    )
                )
                current = channels
            encoders.append(nn.ModuleList(level_blocks))
            if level < len(levels) - 1:
                next_channels = levels[level + 1]
                downsamples.append(
                    _mixed_conv(
                        current,
                        next_channels,
                        kernel_size=config.kernel_size,
                        stride=config.downsample_stride,
                    )
                )
                current = next_channels
        self.encoders = nn.ModuleList(encoders)
        self.downsamples = nn.ModuleList(downsamples)

        decoders: list[nn.ModuleList] = []
        up_convolutions: list[nn.Module] = []
        for reverse_index, level in enumerate(range(len(levels) - 1, -1, -1)):
            channels = levels[level]
            if reverse_index > 0:
                up_convolutions.append(
                    _mixed_conv(current, channels, kernel_size=config.kernel_size)
                )
                current = channels
            level_blocks = []
            for block_index in range(config.residual_blocks_per_resolution):
                block_input = current + channels if block_index == 0 else channels
                level_blocks.append(
                    SpatialContextFiLMResidualBlock3D(
                        block_input,
                        channels,
                        channels,
                        noise_features=config.noise_embedding_features,
                        kernel_size=config.kernel_size,
                        maximum_groups=config.group_norm_maximum_groups,
                        dropout=config.dropout,
                    )
                )
                current = channels
            decoders.append(nn.ModuleList(level_blocks))
        self.decoders = nn.ModuleList(decoders)
        self.up_convolutions = nn.ModuleList(up_convolutions)
        self.output_normalization = nn.GroupNorm(
            _group_count(current, config.group_norm_maximum_groups), current
        )
        self.output_convolution = _mixed_conv(
            current, config.residual_channels, kernel_size=config.kernel_size
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
            raise ValueError("ECRD noisy residual must be [batch,5,x,y,z]")
        expected = (
            noisy_residual.shape[0],
            self.config.condition_channels,
            *noisy_residual.shape[2:],
        )
        if condition.shape != expected:
            raise ValueError(f"ECRD condition shape must be {expected}")
        if noise_coordinate.shape != (noisy_residual.shape[0],):
            raise ValueError("ECRD noise coordinate shape differs")
        if not (
            torch.isfinite(noisy_residual).all()
            and torch.isfinite(condition).all()
            and torch.isfinite(noise_coordinate).all()
        ):
            raise ValueError("ECRD network inputs must be finite")

        context_features = self.conditioner(condition)
        position = normalized_xy_coordinates(noisy_residual)
        hidden = self.input_convolution(torch.cat((noisy_residual, position), dim=1))
        embedded = self.noise_embedding(noise_coordinate)
        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoders):
            for block in blocks:
                hidden = block(hidden, embedded, context_features[level])
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        up_index = 0
        for reverse_index, blocks in enumerate(self.decoders):
            level = len(context_features) - 1 - reverse_index
            skip = skips[level]
            if reverse_index > 0:
                if self.config.preserve_toroidal_resolution:
                    hidden = xy_bilinear_upsample(hidden, skip.shape[2:])
                else:
                    hidden = periodic_trilinear_upsample(hidden, skip.shape[2:])
                hidden = self.up_convolutions[up_index](hidden)
                up_index += 1
            if hidden.shape[2:] != skip.shape[2:]:
                raise RuntimeError("ECRD decoder and skip spatial shapes differ")
            hidden = torch.cat((hidden, skip), dim=1)
            for block in blocks:
                hidden = block(hidden, embedded, context_features[level])
        return self.output_convolution(F.silu(self.output_normalization(hidden)))


class _MeanResidualBlock3D(nn.Module):
    def __init__(self, channels: int, *, maximum_groups: int) -> None:
        super().__init__()
        self.normalization1 = nn.GroupNorm(_group_count(channels, maximum_groups), channels)
        self.convolution1 = _mixed_conv(channels, channels, kernel_size=3)
        self.normalization2 = nn.GroupNorm(_group_count(channels, maximum_groups), channels)
        self.convolution2 = _mixed_conv(channels, channels, kernel_size=3)
        self.channels = int(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError("ECRD mean block shape differs")
        hidden = self.convolution1(F.silu(self.normalization1(inputs)))
        hidden = self.convolution2(F.silu(self.normalization2(hidden)))
        return (inputs + hidden) * math.sqrt(0.5)


class EquivariantResidualMeanHead3D(nn.Module):
    """Small z-equivariant normalized correction to the frozen parent mean."""

    def __init__(self, config: ECRDUNetConfig) -> None:
        super().__init__()
        if not config.mean_head:
            raise ValueError("mean head requested for an arm without one")
        self.input_convolution = _mixed_conv(
            config.condition_channels + config.position_channels,
            config.mean_head_channels,
            kernel_size=3,
        )
        self.blocks = nn.ModuleList(
            [
                _MeanResidualBlock3D(
                    config.mean_head_channels,
                    maximum_groups=config.group_norm_maximum_groups,
                )
                for _ in range(config.mean_head_blocks)
            ]
        )
        self.output_normalization = nn.GroupNorm(
            _group_count(
                config.mean_head_channels, config.group_norm_maximum_groups
            ),
            config.mean_head_channels,
        )
        self.output_convolution = _mixed_conv(
            config.mean_head_channels, 5, kernel_size=3
        )
        self.condition_channels = config.condition_channels
        with torch.no_grad():
            self.output_convolution.weight.zero_()
            if self.output_convolution.bias is not None:
                self.output_convolution.bias.zero_()

    def forward(self, condition: Tensor) -> Tensor:
        if condition.ndim != 5 or condition.shape[1] != self.condition_channels:
            raise ValueError("ECRD mean-head condition shape differs")
        position = normalized_xy_coordinates(condition)
        hidden = self.input_convolution(torch.cat((condition, position), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_convolution(F.silu(self.output_normalization(hidden)))


@dataclass(frozen=True)
class ECRDLossResult:
    loss: Tensor
    edm_loss: Tensor
    unweighted_edm_mse: Tensor
    mean_mse: Tensor
    sigma_minimum: float
    sigma_maximum: float


class ECRDTransition(nn.Module):
    """Conditional mean plus joint normalized stochastic innovation."""

    def __init__(
        self,
        config: ECRDUNetConfig = ECRDUNetConfig(),
        *,
        residual_scales: Sequence[float] = B5_RESIDUAL_SCALES,
        noise_config: MultiscaleNoiseConfig = MultiscaleNoiseConfig(),
        sigma_data: float = 1.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        mean_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        scales = torch.as_tensor(tuple(residual_scales), dtype=torch.float32)
        if scales.shape != (5,) or not torch.all(torch.isfinite(scales) & (scales > 0)):
            raise ValueError("ECRD requires five positive frozen residual scales")
        if sigma_data != 1.0 or p_std <= 0.0 or mean_loss_weight != 1.0:
            raise ValueError("ECRD EDM or mean-loss constants differ")
        self.config = config
        self.noise_config = noise_config
        self.backbone = DeepConditionalResidualUNet3D(config)
        self.mean_head = (
            EquivariantResidualMeanHead3D(config) if config.mean_head else None
        )
        self.register_buffer("residual_scales", scales.reshape(1, 5, 1, 1, 1))
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.mean_loss_weight = float(mean_loss_weight)

    def mean_correction_normalized(self, condition: Tensor) -> Tensor:
        if condition.ndim != 5 or condition.shape[1] != self.config.condition_channels:
            raise ValueError("ECRD condition shape differs")
        if self.mean_head is None:
            return condition.new_zeros((condition.shape[0], 5, *condition.shape[2:]))
        return self.mean_head(condition)

    def denormalize_residual(self, normalized: Tensor) -> Tensor:
        if normalized.ndim < 5 or normalized.shape[-4] != 5:
            raise ValueError("ECRD normalized residual channel axis differs")
        scales = self.residual_scales.to(normalized)
        while scales.ndim < normalized.ndim:
            scales = scales.unsqueeze(1)
        return normalized * scales

    def denoise(self, noisy: Tensor, condition: Tensor, sigma: Tensor) -> Tensor:
        if noisy.ndim != 5 or noisy.shape[1] != 5:
            raise ValueError("ECRD noisy innovation must be [batch,5,x,y,z]")
        expected = (noisy.shape[0], self.config.condition_channels, *noisy.shape[2:])
        if condition.shape != expected:
            raise ValueError("ECRD denoising condition shape differs")
        sigma_values = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype)
        if sigma_values.ndim == 0:
            sigma_values = sigma_values.expand(noisy.shape[0])
        if sigma_values.shape != (noisy.shape[0],) or not torch.all(
            torch.isfinite(sigma_values) & (sigma_values > 0.0)
        ):
            raise ValueError("ECRD sigma values differ")
        sigma_x = _expand_sample_scalar(sigma_values, noisy)
        denominator = torch.sqrt(sigma_x.square() + 1.0)
        c_in = torch.reciprocal(denominator)
        c_skip = torch.reciprocal(sigma_x.square() + 1.0)
        c_out = sigma_x / denominator
        c_noise = torch.log(sigma_values) * 0.25
        network = self.backbone(c_in * noisy, condition, c_noise)
        if network.shape != noisy.shape:
            raise RuntimeError("ECRD backbone output shape differs")
        return c_skip * noisy + c_out * network

    def training_loss(
        self,
        normalized_parent_residual: Tensor,
        condition: Tensor,
        *,
        sigma: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ECRDLossResult:
        target = normalized_parent_residual
        if target.ndim != 5 or target.shape[1] != 5:
            raise ValueError("ECRD parent residual must be [batch,5,x,y,z]")
        if condition.shape != (
            target.shape[0], self.config.condition_channels, *target.shape[2:]
        ):
            raise ValueError("ECRD loss condition shape differs")
        correction = self.mean_correction_normalized(condition)
        mean_mse = (
            torch.mean((correction.float() - target.float()).square())
            if self.mean_head is not None
            else target.new_zeros((), dtype=torch.float32)
        )
        clean = target - correction.detach()
        if sigma is None:
            gaussian = torch.randn(
                (target.shape[0],),
                generator=generator,
                device=target.device,
                dtype=torch.float32,
            )
            sigma_values = torch.exp(self.p_mean + self.p_std * gaussian).to(target)
        else:
            sigma_values = torch.as_tensor(sigma, device=target.device, dtype=target.dtype)
            if sigma_values.ndim == 0:
                sigma_values = sigma_values.expand(target.shape[0])
        if sigma_values.shape != (target.shape[0],) or not torch.all(
            torch.isfinite(sigma_values) & (sigma_values > 0.0)
        ):
            raise ValueError("ECRD training sigma values differ")
        if noise is None:
            noise_values = (
                sample_multiscale_noise(clean, config=self.noise_config, generator=generator)
                if self.config.multiscale_noise
                else torch.randn(
                    clean.shape,
                    generator=generator,
                    device=clean.device,
                    dtype=clean.dtype,
                )
            )
        else:
            noise_values = torch.as_tensor(noise, device=clean.device, dtype=clean.dtype)
        if noise_values.shape != clean.shape or not torch.isfinite(noise_values).all():
            raise ValueError("ECRD training noise shape or values differ")
        sigma_x = _expand_sample_scalar(sigma_values, clean)
        noisy = clean + sigma_x * noise_values
        prediction = self.denoise(noisy, condition, sigma_values)
        per_sample_mse = (prediction.float() - clean.float()).square().flatten(1).mean(1)
        weights = (sigma_values.float().square() + 1.0) / sigma_values.float().square()
        edm_loss = torch.mean(weights * per_sample_mse)
        total = edm_loss + (self.mean_loss_weight * mean_mse if self.mean_head is not None else 0.0)
        return ECRDLossResult(
            loss=total,
            edm_loss=edm_loss,
            unweighted_edm_mse=torch.mean(per_sample_mse),
            mean_mse=mean_mse,
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
            raise ValueError("invalid ECRD sampling schedule")
        ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float64)
        maximum = float(sigma_max) ** (1.0 / float(rho))
        minimum = float(sigma_min) ** (1.0 / float(rho))
        sigma = (maximum + ramp * (minimum - maximum)).pow(float(rho))
        return torch.cat((sigma, sigma.new_zeros(1))).to(dtype=dtype)

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
        """Return `[batch,member,5,x,y,z]` normalized innovations."""

        if condition.ndim != 5 or condition.shape[1] != self.config.condition_channels:
            raise ValueError("ECRD sampling condition shape differs")
        if initial_noise.ndim != 6 or initial_noise.shape[0] != condition.shape[0]:
            raise ValueError("ECRD initial noise must be [batch,member,5,x,y,z]")
        if initial_noise.shape[2:] != (5, *condition.shape[2:]):
            raise ValueError("ECRD initial noise field shape differs")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("ECRD initial noise must be finite")
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
                next_denoised = self.denoise(proposed, expanded_condition, next_batch)
                next_derivative = (proposed - next_denoised) / next_sigma
                sample = sample + (next_sigma - current_sigma) * 0.5 * (
                    derivative + next_derivative
                )
            else:
                sample = proposed
        return sample.reshape(batch, members, *sample.shape[1:])

    def compose_fields(
        self,
        parent_mean: Tensor,
        condition: Tensor,
        normalized_innovation: Tensor,
    ) -> Tensor:
        """Compose canonical one-step fields `[B,M,1,5,x,y,z]`."""

        if parent_mean.ndim != 5 or parent_mean.shape[1] != 5:
            raise ValueError("ECRD parent mean must be [batch,5,x,y,z]")
        if normalized_innovation.ndim != 6 or normalized_innovation.shape[0] != parent_mean.shape[0]:
            raise ValueError("ECRD innovation must be [batch,member,5,x,y,z]")
        if normalized_innovation.shape[2:] != parent_mean.shape[1:]:
            raise ValueError("ECRD innovation and mean shapes differ")
        correction = self.denormalize_residual(
            self.mean_correction_normalized(condition)
        )
        innovation = self.denormalize_residual(normalized_innovation)
        fields = parent_mean[:, None] + correction[:, None] + innovation
        return fields[:, :, None]

    def to_record(self) -> dict[str, Any]:
        return {
            "family": "equivariant_context_conditioned_joint_residual_EDM",
            "config": self.config.to_record(),
            "noise": (
                self.noise_config.to_record()
                if self.config.multiscale_noise
                else {"distribution": "elementwise_standard_normal"}
            ),
            "residual_scales": list(B5_RESIDUAL_SCALES),
            "sigma_data": self.sigma_data,
            "P_mean": self.p_mean,
            "P_std": self.p_std,
            "mean_loss_weight": self.mean_loss_weight,
            "physics_derived_loss_used": False,
        }


def symmetrized_h1_mean(model: nn.Module, context: Tensor) -> Tensor:
    """Four-phase truth-free H1 average from the frozen protocol."""

    if context.ndim != 6 or context.shape[1] < 1 or context.shape[2:] != (
        5,
        *B5_SPATIAL_SHAPE,
    ):
        raise ValueError("H1 symmetrization context shape differs")
    latest = context[:, -1:]
    predictions: list[Tensor] = []
    for shift in range(4):
        rolled = torch.roll(latest, shifts=shift, dims=-1)
        predicted = model(rolled)
        if predicted.shape != latest[:, 0].shape:
            raise RuntimeError("H1 symmetrization prediction shape differs")
        predictions.append(torch.roll(predicted, shifts=-shift, dims=-1))
    return torch.stack(predictions, dim=0).mean(dim=0)
