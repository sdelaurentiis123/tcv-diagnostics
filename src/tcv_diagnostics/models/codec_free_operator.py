"""Codec-free mixed-boundary 3D state-derivative operator.

The model is the controlled deterministic baseline frozen in the post-ECRD
protocol.  It has no latent codec, never downsamples the toroidal axis, and
does not implement a physics-derived training loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import make_conv


STATE_CHANNELS = {"c5p": 5, "e6b": 6}


def _group_count(channels: int, maximum: int) -> int:
    for groups in range(min(channels, maximum), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("every positive channel count has group count one")


def _mixed_conv(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int | Sequence[int] = 1,
) -> nn.Module:
    return make_conv(
        in_channels,
        out_channels,
        spatial=3,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        padding_mode=("zeros", "zeros", "circular"),
    )


def xy_upsample(inputs: Tensor, size: Sequence[int]) -> Tensor:
    """Bilinearly resize x/y independently for every toroidal plane."""

    if inputs.ndim != 5:
        raise ValueError("x/y upsampling expects [batch,channel,x,y,z]")
    target = tuple(int(value) for value in size)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("upsampling target must have three positive sizes")
    batch, channels, n_x, n_y, n_z = inputs.shape
    if target[2] != n_z:
        raise ValueError("codec-free operator cannot resize the toroidal axis")
    planes = inputs.permute(0, 4, 1, 2, 3).reshape(
        batch * n_z, channels, n_x, n_y
    )
    planes = F.interpolate(
        planes,
        size=target[:2],
        mode="bilinear",
        align_corners=False,
    )
    return planes.reshape(
        batch, n_z, channels, target[0], target[1]
    ).permute(0, 2, 3, 4, 1).contiguous()


def spatialize_boundary(boundary: Tensor, *, n_x: int, n_z: int) -> Tensor:
    """Place each Bphi history profile and an explicit mask at its radial side."""

    if boundary.ndim != 4 or boundary.shape[2] != 2:
        raise ValueError("boundary must be [batch,history,side=2,y]")
    if n_x < 2 or n_z <= 0:
        raise ValueError("boundary spatialization needs positive nondegenerate sizes")
    batch, history, _, n_y = boundary.shape
    dtype, device = boundary.dtype, boundary.device
    inner_x = torch.zeros(n_x, dtype=dtype, device=device)
    outer_x = torch.zeros_like(inner_x)
    inner_x[0] = 1.0
    outer_x[-1] = 1.0
    inner_mask = inner_x.reshape(1, 1, n_x, 1, 1).expand(
        batch, history, n_x, n_y, n_z
    )
    outer_mask = outer_x.reshape(1, 1, n_x, 1, 1).expand_as(inner_mask)
    inner = boundary[:, :, 0].reshape(batch, history, 1, n_y, 1)
    outer = boundary[:, :, 1].reshape(batch, history, 1, n_y, 1)
    return torch.cat(
        (
            inner * inner_mask,
            outer * outer_mask,
            inner_mask,
            outer_mask,
        ),
        dim=1,
    )


def normalized_xy_coordinates(reference: Tensor) -> Tensor:
    """Return fixed nonperiodic coordinates without an absolute z coordinate."""

    if reference.ndim != 5:
        raise ValueError("coordinate reference must be five-dimensional")
    batch, _, n_x, n_y, n_z = reference.shape
    x = torch.linspace(-1.0, 1.0, n_x, dtype=reference.dtype, device=reference.device)
    y = torch.linspace(-1.0, 1.0, n_y, dtype=reference.dtype, device=reference.device)
    x = x.reshape(1, 1, n_x, 1, 1).expand(batch, 1, n_x, n_y, n_z)
    y = y.reshape(1, 1, 1, n_y, 1).expand(batch, 1, n_x, n_y, n_z)
    return torch.cat((x, y), dim=1)


@dataclass(frozen=True)
class CodecFreeOperatorConfig:
    state_family: str = "e6b"
    history_frames: int = 1
    base_channels: int = 24
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    blocks_per_level: int = 2
    lead_embedding_channels: int = 128
    group_norm_maximum_groups: int = 8
    kernel_size: int = 3
    predict_boundary: bool = True

    def __post_init__(self) -> None:
        if self.state_family not in STATE_CHANNELS:
            raise ValueError(f"unsupported state family {self.state_family!r}")
        if self.history_frames not in (1, 2):
            raise ValueError("operator history must contain one or two frames")
        values = (
            self.base_channels,
            self.blocks_per_level,
            self.lead_embedding_channels,
            self.group_norm_maximum_groups,
            self.kernel_size,
        )
        if min(values) <= 0:
            raise ValueError("operator dimensions must be positive")
        if self.lead_embedding_channels % 2:
            raise ValueError("lead embedding width must be even")
        if self.kernel_size % 2 != 1:
            raise ValueError("operator kernel size must be odd")
        if not self.channel_multipliers or any(
            int(value) <= 0 for value in self.channel_multipliers
        ):
            raise ValueError("channel multipliers must be positive")
        if self.predict_boundary != (self.state_family == "e6b"):
            raise ValueError("only E6B predicts the retained Bphi state")

    @property
    def volume_channels(self) -> int:
        return STATE_CHANNELS[self.state_family]

    @property
    def boundary_input_channels(self) -> int:
        return 4 * self.history_frames if self.predict_boundary else 0

    @property
    def input_channels(self) -> int:
        return (
            self.volume_channels * self.history_frames
            + self.boundary_input_channels
            + 2
        )

    @property
    def level_channels(self) -> tuple[int, ...]:
        return tuple(
            self.base_channels * int(multiplier)
            for multiplier in self.channel_multipliers
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["channel_multipliers"] = list(self.channel_multipliers)
        record.update(
            {
                "family": "codec_free_mixed_boundary_3d_increment_operator",
                "volume_channels": self.volume_channels,
                "input_channels": self.input_channels,
                "prediction": "standardized_state_derivative",
                "padding_xyz": ["zeros", "zeros", "circular"],
                "downsample_stride_xyz": [2, 2, 1],
                "absolute_z_coordinate": False,
                "physics_derived_loss": False,
                "latent_codec": False,
            }
        )
        return record


class _LeadEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half = channels // 2
        frequencies = torch.exp(
            torch.linspace(0.0, -math.log(10_000.0), half)
        )
        self.register_buffer("frequencies", frequencies)
        self.network = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.channels = int(channels)

    def forward(self, lead_steps: Tensor) -> Tensor:
        values = torch.as_tensor(
            lead_steps,
            dtype=self.frequencies.dtype,
            device=self.frequencies.device,
        ).flatten()
        if values.numel() == 0 or not torch.all(torch.isfinite(values)):
            raise ValueError("lead steps must be finite and nonempty")
        if not torch.all(values > 0):
            raise ValueError("lead steps must be positive")
        angles = torch.log(values).unsqueeze(-1) * self.frequencies.unsqueeze(0)
        return self.network(torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1))


class _ModulatedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        embedding_channels: int,
        kernel_size: int,
        maximum_groups: int,
    ) -> None:
        super().__init__()
        groups = _group_count(channels, maximum_groups)
        self.normalization1 = nn.GroupNorm(groups, channels)
        self.convolution1 = _mixed_conv(
            channels, channels, kernel_size=kernel_size
        )
        self.normalization2 = nn.GroupNorm(groups, channels)
        self.modulation = nn.Linear(embedding_channels, 2 * channels)
        self.convolution2 = _mixed_conv(
            channels, channels, kernel_size=kernel_size
        )
        self.channels = int(channels)

    def forward(self, inputs: Tensor, embedding: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.channels:
            raise ValueError("residual block input shape differs")
        if embedding.ndim != 2 or embedding.shape[0] != inputs.shape[0]:
            raise ValueError("residual block lead embedding shape differs")
        hidden = self.convolution1(F.silu(self.normalization1(inputs)))
        scale, shift = self.modulation(embedding).chunk(2, dim=-1)
        hidden = self.normalization2(hidden)
        hidden = hidden * (1.0 + scale[..., None, None, None])
        hidden = hidden + shift[..., None, None, None]
        hidden = self.convolution2(F.silu(hidden))
        return (inputs + hidden) * math.sqrt(0.5)


class _BoundaryDerivativeHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels + 1, channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(channels, 1, kernel_size=3, padding=1),
            )
            for _ in range(2)
        )
        self.channels = int(channels)

    def forward(self, features: Tensor, current_boundary: Tensor) -> Tensor:
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError("boundary head features differ")
        if current_boundary.ndim != 3 or current_boundary.shape[1] != 2:
            raise ValueError("current boundary must be [batch,side=2,y]")
        if current_boundary.shape[0] != features.shape[0]:
            raise ValueError("boundary batch differs")
        if current_boundary.shape[-1] != features.shape[-2]:
            raise ValueError("boundary y size differs")
        edges = (features[:, :, 0].mean(-1), features[:, :, -1].mean(-1))
        outputs = []
        for side, (edge, head) in enumerate(zip(edges, self.heads)):
            inputs = torch.cat((edge, current_boundary[:, side : side + 1]), dim=1)
            outputs.append(head(inputs).squeeze(1))
        return torch.stack(outputs, dim=1)


class StateDerivativePrediction(NamedTuple):
    volume: Tensor
    boundary: Tensor | None


class StateForecast(NamedTuple):
    volume: Tensor
    boundary: Tensor | None


def normalized_error_metrics(candidate: Tensor, reference: Tensor) -> dict[str, float]:
    """Return unit-clamped maximum and RMS discrepancies in float64."""

    if candidate.shape != reference.shape or candidate.numel() == 0:
        raise ValueError("normalized error inputs must have the same nonempty shape")
    if not torch.isfinite(candidate).all() or not torch.isfinite(reference).all():
        raise ValueError("normalized error inputs must be finite")
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    reference64 = reference.to(torch.float64)
    maximum_scale = torch.maximum(
        torch.max(torch.abs(reference64)),
        torch.ones((), dtype=torch.float64, device=reference.device),
    )
    rms_scale = torch.maximum(
        torch.sqrt(torch.mean(reference64 * reference64)),
        torch.ones((), dtype=torch.float64, device=reference.device),
    )
    return {
        "normalized_maximum_absolute_error": float(
            (torch.max(torch.abs(difference)) / maximum_scale).cpu()
        ),
        "normalized_root_mean_square_error": float(
            (torch.sqrt(torch.mean(difference * difference)) / rms_scale).cpu()
        ),
    }


class CodecFreeIncrementOperator3D(nn.Module):
    """U-shaped full-field operator with x/y-only multiresolution processing."""

    def __init__(self, config: CodecFreeOperatorConfig) -> None:
        super().__init__()
        self.config = config
        levels = config.level_channels
        self.lead_embedding = _LeadEmbedding(config.lead_embedding_channels)
        self.input_projection = _mixed_conv(
            config.input_channels,
            levels[0],
            kernel_size=config.kernel_size,
        )

        encoder_blocks: list[nn.ModuleList] = []
        downsample: list[nn.Module] = []
        for level, channels in enumerate(levels):
            encoder_blocks.append(
                nn.ModuleList(
                    _ModulatedResidualBlock(
                        channels,
                        embedding_channels=config.lead_embedding_channels,
                        kernel_size=config.kernel_size,
                        maximum_groups=config.group_norm_maximum_groups,
                    )
                    for _ in range(config.blocks_per_level)
                )
            )
            if level < len(levels) - 1:
                downsample.append(
                    _mixed_conv(
                        channels,
                        levels[level + 1],
                        kernel_size=config.kernel_size,
                        stride=(2, 2, 1),
                    )
                )
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.downsamples = nn.ModuleList(downsample)

        merges: list[nn.Module] = []
        decoder_blocks: list[nn.ModuleList] = []
        for level in range(len(levels) - 2, -1, -1):
            channels = levels[level]
            merges.append(
                _mixed_conv(
                    levels[level + 1] + channels,
                    channels,
                    kernel_size=config.kernel_size,
                )
            )
            decoder_blocks.append(
                nn.ModuleList(
                    _ModulatedResidualBlock(
                        channels,
                        embedding_channels=config.lead_embedding_channels,
                        kernel_size=config.kernel_size,
                        maximum_groups=config.group_norm_maximum_groups,
                    )
                    for _ in range(config.blocks_per_level)
                )
            )
        self.merges = nn.ModuleList(merges)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.output_normalization = nn.GroupNorm(
            _group_count(levels[0], config.group_norm_maximum_groups), levels[0]
        )
        self.output_projection = _mixed_conv(
            levels[0],
            config.volume_channels,
            kernel_size=config.kernel_size,
        )
        self.boundary_head = (
            _BoundaryDerivativeHead(levels[0]) if config.predict_boundary else None
        )

    def _condition(
        self,
        context: Tensor,
        context_boundary: Tensor | None,
    ) -> Tensor:
        if context.ndim != 6:
            raise ValueError("context must be [batch,history,channel,x,y,z]")
        batch, history, channels, n_x, n_y, n_z = context.shape
        if history != self.config.history_frames:
            raise ValueError("context history differs from configuration")
        if channels != self.config.volume_channels:
            raise ValueError("context field count differs from configuration")
        flattened = context.reshape(batch, history * channels, n_x, n_y, n_z)
        pieces = [flattened]
        if self.config.predict_boundary:
            if context_boundary is None:
                raise ValueError("E6B operator requires Bphi history")
            expected = (batch, history, 2, n_y)
            if tuple(context_boundary.shape) != expected:
                raise ValueError(f"Bphi history shape must be {expected}")
            pieces.append(spatialize_boundary(context_boundary, n_x=n_x, n_z=n_z))
        elif context_boundary is not None:
            raise ValueError("C5P operator does not accept Bphi history")
        pieces.append(normalized_xy_coordinates(flattened))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        context: Tensor,
        lead_steps: Tensor,
        context_boundary: Tensor | None = None,
    ) -> StateDerivativePrediction:
        condition = self._condition(context, context_boundary)
        embedding = self.lead_embedding(lead_steps).to(
            device=context.device, dtype=context.dtype
        )
        if embedding.shape[0] != context.shape[0]:
            raise ValueError("lead count differs from context batch")
        hidden = self.input_projection(condition)
        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                hidden = block(hidden, embedding)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        for offset, (merge, blocks) in enumerate(
            zip(self.merges, self.decoder_blocks)
        ):
            skip = skips[-2 - offset]
            hidden = xy_upsample(hidden, skip.shape[-3:])
            hidden = merge(torch.cat((hidden, skip), dim=1))
            for block in blocks:
                hidden = block(hidden, embedding)

        volume = self.output_projection(F.silu(self.output_normalization(hidden)))
        boundary = None
        if self.boundary_head is not None:
            if context_boundary is None:
                raise AssertionError("E6B boundary vanished after validation")
            boundary = self.boundary_head(hidden, context_boundary[:, -1])
        return StateDerivativePrediction(volume=volume, boundary=boundary)

    def forecast(
        self,
        context: Tensor,
        lead_steps: Tensor,
        context_boundary: Tensor | None = None,
    ) -> StateForecast:
        derivative = self(context, lead_steps, context_boundary)
        lead = torch.as_tensor(
            lead_steps, dtype=context.dtype, device=context.device
        ).reshape(-1, 1, 1, 1, 1)
        volume = context[:, -1] + lead * derivative.volume
        boundary = None
        if derivative.boundary is not None:
            if context_boundary is None:
                raise AssertionError("E6B boundary vanished during forecast")
            boundary_lead = torch.as_tensor(
                lead_steps, dtype=context.dtype, device=context.device
            ).reshape(-1, 1, 1)
            boundary = context_boundary[:, -1] + boundary_lead * derivative.boundary
        return StateForecast(volume=volume, boundary=boundary)

    def to_record(self) -> dict[str, Any]:
        return {
            "architecture": self.config.to_record(),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "physics_derived_loss_used": False,
        }


def state_derivative_loss(
    prediction: StateDerivativePrediction,
    target_volume_derivative: Tensor,
    target_boundary_derivative: Tensor | None = None,
) -> Tensor:
    """Equal-element state MSE; no derived physics quantity enters this loss."""

    if prediction.volume.shape != target_volume_derivative.shape:
        raise ValueError("volume derivative target shape differs")
    loss = F.mse_loss(prediction.volume, target_volume_derivative)
    if prediction.boundary is None:
        if target_boundary_derivative is not None:
            raise ValueError("boundary target supplied to a boundary-free model")
        return loss
    if target_boundary_derivative is None:
        raise ValueError("E6B boundary derivative target is required")
    if prediction.boundary.shape != target_boundary_derivative.shape:
        raise ValueError("boundary derivative target shape differs")
    return loss + F.mse_loss(prediction.boundary, target_boundary_derivative)


def component_balanced_state_derivative_loss(
    prediction: StateDerivativePrediction,
    target_volume_derivative: Tensor,
    target_boundary_derivative: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Average standardized MSE equally over fields and boundary sides.

    This is a data-only loss.  It avoids giving the two retained boundary
    profiles either negligible element-count weight or the one-half total
    weight implied by summing two separately averaged tensors.
    """

    if prediction.volume.shape != target_volume_derivative.shape:
        raise ValueError("volume derivative target shape differs")
    if prediction.volume.ndim != 5:
        raise ValueError("volume derivatives must be [batch,channel,x,y,z]")
    volume_by_field = torch.mean(
        (prediction.volume - target_volume_derivative) ** 2,
        dim=(0, 2, 3, 4),
    )
    component_losses = [value for value in volume_by_field]
    records: dict[str, Tensor] = {
        "volume_mean": torch.mean(volume_by_field),
    }
    if prediction.boundary is None:
        if target_boundary_derivative is not None:
            raise ValueError("boundary target supplied to a boundary-free model")
    else:
        if target_boundary_derivative is None:
            raise ValueError("E6B boundary derivative target is required")
        if prediction.boundary.shape != target_boundary_derivative.shape:
            raise ValueError("boundary derivative target shape differs")
        if prediction.boundary.ndim != 3 or prediction.boundary.shape[1] != 2:
            raise ValueError("boundary derivatives must be [batch,side=2,y]")
        boundary_by_side = torch.mean(
            (prediction.boundary - target_boundary_derivative) ** 2,
            dim=(0, 2),
        )
        component_losses.extend(value for value in boundary_by_side)
        records["boundary_mean"] = torch.mean(boundary_by_side)
    loss = torch.mean(torch.stack(component_losses))
    records["total"] = loss
    return loss, records
