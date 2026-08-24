"""Mixed-boundary full-resolution axial operator transformer.

This is a controlled nonlocal comparator for the codec-free Paper 0 state
operator.  It is implemented locally from standard axial-attention building
blocks; it is not a copy of, or a claim to reproduce, the public GAOT code.
The model retains all toroidal cells, uses no absolute toroidal coordinate,
and predicts direct standardized state derivatives only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .codec_free_operator import (
    STATE_CHANNELS,
    StateDerivativePrediction,
    StateForecast,
    _BoundaryDerivativeHead,
    _LeadEmbedding,
    _group_count,
    _mixed_conv,
    normalized_xy_coordinates,
    spatialize_boundary,
)


def _axis_to_sequences(
    inputs: Tensor,
    *,
    spatial_axis: int,
) -> tuple[Tensor, tuple[int, ...]]:
    """Flatten all but one spatial axis into an attention batch."""

    if inputs.ndim != 5:
        raise ValueError("axial attention expects [batch,channel,x,y,z]")
    if spatial_axis not in (1, 2, 3):
        raise ValueError("channel-last spatial axis must be x=1, y=2, or z=3")
    channel_last = inputs.movedim(1, -1)
    moved = channel_last.movedim(spatial_axis, -2).contiguous()
    shape = tuple(moved.shape)
    return moved.reshape(-1, shape[-2], shape[-1]), shape


def _sequences_to_axis(
    sequences: Tensor,
    *,
    moved_shape: tuple[int, ...],
    spatial_axis: int,
) -> Tensor:
    if tuple(sequences.shape[-2:]) != tuple(moved_shape[-2:]):
        raise ValueError("attention sequence shape differs from saved axis shape")
    moved = sequences.reshape(moved_shape)
    channel_last = moved.movedim(-2, spatial_axis)
    return channel_last.movedim(-1, 1).contiguous()


class _AxialAttention(nn.Module):
    """Content-dependent global mixing along x, y, and z independently."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.normalizations = nn.ModuleList(
            nn.LayerNorm(channels) for _ in range(3)
        )
        self.attentions = nn.ModuleList(
            nn.MultiheadAttention(
                channels,
                heads,
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(3)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for spatial_axis, normalization, attention in zip(
            (1, 2, 3),
            self.normalizations,
            self.attentions,
        ):
            sequences, moved_shape = _axis_to_sequences(
                hidden,
                spatial_axis=spatial_axis,
            )
            normalized = normalization(sequences)
            update, _ = attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            update = _sequences_to_axis(
                update,
                moved_shape=moved_shape,
                spatial_axis=spatial_axis,
            )
            hidden = (hidden + update) * math.sqrt(0.5)
        return hidden


class _AxialOperatorBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        lead_embedding_channels: int,
        expansion: int,
        kernel_size: int,
        maximum_groups: int,
    ) -> None:
        super().__init__()
        groups = _group_count(channels, maximum_groups)
        self.local_normalization = nn.GroupNorm(groups, channels)
        self.local_modulation = nn.Linear(
            lead_embedding_channels,
            2 * channels,
        )
        self.local_convolution = _mixed_conv(
            channels,
            channels,
            kernel_size=kernel_size,
        )
        self.axial_attention = _AxialAttention(channels, heads)
        self.feedforward_normalization = nn.GroupNorm(groups, channels)
        self.feedforward = nn.Sequential(
            nn.Conv3d(channels, expansion * channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(expansion * channels, channels, kernel_size=1),
        )

    def forward(self, inputs: Tensor, lead_embedding: Tensor) -> Tensor:
        normalized = self.local_normalization(inputs)
        scale, shift = self.local_modulation(lead_embedding).chunk(2, dim=-1)
        normalized = normalized * (1.0 + scale[..., None, None, None])
        normalized = normalized + shift[..., None, None, None]
        local = self.local_convolution(F.silu(normalized))
        hidden = (inputs + local) * math.sqrt(0.5)
        hidden = self.axial_attention(hidden)
        feedforward = self.feedforward(
            F.silu(self.feedforward_normalization(hidden))
        )
        return (hidden + feedforward) * math.sqrt(0.5)


@dataclass(frozen=True)
class AxialOperatorConfig:
    state_family: str = "e6b"
    history_frames: int = 1
    auxiliary_context_channels: int = 0
    static_context_channels: int = 0
    width: int = 104
    blocks: int = 4
    attention_heads: int = 4
    feedforward_expansion: int = 2
    lead_embedding_channels: int = 128
    group_norm_maximum_groups: int = 8
    kernel_size: int = 3
    predict_boundary: bool = True
    zero_initialize_output: bool = True

    def __post_init__(self) -> None:
        if self.state_family not in STATE_CHANNELS:
            raise ValueError(f"unsupported state family {self.state_family!r}")
        if self.history_frames not in (1, 2):
            raise ValueError("axial operator history must contain one or two frames")
        dimensions = (
            self.width,
            self.blocks,
            self.attention_heads,
            self.feedforward_expansion,
            self.lead_embedding_channels,
            self.group_norm_maximum_groups,
            self.kernel_size,
        )
        if min(dimensions) <= 0:
            raise ValueError("axial operator dimensions must be positive")
        if self.width % self.attention_heads:
            raise ValueError("operator width must be divisible by attention heads")
        if self.lead_embedding_channels % 2:
            raise ValueError("lead embedding width must be even")
        if self.kernel_size % 2 != 1:
            raise ValueError("operator kernel size must be odd")
        if self.auxiliary_context_channels < 0 or self.static_context_channels < 0:
            raise ValueError("context channel counts must be nonnegative")
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
            (self.volume_channels + self.auxiliary_context_channels)
            * self.history_frames
            + self.static_context_channels
            + self.boundary_input_channels
            + 2
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "family": "mixed_boundary_full_resolution_axial_operator_transformer",
                "volume_channels": self.volume_channels,
                "input_channels": self.input_channels,
                "prediction": "standardized_state_derivative",
                "padding_xyz": ["zeros", "zeros", "circular"],
                "toroidal_downsampling": False,
                "absolute_z_coordinate": False,
                "global_mixing_axes": ["x", "y", "z"],
                "physics_derived_loss": False,
                "latent_codec": False,
                "official_GAOT_reproduction": False,
            }
        )
        return record


class AxialIncrementOperator3D(nn.Module):
    """Codec-free state-derivative model with full-domain axial mixing."""

    def __init__(self, config: AxialOperatorConfig) -> None:
        super().__init__()
        self.config = config
        self.lead_embedding = _LeadEmbedding(config.lead_embedding_channels)
        self.input_projection = _mixed_conv(
            config.input_channels,
            config.width,
            kernel_size=config.kernel_size,
        )
        self.blocks = nn.ModuleList(
            _AxialOperatorBlock(
                config.width,
                heads=config.attention_heads,
                lead_embedding_channels=config.lead_embedding_channels,
                expansion=config.feedforward_expansion,
                kernel_size=config.kernel_size,
                maximum_groups=config.group_norm_maximum_groups,
            )
            for _ in range(config.blocks)
        )
        self.output_normalization = nn.GroupNorm(
            _group_count(config.width, config.group_norm_maximum_groups),
            config.width,
        )
        self.output_projection = _mixed_conv(
            config.width,
            config.volume_channels,
            kernel_size=config.kernel_size,
        )
        self.boundary_head = (
            _BoundaryDerivativeHead(config.width)
            if config.predict_boundary
            else None
        )
        if config.zero_initialize_output:
            nn.init.zeros_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)
            if self.boundary_head is not None:
                for head in self.boundary_head.heads:
                    final = head[-1]
                    if not isinstance(final, nn.Conv1d):
                        raise AssertionError("boundary output layer differs")
                    nn.init.zeros_(final.weight)
                    if final.bias is not None:
                        nn.init.zeros_(final.bias)

    def _condition(
        self,
        context: Tensor,
        context_boundary: Tensor | None,
        auxiliary_context: Tensor | None,
        static_context: Tensor | None,
    ) -> Tensor:
        if context.ndim != 6:
            raise ValueError("context must be [batch,history,channel,x,y,z]")
        batch, history, channels, n_x, n_y, n_z = context.shape
        if history != self.config.history_frames:
            raise ValueError("context history differs from configuration")
        if channels != self.config.volume_channels:
            raise ValueError("context field count differs from configuration")
        pieces = [context.reshape(batch, history * channels, n_x, n_y, n_z)]

        auxiliary_channels = self.config.auxiliary_context_channels
        if auxiliary_channels:
            expected = (batch, history, auxiliary_channels, n_x, n_y, n_z)
            if auxiliary_context is None or tuple(auxiliary_context.shape) != expected:
                raise ValueError(f"auxiliary context shape must be {expected}")
            pieces.append(
                auxiliary_context.reshape(
                    batch,
                    history * auxiliary_channels,
                    n_x,
                    n_y,
                    n_z,
                )
            )
        elif auxiliary_context is not None:
            raise ValueError("operator is not configured for auxiliary context")

        static_channels = self.config.static_context_channels
        if static_channels:
            expected = (batch, static_channels, n_x, n_y, n_z)
            if static_context is None or tuple(static_context.shape) != expected:
                raise ValueError(f"static context shape must be {expected}")
            pieces.append(static_context)
        elif static_context is not None:
            raise ValueError("operator is not configured for static context")

        if self.config.predict_boundary:
            expected = (batch, history, 2, n_y)
            if context_boundary is None or tuple(context_boundary.shape) != expected:
                raise ValueError(f"Bphi history shape must be {expected}")
            pieces.append(spatialize_boundary(context_boundary, n_x=n_x, n_z=n_z))
        elif context_boundary is not None:
            raise ValueError("C5P operator does not accept Bphi history")

        pieces.append(normalized_xy_coordinates(pieces[0]))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        context: Tensor,
        lead_steps: Tensor,
        context_boundary: Tensor | None = None,
        auxiliary_context: Tensor | None = None,
        static_context: Tensor | None = None,
    ) -> StateDerivativePrediction:
        condition = self._condition(
            context,
            context_boundary,
            auxiliary_context,
            static_context,
        )
        embedding = self.lead_embedding(lead_steps).to(
            device=context.device,
            dtype=context.dtype,
        )
        if embedding.shape[0] != context.shape[0]:
            raise ValueError("lead count differs from context batch")
        hidden = self.input_projection(condition)
        for block in self.blocks:
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
        auxiliary_context: Tensor | None = None,
        static_context: Tensor | None = None,
    ) -> StateForecast:
        derivative = self(
            context,
            lead_steps,
            context_boundary,
            auxiliary_context,
            static_context,
        )
        volume_lead = torch.as_tensor(
            lead_steps,
            dtype=context.dtype,
            device=context.device,
        ).reshape(-1, 1, 1, 1, 1)
        volume = context[:, -1] + volume_lead * derivative.volume
        boundary = None
        if derivative.boundary is not None:
            if context_boundary is None:
                raise AssertionError("E6B boundary vanished during forecast")
            boundary_lead = torch.as_tensor(
                lead_steps,
                dtype=context.dtype,
                device=context.device,
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
