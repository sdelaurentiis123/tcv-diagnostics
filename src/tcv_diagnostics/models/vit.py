"""Dependency-light LOLA-style Vision Transformer blocks.

This is a minimal PyTorch port of ``lola/nn/{vit,attention,embedding}.py`` at
LOLA commit ``21a4354b327e6e5ee06da5075ba3bd1dd88c61f1``.  The upstream
license is retained in ``LOLA_LICENSE.txt``.  It intentionally omits xFormers,
local sparse attention, external rearrangement helpers, and stochastic
extensions that are not part of the frozen deterministic O2 experiment.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import PatchifyND, UnpatchifyND


class SineEncoding(nn.Module):
    """Static sinusoidal encoding used by the attributed LOLA ViT."""

    def __init__(self, features: int, omega: float = 1.0e3) -> None:
        super().__init__()
        if features <= 0 or features % 2:
            raise ValueError("sine-encoding features must be positive and even")
        frequencies = torch.linspace(0.0, 1.0, features // 2)
        frequencies = float(omega) ** (-frequencies)
        self.register_buffer("frequencies", frequencies.to(torch.float32))

    def forward(self, coordinates: Tensor) -> Tensor:
        angles = coordinates.unsqueeze(-1) * self.frequencies.to(
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


def apply_rope(query: Tensor, key: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
    """Apply pairwise rotary angles to query and key head features."""

    if query.shape != key.shape:
        raise ValueError("query and key shapes differ")
    if query.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    expected = query.shape[:-1] + (query.shape[-1] // 2,)
    try:
        broadcast = torch.broadcast_shapes(tuple(theta.shape), tuple(expected))
    except RuntimeError as error:
        raise ValueError(
            f"RoPE angles {theta.shape} do not broadcast to {expected}"
        ) from error
    if tuple(broadcast) != tuple(expected):
        raise ValueError(f"RoPE angles {theta.shape} do not broadcast to {expected}")

    query_pairs = query.unflatten(-1, (-1, 2))
    key_pairs = key.unflatten(-1, (-1, 2))
    cosine = torch.cos(theta)
    sine = torch.sin(theta)

    def rotate(values: Tensor) -> Tensor:
        real, imaginary = values[..., 0], values[..., 1]
        return torch.stack(
            (
                real * cosine - imaginary * sine,
                real * sine + imaginary * cosine,
            ),
            dim=-1,
        ).flatten(-2)

    return rotate(query_pairs), rotate(key_pairs)


class MultiheadSelfAttention(nn.Module):
    """LOLA-compatible global self-attention with per-head Q/K RMSNorm."""

    def __init__(
        self,
        channels: int,
        *,
        attention_heads: int = 1,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or attention_heads <= 0 or channels % attention_heads:
            raise ValueError("channels must be divisible by a positive head count")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("attention dropout must lie in [0,1)")
        self.qkv_projection = nn.Linear(channels, 3 * channels, bias=False)
        self.output_projection = nn.Linear(channels, channels)
        head_channels = channels // attention_heads
        self.qk_normalization = (
            nn.RMSNorm(head_channels, elementwise_affine=False, eps=1.0e-5)
            if qk_norm
            else nn.Identity()
        )
        self.channels = int(channels)
        self.heads = int(attention_heads)
        self.head_channels = int(head_channels)
        self.dropout = float(dropout)

    def forward(self, tokens: Tensor, theta: Tensor | None = None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.channels:
            raise ValueError(
                f"attention expects [batch,tokens,{self.channels}], got {tokens.shape}"
            )
        batch, length, _ = tokens.shape
        qkv = self.qkv_projection(tokens).reshape(
            batch,
            length,
            3,
            self.heads,
            self.head_channels,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = self.qk_normalization(query)
        key = self.qk_normalization(key)
        if theta is not None:
            expected = (length, self.channels // 2)
            if theta.shape != expected:
                raise ValueError(f"attention RoPE shape {theta.shape} != {expected}")
            angles = theta.reshape(length, self.heads, self.head_channels // 2)
            angles = angles.permute(1, 0, 2).unsqueeze(0)
            query, key = apply_rope(query, key, angles)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch,
            length,
            self.channels,
        )
        return self.output_projection(attended)


class ViTBlock(nn.Module):
    """One attributed LOLA AdaLN-zero-style deterministic transformer block."""

    def __init__(
        self,
        channels: int,
        *,
        ffn_factor: int = 4,
        spatial: int = 2,
        attention_heads: int = 1,
        qk_norm: bool = True,
        rope: bool = True,
        dropout: float = 0.0,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 2:
            raise ValueError("ViT channels must be positive and even")
        if ffn_factor <= 0 or spatial <= 0:
            raise ValueError("FFN factor and spatial rank must be positive")
        self.checkpointing = bool(checkpointing)
        self.normalization = nn.LayerNorm(channels, elementwise_affine=False)
        self.ada_zero = nn.Parameter(torch.randn(4, channels) * 1.0e-2)
        self.attention = MultiheadSelfAttention(
            channels,
            attention_heads=attention_heads,
            qk_norm=qk_norm,
        )
        if rope:
            amplitude = 1.0e2 ** -torch.rand(channels // 2)
            direction = F.normalize(torch.randn(spatial, channels // 2), dim=0)
            self.theta = nn.Parameter(amplitude * direction)
        else:
            self.theta = None
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_factor * channels),
            nn.SiLU(),
            nn.Identity() if dropout == 0.0 else nn.Dropout(dropout),
            nn.Linear(ffn_factor * channels, channels),
        )

    def _forward(self, tokens: Tensor, coordinates: Tensor, skip: Tensor) -> Tensor:
        theta = (
            None
            if self.theta is None
            else torch.einsum("ld,dc->lc", coordinates, self.theta)
        )
        scale, shift, residual_scale, skip_scale = self.ada_zero
        hidden = (scale + 1.0) * self.normalization(tokens) + shift
        hidden = hidden + self.attention(hidden, theta)
        hidden = self.ffn(hidden)
        hidden = (tokens + residual_scale * hidden) * torch.rsqrt(
            1.0 + residual_scale * residual_scale
        )
        return (hidden + skip_scale * skip) * torch.rsqrt(
            1.0 + skip_scale * skip_scale
        )

    def forward(self, tokens: Tensor, coordinates: Tensor, skip: Tensor) -> Tensor:
        if self.checkpointing and self.training:
            return checkpoint(
                self._forward,
                tokens,
                coordinates,
                skip,
                use_reentrant=False,
            )
        return self._forward(tokens, coordinates, skip)


def regular_grid_coordinates(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Coordinates in the exact flattening order of channels-first grids."""

    dimensions = tuple(int(item) for item in shape)
    if not dimensions or any(item <= 0 for item in dimensions):
        raise ValueError(f"invalid token grid {dimensions}")
    axes = [torch.arange(size, device=device) for size in dimensions]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, len(dimensions)).to(dtype=dtype)


class ViT(nn.Module):
    """LOLA-style N-D ViT without stochastic or sparse-attention extensions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_channels: int = 0,
        hidden_channels: int = 1024,
        hidden_blocks: int = 3,
        attention_heads: int = 1,
        ffn_factor: int = 4,
        spatial: int = 2,
        patch_size: int | Sequence[int] = 1,
        qk_norm: bool = True,
        rope: bool = True,
        dropout: float = 0.0,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch = (int(patch_size),) * int(spatial)
        else:
            patch = tuple(int(item) for item in patch_size)
        if len(patch) != spatial or any(item <= 0 for item in patch):
            raise ValueError(f"invalid {spatial}-D patch {patch}")
        if min(in_channels, out_channels, hidden_channels, hidden_blocks) <= 0:
            raise ValueError("ViT channel and block counts must be positive")
        if condition_channels < 0:
            raise ValueError("condition channels cannot be negative")

        patch_volume = math.prod(patch)
        self.patchify = PatchifyND(patch)
        self.unpatchify = UnpatchifyND(patch)
        self.input_projection = nn.Linear(
            patch_volume * (in_channels + condition_channels),
            hidden_channels,
        )
        self.output_projection = nn.Linear(
            hidden_channels,
            patch_volume * out_channels,
        )
        self.position_encoding = SineEncoding(hidden_channels)
        self.position_projection = nn.Linear(
            spatial * hidden_channels,
            hidden_channels,
        )
        self.blocks = nn.ModuleList(
            ViTBlock(
                hidden_channels,
                ffn_factor=ffn_factor,
                spatial=spatial,
                attention_heads=attention_heads,
                qk_norm=qk_norm,
                rope=rope,
                dropout=dropout,
                checkpointing=checkpointing,
            )
            for _ in range(hidden_blocks)
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.condition_channels = int(condition_channels)
        self.hidden_channels = int(hidden_channels)
        self.spatial = int(spatial)
        self.patch_size = patch

    def forward(self, inputs: Tensor, condition: Tensor | None = None) -> Tensor:
        if inputs.ndim != self.spatial + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"ViT expects [batch,{self.in_channels},{self.spatial} axes], "
                f"got {inputs.shape}"
            )
        if self.condition_channels:
            expected = (inputs.shape[0], self.condition_channels, *inputs.shape[2:])
            if condition is None or condition.shape != expected:
                raise ValueError(f"condition shape must be {expected}")
            combined = torch.cat((inputs, condition), dim=1)
        else:
            if condition is not None:
                raise ValueError("condition supplied to an unconditional ViT")
            combined = inputs

        patches = self.patchify(combined)
        token_grid = tuple(int(item) for item in patches.shape[2:])
        tokens = patches.movedim(1, -1)
        tokens = self.input_projection(tokens).flatten(1, -2)
        coordinates = regular_grid_coordinates(
            token_grid,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        position = self.position_encoding(coordinates).flatten(start_dim=1)
        skip = tokens
        tokens = tokens + self.position_projection(position)
        for block in self.blocks:
            tokens = block(tokens, coordinates, skip)
        tokens = tokens.reshape(inputs.shape[0], *token_grid, self.hidden_channels)
        patches = self.output_projection(tokens).movedim(-1, 1)
        return self.unpatchify(patches)
