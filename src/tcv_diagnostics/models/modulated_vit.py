"""Noise-time-modulated LOLA-style Vision Transformer blocks.

This is a dependency-light PyTorch port of the modulated path in
``lola/nn/vit.py`` at LOLA commit
``21a4354b327e6e5ee06da5075ba3bd1dd88c61f1``. The upstream MIT license is
retained in ``LOLA_LICENSE.txt``. It is separate from ``models.vit`` so the
completed deterministic O2 implementation and its state dictionaries remain
unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import PatchifyND, UnpatchifyND
from .vit import MultiheadSelfAttention, SineEncoding, regular_grid_coordinates


class ModulatedViTBlock(nn.Module):
    """One LOLA AdaLN-zero transformer block conditioned by a global vector."""

    def __init__(
        self,
        channels: int,
        *,
        modulation_features: int,
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
        if modulation_features <= 0:
            raise ValueError("modulation features must be positive")
        if ffn_factor <= 0 or spatial <= 0:
            raise ValueError("FFN factor and spatial rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

        self.checkpointing = bool(checkpointing)
        self.normalization = nn.LayerNorm(channels, elementwise_affine=False)
        self.ada_zero = nn.Sequential(
            nn.Linear(modulation_features, modulation_features),
            nn.SiLU(),
            nn.Linear(modulation_features, 4 * channels),
        )
        # Match LOLA: start the learned modulation branch close to zero while
        # retaining the framework's ordinary bias initialization.
        self.ada_zero[-1].weight.data.mul_(1.0e-2)
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
        self.channels = int(channels)
        self.modulation_features = int(modulation_features)

    def _forward(
        self,
        tokens: Tensor,
        modulation: Tensor,
        coordinates: Tensor,
        skip: Tensor,
    ) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.channels:
            raise ValueError("tokens have invalid shape")
        if modulation.shape != (tokens.shape[0], self.modulation_features):
            raise ValueError("modulation has invalid shape")
        theta = (
            None
            if self.theta is None
            else torch.einsum("ld,dc->lc", coordinates, self.theta)
        )
        parameters = self.ada_zero(modulation).reshape(
            tokens.shape[0], 4, 1, self.channels
        )
        scale, shift, residual_scale, skip_scale = parameters.unbind(dim=1)
        hidden = (scale + 1.0) * self.normalization(tokens) + shift
        hidden = hidden + self.attention(hidden, theta)
        hidden = self.ffn(hidden)
        hidden = (tokens + residual_scale * hidden) * torch.rsqrt(
            1.0 + residual_scale * residual_scale
        )
        return (hidden + skip_scale * skip) * torch.rsqrt(
            1.0 + skip_scale * skip_scale
        )

    def forward(
        self,
        tokens: Tensor,
        modulation: Tensor,
        coordinates: Tensor,
        skip: Tensor,
    ) -> Tensor:
        if self.checkpointing and self.training:
            return checkpoint(
                self._forward,
                tokens,
                modulation,
                coordinates,
                skip,
                use_reentrant=False,
            )
        return self._forward(tokens, modulation, coordinates, skip)


class ModulatedViT(nn.Module):
    """LOLA-style N-D ViT with one global conditioning vector per sample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        condition_channels: int,
        modulation_features: int,
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
        if min(
            in_channels,
            out_channels,
            modulation_features,
            hidden_channels,
            hidden_blocks,
        ) <= 0:
            raise ValueError("ViT channel, block, and modulation counts must be positive")
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
            ModulatedViTBlock(
                hidden_channels,
                modulation_features=modulation_features,
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
        self.modulation_features = int(modulation_features)
        self.hidden_channels = int(hidden_channels)
        self.spatial = int(spatial)
        self.patch_size = patch

    def forward(
        self,
        inputs: Tensor,
        modulation: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        if inputs.ndim != self.spatial + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"ViT expects [batch,{self.in_channels},{self.spatial} axes], "
                f"got {inputs.shape}"
            )
        if modulation.shape != (inputs.shape[0], self.modulation_features):
            raise ValueError(
                "modulation must be [batch,modulation_features], got "
                f"{modulation.shape}"
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
            tokens = block(tokens, modulation, coordinates, skip)
        tokens = tokens.reshape(inputs.shape[0], *token_grid, self.hidden_channels)
        patches = self.output_projection(tokens).movedim(-1, 1)
        return self.unpatchify(patches)


class NoiseTimeEmbedding(nn.Module):
    """LOLA sine/MLP embedding for scalar diffusion-noise time."""

    def __init__(self, features: int) -> None:
        super().__init__()
        if features <= 0 or features % 2:
            raise ValueError("noise-time features must be positive and even")
        self.features = int(features)
        self.network = nn.Sequential(
            SineEncoding(features),
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )

    def forward(self, noise_time: Tensor) -> Tensor:
        if noise_time.ndim != 1:
            raise ValueError("noise time must have shape [batch]")
        return self.network(noise_time)
