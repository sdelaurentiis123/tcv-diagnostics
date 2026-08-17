"""Deterministic deep-compression autoencoder for the matched O1 ladder.

This is a minimum dependency-free PyTorch port of the DCAE structure used by
LOLA at commit 21a4354b327e6e5ee06da5075ba3bd1dd88c61f1, plus the audited TCV
mixed-padding and anisotropic-stride behavior. See LOLA_LICENSE.txt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import (
    ChannelLayerNorm,
    PatchifyND,
    UnpatchifyND,
    make_conv,
)


def _kernel(
    value: int | Sequence[int],
    spatial: int,
) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (value,) * spatial
    else:
        result = tuple(int(item) for item in value)
    if len(result) != spatial or any(item <= 0 or item % 2 == 0 for item in result):
        raise ValueError(f"kernel must have {spatial} positive odd entries: {result}")
    return result


def normalize_strides(
    stride: int | Sequence[int] | Sequence[Sequence[int]],
    *,
    spatial: int,
    transitions: int,
) -> tuple[tuple[int, ...], ...]:
    """Normalize scalar, per-axis, or per-transition stride declarations."""

    if isinstance(stride, int):
        return ((stride,) * spatial,) * transitions
    values = tuple(stride)
    if len(values) == spatial and all(isinstance(item, int) for item in values):
        per_axis = tuple(int(item) for item in values)
        return (per_axis,) * transitions
    if len(values) != transitions:
        raise ValueError(
            f"expected {transitions} stride transitions, got {len(values)}"
        )
    result = tuple(tuple(int(item) for item in level) for level in values)
    if any(len(level) != spatial for level in result):
        raise ValueError(f"each stride needs {spatial} axes, got {result}")
    if any(item <= 0 for level in result for item in level):
        raise ValueError(f"stride entries must be positive, got {result}")
    return result


def padding_modes(
    periodic: bool | Sequence[bool],
    *,
    spatial: int,
    wall_mode: str,
) -> str | tuple[str, ...]:
    if isinstance(periodic, bool):
        return "circular" if periodic else wall_mode
    values = tuple(bool(item) for item in periodic)
    if len(values) != spatial:
        raise ValueError(f"periodic must contain {spatial} entries, got {values}")
    return tuple("circular" if item else wall_mode for item in values)


class ResidualBlock(nn.Module):
    """LOLA DCAE residual block without optional spatial attention."""

    def __init__(
        self,
        channels: int,
        *,
        spatial: int,
        kernel_size: Sequence[int],
        padding_mode: str | Sequence[str],
        ffn_factor: int = 1,
        dropout: float | None = 0.05,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.checkpointing = checkpointing
        padding = tuple(item // 2 for item in kernel_size)
        self.norm = ChannelLayerNorm()
        self.ffn = nn.Sequential(
            make_conv(
                channels,
                ffn_factor * channels,
                spatial=spatial,
                kernel_size=tuple(kernel_size),
                padding=padding,
                padding_mode=padding_mode,
            ),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            make_conv(
                ffn_factor * channels,
                channels,
                spatial=spatial,
                kernel_size=tuple(kernel_size),
                padding=padding,
                padding_mode=padding_mode,
            ),
        )
        with torch.no_grad():
            self.ffn[-1].weight.mul_(1e-2)

    def _forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.ffn(self.norm(inputs))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.checkpointing and self.training:
            return checkpoint(self._forward, inputs, use_reentrant=False)
        return self._forward(inputs)


class DCEncoder(nn.Module):
    """Deep-compression encoder with explicit TCV boundary semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: Sequence[int],
        hidden_blocks: Sequence[int],
        stride: int | Sequence[int] | Sequence[Sequence[int]],
        kernel_size: int | Sequence[int] = 3,
        spatial: int = 3,
        patch_size: int | Sequence[int] = 1,
        periodic: bool | Sequence[bool] = (False, False, True),
        wall_padding_mode: str = "zeros",
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        dropout: float | None = 0.05,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_channels = tuple(int(item) for item in hidden_channels)
        hidden_blocks = tuple(int(item) for item in hidden_blocks)
        if len(hidden_channels) != len(hidden_blocks):
            raise ValueError("hidden_channels and hidden_blocks must have equal length")
        kernel = _kernel(kernel_size, spatial)
        strides = normalize_strides(
            stride,
            spatial=spatial,
            transitions=len(hidden_channels) - 1,
        )
        if isinstance(patch_size, int):
            patch = (patch_size,) * spatial
        else:
            patch = tuple(int(item) for item in patch_size)
        modes = padding_modes(
            periodic,
            spatial=spatial,
            wall_mode=wall_padding_mode,
        )
        conv_kwargs = {
            "kernel_size": kernel,
            "padding": tuple(item // 2 for item in kernel),
            "padding_mode": modes,
        }

        self.patch = PatchifyND(patch)
        self.levels = nn.ModuleList()
        for level, (channels, blocks) in enumerate(
            zip(hidden_channels, hidden_blocks)
        ):
            modules: list[nn.Module] = []
            if level == 0:
                modules.append(
                    make_conv(
                        math.prod(patch) * in_channels,
                        channels,
                        spatial=spatial,
                        **conv_kwargs,
                    )
                )
            else:
                level_stride = strides[level - 1]
                if pixel_shuffle:
                    modules.extend(
                        [
                            PatchifyND(level_stride),
                            make_conv(
                                hidden_channels[level - 1]
                                * math.prod(level_stride),
                                channels,
                                spatial=spatial,
                                identity_init=identity_init,
                                **conv_kwargs,
                            ),
                        ]
                    )
                else:
                    modules.append(
                        make_conv(
                            hidden_channels[level - 1],
                            channels,
                            spatial=spatial,
                            stride=level_stride,
                            identity_init=identity_init,
                            **conv_kwargs,
                        )
                    )
            modules.extend(
                ResidualBlock(
                    channels,
                    spatial=spatial,
                    kernel_size=kernel,
                    padding_mode=modes,
                    ffn_factor=ffn_factor,
                    dropout=dropout,
                    checkpointing=checkpointing,
                )
                for _ in range(blocks)
            )
            if level + 1 == len(hidden_channels):
                modules.append(
                    make_conv(
                        channels,
                        out_channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **conv_kwargs,
                    )
                )
            self.levels.append(nn.Sequential(*modules))

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.patch(inputs)
        for level in self.levels:
            result = level(result)
        return result


class DCDecoder(nn.Module):
    """Inverse deep-compression decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: Sequence[int],
        hidden_blocks: Sequence[int],
        stride: int | Sequence[int] | Sequence[Sequence[int]],
        kernel_size: int | Sequence[int] = 3,
        spatial: int = 3,
        patch_size: int | Sequence[int] = 1,
        periodic: bool | Sequence[bool] = (False, False, True),
        wall_padding_mode: str = "zeros",
        pixel_shuffle: bool = True,
        ffn_factor: int = 1,
        dropout: float | None = 0.05,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_channels = tuple(int(item) for item in hidden_channels)
        hidden_blocks = tuple(int(item) for item in hidden_blocks)
        if len(hidden_channels) != len(hidden_blocks):
            raise ValueError("hidden_channels and hidden_blocks must have equal length")
        kernel = _kernel(kernel_size, spatial)
        strides = normalize_strides(
            stride,
            spatial=spatial,
            transitions=len(hidden_channels) - 1,
        )
        if isinstance(patch_size, int):
            patch = (patch_size,) * spatial
        else:
            patch = tuple(int(item) for item in patch_size)
        modes = padding_modes(
            periodic,
            spatial=spatial,
            wall_mode=wall_padding_mode,
        )
        conv_kwargs = {
            "kernel_size": kernel,
            "padding": tuple(item // 2 for item in kernel),
            "padding_mode": modes,
        }

        self.levels = nn.ModuleList()
        for level in reversed(range(len(hidden_channels))):
            channels = hidden_channels[level]
            modules: list[nn.Module] = []
            if level + 1 == len(hidden_channels):
                modules.append(
                    make_conv(
                        in_channels,
                        channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **conv_kwargs,
                    )
                )
            modules.extend(
                ResidualBlock(
                    channels,
                    spatial=spatial,
                    kernel_size=kernel,
                    padding_mode=modes,
                    ffn_factor=ffn_factor,
                    dropout=dropout,
                    checkpointing=checkpointing,
                )
                for _ in range(hidden_blocks[level])
            )
            if level > 0:
                level_stride = strides[level - 1]
                if pixel_shuffle:
                    modules.extend(
                        [
                            make_conv(
                                channels,
                                hidden_channels[level - 1]
                                * math.prod(level_stride),
                                spatial=spatial,
                                identity_init=identity_init,
                                **conv_kwargs,
                            ),
                            UnpatchifyND(level_stride),
                        ]
                    )
                else:
                    modules.extend(
                        [
                            nn.Upsample(
                                scale_factor=level_stride,
                                mode="nearest",
                            ),
                            make_conv(
                                channels,
                                hidden_channels[level - 1],
                                spatial=spatial,
                                identity_init=identity_init,
                                **conv_kwargs,
                            ),
                        ]
                    )
            else:
                modules.append(
                    make_conv(
                        channels,
                        math.prod(patch) * out_channels,
                        spatial=spatial,
                        **conv_kwargs,
                    )
                )
            self.levels.append(nn.Sequential(*modules))
        self.unpatch = UnpatchifyND(patch)

    def forward(self, inputs: Tensor) -> Tensor:
        result = inputs
        for level in self.levels:
            result = level(result)
        return self.unpatch(result)


class AutoEncoder(nn.Module):
    """Encoder/decoder pair with the frozen LOLA softclip2 latent map."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def saturate(latent: Tensor) -> Tensor:
        return latent * torch.rsqrt(1 + torch.square(latent / 5))

    def encode(self, inputs: Tensor) -> Tensor:
        return self.saturate(self.encoder(inputs))

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(inputs)
        return self.decode(latent), latent


@dataclass(frozen=True)
class CodecConfig:
    name: str
    hidden_channels: tuple[int, ...]
    hidden_blocks: tuple[int, ...]
    latent_channels: int
    strides: tuple[tuple[int, int, int], ...]
    expected_latent_grid: tuple[int, int, int]
    predictor_patch: tuple[int, int, int]


CODEC_CONFIGS = {
    "dcae_l20": CodecConfig(
        name="dcae_l20",
        hidden_channels=(64, 128, 256, 512),
        hidden_blocks=(2, 2, 2, 2),
        latent_channels=64,
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 1)),
        expected_latent_grid=(8, 4, 22),
        predictor_patch=(1, 1, 1),
    ),
    "dcae_l10": CodecConfig(
        name="dcae_l10",
        hidden_channels=(64, 128, 256),
        hidden_blocks=(2, 2, 2),
        latent_channels=32,
        strides=((2, 2, 2), (2, 2, 2)),
        expected_latent_grid=(16, 8, 22),
        predictor_patch=(2, 2, 1),
    ),
}


def latent_shape(
    config: CodecConfig,
    input_grid: Sequence[int] = (64, 32, 88),
) -> tuple[int, int, int, int]:
    grid = tuple(int(item) for item in input_grid)
    for stride in config.strides:
        if any(size % step for size, step in zip(grid, stride)):
            raise ValueError(f"grid {grid} is not divisible by stride {stride}")
        grid = tuple(size // step for size, step in zip(grid, stride))
    if tuple(input_grid) == (64, 32, 88) and grid != config.expected_latent_grid:
        raise AssertionError((config.name, grid, config.expected_latent_grid))
    return (config.latent_channels, *grid)


def build_codec(name: str, pixel_channels: int) -> AutoEncoder:
    """Build one frozen codec candidate from scratch."""

    try:
        config = CODEC_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"unknown codec {name!r}") from exc
    common = {
        "hidden_channels": config.hidden_channels,
        "hidden_blocks": config.hidden_blocks,
        "stride": config.strides,
        "kernel_size": 3,
        "spatial": 3,
        "patch_size": 1,
        "periodic": (False, False, True),
        "wall_padding_mode": "zeros",
        "pixel_shuffle": True,
        "ffn_factor": 1,
        "dropout": 0.05,
        "checkpointing": False,
        "identity_init": True,
    }
    return AutoEncoder(
        DCEncoder(
            pixel_channels,
            config.latent_channels,
            **common,
        ),
        DCDecoder(
            config.latent_channels,
            pixel_channels,
            **common,
        ),
    )


def equal_channel_mae(target: Tensor, prediction: Tensor) -> Tensor:
    """Arithmetic mean of per-channel MAE in standardized coordinates."""

    if target.shape != prediction.shape:
        raise ValueError(
            f"target/prediction shape mismatch: {target.shape} != {prediction.shape}"
        )
    if target.ndim < 3:
        raise ValueError("expected [batch,channel,spatial...] tensors")
    return (target - prediction).abs().flatten(start_dim=2).mean(dim=(0, 2)).mean()
