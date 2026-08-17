"""Small N-D layers for the Paper 0 deterministic models.

Patch ordering follows the LOLA implementation at upstream commit
21a4354b327e6e5ee06da5075ba3bd1dd88c61f1. Per-axis padding incorporates
the predecessor TCV repair: x/y are walls and z is periodic. See
LOLA_LICENSE.txt in this directory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_CONVOLUTIONS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
_PAD_ALIASES = {
    "zeros": "constant",
    "circular": "circular",
    "replicate": "replicate",
    "reflect": "reflect",
}


def _tuple(value: int | Sequence[int], length: int, *, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (value,) * length
    else:
        result = tuple(int(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} entries, got {result}")
    if any(item < 0 for item in result):
        raise ValueError(f"{name} entries must be nonnegative, got {result}")
    return result


def identity_init_(
    convolution: nn.Module,
    in_channels: int,
    out_channels: int,
) -> None:
    """Add a channel-cycling spatial identity to a small random convolution."""

    kernel_size = tuple(int(size) for size in convolution.weight.shape[2:])
    center = tuple(size // 2 for size in kernel_size)
    identity = torch.zeros_like(convolution.weight)
    for output in range(out_channels):
        identity[(output, output % in_channels, *center)] = 1
    with torch.no_grad():
        convolution.weight.mul_(1e-2)
        convolution.weight.add_(identity)


class PerAxisConvNd(nn.Module):
    """Convolution with an independent padding mode on each spatial axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        spatial: int,
        padding: int | Sequence[int],
        padding_mode: Sequence[str],
        identity_init: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if spatial not in _CONVOLUTIONS:
            raise ValueError(f"unsupported spatial rank {spatial}")
        self.spatial = spatial
        self.pads = _tuple(padding, spatial, name="padding")
        modes = tuple(padding_mode)
        if len(modes) != spatial:
            raise ValueError(
                f"padding_mode must contain {spatial} entries, got {modes}"
            )
        try:
            self.modes = tuple(_PAD_ALIASES[mode] for mode in modes)
        except KeyError as exc:
            raise ValueError(f"unsupported padding mode {exc.args[0]!r}") from exc

        convolution = _CONVOLUTIONS[spatial]
        self.conv = convolution(
            in_channels,
            out_channels,
            padding=0,
            **kwargs,
        )
        if identity_init:
            identity_init_(self.conv, in_channels, out_channels)

    @property
    def weight(self) -> Tensor:
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        return self.conv.bias

    def forward(self, inputs: Tensor) -> Tensor:
        result = inputs
        for axis, (amount, mode) in enumerate(zip(self.pads, self.modes)):
            if amount == 0:
                continue
            padding = [0] * (2 * self.spatial)
            position = 2 * (self.spatial - 1 - axis)
            padding[position] = amount
            padding[position + 1] = amount
            if mode == "constant":
                result = F.pad(result, padding, mode=mode, value=0.0)
            else:
                result = F.pad(result, padding, mode=mode)
        return self.conv(result)


def make_conv(
    in_channels: int,
    out_channels: int,
    *,
    spatial: int,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    """Construct a 1-D, 2-D, or 3-D convolution with optional mixed padding."""

    if spatial not in _CONVOLUTIONS:
        raise ValueError(f"unsupported spatial rank {spatial}")
    padding_mode = kwargs.get("padding_mode", "zeros")
    if isinstance(padding_mode, (tuple, list)):
        padding = kwargs.get("padding", 0)
        if isinstance(padding, int):
            pads = (padding,) * spatial
        else:
            pads = tuple(padding)
        if any(amount > 0 for amount in pads):
            return PerAxisConvNd(
                in_channels,
                out_channels,
                spatial=spatial,
                identity_init=identity_init,
                **kwargs,
            )
        kwargs = {**kwargs, "padding_mode": "zeros"}

    convolution = _CONVOLUTIONS[spatial](
        in_channels,
        out_channels,
        **kwargs,
    )
    if identity_init:
        identity_init_(convolution, in_channels, out_channels)
    return convolution


class ChannelLayerNorm(nn.Module):
    """LOLA-compatible unparameterized normalization over the channel axis."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("eps", torch.as_tensor(float(eps)))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim < 3:
            raise ValueError("channel normalization expects [batch,channel,...]")
        variance, mean = torch.var_mean(
            inputs,
            dim=1,
            keepdim=True,
            correction=1,
        )
        return (inputs - mean) * torch.rsqrt(variance + self.eps)


class PatchifyND(nn.Module):
    """Move regular spatial patches into the channel dimension."""

    def __init__(self, patch_size: int | Sequence[int], spatial: int | None = None):
        super().__init__()
        if isinstance(patch_size, int):
            if spatial is None:
                raise ValueError("spatial is required for scalar patch_size")
            self.patch_size = (patch_size,) * spatial
        else:
            self.patch_size = tuple(int(item) for item in patch_size)
        if not self.patch_size or any(item <= 0 for item in self.patch_size):
            raise ValueError(f"invalid patch size {self.patch_size}")

    def forward(self, inputs: Tensor) -> Tensor:
        spatial = len(self.patch_size)
        if inputs.ndim != spatial + 2:
            raise ValueError(
                f"expected [batch,channel,{spatial} spatial axes], "
                f"got shape {tuple(inputs.shape)}"
            )
        batch, channels = inputs.shape[:2]
        dimensions = tuple(int(item) for item in inputs.shape[2:])
        if any(size % patch for size, patch in zip(dimensions, self.patch_size)):
            raise ValueError(
                f"shape {dimensions} is not divisible by {self.patch_size}"
            )
        coarse = tuple(
            size // patch for size, patch in zip(dimensions, self.patch_size)
        )
        interleaved: list[int] = []
        for size, patch in zip(coarse, self.patch_size):
            interleaved.extend((size, patch))
        result = inputs.reshape(batch, channels, *interleaved)
        patch_axes = [3 + 2 * axis for axis in range(spatial)]
        coarse_axes = [2 + 2 * axis for axis in range(spatial)]
        result = result.permute(0, 1, *patch_axes, *coarse_axes).contiguous()
        return result.reshape(
            batch,
            channels * math.prod(self.patch_size),
            *coarse,
        )


class UnpatchifyND(nn.Module):
    """Inverse of PatchifyND for channels-first tensors."""

    def __init__(self, patch_size: int | Sequence[int], spatial: int | None = None):
        super().__init__()
        if isinstance(patch_size, int):
            if spatial is None:
                raise ValueError("spatial is required for scalar patch_size")
            self.patch_size = (patch_size,) * spatial
        else:
            self.patch_size = tuple(int(item) for item in patch_size)
        if not self.patch_size or any(item <= 0 for item in self.patch_size):
            raise ValueError(f"invalid patch size {self.patch_size}")

    def forward(self, inputs: Tensor) -> Tensor:
        spatial = len(self.patch_size)
        if inputs.ndim != spatial + 2:
            raise ValueError(
                f"expected [batch,channel,{spatial} spatial axes], "
                f"got shape {tuple(inputs.shape)}"
            )
        batch, packed_channels = inputs.shape[:2]
        patch_volume = math.prod(self.patch_size)
        if packed_channels % patch_volume:
            raise ValueError(
                f"{packed_channels} channels are not divisible by {patch_volume}"
            )
        channels = packed_channels // patch_volume
        coarse = tuple(int(item) for item in inputs.shape[2:])
        result = inputs.reshape(
            batch,
            channels,
            *self.patch_size,
            *coarse,
        )
        permutation = [0, 1]
        for axis in range(spatial):
            permutation.extend((2 + spatial + axis, 2 + axis))
        result = result.permute(*permutation).contiguous()
        full = tuple(
            size * patch for size, patch in zip(coarse, self.patch_size)
        )
        return result.reshape(batch, channels, *full)
