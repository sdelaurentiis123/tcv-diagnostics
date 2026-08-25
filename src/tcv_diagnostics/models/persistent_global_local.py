"""Persistent global--local joint residual diffusion components.

The model in this module is the bounded old-85604 mechanism pilot frozen in
``POST_ECRD_OLD_85604_PERSISTENT_GLOBAL_LOCAL_PILOT_2026-08-25.md``.
It is deliberately data agnostic: no split, diagnostic, geometry mask, flux,
or held-out artifact is read here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .field_residual_edm import _expand_sample_scalar, _group_count
from .layers import make_conv
from .modulated_vit import NoiseTimeEmbedding


PGL_FIELD_ORDER = ("Ne", "Pe", "Pi", "phi", "Vi")


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
    """Resize only nonperiodic axes and retain every toroidal cell."""

    if inputs.ndim != 5:
        raise ValueError("x/y upsampling expects [batch,channel,x,y,z]")
    target = tuple(int(value) for value in size)
    if len(target) != 3 or any(value <= 0 for value in target):
        raise ValueError("x/y upsampling target must have three positive sizes")
    batch, channels, n_x, n_y, n_z = inputs.shape
    if target[2] != n_z:
        raise ValueError("persistent global-local model cannot resize toroidal z")
    slices = inputs.permute(0, 4, 1, 2, 3).reshape(
        batch * n_z, channels, n_x, n_y
    )
    resized = F.interpolate(
        slices,
        size=target[:2],
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(
        batch, n_z, channels, target[0], target[1]
    ).permute(0, 2, 3, 4, 1).contiguous()


def toroidal_lowpass(inputs: Tensor, *, maximum_mode: int) -> Tensor:
    """Project the last, periodic axis onto stored modes ``|k| <= maximum``."""

    if inputs.ndim < 2 or inputs.shape[-1] < 3:
        raise ValueError("toroidal projection requires a nontrivial last axis")
    maximum = int(maximum_mode)
    available = inputs.shape[-1] // 2
    if maximum < 0 or maximum >= available:
        raise ValueError("low-mode cutoff must leave a nonempty high-mode band")
    original_dtype = inputs.dtype
    spectrum = torch.fft.rfft(inputs.float(), dim=-1)
    mask = torch.arange(
        spectrum.shape[-1], device=spectrum.device
    ) <= maximum
    projected = torch.fft.irfft(
        spectrum * mask.to(spectrum.dtype),
        n=inputs.shape[-1],
        dim=-1,
    )
    return projected.to(dtype=original_dtype)


def toroidal_highpass(inputs: Tensor, *, maximum_mode: int) -> Tensor:
    """Complement of :func:`toroidal_lowpass` on the periodic last axis."""

    return inputs - toroidal_lowpass(inputs, maximum_mode=maximum_mode)


def _component_unit_rms(values: Tensor, *, dimensions: Sequence[int]) -> Tensor:
    rms = torch.mean(values.float().square(), dim=tuple(dimensions), keepdim=True).sqrt()
    if torch.any(~torch.isfinite(rms)) or torch.any(rms <= 1.0e-8):
        raise FloatingPointError("structured-noise component has zero or invalid RMS")
    return values / rms.to(values)


@dataclass(frozen=True)
class PersistentNoiseConfig:
    """Equal-RMS persistent-low-mode plus independent-local corruption."""

    global_weight: float = 1.0
    local_weight: float = 1.0
    global_pool_xy: tuple[int, int] = (4, 4)
    low_mode_maximum: int = 7

    def __post_init__(self) -> None:
        weights = (float(self.global_weight), float(self.local_weight))
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("persistent noise weights must be finite and positive")
        if len(self.global_pool_xy) != 2 or any(
            int(value) <= 0 for value in self.global_pool_xy
        ):
            raise ValueError("persistent noise x/y pooling must be positive")
        if int(self.low_mode_maximum) < 0:
            raise ValueError("persistent noise mode cutoff must be nonnegative")

    @property
    def normalization(self) -> float:
        return math.sqrt(self.global_weight**2 + self.local_weight**2)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["global_pool_xy"] = list(self.global_pool_xy)
        record.update(
            {
                "global_draw_shared_across_future_time": True,
                "global_band": f"abs(k)<={self.low_mode_maximum}",
                "local_band": f"abs(k)>{self.low_mode_maximum}",
                "component_RMS_matching": "per_sample_field",
                "posthoc_spread_multiplier": False,
            }
        )
        return record


@dataclass(frozen=True)
class PersistentNoiseSample:
    total: Tensor
    global_component: Tensor
    local_component: Tensor


def sample_persistent_global_local_noise(
    reference: Tensor,
    *,
    config: PersistentNoiseConfig = PersistentNoiseConfig(),
    generator: torch.Generator | None = None,
) -> PersistentNoiseSample:
    """Draw structured noise for ``[B,K,C,x,y,z]`` joint trajectories."""

    if reference.ndim != 6:
        raise ValueError("persistent noise reference must be [B,K,C,x,y,z]")
    batch, horizon, channels, n_x, n_y, n_z = reference.shape
    factor_x, factor_y = config.global_pool_xy
    if n_x % factor_x or n_y % factor_y:
        raise ValueError("reference x/y sizes must divide persistent pooling")
    if config.low_mode_maximum >= n_z // 2:
        raise ValueError("reference z is too short for the frozen mode split")
    kwargs = {
        "device": reference.device,
        "dtype": reference.dtype,
        "generator": generator,
    }
    coarse = torch.randn(
        (batch, 1, channels, n_x // factor_x, n_y // factor_y, n_z),
        **kwargs,
    )
    global_full = coarse.repeat_interleave(factor_x, dim=-3).repeat_interleave(
        factor_y, dim=-2
    )
    global_full = toroidal_lowpass(
        global_full, maximum_mode=config.low_mode_maximum
    )
    global_full = _component_unit_rms(
        global_full, dimensions=(-3, -2, -1)
    ).expand(batch, horizon, channels, n_x, n_y, n_z)

    local = torch.randn(reference.shape, **kwargs)
    local = toroidal_highpass(local, maximum_mode=config.low_mode_maximum)
    local = _component_unit_rms(local, dimensions=(-3, -2, -1))
    total = (
        config.global_weight * global_full + config.local_weight * local
    ) / config.normalization
    if not (
        torch.isfinite(total).all()
        and torch.isfinite(global_full).all()
        and torch.isfinite(local).all()
    ):
        raise FloatingPointError("persistent global-local noise is non-finite")
    return PersistentNoiseSample(
        total=total,
        global_component=global_full,
        local_component=local,
    )


@dataclass(frozen=True)
class PersistentGlobalLocalConfig:
    """Four-step global-recurrent/local-equivariant denoiser configuration."""

    horizon: int = 4
    fields: int = 5
    base_channels: int = 16
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    residual_blocks_per_resolution: int = 1
    global_channels: int = 24
    global_pool_xy: tuple[int, int] = (4, 4)
    low_mode_maximum: int = 7
    noise_embedding_features: int = 128
    group_norm_maximum_groups: int = 8
    kernel_size: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        counts = (
            self.horizon,
            self.fields,
            self.base_channels,
            self.residual_blocks_per_resolution,
            self.global_channels,
            self.noise_embedding_features,
            self.group_norm_maximum_groups,
            self.kernel_size,
        )
        if min(counts) <= 0:
            raise ValueError("persistent global-local dimensions must be positive")
        if self.horizon != 4 or self.fields != 5:
            raise ValueError("the frozen pilot jointly predicts four C5P frames")
        if not self.channel_multipliers or any(
            int(value) <= 0 for value in self.channel_multipliers
        ):
            raise ValueError("channel multipliers must be positive")
        if len(self.global_pool_xy) != 2 or any(
            int(value) <= 0 for value in self.global_pool_xy
        ):
            raise ValueError("global pooling must be positive")
        if self.low_mode_maximum < 0:
            raise ValueError("global mode cutoff must be nonnegative")
        if self.noise_embedding_features % 2:
            raise ValueError("noise embedding width must be even")
        if self.kernel_size != 3:
            raise ValueError("the frozen pilot uses kernel size three")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    @property
    def level_channels(self) -> tuple[int, ...]:
        return tuple(
            self.base_channels * int(multiplier)
            for multiplier in self.channel_multipliers
        )

    @property
    def local_input_channels(self) -> int:
        return 3 * self.fields + 2

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["channel_multipliers"] = list(self.channel_multipliers)
        record["global_pool_xy"] = list(self.global_pool_xy)
        record.update(
            {
                "field_order": list(PGL_FIELD_ORDER),
                "joint_future_frames": self.horizon,
                "padding_xyz": ["zeros", "zeros", "circular"],
                "toroidal_downsampling": False,
                "absolute_z_coordinate": False,
                "absolute_time_condition": False,
                "global_recurrence": "ConvGRU_over_future_time",
                "global_output_band": f"abs(k)<={self.low_mode_maximum}",
                "local_output_band": f"abs(k)>{self.low_mode_maximum}",
                "physics_derived_loss": False,
            }
        )
        return record


class _ConvGRUCell3D(nn.Module):
    """Mixed-boundary ConvGRU cell used only on the coarse global stream."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        joined = input_channels + hidden_channels
        self.gates = _mixed_conv(joined, 2 * hidden_channels, kernel_size=3)
        self.candidate = _mixed_conv(joined, hidden_channels, kernel_size=3)
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)

    def forward(self, inputs: Tensor, hidden: Tensor) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.input_channels:
            raise ValueError("ConvGRU input shape differs")
        if hidden.shape != (
            inputs.shape[0], self.hidden_channels, *inputs.shape[2:]
        ):
            raise ValueError("ConvGRU hidden shape differs")
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, hidden), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * hidden), dim=1))
        )
        return (1.0 - update) * hidden + update * candidate


class _PersistentGlobalStream3D(nn.Module):
    """Low-mode recurrent stream shared by every local decoding resolution."""

    def __init__(self, config: PersistentGlobalLocalConfig) -> None:
        super().__init__()
        channels = config.global_channels
        self.config = config
        self.noise_embedding = NoiseTimeEmbedding(config.noise_embedding_features)
        self.input_projection = _mixed_conv(
            3 * config.fields,
            channels,
            kernel_size=3,
        )
        self.initial_projection = _mixed_conv(
            2 * config.fields,
            channels,
            kernel_size=3,
        )
        self.noise_projection = nn.Linear(config.noise_embedding_features, channels)
        self.recurrent = _ConvGRUCell3D(channels, channels)
        self.output_normalization = nn.GroupNorm(
            _group_count(channels, config.group_norm_maximum_groups), channels
        )
        self.output_projection = _mixed_conv(channels, config.fields, kernel_size=3)
        with torch.no_grad():
            self.output_projection.weight.zero_()
            if self.output_projection.bias is not None:
                self.output_projection.bias.zero_()

    def _pool(self, inputs: Tensor) -> Tensor:
        factor_x, factor_y = self.config.global_pool_xy
        if inputs.shape[-3] % factor_x or inputs.shape[-2] % factor_y:
            raise ValueError("global stream x/y shape is incompatible with pooling")
        return F.avg_pool3d(
            inputs,
            kernel_size=(factor_x, factor_y, 1),
            stride=(factor_x, factor_y, 1),
        )

    def forward(
        self,
        noisy: Tensor,
        current: Tensor,
        mean: Tensor,
        noise_coordinate: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if noisy.ndim != 6:
            raise ValueError("global stream noisy trajectory must be six-dimensional")
        batch, horizon, fields, n_x, n_y, n_z = noisy.shape
        expected = (batch, self.config.horizon, self.config.fields, n_x, n_y, n_z)
        if tuple(noisy.shape) != expected or tuple(mean.shape) != expected:
            raise ValueError("global stream trajectory shape differs")
        if current.shape != (batch, self.config.fields, n_x, n_y, n_z):
            raise ValueError("global stream current-state shape differs")
        if noise_coordinate.shape != (batch,):
            raise ValueError("global stream noise coordinate shape differs")

        current_low = toroidal_lowpass(
            self._pool(current), maximum_mode=self.config.low_mode_maximum
        )
        first_mean_low = toroidal_lowpass(
            self._pool(mean[:, 0]), maximum_mode=self.config.low_mode_maximum
        )
        hidden = self.initial_projection(torch.cat((current_low, first_mean_low), dim=1))
        embedding = self.noise_projection(self.noise_embedding(noise_coordinate)).reshape(
            batch, self.config.global_channels, 1, 1, 1
        )
        hidden = hidden + embedding

        decoded: list[Tensor] = []
        features: list[Tensor] = []
        for step in range(horizon):
            noisy_low = toroidal_lowpass(
                self._pool(noisy[:, step]),
                maximum_mode=self.config.low_mode_maximum,
            )
            mean_low = toroidal_lowpass(
                self._pool(mean[:, step]),
                maximum_mode=self.config.low_mode_maximum,
            )
            recurrent_input = self.input_projection(
                torch.cat((noisy_low, mean_low, current_low), dim=1)
            ) + embedding
            hidden = self.recurrent(recurrent_input, hidden)
            features.append(hidden)
            decoded.append(
                self.output_projection(F.silu(self.output_normalization(hidden)))
            )

        coarse_features = torch.stack(features, dim=1)
        coarse_output = torch.stack(decoded, dim=1)
        flat_features = coarse_features.reshape(
            batch * horizon,
            self.config.global_channels,
            *coarse_features.shape[-3:],
        )
        flat_output = coarse_output.reshape(
            batch * horizon,
            self.config.fields,
            *coarse_output.shape[-3:],
        )
        full_features = xy_bilinear_upsample(flat_features, (n_x, n_y, n_z)).reshape(
            batch, horizon, self.config.global_channels, n_x, n_y, n_z
        )
        full_output = xy_bilinear_upsample(flat_output, (n_x, n_y, n_z)).reshape(
            batch, horizon, self.config.fields, n_x, n_y, n_z
        )
        return (
            toroidal_lowpass(
                full_output, maximum_mode=self.config.low_mode_maximum
            ),
            full_features,
        )


class _GlobalFiLMResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        global_channels: int,
        noise_features: int,
        maximum_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.normalization1 = nn.GroupNorm(
            _group_count(in_channels, maximum_groups), in_channels
        )
        self.convolution1 = _mixed_conv(in_channels, out_channels, kernel_size=3)
        self.normalization2 = nn.GroupNorm(
            _group_count(out_channels, maximum_groups), out_channels
        )
        self.noise_projection = nn.Linear(noise_features, 2 * out_channels)
        self.global_projection = _mixed_conv(
            global_channels, 2 * out_channels, kernel_size=1, padding=0
        )
        self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
        self.convolution2 = _mixed_conv(out_channels, out_channels, kernel_size=3)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else _mixed_conv(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.global_channels = int(global_channels)
        self.noise_features = int(noise_features)

    def forward(
        self,
        inputs: Tensor,
        noise_embedding: Tensor,
        global_features: Tensor,
    ) -> Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != self.in_channels:
            raise ValueError("local residual-block input shape differs")
        if noise_embedding.shape != (inputs.shape[0], self.noise_features):
            raise ValueError("local residual-block noise embedding differs")
        if global_features.shape != (
            inputs.shape[0], self.global_channels, *inputs.shape[2:]
        ):
            raise ValueError("local residual-block global features differ")
        hidden = self.convolution1(F.silu(self.normalization1(inputs)))
        noise_scale, noise_shift = self.noise_projection(noise_embedding).chunk(2, dim=1)
        global_scale, global_shift = self.global_projection(global_features).chunk(2, dim=1)
        noise_scale = noise_scale.reshape(inputs.shape[0], -1, 1, 1, 1)
        noise_shift = noise_shift.reshape(inputs.shape[0], -1, 1, 1, 1)
        hidden = self.normalization2(hidden) * (
            1.0 + noise_scale + global_scale
        ) + noise_shift + global_shift
        hidden = self.convolution2(self.dropout(F.silu(hidden)))
        return (self.skip(inputs) + hidden) * math.sqrt(0.5)


def _normalized_xy(reference: Tensor) -> Tensor:
    if reference.ndim != 5:
        raise ValueError("x/y coordinates require a five-dimensional reference")
    batch, _, n_x, n_y, n_z = reference.shape
    x = torch.linspace(-1.0, 1.0, n_x, device=reference.device, dtype=reference.dtype)
    y = torch.linspace(-1.0, 1.0, n_y, device=reference.device, dtype=reference.dtype)
    grid_x = x.reshape(1, 1, n_x, 1, 1).expand(batch, 1, n_x, n_y, n_z)
    grid_y = y.reshape(1, 1, 1, n_y, 1).expand(batch, 1, n_x, n_y, n_z)
    return torch.cat((grid_x, grid_y), dim=1)


class _LocalHighModeUNet3D(nn.Module):
    """Full-z local U-Net conditioned by the decoded recurrent global state."""

    def __init__(self, config: PersistentGlobalLocalConfig) -> None:
        super().__init__()
        self.config = config
        levels = config.level_channels
        self.noise_embedding = NoiseTimeEmbedding(config.noise_embedding_features)
        self.input_projection = _mixed_conv(
            config.local_input_channels, levels[0], kernel_size=3
        )
        encoders: list[nn.ModuleList] = []
        downsamples: list[nn.Module] = []
        current = levels[0]
        for level, channels in enumerate(levels):
            blocks: list[nn.Module] = []
            for _ in range(config.residual_blocks_per_resolution):
                blocks.append(
                    _GlobalFiLMResidualBlock3D(
                        current,
                        channels,
                        global_channels=config.global_channels,
                        noise_features=config.noise_embedding_features,
                        maximum_groups=config.group_norm_maximum_groups,
                        dropout=config.dropout,
                    )
                )
                current = channels
            encoders.append(nn.ModuleList(blocks))
            if level < len(levels) - 1:
                next_channels = levels[level + 1]
                downsamples.append(
                    _mixed_conv(
                        current,
                        next_channels,
                        kernel_size=3,
                        stride=(2, 2, 1),
                    )
                )
                current = next_channels
        self.encoders = nn.ModuleList(encoders)
        self.downsamples = nn.ModuleList(downsamples)

        merges: list[nn.Module] = []
        decoders: list[nn.ModuleList] = []
        for level in range(len(levels) - 2, -1, -1):
            channels = levels[level]
            merges.append(
                _mixed_conv(current + channels, channels, kernel_size=3)
            )
            current = channels
            decoders.append(
                nn.ModuleList(
                    _GlobalFiLMResidualBlock3D(
                        current,
                        channels,
                        global_channels=config.global_channels,
                        noise_features=config.noise_embedding_features,
                        maximum_groups=config.group_norm_maximum_groups,
                        dropout=config.dropout,
                    )
                    for _ in range(config.residual_blocks_per_resolution)
                )
            )
        self.merges = nn.ModuleList(merges)
        self.decoders = nn.ModuleList(decoders)
        self.output_normalization = nn.GroupNorm(
            _group_count(current, config.group_norm_maximum_groups), current
        )
        self.output_projection = _mixed_conv(current, config.fields, kernel_size=3)
        with torch.no_grad():
            self.output_projection.weight.zero_()
            if self.output_projection.bias is not None:
                self.output_projection.bias.zero_()

    @staticmethod
    def _global_pyramid(global_features: Tensor, levels: int) -> tuple[Tensor, ...]:
        features = [global_features]
        for _ in range(1, levels):
            features.append(
                F.avg_pool3d(features[-1], kernel_size=(2, 2, 1), stride=(2, 2, 1))
            )
        return tuple(features)

    def forward(
        self,
        noisy: Tensor,
        current: Tensor,
        mean: Tensor,
        global_features: Tensor,
        noise_coordinate: Tensor,
    ) -> Tensor:
        if noisy.ndim != 6:
            raise ValueError("local stream noisy trajectory must be six-dimensional")
        batch, horizon, fields, n_x, n_y, n_z = noisy.shape
        expected = (batch, self.config.horizon, self.config.fields, n_x, n_y, n_z)
        if tuple(noisy.shape) != expected or tuple(mean.shape) != expected:
            raise ValueError("local stream trajectory shape differs")
        if current.shape != (batch, fields, n_x, n_y, n_z):
            raise ValueError("local stream current state differs")
        if global_features.shape != (
            batch, horizon, self.config.global_channels, n_x, n_y, n_z
        ):
            raise ValueError("local stream recurrent global features differ")

        repeated_current = current[:, None].expand(
            batch, horizon, fields, n_x, n_y, n_z
        )
        flat_noisy = noisy.reshape(batch * horizon, fields, n_x, n_y, n_z)
        flat_current = repeated_current.reshape(
            batch * horizon, fields, n_x, n_y, n_z
        )
        flat_mean = mean.reshape(batch * horizon, fields, n_x, n_y, n_z)
        flat_global = global_features.reshape(
            batch * horizon, self.config.global_channels, n_x, n_y, n_z
        )
        coordinates = _normalized_xy(flat_noisy)
        hidden = self.input_projection(
            torch.cat((flat_noisy, flat_current, flat_mean, coordinates), dim=1)
        )
        embedded = self.noise_embedding(
            noise_coordinate[:, None].expand(batch, horizon).reshape(batch * horizon)
        )
        global_pyramid = self._global_pyramid(flat_global, len(self.config.level_channels))

        skips: list[Tensor] = []
        for level, blocks in enumerate(self.encoders):
            for block in blocks:
                hidden = block(hidden, embedded, global_pyramid[level])
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)

        for offset, (merge, blocks) in enumerate(zip(self.merges, self.decoders)):
            level = len(skips) - 2 - offset
            skip = skips[level]
            hidden = xy_bilinear_upsample(hidden, skip.shape[-3:])
            hidden = merge(torch.cat((hidden, skip), dim=1))
            for block in blocks:
                hidden = block(hidden, embedded, global_pyramid[level])
        raw = self.output_projection(F.silu(self.output_normalization(hidden)))
        raw = raw.reshape(batch, horizon, fields, n_x, n_y, n_z)
        return toroidal_highpass(raw, maximum_mode=self.config.low_mode_maximum)


class PersistentGlobalLocalDenoiser3D(nn.Module):
    """Sum of disjoint low-mode recurrent and high-mode local predictions."""

    def __init__(
        self,
        config: PersistentGlobalLocalConfig = PersistentGlobalLocalConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.global_stream = _PersistentGlobalStream3D(config)
        self.local_stream = _LocalHighModeUNet3D(config)

    def forward(
        self,
        noisy: Tensor,
        current: Tensor,
        mean: Tensor,
        noise_coordinate: Tensor,
    ) -> Tensor:
        if not (
            torch.isfinite(noisy).all()
            and torch.isfinite(current).all()
            and torch.isfinite(mean).all()
            and torch.isfinite(noise_coordinate).all()
        ):
            raise ValueError("persistent global-local inputs must be finite")
        global_output, global_features = self.global_stream(
            noisy, current, mean, noise_coordinate
        )
        local_output = self.local_stream(
            noisy, current, mean, global_features, noise_coordinate
        )
        result = global_output + local_output
        if result.shape != noisy.shape:
            raise RuntimeError("persistent global-local denoiser output shape differs")
        return result


@dataclass(frozen=True)
class PersistentGlobalLocalLoss:
    loss: Tensor
    unweighted_mse: Tensor
    per_step_field_mse: Tensor
    sigma_minimum: float
    sigma_maximum: float


class PersistentGlobalLocalEDM(nn.Module):
    """EDM wrapper for one joint four-frame residual trajectory."""

    def __init__(
        self,
        config: PersistentGlobalLocalConfig = PersistentGlobalLocalConfig(),
        *,
        residual_scales: Tensor | Sequence[Sequence[float]],
        noise_config: PersistentNoiseConfig = PersistentNoiseConfig(),
        sigma_data: float = 1.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ) -> None:
        super().__init__()
        scales = torch.as_tensor(residual_scales, dtype=torch.float32)
        if scales.shape != (config.horizon, config.fields):
            raise ValueError("persistent residual scales must be [future,field]")
        if not torch.all(torch.isfinite(scales) & (scales > 0.0)):
            raise ValueError("persistent residual scales must be finite and positive")
        if sigma_data != 1.0 or p_std <= 0.0:
            raise ValueError("persistent EDM constants differ")
        if (
            noise_config.global_pool_xy != config.global_pool_xy
            or noise_config.low_mode_maximum != config.low_mode_maximum
        ):
            raise ValueError("architecture and structured-noise partitions differ")
        self.config = config
        self.noise_config = noise_config
        self.backbone = PersistentGlobalLocalDenoiser3D(config)
        self.register_buffer(
            "residual_scales",
            scales.reshape(1, config.horizon, config.fields, 1, 1, 1),
        )
        self.sigma_data = float(sigma_data)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)

    def normalize_residual(self, residual: Tensor) -> Tensor:
        if residual.ndim != 6 or residual.shape[1:3] != (
            self.config.horizon, self.config.fields
        ):
            raise ValueError("persistent residual shape differs")
        return residual / self.residual_scales.to(residual)

    def denormalize_residual(self, normalized: Tensor) -> Tensor:
        if normalized.ndim < 6 or normalized.shape[-5:-3] != (
            self.config.horizon, self.config.fields
        ):
            raise ValueError("persistent normalized residual axes differ")
        scales = self.residual_scales.to(normalized)
        while scales.ndim < normalized.ndim:
            scales = scales.unsqueeze(1)
        return normalized * scales

    def denoise(
        self,
        noisy: Tensor,
        current: Tensor,
        mean: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        if noisy.ndim != 6:
            raise ValueError("persistent noisy trajectory must be six-dimensional")
        sigma_values = torch.as_tensor(sigma, device=noisy.device, dtype=noisy.dtype)
        if sigma_values.ndim == 0:
            sigma_values = sigma_values.expand(noisy.shape[0])
        if sigma_values.shape != (noisy.shape[0],) or not torch.all(
            torch.isfinite(sigma_values) & (sigma_values > 0.0)
        ):
            raise ValueError("persistent EDM sigma shape or values differ")
        sigma_x = _expand_sample_scalar(sigma_values, noisy)
        denominator = torch.sqrt(sigma_x.square() + 1.0)
        c_in = torch.reciprocal(denominator)
        c_skip = torch.reciprocal(sigma_x.square() + 1.0)
        c_out = sigma_x / denominator
        network = self.backbone(
            c_in * noisy,
            current,
            mean,
            torch.log(sigma_values) * 0.25,
        )
        return c_skip * noisy + c_out * network

    def training_loss(
        self,
        clean: Tensor,
        current: Tensor,
        mean: Tensor,
        *,
        sigma: Tensor | None = None,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> PersistentGlobalLocalLoss:
        if clean.ndim != 6 or clean.shape[1:3] != (
            self.config.horizon, self.config.fields
        ):
            raise ValueError("persistent clean trajectory shape differs")
        if mean.shape != clean.shape or current.shape != (
            clean.shape[0], self.config.fields, *clean.shape[-3:]
        ):
            raise ValueError("persistent EDM condition shape differs")
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
            raise ValueError("persistent training sigma shape or values differ")
        if noise is None:
            noise_values = sample_persistent_global_local_noise(
                clean, config=self.noise_config, generator=generator
            ).total
        else:
            noise_values = torch.as_tensor(noise, device=clean.device, dtype=clean.dtype)
        if noise_values.shape != clean.shape or not torch.isfinite(noise_values).all():
            raise ValueError("persistent training noise shape or values differ")
        sigma_x = _expand_sample_scalar(sigma_values, clean)
        prediction = self.denoise(
            clean + sigma_x * noise_values,
            current,
            mean,
            sigma_values,
        )
        squared = (prediction.float() - clean.float()).square()
        per_sample = squared.flatten(1).mean(1)
        weights = (sigma_values.float().square() + 1.0) / sigma_values.float().square()
        loss = torch.mean(weights * per_sample)
        return PersistentGlobalLocalLoss(
            loss=loss,
            unweighted_mse=torch.mean(per_sample),
            per_step_field_mse=torch.mean(squared, dim=(0, 3, 4, 5)),
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
            raise ValueError("invalid persistent EDM sampling schedule")
        ramp = torch.linspace(0.0, 1.0, steps, device=device, dtype=torch.float64)
        maximum = float(sigma_max) ** (1.0 / float(rho))
        minimum = float(sigma_min) ** (1.0 / float(rho))
        sigma = (maximum + ramp * (minimum - maximum)).pow(float(rho))
        return torch.cat((sigma, sigma.new_zeros(1))).to(dtype=dtype)

    @torch.no_grad()
    def sample_normalized(
        self,
        current: Tensor,
        mean: Tensor,
        initial_noise: Tensor,
        *,
        steps: int = 18,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
    ) -> Tensor:
        """Return normalized residuals as ``[B,M,K,C,x,y,z]``."""

        if initial_noise.ndim != 7 or initial_noise.shape[0] != current.shape[0]:
            raise ValueError("persistent initial noise must be [B,M,K,C,x,y,z]")
        if initial_noise.shape[2:] != mean.shape[1:]:
            raise ValueError("persistent initial-noise trajectory shape differs")
        if not torch.isfinite(initial_noise).all():
            raise ValueError("persistent initial noise must be finite")
        batch, members = initial_noise.shape[:2]
        expanded_current = current[:, None].expand(
            batch, members, *current.shape[1:]
        ).reshape(batch * members, *current.shape[1:]).contiguous()
        expanded_mean = mean[:, None].expand(
            batch, members, *mean.shape[1:]
        ).reshape(batch * members, *mean.shape[1:]).contiguous()
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
            denoised = self.denoise(
                sample, expanded_current, expanded_mean, sigma_batch
            )
            derivative = (sample - denoised) / current_sigma
            proposed = sample + (next_sigma - current_sigma) * derivative
            if float(next_sigma) != 0.0:
                next_batch = next_sigma.expand(sample.shape[0])
                next_denoised = self.denoise(
                    proposed, expanded_current, expanded_mean, next_batch
                )
                next_derivative = (proposed - next_denoised) / next_sigma
                sample = sample + (next_sigma - current_sigma) * 0.5 * (
                    derivative + next_derivative
                )
            else:
                sample = proposed
        return sample.reshape(batch, members, *sample.shape[1:])

    def compose_fields(self, mean: Tensor, normalized_residual: Tensor) -> Tensor:
        if normalized_residual.ndim != 7 or normalized_residual.shape[0] != mean.shape[0]:
            raise ValueError("persistent residual ensemble shape differs")
        if normalized_residual.shape[2:] != mean.shape[1:]:
            raise ValueError("persistent residual and mean trajectories differ")
        return mean[:, None] + self.denormalize_residual(normalized_residual)

    def to_record(self) -> dict[str, Any]:
        return {
            "family": "persistent_global_local_joint_residual_EDM",
            "config": self.config.to_record(),
            "noise": self.noise_config.to_record(),
            "residual_scales": self.residual_scales.reshape(
                self.config.horizon, self.config.fields
            ).detach().cpu().tolist(),
            "sigma_data": self.sigma_data,
            "P_mean": self.p_mean,
            "P_std": self.p_std,
            "physics_derived_loss_used": False,
        }
