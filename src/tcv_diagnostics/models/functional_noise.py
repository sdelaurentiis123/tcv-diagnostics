"""Functional-generative retrofit of the frozen deterministic O2 transition.

The global-noise mechanism is a minimal adaptation of the official
``cddcam/lola_crps`` implementation at commit
``7643376c2949717ee5c2c840584689f529ba77a5``. The upstream MIT license is
retained in ``FGN_LICENSE.txt``. Existing O2 modules are intentionally left
unchanged so that the deterministic parent remains a hash-locked comparator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .o2 import C5POneStepModel, MaskedLatentTransition, O2ViTConfig
from .vit import ViT, ViTBlock, regular_grid_coordinates


@dataclass(frozen=True)
class FunctionalNoiseConfig:
    """Frozen B3 global-noise dimensions and initialization."""

    raw_noise_features: int = 32
    embedded_noise_features: int = 256
    adapter_last_weight_multiplier: float = 1.0e-2

    def __post_init__(self) -> None:
        if self.raw_noise_features <= 0 or self.embedded_noise_features <= 0:
            raise ValueError("functional-noise dimensions must be positive")
        if self.adapter_last_weight_multiplier <= 0.0:
            raise ValueError("adapter initialization multiplier must be positive")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "raw_distribution": "standard_normal",
                "spatial_semantics": "one_global_vector_shared_across_all_tokens",
                "noise_layers": "all",
            }
        )
        return record


@dataclass(frozen=True)
class DeterministicLoadAudit:
    """Exact accounting for loading one deterministic parent transition."""

    parent_key_count: int
    deterministic_child_key_count: int
    new_noise_keys: tuple[str, ...]
    unexpected_parent_keys: tuple[str, ...]
    missing_deterministic_keys: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.parent_key_count == self.deterministic_child_key_count
            and not self.unexpected_parent_keys
            and not self.missing_deterministic_keys
            and bool(self.new_noise_keys)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "parent_key_count": self.parent_key_count,
            "deterministic_child_key_count": self.deterministic_child_key_count,
            "new_noise_key_count": len(self.new_noise_keys),
            "new_noise_keys": list(self.new_noise_keys),
            "unexpected_parent_keys": list(self.unexpected_parent_keys),
            "missing_deterministic_keys": list(self.missing_deterministic_keys),
            "passed": self.passed,
        }


class FunctionalNoiseEmbedding(nn.Sequential):
    """Map one raw global Gaussian vector to a normalized embedding."""

    def __init__(self, raw_features: int, embedded_features: int) -> None:
        super().__init__(
            nn.Linear(raw_features, embedded_features),
            nn.SiLU(),
            nn.Linear(embedded_features, embedded_features),
            nn.LayerNorm(embedded_features),
        )
        self.raw_features = int(raw_features)
        self.embedded_features = int(embedded_features)


class FunctionalNoiseViTBlock(ViTBlock):
    """An O2 block with one additive AdaLN modulation from global noise."""

    def __init__(
        self,
        channels: int,
        *,
        noise_features: int,
        adapter_last_weight_multiplier: float,
        ffn_factor: int = 4,
        spatial: int = 2,
        attention_heads: int = 1,
        qk_norm: bool = True,
        rope: bool = True,
        dropout: float = 0.0,
        checkpointing: bool = False,
    ) -> None:
        super().__init__(
            channels,
            ffn_factor=ffn_factor,
            spatial=spatial,
            attention_heads=attention_heads,
            qk_norm=qk_norm,
            rope=rope,
            dropout=dropout,
            checkpointing=checkpointing,
        )
        if noise_features <= 0:
            raise ValueError("noise features must be positive")
        self.noise_adapter = nn.Sequential(
            nn.Linear(noise_features, noise_features),
            nn.SiLU(),
            nn.Linear(noise_features, 4 * channels),
        )
        with torch.no_grad():
            self.noise_adapter[-1].weight.mul_(adapter_last_weight_multiplier)
            self.noise_adapter[-1].bias.zero_()
        self.noise_features = int(noise_features)
        self.channels = int(channels)

    def _forward_with_noise(
        self,
        tokens: Tensor,
        coordinates: Tensor,
        skip: Tensor,
        noise: Tensor,
    ) -> Tensor:
        if noise.ndim != 2 or noise.shape != (
            tokens.shape[0],
            self.noise_features,
        ):
            raise ValueError(
                "embedded global noise must be [batch,embedded_noise_features]"
            )
        theta = (
            None
            if self.theta is None
            else torch.einsum("ld,dc->lc", coordinates, self.theta)
        )
        scale, shift, residual_scale, skip_scale = self.ada_zero
        noise_modulation = self.noise_adapter(noise).reshape(
            noise.shape[0], 4, self.channels
        )
        noise_scale, noise_shift, noise_residual, noise_skip = (
            item.unsqueeze(1) for item in noise_modulation.unbind(dim=1)
        )
        scale = scale + noise_scale
        shift = shift + noise_shift
        residual_scale = residual_scale + noise_residual
        skip_scale = skip_scale + noise_skip

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
        coordinates: Tensor,
        skip: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            # This exact inherited path is the pre-optimization identity gate.
            return super().forward(tokens, coordinates, skip)
        if self.checkpointing and self.training:
            return checkpoint(
                self._forward_with_noise,
                tokens,
                coordinates,
                skip,
                noise,
                use_reentrant=False,
            )
        return self._forward_with_noise(tokens, coordinates, skip, noise)


class FunctionalNoiseViT(ViT):
    """The deterministic O2 ViT with a global-noise adapter in every block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        noise_features: int,
        adapter_last_weight_multiplier: float,
        condition_channels: int = 0,
        hidden_channels: int = 1024,
        hidden_blocks: int = 3,
        attention_heads: int = 1,
        ffn_factor: int = 4,
        spatial: int = 2,
        patch_size: int | tuple[int, ...] = 1,
        qk_norm: bool = True,
        rope: bool = True,
        dropout: float = 0.0,
        checkpointing: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            condition_channels=condition_channels,
            hidden_channels=hidden_channels,
            hidden_blocks=hidden_blocks,
            attention_heads=attention_heads,
            ffn_factor=ffn_factor,
            spatial=spatial,
            patch_size=patch_size,
            qk_norm=qk_norm,
            rope=rope,
            dropout=dropout,
            checkpointing=checkpointing,
        )
        self.blocks = nn.ModuleList(
            FunctionalNoiseViTBlock(
                hidden_channels,
                noise_features=noise_features,
                adapter_last_weight_multiplier=adapter_last_weight_multiplier,
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
        self.noise_features = int(noise_features)

    def forward(
        self,
        inputs: Tensor,
        condition: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        if noise is None:
            return super().forward(inputs, condition)
        if inputs.ndim != self.spatial + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"ViT expects [batch,{self.in_channels},{self.spatial} axes], "
                f"got {inputs.shape}"
            )
        if noise.shape != (inputs.shape[0], self.noise_features):
            raise ValueError(
                f"noise shape must be {(inputs.shape[0], self.noise_features)}"
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
            tokens = block(tokens, coordinates, skip, noise)
        tokens = tokens.reshape(inputs.shape[0], *token_grid, self.hidden_channels)
        patches = self.output_projection(tokens).movedim(-1, 1)
        return self.unpatchify(patches)


class FunctionalNoiseMaskedLatentTransition(MaskedLatentTransition):
    """Predict an O2 latent increment conditional on one global noise draw."""

    def __init__(
        self,
        *,
        context_frames: int,
        config: O2ViTConfig = O2ViTConfig(),
        noise_config: FunctionalNoiseConfig = FunctionalNoiseConfig(),
    ) -> None:
        nn.Module.__init__(self)
        if context_frames not in (1, 2):
            raise ValueError("functional transition supports one or two context frames")
        self.context_frames = int(context_frames)
        self.config = config
        self.noise_config = noise_config
        self.backbone = FunctionalNoiseViT(
            config.latent_channels,
            config.latent_channels,
            condition_channels=1,
            noise_features=noise_config.embedded_noise_features,
            adapter_last_weight_multiplier=(
                noise_config.adapter_last_weight_multiplier
            ),
            hidden_channels=config.hidden_channels,
            hidden_blocks=config.transformer_blocks,
            attention_heads=config.attention_heads,
            ffn_factor=config.ffn_factor,
            spatial=4,
            patch_size=(1, *config.latent_patch),
            qk_norm=config.qk_normalization,
            rope=config.rope,
            dropout=config.dropout,
            checkpointing=config.activation_checkpointing,
        )
        self.noise_embedding = FunctionalNoiseEmbedding(
            noise_config.raw_noise_features,
            noise_config.embedded_noise_features,
        )

    @staticmethod
    def _is_noise_key(key: str) -> bool:
        return key.startswith("noise_embedding.") or ".noise_adapter." in key

    def load_deterministic_state(
        self,
        parent_state: Mapping[str, Tensor],
    ) -> DeterministicLoadAudit:
        """Load all and only shape-matched deterministic keys, failing closed."""

        child_state = self.state_dict()
        parent_keys = set(parent_state)
        noise_keys = {key for key in child_state if self._is_noise_key(key)}
        deterministic_keys = set(child_state) - noise_keys
        unexpected = tuple(sorted(parent_keys - deterministic_keys))
        missing = tuple(sorted(deterministic_keys - parent_keys))
        shape_mismatches = sorted(
            key
            for key in parent_keys & deterministic_keys
            if tuple(parent_state[key].shape) != tuple(child_state[key].shape)
        )
        if shape_mismatches:
            raise ValueError(
                "deterministic parent shape mismatches: " + ", ".join(shape_mismatches)
            )
        audit = DeterministicLoadAudit(
            parent_key_count=len(parent_keys),
            deterministic_child_key_count=len(deterministic_keys),
            new_noise_keys=tuple(sorted(noise_keys)),
            unexpected_parent_keys=unexpected,
            missing_deterministic_keys=missing,
        )
        if not audit.passed:
            raise ValueError(f"deterministic parent load audit failed: {audit.to_record()}")
        incompatible = self.load_state_dict(dict(parent_state), strict=False)
        if set(incompatible.missing_keys) != noise_keys or incompatible.unexpected_keys:
            raise RuntimeError("PyTorch load result differs from deterministic key audit")
        return audit

    def forward(
        self,
        standardized_context: Tensor,
        raw_noise: Tensor | None = None,
    ) -> Tensor:
        trajectory, mask = self.masked_trajectory(standardized_context)
        channels_first = trajectory.permute(0, 2, 1, 3, 4, 5).contiguous()
        mask_channels_first = mask.permute(0, 2, 1, 3, 4, 5).contiguous()
        embedded = None
        if raw_noise is not None:
            expected = (
                standardized_context.shape[0],
                self.noise_config.raw_noise_features,
            )
            if raw_noise.shape != expected:
                raise ValueError(f"raw noise shape must be {expected}")
            if not torch.isfinite(raw_noise).all():
                raise ValueError("raw noise must be finite")
            embedded = self.noise_embedding(
                raw_noise.to(
                    device=standardized_context.device,
                    dtype=standardized_context.dtype,
                )
            )
        predicted = self.backbone(
            channels_first,
            mask_channels_first,
            noise=embedded,
        )
        return predicted[:, :, -1]


class C5PFunctionalNoiseOneStepModel(C5POneStepModel):
    """Frozen codec plus an H1 functional-noise transition."""

    transition: FunctionalNoiseMaskedLatentTransition

    def __init__(
        self,
        *,
        codec: nn.Module,
        transition: FunctionalNoiseMaskedLatentTransition,
        latent_mean: Tensor,
        latent_standard_deviation: Tensor,
    ) -> None:
        super().__init__(
            codec=codec,
            transition=transition,
            latent_mean=latent_mean,
            latent_standard_deviation=latent_standard_deviation,
        )

    @property
    def raw_noise_features(self) -> int:
        return self.transition.noise_config.raw_noise_features

    def standardized_latent_members(
        self,
        context: Tensor,
        raw_noise: Tensor,
    ) -> Tensor:
        """Return `[batch,member,latent_channel,x,y,z]` before unnormalizing."""

        if raw_noise.ndim != 3:
            raise ValueError("raw noise must be [batch,member,raw_noise_features]")
        batch, members, features = raw_noise.shape
        if batch != context.shape[0] or features != self.raw_noise_features:
            raise ValueError("raw noise batch or feature count differs")
        if members < 1:
            raise ValueError("ensemble size must be positive")
        standardized = self.encode_context(context)
        expanded = standardized[:, None].expand(
            batch,
            members,
            *standardized.shape[1:],
        )
        expanded = expanded.reshape(batch * members, *standardized.shape[1:])
        increments = self.transition(
            expanded,
            raw_noise.reshape(batch * members, features),
        )
        forecast = expanded[:, -1] + increments
        return forecast.reshape(batch, members, *forecast.shape[1:])

    def predict_with_noise(
        self,
        context: Tensor,
        raw_noise: Tensor,
        *,
        horizon: int = 1,
    ) -> Tensor:
        """Predict with an explicit bank and canonical forecast axes."""

        if int(horizon) != 1:
            raise ValueError("B3 is authorized only for a one-step horizon")
        standardized = self.standardized_latent_members(context, raw_noise)
        batch, members = standardized.shape[:2]
        flattened = standardized.reshape(batch * members, *standardized.shape[2:])
        mean = self.latent_mean[:, 0]
        standard_deviation = self.latent_standard_deviation[:, 0]
        latent = flattened * standard_deviation + mean
        decoded = self.codec.decode(latent)
        decoded = decoded.reshape(batch, members, *decoded.shape[1:])
        return decoded[:, :, None]

    def forward(
        self,
        context: Tensor,
        raw_noise: Tensor | None = None,
    ) -> Tensor:
        if raw_noise is None:
            return super().forward(context)
        if raw_noise.ndim != 2:
            raise ValueError("single-member raw noise must be [batch,features]")
        return self.predict_with_noise(context, raw_noise[:, None])[:, 0, 0]

    def predict(self, context: Tensor, horizon: int, ensemble_size: int) -> Tensor:
        """Sample global noise and return canonical one-step ensemble axes."""

        if int(horizon) != 1:
            raise ValueError("B3 is authorized only for a one-step horizon")
        members = int(ensemble_size)
        if members <= 0:
            raise ValueError("ensemble size must be positive")
        raw_noise = torch.randn(
            context.shape[0],
            members,
            self.raw_noise_features,
            device=context.device,
            dtype=context.dtype,
        )
        return self.predict_with_noise(context, raw_noise, horizon=1)


@dataclass(frozen=True)
class FairCRPSResult:
    """Equal-channel fair CRPS and its accuracy/spread decomposition."""

    total: Tensor
    per_channel: Tensor
    accuracy_per_channel: Tensor
    spread_per_channel: Tensor


def fair_crps(
    predictions: Tensor,
    target: Tensor,
) -> FairCRPSResult:
    """Finite-ensemble fair CRPS for `[B,M,C,...]` predictions.

    Pairwise differences are accumulated over unordered pairs to avoid
    materializing a `[B,M,M,C,...]` tensor for 3D volumes.
    """

    if predictions.ndim < 4:
        raise ValueError("predictions must be [batch,member,channel,spatial...]")
    expected_target = (predictions.shape[0], *predictions.shape[2:])
    if target.shape != expected_target:
        raise ValueError(f"target shape must be {expected_target}")
    members = int(predictions.shape[1])
    if members < 2:
        raise ValueError("fair CRPS requires at least two ensemble members")
    if not torch.isfinite(predictions).all() or not torch.isfinite(target).all():
        raise ValueError("fair CRPS inputs must be finite")

    accuracy_pointwise = (predictions - target[:, None]).abs().mean(dim=1)
    pairwise_sum = torch.zeros_like(accuracy_pointwise)
    for left in range(members):
        for right in range(left + 1, members):
            pairwise_sum = pairwise_sum + (
                predictions[:, left] - predictions[:, right]
            ).abs()
    spread_pointwise = pairwise_sum / (members * (members - 1))
    score_pointwise = accuracy_pointwise - spread_pointwise
    reduction_axes = (0, *range(2, score_pointwise.ndim))
    per_channel = score_pointwise.mean(dim=reduction_axes)
    accuracy_per_channel = accuracy_pointwise.mean(dim=reduction_axes)
    spread_per_channel = spread_pointwise.mean(dim=reduction_axes)
    return FairCRPSResult(
        total=per_channel.mean(),
        per_channel=per_channel,
        accuracy_per_channel=accuracy_per_channel,
        spread_per_channel=spread_per_channel,
    )
