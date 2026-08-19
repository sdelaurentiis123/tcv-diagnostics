"""Parent-initialized latent PDE-Refiner for the frozen Paper 0 B4 smoke.

The implementation follows the explicit-denoising construction in Lippe et
al., *PDE-Refiner* (NeurIPS 2023), while retaining the exact masked latent ViT
used by the deterministic C5P-H1 parent.  Existing O2 modules are deliberately
unchanged so the parent remains a hash-locked comparator.

Noise is applied to per-channel standardized codec latents.  It is not field-
space noise, and this module makes no claim that the two are equivalent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .o2 import C5POneStepModel, MaskedLatentTransition, O2ViTConfig
from .vit import SineEncoding, ViT, ViTBlock, regular_grid_coordinates


@dataclass(frozen=True)
class PDERefinerConfig:
    """Frozen B4 refinement conditioning and explicit noise schedule."""

    refinement_steps: int = 3
    minimum_noise_variance: float = 4.0e-7
    level_features: int = 256
    sine_omega: float = 1.0e3

    def __post_init__(self) -> None:
        if self.refinement_steps != 3:
            raise ValueError("the frozen B4 protocol requires three refinements")
        if self.minimum_noise_variance != 4.0e-7:
            raise ValueError("the frozen B4 minimum noise variance is 4e-7")
        if self.level_features != 256 or self.level_features % 2:
            raise ValueError("the frozen B4 level embedding has 256 features")
        if self.sine_omega != 1.0e3:
            raise ValueError("the frozen B4 sine omega is 1000")

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(range(self.refinement_steps + 1))

    @property
    def minimum_noise_standard_deviation(self) -> float:
        return math.sqrt(self.minimum_noise_variance)

    @property
    def standard_deviations(self) -> tuple[float, ...]:
        minimum = self.minimum_noise_standard_deviation
        return tuple(
            minimum ** (level / self.refinement_steps)
            for level in range(1, self.refinement_steps + 1)
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.update(
            {
                "levels": list(self.levels),
                "normalized_level_coordinates": [
                    level / self.refinement_steps for level in self.levels
                ],
                "minimum_noise_standard_deviation": (
                    self.minimum_noise_standard_deviation
                ),
                "standard_deviations": list(self.standard_deviations),
                "noise_coordinate": "per_channel_standardized_codec_latent",
                "network_calls_per_member": self.refinement_steps + 1,
            }
        )
        return record


def explicit_denoising_update(
    noisy_candidate: Tensor,
    predicted_noise: Tensor,
    sigma: float,
) -> Tensor:
    """Apply one explicit B4 denoising update ``z = z_tilde - sigma*eps``."""

    if noisy_candidate.shape != predicted_noise.shape:
        raise ValueError("noisy candidate and predicted noise shapes differ")
    scale = float(sigma)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("refinement standard deviation must be finite and positive")
    return noisy_candidate - scale * predicted_noise


@dataclass(frozen=True)
class RefinerLoadAudit:
    """Exact accounting for loading one deterministic O2 parent."""

    parent_key_count: int
    deterministic_child_key_count: int
    new_refinement_keys: tuple[str, ...]
    unexpected_parent_keys: tuple[str, ...]
    missing_deterministic_keys: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.parent_key_count == self.deterministic_child_key_count
            and bool(self.new_refinement_keys)
            and not self.unexpected_parent_keys
            and not self.missing_deterministic_keys
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "parent_key_count": self.parent_key_count,
            "deterministic_child_key_count": self.deterministic_child_key_count,
            "new_refinement_key_count": len(self.new_refinement_keys),
            "new_refinement_keys": list(self.new_refinement_keys),
            "unexpected_parent_keys": list(self.unexpected_parent_keys),
            "missing_deterministic_keys": list(self.missing_deterministic_keys),
            "passed": self.passed,
        }


class RefinementLevelEmbedding(nn.Module):
    """Embed one normalized discrete refinement level for every batch item."""

    def __init__(self, features: int = 256, *, omega: float = 1.0e3) -> None:
        super().__init__()
        self.encoding = SineEncoding(features, omega=omega)
        self.network = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
            nn.LayerNorm(features),
        )
        self.features = int(features)
        self.omega = float(omega)

    def forward(self, normalized_level: Tensor) -> Tensor:
        if normalized_level.ndim != 1:
            raise ValueError("normalized refinement level must have shape [batch]")
        if not torch.isfinite(normalized_level).all():
            raise ValueError("normalized refinement level must be finite")
        if torch.any(normalized_level < 0.0) or torch.any(normalized_level > 1.0):
            raise ValueError("normalized refinement level must lie in [0,1]")
        return self.network(self.encoding(normalized_level))


class PDERefinerViTBlock(ViTBlock):
    """One parent ViT block plus zero-initialized level conditioning."""

    def __init__(
        self,
        channels: int,
        *,
        level_features: int,
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
        if level_features <= 0:
            raise ValueError("refinement-level features must be positive")
        self.refinement_adapter = nn.Sequential(
            nn.Linear(level_features, level_features),
            nn.SiLU(),
            nn.Linear(level_features, 4 * channels),
        )
        nn.init.zeros_(self.refinement_adapter[-1].weight)
        nn.init.zeros_(self.refinement_adapter[-1].bias)
        self.channels = int(channels)
        self.level_features = int(level_features)

    def _forward_with_refinement(
        self,
        tokens: Tensor,
        coordinates: Tensor,
        skip: Tensor,
        level_embedding: Tensor,
    ) -> Tensor:
        if level_embedding.shape != (tokens.shape[0], self.level_features):
            raise ValueError(
                "level embedding must be [batch,refinement_level_features]"
            )
        theta = (
            None
            if self.theta is None
            else torch.einsum("ld,dc->lc", coordinates, self.theta)
        )
        modulation = self.refinement_adapter(level_embedding).reshape(
            tokens.shape[0], 4, 1, self.channels
        )
        parameters = self.ada_zero.reshape(1, 4, 1, self.channels) + modulation
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
        coordinates: Tensor,
        skip: Tensor,
        level_embedding: Tensor | None = None,
    ) -> Tensor:
        if level_embedding is None:
            return super().forward(tokens, coordinates, skip)
        if self.checkpointing and self.training:
            return checkpoint(
                self._forward_with_refinement,
                tokens,
                coordinates,
                skip,
                level_embedding,
                use_reentrant=False,
            )
        return self._forward_with_refinement(
            tokens,
            coordinates,
            skip,
            level_embedding,
        )


class PDERefinerViT(ViT):
    """The deterministic O2 ViT with a level adapter in every block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        level_features: int,
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
            PDERefinerViTBlock(
                hidden_channels,
                level_features=level_features,
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
        self.level_features = int(level_features)

    def forward(
        self,
        inputs: Tensor,
        condition: Tensor | None = None,
        *,
        level_embedding: Tensor | None = None,
    ) -> Tensor:
        if level_embedding is None:
            return super().forward(inputs, condition)
        if inputs.ndim != self.spatial + 2 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"ViT expects [batch,{self.in_channels},{self.spatial} axes], "
                f"got {inputs.shape}"
            )
        if level_embedding.shape != (inputs.shape[0], self.level_features):
            raise ValueError("refinement-level embedding shape differs")
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
            tokens = block(tokens, coordinates, skip, level_embedding)
        tokens = tokens.reshape(inputs.shape[0], *token_grid, self.hidden_channels)
        patches = self.output_projection(tokens).movedim(-1, 1)
        return self.unpatchify(patches)


class PDERefinerMaskedLatentTransition(MaskedLatentTransition):
    """Shared O2 transition conditioned on a provisional target and level."""

    def __init__(
        self,
        *,
        context_frames: int,
        config: O2ViTConfig = O2ViTConfig(),
        refiner_config: PDERefinerConfig = PDERefinerConfig(),
    ) -> None:
        nn.Module.__init__(self)
        if context_frames != 1:
            raise ValueError("the frozen B4 transition requires exactly one context")
        self.context_frames = int(context_frames)
        self.config = config
        self.refiner_config = refiner_config
        self.backbone = PDERefinerViT(
            config.latent_channels,
            config.latent_channels,
            condition_channels=1,
            level_features=refiner_config.level_features,
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
        self.level_embedding = RefinementLevelEmbedding(
            refiner_config.level_features,
            omega=refiner_config.sine_omega,
        )

    @staticmethod
    def _is_refinement_key(key: str) -> bool:
        return key.startswith("level_embedding.") or ".refinement_adapter." in key

    def load_deterministic_state(
        self,
        parent_state: Mapping[str, Tensor],
    ) -> RefinerLoadAudit:
        """Load every parent tensor and permit only new refinement keys."""

        child_state = self.state_dict()
        parent_keys = set(parent_state)
        refinement_keys = {
            key for key in child_state if self._is_refinement_key(key)
        }
        deterministic_keys = set(child_state) - refinement_keys
        unexpected = tuple(sorted(parent_keys - deterministic_keys))
        missing = tuple(sorted(deterministic_keys - parent_keys))
        shape_mismatches = sorted(
            key
            for key in parent_keys & deterministic_keys
            if tuple(parent_state[key].shape) != tuple(child_state[key].shape)
        )
        if shape_mismatches:
            raise ValueError(
                "deterministic parent shape mismatches: "
                + ", ".join(shape_mismatches)
            )
        audit = RefinerLoadAudit(
            parent_key_count=len(parent_keys),
            deterministic_child_key_count=len(deterministic_keys),
            new_refinement_keys=tuple(sorted(refinement_keys)),
            unexpected_parent_keys=unexpected,
            missing_deterministic_keys=missing,
        )
        if not audit.passed:
            raise ValueError(f"deterministic parent load audit failed: {audit.to_record()}")
        incompatible = self.load_state_dict(dict(parent_state), strict=False)
        if (
            set(incompatible.missing_keys) != refinement_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError("PyTorch load result differs from parent key audit")
        return audit

    def _levels(
        self,
        refinement_level: int | Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(refinement_level, int):
            levels = torch.full(
                (batch,),
                refinement_level,
                device=device,
                dtype=torch.int64,
            )
        else:
            if refinement_level.shape != (batch,):
                raise ValueError("refinement levels must have shape [batch]")
            if refinement_level.dtype == torch.bool:
                raise ValueError("refinement levels must be integer-valued")
            levels = refinement_level.to(device=device)
            if levels.is_floating_point():
                if not torch.equal(levels, levels.round()):
                    raise ValueError("refinement levels must be integer-valued")
                levels = levels.to(torch.int64)
            else:
                levels = levels.to(torch.int64)
        if torch.any(levels < 0) or torch.any(
            levels > self.refiner_config.refinement_steps
        ):
            raise ValueError("refinement level leaves the frozen range 0..3")
        return levels

    def provisional_trajectory(
        self,
        standardized_context: Tensor,
        provisional_target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if standardized_context.ndim != 6:
            raise ValueError("context must be [batch,time,channel,x,y,z]")
        if standardized_context.shape[1] != self.context_frames:
            raise ValueError("B4 context history differs")
        if standardized_context.shape[2] != self.config.latent_channels:
            raise ValueError("B4 context latent channels differ")
        expected = (
            standardized_context.shape[0],
            standardized_context.shape[2],
            *standardized_context.shape[3:],
        )
        if provisional_target.shape != expected:
            raise ValueError(f"provisional target shape must be {expected}")
        trajectory = torch.cat(
            (standardized_context, provisional_target[:, None]),
            dim=1,
        )
        known = torch.ones(
            (
                standardized_context.shape[0],
                self.context_frames,
                1,
                *standardized_context.shape[3:],
            ),
            device=standardized_context.device,
            dtype=standardized_context.dtype,
        )
        unknown = torch.zeros_like(known[:, :1])
        return trajectory, torch.cat((known, unknown), dim=1)

    def forward(
        self,
        standardized_context: Tensor,
        provisional_target: Tensor,
        refinement_level: int | Tensor,
    ) -> Tensor:
        trajectory, mask = self.provisional_trajectory(
            standardized_context,
            provisional_target,
        )
        levels = self._levels(
            refinement_level,
            batch=standardized_context.shape[0],
            device=standardized_context.device,
        )
        normalized = levels.to(dtype=standardized_context.dtype) / float(
            self.refiner_config.refinement_steps
        )
        embedded = self.level_embedding(normalized)
        channels_first = trajectory.permute(0, 2, 1, 3, 4, 5).contiguous()
        mask_channels_first = mask.permute(0, 2, 1, 3, 4, 5).contiguous()
        predicted = self.backbone(
            channels_first,
            mask_channels_first,
            level_embedding=embedded,
        )
        return predicted[:, :, -1]


class C5PPDERefinerOneStepModel(C5POneStepModel):
    """Frozen codec plus the B4 explicit latent denoising transition."""

    transition: PDERefinerMaskedLatentTransition

    def __init__(
        self,
        *,
        codec: nn.Module,
        transition: PDERefinerMaskedLatentTransition,
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
    def refinement_steps(self) -> int:
        return self.transition.refiner_config.refinement_steps

    @property
    def refinement_standard_deviations(self) -> tuple[float, ...]:
        return self.transition.refiner_config.standard_deviations

    def encode_target(self, target: Tensor) -> Tensor:
        if target.ndim != 5:
            raise ValueError("field target must be [batch,channel,x,y,z]")
        with torch.no_grad():
            latent = self.codec.encode(target)
        mean = self.latent_mean[:, 0]
        standard_deviation = self.latent_standard_deviation[:, 0]
        return (latent - mean) / standard_deviation

    def _standardized_stages(
        self,
        standardized_context: Tensor,
        refinement_noise: Tensor,
    ) -> Tensor:
        if refinement_noise.ndim != 7:
            raise ValueError(
                "refinement noise must be [batch,member,level,channel,x,y,z]"
            )
        batch, members, steps = refinement_noise.shape[:3]
        if batch != standardized_context.shape[0]:
            raise ValueError("refinement-noise batch differs from context")
        if members <= 0:
            raise ValueError("ensemble size must be positive")
        if steps != self.refinement_steps:
            raise ValueError("refinement-noise level count differs")
        expected_latent = (
            standardized_context.shape[2],
            *standardized_context.shape[3:],
        )
        if tuple(refinement_noise.shape[3:]) != expected_latent:
            raise ValueError(
                f"refinement-noise latent shape must be {expected_latent}"
            )
        if not refinement_noise.is_floating_point() or not torch.isfinite(
            refinement_noise
        ).all():
            raise ValueError("refinement noise must be finite floating point")
        refinement_noise = refinement_noise.to(
            device=standardized_context.device,
            dtype=standardized_context.dtype,
        )

        zero = torch.zeros_like(standardized_context[:, -1])
        increment = self.transition(standardized_context, zero, 0)
        initial = standardized_context[:, -1] + increment
        current = initial[:, None].expand(batch, members, *initial.shape[1:])
        stages = [current]

        expanded_context = standardized_context[:, None].expand(
            batch,
            members,
            *standardized_context.shape[1:],
        )
        flattened_context = expanded_context.reshape(
            batch * members,
            *standardized_context.shape[1:],
        )
        for level, sigma in enumerate(
            self.refinement_standard_deviations,
            start=1,
        ):
            noisy = current + sigma * refinement_noise[:, :, level - 1]
            predicted_noise = self.transition(
                flattened_context,
                noisy.reshape(batch * members, *noisy.shape[2:]),
                level,
            ).reshape_as(noisy)
            current = explicit_denoising_update(noisy, predicted_noise, sigma)
            stages.append(current)
        return torch.stack(stages, dim=2)

    def standardized_latent_stages(
        self,
        context: Tensor,
        refinement_noise: Tensor,
    ) -> Tensor:
        """Return ``[batch,member,level0..3,latent_channel,x,y,z]``."""

        standardized_context = self.encode_context(context)
        return self._standardized_stages(standardized_context, refinement_noise)

    def decoded_stages_with_noise(
        self,
        context: Tensor,
        refinement_noise: Tensor,
        *,
        horizon: int = 1,
    ) -> Tensor:
        """Decode all four stages as ``[B,M,level,C,x,y,z]``."""

        if int(horizon) != 1:
            raise ValueError("B4 is authorized only for a one-step horizon")
        standardized = self.standardized_latent_stages(context, refinement_noise)
        batch, members, levels = standardized.shape[:3]
        flattened = standardized.reshape(
            batch * members * levels,
            *standardized.shape[3:],
        )
        mean = self.latent_mean[:, 0]
        standard_deviation = self.latent_standard_deviation[:, 0]
        latent = flattened * standard_deviation + mean
        decoded = self.codec.decode(latent)
        return decoded.reshape(batch, members, levels, *decoded.shape[1:])

    def predict_with_noise(
        self,
        context: Tensor,
        refinement_noise: Tensor,
        *,
        horizon: int = 1,
    ) -> Tensor:
        """Return final-stage ``[batch,member,future_time,channel,x,y,z]``."""

        decoded = self.decoded_stages_with_noise(
            context,
            refinement_noise,
            horizon=horizon,
        )
        return decoded[:, :, -1, None]

    def forward(self, context: Tensor) -> Tensor:
        """Return the deterministic level-0 prediction without refinements."""

        standardized = self.encode_context(context)
        zero = torch.zeros_like(standardized[:, -1])
        increment = self.transition(standardized, zero, 0)
        forecast = standardized[:, -1] + increment
        mean = self.latent_mean[:, 0]
        standard_deviation = self.latent_standard_deviation[:, 0]
        return self.codec.decode(forecast * standard_deviation + mean)

    def predict(self, context: Tensor, horizon: int, ensemble_size: int) -> Tensor:
        """Sample independent full-latent noise for all three refinements."""

        if int(horizon) != 1:
            raise ValueError("B4 is authorized only for a one-step horizon")
        members = int(ensemble_size)
        if members <= 0:
            raise ValueError("ensemble size must be positive")
        standardized = self.encode_context(context)
        noise = torch.randn(
            standardized.shape[0],
            members,
            self.refinement_steps,
            standardized.shape[2],
            *standardized.shape[3:],
            device=standardized.device,
            dtype=standardized.dtype,
        )
        stages = self._standardized_stages(standardized, noise)
        final = stages[:, :, -1]
        flattened = final.reshape(
            final.shape[0] * final.shape[1],
            *final.shape[2:],
        )
        mean = self.latent_mean[:, 0]
        standard_deviation = self.latent_standard_deviation[:, 0]
        decoded = self.codec.decode(flattened * standard_deviation + mean)
        decoded = decoded.reshape(
            final.shape[0],
            final.shape[1],
            *decoded.shape[1:],
        )
        return decoded[:, :, None]
