"""Frozen C5P one-step deterministic latent transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .vit import ViT


@dataclass(frozen=True)
class O2ViTConfig:
    latent_channels: int = 32
    hidden_channels: int = 512
    transformer_blocks: int = 16
    attention_heads: int = 4
    ffn_factor: int = 4
    latent_patch: tuple[int, int, int] = (2, 2, 1)
    qk_normalization: bool = True
    rope: bool = True
    dropout: float = 0.05
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.latent_channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("model channel counts must be positive")
        if self.transformer_blocks <= 0 or self.attention_heads <= 0:
            raise ValueError("model block/head counts must be positive")
        if self.hidden_channels % self.attention_heads:
            raise ValueError("hidden channels must be divisible by attention heads")
        if len(self.latent_patch) != 3 or any(item <= 0 for item in self.latent_patch):
            raise ValueError("latent patch must contain three positive values")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["latent_patch"] = list(self.latent_patch)
        record.update(
            {
                "family": "deterministic_masked_latent_ViT_residual",
                "noise_features": 0,
                "target_slot_masked": True,
                "context_mask_channel": True,
                "prediction": "standardized_latent_increment",
            }
        )
        return record


class MaskedLatentTransition(nn.Module):
    """Predict a target-slot increment from one or two ordered latent frames."""

    def __init__(
        self,
        *,
        context_frames: int,
        config: O2ViTConfig = O2ViTConfig(),
    ) -> None:
        super().__init__()
        if context_frames not in (1, 2):
            raise ValueError("frozen O2 supports exactly one or two context frames")
        self.context_frames = int(context_frames)
        self.config = config
        self.backbone = ViT(
            config.latent_channels,
            config.latent_channels,
            condition_channels=1,
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

    def masked_trajectory(self, context: Tensor) -> tuple[Tensor, Tensor]:
        if context.ndim != 6:
            raise ValueError("context must be [batch,time,channel,x,y,z]")
        if context.shape[1] != self.context_frames:
            raise ValueError(
                f"expected {self.context_frames} context frames, got {context.shape[1]}"
            )
        if context.shape[2] != self.config.latent_channels:
            raise ValueError("context latent channel count differs")
        target = torch.zeros_like(context[:, :1])
        trajectory = torch.cat((context, target), dim=1)
        known = torch.ones(
            (context.shape[0], self.context_frames, 1, *context.shape[3:]),
            device=context.device,
            dtype=context.dtype,
        )
        unknown = torch.zeros_like(known[:, :1])
        mask = torch.cat((known, unknown), dim=1)
        return trajectory, mask

    def forward(self, standardized_context: Tensor) -> Tensor:
        trajectory, mask = self.masked_trajectory(standardized_context)
        channels_first = trajectory.permute(0, 2, 1, 3, 4, 5).contiguous()
        mask_channels_first = mask.permute(0, 2, 1, 3, 4, 5).contiguous()
        predicted = self.backbone(channels_first, mask_channels_first)
        return predicted[:, :, -1]


class C5POneStepModel(nn.Module):
    """Frozen codec plus trainable O2 transition with canonical prediction axes."""

    def __init__(
        self,
        *,
        codec: nn.Module,
        transition: MaskedLatentTransition,
        latent_mean: Tensor,
        latent_standard_deviation: Tensor,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(latent_mean, dtype=torch.float32).flatten()
        standard_deviation = torch.as_tensor(
            latent_standard_deviation,
            dtype=torch.float32,
        ).flatten()
        channels = transition.config.latent_channels
        if mean.shape != (channels,) or standard_deviation.shape != (channels,):
            raise ValueError("latent normalization must have one value per channel")
        if not torch.all(torch.isfinite(mean)):
            raise ValueError("latent means must be finite")
        if not torch.all(torch.isfinite(standard_deviation)) or not torch.all(
            standard_deviation > 0
        ):
            raise ValueError("latent standard deviations must be finite and positive")
        self.codec = codec
        self.transition = transition
        self.register_buffer("latent_mean", mean.reshape(1, 1, channels, 1, 1, 1))
        self.register_buffer(
            "latent_standard_deviation",
            standard_deviation.reshape(1, 1, channels, 1, 1, 1),
        )
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)
        self.codec.eval()

    @property
    def context_frames(self) -> int:
        return self.transition.context_frames

    def train(self, mode: bool = True) -> "C5POneStepModel":
        super().train(mode)
        self.codec.eval()
        return self

    def encode_context(self, context: Tensor) -> Tensor:
        if context.ndim != 6 or context.shape[1] != self.context_frames:
            raise ValueError("field context must be [batch,frozen_history,C,x,y,z]")
        batch, history, channels, x, y, z = context.shape
        with torch.no_grad():
            latent = self.codec.encode(
                context.reshape(batch * history, channels, x, y, z)
            )
        latent = latent.reshape(batch, history, *latent.shape[1:])
        return (latent - self.latent_mean) / self.latent_standard_deviation

    def forward(self, context: Tensor) -> Tensor:
        standardized = self.encode_context(context)
        increment = self.transition(standardized)
        forecast_standardized = standardized[:, -1] + increment
        mean = self.latent_mean[:, 0]
        standard_deviation = self.latent_standard_deviation[:, 0]
        forecast_latent = forecast_standardized * standard_deviation + mean
        return self.codec.decode(forecast_latent)

    def predict(self, context: Tensor, horizon: int, ensemble_size: int) -> Tensor:
        """Return ``[batch,member,future_time,channel,x,y,z]``."""

        if int(horizon) != 1:
            raise ValueError("O2 is authorized only for a one-step horizon")
        if int(ensemble_size) <= 0:
            raise ValueError("ensemble size must be positive")
        forecast = self(context)
        return forecast[:, None, None].expand(
            forecast.shape[0],
            int(ensemble_size),
            1,
            *forecast.shape[1:],
        )
