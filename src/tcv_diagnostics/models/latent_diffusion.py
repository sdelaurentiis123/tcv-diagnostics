"""Matched C5P masked latent diffusion for the prospective Paper 0 B2 arm.

The diffusion equations follow ``lola/diffusion.py`` at LOLA commit
``21a4354b327e6e5ee06da5075ba3bd1dd88c61f1``. The production sampler is the
version-locked Azula 0.3.1 Adams-Bashforth implementation. Physics-derived
quantities are deliberately absent from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
from typing import Any

import torch
from torch import Tensor, nn

from .modulated_vit import ModulatedViT, NoiseTimeEmbedding


@dataclass(frozen=True)
class LatentDiffusionViTConfig:
    """Frozen B2 architecture with smaller settings available to unit tests."""

    latent_channels: int = 32
    noise_time_features: int = 256
    hidden_channels: int = 512
    transformer_blocks: int = 16
    attention_heads: int = 4
    ffn_factor: int = 4
    latent_patch: tuple[int, int, int, int] = (1, 2, 2, 1)
    qk_normalization: bool = True
    rope: bool = True
    dropout: float = 0.05
    activation_checkpointing: bool = True
    condition_mask_channels: int = 1

    def __post_init__(self) -> None:
        if min(
            self.latent_channels,
            self.noise_time_features,
            self.hidden_channels,
            self.transformer_blocks,
            self.attention_heads,
            self.ffn_factor,
        ) <= 0:
            raise ValueError("diffusion architecture counts must be positive")
        if self.noise_time_features % 2 or self.hidden_channels % 2:
            raise ValueError("noise-time and hidden features must be even")
        if self.hidden_channels % self.attention_heads:
            raise ValueError("hidden channels must be divisible by attention heads")
        if len(self.latent_patch) != 4 or any(item <= 0 for item in self.latent_patch):
            raise ValueError("latent patch must contain four positive values")
        if self.condition_mask_channels != 1:
            raise ValueError("the frozen B2 model uses one nonredundant mask channel")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["latent_patch"] = list(self.latent_patch)
        record.update(
            {
                "family": "LOLA_style_masked_latent_diffusion",
                "diffusion_time_is_physical_time": False,
                "physics_derived_loss_allowed": False,
            }
        )
        return record


class LogLogitSchedule(nn.Module):
    """LOLA variance-exploding log-logit schedule with alpha equal to one."""

    def __init__(
        self,
        sigma_min: float = 1.0e-3,
        sigma_max: float = 1.0e3,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("noise scales must satisfy 0 < sigma_min < sigma_max")
        if scale <= 0.0:
            raise ValueError("log-logit scale must be positive")
        self.register_buffer(
            "t_min", torch.as_tensor(sigma_min / (1.0 + sigma_min), dtype=torch.float32)
        )
        self.register_buffer(
            "t_max", torch.as_tensor(sigma_max / (1.0 + sigma_max), dtype=torch.float32)
        )
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("shift", torch.as_tensor(shift, dtype=torch.float32))
        self.sigma_min_value = float(sigma_min)
        self.sigma_max_value = float(sigma_max)

    def alpha(self, time: Tensor) -> Tensor:
        return torch.ones_like(time)

    def sigma(self, time: Tensor) -> Tensor:
        values = torch.as_tensor(time, device=self.t_min.device)
        if not torch.all(torch.isfinite(values)):
            raise ValueError("noise times must be finite")
        if torch.any(values < 0.0) or torch.any(values > 1.0):
            raise ValueError("noise times must lie in [0,1]")
        probability = values * (self.t_max - self.t_min) + self.t_min
        return torch.exp(self.scale * torch.logit(probability) + self.shift)

    def forward(self, time: Tensor) -> tuple[Tensor, Tensor]:
        return self.alpha(time), self.sigma(time)


@dataclass
class DenoiserPrediction:
    """Minimal Gaussian mean/variance contract consumed by Azula 0.3.1."""

    mean: Tensor
    var: Tensor


class NoiseConditionedBackbone(nn.Module):
    """Embed scalar diffusion time and modulate every LOLA ViT block."""

    def __init__(
        self,
        config: LatentDiffusionViTConfig = LatentDiffusionViTConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.time_embedding = NoiseTimeEmbedding(config.noise_time_features)
        self.vit = ModulatedViT(
            config.latent_channels,
            config.latent_channels,
            condition_channels=config.condition_mask_channels,
            modulation_features=config.noise_time_features,
            hidden_channels=config.hidden_channels,
            hidden_blocks=config.transformer_blocks,
            attention_heads=config.attention_heads,
            ffn_factor=config.ffn_factor,
            spatial=4,
            patch_size=config.latent_patch,
            qk_norm=config.qk_normalization,
            rope=config.rope,
            dropout=config.dropout,
            checkpointing=config.activation_checkpointing,
        )

    def forward(self, noisy: Tensor, noise_time: Tensor, mask: Tensor) -> Tensor:
        if noise_time.ndim != 1 or noise_time.shape[0] != noisy.shape[0]:
            raise ValueError("expanded noise time must have one scalar per batch item")
        modulation = self.time_embedding(noise_time)
        return self.vit(noisy, modulation, mask)


def _batch_time(time: Tensor, *, batch: int, device: torch.device) -> Tensor:
    values = torch.as_tensor(time, device=device)
    if values.ndim == 0:
        return values.expand(batch)
    if values.shape == (1,):
        return values.expand(batch)
    if values.shape != (batch,):
        raise ValueError(f"noise time must be scalar or [{batch}], got {values.shape}")
    return values


def _expand_scalar_by_sample(values: Tensor, target: Tensor) -> Tensor:
    expanded = values
    while expanded.ndim < target.ndim:
        expanded = expanded[..., None]
    return expanded


class MaskedEDMDenoiser(nn.Module):
    """LOLA EDM-preconditioned Gaussian denoiser with a binary mask input."""

    def __init__(
        self,
        backbone: nn.Module,
        schedule: LogLogitSchedule | None = None,
        *,
        noise_embedding_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if noise_embedding_scale <= 0.0:
            raise ValueError("noise embedding scale must be positive")
        self.backbone = backbone
        self.schedule = LogLogitSchedule() if schedule is None else schedule
        self.noise_embedding_scale = float(noise_embedding_scale)

    def forward(self, noisy: Tensor, time: Tensor, mask: Tensor) -> DenoiserPrediction:
        if noisy.ndim != 6:
            raise ValueError("noisy latent trajectory must be [batch,C,time,x,y,z]")
        expected_mask = (noisy.shape[0], 1, *noisy.shape[2:])
        if mask.shape != expected_mask:
            raise ValueError(f"mask shape must be {expected_mask}")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("conditioning mask must be binary")

        batch_time = _batch_time(time, batch=noisy.shape[0], device=noisy.device)
        alpha, sigma = self.schedule(batch_time)
        alpha_x = _expand_scalar_by_sample(alpha, noisy)
        sigma_x = _expand_scalar_by_sample(sigma, noisy)

        c_in = torch.rsqrt(alpha_x.square() + sigma_x.square())
        c_out = sigma_x * c_in
        c_skip = alpha_x / (alpha_x.square() + sigma_x.square())
        noise_time = self.noise_embedding_scale * torch.log(sigma / alpha)
        residual = self.backbone(c_in * noisy, noise_time, mask.to(noisy.dtype))
        mean = c_skip * noisy + c_out * residual
        variance = sigma_x.square() / (alpha_x.square() + sigma_x.square())
        return DenoiserPrediction(mean=mean, var=variance)


class ConditionedMaskedDenoiser(nn.Module):
    """Clamp exact context values at every reverse-diffusion evaluation."""

    def __init__(self, denoiser: MaskedEDMDenoiser, observed: Tensor, mask: Tensor) -> None:
        super().__init__()
        if observed.ndim != 6:
            raise ValueError("observed trajectory must be [batch,C,time,x,y,z]")
        if mask.shape != (observed.shape[0], 1, *observed.shape[2:]):
            raise ValueError("conditioning mask shape differs from observed trajectory")
        if mask.dtype is not torch.bool:
            raise ValueError("sampling mask must be boolean")
        self.denoiser = denoiser
        self.register_buffer("observed", observed)
        self.register_buffer("mask", mask)

    @property
    def schedule(self) -> LogLogitSchedule:
        return self.denoiser.schedule

    def forward(self, noisy: Tensor, time: Tensor, **_: Any) -> DenoiserPrediction:
        if noisy.shape != self.observed.shape:
            raise ValueError("sample and observed trajectory shapes differ")
        batch_time = _batch_time(time, batch=noisy.shape[0], device=noisy.device)
        alpha, sigma = self.schedule(batch_time)
        scale = torch.sqrt(alpha.square() + sigma.square())
        scale = _expand_scalar_by_sample(scale, noisy)
        expanded_mask = self.mask.expand_as(noisy)
        conditioned = torch.where(expanded_mask, scale * self.observed, noisy)
        prediction = self.denoiser(conditioned, batch_time, self.mask)
        return DenoiserPrediction(
            mean=torch.where(expanded_mask, self.observed, prediction.mean),
            var=prediction.var,
        )


@dataclass
class DenoisingLoss:
    """Complete and slot-separated element-weighted LOLA loss values."""

    complete: Tensor
    context: Tensor
    target: Tensor
    context_elements: int
    target_elements: int

    def recomposed(self) -> Tensor:
        total = self.context_elements + self.target_elements
        return (
            self.context * self.context_elements + self.target * self.target_elements
        ) / total


def masked_denoising_loss(
    denoiser: MaskedEDMDenoiser,
    clean: Tensor,
    mask: Tensor,
    *,
    noise_time: Tensor | None = None,
    noise: Tensor | None = None,
) -> DenoisingLoss:
    """Evaluate original LOLA EDM loss with explicit context/target accounting."""

    if clean.ndim != 6:
        raise ValueError("clean trajectory must be [batch,C,time,x,y,z]")
    if mask.shape != (clean.shape[0], 1, *clean.shape[2:]):
        raise ValueError("mask shape differs from clean trajectory")
    if mask.dtype is not torch.bool:
        raise ValueError("loss mask must be boolean")
    if noise_time is None:
        noise_time = torch.rand(clean.shape[0], device=clean.device)
    else:
        noise_time = _batch_time(
            noise_time, batch=clean.shape[0], device=clean.device
        )
    if noise is None:
        noise = torch.randn_like(clean)
    elif noise.shape != clean.shape:
        raise ValueError("noise shape differs from clean trajectory")

    alpha, sigma = denoiser.schedule(noise_time)
    alpha_x = _expand_scalar_by_sample(alpha, clean)
    sigma_x = _expand_scalar_by_sample(sigma, clean)
    noisy = alpha_x * clean + sigma_x * noise
    expanded_mask = mask.expand_as(clean)
    context_scale = torch.sqrt(alpha_x.square() + sigma_x.square())
    noisy = torch.where(expanded_mask, context_scale * clean, noisy)

    prediction = denoiser(noisy, noise_time, mask)
    element_loss = (prediction.mean - clean).square() / prediction.var.detach()
    context_elements = int(expanded_mask.sum().item())
    target_elements = int((~expanded_mask).sum().item())
    if context_elements <= 0 or target_elements <= 0:
        raise ValueError("loss requires at least one context and one target element")
    complete = element_loss.mean()
    context = element_loss.masked_select(expanded_mask).mean()
    target = element_loss.masked_select(~expanded_mask).mean()
    return DenoisingLoss(
        complete=complete,
        context=context,
        target=target,
        context_elements=context_elements,
        target_elements=target_elements,
    )


def build_azula_ab_sampler(
    denoiser: ConditionedMaskedDenoiser,
    *,
    steps: int = 16,
    order: int = 3,
) -> nn.Module:
    """Construct the exact version-locked production sampler."""

    try:
        version = importlib.metadata.version("azula")
        from azula.sample import ABSampler
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError("B2 prediction requires Azula 0.3.1") from error
    if version != "0.3.1":
        raise RuntimeError(f"B2 requires Azula 0.3.1, found {version}")
    return ABSampler(
        denoiser,
        order=int(order),
        steps=int(steps),
        start=1.0,
        stop=0.0,
    )


class C5PLatentDiffusionModel(nn.Module):
    """Frozen codec plus a trainable B2 denoiser and canonical one-step sampler."""

    def __init__(
        self,
        *,
        codec: nn.Module,
        denoiser: MaskedEDMDenoiser,
        latent_mean: Tensor,
        latent_standard_deviation: Tensor,
        context_frames: int = 2,
        sampler_steps: int = 16,
        sampler_order: int = 3,
    ) -> None:
        super().__init__()
        if context_frames != 2:
            raise ValueError("the frozen B2 arm requires exactly two context frames")
        if sampler_steps <= 0 or sampler_order <= 0:
            raise ValueError("sampler steps and order must be positive")
        mean = torch.as_tensor(latent_mean, dtype=torch.float32).flatten()
        standard_deviation = torch.as_tensor(
            latent_standard_deviation, dtype=torch.float32
        ).flatten()
        latent_channels = int(denoiser.backbone.config.latent_channels)
        if mean.shape != (latent_channels,) or standard_deviation.shape != (
            latent_channels,
        ):
            raise ValueError("latent normalization must have one value per channel")
        if not torch.all(torch.isfinite(mean)):
            raise ValueError("latent means must be finite")
        if not torch.all(torch.isfinite(standard_deviation)) or not torch.all(
            standard_deviation > 0
        ):
            raise ValueError("latent standard deviations must be finite and positive")
        self.codec = codec
        self.denoiser = denoiser
        self.context_frames = int(context_frames)
        self.sampler_steps = int(sampler_steps)
        self.sampler_order = int(sampler_order)
        self.register_buffer("latent_mean", mean.reshape(1, -1, 1, 1, 1, 1))
        self.register_buffer(
            "latent_standard_deviation",
            standard_deviation.reshape(1, -1, 1, 1, 1, 1),
        )
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)
        self.codec.eval()

    def train(self, mode: bool = True) -> "C5PLatentDiffusionModel":
        super().train(mode)
        self.codec.eval()
        return self

    def _encode_fields(self, trajectory: Tensor) -> Tensor:
        if trajectory.ndim != 6:
            raise ValueError("field trajectory must be [batch,time,C,x,y,z]")
        batch, length, channels, x, y, z = trajectory.shape
        with torch.no_grad():
            latent = self.codec.encode(
                trajectory.reshape(batch * length, channels, x, y, z)
            )
        latent = latent.reshape(batch, length, *latent.shape[1:])
        latent = latent.permute(0, 2, 1, 3, 4, 5).contiguous()
        return (latent - self.latent_mean) / self.latent_standard_deviation

    def _decode_target(self, standardized: Tensor) -> Tensor:
        if standardized.ndim != 5:
            raise ValueError("target latent must be [batch,C,x,y,z]")
        mean = self.latent_mean[:, :, 0]
        standard_deviation = self.latent_standard_deviation[:, :, 0]
        latent = standardized * standard_deviation + mean
        return self.codec.decode(latent)

    def training_loss(
        self,
        context: Tensor,
        target: Tensor,
        *,
        noise_time: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> DenoisingLoss:
        if context.ndim != 6 or context.shape[1] != self.context_frames:
            raise ValueError("context must be [batch,2,C,x,y,z]")
        if target.shape != (context.shape[0], *context.shape[2:]):
            raise ValueError("target must be [batch,C,x,y,z] and match context")
        fields = torch.cat((context, target[:, None]), dim=1)
        clean = self._encode_fields(fields)
        mask = torch.zeros(
            (clean.shape[0], 1, clean.shape[2], *clean.shape[3:]),
            dtype=torch.bool,
            device=clean.device,
        )
        mask[:, :, : self.context_frames] = True
        return masked_denoising_loss(
            self.denoiser,
            clean,
            mask,
            noise_time=noise_time,
            noise=noise,
        )

    @torch.no_grad()
    def _sample_standardized_target(
        self,
        context: Tensor,
        *,
        ensemble_size: int,
    ) -> Tensor:
        """Sample standardized target latents for implementation-gate checks."""

        members = int(ensemble_size)
        if members <= 0:
            raise ValueError("ensemble size must be positive")
        if context.ndim != 6 or context.shape[1] != self.context_frames:
            raise ValueError("context must be [batch,2,C,x,y,z]")

        standardized_context = self._encode_fields(context)
        batch = standardized_context.shape[0]
        target_slot = torch.zeros_like(standardized_context[:, :, :1])
        observed = torch.cat((standardized_context, target_slot), dim=2)
        mask = torch.zeros(
            (batch, 1, observed.shape[2], *observed.shape[3:]),
            dtype=torch.bool,
            device=observed.device,
        )
        mask[:, :, : self.context_frames] = True

        observed = (
            observed[:, None]
            .expand(batch, members, *observed.shape[1:])
            .reshape(batch * members, *observed.shape[1:])
            .contiguous()
        )
        mask = (
            mask[:, None]
            .expand(batch, members, *mask.shape[1:])
            .reshape(batch * members, *mask.shape[1:])
            .contiguous()
        )
        conditioned = ConditionedMaskedDenoiser(self.denoiser, observed, mask)
        sampler = build_azula_ab_sampler(
            conditioned,
            steps=self.sampler_steps,
            order=self.sampler_order,
        ).to(observed.device)
        initial = sampler.init(
            observed.shape,
            dtype=observed.dtype,
            device=observed.device,
        )
        sampled = sampler(initial)
        target_latent = sampled[:, :, -1].reshape(
            batch,
            members,
            *sampled.shape[1:2],
            *sampled.shape[3:],
        )
        return target_latent

    @torch.no_grad()
    def predict(self, context: Tensor, horizon: int, ensemble_size: int) -> Tensor:
        """Return one-step fields as [batch,member,time,channel,x,y,z]."""

        if int(horizon) != 1:
            raise ValueError("B2 is authorized only for a one-step horizon")
        target_latent = self._sample_standardized_target(
            context,
            ensemble_size=ensemble_size,
        )
        batch, members = target_latent.shape[:2]
        flattened = target_latent.reshape(batch * members, *target_latent.shape[2:])
        decoded = self._decode_target(flattened)
        decoded = decoded.reshape(batch, members, *decoded.shape[1:])
        return decoded[:, :, None]
