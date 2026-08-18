"""Known-answer tests for the prospective B2 masked latent diffusion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tcv_diagnostics.models.latent_diffusion as diffusion_module
from tcv_diagnostics.models.latent_diffusion import (
    C5PLatentDiffusionModel,
    ConditionedMaskedDenoiser,
    LatentDiffusionViTConfig,
    LogLogitSchedule,
    MaskedEDMDenoiser,
    masked_denoising_loss,
)


class RecordingZeroBackbone(nn.Module):
    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(latent_channels=channels)
        self.last_noisy: torch.Tensor | None = None
        self.last_noise_time: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    def forward(
        self, noisy: torch.Tensor, noise_time: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        self.last_noisy = noisy.detach().clone()
        self.last_noise_time = noise_time.detach().clone()
        self.last_mask = mask.detach().clone()
        return torch.zeros_like(noisy)


class IdentityCodec(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            for channel in range(channels):
                self.projection.weight[channel, channel, 0, 0, 0] = 1.0

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


class FakeSampler(nn.Module):
    def __init__(self, denoiser: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser

    def init(self, shape: tuple[int, ...], **kwargs: object) -> torch.Tensor:
        return torch.randn(shape, **kwargs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _small_config() -> LatentDiffusionViTConfig:
    return LatentDiffusionViTConfig(
        latent_channels=2,
        noise_time_features=8,
        hidden_channels=16,
        transformer_blocks=1,
        attention_heads=2,
        ffn_factor=2,
        latent_patch=(1, 1, 1, 1),
        dropout=0.0,
        activation_checkpointing=False,
    )


def test_frozen_log_logit_schedule_known_answers() -> None:
    schedule = LogLogitSchedule()
    alpha, sigma = schedule(torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(alpha, torch.ones(3), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        sigma,
        torch.tensor([1.0e-3, 1.0, 1.0e3]),
        rtol=5.0e-5,
        atol=1.0e-7,
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        schedule(torch.tensor([-0.01]))


def test_edm_preconditioning_known_answer_at_unit_noise() -> None:
    backbone = RecordingZeroBackbone()
    denoiser = MaskedEDMDenoiser(backbone)
    noisy = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 2, 2, 2)
    mask = torch.zeros(1, 1, 1, 2, 2, 2)
    prediction = denoiser(noisy, torch.tensor([0.5]), mask)
    torch.testing.assert_close(prediction.mean, 0.5 * noisy, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(
        prediction.var,
        torch.full((1, 1, 1, 1, 1, 1), 0.5),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert backbone.last_noise_time is not None
    torch.testing.assert_close(
        backbone.last_noise_time, torch.zeros(1), rtol=0.0, atol=2.0e-6
    )


def test_masked_loss_scales_clean_context_and_accounts_for_every_element() -> None:
    backbone = RecordingZeroBackbone()
    denoiser = MaskedEDMDenoiser(backbone)
    clean = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 1, 1, 2) / 10
    noise = torch.full_like(clean, 0.25)
    mask = torch.zeros(1, 1, 3, 1, 1, 2, dtype=torch.bool)
    mask[:, :, :2] = True
    result = masked_denoising_loss(
        denoiser,
        clean,
        mask,
        noise_time=torch.tensor([0.5]),
        noise=noise,
    )
    assert backbone.last_noisy is not None
    # At sigma=1, the EDM input multiplier is 1/sqrt(2). The observed
    # context was first set to sqrt(2)*clean, so the backbone sees clean.
    expanded = mask.expand_as(clean)
    torch.testing.assert_close(
        backbone.last_noisy.masked_select(expanded),
        clean.masked_select(expanded),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    target_expected = (clean + noise) / (2.0**0.5)
    torch.testing.assert_close(
        backbone.last_noisy.masked_select(~expanded),
        target_expected.masked_select(~expanded),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.context_elements == 8
    assert result.target_elements == 4
    torch.testing.assert_close(result.complete, result.recomposed())
    assert all(
        torch.isfinite(value)
        for value in (result.complete, result.context, result.target)
    )


def test_conditioned_denoiser_returns_exact_clean_context_mean() -> None:
    denoiser = MaskedEDMDenoiser(RecordingZeroBackbone())
    observed = torch.randn(2, 2, 3, 1, 1, 2)
    mask = torch.zeros(2, 1, 3, 1, 1, 2, dtype=torch.bool)
    mask[:, :, :2] = True
    conditioned = ConditionedMaskedDenoiser(denoiser, observed, mask)
    prediction = conditioned(torch.randn_like(observed), torch.tensor(0.5))
    expanded = mask.expand_as(observed)
    torch.testing.assert_close(
        prediction.mean.masked_select(expanded),
        observed.masked_select(expanded),
        rtol=0.0,
        atol=0.0,
    )


def test_c5p_wrapper_training_and_canonical_ensemble_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = IdentityCodec(2)
    backbone = RecordingZeroBackbone(2)
    denoiser = MaskedEDMDenoiser(backbone)
    model = C5PLatentDiffusionModel(
        codec=codec,
        denoiser=denoiser,
        latent_mean=torch.zeros(2),
        latent_standard_deviation=torch.ones(2),
        sampler_steps=2,
        sampler_order=1,
    )
    assert not any(parameter.requires_grad for parameter in model.codec.parameters())
    model.train()
    assert not model.codec.training

    context = torch.randn(2, 2, 2, 2, 2, 2)
    target = torch.randn(2, 2, 2, 2, 2)
    losses = model.training_loss(
        context,
        target,
        noise_time=torch.tensor([0.25, 0.75]),
    )
    assert torch.isfinite(losses.complete)

    monkeypatch.setattr(
        diffusion_module,
        "build_azula_ab_sampler",
        lambda denoiser, **_: FakeSampler(denoiser),
    )
    model.eval()
    torch.manual_seed(91)
    forecast = model.predict(context, horizon=1, ensemble_size=3)
    assert forecast.shape == (2, 3, 1, 2, 2, 2, 2)
    assert torch.isfinite(forecast).all()
    assert not torch.equal(forecast[:, 0], forecast[:, 1])
    with pytest.raises(ValueError, match="one-step"):
        model.predict(context, horizon=2, ensemble_size=3)


def test_frozen_default_configuration_matches_the_manifest() -> None:
    config = LatentDiffusionViTConfig()
    assert config.latent_channels == 32
    assert config.noise_time_features == 256
    assert config.hidden_channels == 512
    assert config.transformer_blocks == 16
    assert config.attention_heads == 4
    assert config.ffn_factor == 4
    assert config.latent_patch == (1, 2, 2, 1)
    assert config.condition_mask_channels == 1
    assert config.activation_checkpointing is True


def test_azula_builder_fails_closed_when_dependency_is_unavailable() -> None:
    # The local unit-test environment intentionally need not include optional
    # model dependencies. Production jobs version-lock Azula before training.
    try:
        version = diffusion_module.importlib.metadata.version("azula")
    except diffusion_module.importlib.metadata.PackageNotFoundError:
        version = None
    if version is None:
        observed = torch.zeros(1, 2, 3, 1, 1, 1)
        mask = torch.zeros(1, 1, 3, 1, 1, 1, dtype=torch.bool)
        mask[:, :, :2] = True
        conditioned = ConditionedMaskedDenoiser(
            MaskedEDMDenoiser(RecordingZeroBackbone()), observed, mask
        )
        with pytest.raises(RuntimeError, match="Azula 0.3.1"):
            diffusion_module.build_azula_ab_sampler(conditioned)
