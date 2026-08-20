"""Known-answer and symmetry tests for ECRD mechanics."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tcv_diagnostics.models.ecrd import (
    DeepConditionalResidualUNet3D,
    ECRDTransition,
    ECRDUNetConfig,
    EquivariantResidualMeanHead3D,
    MultiscaleNoiseConfig,
    compose_multiscale_noise,
    symmetrized_h1_mean,
    xy_bilinear_upsample,
)


def tiny_config(*, arm: str = "ECRD") -> ECRDUNetConfig:
    equivariant = arm != "B5-Context"
    return ECRDUNetConfig(
        arm=arm,
        history_frames=2 if arm == "ECRD-History" else 1,
        base_channels=4,
        channel_multipliers=(1, 2),
        residual_blocks_per_resolution=1,
        conditioner_blocks_per_resolution=1,
        noise_embedding_features=8,
        group_norm_maximum_groups=2,
        preserve_toroidal_resolution=equivariant,
        mean_head=equivariant,
        multiscale_noise=equivariant,
        mean_head_channels=4,
        mean_head_blocks=1,
    )


def test_arm_contracts_and_records() -> None:
    context = tiny_config(arm="B5-Context")
    assert context.condition_channels == 10
    assert context.downsample_stride == (2, 2, 2)
    assert context.to_record()["physics_derived_loss_allowed"] is False
    history = tiny_config(arm="ECRD-History")
    assert history.condition_channels == 15
    assert history.downsample_stride == (2, 2, 1)
    with pytest.raises(ValueError, match="arm/history"):
        ECRDUNetConfig(arm="ECRD-History", history_frames=1)


def test_xy_upsample_never_resizes_z_and_commutes_with_roll() -> None:
    values = torch.randn(2, 3, 4, 3, 11)
    expected = xy_bilinear_upsample(values, (8, 6, 11))
    observed = xy_bilinear_upsample(torch.roll(values, 3, -1), (8, 6, 11))
    torch.testing.assert_close(observed, torch.roll(expected, 3, -1))
    with pytest.raises(ValueError, match="forbidden"):
        xy_bilinear_upsample(values, (8, 6, 22))


def test_multiscale_noise_known_components_and_variance_normalization() -> None:
    config = MultiscaleNoiseConfig(
        global_weight=1.0,
        mesoscale_weight=2.0,
        local_weight=3.0,
        mesoscale_xy=(2, 2),
    )
    global_noise = torch.ones(1, 2, 1, 1, 1)
    meso = torch.full((1, 2, 2, 2, 5), 2.0)
    local = torch.full((1, 2, 4, 4, 5), 3.0)
    result = compose_multiscale_noise(
        global_noise=global_noise,
        mesoscale_noise=meso,
        local_noise=local,
        config=config,
    )
    expected = (1.0 + 4.0 + 9.0) / math.sqrt(14.0)
    torch.testing.assert_close(result, torch.full_like(result, expected))
    assert config.to_record()["posthoc_spread_multiplier"] is False


@pytest.mark.parametrize("shift", [1, 3, 7])
def test_equivariant_generator_and_mean_head_commute_with_z_roll(shift: int) -> None:
    torch.manual_seed(3)
    config = tiny_config()
    generator = DeepConditionalResidualUNet3D(config).eval()
    mean = EquivariantResidualMeanHead3D(config).eval()
    noisy = torch.randn(1, 5, 8, 6, 12)
    condition = torch.randn(1, config.condition_channels, 8, 6, 12)
    sigma = torch.tensor([0.2])
    with torch.inference_mode():
        reference_generator = generator(noisy, condition, sigma)
        shifted_generator = generator(
            torch.roll(noisy, shift, -1),
            torch.roll(condition, shift, -1),
            sigma,
        )
        reference_mean = mean(condition)
        shifted_mean = mean(torch.roll(condition, shift, -1))
    torch.testing.assert_close(
        shifted_generator,
        torch.roll(reference_generator, shift, -1),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        shifted_mean,
        torch.roll(reference_mean, shift, -1),
        atol=2e-6,
        rtol=2e-6,
    )


def test_ecrd_loss_separates_mean_and_innovation_gradients() -> None:
    torch.manual_seed(5)
    config = tiny_config()
    model = ECRDTransition(config)
    target = torch.randn(1, 5, 8, 6, 12)
    condition = torch.randn(1, config.condition_channels, 8, 6, 12)
    sigma = torch.tensor([0.7])
    noise = torch.randn_like(target)
    result = model.training_loss(target, condition, sigma=sigma, noise=noise)
    assert result.loss.requires_grad
    assert result.edm_loss.requires_grad
    assert result.mean_mse.requires_grad
    torch.testing.assert_close(result.loss, result.edm_loss + result.mean_mse)
    result.loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.mean_head.parameters()
        if parameter.requires_grad
    )
    assert model.to_record()["physics_derived_loss_used"] is False


def test_ecrd_sampler_and_canonical_composition_shape() -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = ECRDTransition(config).eval()
    condition = torch.randn(1, config.condition_channels, 8, 6, 12)
    parent = torch.randn(1, 5, 8, 6, 12)
    initial = torch.randn(1, 2, 5, 8, 6, 12)
    sampled = model.sample_normalized(
        condition,
        initial,
        steps=2,
        sigma_max=1.0,
        sigma_min=0.1,
    )
    assert sampled.shape == initial.shape
    fields = model.compose_fields(parent, condition, sampled)
    assert fields.shape == (1, 2, 1, 5, 8, 6, 12)
    assert torch.isfinite(fields).all()


class _BiasedPhaseModel(nn.Module):
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # Artificial phase-dependent bias. Averaging the four prescribed
        # conjugations must reproduce their literal arithmetic definition.
        marker = context[..., 0].mean(dim=(-1, -2, -3), keepdim=True)
        return context[:, -1] + marker


def test_h1_symmetrization_is_literal_four_phase_average() -> None:
    model = _BiasedPhaseModel()
    context = torch.randn(1, 2, 5, 64, 32, 88)
    observed = symmetrized_h1_mean(model, context)
    terms = []
    for shift in range(4):
        value = model(torch.roll(context[:, -1:], shift, -1))
        terms.append(torch.roll(value, -shift, -1))
    expected = torch.stack(terms).mean(0)
    torch.testing.assert_close(observed, expected)


def test_ecrd_has_no_toroidal_stride() -> None:
    model = DeepConditionalResidualUNet3D(tiny_config())
    strides = [
        tuple(module.stride)
        for module in model.modules()
        if isinstance(module, nn.Conv3d)
    ]
    assert strides
    assert all(stride[-1] == 1 for stride in strides)
