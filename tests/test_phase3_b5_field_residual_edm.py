"""Known-answer tests for the B5 joint field-residual EDM mechanics."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from tcv_diagnostics.models.field_residual_edm import (
    B5_FIELD_ORDER,
    B5_RESIDUAL_SCALES,
    FieldResidualUNet3D,
    FieldResidualUNetConfig,
    JointFieldResidualEDM,
    normalized_xy_coordinates,
    periodic_trilinear_upsample,
)


class ZeroBackbone(nn.Module):
    def forward(
        self,
        noisy: torch.Tensor,
        condition: torch.Tensor,
        noise_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        assert condition.shape == (noisy.shape[0], 10, *noisy.shape[2:])
        assert noise_coordinate.shape == (noisy.shape[0],)
        return torch.zeros_like(noisy)


def small_config() -> FieldResidualUNetConfig:
    return FieldResidualUNetConfig(
        base_channels=8,
        channel_multipliers=(1, 2),
        residual_blocks_per_resolution=1,
        noise_embedding_features=8,
        group_norm_maximum_groups=4,
    )


def test_B5_field_residual_default_config_matches_frozen_architecture() -> None:
    config = FieldResidualUNetConfig()
    assert config.level_channels == (32, 64, 128, 128)
    assert config.downsample_count == 3
    assert config.to_record()["name"] == "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI"
    assert config.to_record()["field_order"] == list(B5_FIELD_ORDER)
    assert config.to_record()["padding_by_axis"] == [
        "zeros",
        "zeros",
        "circular",
    ]
    assert config.to_record()["physics_derived_loss_allowed"] is False


def test_B5_static_coordinates_have_no_absolute_toroidal_position() -> None:
    reference = torch.empty(2, 5, 4, 3, 6)
    coordinates = normalized_xy_coordinates(reference)
    assert coordinates.shape == (2, 2, 4, 3, 6)
    torch.testing.assert_close(
        coordinates[:, :, :, :, 1:],
        coordinates[:, :, :, :, :-1],
        rtol=0.0,
        atol=0.0,
    )
    assert coordinates[:, 0].min() == -1.0
    assert coordinates[:, 0].max() == 1.0
    assert coordinates[:, 1].min() == -1.0
    assert coordinates[:, 1].max() == 1.0


def test_B5_periodic_trilinear_upsampling_crosses_toroidal_seam() -> None:
    values = torch.tensor([0.0, 4.0, 8.0, 12.0]).reshape(1, 1, 1, 1, 4)
    result = periodic_trilinear_upsample(values, (2, 2, 8))
    assert result.shape == (1, 1, 2, 2, 8)
    expected = torch.tensor([3.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 9.0])
    torch.testing.assert_close(result[0, 0, 0, 0], expected)
    shifted = periodic_trilinear_upsample(
        torch.roll(values, 1, dims=-1), (2, 2, 8)
    )
    torch.testing.assert_close(shifted, torch.roll(result, 2, dims=-1))


def test_B5_full_field_unet_shape_and_zero_initial_output() -> None:
    torch.manual_seed(4)
    model = FieldResidualUNet3D(small_config()).eval()
    noisy = torch.randn(2, 5, 8, 6, 10)
    condition = torch.randn(2, 10, 8, 6, 10)
    output = model(noisy, condition, torch.tensor([-1.0, 0.5]))
    assert output.shape == noisy.shape
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0.0, atol=0.0)


def test_B5_unet_is_toroidally_equivariant_at_total_stride_shift() -> None:
    torch.manual_seed(11)
    model = FieldResidualUNet3D(small_config()).eval()
    with torch.no_grad():
        model.output_convolution.weight.normal_(mean=0.0, std=0.01)
        if model.output_convolution.bias is not None:
            model.output_convolution.bias.zero_()
    noisy = torch.randn(1, 5, 8, 6, 10)
    condition = torch.randn(1, 10, 8, 6, 10)
    coordinate = torch.tensor([0.2])
    shift = 2  # one stride-2 level in this test configuration
    reference = model(noisy, condition, coordinate)
    shifted = model(
        torch.roll(noisy, shift, dims=-1),
        torch.roll(condition, shift, dims=-1),
        coordinate,
    )
    torch.testing.assert_close(
        shifted,
        torch.roll(reference, shift, dims=-1),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_B5_EDM_preconditioning_has_exact_zero_backbone_answer() -> None:
    edm = JointFieldResidualEDM(ZeroBackbone())
    noisy = torch.ones(2, 5, 2, 2, 2)
    condition = torch.zeros(2, 10, 2, 2, 2)
    prediction = edm.denoise(noisy, condition, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(
        prediction,
        torch.full_like(noisy, 0.2),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_B5_EDM_loss_has_exact_weighted_known_answer() -> None:
    edm = JointFieldResidualEDM(ZeroBackbone())
    clean = torch.zeros(1, 5, 2, 2, 2)
    condition = torch.zeros(1, 10, 2, 2, 2)
    result = edm.training_loss(
        clean,
        condition,
        sigma=torch.tensor([2.0]),
        noise=torch.ones_like(clean),
    )
    assert result.sigma_minimum == result.sigma_maximum == 2.0
    assert result.unweighted_mse.item() == pytest.approx(0.16, rel=1.0e-6)
    assert result.loss.item() == pytest.approx(0.2, rel=1.0e-6)


def test_B5_residual_scaling_round_trip_preserves_nonzero_mean() -> None:
    edm = JointFieldResidualEDM(ZeroBackbone())
    residual = torch.stack(
        [torch.full((2, 2, 2), float(index + 1)) for index in range(5)], dim=0
    )[None]
    normalized = edm.normalize_residual(residual)
    recovered = edm.denormalize_residual(normalized)
    torch.testing.assert_close(recovered, residual, rtol=1.0e-6, atol=1.0e-7)
    assert tuple(edm.residual_scales.flatten().tolist()) == pytest.approx(
        B5_RESIDUAL_SCALES,
        rel=1.0e-7,
        abs=0.0,
    )
    assert torch.all(normalized.mean(dim=(0, 2, 3, 4)) != 0.0)


def test_B5_Karras_schedule_is_monotone_and_has_exact_endpoints() -> None:
    schedule = JointFieldResidualEDM.sampling_schedule(
        steps=18,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert schedule.shape == (19,)
    assert schedule[0].item() == pytest.approx(80.0, rel=1.0e-6)
    assert schedule[-2].item() == pytest.approx(0.002, rel=1.0e-6)
    assert schedule[-1].item() == 0.0
    assert torch.all(schedule[:-1] > schedule[1:])


def test_B5_sampler_is_deterministic_finite_and_member_distinct() -> None:
    edm = JointFieldResidualEDM(ZeroBackbone()).eval()
    condition = torch.zeros(1, 10, 2, 2, 4)
    generator = torch.Generator().manual_seed(1701)
    initial = torch.randn(1, 2, 5, 2, 2, 4, generator=generator)
    first = edm.sample_normalized(
        condition,
        initial,
        steps=4,
        sigma_max=2.0,
        sigma_min=0.01,
        rho=3.0,
    )
    second = edm.sample_normalized(
        condition,
        initial.clone(),
        steps=4,
        sigma_max=2.0,
        sigma_min=0.01,
        rho=3.0,
    )
    assert first.shape == (1, 2, 5, 2, 2, 4)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert torch.sqrt(torch.mean((first[:, 0] - first[:, 1]).square())).item() > 0.0


def test_B5_composition_returns_canonical_axes_and_exact_values() -> None:
    edm = JointFieldResidualEDM(ZeroBackbone())
    mean = torch.full((1, 5, 2, 2, 2), 3.0)
    normalized = torch.ones(1, 2, 5, 2, 2, 2)
    result = edm.compose_fields(mean, normalized)
    assert result.shape == (1, 2, 1, 5, 2, 2, 2)
    expected = mean[:, None, None] + torch.tensor(B5_RESIDUAL_SCALES).reshape(
        1, 1, 1, 5, 1, 1, 1
    )
    torch.testing.assert_close(result, expected.expand_as(result))


def test_B5_model_rejects_wrong_channels_and_nonfinite_sigma() -> None:
    model = FieldResidualUNet3D(small_config())
    with pytest.raises(ValueError, match="noisy residual"):
        model(torch.zeros(1, 4, 4, 4, 4), torch.zeros(1, 10, 4, 4, 4), torch.zeros(1))
    edm = JointFieldResidualEDM(ZeroBackbone())
    with pytest.raises(ValueError, match="finite and positive"):
        edm.denoise(
            torch.zeros(1, 5, 2, 2, 2),
            torch.zeros(1, 10, 2, 2, 2),
            torch.tensor([math.nan]),
        )
