from __future__ import annotations

import math

import pytest
import torch

from tcv_diagnostics.models.persistent_global_local import (
    PersistentGlobalLocalConfig,
    PersistentGlobalLocalDenoiser3D,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
    sample_persistent_global_local_noise,
    toroidal_highpass,
    toroidal_lowpass,
)


def tiny_config() -> PersistentGlobalLocalConfig:
    return PersistentGlobalLocalConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        residual_blocks_per_resolution=1,
        global_channels=4,
        global_pool_xy=(2, 2),
        low_mode_maximum=2,
        noise_embedding_features=16,
        group_norm_maximum_groups=4,
    )


def tiny_noise_config() -> PersistentNoiseConfig:
    return PersistentNoiseConfig(
        global_pool_xy=(2, 2),
        low_mode_maximum=2,
    )


def trajectories(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(501)
    noisy = torch.randn((batch, 4, 5, 8, 8, 12), generator=generator)
    current = torch.randn((batch, 5, 8, 8, 12), generator=generator)
    mean = torch.randn((batch, 4, 5, 8, 8, 12), generator=generator)
    return noisy, current, mean


def test_toroidal_low_high_partition_recovers_known_signal() -> None:
    z = torch.arange(24, dtype=torch.float32)
    low = torch.cos(2.0 * math.pi * 2.0 * z / 24.0)
    high = 0.4 * torch.sin(2.0 * math.pi * 7.0 * z / 24.0)
    values = (low + high).reshape(1, 1, 1, 1, 24)
    observed_low = toroidal_lowpass(values, maximum_mode=3)
    observed_high = toroidal_highpass(values, maximum_mode=3)
    assert torch.allclose(observed_low, low.reshape_as(values), atol=2.0e-6)
    assert torch.allclose(observed_high, high.reshape_as(values), atol=2.0e-6)
    assert torch.allclose(observed_low + observed_high, values, atol=1.0e-7)


def test_persistent_noise_has_shared_global_and_independent_local_time() -> None:
    reference = torch.zeros((2, 4, 5, 8, 8, 12))
    sample = sample_persistent_global_local_noise(
        reference,
        config=tiny_noise_config(),
        generator=torch.Generator().manual_seed(90),
    )
    assert sample.total.shape == reference.shape
    assert torch.equal(sample.global_component[:, 0], sample.global_component[:, 3])
    assert not torch.equal(sample.local_component[:, 0], sample.local_component[:, 3])
    assert torch.max(
        torch.abs(
            toroidal_highpass(sample.global_component, maximum_mode=2)
        )
    ) < 2.0e-6
    assert torch.max(
        torch.abs(toroidal_lowpass(sample.local_component, maximum_mode=2))
    ) < 2.0e-6
    global_rms = sample.global_component.float().square().mean((-3, -2, -1)).sqrt()
    local_rms = sample.local_component.float().square().mean((-3, -2, -1)).sqrt()
    assert torch.allclose(global_rms, torch.ones_like(global_rms), atol=2.0e-5)
    assert torch.allclose(local_rms, torch.ones_like(local_rms), atol=2.0e-5)


def test_configuration_records_physical_axis_contract() -> None:
    record = PersistentGlobalLocalConfig().to_record()
    assert record["joint_future_frames"] == 4
    assert record["field_order"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert record["padding_xyz"] == ["zeros", "zeros", "circular"]
    assert record["toroidal_downsampling"] is False
    assert record["global_recurrence"] == "ConvGRU_over_future_time"
    assert record["physics_derived_loss"] is False


def test_denoiser_shape_gradients_and_nontrivial_shift_equivariance() -> None:
    config = tiny_config()
    model = PersistentGlobalLocalDenoiser3D(config)
    torch.manual_seed(91)
    with torch.no_grad():
        model.global_stream.output_projection.weight.normal_(0.0, 0.02)
        model.local_stream.output_projection.weight.normal_(0.0, 0.02)
    noisy, current, mean = trajectories()
    sigma = torch.tensor([0.25])
    output = model(noisy, current, mean, sigma)
    assert output.shape == noisy.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.eval()
    with torch.no_grad():
        reference = model(noisy, current, mean, sigma)
        for shift in (1, 3, 7):
            shifted = model(
                torch.roll(noisy, shift, -1),
                torch.roll(current, shift, -1),
                torch.roll(mean, shift, -1),
                sigma,
            )
            expected = torch.roll(reference, shift, -1)
            relative = torch.linalg.vector_norm(shifted - expected) / torch.linalg.vector_norm(
                expected
            ).clamp_min(1.0e-8)
            assert float(relative) < 2.0e-5


def test_edm_loss_backward_sampling_and_composition() -> None:
    config = tiny_config()
    scales = torch.linspace(0.2, 1.0, 20).reshape(4, 5)
    model = PersistentGlobalLocalEDM(
        config,
        residual_scales=scales,
        noise_config=tiny_noise_config(),
    )
    clean, current, mean = trajectories()
    noise = sample_persistent_global_local_noise(
        clean,
        config=tiny_noise_config(),
        generator=torch.Generator().manual_seed(92),
    ).total
    result = model.training_loss(
        clean,
        current,
        mean,
        sigma=torch.tensor([0.7]),
        noise=noise,
    )
    assert result.loss.ndim == 0 and torch.isfinite(result.loss)
    assert result.per_step_field_mse.shape == (4, 5)
    result.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    initial = torch.stack(
        [
            sample_persistent_global_local_noise(
                clean,
                config=tiny_noise_config(),
                generator=torch.Generator().manual_seed(seed),
            ).total
            for seed in (93, 94)
        ],
        dim=1,
    )
    model.eval()
    normalized = model.sample_normalized(
        current,
        mean,
        initial,
        steps=2,
        sigma_max=1.0,
        sigma_min=0.1,
    )
    assert normalized.shape == (1, 2, 4, 5, 8, 8, 12)
    fields = model.compose_fields(mean, normalized)
    assert fields.shape == normalized.shape
    assert torch.isfinite(fields).all()


def test_invalid_short_z_and_nondivisible_xy_are_rejected() -> None:
    config = tiny_noise_config()
    with pytest.raises(ValueError, match="divide"):
        sample_persistent_global_local_noise(
            torch.zeros((1, 4, 5, 7, 8, 12)), config=config
        )
    with pytest.raises(ValueError, match="too short"):
        sample_persistent_global_local_noise(
            torch.zeros((1, 4, 5, 8, 8, 4)), config=config
        )
