from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tcv_diagnostics.models.o2 import (
    C5POneStepModel,
    MaskedLatentTransition,
    O2ViTConfig,
)
from tcv_diagnostics.o2_training import (
    O2RunConfig,
    fit_latent_normalization,
    learning_rate_at_step,
    scale_accumulated_gradients,
    validation_loss,
)


class IdentityCodec(nn.Module):
    def encode(self, values):
        return values

    def decode(self, values):
        return values


class LatentFrameDataset:
    def __init__(self):
        self.frames = (0, 1)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        frame = self.frames[index]
        volume = np.stack(
            (
                np.full((2, 1, 1), frame, dtype=np.float32),
                np.full((2, 1, 1), 2 * frame + 1, dtype=np.float32),
            ),
            axis=0,
        )
        return {"volume": volume, "frame_index": np.int64(frame)}


class ValidationWindowDataset:
    target_frames = (498, 499, 500, 501)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = np.float32(index + 1)
        latest = np.full((5, 2, 1, 1), value, dtype=np.float32)
        return {
            "context": latest[None],
            "target": latest,
            "target_frame_index": np.int64(self.target_frames[index]),
        }


class LatestFrame(nn.Module):
    def forward(self, context):
        return context[:, -1]


def tiny_model_config():
    return O2ViTConfig(
        latent_channels=2,
        hidden_channels=8,
        transformer_blocks=1,
        attention_heads=2,
        ffn_factor=2,
        latent_patch=(1, 1, 1),
        qk_normalization=True,
        rope=True,
        dropout=0.0,
        activation_checkpointing=False,
    )


def test_full_and_smoke_budgets_encode_partial_accumulation_exactly():
    full = O2RunConfig.frozen(mode="full", arm="C5P-H2", seed=1701)
    assert full.context_frames == 2
    assert full.train_targets == tuple(range(2, 432))
    assert full.validation_targets == tuple(range(498, 624))
    assert full.latent_fit_frames == tuple(range(432))
    assert full.epochs == 200
    assert full.optimizer_steps_per_epoch == 27
    assert full.final_accumulation_count == 14
    assert full.total_optimizer_steps == 5400
    assert full.warmup_optimizer_steps == 270
    assert full.to_record()["physics_derived_loss_allowed"] is False

    smoke = O2RunConfig.frozen(mode="smoke", arm="C5P-H1", seed=1702)
    assert smoke.context_frames == 1
    assert smoke.epochs == 2
    assert len(smoke.train_targets) == 16
    assert len(smoke.validation_targets) == 4
    assert len(smoke.latent_fit_frames) == 16
    assert smoke.total_optimizer_steps == 2


def test_unfrozen_arm_seed_and_mode_are_rejected():
    with pytest.raises(ValueError):
        O2RunConfig.frozen(mode="full", arm="E6B-H1", seed=1701)
    with pytest.raises(ValueError):
        O2RunConfig.frozen(mode="full", arm="C5P-H1", seed=9)
    with pytest.raises(ValueError):
        O2RunConfig.frozen(mode="other", arm="C5P-H1", seed=1701)


def test_learning_rate_has_frozen_boundaries():
    config = O2RunConfig.frozen(mode="full", arm="C5P-H1", seed=1701)
    assert learning_rate_at_step(config, 1) == pytest.approx(
        config.learning_rate / config.warmup_optimizer_steps
    )
    assert learning_rate_at_step(config, config.warmup_optimizer_steps) == (
        config.learning_rate
    )
    assert learning_rate_at_step(config, config.total_optimizer_steps) == pytest.approx(
        config.minimum_learning_rate
    )
    with pytest.raises(ValueError):
        learning_rate_at_step(config, 0)


def test_final_partial_gradient_is_divided_by_actual_count():
    parameter = nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(32.0)
    scale_accumulated_gradients([parameter], 14)
    assert parameter.grad.item() == pytest.approx(32.0 / 14.0)
    with pytest.raises(ValueError):
        scale_accumulated_gradients([parameter], 0)


def test_latent_normalization_uses_training_frames_and_population_moments():
    result = fit_latent_normalization(
        codec=IdentityCodec(),
        dataset=LatentFrameDataset(),
        frames=(0, 1),
        codec_checkpoint_sha256="a" * 64,
        device=torch.device("cpu"),
        batch_size=1,
        expected_channels=2,
        scientific_authority=False,
    )
    np.testing.assert_allclose(result.mean, [0.5, 2.0])
    np.testing.assert_allclose(result.standard_deviation, [0.5, 1.0])
    assert result.sample_count_per_channel == 4
    assert result.fit_frames == (0, 2)
    assert result.to_record()["held_out_85606_read"] is False
    assert result.to_record()["scientific_authority"] is False


def test_validation_is_chronological_target_only_and_float64_accumulated():
    config = O2RunConfig.frozen(mode="smoke", arm="C5P-H1", seed=1701)
    aggregate, per_channel = validation_loss(
        LatestFrame(),
        ValidationWindowDataset(),
        config,
        torch.device("cpu"),
    )
    assert aggregate == 0.0
    assert per_channel == [0.0] * 5


def test_tiny_transition_overfits_a_known_one_step_increment():
    torch.manual_seed(1701)
    transition = MaskedLatentTransition(
        context_frames=1,
        config=tiny_model_config(),
    )
    model = C5POneStepModel(
        codec=IdentityCodec(),
        transition=transition,
        latent_mean=torch.zeros(2),
        latent_standard_deviation=torch.ones(2),
    )
    generator = torch.Generator().manual_seed(18)
    context = torch.randn(4, 1, 2, 2, 2, 2, generator=generator)
    target = context[:, -1] + 0.25
    optimizer = torch.optim.Adam(model.transition.parameters(), lr=2.0e-2)

    with torch.no_grad():
        initial = (model(context) - target).abs().mean().item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(context) - target).abs().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (model(context) - target).abs().mean().item()
    assert final < 0.25 * initial
