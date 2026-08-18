"""Known-answer tests for the bounded B2 optimization contract."""

from __future__ import annotations

import pytest
import torch

from tcv_diagnostics.b2_training import (
    B2RunConfig,
    FixedValidationNoiseBank,
    fixed_validation_perturbation,
    learning_rate_at_step,
    train_b2_smoke,
)


def test_b2_full_and_smoke_budgets_match_the_frozen_protocol() -> None:
    smoke = B2RunConfig.frozen(mode="smoke", seed=1701)
    assert smoke.context_frames == 2
    assert smoke.train_targets == tuple(range(2, 18))
    assert smoke.validation_targets == tuple(range(498, 502))
    assert smoke.latent_fit_frames == tuple(range(16))
    assert smoke.epochs == 2
    assert smoke.optimizer_steps_per_epoch == 1
    assert smoke.total_optimizer_steps == 2
    assert smoke.latent_trajectory_shape == (32, 3, 16, 8, 22)
    assert smoke.to_record()["physics_derived_loss_allowed"] is False
    assert smoke.to_record()["full_training_authorized"] is False

    full = B2RunConfig.frozen(mode="full", seed=1702)
    assert len(full.train_targets) == 430
    assert len(full.validation_targets) == 126
    assert full.epochs == 200
    assert full.optimizer_steps_per_epoch == 27
    assert full.total_optimizer_steps == 5400
    assert full.final_accumulation_count == 14


def test_b2_unfrozen_smoke_seed_mode_and_schedule_steps_are_rejected() -> None:
    with pytest.raises(ValueError, match="seed 1701"):
        B2RunConfig.frozen(mode="smoke", seed=1702)
    with pytest.raises(ValueError, match="unsupported"):
        B2RunConfig.frozen(mode="other", seed=1701)
    with pytest.raises(ValueError, match="one of"):
        B2RunConfig.frozen(mode="full", seed=9)
    config = B2RunConfig.frozen(mode="smoke", seed=1701)
    with pytest.raises(ValueError, match="outside"):
        learning_rate_at_step(config, 0)


def test_b2_cosine_schedule_starts_at_base_and_ends_at_zero() -> None:
    smoke = B2RunConfig.frozen(mode="smoke", seed=1701)
    assert learning_rate_at_step(smoke, 1) == smoke.learning_rate
    assert learning_rate_at_step(smoke, 2) == pytest.approx(0.0, abs=1.0e-20)
    full = B2RunConfig.frozen(mode="full", seed=1701)
    assert learning_rate_at_step(full, 1) == full.learning_rate
    assert learning_rate_at_step(full, full.total_optimizer_steps) == pytest.approx(
        0.0, abs=1.0e-20
    )


def test_fixed_validation_noise_is_exact_and_independent_of_model_rng() -> None:
    shape = (2, 3, 2, 1, 1)
    time_a, noise_a = fixed_validation_perturbation(
        target_frame=498,
        latent_trajectory_shape=shape,
    )
    torch.manual_seed(999)
    _ = torch.randn(100)
    time_b, noise_b = fixed_validation_perturbation(
        target_frame=498,
        latent_trajectory_shape=shape,
    )
    time_c, noise_c = fixed_validation_perturbation(
        target_frame=499,
        latent_trajectory_shape=shape,
    )
    assert torch.equal(time_a, time_b)
    assert torch.equal(noise_a, noise_b)
    assert not torch.equal(noise_a, noise_c)
    assert not torch.equal(time_a, time_c) or not torch.equal(noise_a, noise_c)

    bank = FixedValidationNoiseBank((498, 499), shape, base_seed=2031905426)
    bank_time, bank_noise = bank.get(498)
    assert torch.equal(bank_time, time_a)
    assert torch.equal(bank_noise, noise_a)
    with pytest.raises(KeyError):
        bank.get(500)


def test_full_training_fails_before_any_data_or_cuda_access() -> None:
    full = B2RunConfig.frozen(mode="full", seed=1701)
    with pytest.raises(RuntimeError, match="full B2 training is not authorized"):
        train_b2_smoke(
            config=full,
            catalog=None,  # type: ignore[arg-type]
            codec_checkpoint=None,  # type: ignore[arg-type]
            codec_checkpoint_sha256="",
            output_directory=None,  # type: ignore[arg-type]
            paper0_commit="a" * 40,
            slurm_job_id="test",
            device=torch.device("cpu"),
        )

