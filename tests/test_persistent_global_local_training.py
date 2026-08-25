from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from tcv_diagnostics.persistent_global_local_training import (
    PGL_MEAN_STEP_WEIGHTS,
    PersistentPilotTrainingConfig,
    evaluate_mean_state,
    fit_parent_residual_scales,
    keyed_sigma_and_noise,
    mean_forecast_trajectory,
    toroidal_equivariance_gate,
    validation_objective,
    weighted_mean_state_loss,
)
from tcv_diagnostics.models.persistent_global_local import (
    PersistentGlobalLocalConfig,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
)


@dataclass
class _Forecast:
    volume: torch.Tensor


class _Mean(nn.Module):
    def __init__(self, increment: float = 0.1) -> None:
        super().__init__()
        self.increment = nn.Parameter(torch.full((5,), float(increment)))

    def forecast(self, context: torch.Tensor, lead_steps: torch.Tensor) -> _Forecast:
        update = self.increment.reshape(1, 5, 1, 1, 1).to(context)
        return _Forecast(context[:, -1] + lead_steps.reshape(-1, 1, 1, 1, 1) * update)


class _Dataset:
    def __init__(self, *, split: str, count: int, start: int) -> None:
        self.split = split
        self.horizon = 4
        self.windows = tuple(range(count))
        self.start = start
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, object]:
        current_frame = self.start + int(index)
        base = np.full((1, 5, 4, 4, 12), current_frame / 1000.0, dtype=np.float32)
        targets = np.stack(
            [base[0] + np.float32(0.2 * step) for step in range(1, 5)], axis=0
        )
        return {
            "context": base,
            "targets": targets,
            "current_frame_index": np.int64(current_frame),
            "target_frame_indices": np.arange(
                current_frame + 1, current_frame + 5, dtype=np.int64
            ),
        }


def _tiny_edm(*, z: int = 12) -> PersistentGlobalLocalEDM:
    del z
    config = PersistentGlobalLocalConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        global_channels=4,
        global_pool_xy=(2, 2),
        low_mode_maximum=2,
        noise_embedding_features=16,
        group_norm_maximum_groups=4,
    )
    return PersistentGlobalLocalEDM(
        config,
        residual_scales=torch.ones((4, 5)),
        noise_config=PersistentNoiseConfig(
            global_pool_xy=(2, 2), low_mode_maximum=2
        ),
    )


def test_frozen_training_budgets_and_mean_weights() -> None:
    smoke = PersistentPilotTrainingConfig(mode="smoke")
    pilot = PersistentPilotTrainingConfig(mode="pilot")
    assert smoke.total_optimizer_steps == 4
    assert pilot.total_optimizer_steps == 4280
    assert pilot.validation_epochs == tuple(range(2, 21, 2))
    assert PGL_MEAN_STEP_WEIGHTS == (0.625, 0.125, 0.125, 0.125)
    assert pilot.to_record()["physics_derived_loss_used"] is False
    with pytest.raises(ValueError, match="seed 1702"):
        PersistentPilotTrainingConfig(mode="smoke", seed=1701)


def test_mean_rollout_and_weighted_state_loss_use_no_truth_feedback() -> None:
    model = _Mean(0.1)
    context = torch.zeros((1, 1, 5, 4, 4, 12))
    target = torch.stack(
        [torch.full((1, 5, 4, 4, 12), 0.2 * step) for step in range(1, 5)],
        dim=1,
    )
    mean = mean_forecast_trajectory(model, context)
    assert torch.allclose(
        mean[:, :, 0, 0, 0, 0],
        torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
    )
    loss, fields = weighted_mean_state_loss(mean, target, torch.ones(5))
    expected = sum(
        weight * (0.1 * step) ** 2
        for step, weight in enumerate(PGL_MEAN_STEP_WEIGHTS, start=1)
    )
    assert float(loss.detach()) == pytest.approx(expected, rel=1.0e-6)
    assert fields.shape == (4, 5)


def test_scale_fit_uses_all_training_windows_and_future_step_field_rms() -> None:
    model = _Mean(0.1)
    scales, record = fit_parent_residual_scales(
        parent_mean=model,
        dataset=_Dataset(split="train", count=428, start=0),
        device=torch.device("cpu"),
    )
    expected = torch.tensor([0.1, 0.2, 0.3, 0.4]).reshape(4, 1).expand(4, 5)
    assert torch.allclose(scales, expected, atol=2.0e-6)
    assert record["training_window_count"] == 428
    assert record["physics_derived_quantity"] is False
    assert record["held_out_85606_read"] is False


def test_keyed_noise_is_reproducible_and_has_expected_shape() -> None:
    reference = torch.zeros((1, 4, 5, 4, 4, 12))
    config = PersistentNoiseConfig(global_pool_xy=(2, 2), low_mode_maximum=2)
    first = keyed_sigma_and_noise(
        base_seed=12,
        epoch_zero_based=3,
        current_frame=17,
        probe=1,
        reference=reference,
        noise_config=config,
    )
    second = keyed_sigma_and_noise(
        base_seed=12,
        epoch_zero_based=3,
        current_frame=17,
        probe=1,
        reference=reference,
        noise_config=config,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[1].shape == reference.shape


def test_smoke_validation_is_state_only_and_chronological() -> None:
    model = _Mean(0.1)
    dataset = _Dataset(split="validation", count=4, start=496)
    record = validation_objective(
        mean_model=model,
        edm=_tiny_edm(),
        dataset=dataset,
        derivative_rms=torch.ones(5),
        config=PersistentPilotTrainingConfig(mode="smoke"),
        device=torch.device("cpu"),
    )
    assert tuple(record["blocks"]) == ("SMOKE",)
    assert record["blocks"]["SMOKE"]["current_frames"] == [496, 497, 498, 499]
    assert record["physics_metric_used"] is False
    assert record["future_truth_used_as_condition"] is False
    state = evaluate_mean_state(
        mean_model=model, dataset=dataset, device=torch.device("cpu")
    )
    assert state["window_count"] == 4
    assert state["horizons"]["1"]["mean_field_mse"] == pytest.approx(0.01)
    assert state["horizons"]["4"]["mean_field_mse"] == pytest.approx(0.16)


def test_production_equivariance_gate_requires_full_toroidal_axis() -> None:
    edm = _tiny_edm()
    current = torch.randn((1, 5, 4, 4, 88))
    mean = torch.randn((1, 4, 5, 4, 4, 12))
    clean = torch.randn_like(mean)
    with pytest.raises(ValueError, match="all 88"):
        toroidal_equivariance_gate(
            edm=edm,
            current=current[..., :12],
            mean=mean,
            clean_reference=clean,
        )
