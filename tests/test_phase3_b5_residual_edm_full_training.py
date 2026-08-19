"""Known-answer tests for frozen B5 full-training mechanics."""

from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np
import pytest
import torch
from torch import nn

from tcv_diagnostics.b5_residual_edm_full_training import (
    B5EDMFullConfig,
    B5ResidualOneStepDataset,
    accumulation_groups,
    full_learning_rate,
    full_training_order,
    full_validation_seed_bank,
    keyed_full_sigma_and_noise,
    scientific_sampler_seed_bank,
    select_earliest_lowest_candidate,
    sigma_and_noise_from_uint64,
    train_b5_edm_full,
    update_ema_model,
)
from tcv_diagnostics.model_training_data import VOLUME_SHAPE
from tcv_diagnostics.models.field_residual_edm import B5_RESIDUAL_SCALES


class FakeWindows:
    context_frames = 1
    augment = False
    fields = ("Ne", "Pe", "Pi", "phi", "Vi")

    def __init__(self, split: str) -> None:
        self.split = split
        self.target_frames = (
            tuple(range(2, 432)) if split == "train" else tuple(range(498, 624))
        )

    def __getitem__(self, index: int) -> dict:
        target = self.target_frames[index]
        context = np.full((1, 5, *VOLUME_SHAPE), target - 1, dtype=np.float32)
        truth = np.full((5, *VOLUME_SHAPE), target + 1, dtype=np.float32)
        return {
            "context": context,
            "target": truth,
            "context_frame_indices": np.asarray([target - 1], dtype=np.int64),
            "target_frame_index": np.int64(target),
            "toroidal_roll": np.int64(0),
        }


class FakeForecast:
    sha256 = "fake"

    def __init__(self, target_frames: tuple[int, ...]) -> None:
        self.target_frames = target_frames

    def read(self, start: int, stop: int) -> np.ndarray:
        assert stop == start + 1
        target = self.target_frames[start]
        return np.full((1, 5, *VOLUME_SHAPE), target, dtype=np.float32)


class TinyState(nn.Module):
    def __init__(self, weight: float, fixed: float = 7.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([weight], dtype=torch.float32))
        self.register_buffer("fixed", torch.tensor([fixed], dtype=torch.float32))


def test_B5_full_config_budget_order_and_cosine_schedule_are_exact() -> None:
    config = B5EDMFullConfig()
    assert config.target_presentations == 43_000
    assert config.optimizer_steps_per_epoch == 108
    assert config.total_optimizer_steps == 10_800
    assert config.validation_completed_epochs == tuple(range(5, 101, 5))
    order = full_training_order(config)
    assert order.shape == (100, 430) and order.dtype == np.int64
    assert order[0, :12].tolist() == [
        285,
        245,
        370,
        398,
        307,
        179,
        86,
        413,
        252,
        157,
        324,
        358,
    ]
    rates = [
        full_learning_rate(config, index)
        for index in (0, config.total_optimizer_steps // 2, 10_799)
    ]
    assert rates[0] == 1.0e-4
    assert rates[-1] == 1.0e-6
    assert rates[0] > rates[1] > rates[-1]
    assert math.isclose(rates[1], 5.0492799850180375e-05, rel_tol=1e-14)


def test_B5_full_accumulation_preserves_every_target_and_partial_group() -> None:
    epoch = full_training_order()[0]
    groups = accumulation_groups(epoch)
    assert len(groups) == 108
    assert [len(group) for group in groups] == [4] * 107 + [2]
    np.testing.assert_array_equal(np.concatenate(groups), epoch)
    assert sum(map(len, groups)) == 430
    with pytest.raises(ValueError, match="not a target permutation"):
        accumulation_groups(np.full(430, 2, dtype=np.int64))


def test_B5_full_seed_banks_are_exact_independent_and_prefix_stable() -> None:
    validation = full_validation_seed_bank()
    scientific = scientific_sampler_seed_bank()
    assert validation.shape == (126, 4) and validation.dtype == np.uint64
    assert scientific.shape == (126, 32) and scientific.dtype == np.uint64
    assert int(validation[0, 0]) == 13_728_382_829_692_652_680
    assert not np.array_equal(validation, scientific[:, :4])
    np.testing.assert_array_equal(scientific[:, :4], scientific[:, :8][:, :4])


def test_B5_full_training_and_validation_corruptions_are_byte_locked() -> None:
    sigma, noise = keyed_full_sigma_and_noise(
        seed=67_502,
        epoch_zero_based=0,
        target_frame=285,
        spatial_shape=(2, 3, 4),
    )
    repeated_sigma, repeated = keyed_full_sigma_and_noise(
        seed=67_502,
        epoch_zero_based=0,
        target_frame=285,
        spatial_shape=(2, 3, 4),
    )
    assert float(sigma) == pytest.approx(0.037610214203596115, rel=0.0)
    assert sigma == repeated_sigma and np.array_equal(noise, repeated)
    assert hashlib.sha256(noise.tobytes(order="C")).hexdigest() == (
        "3f1b374f152b7f4dfdaa79bdea1969651d36b65f063f3f196c0e17ff4a2daf68"
    )
    _, changed = keyed_full_sigma_and_noise(
        seed=67_502,
        epoch_zero_based=1,
        target_frame=285,
        spatial_shape=(2, 3, 4),
    )
    assert not np.array_equal(noise, changed)

    validation_seed = full_validation_seed_bank()[0, 0]
    validation_sigma, validation_noise = sigma_and_noise_from_uint64(
        validation_seed, spatial_shape=(2, 3, 4)
    )
    assert float(validation_sigma) == pytest.approx(0.18121378123760223, rel=0.0)
    assert hashlib.sha256(validation_noise.tobytes(order="C")).hexdigest() == (
        "2d7b41739df1422a6043eb5d7ea129ed21e92cd0098014ed1b2c15bc29f179b5"
    )


@pytest.mark.parametrize("split,target", [("train", 5), ("validation", 501)])
def test_B5_full_dataset_joins_context_mean_and_truth_without_time(
    split: str, target: int
) -> None:
    windows = FakeWindows(split)
    dataset = B5ResidualOneStepDataset(
        windows,
        FakeForecast(windows.target_frames),
        split=split,
    )
    item = dataset[dataset.index_for_target(target)]
    assert item["condition"].shape == (10, *VOLUME_SHAPE)
    assert item["normalized_residual"].shape == (5, *VOLUME_SHAPE)
    np.testing.assert_array_equal(item["condition"][:5], target - 1)
    np.testing.assert_array_equal(item["condition"][5:], target)
    expected = 1.0 / np.asarray(B5_RESIDUAL_SCALES, dtype=np.float32)
    np.testing.assert_allclose(
        item["normalized_residual"][:, 0, 0, 0], expected, rtol=1e-6
    )
    assert item["target_truth_used_as_condition"] is False
    assert item["absolute_time_used_as_condition"] is False


def test_B5_full_EMA_formula_and_fixed_buffers_are_exact() -> None:
    ema = TinyState(1.0)
    raw = TinyState(3.0)
    update_ema_model(ema, raw, decay=0.999)
    assert float(ema.weight.detach()) == pytest.approx(1.002, abs=1e-7)
    assert torch.equal(ema.fixed, raw.fixed)
    changed_buffer = TinyState(3.0, fixed=8.0)
    with pytest.raises(RuntimeError, match="fixed buffer"):
        update_ema_model(ema, changed_buffer, decay=0.999)


def test_B5_full_selection_uses_earliest_numerical_minimum_only() -> None:
    records = [
        {
            "completed_epoch": epoch,
            "validation": {"mean_EDM_loss": 2.0 - epoch / 100.0},
        }
        for epoch in range(5, 101, 5)
    ]
    records[6]["validation"]["mean_EDM_loss"] = 0.25
    records[11]["validation"]["mean_EDM_loss"] = 0.25
    selected = select_earliest_lowest_candidate(records)
    assert selected["completed_epoch"] == 35


def test_B5_full_training_scope_does_not_sample_or_score_scientific_forecast() -> None:
    source = inspect.getsource(train_b5_edm_full)
    assert "scientific_sampler_seed_bank(" not in source
    assert ".sample_normalized(" not in source
    assert "compute_flux" not in source
    assert "compute_spectrum" not in source
    assert '"scientific_forecast_generated": False' in source
    assert '"held_out_85606_read": False' in source
