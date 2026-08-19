"""Known-answer tests for bounded B5 residual-EDM smoke mechanics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tcv_diagnostics.b5_residual_edm_training import (
    B5EDMSmokeConfig,
    B5ResidualSmokeDataset,
    keyed_sigma_and_noise,
    module_state_sha256,
    sampler_initial_noise,
    smoke_target_sequence,
)
from tcv_diagnostics.model_data import array_sha256
from tcv_diagnostics.model_training_data import VOLUME_SHAPE
from tcv_diagnostics.models.field_residual_edm import B5_RESIDUAL_SCALES


class FakeWindows:
    split = "train"
    context_frames = 1
    target_frames = tuple(range(2, 10))
    augment = False
    fields = ("Ne", "Pe", "Pi", "phi", "Vi")

    def __getitem__(self, index: int) -> dict:
        target = self.target_frames[index]
        context = np.full((1, 5, *VOLUME_SHAPE), target - 1, dtype=np.float32)
        truth = np.full((5, *VOLUME_SHAPE), target + 1, dtype=np.float32)
        return {
            "context": context,
            "target": truth,
            "context_frame_indices": np.asarray([target - 1], dtype=np.int64),
            "target_frame_index": np.int64(target),
        }


class FakeForecast:
    target_frames = tuple(range(2, 432))
    sha256 = "fake"

    def read(self, start: int, stop: int) -> np.ndarray:
        assert stop == start + 1
        target = start + 2
        return np.full((1, 5, *VOLUME_SHAPE), target, dtype=np.float32)


def test_B5_smoke_config_and_seeded_cycle_are_exact() -> None:
    config = B5EDMSmokeConfig()
    sequence = smoke_target_sequence()
    assert sequence[:8] == (6, 3, 9, 2, 4, 5, 8, 7)
    assert len(sequence) == 64
    assert all(sequence[index : index + 8] == sequence[:8] for index in range(0, 64, 8))
    assert {target: sequence.count(target) for target in range(2, 10)} == {
        target: 8 for target in range(2, 10)
    }
    assert config.target_sequence == sequence
    assert config.to_record()["scientific_result"] is False
    assert config.to_record()["full_training_authorized"] is False


def test_B5_keyed_sigma_and_noise_are_reproducible_and_byte_locked() -> None:
    target = smoke_target_sequence()[0]
    first_sigma, first = keyed_sigma_and_noise(
        seed=67_002,
        ordinal=0,
        target_frame=target,
        spatial_shape=(2, 3, 4),
    )
    second_sigma, second = keyed_sigma_and_noise(
        seed=67_002,
        ordinal=0,
        target_frame=target,
        spatial_shape=(2, 3, 4),
    )
    assert float(first_sigma) == pytest.approx(1.6987277269363403, rel=0.0)
    assert first.dtype == np.float32 and first.shape == (5, 2, 3, 4)
    assert np.array_equal(first, second)
    assert first_sigma == second_sigma
    assert array_sha256(first) == (
        "8909702753c1db72ec8062da9351924d9be82d915e3df8946c27439f51bfb1f0"
    )


def test_B5_sampler_initial_members_are_independent_and_byte_locked() -> None:
    values = sampler_initial_noise(spatial_shape=(2, 3, 4))
    assert values.shape == (2, 5, 2, 3, 4)
    assert values.dtype == np.float32
    assert not np.array_equal(values[0], values[1])
    assert array_sha256(values) == (
        "82c8111f8ef1a0c84904a5da91c076a93f64fa718476b54049abae661c2daf92"
    )


def test_B5_smoke_dataset_joins_context_mean_and_truth_without_leakage() -> None:
    dataset = B5ResidualSmokeDataset(FakeWindows(), FakeForecast())
    item = dataset[3]
    target = 5
    assert int(item["target_frame_index"]) == target
    assert int(item["context_frame_index"]) == target - 1
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


def test_B5_smoke_dataset_rejects_augmented_or_wrong_target_windows() -> None:
    windows = FakeWindows()
    windows.augment = True
    with pytest.raises(ValueError, match="window dataset"):
        B5ResidualSmokeDataset(windows, FakeForecast())


def test_B5_module_state_hash_changes_with_a_parameter() -> None:
    module = nn.Linear(3, 2)
    first = module_state_sha256(module)
    assert first == module_state_sha256(module)
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    assert module_state_sha256(module) != first
