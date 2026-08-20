"""Leakage, augmentation, artifact, and noise tests for ECRD data mechanics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tcv_diagnostics.ecrd_data as data
from tcv_diagnostics.b5_residual_edm_full_training import (
    keyed_full_sigma_and_noise,
    sigma_and_noise_from_uint64,
)
from tcv_diagnostics.models.ecrd import MultiscaleNoiseConfig


class _Windows:
    split = "train"
    context_frames = 1
    target_frames = data.ECRD_TRAIN_TARGETS
    fields = ("Ne", "Pe", "Pi", "phi", "Vi")
    augment = True

    def __init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int):
        target = self.target_frames[index]
        context = np.zeros((1, 5, 64, 32, 88), dtype=np.float32)
        truth = np.zeros((5, 64, 32, 88), dtype=np.float32)
        context[..., 0] = 2.0
        truth[..., 0] = 5.0
        return {
            "target_frame_index": target,
            "context_frame_indices": np.asarray([target - 1]),
            "context": np.roll(context, 3, axis=-1),
            "target": np.roll(truth, 3, axis=-1),
            "toroidal_roll": 3,
        }


class _Parent:
    split = "train"
    target_frames = data.ECRD_TRAIN_TARGETS

    def read(self, start: int, stop: int) -> np.ndarray:
        result = np.zeros((stop - start, 5, 64, 32, 88), dtype=np.float32)
        result[..., 0] = 1.0
        return result


class _LegacyForecast:
    target_frames = data.ECRD_TRAIN_TARGETS
    sha256 = "a" * 64

    def read(self, start: int, stop: int) -> np.ndarray:
        return np.full((stop - start, 5, 2, 2, 4), 3.0, dtype=np.float32)


def test_unsymmetrized_h1_adapter_is_read_only_and_split_explicit() -> None:
    legacy = _LegacyForecast()
    parent = data.FrozenH1ParentAdapter(legacy, split="train")
    assert parent.split == "train"
    assert parent.target_frames == data.ECRD_TRAIN_TARGETS
    assert parent.sha256 == legacy.sha256
    np.testing.assert_array_equal(parent.read(0, 1), legacy.read(0, 1))
    with pytest.raises(ValueError, match="targets"):
        data.FrozenH1ParentAdapter(legacy, split="validation")


def test_dataset_rolls_parent_with_context_and_truth() -> None:
    dataset = data.ECRDResidualDataset(
        _Windows(), _Parent(), split="train", history_frames=1, augment=True
    )
    item = dataset[0]
    assert item["condition"].shape == (10, 64, 32, 88)
    assert item["toroidal_roll"] == 3
    assert np.all(item["condition"][:5, ..., 3] == 2.0)
    assert np.all(item["condition"][5:, ..., 3] == 1.0)
    scales = np.asarray(data.B5_RESIDUAL_SCALES, dtype=np.float32)
    np.testing.assert_allclose(
        item["normalized_parent_residual"][:, 0, 0, 3],
        4.0 / scales,
    )
    assert item["target_truth_used_as_condition"] is False


def test_validation_augmentation_fails_closed() -> None:
    windows = _Windows()
    windows.split = "validation"
    windows.target_frames = data.ECRD_VALIDATION_TARGETS
    parent = _Parent()
    parent.split = "validation"
    parent.target_frames = data.ECRD_VALIDATION_TARGETS
    with pytest.raises(ValueError, match="forbidden"):
        data.ECRDResidualDataset(
            windows,
            parent,
            split="validation",
            history_frames=1,
            augment=True,
        )


def test_keyed_corruption_is_repeatable_and_scale_specific() -> None:
    first_sigma, first = data.keyed_ecrd_sigma_and_noise(
        base_seed=67502,
        epoch_zero_based=3,
        target_frame=20,
        multiscale=True,
        spatial_shape=(8, 8, 12),
        config=MultiscaleNoiseConfig(mesoscale_xy=(2, 2)),
    )
    second_sigma, second = data.keyed_ecrd_sigma_and_noise(
        base_seed=67502,
        epoch_zero_based=3,
        target_frame=20,
        multiscale=True,
        spatial_shape=(8, 8, 12),
        config=MultiscaleNoiseConfig(mesoscale_xy=(2, 2)),
    )
    white_sigma, white = data.keyed_ecrd_sigma_and_noise(
        base_seed=67502,
        epoch_zero_based=3,
        target_frame=20,
        multiscale=False,
        spatial_shape=(8, 8, 12),
        config=MultiscaleNoiseConfig(mesoscale_xy=(2, 2)),
    )
    assert first_sigma == second_sigma == white_sigma
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, white)
    b5_sigma, b5_white = keyed_full_sigma_and_noise(
        seed=67502,
        epoch_zero_based=3,
        target_frame=20,
        spatial_shape=(8, 8, 12),
    )
    assert white_sigma == b5_sigma
    np.testing.assert_array_equal(white, b5_white)


def test_validation_white_noise_is_exactly_paired_with_b5() -> None:
    sigma, noise = data.validation_sigma_and_noise_from_uint64(
        12345,
        multiscale=False,
        spatial_shape=(8, 8, 12),
    )
    expected_sigma, expected_noise = sigma_and_noise_from_uint64(
        12345,
        spatial_shape=(8, 8, 12),
    )
    assert sigma == expected_sigma
    np.testing.assert_array_equal(noise, expected_noise)


def test_multiscale_noise_has_local_full_rank_and_correlated_neighbors() -> None:
    values = np.stack(
        [
            data.multiscale_noise_from_uint64(
                seed,
                spatial_shape=(8, 8, 12),
                config=MultiscaleNoiseConfig(mesoscale_xy=(2, 2)),
            )
            for seed in range(256)
        ]
    )
    variances = np.var(values, axis=0)
    assert float(np.min(variances)) > 0.5
    assert 0.8 < float(np.mean(variances)) < 1.2
    same_patch = np.corrcoef(values[:, 0, 0, 0, 0], values[:, 0, 1, 1, 0])[0, 1]
    far_patch = np.corrcoef(values[:, 0, 0, 0, 0], values[:, 0, 6, 6, 0])[0, 1]
    assert same_patch > far_patch + 0.05


def test_parent_writer_refuses_held_out_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="held-out"):
        data.ECRDParentMeanWriter(
            tmp_path / "parent.h5",
            split="train",
            target_frames=data.ECRD_TRAIN_TARGETS,
            metadata={"target_truth_read": False, "source": "85606"},
        )


def test_parent_artifact_exposes_bounded_execution_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "ECRD_TRAIN_TARGETS", (2,))
    monkeypatch.setattr(data, "VOLUME_SHAPE", (2, 2, 4))
    path = tmp_path / "parent.h5"
    with data.ECRDParentMeanWriter(
        path,
        split="train",
        target_frames=(2,),
        metadata={
            "target_truth_read": False,
            "artifact_authority": "bounded_non_scientific_engineering_smoke_only",
            "execution_device": "cpu-smoke",
        },
    ) as writer:
        writer.append(
            target_frame=2,
            standardized_parent_mean=np.zeros((5, 2, 2, 4), dtype=np.float32),
            inference_seconds=1.0,
        )
        writer.finalize()
    artifact = data.ECRDParentMeanArtifact(
        path,
        split="train",
        expected_sha256=data.sha256_path(path),
    )
    try:
        assert artifact.artifact_authority == (
            "bounded_non_scientific_engineering_smoke_only"
        )
        assert artifact.execution_device == "cpu-smoke"
    finally:
        artifact.close()
