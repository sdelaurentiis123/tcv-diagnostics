from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

import tcv_diagnostics.persistent_global_local_forecast as forecast_module
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.models.persistent_global_local import (
    PersistentGlobalLocalConfig,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
)
from tcv_diagnostics.persistent_global_local_forecast import (
    PGL_SCIENTIFIC_SEED_BANK_SHA256,
    PGLForecastArtifact,
    PGLForecastSchema,
    PGLForecastWriter,
    evaluation_seed_rows,
    initial_noise_from_uint64,
    tensor_sha256,
)


def _tiny_config() -> PersistentGlobalLocalConfig:
    return PersistentGlobalLocalConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        global_channels=4,
        global_pool_xy=(2, 2),
        low_mode_maximum=2,
        noise_embedding_features=16,
        group_norm_maximum_groups=4,
    )


def _tiny_noise_config() -> PersistentNoiseConfig:
    return PersistentNoiseConfig(global_pool_xy=(2, 2), low_mode_maximum=2)


def _seed_bank() -> np.ndarray:
    return np.arange(126 * 32, dtype=np.uint64).reshape(126, 32)


def test_seed_rows_use_the_one_frame_target_index() -> None:
    bank = _seed_bank()
    rows = evaluation_seed_rows(bank, (497, 501, 619))
    np.testing.assert_array_equal(rows[0], bank[0])
    np.testing.assert_array_equal(rows[1], bank[4])
    np.testing.assert_array_equal(rows[2], bank[122])


def test_structured_noise_expansion_is_seed_stable() -> None:
    model = PersistentGlobalLocalEDM(
        _tiny_config(),
        residual_scales=torch.ones((4, 5)),
        noise_config=_tiny_noise_config(),
    )
    reference = torch.zeros((1, 4, 5, 8, 8, 12))
    first = initial_noise_from_uint64(9001, reference=reference, model=model)
    second = initial_noise_from_uint64(np.uint64(9001), reference=reference, model=model)
    other = initial_noise_from_uint64(9002, reference=reference, model=model)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert len(tensor_sha256(first)) == 64
    assert tensor_sha256(first) == tensor_sha256(second)


def test_forecast_writer_and_artifact_round_trip(tmp_path, monkeypatch) -> None:
    bank = _seed_bank()
    monkeypatch.setattr(
        forecast_module,
        "load_scientific_sampler_seed_bank",
        lambda path, digest: bank,
    )
    schema = PGLForecastSchema(
        starts=(497, 500),
        members=2,
        horizon=4,
        fields=5,
        volume_shape=(2, 2, 8),
    )
    path = tmp_path / "forecast.h5"
    lock = "1" * 64
    rows = evaluation_seed_rows(bank, schema.starts)[:, : schema.members]
    with PGLForecastWriter(
        path,
        paper0_commit="a" * 40,
        manifest_sha256=lock,
        training_result_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        seed_bank_path=tmp_path / "bank.npy",
        seed_bank_sha256=PGL_SCIENTIFIC_SEED_BANK_SHA256,
        schema=schema,
    ) as writer:
        for index, start in enumerate(schema.starts):
            values = np.full(schema.forecast_shape[1:], index + 0.25, dtype=np.float32)
            mean = np.full(schema.mean_shape[1:], index + 0.5, dtype=np.float32)
            parent = np.full(schema.mean_shape[1:], index + 0.75, dtype=np.float32)
            hashes = [
                hashlib.sha256(f"{start}:{member}".encode()).hexdigest()
                for member in range(schema.members)
            ]
            writer.append(
                current_frame=start,
                standardized_forecast=values,
                selected_mean=mean,
                parent_mean=parent,
                inference_seconds=1.25 + index,
                sampler_seed_row=rows[index],
                initial_noise_sha256=hashes,
            )
        writer.finalize()
    digest = sha256_path(path)
    with PGLForecastArtifact(
        path,
        expected_sha256=digest,
        manifest_sha256=lock,
        training_result_sha256="2" * 64,
        checkpoint_sha256="3" * 64,
        seed_bank_path=tmp_path / "bank.npy",
        seed_bank_sha256=PGL_SCIENTIFIC_SEED_BANK_SHA256,
        schema=schema,
    ) as artifact:
        assert artifact.read_forecast(1).shape == schema.forecast_shape[1:]
        assert artifact.read_mean(0, parent=False).shape == schema.mean_shape[1:]
        assert artifact.read_mean(0, parent=True).shape == schema.mean_shape[1:]
        assert np.array_equal(
            artifact.read_forecast_horizon(1, 4), artifact.read_forecast(1)[:, 3]
        )
        assert np.array_equal(
            artifact.read_mean_horizon(0, 2, parent=False),
            artifact.read_mean(0, parent=False)[1],
        )
        assert np.array_equal(
            artifact.read_mean_horizon(0, 3, parent=True),
            artifact.read_mean(0, parent=True)[2],
        )
        with pytest.raises(ValueError, match="horizon"):
            artifact.read_forecast_horizon(0, 0)
        assert artifact.timing_record()["start_count"] == 2
