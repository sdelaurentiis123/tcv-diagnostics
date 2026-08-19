"""Regression tests for truth-separated B5 M32 forecast artifacts."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tcv_diagnostics.b5_residual_edm_forecast import (
    B5_FORECAST_AXES,
    B5ForecastArtifact,
    B5ForecastSchema,
    B5ForecastWriter,
    generate_selected_b5_forecasts,
    initial_noise_from_uint64,
    load_scientific_sampler_seed_bank,
    load_selected_b5_model,
    sample_b5_target_from_seeds,
    save_scientific_sampler_seed_bank,
)
from tcv_diagnostics.b5_residual_edm_full_training import (
    B5_SCIENTIFIC_BANK_NPY_SHA256,
    scientific_sampler_seed_bank,
)
from tcv_diagnostics.codec_training import sha256_path


def tiny_schema() -> B5ForecastSchema:
    return B5ForecastSchema(
        members=3,
        future_frames=1,
        channels=2,
        volume_shape=(2, 3, 4),
    )


def saved_bank(tmp_path: Path) -> tuple[np.ndarray, Path, str]:
    bank = scientific_sampler_seed_bank()
    path = tmp_path / "scientific_sampler_seeds_M32.npy"
    digest = save_scientific_sampler_seed_bank(path, bank)
    return bank, path, digest


def noise_hashes(seeds: np.ndarray) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            initial_noise_from_uint64(seed, spatial_shape=(2, 3, 4)).tobytes()
        ).hexdigest()
        for seed in seeds
    )


def test_B5_frozen_forecast_schema_has_canonical_axes_shape_and_payload() -> None:
    schema = B5ForecastSchema.frozen()
    assert B5_FORECAST_AXES == (
        "target_frame",
        "ensemble_member",
        "future_time",
        "channel",
        "x",
        "y",
        "stored_toroidal_z",
    )
    assert schema.per_target_shape == (32, 1, 5, 64, 32, 88)
    assert schema.scientific_seed_shape == (126, 32)
    assert 126 * np.prod(schema.per_target_shape) * 4 == 14_533_263_360


def test_B5_scientific_seed_bank_persistence_is_exact_and_no_overwrite(
    tmp_path: Path,
) -> None:
    bank, path, digest = saved_bank(tmp_path)
    assert digest == B5_SCIENTIFIC_BANK_NPY_SHA256
    np.testing.assert_array_equal(load_scientific_sampler_seed_bank(path, digest), bank)
    assert int(bank[0, 0]) == 6_157_905_366_534_159_321
    with pytest.raises(FileExistsError):
        save_scientific_sampler_seed_bank(path, bank)
    with pytest.raises(ValueError, match="SHA-256"):
        load_scientific_sampler_seed_bank(path, "0" * 64)


def test_B5_seed_expansion_is_reproducible_and_byte_locked() -> None:
    seed = scientific_sampler_seed_bank()[0, 0]
    first = initial_noise_from_uint64(seed, spatial_shape=(2, 3, 4))
    repeated = initial_noise_from_uint64(seed, spatial_shape=(2, 3, 4))
    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (5, 2, 3, 4) and first.dtype == np.float32
    assert hashlib.sha256(first.tobytes()).hexdigest() == (
        "b2ed790cac5812af332a5cf656d1d6161cf5ac24c29baa9bbd34d50f189e4d52"
    )


def test_B5_writer_reader_lock_order_axes_seeds_and_timing(tmp_path: Path) -> None:
    schema = tiny_schema()
    bank, bank_path, bank_digest = saved_bank(tmp_path)
    path = tmp_path / "b5_forecast.h5"
    frames = (498, 499)
    metadata = {
        "source_kind": "selected_B5_residual_EDM",
        "arm": "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI",
        "context_frames": 1,
        "target_truth_read": False,
    }
    first = np.zeros(schema.per_target_shape, dtype=np.float32)
    second = np.ones_like(first)
    with B5ForecastWriter(
        path,
        target_frames=frames,
        metadata=metadata,
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=first,
            inference_seconds=1.25,
            sampler_seed_row=bank[0, :3],
            initial_noise_sha256=noise_hashes(bank[0, :3]),
        )
        writer.append(
            target_frame=499,
            standardized_forecast=second,
            inference_seconds=2.75,
            sampler_seed_row=bank[1, :3],
            initial_noise_sha256=noise_hashes(bank[1, :3]),
        )
        writer.finalize()

    with B5ForecastArtifact(
        path,
        expected_sha256=sha256_path(path),
        target_frames=frames,
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as artifact:
        np.testing.assert_array_equal(artifact.read(0, 1), first[None])
        np.testing.assert_array_equal(artifact.read(1, 2), second[None])
        assert artifact.metadata == metadata
        timing = artifact.timing_record()
        assert timing["target_count"] == 2
        assert timing["ensemble_members_per_target"] == 3
        assert timing["network_evaluations_per_member"] == 35
        assert timing["total_seconds"] == pytest.approx(4.0)


def test_B5_writer_rejects_heldout_reordering_and_wrong_seed_row(
    tmp_path: Path,
) -> None:
    schema = tiny_schema()
    bank, bank_path, bank_digest = saved_bank(tmp_path)
    values = np.zeros(schema.per_target_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="held-out"):
        B5ForecastWriter(
            tmp_path / "forbidden.h5",
            target_frames=(498,),
            metadata={"source": "/secret/85606/file.h5"},
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_digest,
            schema=schema,
        )
    with B5ForecastWriter(
        tmp_path / "wrong_seed.h5",
        target_frames=(498,),
        metadata={"target_truth_read": False},
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="seed row"):
            writer.append(
                target_frame=498,
                standardized_forecast=values,
                inference_seconds=0.1,
                sampler_seed_row=bank[1, :3],
                initial_noise_sha256=noise_hashes(bank[1, :3]),
            )


def test_B5_artifact_rejects_tampered_stored_seed(tmp_path: Path) -> None:
    schema = tiny_schema()
    bank, bank_path, bank_digest = saved_bank(tmp_path)
    path = tmp_path / "tampered.h5"
    with B5ForecastWriter(
        path,
        target_frames=(498,),
        metadata={"target_truth_read": False},
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=np.zeros(schema.per_target_shape, dtype=np.float32),
            inference_seconds=0.1,
            sampler_seed_row=bank[0, :3],
            initial_noise_sha256=noise_hashes(bank[0, :3]),
        )
        writer.finalize()
    with h5py.File(path, "r+") as handle:
        handle["sampler_seed_uint64"][0, 0] = np.uint64(0)
    with pytest.raises(ValueError, match="stored sampler seeds"):
        B5ForecastArtifact(
            path,
            expected_sha256=sha256_path(path),
            target_frames=(498,),
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_digest,
            schema=schema,
        )


def test_B5_selected_model_loader_rejects_hash_before_torch_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_selected_b5_model(
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            device=torch.device("cpu"),
            training_commit="a" * 40,
        )


def test_B5_sampling_rejects_noncanonical_context_before_model_use() -> None:
    with pytest.raises(ValueError, match="context shape"):
        sample_b5_target_from_seeds(
            model=object(),  # type: ignore[arg-type]
            context=torch.zeros(1, 1, 5, 2, 3, 4),
            deterministic_mean=torch.zeros(1, 5, 2, 3, 4),
            complete_member_seeds=np.zeros(32, dtype=np.uint64),
            member_batch_size=8,
        )


def test_B5_generator_source_freezes_truth_lock_seed_prefixes_and_smoke() -> None:
    source = inspect.getsource(generate_selected_b5_forecasts)
    assert "tuple(range(498, 502))" in source
    assert "B5_FULL_VALIDATION_TARGETS" in source
    assert "dataset.target_truth_read is not False" in source
    assert '"target" in item' in source
    assert "member_prefixes_regenerated" in source
    assert "complete_M32_generated_once" in source
    assert "network_evaluations_per_member" in source
