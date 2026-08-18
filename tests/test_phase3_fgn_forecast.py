"""Regression tests for truth-free B3 FGN scientific forecast artifacts."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.fgn_forecast import (
    FGN_FORECAST_AXES,
    FGN_SCIENTIFIC_NOISE_SEED,
    FGNForecastArtifact,
    FGNForecastSchema,
    FGNForecastWriter,
    _row_sha256,
    generate_selected_fgn_forecasts,
    load_scientific_noise_bank,
    load_selected_fgn_model,
    sample_fgn_target_from_noise,
    save_scientific_noise_bank,
    scientific_noise_bank,
)
from tcv_diagnostics.fgn_training import ParentArtifacts, validation_noise_bank


def _tiny_schema() -> FGNForecastSchema:
    return FGNForecastSchema(
        members=3,
        future_frames=1,
        channels=2,
        volume_shape=(2, 3, 4),
        raw_noise_features=32,
    )


def _saved_bank(tmp_path: Path) -> tuple[np.ndarray, Path, str]:
    bank = scientific_noise_bank()
    path = tmp_path / "scientific_noise_M32.npy"
    digest = save_scientific_noise_bank(path, bank)
    return bank, path, digest


def test_frozen_fgn_forecast_schema_has_canonical_axes_and_shape() -> None:
    schema = FGNForecastSchema.frozen()
    assert FGN_FORECAST_AXES == (
        "target_frame",
        "ensemble_member",
        "future_time",
        "channel",
        "x",
        "y",
        "stored_toroidal_z",
    )
    assert schema.per_target_shape == (32, 1, 5, 64, 32, 88)
    assert schema.scientific_noise_shape == (126, 32, 32)


def test_scientific_noise_bank_is_exact_and_independent_of_selection_bank() -> None:
    first = scientific_noise_bank()
    repeated = scientific_noise_bank()
    selection = validation_noise_bank()

    assert FGN_SCIENTIFIC_NOISE_SEED == 31_032
    assert first.shape == (126, 32, 32)
    assert first.dtype == np.dtype("f4")
    np.testing.assert_array_equal(first, repeated)
    assert first[0, :2, :4].reshape(-1).view("u4").tolist() == [
        1058853222,
        1053251522,
        3197645008,
        1028362331,
        3206491276,
        1065798268,
        1057936324,
        1048909588,
    ]
    assert hashlib.sha256(memoryview(first)).hexdigest() == (
        "5bd4bb603f70ffc0531238ba78e4754b57f978f50b47b0cf46aeb1e15140e247"
    )
    assert not np.array_equal(first[:, :2], selection)


def test_scientific_noise_persistence_is_hash_checked_and_no_overwrite(
    tmp_path: Path,
) -> None:
    bank, path, digest = _saved_bank(tmp_path)
    assert digest == "1449777a61d40af49ccb3bd6bed5edcba0fd8afe24d113e6175218c04865aa9c"
    np.testing.assert_array_equal(load_scientific_noise_bank(path, digest), bank)
    with pytest.raises(FileExistsError):
        save_scientific_noise_bank(path, bank)
    with pytest.raises(ValueError, match="SHA-256"):
        load_scientific_noise_bank(path, "0" * 64)


def test_fgn_writer_reader_lock_order_axes_noise_hash_and_timing(
    tmp_path: Path,
) -> None:
    schema = _tiny_schema()
    bank, bank_path, bank_digest = _saved_bank(tmp_path)
    path = tmp_path / "fgn_forecast.h5"
    frames = (498, 499)
    metadata = {
        "source_kind": "selected_B3_FGN",
        "arm": "B3-FGN-H1",
        "context_frames": 1,
        "target_truth_read": False,
    }
    first = np.zeros(schema.per_target_shape, dtype=np.float32)
    second = np.ones_like(first)
    with FGNForecastWriter(
        path,
        target_frames=frames,
        metadata=metadata,
        noise_bank_path=bank_path,
        noise_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=first,
            inference_seconds=1.25,
            raw_noise_row_sha256=_row_sha256(bank[0, :3, :32]),
        )
        writer.append(
            target_frame=499,
            standardized_forecast=second,
            inference_seconds=2.75,
            raw_noise_row_sha256=_row_sha256(bank[1, :3, :32]),
        )
        writer.finalize()

    with FGNForecastArtifact(
        path,
        expected_sha256=sha256_path(path),
        target_frames=frames,
        noise_bank_path=bank_path,
        noise_bank_sha256=bank_digest,
        schema=schema,
    ) as artifact:
        np.testing.assert_array_equal(artifact.read(0, 1), first[None])
        np.testing.assert_array_equal(artifact.read(1, 2), second[None])
        assert artifact.metadata == metadata
        timing = artifact.timing_record()
        assert timing["target_count"] == 2
        assert timing["ensemble_members_per_target"] == 3
        assert timing["network_evaluations_per_member"] == 1
        assert timing["total_seconds"] == pytest.approx(4.0)


def test_fgn_writer_rejects_heldout_reordering_and_nonfinite(tmp_path: Path) -> None:
    schema = _tiny_schema()
    bank, bank_path, bank_digest = _saved_bank(tmp_path)
    values = np.zeros(schema.per_target_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="held-out"):
        FGNForecastWriter(
            tmp_path / "forbidden.h5",
            target_frames=(498,),
            metadata={"source": "/secret/85606/file.h5"},
            noise_bank_path=bank_path,
            noise_bank_sha256=bank_digest,
            schema=schema,
        )

    with FGNForecastWriter(
        tmp_path / "order.h5",
        target_frames=(498, 499),
        metadata={"target_truth_read": False},
        noise_bank_path=bank_path,
        noise_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="differs"):
            writer.append(
                target_frame=499,
                standardized_forecast=values,
                inference_seconds=0.1,
                raw_noise_row_sha256=_row_sha256(bank[1, :3]),
            )
        with pytest.raises(RuntimeError, match="every target"):
            writer.finalize()

    nonfinite = values.copy()
    nonfinite[0, 0, 0, 0, 0, 0] = np.nan
    with FGNForecastWriter(
        tmp_path / "nonfinite.h5",
        target_frames=(498,),
        metadata={"target_truth_read": False},
        noise_bank_path=bank_path,
        noise_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="finite"):
            writer.append(
                target_frame=498,
                standardized_forecast=nonfinite,
                inference_seconds=0.1,
                raw_noise_row_sha256=_row_sha256(bank[0, :3]),
            )


def test_fgn_artifact_rejects_wrong_stored_noise_row_hash(tmp_path: Path) -> None:
    schema = _tiny_schema()
    bank, bank_path, bank_digest = _saved_bank(tmp_path)
    path = tmp_path / "tampered.h5"
    with FGNForecastWriter(
        path,
        target_frames=(498,),
        metadata={"target_truth_read": False},
        noise_bank_path=bank_path,
        noise_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=np.zeros(schema.per_target_shape, dtype=np.float32),
            inference_seconds=0.1,
            raw_noise_row_sha256=_row_sha256(bank[0, :3]),
        )
        writer.finalize()
    with h5py.File(path, "r+") as handle:
        handle["raw_noise_row_sha256"][0] = ("0" * 64).encode("ascii")
    with pytest.raises(ValueError, match="raw-noise row hashes"):
        FGNForecastArtifact(
            path,
            expected_sha256=sha256_path(path),
            target_frames=(498,),
            noise_bank_path=bank_path,
            noise_bank_sha256=bank_digest,
            schema=schema,
        )


def test_selected_model_loader_rejects_hash_before_torch_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    artifacts = ParentArtifacts(
        checkpoint_path=tmp_path / "parent.pt",
        checkpoint_sha256="1" * 64,
        codec_path=tmp_path / "codec.pt",
        codec_sha256="2" * 64,
        latent_normalization_path=tmp_path / "latent.json",
        latent_normalization_sha256="3" * 64,
    )
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_selected_fgn_model(
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            artifacts=artifacts,
            device=torch.device("cpu"),
            training_commit="a" * 40,
        )


def test_fgn_sampling_rejects_noncanonical_context_before_model_use() -> None:
    with pytest.raises(ValueError, match="context shape"):
        sample_fgn_target_from_noise(
            model=object(),  # type: ignore[arg-type]
            context=torch.zeros(1, 1, 5, 2, 3, 4),
            complete_raw_noise=np.zeros((32, 32), dtype=np.float32),
            member_batch_size=8,
        )


def test_generator_source_freezes_truth_lock_full_bank_and_four_target_smoke() -> None:
    source = inspect.getsource(generate_selected_fgn_forecasts)
    assert "tuple(range(498, 502))" in source
    assert "range(FGN_SCIENTIFIC_TARGET_START, FGN_SCIENTIFIC_TARGET_STOP)" in source
    assert "dataset.target_truth_read is not False" in source
    assert "member_prefixes_regenerated" in source
    assert "direct_functional_noise_single_pass" in source
