"""Regression tests for truth-free B4 PDE-Refiner forecast artifacts."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.pde_refiner_forecast import (
    B4_FINAL_FORECAST_AXES,
    B4_SCIENTIFIC_SEED_BANK_SEED,
    B4_SCIENTIFIC_SEED_NPY_SHA256,
    B4_SCIENTIFIC_SEED_RAW_SHA256,
    B4_STAGE_FORECAST_AXES,
    PDERefinerFinalForecastArtifact,
    PDERefinerFinalForecastWriter,
    PDERefinerForecastSchema,
    PDERefinerStageForecastArtifact,
    PDERefinerStageForecastWriter,
    _array_sha256,
    generate_selected_pde_refiner_forecasts,
    load_scientific_refiner_seed_bank,
    load_selected_pde_refiner_model,
    refinement_noise_from_scientific_seeds,
    save_scientific_refiner_seed_bank,
    scientific_refiner_seed_bank,
)
from tcv_diagnostics.pde_refiner_training import (
    RefinerParentArtifacts,
    validation_seed_bank,
)


def _tiny_schema() -> PDERefinerForecastSchema:
    return PDERefinerForecastSchema(
        final_members=3,
        stage_members=2,
        stages=4,
        future_frames=1,
        channels=2,
        volume_shape=(2, 3, 4),
    )


def _saved_bank(tmp_path: Path) -> tuple[np.ndarray, Path, str]:
    bank = scientific_refiner_seed_bank()
    path = tmp_path / "scientific_seed_bank.npy"
    digest = save_scientific_refiner_seed_bank(path, bank)
    return bank, path, digest


def _metadata() -> dict:
    return {
        "source_kind": "selected_B4_PDE_Refiner",
        "arm": "B4-PDE-Refiner-H1",
        "seed": 1701,
        "context_frames": 1,
        "target_truth_read": False,
        "absolute_time_input": False,
        "member_prefixes_regenerated": False,
    }


def test_frozen_b4_forecast_schema_has_canonical_axes_and_shapes() -> None:
    schema = PDERefinerForecastSchema.frozen()
    assert B4_FINAL_FORECAST_AXES == (
        "target_frame",
        "ensemble_member",
        "future_time",
        "channel",
        "x",
        "y",
        "stored_toroidal_z",
    )
    assert B4_STAGE_FORECAST_AXES == (
        "target_frame",
        "ensemble_member",
        "refinement_stage",
        "channel",
        "x",
        "y",
        "stored_toroidal_z",
    )
    assert schema.final_per_target_shape == (32, 1, 5, 64, 32, 88)
    assert schema.stage_per_target_shape == (4, 4, 5, 64, 32, 88)
    assert schema.scientific_seed_shape == (126, 32, 3)


def test_scientific_seed_bank_is_exact_and_independent_of_selection() -> None:
    first = scientific_refiner_seed_bank()
    repeated = scientific_refiner_seed_bank()
    selection = validation_seed_bank()

    assert B4_SCIENTIFIC_SEED_BANK_SEED == 41_032
    assert first.shape == (126, 32, 3)
    assert first.dtype == np.dtype("u8")
    np.testing.assert_array_equal(first, repeated)
    assert hashlib.sha256(first.tobytes(order="C")).hexdigest() == (
        B4_SCIENTIFIC_SEED_RAW_SHA256
    )
    assert first[0, :2].reshape(-1).tolist() == [
        10266805173566249757,
        9027700403424445416,
        6105775998067350031,
        6296489915077523366,
        6738869073669637664,
        10319338025778149299,
    ]
    assert not np.array_equal(first[:, :2], selection)


def test_scientific_seed_persistence_is_exact_hash_checked_and_exclusive(
    tmp_path: Path,
) -> None:
    bank, path, digest = _saved_bank(tmp_path)
    assert digest == B4_SCIENTIFIC_SEED_NPY_SHA256
    np.testing.assert_array_equal(
        load_scientific_refiner_seed_bank(path, digest), bank
    )
    with pytest.raises(FileExistsError):
        save_scientific_refiner_seed_bank(path, bank)
    with pytest.raises(ValueError, match="SHA-256"):
        load_scientific_refiner_seed_bank(path, "0" * 64)


def test_scientific_seeds_expand_reproducibly_to_full_latent_noise() -> None:
    seeds = scientific_refiner_seed_bank()[0, :2]
    first = refinement_noise_from_scientific_seeds(seeds)
    repeated = refinement_noise_from_scientific_seeds(seeds)
    assert first.shape == (2, 3, 32, 16, 8, 22)
    assert first.dtype == np.dtype("f4")
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first[0, 0], first[0, 1])
    assert first[0, 0].reshape(-1)[:4].view("u4").tolist() == [
        3216337385,
        3208426640,
        3215178867,
        1065769725,
    ]


def test_b4_final_and_stage_writers_lock_order_schema_hashes_and_prefix(
    tmp_path: Path,
) -> None:
    schema = _tiny_schema()
    bank, bank_path, bank_digest = _saved_bank(tmp_path)
    final_path = tmp_path / "final.h5"
    stage_path = tmp_path / "stages.h5"
    frames = (498, 499)
    stages = np.zeros(schema.stage_per_target_shape, dtype=np.float32)
    stages[:, 3] = np.arange(np.prod(stages[:, 3].shape), dtype=np.float32).reshape(
        stages[:, 3].shape
    )
    final = np.zeros(schema.final_per_target_shape, dtype=np.float32)
    final[: schema.stage_members, 0] = stages[:, 3]
    final[2] = 1.0
    with PDERefinerFinalForecastWriter(
        final_path,
        target_frames=frames,
        metadata=_metadata(),
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as final_writer:
        with PDERefinerStageForecastWriter(
            stage_path,
            target_frames=frames,
            metadata=_metadata(),
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_digest,
            schema=schema,
        ) as stage_writer:
            for offset, frame in enumerate(frames):
                digest = _array_sha256(bank[offset])
                final_writer.append(
                    target_frame=frame,
                    standardized_forecast=final + offset,
                    inference_seconds=1.0 + offset,
                    seed_row_sha256=digest,
                )
                stage_writer.append(
                    target_frame=frame,
                    standardized_forecast=stages + offset,
                    inference_seconds=1.0 + offset,
                    seed_row_sha256=digest,
                )
            stage_writer.finalize()
            final_writer.finalize()

    with PDERefinerFinalForecastArtifact(
        final_path,
        expected_sha256=sha256_path(final_path),
        target_frames=frames,
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as final_artifact:
        with PDERefinerStageForecastArtifact(
            stage_path,
            expected_sha256=sha256_path(stage_path),
            target_frames=frames,
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_digest,
            schema=schema,
        ) as stage_artifact:
            observed_final = final_artifact.read(0, 1)[0]
            observed_stages = stage_artifact.read(0, 1)[0]
            assert np.array_equal(
                observed_final[: schema.stage_members, 0],
                observed_stages[:, 3],
            )
            assert final_artifact.metadata == _metadata()
            assert final_artifact.timing_record()["total_seconds"] == 3.0


def test_b4_writer_rejects_heldout_order_and_nonfinite(tmp_path: Path) -> None:
    schema = _tiny_schema()
    bank, bank_path, bank_digest = _saved_bank(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        PDERefinerFinalForecastWriter(
            tmp_path / "forbidden.h5",
            target_frames=(498,),
            metadata={"source": "/secret/85606/data.h5"},
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_digest,
            schema=schema,
        )
    values = np.zeros(schema.final_per_target_shape, dtype=np.float32)
    with PDERefinerFinalForecastWriter(
        tmp_path / "order.h5",
        target_frames=(498, 499),
        metadata=_metadata(),
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="differs"):
            writer.append(
                target_frame=499,
                standardized_forecast=values,
                inference_seconds=0.1,
                seed_row_sha256=_array_sha256(bank[1]),
            )
    values[0, 0, 0, 0, 0, 0] = np.nan
    with PDERefinerFinalForecastWriter(
        tmp_path / "nonfinite.h5",
        target_frames=(498,),
        metadata=_metadata(),
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="finite"):
            writer.append(
                target_frame=498,
                standardized_forecast=values,
                inference_seconds=0.1,
                seed_row_sha256=_array_sha256(bank[0]),
            )


def test_b4_artifact_rejects_tampered_seed_row_hash(tmp_path: Path) -> None:
    schema = _tiny_schema()
    bank, bank_path, bank_digest = _saved_bank(tmp_path)
    path = tmp_path / "tampered.h5"
    with PDERefinerFinalForecastWriter(
        path,
        target_frames=(498,),
        metadata=_metadata(),
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_digest,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=np.zeros(
                schema.final_per_target_shape, dtype=np.float32
            ),
            inference_seconds=0.1,
            seed_row_sha256=_array_sha256(bank[0]),
        )
        writer.finalize()
    with h5py.File(path, "r+") as handle:
        handle["seed_row_sha256"][0] = ("0" * 64).encode("ascii")
    with pytest.raises(ValueError, match="seed-row hashes"):
        PDERefinerFinalForecastArtifact(
            path,
            expected_sha256=sha256_path(path),
            target_frames=(498,),
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_digest,
            schema=schema,
        )


def test_selected_b4_loader_rejects_hash_before_torch_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    artifacts = RefinerParentArtifacts(
        checkpoint_path=tmp_path / "parent.pt",
        checkpoint_sha256="1" * 64,
        codec_path=tmp_path / "codec.pt",
        codec_sha256="2" * 64,
        latent_normalization_path=tmp_path / "latent.json",
        latent_normalization_sha256="3" * 64,
    )
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_selected_pde_refiner_model(
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            artifacts=artifacts,
            device=torch.device("cpu"),
            training_commit="a" * 40,
        )


def test_b4_generator_source_freezes_truth_lock_both_artifacts_and_prefix() -> None:
    source = inspect.getsource(generate_selected_pde_refiner_forecasts)
    assert "tuple(range(498, 502))" in source
    assert "range(B4_SCIENTIFIC_TARGET_START, B4_SCIENTIFIC_TARGET_STOP)" in source
    assert "dataset.target_truth_read is not False" in source
    assert "PDERefinerFinalForecastWriter" in source
    assert "PDERefinerStageForecastWriter" in source
    assert source.index("stage_writer.finalize()") < source.index(
        "final_writer.finalize()"
    )
    assert "member_prefixes_regenerated" in source
