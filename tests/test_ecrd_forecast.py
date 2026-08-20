"""Canonical axes, seed pairing, and truth-lock tests for ECRD forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcv_diagnostics.b5_residual_edm_forecast import (
    B5ForecastSchema,
    initial_noise_from_uint64,
    save_scientific_sampler_seed_bank,
)
from tcv_diagnostics.b5_residual_edm_full_training import (
    scientific_sampler_seed_bank,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_forecast import (
    ECRDForecastArtifact,
    ECRDForecastWriter,
    initial_noise_for_arm,
)


def _seed_bank(tmp_path: Path) -> tuple[Path, str, np.ndarray]:
    values = scientific_sampler_seed_bank()
    path = tmp_path / "scientific_sampler_seed_bank.npy"
    digest = save_scientific_sampler_seed_bank(path, values)
    return path, digest, values


def test_arm_noise_is_paired_repeatable_and_not_posthoc_scaled() -> None:
    expected = initial_noise_from_uint64(1234, spatial_shape=(8, 8, 12))
    observed = initial_noise_for_arm(
        1234, arm="B5-Context", spatial_shape=(8, 8, 12)
    )
    np.testing.assert_array_equal(observed, expected)
    first = initial_noise_for_arm(1234, arm="ECRD", spatial_shape=(8, 8, 12))
    second = initial_noise_for_arm(1234, arm="ECRD", spatial_shape=(8, 8, 12))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, expected)


def test_forecast_writer_round_trip_keeps_canonical_axes(tmp_path: Path) -> None:
    bank_path, bank_sha, bank = _seed_bank(tmp_path)
    schema = B5ForecastSchema(members=2, volume_shape=(2, 3, 4))
    output = tmp_path / "forecast.h5"
    with ECRDForecastWriter(
        output,
        target_frames=(498,),
        arm="ECRD",
        model_seed=1702,
        history_frames=1,
        parent_kind="four_phase_symmetrized_H1",
        metadata={"target_truth_read": False},
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_sha,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=np.zeros(schema.per_target_shape, dtype=np.float32),
            inference_seconds=0.5,
            sampler_seed_row=bank[0, :2],
            initial_noise_sha256=("a" * 64, "b" * 64),
        )
        writer.finalize()
    with ECRDForecastArtifact(
        output,
        expected_sha256=sha256_path(output),
        target_frames=(498,),
        arm="ECRD",
        model_seed=1702,
        seed_bank_path=bank_path,
        seed_bank_sha256=bank_sha,
        schema=schema,
    ) as artifact:
        assert artifact.read(0, 1).shape == (1, 2, 1, 5, 2, 3, 4)
        assert artifact.metadata["target_truth_read"] is False


def test_forecast_writer_rejects_wrong_parent_and_held_out_metadata(
    tmp_path: Path,
) -> None:
    bank_path, bank_sha, _ = _seed_bank(tmp_path)
    schema = B5ForecastSchema(members=2, volume_shape=(2, 3, 4))
    with pytest.raises(ValueError, match="parent"):
        ECRDForecastWriter(
            tmp_path / "wrong_parent.h5",
            target_frames=(498,),
            arm="ECRD",
            model_seed=1701,
            history_frames=1,
            parent_kind="original_unsymmetrized_H1",
            metadata={"target_truth_read": False},
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_sha,
            schema=schema,
        )
    with pytest.raises(ValueError, match="held-out"):
        ECRDForecastWriter(
            tmp_path / "held_out.h5",
            target_frames=(498,),
            arm="B5-Context",
            model_seed=1701,
            history_frames=1,
            parent_kind="original_unsymmetrized_H1",
            metadata={"source": "85606"},
            seed_bank_path=bank_path,
            seed_bank_sha256=bank_sha,
            schema=schema,
        )
