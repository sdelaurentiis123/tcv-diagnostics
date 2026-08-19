"""Truth-separation and artifact tests for the B5 H1 training forecast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcv_diagnostics.b5_residual_forecast import (
    B5TrainingForecastArtifact,
    B5TrainingForecastWriter,
)
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_training_data import VOLUME_SHAPE


def test_b5_training_forecast_roundtrip_locks_truth_axes_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "forecast.h5"
    values = np.zeros((5, *VOLUME_SHAPE), dtype=np.float32)
    metadata = {"arm": "C5P-H1", "seed": 1701, "target_truth_read": False}
    with B5TrainingForecastWriter(
        path, target_frames=(2, 3), metadata=metadata
    ) as writer:
        writer.append(target_frame=2, standardized_forecast=values, inference_seconds=0.1)
        writer.append(target_frame=3, standardized_forecast=values + 1, inference_seconds=0.2)
        writer.finalize()
    digest = sha256_path(path)
    with B5TrainingForecastArtifact(
        path, expected_sha256=digest, target_frames=(2, 3)
    ) as artifact:
        np.testing.assert_array_equal(artifact.read(0, 1), values[None])
        np.testing.assert_array_equal(artifact.read(1, 2), (values + 1)[None])
        assert artifact.metadata == metadata
        assert artifact.timing_record()["total_seconds"] == pytest.approx(0.3)


def test_b5_writer_rejects_heldout_metadata_reordering_and_incomplete_stream(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="held-out"):
        B5TrainingForecastWriter(
            tmp_path / "heldout.h5",
            target_frames=(2,),
            metadata={"source": "/forbidden/85606", "target_truth_read": False},
        )
    with pytest.raises(ValueError, match="truth"):
        B5TrainingForecastWriter(
            tmp_path / "truth.h5",
            target_frames=(2,),
            metadata={"target_truth_read": True},
        )
    values = np.zeros((5, *VOLUME_SHAPE), dtype=np.float32)
    with B5TrainingForecastWriter(
        tmp_path / "order.h5",
        target_frames=(2, 3),
        metadata={"target_truth_read": False},
    ) as writer:
        with pytest.raises(ValueError, match="differs"):
            writer.append(
                target_frame=3,
                standardized_forecast=values,
                inference_seconds=0.0,
            )
        with pytest.raises(RuntimeError, match="every target"):
            writer.finalize()


def test_b5_artifact_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "forecast.h5"
    values = np.zeros((5, *VOLUME_SHAPE), dtype=np.float32)
    with B5TrainingForecastWriter(
        path,
        target_frames=(2,),
        metadata={"target_truth_read": False},
    ) as writer:
        writer.append(
            target_frame=2,
            standardized_forecast=values,
            inference_seconds=0.0,
        )
        writer.finalize()
    with pytest.raises(ValueError, match="SHA-256"):
        B5TrainingForecastArtifact(
            path,
            expected_sha256="0" * 64,
            target_frames=(2,),
        )
