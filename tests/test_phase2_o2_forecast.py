from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_training_data import VOLUME_SHAPE
from tcv_diagnostics.o2_forecast import O2ForecastArtifact, O2ForecastWriter


def test_forecast_writer_and_reader_lock_order_axes_hash_and_timing(tmp_path: Path):
    path = tmp_path / "forecast.h5"
    frames = (498, 499)
    metadata = {
        "arm": "C5P-H1",
        "seed": 1701,
        "held_out_85606_read": False,
    }
    first = np.zeros((5, *VOLUME_SHAPE), dtype=np.float32)
    second = np.ones_like(first)
    with O2ForecastWriter(path, target_frames=frames, metadata=metadata) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=first,
            inference_seconds=0.1,
        )
        writer.append(
            target_frame=499,
            standardized_forecast=second,
            inference_seconds=0.2,
        )
        writer.finalize()

    digest = sha256_path(path)
    with O2ForecastArtifact(
        path,
        expected_sha256=digest,
        target_frames=frames,
    ) as artifact:
        np.testing.assert_array_equal(artifact.read(0, 1), first[None])
        np.testing.assert_array_equal(artifact.read(1, 2), second[None])
        timing = artifact.timing_record()
        assert timing["target_count"] == 2
        assert timing["total_seconds"] == pytest.approx(0.3)
        assert artifact.metadata == metadata


def test_forecast_writer_rejects_reordering_nonfinite_and_incomplete_stream(tmp_path: Path):
    values = np.zeros((5, *VOLUME_SHAPE), dtype=np.float32)
    with O2ForecastWriter(
        tmp_path / "order.h5",
        target_frames=(498, 499),
        metadata={"arm": "C5P-H1"},
    ) as writer:
        with pytest.raises(ValueError, match="differs"):
            writer.append(
                target_frame=499,
                standardized_forecast=values,
                inference_seconds=0.1,
            )
        with pytest.raises(RuntimeError, match="every target"):
            writer.finalize()

    nonfinite = values.copy()
    nonfinite[0, 0, 0, 0] = np.nan
    with O2ForecastWriter(
        tmp_path / "finite.h5",
        target_frames=(498,),
        metadata={"arm": "C5P-H1"},
    ) as writer:
        with pytest.raises(ValueError, match="finite"):
            writer.append(
                target_frame=498,
                standardized_forecast=nonfinite,
                inference_seconds=0.1,
            )


def test_forecast_artifact_rejects_wrong_hash_and_held_out_value(tmp_path: Path):
    with pytest.raises(ValueError, match="held-out"):
        O2ForecastWriter(
            tmp_path / "forbidden.h5",
            target_frames=(498,),
            metadata={"source": "/tmp/85606/checkpoint.pt"},
        )

    path = tmp_path / "valid.h5"
    values = np.zeros((5, *VOLUME_SHAPE), dtype=np.float32)
    with O2ForecastWriter(
        path,
        target_frames=(498,),
        metadata={"arm": "C5P-H1"},
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=values,
            inference_seconds=0.0,
        )
        writer.finalize()
    with pytest.raises(ValueError, match="SHA-256"):
        O2ForecastArtifact(
            path,
            expected_sha256="0" * 64,
            target_frames=(498,),
        )
