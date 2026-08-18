from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tcv_diagnostics.b2_forecast import (
    B2_EVALUATION_SEED_TAG,
    B2_FORECAST_AXES,
    B2ForecastArtifact,
    B2ForecastSchema,
    B2ForecastWriter,
    _initial_sample_from_standard_normal,
    initial_standard_normal,
    sampler_seed,
)
from tcv_diagnostics.codec_training import sha256_path


def _tiny_schema() -> B2ForecastSchema:
    return B2ForecastSchema(
        members=3,
        future_frames=1,
        channels=2,
        volume_shape=(2, 3, 4),
        latent_channels=2,
        trajectory_frames=3,
        latent_shape=(1, 2, 2),
    )


def test_frozen_b2_forecast_schema_has_canonical_axes_and_shape():
    schema = B2ForecastSchema.frozen()
    assert B2_FORECAST_AXES == (
        "target_frame",
        "ensemble_member",
        "future_time",
        "channel",
        "x",
        "y",
        "stored_toroidal_z",
    )
    assert schema.per_target_shape == (32, 1, 5, 64, 32, 88)
    assert schema.initial_noise_shape == (32, 32, 3, 16, 8, 22)


def test_cpu_initial_noise_is_exact_reproducible_and_target_separated():
    first, first_seed, first_hash = initial_standard_normal(
        model_seed=1701,
        target_frame=498,
        shape=(2, 3),
    )
    repeated, repeated_seed, repeated_hash = initial_standard_normal(
        model_seed=1701,
        target_frame=498,
        shape=(2, 3),
    )
    other, other_seed, other_hash = initial_standard_normal(
        model_seed=1701,
        target_frame=499,
        shape=(2, 3),
    )

    assert B2_EVALUATION_SEED_TAG == 1_110_590_806
    assert first_seed == repeated_seed == 14_989_464_630_366_940_288
    assert first_hash == repeated_hash == (
        "8b06b631587ee4ff65410578258dd1b58087328d1cbdce087d5de4c2b4dec810"
    )
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_allclose(
        first,
        np.asarray(
            [
                [-2.5382845, 1.1250906, 0.3853605],
                [1.9638722, -0.5622241, -0.33607596],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert other_seed != first_seed
    assert other_hash != first_hash
    assert not np.array_equal(other, first)


def test_sampler_seed_rejects_unfrozen_model_seed():
    with pytest.raises(ValueError, match="three-seed"):
        sampler_seed(42, 498)


def test_source_freezes_a_distinct_four_target_non_scientific_smoke() -> None:
    import inspect
    from tcv_diagnostics.b2_forecast import generate_selected_b2_forecasts

    source = inspect.getsource(generate_selected_b2_forecasts)
    assert "tuple(range(498, 502))" in source
    assert "bounded_non_scientific_B2_LDM_H2_M32_forecast_smoke_85604" in source
    assert "bounded_smoke: bool = False" in source


def test_external_initialization_matches_azula_default_unit_prior_formula():
    class Schedule:
        def __call__(self, time: torch.Tensor):
            assert float(time) == 1.0
            return torch.tensor(2.0), torch.tensor(3.0)

    model = SimpleNamespace(
        denoiser=SimpleNamespace(schedule=Schedule()),
    )
    normal = torch.tensor([[[[[[1.0]]]]], [[[[[-2.0]]]]]])
    actual = _initial_sample_from_standard_normal(
        standard_normal=normal,
        model=model,
    )
    expected = torch.sqrt(torch.tensor(13.0)) * normal
    torch.testing.assert_close(actual, expected)


def test_b2_writer_reader_lock_order_axes_seed_hash_and_timing(tmp_path: Path):
    schema = _tiny_schema()
    path = tmp_path / "b2_forecast.h5"
    frames = (498, 499)
    metadata = {"arm": "B2-LDM-H2", "checkpoint": "selected.pt"}
    first = np.zeros(schema.per_target_shape, dtype=np.float32)
    second = np.ones_like(first)
    with B2ForecastWriter(
        path,
        target_frames=frames,
        model_seed=1701,
        metadata=metadata,
        schema=schema,
    ) as writer:
        writer.append(
            target_frame=498,
            standardized_forecast=first,
            inference_seconds=1.25,
            sampler_seed_uint64=sampler_seed(1701, 498),
            initial_noise_sha256="a" * 64,
        )
        writer.append(
            target_frame=499,
            standardized_forecast=second,
            inference_seconds=2.75,
            sampler_seed_uint64=sampler_seed(1701, 499),
            initial_noise_sha256="b" * 64,
        )
        writer.finalize()

    with B2ForecastArtifact(
        path,
        expected_sha256=sha256_path(path),
        target_frames=frames,
        model_seed=1701,
        schema=schema,
    ) as artifact:
        np.testing.assert_array_equal(artifact.read(0, 1), first[None])
        np.testing.assert_array_equal(artifact.read(1, 2), second[None])
        assert artifact.metadata == metadata
        timing = artifact.timing_record()
        assert timing["target_count"] == 2
        assert timing["ensemble_members_per_target"] == 3
        assert timing["total_seconds"] == pytest.approx(4.0)


def test_b2_writer_rejects_reordering_seed_nonfinite_and_held_out(tmp_path: Path):
    schema = _tiny_schema()
    values = np.zeros(schema.per_target_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="held-out"):
        B2ForecastWriter(
            tmp_path / "forbidden.h5",
            target_frames=(498,),
            model_seed=1701,
            metadata={"source": "/secret/85606/file.h5"},
            schema=schema,
        )

    with B2ForecastWriter(
        tmp_path / "order.h5",
        target_frames=(498, 499),
        model_seed=1701,
        metadata={"arm": "B2-LDM-H2"},
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="differs"):
            writer.append(
                target_frame=499,
                standardized_forecast=values,
                inference_seconds=0.1,
                sampler_seed_uint64=sampler_seed(1701, 499),
                initial_noise_sha256="a" * 64,
            )
        with pytest.raises(RuntimeError, match="every target"):
            writer.finalize()

    nonfinite = values.copy()
    nonfinite[0, 0, 0, 0, 0, 0] = np.nan
    with B2ForecastWriter(
        tmp_path / "bad_values.h5",
        target_frames=(498,),
        model_seed=1701,
        metadata={"arm": "B2-LDM-H2"},
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="finite"):
            writer.append(
                target_frame=498,
                standardized_forecast=nonfinite,
                inference_seconds=0.1,
                sampler_seed_uint64=sampler_seed(1701, 498),
                initial_noise_sha256="a" * 64,
            )

    with B2ForecastWriter(
        tmp_path / "bad_seed.h5",
        target_frames=(498,),
        model_seed=1701,
        metadata={"arm": "B2-LDM-H2"},
        schema=schema,
    ) as writer:
        with pytest.raises(ValueError, match="sampler seed"):
            writer.append(
                target_frame=498,
                standardized_forecast=values,
                inference_seconds=0.1,
                sampler_seed_uint64=1,
                initial_noise_sha256="a" * 64,
            )
