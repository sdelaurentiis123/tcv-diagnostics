"""Known-answer and fail-closed tests for full-only B4 training."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tcv_diagnostics.pde_refiner_full_training import (
    B4_FULL_LEVEL_COUNTS,
    B4_FULL_LEVEL_RAW_SHA256,
    PDERefinerFullConfig,
    full_learning_rate,
    full_training_levels,
    train_pde_refiner_full,
)
from tcv_diagnostics.pde_refiner_training import (
    PDERefinerSmokeConfig,
    RefinerParentArtifacts,
)


def dummy_artifacts(tmp_path: Path) -> RefinerParentArtifacts:
    return RefinerParentArtifacts(
        checkpoint_path=tmp_path / "parent.pt",
        checkpoint_sha256="0" * 64,
        codec_path=tmp_path / "codec.pt",
        codec_sha256="0" * 64,
        latent_normalization_path=tmp_path / "normalization.json",
        latent_normalization_sha256="0" * 64,
    )


def test_full_budget_is_exactly_the_frozen_seed1701_contract() -> None:
    config = PDERefinerFullConfig.frozen(seed=1701)
    assert config.epochs == 100
    assert config.train_targets == tuple(range(2, 432))
    assert config.validation_targets == tuple(range(498, 624))
    assert config.microbatch_targets == 1
    assert config.gradient_accumulation_targets == 16
    assert config.validation_members == 2
    assert config.optimizer_steps_per_epoch == 27
    assert config.total_optimizer_steps == 2700
    assert config.validation_completed_epochs == tuple(range(5, 101, 5))
    assert config.peak_learning_rate == 1e-4
    assert config.minimum_learning_rate == 1e-6
    assert config.betas == (0.9, 0.999)
    assert config.weight_decay == 1e-5
    assert config.ema_decay == 0.995
    assert config.training_precision == "float32_no_autocast_TF32_disabled"
    record = config.to_record()
    assert record["mode"] == "full"
    assert record["full_training_authorized"] is True
    assert record["scientific_result"] is False
    assert record["training_complete_is_scientific_acceptance"] is False
    assert record["physics_derived_loss_allowed"] is False
    assert record["absolute_time_input_allowed"] is False
    with pytest.raises(ValueError, match="seed 1701"):
        PDERefinerFullConfig.frozen(seed=1702)


def test_inclusive_cosine_schedule_has_exact_endpoints_and_is_monotone() -> None:
    config = PDERefinerFullConfig.frozen(seed=1701)
    values = np.asarray(
        [full_learning_rate(config, step) for step in range(2700)],
        dtype=np.float64,
    )
    assert values[0] == 1e-4
    assert values[-1] == 1e-6
    assert np.all(np.diff(values) < 0.0)
    assert values[1349] > values[1350]
    with pytest.raises(ValueError, match="outside"):
        full_learning_rate(config, -1)
    with pytest.raises(ValueError, match="outside"):
        full_learning_rate(config, 2700)
    with pytest.raises(ValueError, match="frozen full config"):
        full_learning_rate(replace(config, epochs=101), 0)


def test_full_level_matrix_matches_prospective_bytes_counts_and_prefix() -> None:
    config = PDERefinerFullConfig.frozen(seed=1701)
    values = full_training_levels(config)
    assert values.shape == (100, 430)
    assert values.dtype == np.int64
    assert values.flags.c_contiguous
    assert hashlib.sha256(values.tobytes(order="C")).hexdigest() == (
        B4_FULL_LEVEL_RAW_SHA256
    )
    assert tuple(np.bincount(values.reshape(-1), minlength=4)) == (
        B4_FULL_LEVEL_COUNTS
    )
    assert values[0, :16].tolist() == [
        3,
        2,
        0,
        0,
        2,
        1,
        0,
        3,
        0,
        1,
        3,
        1,
        1,
        3,
        1,
        1,
    ]
    assert all(set(map(int, np.unique(row))) == {0, 1, 2, 3} for row in values)


def test_full_wrapper_rejects_smoke_or_modified_budget_before_writes(
    tmp_path: Path,
) -> None:
    artifacts = dummy_artifacts(tmp_path)
    smoke = PDERefinerSmokeConfig.frozen(seed=1701)
    with pytest.raises(ValueError, match="100-epoch budget"):
        train_pde_refiner_full(
            config=smoke,  # type: ignore[arg-type]
            catalog=None,  # type: ignore[arg-type]
            artifacts=artifacts,
            output_directory=tmp_path / "smoke",
            paper0_commit="deadbeef",
            slurm_job_id="0",
            device=torch.device("cpu"),
        )
    modified = replace(PDERefinerFullConfig.frozen(seed=1701), epochs=101)
    with pytest.raises(ValueError, match="100-epoch budget"):
        train_pde_refiner_full(
            config=modified,
            catalog=None,  # type: ignore[arg-type]
            artifacts=artifacts,
            output_directory=tmp_path / "modified",
            paper0_commit="deadbeef",
            slurm_job_id="0",
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "smoke").exists()
    assert not (tmp_path / "modified").exists()


def test_full_source_does_not_expose_smoke_or_scientific_scoring() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/tcv_diagnostics/pde_refiner_full_training.py"
    ).read_text(encoding="utf-8")
    assert "def train_pde_refiner_full(" in source
    assert "def train_pde_refiner_smoke(" not in source
    assert "85606" in source
    assert '"held_out_85606_read": False' in source
    assert '"H_det_evaluated": False' in source
    assert '"H_prob_evaluated": False' in source
    assert '"assimilation_allowed": False' in source
    assert '"diagnostic_ranking_allowed": False' in source
