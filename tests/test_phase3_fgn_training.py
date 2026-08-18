"""Regression tests for bounded B3 FGN training mechanics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.fgn_training import (
    FGNRunConfig,
    ParentArtifacts,
    _parameter_groups,
    _verify_parent_payload,
    learning_rate_at_step,
    save_validation_noise_bank,
    train_fgn_smoke,
    validation_noise_bank,
)
from tcv_diagnostics.model_data import write_strict_json_atomic
from tcv_diagnostics.models.functional_noise import (
    FunctionalNoiseConfig,
    FunctionalNoiseMaskedLatentTransition,
)
from tcv_diagnostics.models.o2 import O2ViTConfig


def _tiny_model_config() -> O2ViTConfig:
    return O2ViTConfig(
        latent_channels=2,
        hidden_channels=8,
        transformer_blocks=2,
        attention_heads=2,
        ffn_factor=2,
        latent_patch=(1, 1, 1),
        qk_normalization=True,
        rope=True,
        dropout=0.0,
        activation_checkpointing=False,
    )


def test_frozen_smoke_and_prospective_full_budgets() -> None:
    smoke = FGNRunConfig.frozen(mode="smoke", seed=1701)
    assert smoke.epochs == 2
    assert smoke.train_targets == tuple(range(2, 18))
    assert smoke.validation_targets == tuple(range(498, 502))
    assert smoke.ensemble_members == 2
    assert smoke.optimizer_steps_per_epoch == 1
    assert smoke.total_optimizer_steps == 2
    assert smoke.to_record()["scientific_result"] is False
    assert smoke.to_record()["full_training_authorized"] is False

    full = FGNRunConfig.frozen(mode="full", seed=1701)
    assert full.epochs == 100
    assert len(full.train_targets) == 430
    assert len(full.validation_targets) == 126
    assert full.optimizer_steps_per_epoch == 27
    assert full.total_optimizer_steps == 2700
    assert full.warmup_optimizer_steps == 270
    assert full.to_record()["prospective_full_budget"] is True
    assert full.to_record()["full_training_authorized"] is False
    with pytest.raises(ValueError, match="seed 1701"):
        FGNRunConfig.frozen(mode="smoke", seed=1702)


def test_staged_learning_rate_known_answers() -> None:
    smoke = FGNRunConfig.frozen(mode="smoke", seed=1701)
    for step in (1, 2):
        assert learning_rate_at_step(smoke, step, group="common") == 3.0e-5
        assert learning_rate_at_step(smoke, step, group="new") == 1.0e-4

    full = FGNRunConfig.frozen(mode="full", seed=1701)
    assert learning_rate_at_step(full, 1, group="common") == pytest.approx(
        3.0e-5 / 270
    )
    assert learning_rate_at_step(full, 270, group="common") == pytest.approx(3.0e-5)
    assert learning_rate_at_step(full, 270, group="new") == pytest.approx(1.0e-4)
    assert learning_rate_at_step(full, 2700, group="common") == pytest.approx(0.0)
    assert learning_rate_at_step(full, 2700, group="new") == pytest.approx(0.0)
    with pytest.raises(ValueError, match="outside"):
        learning_rate_at_step(full, 0, group="common")
    with pytest.raises(ValueError, match="unknown"):
        learning_rate_at_step(full, 1, group="codec")


def test_validation_noise_bank_is_frozen_and_roundtrips(tmp_path: Path) -> None:
    first = validation_noise_bank()
    second = validation_noise_bank()
    assert first.shape == (126, 2, 32)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    np.testing.assert_allclose(
        first.reshape(-1)[:4],
        np.asarray(
            [
                0.7661535739898682,
                -1.1828441619873047,
                0.20746085047721863,
                -0.7368801832199097,
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert hashlib.sha256(first.tobytes()).hexdigest() == (
        "48649979e4dab64ebc28d28e124947bd9aadaecfdceefa3f55c5710d5e9b3786"
    )
    path = tmp_path / "validation_noise.npy"
    digest = save_validation_noise_bank(path, first)
    assert digest == sha256_path(path)
    assert np.array_equal(np.load(path, allow_pickle=False), first)
    with pytest.raises(FileExistsError):
        save_validation_noise_bank(path, first)
    with pytest.raises(ValueError, match="seed differs"):
        validation_noise_bank(seed=1)


def test_staged_parameter_groups_cover_transition_once() -> None:
    transition = FunctionalNoiseMaskedLatentTransition(
        context_frames=1,
        config=_tiny_model_config(),
        noise_config=FunctionalNoiseConfig(
            raw_noise_features=3,
            embedded_noise_features=6,
        ),
    )
    groups = _parameter_groups(transition)
    assert groups.common
    assert groups.new
    assert all(
        name.startswith("noise_embedding.") or ".noise_adapter." in name
        for name in groups.new_names
    )
    assert all(
        not name.startswith("noise_embedding.") and ".noise_adapter." not in name
        for name in groups.common_names
    )
    assert {id(item) for item in groups.all_parameters} == {
        id(item) for item in transition.parameters()
    }
    assert groups.to_record()["total_parameter_count"] == sum(
        item.numel() for item in transition.parameters()
    )


def test_parent_artifact_verification_is_hash_and_metadata_strict(
    tmp_path: Path,
) -> None:
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"frozen-codec-placeholder")
    codec_sha = sha256_path(codec)
    normalization = {
        "schema_version": 1,
        "kind": "per_latent_channel_training_only_population_moments",
        "mean": [0.1, -0.2],
        "population_standard_deviation": [0.7, 1.3],
        "sample_count_per_channel": 1,
        "fit_frames": [0, 432],
        "codec_checkpoint_sha256": codec_sha,
        "scientific_authority": True,
        "held_out_85606_read": False,
    }
    normalization_path = tmp_path / "latent_normalization.json"
    write_strict_json_atomic(normalization_path, normalization)
    model_config = _tiny_model_config()
    parent_path = tmp_path / "parent.pt"
    torch.save(
        {
            "kind": "selected_O2_transition",
            "config": {"arm": "C5P-H1", "seed": 1701},
            "epoch": 193,
            "global_step": 5238,
            "validation_loss": 0.04558250684515488,
            "model_config": model_config.to_record(),
            "codec_checkpoint": {
                "path": str(codec),
                "sha256": codec_sha,
                "trainable": False,
            },
            "latent_normalization": normalization,
            "transition_state": {},
        },
        parent_path,
    )
    artifacts = ParentArtifacts(
        checkpoint_path=parent_path,
        checkpoint_sha256=sha256_path(parent_path),
        codec_path=codec,
        codec_sha256=codec_sha,
        latent_normalization_path=normalization_path,
        latent_normalization_sha256=sha256_path(normalization_path),
    )
    payload, observed_normalization = _verify_parent_payload(
        artifacts=artifacts,
        model_config=model_config,
    )
    assert payload["kind"] == "selected_O2_transition"
    assert observed_normalization == normalization

    bad = replace(artifacts, checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _verify_parent_payload(artifacts=bad, model_config=model_config)


def test_smoke_entrypoint_rejects_full_or_modified_budget_before_writes(
    tmp_path: Path,
) -> None:
    dummy = ParentArtifacts(
        checkpoint_path=tmp_path / "parent.pt",
        checkpoint_sha256="0" * 64,
        codec_path=tmp_path / "codec.pt",
        codec_sha256="0" * 64,
        latent_normalization_path=tmp_path / "normalization.json",
        latent_normalization_sha256="0" * 64,
    )
    full = FGNRunConfig.frozen(mode="full", seed=1701)
    with pytest.raises(ValueError, match="bounded budget"):
        train_fgn_smoke(
            config=full,
            catalog=None,  # type: ignore[arg-type]
            artifacts=dummy,
            output_directory=tmp_path / "full",
            paper0_commit="deadbeef",
            slurm_job_id="0",
            device=torch.device("cpu"),
        )
    modified = replace(
        FGNRunConfig.frozen(mode="smoke", seed=1701),
        epochs=3,
    )
    with pytest.raises(ValueError, match="bounded budget"):
        train_fgn_smoke(
            config=modified,
            catalog=None,  # type: ignore[arg-type]
            artifacts=dummy,
            output_directory=tmp_path / "modified",
            paper0_commit="deadbeef",
            slurm_job_id="0",
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "full").exists()
    assert not (tmp_path / "modified").exists()
