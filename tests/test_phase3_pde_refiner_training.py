"""Regression tests for the bounded B4 PDE-Refiner smoke mechanics."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import write_strict_json_atomic
from tcv_diagnostics.models.o2 import O2ViTConfig
from tcv_diagnostics.models.pde_refiner import (
    PDERefinerConfig,
    PDERefinerMaskedLatentTransition,
)
from tcv_diagnostics.pde_refiner_training import (
    B4_LATENT_SHAPE,
    PDERefinerSmokeConfig,
    RefinerParentArtifacts,
    TransitionEMA,
    _parameter_groups,
    _verify_parent_payload,
    refinement_noise_from_seeds,
    refinement_training_pair,
    save_numpy_exclusive,
    smoke_training_levels,
    train_pde_refiner_smoke,
    validation_seed_bank,
)


def tiny_model_config() -> O2ViTConfig:
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


def test_only_frozen_bounded_smoke_budget_is_executable() -> None:
    config = PDERefinerSmokeConfig.frozen(seed=1701)
    assert config.epochs == 2
    assert config.train_targets == tuple(range(2, 18))
    assert config.validation_targets == tuple(range(498, 502))
    assert config.validation_members == 2
    assert config.optimizer_steps_per_epoch == 1
    assert config.total_optimizer_steps == 2
    assert config.learning_rate == 1e-4
    assert config.betas == (0.9, 0.999)
    assert config.weight_decay == 1e-5
    assert config.ema_decay == 0.995
    assert config.training_precision == "float32_no_autocast_TF32_disabled"
    record = config.to_record()
    assert record["scientific_result"] is False
    assert record["full_training_authorized"] is False
    assert record["physics_derived_loss_allowed"] is False
    assert record["absolute_time_input_allowed"] is False
    with pytest.raises(ValueError, match="seed 1701"):
        PDERefinerSmokeConfig.frozen(seed=1702)


def test_training_level_draws_are_frozen_and_exercise_every_level() -> None:
    levels = smoke_training_levels(PDERefinerSmokeConfig.frozen(seed=1701))
    expected = np.asarray(
        [
            [3, 2, 0, 0, 2, 1, 0, 3, 0, 1, 3, 1, 1, 3, 1, 1],
            [2, 0, 1, 3, 3, 1, 0, 1, 0, 1, 2, 2, 3, 2, 0, 2],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(levels, expected)
    assert np.bincount(levels.reshape(-1), minlength=4).tolist() == [8, 10, 7, 7]
    assert all(set(map(int, row)) == {0, 1, 2, 3} for row in levels)
    assert hashlib.sha256(levels.tobytes()).hexdigest() == (
        "1fe9416df9f7ca272b3111815800a2f3efab4a6ed7e5c29353d6b05c5472570e"
    )


def test_validation_seed_bank_and_full_latent_noise_are_reproducible() -> None:
    first = validation_seed_bank()
    second = validation_seed_bank()
    assert first.shape == (126, 2, 3)
    assert first.dtype == np.uint64
    assert np.array_equal(first, second)
    assert first.reshape(-1)[:4].tolist() == [
        8937439276147795681,
        577436771100881191,
        16013722266247506046,
        5935315524604458904,
    ]
    assert hashlib.sha256(first.tobytes()).hexdigest() == (
        "85409dcad8eb2800bcd703a35aee502c59add718cd5956780f2ade7555f544ca"
    )
    noise1 = refinement_noise_from_seeds(first[0])
    noise2 = refinement_noise_from_seeds(first[0])
    assert noise1.shape == (2, 3, *B4_LATENT_SHAPE)
    assert noise1.dtype == np.float32
    assert np.array_equal(noise1, noise2)
    np.testing.assert_array_equal(
        noise1[0, 0].reshape(-1)[:3],
        np.asarray(
            [-0.37839895486831665, 1.2604801654815674, 0.678612232208252],
            dtype=np.float32,
        ),
    )
    assert not np.array_equal(noise1[0, 0], noise1[0, 1])
    assert not np.array_equal(noise1[0, 0], noise1[1, 0])
    with pytest.raises(ValueError, match="seed-bank seed differs"):
        validation_seed_bank(seed=1)
    with pytest.raises(ValueError, match="uint64"):
        refinement_noise_from_seeds(first[0].astype(np.int64))


def test_numpy_artifacts_are_exclusive_and_hash_stable(tmp_path: Path) -> None:
    path = tmp_path / "levels.npy"
    levels = smoke_training_levels(PDERefinerSmokeConfig.frozen(seed=1701))
    digest = save_numpy_exclusive(path, levels)
    assert digest == sha256_path(path)
    assert digest == "799443c5c4102f4587a24a701aa7ac6644465ddd9755904746013534dfaacdaa"
    assert np.array_equal(np.load(path, allow_pickle=False), levels)
    with pytest.raises(FileExistsError):
        save_numpy_exclusive(path, levels)


def test_mixed_level_training_pair_matches_explicit_formulas() -> None:
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(2, 1, 2)
    target = torch.tensor([[2.0, 0.0], [7.0, 8.0]]).reshape(2, 1, 2)
    noise = torch.tensor([[9.0, -2.0], [0.5, -1.5]]).reshape(2, 1, 2)
    levels = torch.tensor([0, 2], dtype=torch.int64)
    sigmas = PDERefinerConfig().standard_deviations
    provisional, objective = refinement_training_pair(
        previous=previous,
        target=target,
        levels=levels,
        noise=noise,
        standard_deviations=sigmas,
    )
    torch.testing.assert_close(
        provisional[0],
        torch.zeros_like(provisional[0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        objective[0],
        target[0] - previous[0],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        provisional[1],
        target[1] + sigmas[1] * noise[1],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(objective[1], noise[1], rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="leaves 0..3"):
        refinement_training_pair(
            previous=previous,
            target=target,
            levels=torch.tensor([0, 4]),
            noise=noise,
            standard_deviations=sigmas,
        )


def test_transition_ema_updates_exactly_and_restores_raw_state() -> None:
    module = nn.Sequential(nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        module[0].weight.fill_(2.0)
    ema = TransitionEMA(module, decay=0.5)
    with torch.no_grad():
        module[0].weight.fill_(4.0)
    ema.update(module)
    assert ema.updates == 1
    torch.testing.assert_close(
        ema.shadow["0.weight"],
        torch.full((2, 2), 3.0),
        rtol=0.0,
        atol=0.0,
    )
    with ema.applied_to(module):
        torch.testing.assert_close(
            module[0].weight,
            torch.full((2, 2), 3.0),
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        module[0].weight,
        torch.full((2, 2), 4.0),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="strictly between"):
        TransitionEMA(module, decay=1.0)


def test_parameter_accounting_covers_parent_and_refinement_once() -> None:
    transition = PDERefinerMaskedLatentTransition(
        context_frames=1,
        config=tiny_model_config(),
        refiner_config=PDERefinerConfig(),
    )
    groups = _parameter_groups(transition)
    assert groups.parent
    assert groups.refinement
    assert all(
        name.startswith("level_embedding.") or ".refinement_adapter." in name
        for name in groups.refinement_names
    )
    assert all(
        not name.startswith("level_embedding.")
        and ".refinement_adapter." not in name
        for name in groups.parent_names
    )
    assert {id(item) for item in groups.all_parameters} == {
        id(item) for item in transition.parameters()
    }
    assert groups.to_record()["total_parameter_count"] == sum(
        item.numel() for item in transition.parameters()
    )


def test_parent_artifacts_are_hash_and_metadata_strict(tmp_path: Path) -> None:
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
    normalization_path = tmp_path / "normalization.json"
    write_strict_json_atomic(normalization_path, normalization)
    model_config = tiny_model_config()
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
    artifacts = RefinerParentArtifacts(
        checkpoint_path=parent_path,
        checkpoint_sha256=sha256_path(parent_path),
        codec_path=codec,
        codec_sha256=codec_sha,
        latent_normalization_path=normalization_path,
        latent_normalization_sha256=sha256_path(normalization_path),
    )
    payload, observed = _verify_parent_payload(
        artifacts=artifacts,
        model_config=model_config,
    )
    assert payload["kind"] == "selected_O2_transition"
    assert observed == normalization
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        _verify_parent_payload(
            artifacts=replace(artifacts, checkpoint_sha256="0" * 64),
            model_config=model_config,
        )


def test_smoke_wrapper_rejects_modified_budget_before_writes(tmp_path: Path) -> None:
    dummy = RefinerParentArtifacts(
        checkpoint_path=tmp_path / "parent.pt",
        checkpoint_sha256="0" * 64,
        codec_path=tmp_path / "codec.pt",
        codec_sha256="0" * 64,
        latent_normalization_path=tmp_path / "normalization.json",
        latent_normalization_sha256="0" * 64,
    )
    modified = replace(
        PDERefinerSmokeConfig.frozen(seed=1701),
        epochs=3,
    )
    output = tmp_path / "modified"
    with pytest.raises(ValueError, match="bounded budget"):
        train_pde_refiner_smoke(
            config=modified,
            catalog=None,  # type: ignore[arg-type]
            artifacts=dummy,
            output_directory=output,
            paper0_commit="deadbeef",
            slurm_job_id="0",
            device=torch.device("cpu"),
        )
    assert not output.exists()
