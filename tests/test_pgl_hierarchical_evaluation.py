from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from torch import nn

from tcv_diagnostics.b5_residual_edm_training import module_state_sha256
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.models.persistent_global_local import (
    PersistentGlobalLocalConfig,
    PersistentGlobalLocalEDM,
    PersistentNoiseConfig,
)
from tcv_diagnostics.pgl_hierarchical_evaluation import (
    PGL_HIERARCHICAL_PARENT_CHECKPOINT_SHA256,
    authorize_hierarchical_training_result,
    load_hierarchical_checkpoint_state,
)
from paper0.tools.score_pgl_hierarchical_physics import verify_generation


def _model() -> PersistentGlobalLocalEDM:
    config = PersistentGlobalLocalConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        global_channels=4,
        global_pool_xy=(2, 2),
        low_mode_maximum=2,
        noise_embedding_features=16,
        group_norm_maximum_groups=4,
    )
    noise = PersistentNoiseConfig(global_pool_xy=(2, 2), low_mode_maximum=2)
    return PersistentGlobalLocalEDM(
        config, residual_scales=torch.ones((4, 5)), noise_config=noise
    )


def test_generation_and_scoring_commits_are_separate_authorities(tmp_path: Path) -> None:
    forecast = tmp_path / "forecast.h5"
    forecast.write_bytes(b"immutable forecast")
    generation_commit = "a" * 40
    scoring_commit = "b" * 40
    result = {
        "scope": "old_85604_pgl_hierarchical_truth_free_forecast_generation",
        "status": "truth_free_forecast_completed_and_hash_closed",
        "arm": "CONTROL",
        "optimizer_update": 107,
        "paper0_commit": generation_commit,
        "manifest": {"sha256": "manifest"},
        "training_result": {"sha256": "training"},
        "checkpoint": {"sha256": "checkpoint"},
        "forecast": {"path": str(forecast), "sha256": sha256_path(forecast)},
        "start_count": 36,
        "ensemble_members": 32,
        "future_frames": 4,
        "target_truth_read": False,
        "physics_diagnostics_scored": False,
        "checkpoint_selection_performed": False,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    result_path = tmp_path / "generation.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    args = SimpleNamespace(
        generation_result=result_path,
        generation_result_sha256=sha256_path(result_path),
        forecast=forecast,
        forecast_sha256=sha256_path(forecast),
        arm="CONTROL",
        optimizer_update=107,
        paper0_commit=scoring_commit,
        generation_paper0_commit=generation_commit,
        manifest_sha256="manifest",
        training_result_sha256="training",
        checkpoint_sha256="checkpoint",
    )
    assert verify_generation(args)["paper0_commit"] == generation_commit
    args.generation_paper0_commit = "c" * 40
    with pytest.raises(ValueError, match="contract differs"):
        verify_generation(args)


def _artifacts(
    tmp_path: Path, *, arm: str = "TRANSPORT", update: int = 214
) -> tuple[Path, str, Path, str, nn.Module, PersistentGlobalLocalEDM]:
    mean = nn.Conv3d(5, 5, 1)
    model = _model()
    physics = arm == "TRANSPORT"
    checkpoint = tmp_path / f"fixed_update_{update:04d}.pt"
    state_hash = {
        "raw_mean": module_state_sha256(mean),
        "raw_stochastic": module_state_sha256(model),
        "ema_mean": module_state_sha256(mean),
        "ema_stochastic": module_state_sha256(model),
    }
    payload = {
        "schema_version": 1,
        "kind": "pgl_hierarchical_transport_fixed_update",
        "development_run": "85604",
        "paper0_commit": "a" * 40,
        "arm": arm,
        "mode": "screen",
        "seed": 1702,
        "optimizer_update": update,
        "equivalent_epochs": update / 214,
        "mean_model_state": mean.state_dict(),
        "stochastic_model_state": model.state_dict(),
        "state_sha256": state_hash,
        "checkpoint_selection_performed": False,
        "physics_derived_training_loss_used": physics,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
    torch.save(payload, checkpoint)
    checkpoint_sha = sha256_path(checkpoint)
    records = []
    for fixed in (107, 214, 428):
        path = checkpoint if fixed == update else tmp_path / f"placeholder_{fixed}.pt"
        if fixed != update:
            path.write_bytes(b"placeholder")
        records.append(
            {
                "optimizer_update": fixed,
                "equivalent_epochs": fixed / 214,
                "path": str(path),
                "sha256": checkpoint_sha if fixed == update else sha256_path(path),
                "selection": "fixed_duration_no_selection",
                "state_sha256": state_hash,
            }
        )
    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_hierarchical_transport_training",
        "status": "screen_training_completed",
        "mode": "screen",
        "arm": arm,
        "development_run": "85604",
        "seed": 1702,
        "paper0_commit": "a" * 40,
        "completed_optimizer_updates": 428,
        "fresh_optimizer": True,
        "full_sampler_compute_control_executed": True,
        "checkpoint_selection_performed": False,
        "future_truth_used_by_sampler": False,
        "physics_derived_training_loss_used": physics,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
        "training": {
            "optimizer_updates": 428,
            "training_windows": 856,
            "checkpoint_updates": [107, 214, 428],
            "sampler_members": 4,
            "sampler_steps": 18,
            "stochastic_learning_rate": 1.0e-6,
            "mean_learning_rate": 1.0e-7,
            "physics_derived_training_loss_used": physics,
        },
        "checkpoints": records,
        "parent": {
            "selected_checkpoint": {
                "sha256": PGL_HIERARCHICAL_PARENT_CHECKPOINT_SHA256
            }
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path, sha256_path(result_path), checkpoint, checkpoint_sha, mean, model


def test_authority_and_both_ema_branches_reload(tmp_path: Path) -> None:
    result_path, result_sha, checkpoint, checkpoint_sha, mean, model = _artifacts(
        tmp_path
    )
    result = authorize_hierarchical_training_result(
        result_path=result_path,
        result_sha256=result_sha,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        arm="TRANSPORT",
        optimizer_update=214,
    )
    for parameter in mean.parameters():
        parameter.data.zero_()
    for parameter in model.parameters():
        parameter.data.zero_()
    provenance = load_hierarchical_checkpoint_state(
        selected_mean=mean,
        stochastic_model=model,
        training_result=result,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        arm="TRANSPORT",
        optimizer_update=214,
        device=torch.device("cpu"),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert module_state_sha256(mean) == payload["state_sha256"]["ema_mean"]
    assert module_state_sha256(model) == payload["state_sha256"]["ema_stochastic"]
    assert provenance["mean_branch_updated"] is True
    assert provenance["physics_derived_training_loss_used"] is True


def test_wrong_physics_label_or_unregistered_update_fails(tmp_path: Path) -> None:
    result_path, _, checkpoint, checkpoint_sha, *_ = _artifacts(tmp_path)
    result = json.loads(result_path.read_text())
    result["physics_derived_training_loss_used"] = False
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        authorize_hierarchical_training_result(
            result_path=result_path,
            result_sha256=sha256_path(result_path),
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            arm="TRANSPORT",
            optimizer_update=214,
        )
    with pytest.raises(ValueError, match="update"):
        authorize_hierarchical_training_result(
            result_path=result_path,
            result_sha256=sha256_path(result_path),
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            arm="TRANSPORT",
            optimizer_update=999,
        )
