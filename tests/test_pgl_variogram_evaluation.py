from __future__ import annotations

import json
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
from tcv_diagnostics.pgl_variogram_evaluation import (
    PGL_VARIOGRAM_PARENT_CHECKPOINT_SHA256,
    authorize_variogram_training_result,
    load_variogram_checkpoint_state,
)


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
        config,
        residual_scales=torch.ones((4, 5)),
        noise_config=noise,
    )


def _artifacts(tmp_path: Path, *, arm: str = "B") -> tuple[Path, str, Path, str]:
    mean = nn.Linear(3, 2)
    model = _model()
    physics = arm in ("C", "D")
    checkpoint = tmp_path / "selected.pt"
    payload = {
        "schema_version": 1,
        "kind": "pgl_variogram_fixed_final_EMA_warm_start",
        "development_run": "85604",
        "arm": arm,
        "mode": "screen",
        "seed": 1702,
        "paper0_commit": "a" * 40,
        "completed_epoch": 1,
        "optimizer_updates": 214,
        "training": {},
        "mean_model_state": {name: value.detach().clone() for name, value in mean.state_dict().items()},
        "stochastic_model_state": {name: value.detach().clone() for name, value in model.state_dict().items()},
        "parent_mean_state_sha256": module_state_sha256(mean),
        "parent_stochastic_state_sha256": "1" * 64,
        "checkpoint_selection_performed": False,
        "physics_derived_training_loss_used": physics,
        "held_out_run_read": False,
        "held_out_85606_read": False,
        "new_segment_read": False,
        "new_nersc_data_read": False,
    }
    torch.save(payload, checkpoint)
    checkpoint_sha = sha256_path(checkpoint)
    result = {
        "schema_version": 1,
        "scope": "post_ecrd_old_85604_pgl_variogram_warm_start",
        "status": "screen_training_completed",
        "mode": "screen",
        "arm": arm,
        "development_run": "85604",
        "seed": 1702,
        "paper0_commit": "a" * 40,
        "completed_epochs": 1,
        "completed_optimizer_updates": 214,
        "mean_frozen_bitwise": True,
        "fresh_optimizer": True,
        "full_sampler_compute_control_executed": True,
        "sampler_members": 4,
        "sampler_steps": 18,
        "network_evaluations_per_member": 35,
        "checkpoint_selection_performed": False,
        "future_truth_used_by_sampler": False,
        "physics_derived_training_loss_used": physics,
        "held_out_run_read": False,
        "held_out_85606_read": False,
        "new_segment_read": False,
        "new_nersc_data_read": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
        "training": {
            "fixed_final_ema_no_checkpoint_selection": True,
            "deterministic_mean_frozen": True,
            "optimizer_updates": 214,
            "training_windows": 428,
            "learning_rate": 1.0e-6,
            "arm": arm,
            "physics_derived_training_loss_used": physics,
        },
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "selection": "fixed_final_EMA_no_checkpoint_selection",
        },
        "parent": {
            "selected_checkpoint": {"sha256": PGL_VARIOGRAM_PARENT_CHECKPOINT_SHA256}
        },
        "initial_state_sha256": {
            "mean": module_state_sha256(mean),
            "stochastic": "1" * 64,
        },
        "final_state_sha256": {
            "mean": module_state_sha256(mean),
            "stochastic_ema": module_state_sha256(model),
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path, sha256_path(result_path), checkpoint, checkpoint_sha


def test_authority_and_checkpoint_reload_preserve_mean(tmp_path: Path) -> None:
    result_path, result_sha, checkpoint, checkpoint_sha = _artifacts(tmp_path)
    result = authorize_variogram_training_result(
        result_path=result_path,
        result_sha256=result_sha,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        arm="B",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mean = nn.Linear(3, 2)
    mean.load_state_dict(payload["mean_model_state"])
    mean_hash = module_state_sha256(mean)
    model = _model()
    provenance = load_variogram_checkpoint_state(
        selected_mean=mean,
        stochastic_model=model,
        training_result=result,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        arm="B",
        device=torch.device("cpu"),
    )
    assert module_state_sha256(mean) == mean_hash
    assert provenance["deterministic_mean_frozen"] is True
    assert provenance["physics_derived_training_loss_used"] is False


def test_transport_loss_label_is_mandatory(tmp_path: Path) -> None:
    result_path, result_sha, checkpoint, checkpoint_sha = _artifacts(tmp_path, arm="C")
    authorize_variogram_training_result(
        result_path=result_path,
        result_sha256=result_sha,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        arm="C",
    )
    record = json.loads(result_path.read_text())
    record["physics_derived_training_loss_used"] = False
    result_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        authorize_variogram_training_result(
            result_path=result_path,
            result_sha256=sha256_path(result_path),
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            arm="C",
        )

