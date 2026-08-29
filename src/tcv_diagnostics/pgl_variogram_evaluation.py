"""Authority checks for evaluating bounded PGL variogram warm starts.

The original persistent-model evaluator is intentionally locked to its epoch-20
checkpoint.  This module validates one fixed-final variogram checkpoint and
applies only its stored stochastic EMA state to an already-authorized copy of
that model.  It never selects a checkpoint or discovers experiment paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .b5_residual_edm_training import module_state_sha256
from .codec_training import sha256_path
from .model_data import assert_development_path, load_strict_json
from .models.persistent_global_local import PersistentGlobalLocalEDM
from .pgl_variogram_training import PGL_VARIOGRAM_ARMS


PGL_VARIOGRAM_PARENT_CHECKPOINT_SHA256 = (
    "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
)


def authorize_variogram_training_result(
    *,
    result_path: Path,
    result_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    arm: str,
) -> dict[str, Any]:
    """Validate one completed, fixed-budget arm without result-based selection."""

    result_file = Path(result_path)
    checkpoint_file = Path(checkpoint_path)
    assert_development_path(result_file)
    assert_development_path(checkpoint_file)
    if arm not in PGL_VARIOGRAM_ARMS:
        raise ValueError("variogram evaluation arm differs")
    if sha256_path(result_file) != str(result_sha256):
        raise ValueError("variogram training result SHA-256 differs")
    if sha256_path(checkpoint_file) != str(checkpoint_sha256):
        raise ValueError("variogram fixed-final checkpoint SHA-256 differs")
    result = load_strict_json(result_file)
    physics_loss = arm in ("C", "D")
    selected = result.get("selected_checkpoint", {})
    parent = result.get("parent", {}).get("selected_checkpoint", {})
    training = result.get("training", {})
    if (
        result.get("schema_version") != 1
        or result.get("scope")
        != "post_ecrd_old_85604_pgl_variogram_warm_start"
        or result.get("status") != "screen_training_completed"
        or result.get("mode") != "screen"
        or result.get("arm") != arm
        or result.get("development_run") != "85604"
        or result.get("seed") != 1702
        or result.get("completed_epochs") != 1
        or result.get("completed_optimizer_updates") != 214
        or result.get("mean_frozen_bitwise") is not True
        or result.get("fresh_optimizer") is not True
        or result.get("full_sampler_compute_control_executed") is not True
        or result.get("sampler_members") != 4
        or result.get("sampler_steps") != 18
        or result.get("network_evaluations_per_member") != 35
        or result.get("checkpoint_selection_performed") is not False
        or result.get("future_truth_used_by_sampler") is not False
        or result.get("physics_derived_training_loss_used") is not physics_loss
        or result.get("held_out_run_read") is not False
        or result.get("held_out_85606_read") is not False
        or result.get("new_segment_read") is not False
        or result.get("new_nersc_data_read") is not False
        or result.get("assimilation_performed") is not False
        or result.get("diagnostic_ranking_performed") is not False
        or result.get("steering_performed") is not False
        or training.get("fixed_final_ema_no_checkpoint_selection") is not True
        or training.get("deterministic_mean_frozen") is not True
        or training.get("optimizer_updates") != 214
        or training.get("training_windows") != 428
        or training.get("learning_rate") != 1.0e-6
        or training.get("arm") != arm
        or training.get("physics_derived_training_loss_used") is not physics_loss
        or Path(str(selected.get("path", ""))).resolve(strict=True)
        != checkpoint_file.resolve(strict=True)
        or selected.get("sha256") != str(checkpoint_sha256)
        or selected.get("selection") != "fixed_final_EMA_no_checkpoint_selection"
        or parent.get("sha256") != PGL_VARIOGRAM_PARENT_CHECKPOINT_SHA256
    ):
        raise ValueError("variogram fixed-final training result contract differs")
    initial = result.get("initial_state_sha256", {})
    final = result.get("final_state_sha256", {})
    if (
        initial.get("mean") != final.get("mean")
        or any(len(str(value)) != 64 for value in (*initial.values(), *final.values()))
    ):
        raise ValueError("variogram result state hashes differ")
    return result


def load_variogram_checkpoint_state(
    *,
    selected_mean: nn.Module,
    stochastic_model: PersistentGlobalLocalEDM,
    training_result: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    arm: str,
    device: torch.device,
) -> dict[str, Any]:
    """Load the fixed-final stochastic EMA while proving the mean is unchanged."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    physics_loss = arm in ("C", "D")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "pgl_variogram_fixed_final_EMA_warm_start"
        or payload.get("development_run") != "85604"
        or payload.get("arm") != arm
        or payload.get("mode") != "screen"
        or payload.get("seed") != 1702
        or payload.get("completed_epoch") != 1
        or payload.get("optimizer_updates") != 214
        or payload.get("checkpoint_selection_performed") is not False
        or payload.get("physics_derived_training_loss_used") is not physics_loss
        or payload.get("held_out_run_read") is not False
        or payload.get("held_out_85606_read") is not False
        or payload.get("new_segment_read") is not False
        or payload.get("new_nersc_data_read") is not False
        or payload.get("paper0_commit") != training_result.get("paper0_commit")
    ):
        raise ValueError("variogram checkpoint payload contract differs")
    mean_state = selected_mean.state_dict()
    stored_mean = payload.get("mean_model_state", {})
    if set(stored_mean) != set(mean_state) or not all(
        torch.equal(stored_mean[name].to(device), mean_state[name]) for name in mean_state
    ):
        raise ValueError("variogram checkpoint changed the frozen deterministic mean")
    stochastic_model.load_state_dict(payload["stochastic_model_state"], strict=True)
    stochastic_model.eval()
    stochastic_model.requires_grad_(False)
    observed_hash = module_state_sha256(stochastic_model)
    expected_hash = training_result["final_state_sha256"]["stochastic_ema"]
    if observed_hash != expected_hash:
        raise ValueError("variogram stochastic EMA state hash differs")
    return {
        "arm": arm,
        "selected_checkpoint": {
            "path": str(Path(checkpoint_path)),
            "sha256": str(checkpoint_sha256),
            "selection": "fixed_final_EMA_no_checkpoint_selection",
            "checkpoint_reload_bitwise": True,
            "stochastic_state_sha256": observed_hash,
        },
        "deterministic_mean_frozen": True,
        "physics_derived_training_loss_used": physics_loss,
    }
