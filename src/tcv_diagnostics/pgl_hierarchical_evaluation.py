"""Authority checks for fixed-update hierarchical PGL evaluation.

The two training arms share one parent and optimization budget.  This module
authorizes one preregistered checkpoint without searching, selecting, or
opening validation truth, then reloads both EMA branches bitwise.
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
from .pgl_hierarchical_training import (
    PGL_HIERARCHICAL_ARMS,
    PGL_HIERARCHICAL_CHECKPOINT_UPDATES,
)


PGL_HIERARCHICAL_PARENT_CHECKPOINT_SHA256 = (
    "4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb"
)


def authorize_hierarchical_training_result(
    *,
    result_path: Path,
    result_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    arm: str,
    optimizer_update: int,
) -> dict[str, Any]:
    """Authorize one fixed-duration checkpoint from a complete two-epoch run."""

    result_file = Path(result_path)
    checkpoint_file = Path(checkpoint_path)
    assert_development_path(result_file)
    assert_development_path(checkpoint_file)
    if arm not in PGL_HIERARCHICAL_ARMS:
        raise ValueError("hierarchical evaluation arm differs")
    if int(optimizer_update) not in PGL_HIERARCHICAL_CHECKPOINT_UPDATES:
        raise ValueError("hierarchical evaluation update differs")
    if sha256_path(result_file) != str(result_sha256):
        raise ValueError("hierarchical training-result SHA-256 differs")
    if sha256_path(checkpoint_file) != str(checkpoint_sha256):
        raise ValueError("hierarchical checkpoint SHA-256 differs")
    result = load_strict_json(result_file)
    physics_loss = arm == "TRANSPORT"
    training = result.get("training", {})
    parent = result.get("parent", {}).get("selected_checkpoint", {})
    checkpoints = {
        int(record.get("optimizer_update", -1)): record
        for record in result.get("checkpoints", [])
    }
    selected = checkpoints.get(int(optimizer_update), {})
    if (
        result.get("schema_version") != 1
        or result.get("scope")
        != "post_ecrd_old_85604_pgl_hierarchical_transport_training"
        or result.get("status") != "screen_training_completed"
        or result.get("mode") != "screen"
        or result.get("arm") != arm
        or result.get("development_run") != "85604"
        or result.get("seed") != 1702
        or result.get("completed_optimizer_updates") != 428
        or len(checkpoints) != 3
        or result.get("fresh_optimizer") is not True
        or result.get("full_sampler_compute_control_executed") is not True
        or result.get("checkpoint_selection_performed") is not False
        or result.get("future_truth_used_by_sampler") is not False
        or result.get("physics_derived_training_loss_used") is not physics_loss
        or result.get("held_out_85606_read") is not False
        or result.get("new_nersc_data_read") is not False
        or result.get("assimilation_performed") is not False
        or result.get("diagnostic_ranking_performed") is not False
        or result.get("steering_performed") is not False
        or training.get("optimizer_updates") != 428
        or training.get("training_windows") != 856
        or training.get("checkpoint_updates") != [107, 214, 428]
        or training.get("sampler_members") != 4
        or training.get("sampler_steps") != 18
        or training.get("stochastic_learning_rate") != 1.0e-6
        or training.get("mean_learning_rate") != 1.0e-7
        or training.get("physics_derived_training_loss_used") is not physics_loss
        or parent.get("sha256") != PGL_HIERARCHICAL_PARENT_CHECKPOINT_SHA256
        or Path(str(selected.get("path", ""))).resolve(strict=True)
        != checkpoint_file.resolve(strict=True)
        or selected.get("sha256") != str(checkpoint_sha256)
        or selected.get("selection") != "fixed_duration_no_selection"
    ):
        raise ValueError("hierarchical fixed-update training contract differs")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in selected.get("state_sha256", {}).values()
    ):
        raise ValueError("hierarchical checkpoint state hashes differ")
    return result


def load_hierarchical_checkpoint_state(
    *,
    selected_mean: nn.Module,
    stochastic_model: PersistentGlobalLocalEDM,
    training_result: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    arm: str,
    optimizer_update: int,
    device: torch.device,
) -> dict[str, Any]:
    """Reload both EMA states and prove they match the recorded checkpoint."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    physics_loss = arm == "TRANSPORT"
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "pgl_hierarchical_transport_fixed_update"
        or payload.get("development_run") != "85604"
        or payload.get("arm") != arm
        or payload.get("mode") != "screen"
        or payload.get("seed") != 1702
        or payload.get("optimizer_update") != int(optimizer_update)
        or payload.get("checkpoint_selection_performed") is not False
        or payload.get("physics_derived_training_loss_used") is not physics_loss
        or payload.get("held_out_85606_read") is not False
        or payload.get("new_nersc_data_read") is not False
        or payload.get("paper0_commit") != training_result.get("paper0_commit")
    ):
        raise ValueError("hierarchical checkpoint payload contract differs")
    selected_mean.load_state_dict(payload["mean_model_state"], strict=True)
    stochastic_model.load_state_dict(payload["stochastic_model_state"], strict=True)
    selected_mean.eval().requires_grad_(False)
    stochastic_model.eval().requires_grad_(False)
    observed = {
        "ema_mean": module_state_sha256(selected_mean),
        "ema_stochastic": module_state_sha256(stochastic_model),
    }
    expected = payload.get("state_sha256", {})
    if any(observed[name] != expected.get(name) for name in observed):
        raise ValueError("hierarchical EMA checkpoint reload differs")
    return {
        "arm": arm,
        "optimizer_update": int(optimizer_update),
        "equivalent_epochs": float(payload["equivalent_epochs"]),
        "selected_checkpoint": {
            "path": str(Path(checkpoint_path)),
            "sha256": str(checkpoint_sha256),
            "selection": "fixed_duration_no_selection",
            "checkpoint_reload_bitwise": True,
            "state_sha256": observed,
        },
        "mean_branch_updated": True,
        "stochastic_branch_updated": True,
        "physics_derived_training_loss_used": physics_loss,
    }
