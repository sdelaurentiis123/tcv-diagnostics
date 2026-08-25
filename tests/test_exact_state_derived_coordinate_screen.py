"""Known-answer checks for the old-85604 phi-plus-Vi screen."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paper0.tools.train_exact_state_phi_repair_screen import (
    E6B_FIELDS,
    authorize_manifest,
    build_model,
    repair_tensor_batch,
)


def _manifest() -> dict:
    return {
        "scope": "post_ecrd_old_85604_exact_state_derived_coordinate_screen",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "screen_training_authorized": True,
        "three_seed_scaling_authorized": False,
        "state": {
            "predicted_volume_fields": list(E6B_FIELDS),
            "predicted_boundary": "Bphi",
            "auxiliary_context_fields": ["phi", "Vi"],
            "future_auxiliary_context_allowed": False,
            "validated_elliptic_closure_available": True,
            "rollout_requires_external_elliptic_operator": True,
        },
        "split": {
            "training_frames": [0, 432],
            "guard_frames": [432, 496],
            "validation_frames": [496, 624],
            "training_pair_count": 431,
            "validation_pair_count": 127,
            "lead_steps": [1],
            "history_frames": 1,
        },
        "architectures": {"local_current_phi_vi": {}},
        "optimization": {"screen_seed": 1701},
    }


def test_manifest_authorizes_only_causal_phi_vi_arm() -> None:
    authorize_manifest(_manifest(), architecture="local_current_phi_vi", seed=1701)
    broken = _manifest()
    broken["state"]["auxiliary_context_fields"] = ["phi"]
    with pytest.raises(ValueError, match="auxiliary context"):
        authorize_manifest(
            broken,
            architecture="local_current_phi_vi",
            seed=1701,
        )


def test_phi_vi_model_matches_local_budget_and_uses_two_auxiliary_channels() -> None:
    record = {
        "base_channels": 24,
        "channel_multipliers": [1, 2, 4],
        "blocks_per_level": 2,
        "lead_embedding_channels": 128,
        "group_norm_maximum_groups": 8,
        "kernel_size": 3,
        "zero_initialize_output": True,
    }
    phi, _ = build_model("local_current_phi", record)
    phi_vi, config = build_model("local_current_phi_vi", record)
    phi_count = sum(parameter.numel() for parameter in phi.parameters())
    phi_vi_count = sum(parameter.numel() for parameter in phi_vi.parameters())
    assert config.auxiliary_context_channels == 2
    assert abs(phi_vi_count - phi_count) / phi_count < 0.03
    assert config.to_record()["downsample_stride_xyz"][-1] == 1


def test_tensor_batch_enforces_two_auxiliary_channels() -> None:
    item = {
        "context": np.zeros((1, 6, 2, 2, 2), dtype=np.float32),
        "lead_steps": np.float32(1.0),
        "target_derivative": np.zeros((6, 2, 2, 2), dtype=np.float32),
        "context_boundary": np.zeros((1, 2, 2), dtype=np.float32),
        "target_boundary_derivative": np.zeros((2, 2), dtype=np.float32),
        "auxiliary_context": np.zeros((1, 1, 2, 2, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="auxiliary-context shape"):
        repair_tensor_batch(
            item,
            torch.device("cpu"),
            auxiliary_channels=2,
        )
