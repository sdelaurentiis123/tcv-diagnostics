"""Known-answer checks for the old-85604 exact-state phi repair screen."""

from __future__ import annotations

import pytest

from paper0.tools.train_exact_state_phi_repair_screen import (
    ARCHITECTURES,
    E6B_FIELDS,
    authorize_manifest,
    build_model,
)


def _manifest() -> dict:
    return {
        "scope": "post_ecrd_old_85604_exact_state_phi_repair_screen",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "screen_training_authorized": True,
        "three_seed_scaling_authorized": False,
        "state": {
            "predicted_volume_fields": list(E6B_FIELDS),
            "predicted_boundary": "Bphi",
            "auxiliary_context_fields": ["phi"],
            "future_auxiliary_context_allowed": False,
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
        "architectures": {name: {} for name in ARCHITECTURES},
        "optimization": {"screen_seed": 1701},
    }


def test_manifest_locks_old_85604_scope_and_causal_phi() -> None:
    authorize_manifest(_manifest(), architecture="local_current_phi", seed=1701)
    broken = _manifest()
    broken["state"]["future_auxiliary_context_allowed"] = True
    with pytest.raises(ValueError, match="future auxiliary"):
        authorize_manifest(broken, architecture="local_current_phi", seed=1701)


def test_local_and_axial_repair_models_are_parameter_matched() -> None:
    local, local_config = build_model(
        "local_current_phi",
        {
            "base_channels": 24,
            "channel_multipliers": [1, 2, 4],
            "blocks_per_level": 2,
            "lead_embedding_channels": 128,
            "group_norm_maximum_groups": 8,
            "kernel_size": 3,
            "zero_initialize_output": True,
        },
    )
    axial, axial_config = build_model(
        "axial_current_phi",
        {
            "width": 104,
            "blocks": 4,
            "attention_heads": 4,
            "feedforward_expansion": 2,
            "lead_embedding_channels": 128,
            "group_norm_maximum_groups": 8,
            "kernel_size": 3,
            "zero_initialize_output": True,
        },
    )
    local_count = sum(parameter.numel() for parameter in local.parameters())
    axial_count = sum(parameter.numel() for parameter in axial.parameters())
    assert local_config.auxiliary_context_channels == 1
    assert axial_config.auxiliary_context_channels == 1
    assert abs(local_count - axial_count) / local_count < 0.03
    assert local_config.to_record()["downsample_stride_xyz"][-1] == 1
    assert axial_config.to_record()["toroidal_downsampling"] is False


def test_trainer_source_freezes_no_future_truth_and_no_physics_loss() -> None:
    from pathlib import Path

    source = Path("paper0/tools/train_exact_state_phi_repair_screen.py").read_text(
        encoding="utf-8"
    )
    assert 'auxiliary_context_fields=("phi",)' in source
    assert '"future_auxiliary_context_read": False' in source
    assert '"physics_derived_loss_used": False' in source
    assert "advance_to_three_seed_scaling" in source
