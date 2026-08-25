"""Known-answer checks for the old-85604 C5P multi-lead screen."""

from __future__ import annotations

import copy

import pytest

from paper0.tools.train_codec_free_stage2_multilead import (
    FIELDS,
    LEADS,
    authorize_manifest,
    screen_decision,
)
from tcv_diagnostics.state_operator_data import plan_lead_pairs


def _manifest() -> dict:
    return {
        "scope": "post_ecrd_old_85604_stage2_multilead_screen",
        "development_run": "85604",
        "held_out_85606_read": False,
        "held_out_85606_access_allowed": False,
        "guard_frames_read_allowed": False,
        "screen_training_authorized": True,
        "three_seed_scaling_authorized": False,
        "state_family": "c5p",
        "split": {
            "training_frames": [0, 432],
            "guard_frames": [432, 496],
            "validation_frames": [496, 624],
            "lead_steps": list(LEADS),
            "history_frames": 1,
            "training_pair_count": 2129,
            "validation_pair_count": 609,
            "training_pairs_by_lead": {
                "1": 431,
                "2": 430,
                "4": 428,
                "8": 424,
                "16": 416,
            },
            "validation_pairs_by_lead": {
                "1": 127,
                "2": 126,
                "4": 124,
                "8": 120,
                "16": 112,
            },
        },
        "optimization": {
            "screen_seed": 1701,
            "initialize_from_parent_model_only": True,
            "restore_parent_optimizer": False,
        },
    }


def _evaluation(*, ratio: float, lead1_mse: float, longer_mse: float) -> dict:
    per_lead = {}
    for lead in LEADS:
        model_mse = lead1_mse if lead == 1 else longer_mse
        per_lead[str(lead)] = {
            "shared_field_mean_model_derivative_mse": model_mse,
            "per_field": {
                field: {"persistence_relative_skill": 0.2} for field in FIELDS
            },
        }
    return {
        "per_lead": per_lead,
        "mean_shared_persistence_normalized_mse_ratio": ratio,
    }


def test_manifest_and_pair_counts_match_frozen_chronology() -> None:
    authorize_manifest(_manifest(), seed=1701)
    train = plan_lead_pairs(
        split="train", lead_steps=LEADS, history_frames=1
    )
    validation = plan_lead_pairs(
        split="validation", lead_steps=LEADS, history_frames=1
    )
    assert len(train) == 2129
    assert len(validation) == 609
    assert not any(432 <= frame < 496 for pair in train + validation for frame in (pair.current, pair.target))


def test_manifest_rejects_scaling_or_another_seed() -> None:
    manifest = _manifest()
    manifest["three_seed_scaling_authorized"] = True
    with pytest.raises(ValueError, match="three-seed"):
        authorize_manifest(manifest, seed=1701)
    with pytest.raises(ValueError, match="seed"):
        authorize_manifest(_manifest(), seed=1702)


def test_screen_decision_requires_every_prospective_gate() -> None:
    parent = _evaluation(ratio=0.5, lead1_mse=0.005322, longer_mse=1.0)
    child = _evaluation(ratio=0.44, lead1_mse=0.0054, longer_mse=0.9)
    gates = {
        "maximum_lead1_shared_mse": 0.005588458639715578,
        "minimum_parent_improvement_fraction": 0.10,
        "minimum_improved_longer_lead_count": 3,
    }
    decision = screen_decision(
        parent_evaluation=parent,
        best_validation=child,
        training_gate=True,
        frozen_gates=gates,
    )
    assert decision["parent_improvement_fraction"] == pytest.approx(0.12)
    assert decision["longer_lead_improved_count"] == 4
    assert decision["advance_to_three_seed_scaling"] is True

    failed_child = copy.deepcopy(child)
    failed_child["per_lead"]["16"]["per_field"]["phi"][
        "persistence_relative_skill"
    ] = -0.01
    failed = screen_decision(
        parent_evaluation=parent,
        best_validation=failed_child,
        training_gate=True,
        frozen_gates=gates,
    )
    assert failed["advance_to_three_seed_scaling"] is False
    assert failed["screen_gates"][
        "every_c5p_field_positive_skill_at_every_lead"
    ] is False
