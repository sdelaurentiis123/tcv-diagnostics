from __future__ import annotations

import copy

import pytest

from tcv_diagnostics.o2_matrix import O2_MATRIX_ORDER, finalize_o2_matrix


def _run(index: int, arm: str, seed: int, passes: bool) -> dict:
    return {
        "scope": "O2_selected_checkpoint_scientific_evaluation",
        "status": "completed",
        "scientific_authority": True,
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_used_during_forecast_generation": False,
        "physics_derived_training_loss_used": False,
        "training_run_index": index,
        "arm": arm,
        "seed": seed,
        "gate": {"passes": passes, "status": "pass" if passes else "fail"},
        "O2_seed_accepted": passes,
        "O3_launch_allowed": False,
    }


def _matrix(passes):
    return [
        _run(index, arm, seed, accepted)
        for (index, arm, seed), accepted in zip(O2_MATRIX_ORDER, passes)
    ]


def test_all_three_seeds_are_required_and_seed_averaging_cannot_rescue_arm():
    result = finalize_o2_matrix(_matrix([True, True, False, True, True, True]))
    assert result["arms"]["C5P-H1"]["passing_seed_count"] == 2
    assert result["arms"]["C5P-H1"]["accepted"] is False
    assert result["accepted_arms"] == ["C5P-H2"]
    assert result["new_O3_protocol_may_be_frozen"] is True
    assert result["O3_launch_allowed"] is False
    assert result["seed_averaging_used"] is False


def test_no_passing_arm_stops_before_o3_and_two_passing_arms_are_both_retained():
    failed = finalize_o2_matrix(_matrix([False] * 6))
    assert failed["accepted_arms"] == []
    assert failed["new_O3_protocol_may_be_frozen"] is False
    assert failed["disposition"] == "stop_and_report_deterministic_one_step_failure"

    passed = finalize_o2_matrix(_matrix([True] * 6))
    assert passed["accepted_arms"] == ["C5P-H1", "C5P-H2"]
    assert passed["disposition"] == (
        "retain_both_arms_through_first_new_short_O3_comparison"
    )


def test_matrix_rejects_reordering_and_acceptance_gate_disagreement():
    runs = _matrix([True] * 6)
    reordered = copy.deepcopy(runs)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="order"):
        finalize_o2_matrix(reordered)
    inconsistent = copy.deepcopy(runs)
    inconsistent[0]["O2_seed_accepted"] = False
    with pytest.raises(ValueError, match="acceptance"):
        finalize_o2_matrix(inconsistent)
