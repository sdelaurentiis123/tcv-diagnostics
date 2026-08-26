"""Known-answer tests for the matched state-view paired reduction."""

from __future__ import annotations

import copy

import pytest

from paper0.tools.reduce_matched_state_multilead import reduce_pair
from paper0.tools.train_matched_state_multilead import LEADS


def _result(family: str, *, ratio: float, epoch: int = 9) -> dict:
    per_lead = {}
    for lead in LEADS:
        per_lead[str(lead)] = {
            "shared_field_mean_model_derivative_mse": ratio * lead,
            "shared_field_persistence_relative_skill": 0.5,
        }
    return {
        "training_gate": {"passed": True},
        "transition_gate": {"passed": True},
        "best_checkpoint": {
            "selection_metric": ratio,
            "epoch": epoch,
            "selected_at_budget_boundary": epoch == 12,
        },
        "best_validation": {"per_lead": per_lead},
        "family": family,
    }


def _reduce(c5p: dict, e6b: dict) -> dict:
    return reduce_pair(
        c5p=c5p,
        e6b=e6b,
        c5p_lock={"path": "c5p.json", "sha256": "a" * 64},
        e6b_lock={"path": "e6b.json", "sha256": "b" * 64},
        manifest_lock={"path": "manifest.json", "sha256": "c" * 64},
        paper0_commit="d" * 40,
        training_commit="e" * 40,
        slurm_job_id="known-answer",
    )


def test_reduction_authorizes_only_a_two_arm_transition_pass() -> None:
    result = _reduce(_result("c5p", ratio=1.0), _result("e6b", ratio=0.8))
    assert result["paired_physics_evaluation_authorized"]
    assert result["decision"] == "run_causal_paired_derived_field_physics_evaluation"
    assert result["transition_comparison"][
        "median_e6b_over_c5p_shared_mse"
    ] == pytest.approx(0.8)
    assert not result["duration_censored_for_any_arm"]

    failed = _result("e6b", ratio=0.8)
    failed["transition_gate"]["passed"] = False
    stopped = _reduce(_result("c5p", ratio=1.0), failed)
    assert not stopped["paired_physics_evaluation_authorized"]
    assert stopped["decision"] == "stop_before_paired_physics_and_record_transition_failure"


def test_reduction_reports_budget_boundary_without_extending() -> None:
    c5p = _result("c5p", ratio=1.0, epoch=12)
    e6b = _result("e6b", ratio=1.2, epoch=10)
    result = _reduce(c5p, e6b)
    assert result["selected_at_budget_boundary"] == {
        "c5p": True,
        "e6b": False,
    }
    assert result["duration_censored_for_any_arm"]
    assert result["three_seed_scaling_authorized"] is False


def test_reduction_rejects_nonfinite_comparison() -> None:
    e6b = copy.deepcopy(_result("e6b", ratio=1.0))
    e6b["best_validation"]["per_lead"]["4"][
        "shared_field_mean_model_derivative_mse"
    ] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _reduce(_result("c5p", ratio=1.0), e6b)
