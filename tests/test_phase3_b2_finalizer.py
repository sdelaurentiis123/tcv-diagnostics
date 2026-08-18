from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_b2_evaluation",
    ROOT / "paper0/tools/finalize_b2_evaluation.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _training_matrix() -> dict[str, object]:
    return {
        "scope": "phase3_B2_LDM_H2_full_training_matrix_frozen",
        "status": "completed_pending_bounded_evaluator_smoke",
        "paper0_commit": "a" * 40,
        "development_run": "85604",
        "held_out_85606_read": False,
        "seeds": [1701, 1702, 1703],
        "seed_count": 3,
        "all_training_histories_complete": True,
        "all_checkpoint_choices_frozen_before_probabilistic_metrics": True,
        "probabilistic_scientific_gate_evaluated": False,
        "runs": [
            {
                "seed": seed,
                "training_complete": True,
                "scientific_acceptance_evaluated": False,
            }
            for seed in (1701, 1702, 1703)
        ],
    }


def _comparators() -> dict[str, object]:
    return {
        "scope": "phase3_B2_frozen_paired_deterministic_comparators_85604",
        "status": "completed_before_B2_scientific_acceptance",
        "development_run": "85604",
        "held_out_85606_read": False,
        "B2_forecasts_or_scores_read": False,
        "deterministic_model_retrained": False,
        "deterministic_checkpoint_reselected": False,
        "seeds": [1701, 1702, 1703],
        "seed_count": 3,
        "scientific_acceptance_evaluated": False,
        "best_uncompressed": {"name": "training_only_toroidal_spectral_AR1"},
        "runs": [{"seed": seed} for seed in (1701, 1702, 1703)],
    }


def test_finalizer_accepts_only_complete_ordered_training_and_comparator_inputs() -> (
    None
):
    training = MODULE._validate_training_matrix(
        _training_matrix(), paper0_commit="a" * 40
    )
    comparators, best = MODULE._validate_comparators(_comparators())
    assert tuple(training) == (1701, 1702, 1703)
    assert tuple(comparators) == (1701, 1702, 1703)
    assert best["name"] == "training_only_toroidal_spectral_AR1"

    contaminated = _comparators()
    contaminated["B2_forecasts_or_scores_read"] = True
    with pytest.raises(ValueError, match="comparator matrix"):
        MODULE._validate_comparators(contaminated)


def test_finalizer_requires_exact_four_target_smoke_and_training_matrix() -> None:
    smoke = {
        "scope": "bounded_non_scientific_B2_evaluator_smoke_85604",
        "status": "bounded_evaluator_smoke_completed",
        "paper0_commit": "a" * 40,
        "seed": 1701,
        "target_frames": [498, 502],
        "target_count": 4,
        "ensemble_members": 32,
        "held_out_85606_read": False,
        "truth_opened_only_after_forecast_hash": True,
        "full_probabilistic_evaluation_preconditions_passed": True,
        "probabilistic_scientific_gate_evaluated": False,
        "training_matrix": {"sha256": "m" * 64},
    }
    MODULE._validate_smoke(
        smoke,
        paper0_commit="a" * 40,
        training_matrix_sha256="m" * 64,
    )
    smoke["target_count"] = 126
    with pytest.raises(ValueError, match="smoke contract"):
        MODULE._validate_smoke(
            smoke,
            paper0_commit="a" * 40,
            training_matrix_sha256="m" * 64,
        )
