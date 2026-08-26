"""Known-answer tests for freezing matched-state bounded generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper0.tools.freeze_matched_state_bounded_generation import freeze_manifest


def _write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, authorize: bool = True) -> tuple[Path, str]:
    locks = {}
    for family in ("c5p", "e6b"):
        checkpoint = tmp_path / f"{family}.pt"
        checkpoint.write_bytes(family.encode("utf-8"))
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result = {
            "status": "passed",
            "family": family,
            "seed": 1701,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "training_gate": {"passed": True},
            "transition_gate": {"passed": True},
            "best_checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "epoch": 7,
                "selection_metric": 0.4,
            },
        }
        result_path = tmp_path / f"{family}.json"
        locks[family] = {
            "path": str(result_path),
            "sha256": _write(result_path, result),
        }
    reduction = {
        "scope": "post_ecrd_old_85604_matched_state_multilead_reduction",
        "status": "completed",
        "development_run": "85604",
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
        "paired_physics_evaluation_authorized": authorize,
        "decision": (
            "run_causal_paired_derived_field_physics_evaluation"
            if authorize
            else "stop_before_paired_physics_and_record_transition_failure"
        ),
        "results": locks,
    }
    path = tmp_path / "reduction.json"
    return path, _write(path, reduction)


def test_freeze_locks_both_models_and_truth_free_generation(tmp_path) -> None:
    reduction, digest = _fixture(tmp_path)
    manifest = freeze_manifest(
        paired_reduction=reduction,
        paired_reduction_sha256=digest,
        paper0_commit="a" * 40,
    )
    assert set(manifest["evidence"]["models"]) == {"c5p", "e6b"}
    assert not manifest["evaluation"]["target_truth_used_during_generation"]
    assert manifest["evaluation"]["e6b_boundary_source"] == "predicted_Bphi"
    assert manifest["downstream"]["e6b_exact_phi_required"]
    assert not manifest["downstream"]["exact_phi_future_truth_allowed"]


def test_freeze_refuses_a_failed_paired_transition(tmp_path) -> None:
    reduction, digest = _fixture(tmp_path, authorize=False)
    with pytest.raises(ValueError, match="does not authorize"):
        freeze_manifest(
            paired_reduction=reduction,
            paired_reduction_sha256=digest,
            paper0_commit="a" * 40,
        )
