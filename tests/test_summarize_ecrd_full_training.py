"""Tests for the ECRD full-training finalizer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tcv_diagnostics.ecrd_training import (
    ECRDTrainingConfig,
    frozen_parameter_counts,
    model_config_record,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/summarize_ecrd_full_training.py"
LAUNCHER = ROOT / "cluster/ecrd_full_training_finalize.sbatch"
SPEC = importlib.util.spec_from_file_location("summarize_ecrd_full", TOOL)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _result(*, arm: str, seed: int, commit: str, slurm: str) -> dict[str, object]:
    return {
        "scope": "ECRD_matched_model_development_training_85604",
        "status": "training_completed_checkpoint_selected",
        "mode": "full",
        "arm": arm,
        "seed": seed,
        "paper0_commit": commit,
        "slurm_job_id": slurm,
        "development_run": "85604",
        "training": json.loads(
            json.dumps(ECRDTrainingConfig(arm=arm, seed=seed, mode="full").to_record())
        ),
        "model": model_config_record(arm),
        "parameter_count": frozen_parameter_counts()[arm],
        "completed_epochs": 100,
        "completed_optimizer_steps": 10800,
        "target_presentations": 43000,
        "candidate_count": 20,
        "checkpoint_reload_bitwise_exact": True,
        "training_performed": True,
        "validation_frames_read": True,
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "target_truth_used_as_condition": False,
        "absolute_time_used_as_condition": False,
        "guard_frames_read": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "assimilation_performed": False,
        "diagnostic_ranking_performed": False,
        "steering_performed": False,
        "artifacts": {
            "candidate_checkpoints": [
                {"completed_epoch": epoch} for epoch in range(5, 101, 5)
            ]
        },
    }


def test_new_task_matrix_is_exact() -> None:
    assert SUMMARY.NEW_TASKS == (
        (0, "B5", "b5", 1702),
        (1, "B5", "b5", 1703),
        (2, "B5-Context", "b5_context", 1701),
        (3, "B5-Context", "b5_context", 1702),
        (4, "B5-Context", "b5_context", 1703),
        (5, "ECRD", "ecrd", 1701),
        (6, "ECRD", "ecrd", 1702),
        (7, "ECRD", "ecrd", 1703),
        (8, "ECRD-History", "ecrd_history", 1701),
        (9, "ECRD-History", "ecrd_history", 1702),
        (10, "ECRD-History", "ecrd_history", 1703),
    )
    assert len(SUMMARY.expected_artifact_relatives()) == 27


def test_result_audit_requires_complete_budget_and_finished_wandb() -> None:
    commit = "a" * 40
    result = _result(arm="ECRD", seed=1702, commit=commit, slurm="123_6")
    tracking = {
        "mode": "online",
        "remote_state_after_finish": "finished",
        "remote_presence_verified_after_finish": True,
    }
    SUMMARY.audit_result_record(
        result,
        tracking,
        arm="ECRD",
        seed=1702,
        training_commit=commit,
        expected_slurm_job_id="123_6",
    )
    result["completed_optimizer_steps"] = 10799
    with pytest.raises(RuntimeError, match="completed_optimizer_steps"):
        SUMMARY.audit_result_record(
            result,
            tracking,
            arm="ECRD",
            seed=1702,
            training_commit=commit,
            expected_slurm_job_id="123_6",
        )


def test_finalizer_has_no_scientific_evaluation_dependency() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "score_ecrd_forecast" not in source
    assert "tcv_diagnostics.transport" not in source
    assert "tcv_diagnostics.spect" not in source
    assert '"scientific_result": False' in source
    assert '"held_out_85606_read": False' in source


def test_finalizer_launcher_is_read_only_and_dependency_ready() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --gres" not in source
    assert 'readonly ARRAY_JOB_ID="6913340"' in source
    assert 'readonly TRAINING_COMMIT="d822ee2147a98713f1b2ecdfd0f5a4077eded062"' in source
    assert "0a3d2ef6ea45c133c2907bb855ae755edf429278b7f497698c9e470095e81b8d" in source
    assert "summarize_ecrd_full_training.py" in source
    assert "evaluate_ecrd_checkpoint.py" not in source
    assert '"scientific_result": False' in source
    assert '"held_out_85606_read": False' in source
