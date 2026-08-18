"""Regression lock for the completed six-run O2 training freeze, job 6895637."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase2_o2_training_freeze_6895637.json"
RESULT_SHA256 = "dd8951e39e60d1631866ebe7af7c4d529ad543daf211233369b8fec9936ee837"


def test_completed_o2_training_matrix_is_immutable_and_scientifically_undecided():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["scope"] == "phase2_C5P_O2_full_training_matrix_frozen"
    assert result["status"] == "completed_pending_scientific_O2_evaluation"
    assert result["development_run"] == "85604"
    assert result["held_out_85606_read"] is False
    assert result["training_commit"] == (
        "9035bc3ce9d2351cd17586f4429af8116d43a47e"
    )
    assert result["training_slurm_job_id"] == "6894980"
    assert result["audit_commit"] == (
        "ac164eb06fe62cf91dd0be11ee3cbb5565cec639"
    )
    assert result["audit_slurm_job_id"] == "6895637"
    assert result["checkpoint_choice_frozen_before_reference_or_physics_metrics"] is True
    assert result["O2_scientific_evaluation_completed"] is False
    assert result["O2_accepted_arms"] == []
    assert result["O3_launch_allowed"] is False


def test_all_six_selected_checkpoints_and_training_only_summaries_are_frozen():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    expected = [
        (0, "C5P-H1", 1701, 193, "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"),
        (1, "C5P-H1", 1702, 198, "d15c74717fad6a3ccf5b5af895e3eefb7271667f4bbde2164514a61a526bc0e8"),
        (2, "C5P-H1", 1703, 198, "a718b2135c7019d05541bd5ffb029ce9408df8225603cffc957c42d2ce5abae3"),
        (3, "C5P-H2", 1701, 198, "3b971b2081901469e1f98adbe27b5cdbf3281d08a99ee28e0d8d8b1577722a84"),
        (4, "C5P-H2", 1702, 199, "5edc3e002730eb78232967255cfab66ee860b8b3858eed007f7061341b5c36eb"),
        (5, "C5P-H2", 1703, 191, "a70bd271117f1b0afb21258e4c5d7d4eb4919dc4a528509ccbf6ac2464622d85"),
    ]
    observed = [
        (
            int(run["run_index"]),
            run["arm"],
            int(run["seed"]),
            int(run["selected_epoch"]),
            run["selected_checkpoint"]["sha256"],
        )
        for run in result["runs"]
    ]
    assert observed == expected
    assert all(run["parameter_count"] == 51_612_800 for run in result["runs"])
    assert all(run["wandb"]["remote_state"] == "finished" for run in result["runs"])
    comparison = result["training_loss_comparison_only"]
    assert comparison["may_select_an_arm"] is False
    assert comparison["C5P-H1"]["mean_selected_validation_equal_channel_mae"] == (
        0.04545187584443883
    )
    assert comparison["C5P-H2"]["mean_selected_validation_equal_channel_mae"] == (
        0.04569007103892144
    )
