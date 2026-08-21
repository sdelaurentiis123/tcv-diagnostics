"""Pure mapping tests for ECRD W&B monitoring records."""

from __future__ import annotations

from tcv_diagnostics.ecrd_wandb_tracking import (
    ecrd_epoch_metrics,
    ecrd_result_summary,
    verify_ecrd_remote_finished_run,
)

from types import SimpleNamespace

import pytest


def test_epoch_mapping_preserves_validation_blocks() -> None:
    validation = {
        "checkpoint_score": 2.5,
        "aggregate": {"objective": 2.4, "edm_loss": 2.0, "mean_mse": 0.4},
        "blocks": {
            name: {"objective": value, "edm_loss": value - 0.4, "mean_mse": 0.4}
            for name, value in (("V00", 2.0), ("V01", 2.5), ("V02", 3.0))
        },
        "wall_seconds": 7.0,
    }
    record = {
        "completed_epoch": 5,
        "global_optimizer_step": 540,
        "first_learning_rate": 1.0e-4,
        "last_learning_rate": 9.0e-5,
        "mean_preclip_gradient_norm": 1.2,
        "maximum_preclip_gradient_norm": 2.2,
        "train_target_count": 430,
        "train_mean_objective": 3.0,
        "train_mean_edm_loss": 2.4,
        "train_mean_unweighted_edm_mse": 0.8,
        "train_mean_mean_mse": 0.6,
        "validation_candidate": True,
        "validation": validation,
        "epoch_wall_seconds": 12.0,
    }
    metrics = ecrd_epoch_metrics(record)
    assert metrics["validation/checkpoint_score"] == 2.5
    assert metrics["validation/V00/objective"] == 2.0
    assert metrics["validation/V02/mean_head_MSE"] == 0.4


def test_result_summary_keeps_scope_and_selected_hash() -> None:
    result = {
        "status": "training_completed_checkpoint_selected",
        "completed_epochs": 1,
        "completed_optimizer_steps": 2,
        "candidate_count": 1,
        "selected_completed_epoch": 1,
        "selected_validation": {"checkpoint_score": 2.0},
        "checkpoint_reload_bitwise_exact": True,
        "parameter_count": 11,
        "peak_cuda_memory_GiB": 1.5,
        "wall_seconds": 3.0,
        "paper0_commit": "abc",
        "artifacts": {
            "selected_checkpoint": {"sha256": "d" * 64},
            "history": {"sha256": "e" * 64},
        },
        "physics_derived_loss_used": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
    }
    summary = ecrd_result_summary(result)
    assert summary["final/selected_validation_objective"] == 2.0
    assert summary["provenance/selected_checkpoint_sha256"] == "d" * 64
    assert summary["scope/held_out_85606_read"] is False


def test_ecrd_remote_completion_retries_read_api_propagation() -> None:
    states = iter(("running", "running", "finished"))
    sleeps: list[float] = []

    class SequencedWandb:
        @staticmethod
        def Api(*, timeout: int):
            return SimpleNamespace(
                run=lambda path: SimpleNamespace(
                    id="expected",
                    state=next(states),
                )
            )

    remote = verify_ecrd_remote_finished_run(
        module=SequencedWandb,
        remote_path="entity/project/expected",
        expected_run_id="expected",
        retry_delays_seconds=(0.1, 0.2),
        sleep=sleeps.append,
    )
    assert remote.state == "finished"
    assert sleeps == [0.1, 0.2]


def test_ecrd_remote_identity_mismatch_fails_without_retry() -> None:
    sleeps: list[float] = []

    class WrongRunWandb:
        @staticmethod
        def Api(*, timeout: int):
            return SimpleNamespace(
                run=lambda path: SimpleNamespace(id="wrong", state="finished")
            )

    with pytest.raises(RuntimeError, match="identity"):
        verify_ecrd_remote_finished_run(
            module=WrongRunWandb,
            remote_path="entity/project/expected",
            expected_run_id="expected",
            retry_delays_seconds=(0.1,),
            sleep=sleeps.append,
        )
    assert sleeps == []
