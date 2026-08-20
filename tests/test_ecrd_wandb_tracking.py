"""Pure mapping tests for ECRD W&B monitoring records."""

from __future__ import annotations

from tcv_diagnostics.ecrd_wandb_tracking import ecrd_epoch_metrics


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
