"""Network-free tests for the required full B5 W&B mirror."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tcv_diagnostics.b5_residual_edm_full_wandb_tracking import (
    B5EDMFullOnlineWandbTracker,
    b5_full_epoch_metrics,
    b5_full_result_summary,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="b5-joint-field-residual-edm-full",
    run_id="p0b5full-123-s1701",
    run_name="b5-joint-field-residual-edm-full-s1701-j123",
    job_type="phase3_b5_joint_field_residual_edm_full",
    tags=("paper0", "phase3", "b5", "residual-edm", "full"),
)


def validation(value: float = 0.5) -> dict:
    return {
        "mean_EDM_loss": value,
        "mean_unweighted_MSE": value / 2,
        "probe_count": 504,
        "minimum_sigma": 0.01,
        "maximum_sigma": 20.0,
        "wall_seconds": 3.0,
    }


def epoch_record(*, performed: bool) -> dict:
    return {
        "completed_epoch": 5,
        "global_optimizer_step": 540,
        "train_target_count": 430,
        "train_mean_EDM_loss": 0.4,
        "train_mean_unweighted_MSE": 0.2,
        "train_minimum_sigma": 0.01,
        "train_maximum_sigma": 20.0,
        "mean_preclip_gradient_norm": 0.8,
        "maximum_preclip_gradient_norm": 0.9,
        "first_learning_rate": 9.9e-5,
        "last_learning_rate": 9.8e-5,
        "EMA_updates": 540,
        "validation_candidate": performed,
        "validation": validation() if performed else None,
        "epoch_wall_seconds": 10.0,
    }


def result_record() -> dict:
    artifact = lambda value: {"sha256": value * 64}
    return {
        "status": "training_completed_checkpoint_selected",
        "completed_epochs": 100,
        "completed_optimizer_steps": 10_800,
        "EMA_updates": 10_800,
        "candidate_count": 20,
        "selected_completed_epoch": 50,
        "selected_optimizer_step": 5_400,
        "selected_validation": validation(0.4),
        "final_candidate_validation": validation(0.5),
        "checkpoint_reload_bitwise_exact": True,
        "all_losses_and_gradients_finite": True,
        "parameter_count": 11_604_709,
        "peak_cuda_bytes": 123,
        "peak_cuda_GiB": 0.1,
        "wall_seconds": 456.0,
        "paper0_commit": "a" * 40,
        "artifacts": {
            "selected_checkpoint": artifact("b"),
            "final_training_state": artifact("c"),
            "history": artifact("d"),
            "training_order": artifact("e"),
            "validation_seed_bank": artifact("f"),
        },
        "physics_derived_loss_used": False,
        "physics_metric_used_for_checkpoint_selection": False,
        "held_out_85606_read": False,
        "scientific_forecast_generated": False,
        "scientific_acceptance_evaluated": False,
    }


class FakeRun:
    def __init__(self) -> None:
        self.id = SPEC.run_id
        self.url = "https://wandb.ai/example/run"
        self.offline = False
        self.summary: dict = {}
        self.logs: list[tuple[dict, int, bool]] = []
        self.exit_code = None

    def log(self, metrics: dict, *, step: int, commit: bool) -> None:
        self.logs.append((metrics, step, commit))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class FakeWandb:
    __version__ = "test"

    def __init__(self) -> None:
        self.run = FakeRun()

    def Api(self, *, timeout: int):
        return SimpleNamespace(
            api_key="configured",
            viewer=SimpleNamespace(entity=SPEC.entity, username="test-user"),
            run=lambda path: SimpleNamespace(id=SPEC.run_id, state="finished"),
        )

    @staticmethod
    def Settings(**kwargs):
        return kwargs

    def init(self, **kwargs):
        return self.run


def test_B5_full_sparse_validation_metrics_do_not_invent_values() -> None:
    absent = b5_full_epoch_metrics(epoch_record(performed=False))
    assert absent["validation/performed"] == 0
    assert "validation/mean_EDM_loss" not in absent
    present = b5_full_epoch_metrics(epoch_record(performed=True))
    assert present["validation/performed"] == 1
    assert present["validation/mean_EDM_loss"] == 0.5
    assert present["validation/probe_count"] == 504


def test_B5_full_result_summary_keeps_training_separate_from_science() -> None:
    summary = b5_full_result_summary(result_record())
    assert summary["final/training_completed"] is True
    assert summary["final/selected_completed_epoch"] == 50
    assert summary["compute/parameter_count"] == 11_604_709
    assert summary["scope/scientific_forecast_generated"] is False
    assert summary["scope/scientific_acceptance_evaluated"] is False


def test_B5_full_tracker_logs_100_epochs_and_verifies_remote_finish(
    tmp_path: Path,
) -> None:
    fake = FakeWandb()
    tracker = B5EDMFullOnlineWandbTracker.start(
        spec=SPEC,
        config={"held_out_85606_read": False},
        tracking_directory=tmp_path / "wandb",
        wandb_module=fake,
    )
    for _ in range(100):
        tracker.log_epoch(epoch_record(performed=False))
    record = tracker.finish_success(result_record())
    assert len(fake.run.logs) == 100
    assert fake.run.exit_code == 0
    assert record["remote_state_after_finish"] == "finished"
    assert record["epochs_logged"] == 100
    assert record["samples_uploaded"] is False
