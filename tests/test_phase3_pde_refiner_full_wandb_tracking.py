"""Network-free tests for the required full B4 W&B mirror."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tcv_diagnostics.pde_refiner_full_wandb_tracking import (
    PDERefinerFullOnlineWandbTracker,
    full_refiner_epoch_metrics,
    full_refiner_result_summary,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="b4-pde-refiner-h1-full",
    run_id="p0b4full-123-s1701",
    run_name="b4-pde-refiner-h1-full-s1701-j123",
    job_type="phase3_b4_pde_refiner_full",
    tags=("paper0", "phase3", "b4", "pde-refiner", "full"),
)


def validation(value: float = 0.5) -> dict:
    return {
        "ensemble_mean_equal_channel_decoded_standardized_field_MAE": value,
        "final_MAE_by_channel": {
            field: value for field in ("Ne", "Pe", "Pi", "phi", "Vi")
        },
        "equal_channel_MAE_by_level": [value + 0.03, value + 0.02, value + 0.01, value],
    }


def epoch_record(*, performed: bool) -> dict:
    return {
        "epoch": 4,
        "completed_epoch": 5,
        "global_step": 135,
        "learning_rate": 9.9e-5,
        "EMA_decay": 0.995,
        "EMA_updates": 135,
        "train_standardized_latent_MSE": 0.4,
        "train_MSE_by_level": {str(level): 0.4 + level for level in range(4)},
        "train_count_by_level": {"0": 108, "1": 107, "2": 107, "3": 108},
        "mean_preclip_total_gradient_norm": 0.8,
        "maximum_preclip_total_gradient_norm": 0.9,
        "mean_preclip_parent_gradient_norm": 0.7,
        "mean_preclip_refinement_gradient_norm": 0.2,
        "validation_performed": performed,
        "validation": validation() if performed else None,
        "selected_so_far": 4 if performed else None,
        "epoch_wall_seconds": 10.0,
    }


def result_record() -> dict:
    return {
        "completed_epochs": 100,
        "completed_optimizer_steps": 2700,
        "EMA_updates": 2700,
        "validation_candidates_evaluated": 20,
        "selected_epoch": 49,
        "selected_completed_epoch": 50,
        "selected_optimizer_step": 1350,
        "selected_validation": validation(0.4),
        "final_validation": validation(0.5),
        "preoptimization_parent_identity": {"bitwise_exact": True},
        "checkpoint_reload_bitwise_exact": True,
        "codec_bitwise_unchanged": True,
        "parent_parameter_gradient_seen": True,
        "refinement_parameter_gradient_seen": True,
        "all_four_training_levels_exercised": True,
        "parameter_count": 61_218_944,
        "network_calls_per_unamortized_member": 4,
        "peak_cuda_bytes": 123,
        "wall_seconds": 456.0,
        "training_dtype": "float32",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "paper0_commit": "a" * 40,
        "selected_checkpoint": {"sha256": "b" * 64},
        "final_training_state": {"sha256": "c" * 64},
        "history": {"sha256": "d" * 64},
        "validation_seed_bank": {"sha256": "e" * 64},
        "training_levels": {"sha256": "f" * 64},
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "scientific_result": False,
        "training_complete_is_scientific_acceptance": False,
        "H_det_evaluated": False,
        "H_prob_evaluated": False,
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


def test_sparse_validation_metrics_do_not_invent_values() -> None:
    absent = full_refiner_epoch_metrics(epoch_record(performed=False))
    assert absent["validation/performed"] == 0
    assert absent["validation/selected_epoch_so_far"] == -1
    assert not any(key.startswith("validation/level_") for key in absent)
    present = full_refiner_epoch_metrics(epoch_record(performed=True))
    assert present["validation/performed"] == 1
    assert present["validation/selected_epoch_so_far"] == 4
    assert present["validation/level_3_equal_channel_mae"] == 0.5
    assert present["validation/final_channel_mae/phi"] == 0.5


def test_result_summary_keeps_training_separate_from_scientific_gates() -> None:
    summary = full_refiner_result_summary(result_record())
    assert summary["final/completed_epochs"] == 100
    assert summary["final/validation_candidates_evaluated"] == 20
    assert summary["compute/parameter_count"] == 61_218_944
    assert summary["precision/tf32_disabled"] is True
    assert summary["scope/scientific_result"] is False
    assert summary["scope/training_complete_is_scientific_acceptance"] is False
    assert summary["scope/H_det_evaluated"] is False
    assert summary["scope/H_prob_evaluated"] is False


def test_full_tracker_logs_100_epochs_and_verifies_remote_finish(
    tmp_path: Path,
) -> None:
    fake = FakeWandb()
    tracker = PDERefinerFullOnlineWandbTracker.start(
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
    assert record["checkpoints_uploaded"] is False
