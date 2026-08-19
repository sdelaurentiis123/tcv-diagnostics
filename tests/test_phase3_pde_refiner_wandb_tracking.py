"""Network-free tests for the required B4 PDE-Refiner W&B mirror."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tcv_diagnostics.pde_refiner_wandb_tracking import (
    PDERefinerOnlineWandbTracker,
    pde_refiner_wandb_epoch_metrics,
    pde_refiner_wandb_result_summary,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="b4-pde-refiner-h1-smoke",
    run_id="p0b4smoke-123-s1701",
    run_name="b4-pde-refiner-h1-smoke-s1701-j123",
    job_type="phase3_b4_pde_refiner_smoke",
    tags=("paper0", "phase3", "b4", "pde-refiner"),
)


def fields(value: float) -> dict[str, float]:
    return {field: value for field in ("Ne", "Pe", "Pi", "phi", "Vi")}


def epoch_record() -> dict:
    return {
        "epoch": 0,
        "global_step": 1,
        "learning_rate": 1e-4,
        "EMA_decay": 0.995,
        "EMA_updates": 1,
        "train_standardized_latent_MSE": 0.4,
        "train_MSE_by_level": {str(level): 0.4 + level for level in range(4)},
        "train_count_by_level": {"0": 4, "1": 6, "2": 2, "3": 4},
        "validation_ensemble_mean_equal_channel_decoded_standardized_field_MAE": 0.5,
        "validation_final_MAE_by_channel": fields(0.5),
        "validation_equal_channel_MAE_by_level": [0.6, 0.55, 0.52, 0.5],
        "mean_preclip_total_gradient_norm": 0.8,
        "maximum_preclip_total_gradient_norm": 0.9,
        "mean_preclip_parent_gradient_norm": 0.7,
        "mean_preclip_refinement_gradient_norm": 0.2,
        "selected_so_far": 0,
        "epoch_wall_seconds": 10.0,
    }


def final_result() -> dict:
    validation = {
        "ensemble_mean_equal_channel_decoded_standardized_field_MAE": 0.5
    }
    return {
        "completed_epochs": 1,
        "completed_optimizer_steps": 1,
        "EMA_updates": 1,
        "selected_epoch": 0,
        "selected_optimizer_step": 1,
        "selected_validation": validation,
        "final_validation": validation,
        "preoptimization_parent_identity": {"bitwise_exact": True},
        "checkpoint_reload_bitwise_exact": True,
        "codec_bitwise_unchanged": True,
        "parent_parameter_gradient_seen": True,
        "refinement_parameter_gradient_seen": True,
        "all_four_training_levels_exercised": True,
        "member_and_stage_probe": {
            "level0_shared_bitwise_across_members": True,
            "nonzero_final_diversity_in_every_field": True,
            "final_member_RMS_difference_by_field": fields(0.1),
        },
        "parameter_count": 123,
        "network_calls_per_member": 4,
        "peak_cuda_bytes": 456,
        "wall_seconds": 78.0,
        "training_dtype": "float32",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "paper0_commit": "a" * 40,
        "selected_checkpoint": {"sha256": "b" * 64},
        "final_training_state": {"sha256": "c" * 64},
        "history": {"sha256": "d" * 64},
        "validation_seed_bank": {"sha256": "e" * 64},
        "training_levels": {"sha256": "f" * 64},
        "validation_decoded_stages": {"sha256": "1" * 64},
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "scientific_result": False,
        "full_B4_training_authorized": False,
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
        self.init_kwargs = None

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
        self.init_kwargs = kwargs
        return self.run


def test_metric_names_expose_levels_ema_precision_and_diversity() -> None:
    metrics = pde_refiner_wandb_epoch_metrics(epoch_record())
    assert metrics["optimizer/ema_updates"] == 1
    assert metrics["train/level_2_count"] == 2
    assert metrics["validation/level_3_equal_channel_mae"] == 0.5
    assert metrics["validation/final_channel_mae/phi"] == 0.5
    summary = pde_refiner_wandb_result_summary(final_result())
    assert summary["compute/network_calls_per_member"] == 4
    assert summary["precision/tf32_disabled"] is True
    assert summary["ensemble/nonzero_final_diversity_in_every_field"] is True
    assert summary["scope/scientific_result"] is False
    assert summary["scope/full_B4_training_authorized"] is False


def test_online_tracker_logs_and_verifies_remote_finish(tmp_path: Path) -> None:
    fake = FakeWandb()
    tracker = PDERefinerOnlineWandbTracker.start(
        spec=SPEC,
        config={"held_out_85606_read": False},
        tracking_directory=tmp_path / "wandb",
        wandb_module=fake,
    )
    tracker.log_epoch(epoch_record())
    record = tracker.finish_success(final_result())
    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["resume"] == "never"
    assert fake.run.logs[0][1] == 1
    assert fake.run.exit_code == 0
    assert record["remote_state_after_finish"] == "finished"
    assert record["epochs_logged"] == 1
    assert record["checkpoints_uploaded"] is False
