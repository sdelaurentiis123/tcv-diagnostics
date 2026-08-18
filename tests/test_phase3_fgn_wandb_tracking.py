"""Network-free tests for the required B3 FGN W&B mirror."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tcv_diagnostics.fgn_wandb_tracking import (
    FGNOnlineWandbTracker,
    fgn_wandb_epoch_metrics,
    fgn_wandb_result_summary,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="b3-fgn-h1-smoke",
    run_id="p0b3smoke-123-s1701",
    run_name="b3-fgn-h1-smoke-s1701-j123",
    job_type="phase3_b3_fgn_smoke",
    tags=("paper0", "phase3", "b3", "fgn"),
)


def _fields(value: float) -> dict[str, float]:
    return {field: value for field in ("Ne", "Pe", "Pi", "phi", "Vi")}


def epoch_record() -> dict:
    return {
        "epoch": 0,
        "global_step": 1,
        "common_learning_rate": 3.0e-5,
        "new_learning_rate": 1.0e-4,
        "mean_preclip_total_gradient_norm": 0.8,
        "maximum_preclip_total_gradient_norm": 0.9,
        "mean_preclip_common_gradient_norm": 0.7,
        "mean_preclip_new_gradient_norm": 0.2,
        "train_equal_channel_fair_crps": 0.4,
        "train_fair_crps_by_channel": _fields(0.4),
        "train_accuracy_by_channel": _fields(0.7),
        "train_spread_by_channel": _fields(0.3),
        "validation_equal_channel_fair_crps": 0.5,
        "validation_fair_crps_by_channel": _fields(0.5),
        "validation_accuracy_by_channel": _fields(0.8),
        "validation_spread_by_channel": _fields(0.3),
        "selected_so_far": 0,
        "epoch_wall_seconds": 10.0,
    }


def final_result() -> dict:
    return {
        "completed_epochs": 1,
        "completed_optimizer_steps": 1,
        "selected_epoch": 0,
        "selected_validation": {"equal_channel_fair_crps": 0.5},
        "final_validation": {"equal_channel_fair_crps": 0.5},
        "preoptimization_parent_identity": {"bitwise_exact": True},
        "checkpoint_reload_bitwise_exact": True,
        "codec_bitwise_unchanged": True,
        "common_parameter_gradient_seen": True,
        "new_parameter_gradient_seen": True,
        "member_probe": {
            "latent_member_rms_difference": 0.5,
            "field_member_rms_difference": 0.2,
            "nonzero_latent_diversity": True,
            "nonzero_field_diversity": True,
        },
        "parameter_count": 123,
        "peak_cuda_bytes": 456,
        "wall_seconds": 78.0,
        "paper0_commit": "a" * 40,
        "selected_checkpoint": {"sha256": "b" * 64},
        "final_training_state": {"sha256": "c" * 64},
        "history": {"sha256": "d" * 64},
        "validation_noise_bank": {"sha256": "e" * 64},
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "scientific_result": False,
        "full_B3_training_authorized": False,
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


def test_fgn_metric_names_separate_accuracy_spread_and_parameter_groups() -> None:
    metrics = fgn_wandb_epoch_metrics(epoch_record())
    assert metrics["optimizer/common_learning_rate"] == 3.0e-5
    assert metrics["optimizer/new_learning_rate"] == 1.0e-4
    assert metrics["validation/fair_crps/phi"] == 0.5
    assert metrics["validation/accuracy/phi"] == 0.8
    assert metrics["validation/spread/phi"] == 0.3
    summary = fgn_wandb_result_summary(final_result())
    assert summary["ensemble/nonzero_field_diversity"] is True
    assert summary["final/preoptimization_parent_bitwise_exact"] is True
    assert summary["scope/scientific_result"] is False


def test_fgn_online_tracker_logs_and_verifies_remote_finish(tmp_path: Path) -> None:
    fake = FakeWandb()
    tracker = FGNOnlineWandbTracker.start(
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
    assert record["checkpoints_uploaded"] is False
