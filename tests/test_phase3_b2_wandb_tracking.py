"""Network-free tests for the required B2 W&B mirror."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tcv_diagnostics.b2_wandb_tracking import (
    B2OnlineWandbTracker,
    b2_wandb_epoch_metrics,
    b2_wandb_result_summary,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="b2-ldm-smoke",
    run_id="p0b2smoke-123",
    run_name="b2-ldm-smoke-s1701-j123",
    job_type="phase3_b2_ldm_smoke",
    tags=("paper0", "phase3", "b2"),
)


def epoch_record() -> dict:
    return {
        "epoch": 0,
        "global_step": 1,
        "learning_rate": 1.0e-4,
        "mean_preclip_gradient_norm": 0.8,
        "maximum_preclip_gradient_norm": 0.8,
        "train_complete_denoising_loss": 1.0,
        "train_context_denoising_loss": 0.9,
        "train_target_denoising_loss": 1.2,
        "validation_complete_denoising_loss": 1.1,
        "validation_context_denoising_loss": 1.0,
        "validation_target_denoising_loss": 1.3,
        "selected_so_far": 0,
        "epoch_wall_seconds": 10.0,
    }


def final_result() -> dict:
    return {
        "completed_epochs": 1,
        "completed_optimizer_steps": 1,
        "selected_epoch": 0,
        "selected_validation": {"complete": 1.1, "context": 1.0, "target": 1.3},
        "final_validation": {"complete": 1.1, "context": 1.0, "target": 1.3},
        "checkpoint_reload_bitwise_exact": True,
        "sampler_probe": {
            "latent_member_rms_difference": 0.5,
            "field_member_rms_difference": 0.2,
            "nonzero_latent_diversity": True,
        },
        "parameter_count": 123,
        "peak_cuda_bytes": 456,
        "wall_seconds": 78.0,
        "paper0_commit": "a" * 40,
        "selected_checkpoint": {"sha256": "b" * 64},
        "final_training_state": {"sha256": "c" * 64},
        "history": {"sha256": "d" * 64},
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
        "scientific_result": False,
        "full_B2_training_authorized": False,
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


def test_b2_metric_names_separate_context_and_target() -> None:
    metrics = b2_wandb_epoch_metrics(epoch_record())
    assert metrics["train/context_denoising_loss"] == 0.9
    assert metrics["train/target_denoising_loss"] == 1.2
    summary = b2_wandb_result_summary(final_result())
    assert summary["sampler/nonzero_latent_diversity"] is True
    assert summary["scope/scientific_result"] is False


def test_b2_online_tracker_logs_and_verifies_remote_finish(tmp_path: Path) -> None:
    fake = FakeWandb()
    tracker = B2OnlineWandbTracker.start(
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

