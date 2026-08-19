"""Network-free tests for the required B5 residual-EDM W&B mirror."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tcv_diagnostics.b5_residual_edm_wandb_tracking import (
    B5EDMOnlineWandbTracker,
    b5_edm_wandb_result_summary,
    b5_edm_wandb_step_metrics,
)
from tcv_diagnostics.wandb_tracking import WandbRunSpec


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="b5-joint-field-residual-edm-smoke",
    run_id="p0b5edmsmoke-123-s1701",
    run_name="b5-joint-field-residual-edm-smoke-s1701-j123",
    job_type="phase3_b5_joint_field_residual_edm_smoke",
    tags=("paper0", "phase3", "b5", "residual-edm", "smoke"),
)


def step_record(step: int = 1) -> dict:
    return {
        "global_step": step,
        "target_frame": 6,
        "context_frame": 5,
        "sigma": 1.25,
        "EDM_loss": 0.75,
        "unweighted_MSE": 0.5,
        "learning_rate": 1.0e-4,
        "preclip_gradient_norm": 0.8,
        "step_wall_seconds": 2.0,
    }


def final_result(*, steps: int = 64, passed: bool = True) -> dict:
    return {
        "status": "passed" if passed else "failed",
        "all_mechanical_gates_passed": passed,
        "completed_optimizer_steps": steps,
        "initial_fixed_probe": {"mean_EDM_loss": 1.0},
        "final_fixed_probe": {"mean_EDM_loss": 0.5},
        "fixed_probe_relative_change": -0.5,
        "checkpoint_reload_bitwise_exact": True,
        "toroidal_equivariance": {"passed": True},
        "sampler_probe": {
            "normalized_residual_member_RMS_difference": 0.2,
            "standardized_field_member_RMS_difference": 0.01,
            "nonzero_member_diversity": True,
            "network_evaluations_per_member": 35,
        },
        "parameter_count": 123,
        "peak_cuda_bytes": 456,
        "peak_cuda_GiB": 0.000001,
        "wall_seconds": 78.0,
        "paper0_commit": "a" * 40,
        "artifacts": {
            "smoke_checkpoint": {"sha256": "b" * 64},
            "history": {"sha256": "c" * 64},
        },
        "scientific_result": False,
        "full_B5_training_authorized": False,
        "validation_frames_read": False,
        "held_out_85606_read": False,
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


def test_B5_WandB_metric_names_preserve_loss_noise_and_scope() -> None:
    metrics = b5_edm_wandb_step_metrics(step_record())
    assert metrics["optimizer/global_step"] == 1
    assert metrics["data/target_frame"] == 6
    assert metrics["noise/sigma"] == 1.25
    assert metrics["train/EDM_loss"] == 0.75
    summary = b5_edm_wandb_result_summary(final_result())
    assert summary["compute/network_evaluations_per_member"] == 35
    assert summary["ensemble/nonzero_member_diversity"] is True
    assert summary["scope/scientific_result"] is False
    assert summary["scope/full_B5_training_authorized"] is False


def test_B5_WandB_tracker_requires_exactly_64_steps_and_remote_finish(
    tmp_path: Path,
) -> None:
    fake = FakeWandb()
    tracker = B5EDMOnlineWandbTracker.start(
        spec=SPEC,
        config={"held_out_85606_read": False},
        tracking_directory=tmp_path / "wandb",
        wandb_module=fake,
    )
    for step in range(1, 65):
        tracker.log_step(step_record(step))
    record = tracker.finish_success(final_result())
    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["resume"] == "never"
    assert fake.run.logs[-1][1] == 64
    assert fake.run.exit_code == 0
    assert record["remote_state_after_finish"] == "finished"
    assert record["steps_logged"] == 64
    assert record["checkpoints_uploaded"] is False

    short_fake = FakeWandb()
    short = B5EDMOnlineWandbTracker.start(
        spec=SPEC,
        config={},
        tracking_directory=tmp_path / "short-wandb",
        wandb_module=short_fake,
    )
    short.log_step(step_record())
    with pytest.raises(RuntimeError, match="expected 64"):
        short.finish_success(final_result())
