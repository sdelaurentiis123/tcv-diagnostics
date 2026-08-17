"""Unit tests for fail-closed online W&B tracking without network access."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tcv_diagnostics.wandb_tracking import (
    OnlineWandbTracker,
    WandbRunSpec,
    wandb_epoch_metrics,
    wandb_result_summary,
)


SPEC = WandbRunSpec(
    entity="sdelaurentiis123-columbia-university",
    project="tcv-diagnostics-paper0",
    group="o1-dcae-l20-r1",
    run_id="p0o1r1-123-0",
    run_name="o1-r1-c5p-s1701-j123-t0",
    job_type="phase2_o1_codec_full",
    tags=("paper0", "phase2", "85604-only"),
)


def epoch_record() -> dict:
    return {
        "epoch": 2,
        "global_step": 81,
        "learning_rate": 1.5e-4,
        "mean_preclip_gradient_norm": 0.8,
        "maximum_preclip_gradient_norm": 1.2,
        "train_equal_channel_mae": 0.3,
        "validation_equal_channel_mae": 0.4,
        "validation_mae_by_channel": {"Ne": 0.2, "phi": 0.6},
        "selected_so_far": 1,
        "epoch_wall_seconds": 10.0,
    }


def final_result() -> dict:
    return {
        "completed_epochs": 1,
        "completed_optimizer_steps": 27,
        "selected_epoch": 0,
        "selected_validation_equal_channel_mae": 0.4,
        "final_validation_equal_channel_mae": 0.4,
        "final_validation_mae_by_channel": {"Ne": 0.2, "phi": 0.6},
        "checkpoint_reload_bitwise_exact": True,
        "parameter_count": 123,
        "peak_cuda_bytes": 456,
        "wall_seconds": 78.0,
        "paper0_commit": "a" * 40,
        "selected_checkpoint": {"sha256": "b" * 64},
        "final_training_state": {"sha256": "c" * 64},
        "history": {"sha256": "d" * 64},
        "held_out_85606_read": False,
        "physics_derived_loss_used": False,
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

    def __init__(self, *, api_key: str = "configured") -> None:
        self.api_key = api_key
        self.run = FakeRun()
        self.init_kwargs = None

    def Api(self, *, timeout: int):
        return SimpleNamespace(
            api_key=self.api_key,
            viewer=SimpleNamespace(
                entity=SPEC.entity,
                username="sdelaurentiis123",
            ),
            run=lambda path: SimpleNamespace(id=SPEC.run_id, state="finished"),
        )

    @staticmethod
    def Settings(**kwargs):
        return kwargs

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


class TestWandbTracking(unittest.TestCase):
    def test_epoch_and_final_metric_names_are_explicit(self) -> None:
        metrics = wandb_epoch_metrics(epoch_record())
        self.assertEqual(metrics["validation/channel_mae/phi"], 0.6)
        self.assertEqual(metrics["optimizer/global_step"], 81)
        summary = wandb_result_summary(final_result())
        self.assertTrue(summary["final/checkpoint_reload_bitwise_exact"])
        self.assertFalse(summary["scope/held_out_85606_read"])

    def test_online_run_is_required_and_verified_after_finish(self) -> None:
        fake = FakeWandb()
        with tempfile.TemporaryDirectory() as temporary:
            tracker = OnlineWandbTracker.start(
                spec=SPEC,
                config={"held_out_85606_read": False},
                tracking_directory=Path(temporary) / "wandb",
                wandb_module=fake,
            )
            tracker.log_epoch(epoch_record())
            record = tracker.finish_success(final_result())
        self.assertEqual(fake.init_kwargs["mode"], "online")
        self.assertEqual(fake.init_kwargs["resume"], "never")
        self.assertFalse(fake.init_kwargs["save_code"])
        self.assertEqual(fake.run.logs[0][1], 81)
        self.assertEqual(fake.run.exit_code, 0)
        self.assertTrue(record["remote_presence_verified_after_finish"])
        self.assertFalse(record["checkpoints_uploaded"])

    def test_missing_api_key_fails_before_wandb_init(self) -> None:
        fake = FakeWandb(api_key="")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "no API key"):
                OnlineWandbTracker.start(
                    spec=SPEC,
                    config={},
                    tracking_directory=Path(temporary) / "wandb",
                    wandb_module=fake,
                )
        self.assertIsNone(fake.init_kwargs)


if __name__ == "__main__":
    unittest.main()
