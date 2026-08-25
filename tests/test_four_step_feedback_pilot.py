"""Known-answer checks for the old-85604 four-step feedback pilot."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paper0.tools.train_codec_free_four_step_feedback import (
    SELECTION_STARTS,
    authorize_manifest,
    state_gate,
)
from tcv_diagnostics.autoregressive_training import (
    autoregressive_forecast_sequence,
    feedback_loss_weights,
    plan_autoregressive_windows,
    state_rms_normalized_mse,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/post_ecrd_old_85604_four_step_feedback_pilot.json"
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_four_step_feedback_pilot.sbatch"


class _UnitIncrement:
    def forecast(self, context: torch.Tensor, lead_steps: torch.Tensor) -> SimpleNamespace:
        increment = lead_steps.reshape(-1, 1, 1, 1, 1)
        return SimpleNamespace(volume=context[:, -1] + increment)


def test_window_counts_and_boundaries_are_frozen() -> None:
    train = plan_autoregressive_windows(split="train", horizon=4)
    validation4 = plan_autoregressive_windows(split="validation", horizon=4)
    validation8 = plan_autoregressive_windows(split="validation", horizon=8)
    assert len(train) == 428
    assert len(validation4) == 124
    assert len(validation8) == 120
    assert train[0].current == 0 and train[-1].targets[-1] == 431
    assert validation8[0].current == 496 and validation8[-1].targets[-1] == 623
    consumed = {
        frame
        for window in train + validation8
        for frame in (window.current, *window.targets)
    }
    assert not any(432 <= frame < 496 for frame in consumed)


def test_loss_weights_retain_the_one_step_term() -> None:
    assert feedback_loss_weights(horizon=4, direct_one_step_weight=0.5) == (
        0.625,
        0.125,
        0.125,
        0.125,
    )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        feedback_loss_weights(horizon=4, direct_one_step_weight=1.1)


def test_autoregressive_sequence_uses_complete_predictions() -> None:
    context = torch.zeros(1, 1, 2, 2, 2, 3)
    sequence = autoregressive_forecast_sequence(_UnitIncrement(), context, steps=4)
    assert len(sequence) == 4
    for step, state in enumerate(sequence, start=1):
        assert torch.equal(state, torch.full_like(state, float(step)))


def test_state_loss_is_equal_field_and_training_scale_normalized() -> None:
    candidate = torch.tensor([1.0, 4.0]).reshape(1, 2, 1, 1, 1)
    target = torch.zeros_like(candidate)
    total, per_field = state_rms_normalized_mse(
        candidate, target, torch.tensor([1.0, 2.0])
    )
    assert torch.equal(per_field, torch.tensor([1.0, 4.0]))
    assert float(total) == pytest.approx(2.5)


def test_manifest_authorizes_only_the_frozen_old_85604_pilot() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    authorize_manifest(manifest, seed=1702)
    assert tuple(manifest["split"]["selection_current_frames"]) == SELECTION_STARTS
    with pytest.raises(ValueError, match="seed"):
        authorize_manifest(manifest, seed=1701)
    broken = copy.deepcopy(manifest)
    broken["held_out_85606_access_allowed"] = True
    with pytest.raises(ValueError, match="scope"):
        authorize_manifest(broken, seed=1702)
    broken = copy.deepcopy(manifest)
    broken["loss"]["physics_derived_quantities_used"] = True
    with pytest.raises(ValueError, match="loss"):
        authorize_manifest(broken, seed=1702)


def test_launcher_requests_one_startable_gpu_and_online_tracking() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=20G" in source
    assert "#SBATCH --time=02:00:00" in source
    assert "export WANDB_MODE=online" in source
    assert "--array" not in source
    assert "85606" not in source


def _evaluation(h1: float, h4: float, h8: float) -> dict:
    return {
        "horizons": {
            str(horizon): {"mean_field_model_state_mse": value}
            for horizon, value in ((1, h1), (4, h4), (8, h8))
        }
    }


def test_state_gate_requires_retention_and_long_horizon_improvement() -> None:
    thresholds = {
        "maximum_one_step_error_ratio_to_parent": 1.05,
        "maximum_four_step_error_ratio_to_parent": 1.0,
        "maximum_eight_step_error_ratio_to_parent": 1.0,
        "minimum_mean_four_eight_improvement_fraction": 0.05,
    }
    passed = state_gate(
        parent=_evaluation(1.0, 2.0, 4.0),
        candidate=_evaluation(1.04, 1.8, 3.6),
        thresholds=thresholds,
    )
    assert passed["passed"] is True
    assert passed["mean_four_eight_improvement_fraction"] == pytest.approx(0.10)
    failed = state_gate(
        parent=_evaluation(1.0, 2.0, 4.0),
        candidate=_evaluation(1.06, 1.8, 3.6),
        thresholds=thresholds,
    )
    assert failed["passed"] is False
    assert failed["gates"]["one_step_error_retained"] is False
