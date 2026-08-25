"""Known-answer checks for the old-85604 bounded rollout primitives."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tcv_diagnostics.bounded_rollout import (
    FIELDS,
    FieldErrorAccumulator,
    autoregressive_forecast_path,
    direct_forecast,
    method_schedule,
)


class _ExactIncrementModel:
    def forecast(self, context: torch.Tensor, lead: torch.Tensor):
        value = lead.reshape(-1, 1, 1, 1, 1)
        return SimpleNamespace(
            volume=context[:, -1] + value,
            boundary=None,
        )


def test_frozen_method_schedule_is_exact() -> None:
    assert method_schedule(4) == {
        "direct": None,
        "autoregressive_lead1": 1,
        "autoregressive_lead2": 2,
    }
    assert method_schedule(8) == {
        "direct": None,
        "autoregressive_lead1": 1,
        "autoregressive_lead2": 2,
        "autoregressive_lead4": 4,
    }
    with pytest.raises(ValueError, match="four or eight"):
        method_schedule(16)


def test_direct_and_composed_paths_use_only_predicted_state() -> None:
    current = torch.zeros(2, len(FIELDS), 3, 2, 4)
    model = _ExactIncrementModel()
    direct = direct_forecast(model, current, horizon=8)
    path = autoregressive_forecast_path(model, current, step=2, horizon=8)
    assert len(path) == 4
    assert torch.equal(direct, torch.full_like(current, 8.0))
    for depth, state in enumerate(path, start=1):
        assert torch.equal(state, torch.full_like(current, 2.0 * depth))


def test_composition_step_must_divide_horizon() -> None:
    current = torch.zeros(1, len(FIELDS), 2, 2, 2)
    with pytest.raises(ValueError, match="must divide"):
        autoregressive_forecast_path(_ExactIncrementModel(), current, step=3, horizon=8)


def test_field_error_accumulator_reports_known_skill() -> None:
    truth = np.ones((2, len(FIELDS), 2, 1, 1), dtype=np.float32)
    persistence = np.zeros_like(truth)
    candidate = np.full_like(truth, 0.5)
    baseline = FieldErrorAccumulator.empty()
    baseline.update(persistence, truth)
    baseline_record = baseline.finalize()
    baseline_mse = {
        field: baseline_record["per_field"][field]["mse"] for field in FIELDS
    }
    model = FieldErrorAccumulator.empty()
    frame_rmse = model.update(candidate, truth)
    result = model.finalize(persistence_mse=baseline_mse)
    assert frame_rmse.shape == (2, len(FIELDS))
    for field in FIELDS:
        assert result["per_field"][field]["mse"] == pytest.approx(0.25)
        assert result["per_field"][field]["rmse"] == pytest.approx(0.5)
        assert result["per_field"][field]["persistence_relative_skill"] == (
            pytest.approx(0.75)
        )
