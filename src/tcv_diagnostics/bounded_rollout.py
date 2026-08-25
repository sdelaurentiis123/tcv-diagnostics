"""Bounded direct-versus-autoregressive state-forecast primitives.

The routines in this module contain no data access and no physics-derived
training objective.  They implement the inference composition frozen in
Paper 0 amendment A023.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor


FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
HORIZON_METHOD_STEPS: Mapping[int, Mapping[str, int | None]] = {
    4: {
        "direct": None,
        "autoregressive_lead1": 1,
        "autoregressive_lead2": 2,
    },
    8: {
        "direct": None,
        "autoregressive_lead1": 1,
        "autoregressive_lead2": 2,
        "autoregressive_lead4": 4,
    },
}


def method_schedule(horizon: int) -> dict[str, int | None]:
    """Return a copy of the prospectively frozen composition schedule."""

    value = int(horizon)
    if value not in HORIZON_METHOD_STEPS:
        raise ValueError("bounded rollout horizon must be four or eight")
    return dict(HORIZON_METHOD_STEPS[value])


def direct_forecast(model: Any, current: Tensor, *, horizon: int) -> Tensor:
    """Forecast one terminal state directly from a one-frame context."""

    if current.ndim != 5 or current.shape[1] != len(FIELDS):
        raise ValueError("current state must be [batch,5,x,y,z]")
    lead = torch.full(
        (current.shape[0],),
        float(horizon),
        dtype=current.dtype,
        device=current.device,
    )
    forecast = model.forecast(current.unsqueeze(1), lead)
    if forecast.boundary is not None:
        raise ValueError("C5P bounded rollout cannot produce boundary state")
    if forecast.volume.shape != current.shape:
        raise ValueError("direct forecast state shape differs")
    return forecast.volume


def autoregressive_forecast_path(
    model: Any,
    current: Tensor,
    *,
    step: int,
    horizon: int,
) -> tuple[Tensor, ...]:
    """Compose a model without using intermediate or future truth.

    Every returned tensor is the complete predicted C5P state after another
    autonomous composition.  Only the preceding prediction is passed back to
    the model.
    """

    step_value = int(step)
    horizon_value = int(horizon)
    if step_value <= 0 or horizon_value <= 0 or horizon_value % step_value:
        raise ValueError("composition step must divide the terminal horizon")
    if current.ndim != 5 or current.shape[1] != len(FIELDS):
        raise ValueError("current state must be [batch,5,x,y,z]")
    state = current
    path = []
    lead = torch.full(
        (current.shape[0],),
        float(step_value),
        dtype=current.dtype,
        device=current.device,
    )
    for _ in range(horizon_value // step_value):
        forecast = model.forecast(state.unsqueeze(1), lead)
        if forecast.boundary is not None:
            raise ValueError("C5P bounded rollout cannot produce boundary state")
        state = forecast.volume
        if state.shape != current.shape:
            raise ValueError("autoregressive state shape differs")
        path.append(state)
    return tuple(path)


@dataclass
class FieldErrorAccumulator:
    """Stream standardized per-field squared errors and frame RMSE values."""

    squared_error_sum: np.ndarray
    element_count: int
    per_frame_rmse: list[np.ndarray]

    @classmethod
    def empty(cls) -> "FieldErrorAccumulator":
        return cls(
            squared_error_sum=np.zeros(len(FIELDS), dtype=np.float64),
            element_count=0,
            per_frame_rmse=[],
        )

    def update(self, candidate: np.ndarray, truth: np.ndarray) -> np.ndarray:
        first = np.asarray(candidate, dtype=np.float64)
        second = np.asarray(truth, dtype=np.float64)
        if first.shape != second.shape or first.ndim != 5:
            raise ValueError("state errors require matched [batch,5,x,y,z] arrays")
        if first.shape[1] != len(FIELDS):
            raise ValueError("state error field count differs")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError("state error arrays must be finite")
        difference = first - second
        squared = difference * difference
        self.squared_error_sum += np.sum(
            squared,
            axis=(0, 2, 3, 4),
            dtype=np.float64,
        )
        spatial_count = int(np.prod(first.shape[2:]))
        self.element_count += first.shape[0] * spatial_count
        frame_rmse = np.sqrt(np.mean(squared, axis=(2, 3, 4)))
        self.per_frame_rmse.append(frame_rmse)
        return frame_rmse

    def finalize(
        self,
        *,
        persistence_mse: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        if self.element_count <= 0 or not self.per_frame_rmse:
            raise ValueError("cannot finalize an empty state-error accumulator")
        mse = self.squared_error_sum / float(self.element_count)
        rmse = np.sqrt(mse)
        frames = np.concatenate(self.per_frame_rmse, axis=0)
        result = {
            "frame_count": int(frames.shape[0]),
            "scalar_count_per_field": int(self.element_count),
            "per_field": {},
        }
        for index, field in enumerate(FIELDS):
            record: dict[str, float] = {
                "mse": float(mse[index]),
                "rmse": float(rmse[index]),
                "per_frame_rmse_mean": float(np.mean(frames[:, index])),
                "per_frame_rmse_median": float(np.median(frames[:, index])),
                "per_frame_rmse_maximum": float(np.max(frames[:, index])),
            }
            if persistence_mse is not None:
                baseline = float(persistence_mse[field])
                if not math.isfinite(baseline) or baseline <= 0.0:
                    raise ValueError("persistence MSE must be finite and positive")
                record["persistence_relative_skill"] = float(
                    1.0 - mse[index] / baseline
                )
            result["per_field"][field] = record
        result["mean_field_mse"] = float(np.mean(mse))
        result["mean_field_rmse"] = float(np.mean(rmse))
        if persistence_mse is not None:
            baseline_mean = float(
                np.mean([float(persistence_mse[field]) for field in FIELDS])
            )
            result["mean_field_persistence_relative_skill"] = float(
                1.0 - np.mean(mse) / baseline_mean
            )
        return result
