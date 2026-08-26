"""Bounded deterministic rollout primitives for C5P and saved-state E6B."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .model_training_data import FAMILY_FIELDS


def _validate_state(
    volume: Tensor,
    boundary: Tensor | None,
    *,
    family: str,
) -> None:
    if family not in FAMILY_FIELDS:
        raise ValueError(f"unsupported state family {family!r}")
    if volume.ndim != 5 or volume.shape[1] != len(FAMILY_FIELDS[family]):
        raise ValueError("volume must be [batch,field,x,y,z] for its family")
    if family == "c5p":
        if boundary is not None:
            raise ValueError("C5P state cannot contain a boundary profile")
    else:
        expected = (volume.shape[0], 2, volume.shape[-2])
        if boundary is None or tuple(boundary.shape) != expected:
            raise ValueError(f"E6B boundary must have shape {expected}")


def direct_state_forecast(
    model: Any,
    volume: Tensor,
    boundary: Tensor | None,
    *,
    family: str,
    horizon: int,
) -> tuple[Tensor, Tensor | None]:
    """Forecast one terminal state without reading intermediate truth."""

    _validate_state(volume, boundary, family=family)
    horizon_value = int(horizon)
    if horizon_value <= 0:
        raise ValueError("forecast horizon must be positive")
    lead = torch.full(
        (volume.shape[0],),
        float(horizon_value),
        dtype=volume.dtype,
        device=volume.device,
    )
    context_boundary = None if boundary is None else boundary.unsqueeze(1)
    forecast = model.forecast(
        volume.unsqueeze(1),
        lead,
        context_boundary,
    )
    _validate_state(forecast.volume, forecast.boundary, family=family)
    return forecast.volume, forecast.boundary


def autoregressive_state_forecast_path(
    model: Any,
    volume: Tensor,
    boundary: Tensor | None,
    *,
    family: str,
    step: int,
    horizon: int,
) -> tuple[tuple[Tensor, Tensor | None], ...]:
    """Compose complete predicted states without intervening target truth."""

    _validate_state(volume, boundary, family=family)
    step_value = int(step)
    horizon_value = int(horizon)
    if step_value <= 0 or horizon_value <= 0 or horizon_value % step_value:
        raise ValueError("composition step must divide the terminal horizon")
    state = volume
    side_state = boundary
    path: list[tuple[Tensor, Tensor | None]] = []
    for _ in range(horizon_value // step_value):
        state, side_state = direct_state_forecast(
            model,
            state,
            side_state,
            family=family,
            horizon=step_value,
        )
        path.append((state, side_state))
    return tuple(path)
