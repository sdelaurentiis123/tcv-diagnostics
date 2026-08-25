"""Leakage-safe short-unroll utilities for the old-85604 state operator.

The helpers in this module expose a deterministic transition to its own
predicted states.  They do not define or evaluate any derived physics loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from .model_training_data import (
    FAMILY_FIELDS,
    GUARD_INTERVAL,
    ModelDatasetCatalog,
    TRAIN_INTERVAL,
    VALIDATION_INTERVAL,
    toroidal_roll,
)
from .state_operator_data import LeadTimeStateDataset


@dataclass(frozen=True, order=True)
class AutoregressiveWindow:
    """One current frame and every consecutive target through a fixed horizon."""

    current: int
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.current < 0 or not self.targets:
            raise ValueError("autoregressive window must be nonempty and nonnegative")
        expected = tuple(range(self.current + 1, self.current + len(self.targets) + 1))
        if self.targets != expected:
            raise ValueError("autoregressive targets must be consecutive")

    @property
    def horizon(self) -> int:
        return len(self.targets)


def plan_autoregressive_windows(
    *,
    split: str,
    horizon: int,
    history_frames: int = 1,
    current_interval: Sequence[int] | None = None,
) -> tuple[AutoregressiveWindow, ...]:
    """Plan consecutive windows wholly contained in one frozen split."""

    if history_frames != 1:
        raise ValueError("this short-unroll phase requires one history frame")
    if int(horizon) <= 0:
        raise ValueError("autoregressive horizon must be positive")
    if split == "train":
        source_start, source_stop = TRAIN_INTERVAL
    elif split == "validation":
        source_start, source_stop = VALIDATION_INTERVAL
    else:
        raise ValueError(f"unsupported autoregressive split {split!r}")

    earliest = source_start
    latest_exclusive = source_stop - int(horizon)
    if current_interval is not None:
        if len(current_interval) != 2:
            raise ValueError("current interval must contain start and stop")
        requested_start, requested_stop = map(int, current_interval)
        if requested_start < earliest or requested_stop > source_stop:
            raise ValueError("current interval leaves the frozen source split")
        if requested_stop <= requested_start:
            raise ValueError("current interval must be nonempty")
        earliest = requested_start
        latest_exclusive = min(latest_exclusive, requested_stop)
    if latest_exclusive <= earliest:
        raise ValueError("requested split and horizon produce no windows")

    windows = tuple(
        AutoregressiveWindow(
            current=current,
            targets=tuple(range(current + 1, current + int(horizon) + 1)),
        )
        for current in range(earliest, latest_exclusive)
    )
    consumed = {
        frame
        for window in windows
        for frame in (window.current, *window.targets)
    }
    if any(GUARD_INTERVAL[0] <= frame < GUARD_INTERVAL[1] for frame in consumed):
        raise ValueError("an autoregressive window consumes a guard frame")
    if min(consumed) < source_start or max(consumed) >= source_stop:
        raise ValueError("an autoregressive window leaves its source split")
    return windows


class AutoregressiveStateWindowDataset:
    """Consecutive C5P targets with one shared toroidal augmentation."""

    def __init__(
        self,
        catalog: ModelDatasetCatalog,
        *,
        family: str,
        split: str,
        horizon: int,
        augment: bool,
        seed: int,
        current_interval: Sequence[int] | None = None,
    ) -> None:
        if family != "c5p":
            raise ValueError("the authorized short-unroll dataset is C5P only")
        if split != "train" and augment:
            raise ValueError("validation augmentation is prohibited")
        self.catalog = catalog
        self.family = family
        self.fields = FAMILY_FIELDS[family]
        self.split = split
        self.horizon = int(horizon)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.windows = plan_autoregressive_windows(
            split=split,
            horizon=self.horizon,
            current_interval=current_interval,
        )
        self._sources = {
            step: LeadTimeStateDataset(
                catalog,
                family=family,
                split=split,
                lead_steps=(step,),
                history_frames=1,
                augment=False,
                seed=seed,
                current_interval=current_interval,
            )
            for step in range(1, self.horizon + 1)
        }
        self._indices = {
            step: {pair.current: index for index, pair in enumerate(dataset.pairs)}
            for step, dataset in self._sources.items()
        }
        for window in self.windows:
            if any(window.current not in self._indices[step] for step in self._sources):
                raise ValueError("source datasets do not cover every common window")

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[int(index)]
        items = [
            self._sources[step][self._indices[step][window.current]]
            for step in range(1, self.horizon + 1)
        ]
        for step, item in enumerate(items, start=1):
            if int(item["current_frame_index"]) != window.current:
                raise ValueError("source current frame differs")
            if int(item["target_frame_index"]) != window.targets[step - 1]:
                raise ValueError("source target frame differs")
            if float(item["lead_steps"]) != float(step):
                raise ValueError("source lead differs")
        context = np.asarray(items[0]["context"], dtype=np.float32)
        targets = np.stack(
            [np.asarray(item["target"], dtype=np.float32) for item in items],
            axis=0,
        )
        roll = 0
        if self.augment:
            roll = toroidal_roll(
                seed=self.seed,
                epoch=self.epoch,
                frame=window.targets[-1],
            )
            context = np.ascontiguousarray(np.roll(context, roll, axis=-1))
            targets = np.ascontiguousarray(np.roll(targets, roll, axis=-1))
        return {
            "context": np.ascontiguousarray(context, dtype=np.float32),
            "targets": np.ascontiguousarray(targets, dtype=np.float32),
            "current_frame_index": np.int64(window.current),
            "target_frame_indices": np.asarray(window.targets, dtype=np.int64),
            "toroidal_roll": np.int64(roll),
        }

    def close(self) -> None:
        for dataset in self._sources.values():
            dataset.close()

    def __enter__(self) -> "AutoregressiveStateWindowDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def feedback_loss_weights(*, horizon: int, direct_one_step_weight: float) -> tuple[float, ...]:
    """Combine a retained one-step term with an equal-step rollout mean."""

    if int(horizon) <= 0:
        raise ValueError("feedback horizon must be positive")
    direct = float(direct_one_step_weight)
    if not 0.0 <= direct <= 1.0:
        raise ValueError("direct one-step weight must lie in [0,1]")
    rollout = (1.0 - direct) / int(horizon)
    weights = [rollout] * int(horizon)
    weights[0] += direct
    if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1.0e-12):
        raise AssertionError("feedback loss weights do not sum to one")
    return tuple(float(value) for value in weights)


def state_rms_normalized_mse(
    candidate: Tensor,
    target: Tensor,
    volume_derivative_rms: Tensor,
) -> tuple[Tensor, Tensor]:
    """Equal-field state MSE in fixed train-derivative RMS units."""

    if candidate.shape != target.shape or candidate.ndim != 5:
        raise ValueError("state tensors must share [batch,field,x,y,z] shape")
    scale = torch.as_tensor(
        volume_derivative_rms,
        dtype=candidate.dtype,
        device=candidate.device,
    )
    if scale.ndim != 1 or scale.numel() != candidate.shape[1]:
        raise ValueError("state RMS must contain one value per field")
    if not torch.isfinite(scale).all() or not torch.all(scale > 0):
        raise ValueError("state RMS must be finite and positive")
    error = (candidate - target) / scale.reshape(1, -1, 1, 1, 1)
    per_field = torch.mean(error.square(), dim=(0, 2, 3, 4))
    return torch.mean(per_field), per_field


class _StateForecast(Protocol):
    volume: Tensor


class ForecastOperator(Protocol):
    def forecast(self, context: Tensor, lead_steps: Tensor) -> _StateForecast: ...


def autoregressive_forecast_sequence(
    model: ForecastOperator,
    context: Tensor,
    *,
    steps: int,
) -> tuple[Tensor, ...]:
    """Forecast consecutive states without intermediate or future truth."""

    if context.ndim != 6 or context.shape[1] != 1:
        raise ValueError("short rollout context must be [batch,history=1,field,x,y,z]")
    if int(steps) <= 0:
        raise ValueError("rollout steps must be positive")
    current = context
    outputs: list[Tensor] = []
    for _ in range(int(steps)):
        lead = torch.ones(
            current.shape[0], dtype=current.dtype, device=current.device
        )
        forecast = model.forecast(current, lead)
        if forecast.volume.shape != current[:, -1].shape:
            raise ValueError("forecast state shape differs from current state")
        outputs.append(forecast.volume)
        current = forecast.volume.unsqueeze(1)
    return tuple(outputs)
