"""Leakage-safe lead-time pairs for codec-free 85604 state operators.

This module extends the already verified Paper 0 model catalog without
changing its immutable split, normalization, or shard integrity logic.  It
supports the reduced C5P control and the exact saved-state candidate E6B.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

from .model_training_data import (
    FAMILY_FIELDS,
    GUARD_INTERVAL,
    ModelDatasetCatalog,
    TRAIN_INTERVAL,
    VALIDATION_INTERVAL,
    toroidal_roll,
)


DEFAULT_LEAD_STEPS = (1, 2, 4, 8, 16)


@dataclass(frozen=True, order=True)
class LeadPair:
    """One current/target relation measured in saved-frame steps."""

    current: int
    target: int
    lead: int

    def __post_init__(self) -> None:
        if self.current < 0 or self.target <= self.current:
            raise ValueError("lead pair indices must be ordered and nonnegative")
        if self.lead != self.target - self.current:
            raise ValueError("lead pair target-current differs from lead")


def plan_lead_pairs(
    *,
    split: str,
    lead_steps: Iterable[int],
    history_frames: int = 1,
    current_interval: Sequence[int] | None = None,
) -> tuple[LeadPair, ...]:
    """Construct all requested chronological pairs inside one frozen split.

    The function never treats the returned pairs as independent samples.  An
    optional current interval allows a prospective chronological prefix to be
    selected without moving the split boundary.
    """

    if history_frames not in (1, 2):
        raise ValueError("state-operator history must contain one or two frames")
    if split == "train":
        source_start, source_stop = TRAIN_INTERVAL
    elif split == "validation":
        source_start, source_stop = VALIDATION_INTERVAL
    else:
        raise ValueError(f"unsupported state-operator split {split!r}")

    leads = tuple(sorted({int(value) for value in lead_steps}))
    if not leads or leads[0] <= 0:
        raise ValueError("lead steps must be a nonempty set of positive integers")

    earliest = source_start + history_frames - 1
    latest_exclusive = source_stop
    if current_interval is not None:
        if len(current_interval) != 2:
            raise ValueError("current interval must contain start and stop")
        requested_start, requested_stop = map(int, current_interval)
        if requested_start < earliest or requested_stop > latest_exclusive:
            raise ValueError("current interval leaves the frozen source split")
        if requested_stop <= requested_start:
            raise ValueError("current interval must be nonempty")
        earliest, latest_exclusive = requested_start, requested_stop

    pairs = tuple(
        LeadPair(current=current, target=current + lead, lead=lead)
        for current in range(earliest, latest_exclusive)
        for lead in leads
        if current + lead < source_stop
    )
    if not pairs:
        raise ValueError("requested split and leads produce no forecast pairs")

    consumed = {
        frame
        for pair in pairs
        for frame in (
            *range(pair.current - history_frames + 1, pair.current + 1),
            pair.target,
        )
    }
    if any(GUARD_INTERVAL[0] <= frame < GUARD_INTERVAL[1] for frame in consumed):
        raise ValueError("a state-operator pair consumes a guard frame")
    if min(consumed) < source_start or max(consumed) >= source_stop:
        raise ValueError("a state-operator pair leaves its source split")
    return pairs


class LeadTimeStateDataset:
    """Standardized field and boundary derivatives at specified lead times."""

    def __init__(
        self,
        catalog: ModelDatasetCatalog,
        *,
        family: str,
        split: str,
        lead_steps: Iterable[int],
        history_frames: int,
        augment: bool,
        seed: int,
        current_interval: Sequence[int] | None = None,
        return_physical: bool = False,
    ) -> None:
        if family not in FAMILY_FIELDS:
            raise ValueError(f"unsupported state family {family!r}")
        if split != "train" and augment:
            raise ValueError("validation augmentation is prohibited")
        self.catalog = catalog
        self.family = family
        self.fields = FAMILY_FIELDS[family]
        self.split = split
        self.history_frames = int(history_frames)
        self.pairs = plan_lead_pairs(
            split=split,
            lead_steps=lead_steps,
            history_frames=self.history_frames,
            current_interval=current_interval,
        )
        self.augment = bool(augment)
        self.seed = int(seed)
        self.return_physical = bool(return_physical)
        self.epoch = 0
        self._handles: dict[Path, h5py.File] = {}
        consumed = sorted(
            {
                frame
                for pair in self.pairs
                for frame in (
                    *range(
                        pair.current - self.history_frames + 1,
                        pair.current + 1,
                    ),
                    pair.target,
                )
            }
        )
        self.consumed_frames = tuple(consumed)
        self.catalog.verify_consumed_frames(self.consumed_frames)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _frame(
        self, frame: int
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        shard, local = self.catalog.locate(frame)
        if shard.path not in self.catalog._verified:
            raise RuntimeError("refusing to read a shard before integrity verification")
        handle = self._handle(shard.path)
        if int(handle["coordinates/frame_index"][local]) != frame:
            raise ValueError("stored frame differs from requested frame")
        raw = [np.asarray(handle[f"fields/{field}"][local]) for field in self.fields]
        volume = self.catalog.normalization.encode_volume(self.fields, raw)
        physical_volume = (
            np.ascontiguousarray(np.stack(raw, axis=0), dtype=np.float32)
            if self.return_physical
            else None
        )
        boundary = None
        physical_boundary = None
        if self.family == "e6b":
            raw_boundary = np.asarray(handle["boundary/Bphi"][local])
            boundary = self.catalog.normalization.encode_boundary(raw_boundary)
            if self.return_physical:
                physical_boundary = np.ascontiguousarray(
                    raw_boundary, dtype=np.float32
                )
        return volume, boundary, physical_volume, physical_boundary

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[int(index)]
        context_indices = tuple(
            range(pair.current - self.history_frames + 1, pair.current + 1)
        )
        context_items = [self._frame(frame) for frame in context_indices]
        target_item = self._frame(pair.target)
        context = np.stack([item[0] for item in context_items], axis=0)
        target = target_item[0]
        derivative = (target - context[-1]) / np.float32(pair.lead)

        roll = 0
        if self.augment:
            roll = toroidal_roll(
                seed=self.seed,
                epoch=self.epoch,
                frame=pair.target,
            )
            context = np.ascontiguousarray(np.roll(context, roll, axis=-1))
            target = np.ascontiguousarray(np.roll(target, roll, axis=-1))
            derivative = np.ascontiguousarray(
                np.roll(derivative, roll, axis=-1)
            )

        result: dict[str, Any] = {
            "context": np.ascontiguousarray(context, dtype=np.float32),
            "target": np.ascontiguousarray(target, dtype=np.float32),
            "target_derivative": np.ascontiguousarray(
                derivative, dtype=np.float32
            ),
            "context_frame_indices": np.asarray(context_indices, dtype=np.int64),
            "current_frame_index": np.int64(pair.current),
            "target_frame_index": np.int64(pair.target),
            "lead_steps": np.float32(pair.lead),
            "toroidal_roll": np.int64(roll),
        }

        if self.family == "e6b":
            context_boundary = np.stack(
                [item[1] for item in context_items], axis=0
            )
            target_boundary = target_item[1]
            if target_boundary is None:
                raise AssertionError("E6B target boundary was not loaded")
            boundary_derivative = (
                target_boundary - context_boundary[-1]
            ) / np.float32(pair.lead)
            result.update(
                {
                    "context_boundary": np.ascontiguousarray(
                        context_boundary, dtype=np.float32
                    ),
                    "target_boundary": np.ascontiguousarray(
                        target_boundary, dtype=np.float32
                    ),
                    "target_boundary_derivative": np.ascontiguousarray(
                        boundary_derivative, dtype=np.float32
                    ),
                }
            )

        if self.return_physical:
            physical_context = np.stack(
                [item[2] for item in context_items], axis=0
            )
            physical_target = target_item[2]
            if physical_target is None:
                raise AssertionError("physical target was not loaded")
            if self.augment:
                physical_context = np.ascontiguousarray(
                    np.roll(physical_context, roll, axis=-1)
                )
                physical_target = np.ascontiguousarray(
                    np.roll(physical_target, roll, axis=-1)
                )
            result["physical_context"] = np.ascontiguousarray(
                physical_context, dtype=np.float32
            )
            result["physical_target"] = np.ascontiguousarray(
                physical_target, dtype=np.float32
            )
            if self.family == "e6b":
                result["physical_context_boundary"] = np.ascontiguousarray(
                    np.stack([item[3] for item in context_items], axis=0),
                    dtype=np.float32,
                )
                result["physical_target_boundary"] = np.ascontiguousarray(
                    target_item[3], dtype=np.float32
                )
        return result

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        if hasattr(self, "_handles"):
            self.close()

