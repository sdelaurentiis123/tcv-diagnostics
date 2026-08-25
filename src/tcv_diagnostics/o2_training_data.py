"""Leakage-safe C5P history windows for the newly frozen O2 continuation.

The completed O1 launchers hash-lock ``model_training_data.py``.  This adapter
therefore adds O2 target-to-history routing without changing that historical
dependency.  All actual frame location, shard verification, and normalization
continue to use its verified ``ModelDatasetCatalog``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

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


O2_TRAIN_TARGET_INTERVAL = (2, 432)
O2_VALIDATION_TARGET_INTERVAL = (498, 624)


def strict_o2_targets(
    targets: Iterable[int],
    *,
    split: str,
    context_frames: int,
) -> tuple[int, ...]:
    """Validate a contiguous subset of the frozen matched O2 target indices."""

    if context_frames not in (1, 2):
        raise ValueError("O2 context must contain one or two frames")
    values = tuple(int(target) for target in targets)
    if not values or values != tuple(range(values[0], values[-1] + 1)):
        raise ValueError("O2 targets must be nonempty, unique, ordered, and contiguous")
    if split == "train":
        allowed = O2_TRAIN_TARGET_INTERVAL
    elif split == "validation":
        allowed = O2_VALIDATION_TARGET_INTERVAL
    else:
        raise ValueError(f"unsupported O2 split {split!r}")
    if values[0] < allowed[0] or values[-1] >= allowed[1]:
        raise ValueError(f"O2 {split} targets leave frozen interval {allowed}")
    consumed = [
        frame
        for target in values
        for frame in range(target - context_frames, target + 1)
    ]
    if any(GUARD_INTERVAL[0] <= frame < GUARD_INTERVAL[1] for frame in consumed):
        raise ValueError("O2 window would consume a guard frame")
    source_interval = TRAIN_INTERVAL if split == "train" else VALIDATION_INTERVAL
    if min(consumed) < source_interval[0] or max(consumed) >= source_interval[1]:
        raise ValueError("O2 window leaves its frozen source split")
    return values


class OneStepWindowDataset:
    """Matched C5P history/target windows with one shared deterministic z roll."""

    def __init__(
        self,
        catalog: ModelDatasetCatalog,
        *,
        split: str,
        target_frames: Iterable[int],
        context_frames: int,
        augment: bool,
        seed: int,
        return_physical: bool = False,
    ) -> None:
        if split != "train" and augment:
            raise ValueError("validation augmentation is prohibited")
        self.catalog = catalog
        self.split = split
        self.context_frames = int(context_frames)
        self.target_frames = strict_o2_targets(
            target_frames,
            split=split,
            context_frames=self.context_frames,
        )
        self.fields = FAMILY_FIELDS["c5p"]
        self.augment = bool(augment)
        self.seed = int(seed)
        self.return_physical = bool(return_physical)
        self.epoch = 0
        self._handles: dict[Path, h5py.File] = {}
        consumed = sorted(
            {
                frame
                for target in self.target_frames
                for frame in range(target - self.context_frames, target + 1)
            }
        )
        self.consumed_frames = tuple(consumed)
        self.catalog.verify_consumed_frames(self.consumed_frames)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.target_frames)

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _frame(self, frame: int) -> tuple[np.ndarray, np.ndarray | None]:
        shard, local = self.catalog.locate(frame)
        if shard.path not in self.catalog._verified:
            raise RuntimeError("refusing to read a shard before integrity verification")
        handle = self._handle(shard.path)
        stored = int(handle["coordinates/frame_index"][local])
        if stored != frame:
            raise ValueError(f"stored frame {stored} differs from request {frame}")
        raw = [np.asarray(handle[f"fields/{field}"][local]) for field in self.fields]
        standardized = self.catalog.normalization.encode_volume(self.fields, raw)
        physical = (
            np.ascontiguousarray(np.stack(raw, axis=0), dtype=np.float32)
            if self.return_physical
            else None
        )
        return standardized, physical

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = self.target_frames[int(index)]
        context_indices = tuple(range(target - self.context_frames, target))
        context_items = [self._frame(frame) for frame in context_indices]
        target_item = self._frame(target)
        context = np.stack([frame[0] for frame in context_items], axis=0)
        target_volume = target_item[0]
        roll = 0
        if self.augment:
            roll = toroidal_roll(
                seed=self.seed,
                epoch=self.epoch,
                frame=target,
            )
            context = np.ascontiguousarray(np.roll(context, roll, axis=-1))
            target_volume = np.ascontiguousarray(
                np.roll(target_volume, roll, axis=-1)
            )
        result: dict[str, Any] = {
            "context": np.ascontiguousarray(context, dtype=np.float32),
            "target": np.ascontiguousarray(target_volume, dtype=np.float32),
            "context_frame_indices": np.asarray(context_indices, dtype=np.int64),
            "target_frame_index": np.int64(target),
            "toroidal_roll": np.int64(roll),
        }
        if self.return_physical:
            physical_context = np.stack([frame[1] for frame in context_items], axis=0)
            physical_target = target_item[1]
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
                physical_context,
                dtype=np.float32,
            )
            result["physical_target"] = np.ascontiguousarray(
                physical_target,
                dtype=np.float32,
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
