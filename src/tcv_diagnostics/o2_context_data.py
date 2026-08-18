"""Truth-free validation contexts for frozen Paper 0 O2 forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog
from .o2_training_data import strict_o2_targets


class OneStepContextDataset:
    """Validation forecast contexts that never read the target field."""

    def __init__(
        self,
        catalog: ModelDatasetCatalog,
        *,
        target_frames: Iterable[int],
        context_frames: int,
        return_physical: bool = False,
    ) -> None:
        self.catalog = catalog
        self.split = "validation"
        self.context_frames = int(context_frames)
        self.target_frames = strict_o2_targets(
            target_frames,
            split="validation",
            context_frames=self.context_frames,
        )
        self.fields = FAMILY_FIELDS["c5p"]
        self.return_physical = bool(return_physical)
        self.target_truth_read = False
        self._handles: dict[Path, h5py.File] = {}
        self.consumed_frames = tuple(
            sorted(
                {
                    frame
                    for target in self.target_frames
                    for frame in range(target - self.context_frames, target)
                }
            )
        )
        self.catalog.verify_consumed_frames(self.consumed_frames)

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
        if target in context_indices:
            raise RuntimeError("forecast context unexpectedly contains target truth")
        context_items = [self._frame(frame) for frame in context_indices]
        result: dict[str, Any] = {
            "context": np.ascontiguousarray(
                np.stack([item[0] for item in context_items], axis=0),
                dtype=np.float32,
            ),
            "context_frame_indices": np.asarray(context_indices, dtype=np.int64),
            "target_frame_index": np.int64(target),
            "target_truth_read": False,
        }
        if self.return_physical:
            physical = [item[1] for item in context_items]
            if any(item is None for item in physical):
                raise AssertionError("physical forecast context was not loaded")
            result["physical_context"] = np.ascontiguousarray(
                np.stack(physical, axis=0), dtype=np.float32
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
