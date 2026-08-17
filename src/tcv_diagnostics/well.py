"""Read-only virtual trajectory over the two audited 85604 Well shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import numpy as np

from .data_protocol import C5_FIELDS, require_allowed_file


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class WellShard:
    path: Path
    global_start: int
    frames: int
    spatial_shape: tuple[int, int, int]
    fields: tuple[str, ...]

    @property
    def global_stop(self) -> int:
        return self.global_start + self.frames


class VirtualWellTrajectory:
    """Concatenate storage shards without accepting their legacy split semantics."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        required_fields: Sequence[str] = C5_FIELDS,
    ) -> None:
        if len(shard_paths) != 2:
            raise ValueError("Phase 1 expects exactly two audited 85604 storage shards")
        self.required_fields = tuple(required_fields)
        self.shards: list[WellShard] = []
        global_start = 0
        reference_spatial_shape: tuple[int, int, int] | None = None
        reference_axes: dict[str, np.ndarray] = {}
        time_parts: list[np.ndarray] = []

        for path_text in shard_paths:
            path = require_allowed_file(path_text)
            with h5py.File(path, "r") as handle:
                if "t0_fields" not in handle or "dimensions/time" not in handle:
                    raise ValueError(f"invalid Well shard schema: {path}")
                raw_names = handle["t0_fields"].attrs.get("field_names")
                if raw_names is None:
                    fields = tuple(sorted(handle["t0_fields"].keys()))
                else:
                    fields = tuple(_decode(item) for item in np.asarray(raw_names).flat)
                missing = [field for field in self.required_fields if field not in fields]
                if missing:
                    raise ValueError(f"{path} is missing required fields {missing}")

                shapes = {
                    tuple(int(size) for size in handle[f"t0_fields/{field}"].shape)
                    for field in self.required_fields
                }
                if len(shapes) != 1:
                    raise ValueError(f"field shapes disagree in {path}: {sorted(shapes)}")
                shape = next(iter(shapes))
                if len(shape) != 5 or shape[0] != 1:
                    raise ValueError(
                        f"expected [1,time,x,y,z] fields in {path}, got {shape}"
                    )
                frames = shape[1]
                spatial_shape = shape[2:]
                if reference_spatial_shape is None:
                    reference_spatial_shape = spatial_shape
                elif spatial_shape != reference_spatial_shape:
                    raise ValueError("Well shard spatial shapes disagree")

                time = np.asarray(handle["dimensions/time"][...], dtype=np.float64)
                if time.shape != (frames,):
                    raise ValueError(f"time shape {time.shape} disagrees with {frames} frames")
                time_parts.append(time)
                for axis in ("x", "y", "z"):
                    values = np.asarray(handle[f"dimensions/{axis}"][...])
                    if axis not in reference_axes:
                        reference_axes[axis] = values
                    elif not np.array_equal(values, reference_axes[axis]):
                        raise ValueError(f"Well shard {axis} coordinates disagree")

            self.shards.append(
                WellShard(path, global_start, frames, spatial_shape, fields)
            )
            global_start += frames

        self.time = np.concatenate(time_parts)
        self.axes = reference_axes
        self.spatial_shape = reference_spatial_shape
        self.total_frames = global_start
        differences = np.diff(self.time)
        if differences.size == 0 or not np.all(differences > 0):
            raise ValueError("global Well time must be strictly increasing")
        if not np.allclose(differences, differences[0], rtol=0.0, atol=1e-10):
            raise ValueError("global Well time must have uniform cadence")

    def _segments(self, start: int, stop: int) -> Iterator[tuple[WellShard, int, int]]:
        if start < 0 or stop <= start or stop > self.total_frames:
            raise ValueError(
                f"invalid global slice [{start}, {stop}) for {self.total_frames} frames"
            )
        for shard in self.shards:
            overlap_start = max(start, shard.global_start)
            overlap_stop = min(stop, shard.global_stop)
            if overlap_start < overlap_stop:
                yield (
                    shard,
                    overlap_start - shard.global_start,
                    overlap_stop - shard.global_start,
                )

    def iter_field_chunks(
        self,
        field: str,
        start: int,
        stop: int,
        *,
        chunk_frames: int = 8,
        strides: tuple[int, int, int] = (1, 1, 1),
    ) -> Iterator[np.ndarray]:
        if field not in self.required_fields:
            raise KeyError(f"field {field} is not in required field set")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if len(strides) != 3 or any(stride <= 0 for stride in strides):
            raise ValueError("spatial strides must be three positive integers")
        spatial_slices = tuple(slice(None, None, stride) for stride in strides)
        for shard, local_start, local_stop in self._segments(start, stop):
            with h5py.File(shard.path, "r") as handle:
                dataset = handle[f"t0_fields/{field}"]
                cursor = local_start
                while cursor < local_stop:
                    chunk_stop = min(cursor + chunk_frames, local_stop)
                    selection = (0, slice(cursor, chunk_stop), *spatial_slices)
                    yield np.asarray(dataset[selection])
                    cursor = chunk_stop

    def read_field(
        self,
        field: str,
        start: int,
        stop: int,
        *,
        chunk_frames: int = 8,
        strides: tuple[int, int, int] = (1, 1, 1),
    ) -> np.ndarray:
        chunks = list(
            self.iter_field_chunks(
                field,
                start,
                stop,
                chunk_frames=chunk_frames,
                strides=strides,
            )
        )
        if not chunks:
            raise ValueError("requested slice produced no data")
        return np.concatenate(chunks, axis=0)

