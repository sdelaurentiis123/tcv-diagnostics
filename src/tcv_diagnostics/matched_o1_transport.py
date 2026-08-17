"""Immutable native-81 truth and geometry access for matched O1 transport."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from .codec_training import sha256_path
from .codec_transport import (
    CodecTransportGeometry,
    build_codec_transport_geometry,
)
from .model_data import assert_development_path
from .transport import SingleNullTopology, toroidal_wedge_spacing


NATIVE_SHAPE = (64, 32, 81)
NATIVE_TRUTH_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")


@dataclass(frozen=True)
class NativeTruthShard:
    index: int
    start: int
    stop: int
    path: Path
    sha256: str


class NativeTruthCatalog:
    """Hash-checked access to the canonical 85604 native-81 shards."""

    def __init__(self, compact_result: Mapping[str, Any]) -> None:
        if compact_result.get("development_run") != "85604":
            raise ValueError("native truth result is not for development run 85604")
        if compact_result.get("held_out_85606_read") is not False:
            raise ValueError("native truth result does not preserve the blind lock")
        if compact_result.get("training_performed") is not False:
            raise ValueError("native truth result unexpectedly reports training")
        if (
            compact_result.get("decision", {}).get(
                "all_frame_bidirectional_closure_validated"
            )
            is not True
            or not compact_result.get("ordered_gates")
            or not all(compact_result["ordered_gates"].values())
        ):
            raise ValueError("native truth all-frame gates did not pass")
        records = compact_result.get("extraction", {}).get("canonical_shards", [])
        shards = []
        for index, record in enumerate(records):
            path = Path(record["canonical_file"])
            assert_development_path(path)
            shards.append(
                NativeTruthShard(
                    index=index,
                    start=index * 78,
                    stop=(index + 1) * 78,
                    path=path,
                    sha256=str(record["canonical_file_sha256"]),
                )
            )
        if len(shards) != 8:
            raise ValueError("native truth must contain eight canonical shards")
        covered = [frame for shard in shards for frame in range(shard.start, shard.stop)]
        if covered != list(range(624)):
            raise ValueError("native truth shards do not cover 0..623 exactly once")
        self.shards = tuple(shards)
        self._verified: set[Path] = set()

    def _verify(self, shard: NativeTruthShard) -> None:
        if shard.path in self._verified:
            return
        if not shard.path.is_file():
            raise FileNotFoundError(shard.path)
        actual = sha256_path(shard.path)
        if actual != shard.sha256:
            raise ValueError(f"native truth shard hash differs: {shard.path}")
        with h5py.File(shard.path, "r") as handle:
            frames = np.asarray(handle["frame_index"][:], dtype=np.int64)
            if not np.array_equal(frames, np.arange(shard.start, shard.stop)):
                raise ValueError("native truth frame coordinates differ")
            for field in NATIVE_TRUTH_FIELDS:
                if handle[field].shape != (78, *NATIVE_SHAPE):
                    raise ValueError(f"native truth {field} shape differs")
            if handle["saved_midpoint"].shape != (78, 2, 32):
                raise ValueError("native truth boundary shape differs")
        self._verified.add(shard.path)

    def read(
        self,
        start: int,
        stop: int,
        *,
        fields: Sequence[str] = ("Ne", "Pe", "Pi", "phi"),
    ) -> dict[str, np.ndarray]:
        if start < 0 or stop > 624 or stop <= start:
            raise ValueError("native truth interval is invalid")
        requested = tuple(str(field) for field in fields)
        if not requested or not set(requested).issubset(NATIVE_TRUTH_FIELDS):
            raise ValueError("native truth field request is invalid")
        pieces = {field: [] for field in requested}
        covered: list[int] = []
        for shard in self.shards:
            overlap_start = max(start, shard.start)
            overlap_stop = min(stop, shard.stop)
            if overlap_start >= overlap_stop:
                continue
            self._verify(shard)
            local_start = overlap_start - shard.start
            local_stop = overlap_stop - shard.start
            with h5py.File(shard.path, "r") as handle:
                for field in requested:
                    values = np.asarray(
                        handle[field][local_start:local_stop],
                        dtype=np.float64,
                    )
                    if not np.all(np.isfinite(values)):
                        raise ValueError(f"native truth {field} is non-finite")
                    pieces[field].append(values)
            covered.extend(range(overlap_start, overlap_stop))
        if covered != list(range(start, stop)):
            raise ValueError("native truth interval was not covered exactly once")
        return {
            field: np.concatenate(values, axis=0)
            for field, values in pieces.items()
        }


def _h5_array(handle: h5py.File, name: str) -> np.ndarray:
    dataset = handle[name]
    values = np.asarray(dataset[...], dtype=np.float64)
    for attribute in ("_FillValue", "missing_value"):
        if attribute in dataset.attrs:
            fill = float(np.asarray(dataset.attrs[attribute]).reshape(-1)[0])
            values = np.where(values == fill, np.nan, values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"geometry array {name} contains non-finite values")
    return values


def load_transport_geometry(
    *,
    geometry_path: Path,
    geometry_manifest: Mapping[str, Any],
) -> CodecTransportGeometry:
    """Load the previously verified native Hermes transport geometry."""

    path = Path(geometry_path)
    assert_development_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    source = geometry_manifest.get("sources", {}).get("geometry", {})
    if sha256_path(path) != source.get("sha256"):
        raise ValueError("geometry hash differs from the frozen manifest")
    grid = geometry_manifest["grid"]
    offset = int(grid["model_x_to_grid_x_offset"])
    n_x, n_y = map(int, grid["model_shape"])
    model_slice = slice(offset, offset + n_x)
    topology_record = grid["topology"]
    topology = SingleNullTopology(
        separatrix_x_index=int(topology_record["model_first_sol_x"]),
        core_lower_y=int(topology_record["core_y_inclusive"][0]),
        core_upper_y=int(topology_record["core_y_inclusive"][1]),
        pfr_lower_y=int(topology_record["inner_leg_y_inclusive"][1]),
        pfr_upper_y=int(topology_record["outer_leg_y_inclusive"][0]),
    )
    with h5py.File(path, "r") as handle:
        arrays = {
            name: _h5_array(handle, stored)[model_slice]
            for name, stored in {
                "jacobian": "J",
                "g11": "g11",
                "g23": "g23",
                "bxy": "Bxy",
                "z_shift": "zShift",
                "dy": "dy",
                "penalty_mask": "penalty_mask",
            }.items()
        }
        shift_angle = _h5_array(handle, "ShiftAngle")[model_slice]
        separatrix_radius = _h5_array(handle, "Rxy_xlow")[
            int(topology_record["ixseps1"])
        ]
    result = build_codec_transport_geometry(
        **arrays,
        shift_angle=shift_angle,
        separatrix_face_major_radius=separatrix_radius,
        dz=toroidal_wedge_spacing(int(grid["native_z_cells"]), zperiod=5),
        topology=topology,
    )
    if result.jacobian.shape != (64, 32):
        raise ValueError("transport geometry crop differs from 64x32")
    if int(np.sum(result.separatrix_face_mask)) != 16:
        raise ValueError("confined separatrix does not contain 16 face rows")
    return result
