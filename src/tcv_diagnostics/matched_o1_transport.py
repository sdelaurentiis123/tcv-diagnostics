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
MODEL_SHAPE = (64, 32, 88)
NATIVE_TRUTH_FIELDS = ("Ne", "Pe", "Pi", "Vort", "phi")
CANDIDATE_NATIVE_FIELDS = {
    "c5p": ("Ne", "Pe", "Pi", "phi"),
    "e6b": ("Ne", "Pe", "Pi", "Vort"),
}
E6B_COMMON_COMPONENTS = ("Ne", "Pe", "Pi", "NVi")


def _text_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class MatchedCandidateArtifact:
    """Hash-checked access to one codec reconstruction split artifact."""

    def __init__(
        self,
        path: Path,
        *,
        sha256: str,
        family: str,
        codec: str,
        seed: int,
        checkpoint_sha256: str,
        frames: Sequence[int],
    ) -> None:
        if family not in CANDIDATE_NATIVE_FIELDS:
            raise ValueError(f"unsupported matched candidate family {family!r}")
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != str(sha256):
            raise ValueError("matched candidate artifact hash differs")
        self.family = family
        self.codec = codec
        self.seed = int(seed)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.frames = tuple(int(frame) for frame in frames)
        if not self.frames or self.frames != tuple(
            range(self.frames[0], self.frames[-1] + 1)
        ):
            raise ValueError("matched candidate frames must be contiguous")
        self._verify_schema()

    def _verify_schema(self) -> None:
        count = len(self.frames)
        with h5py.File(self.path, "r") as handle:
            expected_attributes = {
                "schema_version": 1,
                "development_run": "85604",
                "held_out_85606_read": False,
                "family": self.family,
                "codec": self.codec,
                "seed": self.seed,
                "checkpoint_sha256": self.checkpoint_sha256,
                "zperiod": 5,
            }
            for name, expected in expected_attributes.items():
                actual = handle.attrs.get(name)
                if isinstance(expected, str):
                    actual = _text_attribute(actual)
                elif isinstance(expected, bool):
                    actual = bool(actual)
                else:
                    actual = int(actual)
                if actual != expected:
                    raise ValueError(f"matched candidate attribute {name} differs")
            if tuple(int(value) for value in handle.attrs["native_shape"]) != NATIVE_SHAPE:
                raise ValueError("matched candidate native shape differs")
            stored_frames = np.asarray(
                handle["coordinates/frame_index"][:], dtype=np.int64
            )
            if not np.array_equal(stored_frames, self.frames):
                raise ValueError("matched candidate frame coordinates differ")
            candidate = handle["candidate"]
            required_native = CANDIDATE_NATIVE_FIELDS[self.family]
            if set(candidate.keys()) != set(required_native):
                raise ValueError("matched candidate native fields differ")
            for field in required_native:
                dataset = candidate[field]
                if dataset.shape != (count, *NATIVE_SHAPE) or dataset.dtype != np.dtype(
                    "f4"
                ):
                    raise ValueError(f"matched candidate {field} schema differs")
            if self.family == "c5p":
                if "boundary" in handle or "model88" in handle:
                    raise ValueError("C5P candidate contains forbidden E6B side state")
            else:
                boundary = handle["boundary/Bphi"]
                if boundary.shape != (count, 2, 32) or boundary.dtype != np.dtype("f4"):
                    raise ValueError("matched E6B boundary schema differs")
                if _text_attribute(boundary.attrs.get("policy")) != (
                    "exact_bypass_from_model_dataset"
                ):
                    raise ValueError("matched E6B boundary policy differs")
                model = handle["model88"]
                if set(model.keys()) != set(E6B_COMMON_COMPONENTS):
                    raise ValueError("matched E6B common components differ")
                for field in E6B_COMMON_COMPONENTS:
                    dataset = model[field]
                    if dataset.shape != (count, *MODEL_SHAPE) or dataset.dtype != np.dtype(
                        "f4"
                    ):
                        raise ValueError(f"matched E6B model88 {field} schema differs")

    def _indices(self, start: int, stop: int) -> tuple[int, int]:
        if start < self.frames[0] or stop > self.frames[-1] + 1 or stop <= start:
            raise ValueError("matched candidate read leaves its frame interval")
        return start - self.frames[0], stop - self.frames[0]

    def read_native(self, start: int, stop: int) -> dict[str, np.ndarray]:
        local_start, local_stop = self._indices(start, stop)
        with h5py.File(self.path, "r") as handle:
            return {
                field: np.asarray(
                    handle[f"candidate/{field}"][local_start:local_stop],
                    dtype=np.float64,
                )
                for field in CANDIDATE_NATIVE_FIELDS[self.family]
            }

    def read_model88(self, start: int, stop: int) -> dict[str, np.ndarray]:
        if self.family != "e6b":
            raise ValueError("only E6B candidates contain model88 components")
        local_start, local_stop = self._indices(start, stop)
        with h5py.File(self.path, "r") as handle:
            return {
                field: np.asarray(
                    handle[f"model88/{field}"][local_start:local_stop],
                    dtype=np.float64,
                )
                for field in E6B_COMMON_COMPONENTS
            }

    def read_boundary(self, start: int, stop: int) -> np.ndarray:
        if self.family != "e6b":
            raise ValueError("only E6B candidates contain a boundary bypass")
        local_start, local_stop = self._indices(start, stop)
        with h5py.File(self.path, "r") as handle:
            return np.asarray(
                handle["boundary/Bphi"][local_start:local_stop],
                dtype=np.float32,
            )


class MatchedPhiArtifact:
    """Hash-checked exact E6B potential reconstructed on the native grid."""

    def __init__(
        self,
        path: Path,
        *,
        sha256: str,
        source_candidate_sha256: str,
        frames: Sequence[int],
    ) -> None:
        self.path = Path(path)
        assert_development_path(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_path(self.path)
        if self.sha256 != str(sha256):
            raise ValueError("matched phi artifact hash differs")
        self.source_candidate_sha256 = str(source_candidate_sha256)
        self.frames = tuple(int(frame) for frame in frames)
        if not self.frames or self.frames != tuple(
            range(self.frames[0], self.frames[-1] + 1)
        ):
            raise ValueError("matched phi frames must be contiguous")
        with h5py.File(self.path, "r") as handle:
            if (
                int(handle.attrs.get("schema_version", -1)) != 1
                or _text_attribute(handle.attrs.get("development_run")) != "85604"
                or bool(handle.attrs.get("held_out_85606_read"))
                or int(handle.attrs.get("zperiod", -1)) != 5
                or bool(handle.attrs.get("truth_layout"))
                or _text_attribute(handle.attrs.get("source_input_sha256"))
                != self.source_candidate_sha256
            ):
                raise ValueError("matched phi artifact attributes differ")
            stored_frames = np.asarray(handle["frame_index"][:], dtype=np.int64)
            if not np.array_equal(stored_frames, self.frames):
                raise ValueError("matched phi frame coordinates differ")
            dataset = handle["phi"]
            if dataset.shape != (len(self.frames), *NATIVE_SHAPE) or dataset.dtype != np.dtype(
                "f8"
            ):
                raise ValueError("matched phi dataset schema differs")

    def read(self, start: int, stop: int) -> np.ndarray:
        if start < self.frames[0] or stop > self.frames[-1] + 1 or stop <= start:
            raise ValueError("matched phi read leaves its frame interval")
        with h5py.File(self.path, "r") as handle:
            values = np.asarray(
                handle["phi"][
                    start - self.frames[0] : stop - self.frames[0]
                ],
                dtype=np.float64,
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("matched phi contains non-finite values")
        return values


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
