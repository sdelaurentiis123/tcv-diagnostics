"""Verified frame access for the frozen Paper 0 model experiments.

Only this module maps model-training indices to the immutable 85604 HDF5
shards.  It keeps integrity checks, split rules, preprocessing, augmentation,
and HDF5 lifetime outside the model and trainer implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from .model_data import (
    assert_development_path,
    load_strict_json,
    sha256_file,
)


OFFICIAL_DATASET_RESULT_SHA256 = (
    "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
)
OFFICIAL_NORMALIZATION_SHA256 = (
    "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
)
OFFICIAL_ARTIFACT_INDEX_SHA256 = (
    "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01"
)
OFFICIAL_ARTIFACT_ROOT = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/"
    "phase2_model_dataset/job_6893525"
)

VOLUME_SHAPE = (64, 32, 88)
TRAIN_INTERVAL = (0, 432)
GUARD_INTERVAL = (432, 496)
VALIDATION_INTERVAL = (496, 624)
FAMILY_FIELDS = {
    "c5p": ("Ne", "Pe", "Pi", "phi", "Vi"),
    "e6b": ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort"),
}


def _strict_contiguous_frames(
    frames: Iterable[int],
    *,
    split: str,
) -> tuple[int, ...]:
    values = tuple(int(frame) for frame in frames)
    if not values:
        raise ValueError("a model dataset cannot be empty")
    if values != tuple(range(values[0], values[-1] + 1)):
        raise ValueError("model frames must be unique, ordered, and contiguous")
    if split == "train":
        allowed = TRAIN_INTERVAL
    elif split == "validation":
        allowed = VALIDATION_INTERVAL
    else:
        raise ValueError(f"unsupported split {split!r}")
    if values[0] < allowed[0] or values[-1] >= allowed[1]:
        raise ValueError(
            f"{split} frames {values[0]}..{values[-1]} leave frozen {allowed}"
        )
    if any(GUARD_INTERVAL[0] <= frame < GUARD_INTERVAL[1] for frame in values):
        raise ValueError("guard frames are prohibited")
    return values


def epoch_order(frames: Sequence[int], *, seed: int, epoch: int) -> tuple[int, ...]:
    """Return one deterministic permutation containing every frame once."""

    values = np.asarray(tuple(int(item) for item in frames), dtype=np.int64)
    if values.size == 0 or np.unique(values).size != values.size:
        raise ValueError("epoch frames must be nonempty and unique")
    generator = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(epoch), 0x50415030])
    )
    return tuple(int(item) for item in generator.permutation(values))


def toroidal_roll(*, seed: int, epoch: int, frame: int, size: int = 88) -> int:
    """Stateless deterministic training roll for one epoch/frame pair."""

    if size <= 0:
        raise ValueError("roll size must be positive")
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(epoch), int(frame), 0x5A504552]
        )
    )
    return int(generator.integers(0, size))


@dataclass(frozen=True)
class FieldNormalization:
    mean: float
    standard_deviation: float
    transform: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("normalization mean must be finite")
        if not np.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("normalization standard deviation must be finite/positive")
        if self.transform.get("name") not in {"identity", "log_offset"}:
            raise ValueError(f"unsupported transform {self.transform}")

    def encode(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if self.transform["name"] == "log_offset":
            offset = float(self.transform["offset"])
            argument = array + offset
            if not np.all(argument > 0):
                raise ValueError("log-offset normalization received nonpositive data")
            array = np.log(argument)
        result = (array - self.mean) / self.standard_deviation
        if not np.all(np.isfinite(result)):
            raise ValueError("normalization produced non-finite values")
        return np.asarray(result, dtype=np.float32)

    def decode(self, values: np.ndarray) -> np.ndarray:
        array = (
            np.asarray(values, dtype=np.float64) * self.standard_deviation
            + self.mean
        )
        if self.transform["name"] == "log_offset":
            array = np.exp(array) - float(self.transform["offset"])
        if not np.all(np.isfinite(array)):
            raise ValueError("inverse normalization produced non-finite values")
        return array


class ModelNormalization:
    """Frozen training-only scalar normalization for volumes and Bphi sides."""

    def __init__(self, record: Mapping[str, Any]) -> None:
        if record.get("development_run") != "85604":
            raise ValueError("normalization record is not for development run 85604")
        if record.get("held_out_85606_read") is not False:
            raise ValueError("normalization record does not preserve held-out status")
        if tuple(record.get("fit_frames", ())) != TRAIN_INTERVAL:
            raise ValueError("normalization was not fit on frozen training frames")
        records = record.get("records")
        if not isinstance(records, Mapping):
            raise ValueError("normalization records are missing")
        self.records = {
            name: FieldNormalization(
                mean=float(value["mean"]),
                standard_deviation=float(
                    value["population_standard_deviation"]
                ),
                transform=dict(value["transform"]),
            )
            for name, value in records.items()
        }
        required = set().union(*map(set, FAMILY_FIELDS.values())) | {
            "Bphi/inner",
            "Bphi/outer",
        }
        missing = required - set(self.records)
        if missing:
            raise ValueError(f"normalization is missing {sorted(missing)}")

    def encode_volume(
        self,
        fields: Sequence[str],
        values: Sequence[np.ndarray],
    ) -> np.ndarray:
        if len(fields) != len(values):
            raise ValueError("field/value normalization lengths differ")
        encoded = [self.records[field].encode(value) for field, value in zip(fields, values)]
        result = np.stack(encoded, axis=0)
        if result.shape != (len(fields), *VOLUME_SHAPE):
            raise ValueError(f"normalized volume has unexpected shape {result.shape}")
        return np.ascontiguousarray(result, dtype=np.float32)

    def decode_volume(
        self,
        fields: Sequence[str],
        values: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != (len(fields), *VOLUME_SHAPE):
            raise ValueError(f"standardized volume has unexpected shape {array.shape}")
        return np.stack(
            [self.records[field].decode(array[index]) for index, field in enumerate(fields)],
            axis=0,
        )

    def encode_boundary(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != (2, 32):
            raise ValueError(f"Bphi has unexpected shape {array.shape}")
        result = np.stack(
            [
                self.records["Bphi/inner"].encode(array[0]),
                self.records["Bphi/outer"].encode(array[1]),
            ],
            axis=0,
        )
        return np.ascontiguousarray(result, dtype=np.float32)

    def decode_boundary(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        if array.shape != (2, 32):
            raise ValueError(f"standardized Bphi has unexpected shape {array.shape}")
        return np.stack(
            [
                self.records["Bphi/inner"].decode(array[0]),
                self.records["Bphi/outer"].decode(array[1]),
            ],
            axis=0,
        )


@dataclass(frozen=True)
class Shard:
    index: int
    start: int
    stop: int
    path: Path
    sha256: str


class ModelDatasetCatalog:
    """Integrity-checked description of the immutable model shards."""

    def __init__(
        self,
        dataset_record: Mapping[str, Any],
        normalization_record: Mapping[str, Any],
        *,
        artifact_root: Path,
    ) -> None:
        assert_development_path(artifact_root)
        if dataset_record.get("development_run") != "85604":
            raise ValueError("model dataset is not development run 85604")
        if dataset_record.get("held_out_85606_read") is not False:
            raise ValueError("model dataset does not preserve held-out status")
        if dataset_record.get("training_performed") is not False:
            raise ValueError("model dataset artifact unexpectedly reports training")
        if not dataset_record.get("gates", {}).get("all_passed"):
            raise ValueError("model dataset engineering gates did not pass")
        dataset = dataset_record.get("dataset", {})
        if tuple(dataset.get("fields", ())) != (
            "Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi", "Vi"
        ):
            raise ValueError("model dataset field inventory differs")
        if int(dataset.get("frame_count", -1)) != VALIDATION_INTERVAL[1]:
            raise ValueError("model dataset frame count differs")

        shards: list[Shard] = []
        for item in dataset.get("shards", ()):
            source_path = Path(item["path"])
            assert_development_path(source_path)
            path = artifact_root / "shards" / source_path.name
            assert_development_path(path)
            shards.append(
                Shard(
                    index=int(item["index"]),
                    start=int(item["global_start_inclusive"]),
                    stop=int(item["global_stop_exclusive"]),
                    path=path,
                    sha256=str(item["sha256"]),
                )
            )
        if [shard.index for shard in shards] != list(range(8)):
            raise ValueError("model shard indices are incomplete or unordered")
        covered = [frame for shard in shards for frame in range(shard.start, shard.stop)]
        if covered != list(range(VALIDATION_INTERVAL[1])):
            raise ValueError("model shards do not cover frames 0..623 exactly once")
        self.artifact_root = artifact_root
        self.shards = tuple(shards)
        self.normalization = ModelNormalization(normalization_record)
        self._verified: set[Path] = set()

    def locate(self, frame: int) -> tuple[Shard, int]:
        index = int(frame)
        for shard in self.shards:
            if shard.start <= index < shard.stop:
                return shard, index - shard.start
        raise IndexError(f"frame {index} is outside the model dataset")

    def verify_consumed_frames(self, frames: Sequence[int]) -> tuple[Path, ...]:
        required = {self.locate(int(frame))[0] for frame in frames}
        verified: list[Path] = []
        for shard in sorted(required, key=lambda item: item.index):
            if shard.path not in self._verified:
                if not shard.path.is_file():
                    raise FileNotFoundError(shard.path)
                actual = sha256_file(shard.path)
                if actual != shard.sha256:
                    raise ValueError(
                        f"shard {shard.index} SHA-256 mismatch: {actual} != {shard.sha256}"
                    )
                self._verify_shard_schema(shard)
                self._verified.add(shard.path)
            verified.append(shard.path)
        return tuple(verified)

    @staticmethod
    def _verify_shard_schema(shard: Shard) -> None:
        with h5py.File(shard.path, "r") as handle:
            if str(handle.attrs["development_run"]) != "85604":
                raise ValueError("shard development-run attribute differs")
            if bool(handle.attrs["held_out_85606_read"]):
                raise ValueError("shard held-out-read attribute is true")
            if int(handle.attrs["zperiod"]) != 5:
                raise ValueError("shard zperiod is not 5")
            if int(handle.attrs["global_start_inclusive"]) != shard.start:
                raise ValueError("shard start attribute differs")
            if int(handle.attrs["global_stop_exclusive"]) != shard.stop:
                raise ValueError("shard stop attribute differs")
            count = shard.stop - shard.start
            for field in set().union(*map(set, FAMILY_FIELDS.values())):
                dataset = handle[f"fields/{field}"]
                if dataset.shape != (count, *VOLUME_SHAPE) or dataset.dtype != np.dtype("f4"):
                    raise ValueError(f"shard {field} schema differs")
            boundary = handle["boundary/Bphi"]
            if boundary.shape != (count, 2, 32) or boundary.dtype != np.dtype("f4"):
                raise ValueError("shard Bphi schema differs")
            expected_frames = np.arange(shard.start, shard.stop, dtype=np.int64)
            if not np.array_equal(handle["coordinates/frame_index"][:], expected_frames):
                raise ValueError("shard frame coordinate differs")


def load_official_catalog(
    artifact_root: Path = OFFICIAL_ARTIFACT_ROOT,
) -> ModelDatasetCatalog:
    """Load and byte-verify the official 85604 dataset records and index."""

    root = Path(artifact_root)
    assert_development_path(root)
    dataset_path = root / "model_dataset_manifest.json"
    normalization_path = root / "normalization.json"
    artifact_index_path = root / "artifact_sha256.txt"
    expected = {
        dataset_path: OFFICIAL_DATASET_RESULT_SHA256,
        normalization_path: OFFICIAL_NORMALIZATION_SHA256,
        artifact_index_path: OFFICIAL_ARTIFACT_INDEX_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"official artifact SHA-256 mismatch: {path}: {actual}")
    return ModelDatasetCatalog(
        load_strict_json(dataset_path),
        load_strict_json(normalization_path),
        artifact_root=root,
    )


class CodecFrameDataset:
    """One-frame standardized codec examples with deterministic z rolls."""

    def __init__(
        self,
        catalog: ModelDatasetCatalog,
        *,
        family: str,
        split: str,
        frames: Iterable[int],
        augment: bool,
        seed: int,
    ) -> None:
        if family not in FAMILY_FIELDS:
            raise ValueError(f"unsupported state family {family!r}")
        if split != "train" and augment:
            raise ValueError("validation augmentation is prohibited")
        self.catalog = catalog
        self.family = family
        self.fields = FAMILY_FIELDS[family]
        self.split = split
        self.frames = _strict_contiguous_frames(frames, split=split)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self._handles: dict[Path, h5py.File] = {}
        self.catalog.verify_consumed_frames(self.frames)

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frames)

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame = self.frames[int(index)]
        shard, local = self.catalog.locate(frame)
        if shard.path not in self.catalog._verified:
            raise RuntimeError("refusing to read a shard before integrity verification")
        handle = self._handle(shard.path)
        stored_frame = int(handle["coordinates/frame_index"][local])
        if stored_frame != frame:
            raise ValueError(f"stored frame {stored_frame} differs from request {frame}")
        raw = [np.asarray(handle[f"fields/{field}"][local]) for field in self.fields]
        volume = self.catalog.normalization.encode_volume(self.fields, raw)
        roll = 0
        if self.augment:
            roll = toroidal_roll(seed=self.seed, epoch=self.epoch, frame=frame)
            volume = np.ascontiguousarray(np.roll(volume, roll, axis=-1))
        item: dict[str, Any] = {
            "volume": volume,
            "frame_index": np.int64(frame),
            "toroidal_roll": np.int64(roll),
        }
        if self.family == "e6b":
            boundary = np.asarray(handle["boundary/Bphi"][local])
            item["boundary"] = self.catalog.normalization.encode_boundary(boundary)
        return item

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
