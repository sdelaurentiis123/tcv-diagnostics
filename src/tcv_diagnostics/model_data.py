"""Auditable primitives for the shared 85604 model dataset.

This module contains no model code and performs no held-out access. It keeps
array hashing, mergeable training-only moments, transforms, interval routing,
and strict artifact writes independently testable before cluster execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VOLUME_FIELDS = ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort", "phi", "Vi")
STATE_VIEWS = {
    "E6B-H1": ("Ne", "Pe", "Pi", "NVe", "NVi", "Vort"),
    "C5P-H2": ("Ne", "Pe", "Pi", "phi", "Vi"),
    "C5P-H1": ("Ne", "Pe", "Pi", "phi", "Vi"),
}


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting NaN and infinity spellings."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path, *, block_bytes: int = 1024 * 1024) -> str:
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash shape, canonical little-endian dtype, and contiguous array bytes."""

    array = np.asarray(values)
    if array.dtype.hasobject or np.iscomplexobj(array):
        raise TypeError("array hashing requires a non-object real array")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array, dtype=canonical_dtype)
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def canonical_float64_sha256(values: np.ndarray) -> str:
    """Reproduce the canonical extraction-record float64 array hash."""

    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = json.dumps(
        {"dtype": "<f8", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def assert_development_path(path: Path) -> None:
    if "85606" in str(path).lower():
        raise ValueError(f"held-out 85606 path is prohibited: {path}")


def source_segments(
    sources: Sequence[Mapping[str, Any]],
    start: int,
    stop: int,
) -> list[tuple[Mapping[str, Any], int, int, int, int]]:
    """Map one global half-open interval to source-local intervals."""

    if start < 0 or stop <= start:
        raise ValueError(f"invalid global interval [{start},{stop})")
    segments: list[tuple[Mapping[str, Any], int, int, int, int]] = []
    covered: list[int] = []
    for source in sources:
        source_start = int(source["global_start_inclusive"])
        source_stop = int(source["global_stop_exclusive"])
        if source_stop <= source_start:
            raise ValueError("source interval is empty or reversed")
        overlap_start = max(start, source_start)
        overlap_stop = min(stop, source_stop)
        if overlap_start < overlap_stop:
            segments.append(
                (
                    source,
                    overlap_start - source_start,
                    overlap_stop - source_start,
                    overlap_start,
                    overlap_stop,
                )
            )
            covered.extend(range(overlap_start, overlap_stop))
    if covered != list(range(start, stop)):
        raise ValueError(
            f"sources do not cover global interval [{start},{stop}) exactly once"
        )
    return segments


def validate_intervals(
    intervals: Iterable[Sequence[int]],
    *,
    expected_start: int,
    expected_stop: int,
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(item[0]), int(item[1])) for item in intervals)
    if any(stop <= start for start, stop in normalized):
        raise ValueError("all intervals must be nonempty and increasing")
    covered = [
        index for start, stop in normalized for index in range(start, stop)
    ]
    if covered != list(range(expected_start, expected_stop)):
        raise ValueError("intervals do not provide exact ordered coverage")
    return normalized


def finite_real_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.hasobject or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real numeric array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def apply_moment_transform(
    field: str,
    values: np.ndarray,
    transforms: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    """Apply the frozen pre-normalization transform in float64."""

    array = np.asarray(finite_real_array(field, values), dtype=np.float64)
    if field not in transforms:
        raise KeyError(f"no normalization transform is frozen for {field}")
    specification = transforms[field]
    name = specification["name"]
    if name == "identity":
        return array
    if name == "log_offset":
        offset = float(specification["offset"])
        argument = array + offset
        if not np.all(argument > 0.0):
            raise ValueError(f"{field} log transform has a nonpositive argument")
        transformed = np.log(argument)
        if not np.all(np.isfinite(transformed)):
            raise ValueError(f"{field} log transform produced non-finite values")
        return transformed
    raise ValueError(f"unsupported normalization transform {name!r} for {field}")


@dataclass
class StreamingMoments:
    """Mergeable population moments using the parallel variance recurrence."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(finite_real_array("moment values", values), dtype=np.float64)
        if array.size == 0:
            raise ValueError("cannot update moments with an empty array")
        batch_count = int(array.size)
        batch_mean = float(np.mean(array, dtype=np.float64))
        centered = array - batch_mean
        batch_m2 = float(np.sum(centered * centered, dtype=np.float64))
        self.merge(StreamingMoments(batch_count, batch_mean, batch_m2))

    def merge(self, other: "StreamingMoments") -> None:
        if other.count < 0 or self.count < 0:
            raise ValueError("moment counts cannot be negative")
        if not all(
            math.isfinite(value)
            for value in (self.mean, self.m2, other.mean, other.m2)
        ):
            raise ValueError("moments must be finite")
        if self.m2 < 0.0 or other.m2 < 0.0:
            raise ValueError("M2 cannot be negative")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = int(other.count)
            self.mean = float(other.mean)
            self.m2 = float(other.m2)
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    def finalize(self) -> dict[str, int | float]:
        if self.count <= 0:
            raise ValueError("cannot finalize empty moments")
        if self.m2 < 0.0 or not math.isfinite(self.m2):
            raise ValueError("invalid M2")
        variance = self.m2 / self.count
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "M2": float(self.m2),
            "population_variance": float(variance),
            "population_standard_deviation": float(math.sqrt(variance)),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StreamingMoments":
        required = {
            "count",
            "mean",
            "M2",
            "population_variance",
            "population_standard_deviation",
        }
        if set(record) != required:
            raise ValueError("moment record keys differ from the frozen schema")
        result = cls(
            count=int(record["count"]),
            mean=float(record["mean"]),
            m2=float(record["M2"]),
        )
        finalized = result.finalize()
        for key in ("population_variance", "population_standard_deviation"):
            if not math.isclose(
                float(record[key]),
                float(finalized[key]),
                rel_tol=1e-15,
                abs_tol=0.0,
            ):
                raise ValueError(f"inconsistent stored moment value {key}")
        return result


def relative_l2_by_frame(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    reference_array = np.asarray(
        finite_real_array("relative-L2 reference", reference), dtype=np.float64
    )
    candidate_array = np.asarray(
        finite_real_array("relative-L2 candidate", candidate), dtype=np.float64
    )
    if reference_array.shape != candidate_array.shape:
        raise ValueError("relative-L2 arrays have different shapes")
    if reference_array.ndim < 2:
        raise ValueError("relative-L2 arrays require a leading frame axis")
    axes = tuple(range(1, reference_array.ndim))
    numerator = np.sum(
        (candidate_array - reference_array) ** 2,
        axis=axes,
        dtype=np.float64,
    )
    denominator = np.sum(
        reference_array**2,
        axis=axes,
        dtype=np.float64,
    )
    result = np.empty(reference_array.shape[0], dtype=np.float64)
    positive = denominator > 0.0
    result[positive] = np.sqrt(numerator[positive] / denominator[positive])
    zero = ~positive
    result[zero] = np.where(numerator[zero] == 0.0, 0.0, np.inf)
    return result


def records_close(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, bool]:
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("moment tolerances cannot be negative")
    if set(reference) != set(candidate):
        raise ValueError("moment record keys differ")
    result: dict[str, bool] = {}
    for key in reference:
        if key == "count":
            result[key] = int(reference[key]) == int(candidate[key])
        else:
            result[key] = math.isclose(
                float(reference[key]),
                float(candidate[key]),
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
    return result


def write_strict_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write finite JSON without replacing an existing final artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary file {temporary}")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
