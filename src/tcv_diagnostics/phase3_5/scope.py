"""Scope, split, and output guards for Paper 0 Phase 3.5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRAIN_FRAMES = (0, 432)
TRAIN_TARGETS = (2, 432)
GUARD_FRAMES = (432, 496)
VALIDATION_FRAMES = (496, 624)
VALIDATION_TARGETS = (498, 624)


def sha256_path(path: Path, *, block_bytes: int = 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_phase3_5_data_path(path: Path) -> Path:
    """Reject any path that could identify the held-out simulation."""

    value = Path(path)
    normalized = str(value).lower()
    if "85606" in normalized:
        raise ValueError(f"Phase 3.5 held-out path is prohibited: {value}")
    return value


def assert_no_held_out_metadata(value: Any, *, location: str = "metadata") -> None:
    """Recursively reject metadata that claims or points to held-out access."""

    if isinstance(value, str):
        if "85606" in value.lower():
            raise ValueError(f"held-out identifier found in {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if "85606" in key_text and item not in (False, None, "unopened", "closed"):
                raise ValueError(f"held-out metadata is not closed at {location}.{key}")
            assert_no_held_out_metadata(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_held_out_metadata(item, location=f"{location}[{index}]")


def validate_phase3_5_frames(
    frames: Iterable[int],
    *,
    split: str,
    targets: bool,
) -> tuple[int, ...]:
    values = tuple(int(item) for item in frames)
    if not values or values != tuple(range(values[0], values[-1] + 1)):
        raise ValueError("Phase 3.5 frames must be nonempty, ordered, and contiguous")
    if split == "train":
        allowed = TRAIN_TARGETS if targets else TRAIN_FRAMES
    elif split == "validation":
        allowed = VALIDATION_TARGETS if targets else VALIDATION_FRAMES
    else:
        raise ValueError(f"unsupported Phase 3.5 split {split!r}")
    if values[0] < allowed[0] or values[-1] >= allowed[1]:
        raise ValueError(f"frames leave frozen {split} interval {allowed}")
    if any(GUARD_FRAMES[0] <= frame < GUARD_FRAMES[1] for frame in values):
        raise ValueError("Phase 3.5 guard frame read is prohibited")
    return values


@dataclass(frozen=True)
class Phase35Block:
    identifier: str
    split: str
    start: int
    stop: int
    matched_start: int
    matched_stop: int

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))

    @property
    def matched_frames(self) -> tuple[int, ...]:
        return tuple(range(self.matched_start, self.matched_stop))

    def __post_init__(self) -> None:
        validate_phase3_5_frames(self.frames, split=self.split, targets=True)
        validate_phase3_5_frames(self.matched_frames, split=self.split, targets=True)
        if self.matched_start < self.start or self.matched_stop > self.stop:
            raise ValueError("matched Phase 3.5 block leaves its parent block")


@dataclass(frozen=True)
class Phase35Protocol:
    path: Path
    record: Mapping[str, Any]
    blocks: tuple[Phase35Block, ...]

    @property
    def training_blocks(self) -> tuple[Phase35Block, ...]:
        return tuple(block for block in self.blocks if block.split == "train")

    @property
    def validation_blocks(self) -> tuple[Phase35Block, ...]:
        return tuple(block for block in self.blocks if block.split == "validation")

    @property
    def matched_sample_count(self) -> int:
        return int(self.record["blocks"]["matched_sample_count"])

    def block_for_target(self, target: int) -> Phase35Block:
        matches = [block for block in self.blocks if block.start <= target < block.stop]
        if len(matches) != 1:
            raise ValueError(f"target {target} has {len(matches)} Phase 3.5 blocks")
        return matches[0]


def _block(item: Mapping[str, Any], *, split: str) -> Phase35Block:
    start, stop = (int(value) for value in item["range"])
    matched_start, matched_stop = (int(value) for value in item["matched_range"])
    return Phase35Block(
        identifier=str(item["id"]),
        split=split,
        start=start,
        stop=stop,
        matched_start=matched_start,
        matched_stop=matched_stop,
    )


def load_phase3_5_protocol(path: Path, *, root: Path | None = None) -> Phase35Protocol:
    source = Path(path)
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("Phase 3.5 manifest must be a JSON object")
    if record.get("development_run") != "85604":
        raise ValueError("Phase 3.5 manifest is not for development run 85604")
    if record.get("held_out_85606_access_allowed") is not False:
        raise ValueError("Phase 3.5 manifest does not keep held-out access closed")
    data = record.get("data", {})
    expected = {
        "raw_training_frames": list(TRAIN_FRAMES),
        "training_targets": list(TRAIN_TARGETS),
        "guard_frames": list(GUARD_FRAMES),
        "raw_validation_frames": list(VALIDATION_FRAMES),
        "validation_targets": list(VALIDATION_TARGETS),
        "periodic_axes_xyz": [False, False, True],
        "zperiod": 5,
        "mode_mapping": "n=5k",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Phase 3.5 data setting {key} differs")
    protocol_record = record.get("protocol", {})
    protocol_path = Path(protocol_record.get("path", ""))
    if root is not None:
        protocol_path = Path(root) / protocol_path
    elif not protocol_path.is_absolute():
        protocol_path = source.parents[2] / protocol_path
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if sha256_path(protocol_path) != protocol_record.get("sha256"):
        raise ValueError("Phase 3.5 protocol SHA-256 differs")
    for amendment in record.get("clarifying_amendments", ()):
        amendment_path = Path(amendment.get("path", ""))
        if root is not None:
            amendment_path = Path(root) / amendment_path
        elif not amendment_path.is_absolute():
            amendment_path = source.parents[2] / amendment_path
        if not amendment_path.is_file():
            raise FileNotFoundError(amendment_path)
        if sha256_path(amendment_path) != amendment.get("sha256"):
            raise ValueError("Phase 3.5 clarifying-amendment SHA-256 differs")
    block_record = record.get("blocks", {})
    blocks = tuple(
        [_block(item, split="train") for item in block_record.get("training", ())]
        + [
            _block(item, split="validation")
            for item in block_record.get("validation", ())
        ]
    )
    if len(blocks) != 13:
        raise ValueError("Phase 3.5 must contain ten train and three validation blocks")
    matched_count = int(block_record.get("matched_sample_count", -1))
    if matched_count != 42 or any(len(block.matched_frames) != 42 for block in blocks):
        raise ValueError("Phase 3.5 matched block count differs")
    return Phase35Protocol(path=source, record=record, blocks=blocks)


def exclusive_output(path: Path) -> Path:
    """Fail before writing when an immutable output already exists."""

    target = assert_phase3_5_data_path(Path(path))
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def assert_exact_targets(actual: Sequence[int], expected: Sequence[int]) -> None:
    if tuple(int(value) for value in actual) != tuple(int(value) for value in expected):
        raise ValueError("Phase 3.5 target order differs from its frozen sequence")
