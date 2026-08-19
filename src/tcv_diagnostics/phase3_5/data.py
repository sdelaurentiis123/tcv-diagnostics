"""Hash-locked 85604-only artifact access for Phase 3.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tcv_diagnostics.b5_residual_forecast import B5TrainingForecastArtifact
from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    ModelDatasetCatalog,
)
from tcv_diagnostics.o2_forecast import O2ForecastArtifact
from tcv_diagnostics.residual_kl_oracle import gauge_fixed_residual

from .scope import assert_exact_targets, assert_phase3_5_data_path, validate_phase3_5_frames


ARTIFACT_ROOT = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525"
)
MODEL_DATASET_MANIFEST_SHA256 = "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
MODEL_NORMALIZATION_SHA256 = "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
MODEL_ARTIFACT_INDEX_SHA256 = "6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01"
H1_TRAINING_FORECAST = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_h1_residual_audit/"
    "job_6901393/audit/h1_training_forecast.h5"
)
H1_TRAINING_FORECAST_SHA256 = "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
H1_VALIDATION_FORECAST = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_evaluation_full/"
    "job_6896117/task_0_c5p_h1_seed_1701/forecast.h5"
)
H1_VALIDATION_FORECAST_SHA256 = "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
H1_CHECKPOINT = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o2_full/"
    "job_6894980/task_0_c5p_h1_seed_1701/selected.pt"
)
H1_CHECKPOINT_SHA256 = "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
H1_CODEC = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_r2/"
    "job_6894463/task_0_c5p_seed_1701/selected.pt"
)
H1_CODEC_SHA256 = "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
B5_CHECKPOINT = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_field_residual_edm_full/"
    "job_6901531/b5_joint_field_residual_edm_seed_1701/selected.pt"
)
B5_CHECKPOINT_SHA256 = "255904ef362c4d3f0fdb873131cd0b30bc02ea384e76e244d50698bd50df0c72"
B5_FORECAST = Path(
    "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b5_residual_edm_evaluation_full/"
    "job_6901587/b5_joint_field_residual_edm_seed_1701/forecast_M32.h5"
)
B5_FORECAST_SHA256 = "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
B5_SEEDS = B5_FORECAST.with_name("scientific_sampler_seeds_M32.npy")
B5_SEEDS_SHA256 = "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
GEOMETRY = Path("/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc")
GEOMETRY_SHA256 = "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
NATIVE_RESULT_SHA256 = "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"


@dataclass(frozen=True)
class LockedArtifact:
    name: str
    path: Path
    sha256: str

    def verify(self) -> dict[str, str | int]:
        source = assert_phase3_5_data_path(self.path).resolve(strict=True)
        actual = sha256_path(source)
        if actual != self.sha256:
            raise ValueError(f"Phase 3.5 artifact hash differs for {self.name}: {actual}")
        return {"name": self.name, "path": str(source), "sha256": actual, "bytes": source.stat().st_size}


PRIMARY_ARTIFACTS = (
    LockedArtifact(
        "model_dataset_manifest",
        ARTIFACT_ROOT / "model_dataset_manifest.json",
        MODEL_DATASET_MANIFEST_SHA256,
    ),
    LockedArtifact(
        "model_normalization",
        ARTIFACT_ROOT / "normalization.json",
        MODEL_NORMALIZATION_SHA256,
    ),
    LockedArtifact(
        "model_artifact_index",
        ARTIFACT_ROOT / "artifact_sha256.txt",
        MODEL_ARTIFACT_INDEX_SHA256,
    ),
    LockedArtifact("H1_training_forecast", H1_TRAINING_FORECAST, H1_TRAINING_FORECAST_SHA256),
    LockedArtifact("H1_validation_forecast", H1_VALIDATION_FORECAST, H1_VALIDATION_FORECAST_SHA256),
    LockedArtifact("H1_checkpoint", H1_CHECKPOINT, H1_CHECKPOINT_SHA256),
    LockedArtifact("H1_codec", H1_CODEC, H1_CODEC_SHA256),
    LockedArtifact("B5_checkpoint", B5_CHECKPOINT, B5_CHECKPOINT_SHA256),
    LockedArtifact("B5_forecast", B5_FORECAST, B5_FORECAST_SHA256),
    LockedArtifact("B5_seed_bank", B5_SEEDS, B5_SEEDS_SHA256),
    LockedArtifact("geometry", GEOMETRY, GEOMETRY_SHA256),
)


def verify_primary_artifacts() -> list[dict[str, str | int]]:
    return [artifact.verify() for artifact in PRIMARY_ARTIFACTS]


def _split_for_frames(frames: Sequence[int]) -> str:
    if min(frames) >= 0 and max(frames) < 432:
        return "train"
    if min(frames) >= 496 and max(frames) < 624:
        return "validation"
    raise ValueError("Phase 3.5 frames are not inside one permitted region")


def load_c5_frames(
    catalog: ModelDatasetCatalog,
    frames: Iterable[int],
    *,
    physical: bool,
) -> np.ndarray:
    requested = tuple(int(value) for value in frames)
    split = _split_for_frames(requested)
    validate_phase3_5_frames(requested, split=split, targets=False)
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split=split,
        frames=requested,
        augment=False,
        seed=1701,
        return_physical=physical,
    )
    key = "physical_volume" if physical else "volume"
    try:
        result = np.stack([dataset[index][key] for index in range(len(dataset))])
    finally:
        dataset.close()
    if not np.all(np.isfinite(result)):
        raise ValueError("Phase 3.5 C5 frame request contains non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def load_exact_frames(
    catalog: ModelDatasetCatalog,
    frames: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    requested = tuple(int(value) for value in frames)
    split = _split_for_frames(requested)
    validate_phase3_5_frames(requested, split=split, targets=False)
    dataset = CodecFrameDataset(
        catalog,
        family="e6b",
        split=split,
        frames=requested,
        augment=False,
        seed=1701,
        return_physical=True,
    )
    try:
        state = np.stack([dataset[index]["physical_volume"] for index in range(len(dataset))])
        boundary = np.stack([dataset[index]["physical_boundary"] for index in range(len(dataset))])
    finally:
        dataset.close()
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(boundary)):
        raise ValueError("Phase 3.5 exact-state request contains non-finite values")
    return np.ascontiguousarray(state, dtype=np.float32), np.ascontiguousarray(boundary, dtype=np.float32)


def load_h1_forecasts(
    *,
    training_targets: Sequence[int],
    validation_targets: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    train = tuple(int(value) for value in training_targets)
    validation = tuple(int(value) for value in validation_targets)
    validate_phase3_5_frames(train, split="train", targets=True)
    validate_phase3_5_frames(validation, split="validation", targets=True)
    with B5TrainingForecastArtifact(
        H1_TRAINING_FORECAST,
        expected_sha256=H1_TRAINING_FORECAST_SHA256,
        target_frames=tuple(range(2, 432)),
    ) as artifact:
        start = train[0] - 2
        training = artifact.read(start, start + len(train))
        assert_exact_targets(artifact.target_frames[start : start + len(train)], train)
    with O2ForecastArtifact(
        H1_VALIDATION_FORECAST,
        expected_sha256=H1_VALIDATION_FORECAST_SHA256,
        target_frames=tuple(range(498, 624)),
    ) as artifact:
        start = validation[0] - 498
        later = artifact.read(start, start + len(validation))
        assert_exact_targets(artifact.target_frames[start : start + len(validation)], validation)
    if not np.all(np.isfinite(training)) or not np.all(np.isfinite(later)):
        raise ValueError("Phase 3.5 H1 forecast contains non-finite values")
    return training, later


def corrected_h1_residuals(
    training_truth: np.ndarray,
    training_h1: np.ndarray,
    validation_truth: np.ndarray,
    validation_h1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    training = gauge_fixed_residual(training_truth, training_h1)
    validation = gauge_fixed_residual(validation_truth, validation_h1)
    if not np.all(np.isfinite(training)) or not np.all(np.isfinite(validation)):
        raise ValueError("Phase 3.5 corrected H1 residual contains non-finite values")
    bias = np.mean(training, axis=(0, 4), dtype=np.float64)
    training_corrected = training - bias[None, ..., None]
    validation_corrected = validation - bias[None, ..., None]
    return (
        np.asarray(training_corrected, dtype=np.float32),
        np.asarray(validation_corrected, dtype=np.float32),
        np.asarray(bias, dtype=np.float64),
    )


def decode_c5(catalog: ModelDatasetCatalog, standardized: np.ndarray) -> np.ndarray:
    values = np.asarray(standardized)
    if values.ndim != 5 or values.shape[1] != len(FAMILY_FIELDS["c5p"]):
        raise ValueError("C5 decode expects [sample,5,x,y,z]")
    decoded = np.stack(
        [catalog.normalization.decode_volume(FAMILY_FIELDS["c5p"], sample) for sample in values]
    )
    return np.asarray(decoded, dtype=np.float64)
