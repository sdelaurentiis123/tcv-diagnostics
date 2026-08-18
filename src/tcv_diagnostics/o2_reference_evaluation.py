"""Training-only AR fit and truth-free O2 reference forecast generation."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from .codec_training import sha256_path
from .model_training_data import CodecFrameDataset, ModelDatasetCatalog
from .o2_context_data import OneStepContextDataset
from .o2_forecast import O2ForecastWriter
from .o2_references import (
    SpectralAR1,
    fit_spectral_ar1,
    persistence,
    two_frame_linear_extrapolation,
)
from .o2_training_data import strict_o2_targets


O2_REFERENCE_NAMES = (
    "persistence",
    "spectral_ar1",
    "linear_extrapolation",
)


def fit_training_only_o2_ar1(
    catalog: ModelDatasetCatalog,
    *,
    relative_ridge: float = 1.0e-8,
) -> SpectralAR1:
    """Fit C5P toroidal AR(1) on pairs 0->1 through 430->431 only."""

    frames = tuple(range(432))
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=frames,
        augment=False,
        seed=1701,
        return_physical=False,
    )

    def pairs():
        previous = np.asarray(dataset[0]["volume"], dtype=np.float32)
        for index in range(1, len(dataset)):
            current = np.asarray(dataset[index]["volume"], dtype=np.float32)
            yield previous, current
            previous = current

    try:
        result = fit_spectral_ar1(pairs(), relative_ridge=relative_ridge)
    finally:
        dataset.close()
    if result.pair_count != 431 or result.spatial_sample_count_per_pair != 64 * 32:
        raise RuntimeError("training-only O2 AR(1) fit counts differ")
    return result


def reference_prediction(
    name: str,
    context: np.ndarray,
    *,
    spectral_ar1: SpectralAR1 | None,
) -> np.ndarray:
    """Dispatch one frozen uncompressed standardized reference prediction."""

    if name == "persistence":
        return persistence(context)
    if name == "linear_extrapolation":
        return two_frame_linear_extrapolation(context)
    if name == "spectral_ar1":
        if spectral_ar1 is None:
            raise ValueError("spectral AR(1) reference requires a training-only fit")
        return spectral_ar1.predict(persistence(context))
    raise ValueError(f"unsupported O2 reference {name!r}")


def generate_o2_reference_forecast(
    *,
    catalog: ModelDatasetCatalog,
    name: str,
    target_frames: Sequence[int],
    output: Path,
    spectral_ar1: SpectralAR1 | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Generate a reference artifact using context fields and no target read."""

    if name not in O2_REFERENCE_NAMES:
        raise ValueError(f"unsupported O2 reference {name!r}")
    context_frames = 2 if name == "linear_extrapolation" else 1
    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=context_frames,
    )
    dataset = OneStepContextDataset(
        catalog,
        target_frames=targets,
        context_frames=context_frames,
        return_physical=False,
    )
    record_metadata = {
        **metadata,
        "source_kind": "uncompressed_reference",
        "reference_name": name,
        "context_frames": context_frames,
        "target_truth_read": False,
        "validation_tuning_used": False,
        "training_fit_region": [0, 432] if name == "spectral_ar1" else None,
    }
    wall_started = time.monotonic()
    try:
        with O2ForecastWriter(
            output,
            target_frames=targets,
            metadata=record_metadata,
        ) as writer:
            for index, target in enumerate(targets):
                item = dataset[index]
                if item.get("target_truth_read") is not False or "target" in item:
                    raise RuntimeError("reference context unexpectedly contains truth")
                started = time.perf_counter()
                forecast = reference_prediction(
                    name,
                    item["context"],
                    spectral_ar1=spectral_ar1,
                )
                elapsed = time.perf_counter() - started
                writer.append(
                    target_frame=target,
                    standardized_forecast=forecast,
                    inference_seconds=elapsed,
                )
            writer.finalize()
    finally:
        dataset.close()
    return {
        "schema_version": 1,
        "scope": "O2_uncompressed_reference_forecast_generation",
        "development_run": "85604",
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "target_truth_read_during_forecast_generation": False,
        "validation_tuning_used": False,
        "reference_name": name,
        "context_frames": context_frames,
        "target_frames": [targets[0], targets[-1] + 1],
        "target_count": len(targets),
        "forecast": {
            "path": str(Path(output).resolve(strict=True)),
            "sha256": sha256_path(Path(output)),
        },
        "wall_seconds": time.monotonic() - wall_started,
        "metadata": record_metadata,
    }
