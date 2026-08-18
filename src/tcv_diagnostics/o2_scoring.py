"""Truth-separated scoring of immutable Paper 0 O2 forecast artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .codec_transport import (
    CodecTransportGeometry,
    TransportComparisonAccumulator,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .matched_codec_metrics import MatchedCodecAccumulator
from .matched_o1_transport import NativeTruthCatalog
from .model_training_data import (
    CodecFrameDataset,
    FAMILY_FIELDS,
    ModelDatasetCatalog,
)
from .o2_evaluation import (
    O2_BLOCK_FRAMES,
    O2_FIELDS,
    O2_VALIDATION_BLOCKS,
    O2_VALIDATION_TARGETS,
    O2MetricAccumulator,
    O2_VIEW,
    o2_training_materiality,
    validation_blocks,
)
from .o2_forecast import O2ForecastArtifact
from .o2_training_data import OneStepWindowDataset, strict_o2_targets
from .resampling import periodic_resample_float32


O2_TRANSPORT_COMPARISON = {
    "truth_vs_forecast": ("truth", "forecast"),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def compute_o2_training_materiality(
    catalog: ModelDatasetCatalog,
    *,
    chunk_frames: int = 8,
) -> dict[str, Any]:
    """Compute inherited material bands from 85604 training truth only."""

    if not 1 <= int(chunk_frames) <= 32:
        raise ValueError("training materiality chunk size must lie in 1..32")
    frames = tuple(range(432))
    dataset = CodecFrameDataset(
        catalog,
        family="c5p",
        split="train",
        frames=frames,
        augment=False,
        seed=1701,
        return_physical=True,
    )
    accumulator = MatchedCodecAccumulator(spec=O2_VIEW, n_z=88, zperiod=5)
    try:
        for start in range(0, len(frames), int(chunk_frames)):
            stop = min(start + int(chunk_frames), len(frames))
            items = [dataset[index] for index in range(start, stop)]
            standardized = np.stack([item["volume"] for item in items], axis=0)
            physical = np.stack(
                [item["physical_volume"] for item in items], axis=0
            )
            accumulator.update(
                standardized,
                standardized,
                physical,
                physical,
            )
    finally:
        dataset.close()
    truth_metrics = accumulator.finalize()
    materiality = o2_training_materiality(truth_metrics)
    if materiality["source_split"] != "85604_training_[0,432)":
        raise RuntimeError("O2 materiality source split differs")
    return {
        "schema_version": 1,
        "scope": "O2_training_truth_materiality",
        "development_run": "85604",
        "held_out_85606_read": False,
        "training_frames": [0, 432],
        "validation_truth_used_to_select_bands": False,
        "materiality": materiality,
        "training_truth_spectral_metrics": truth_metrics,
    }


def _per_target_error(
    truth: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, Any]:
    truth_array = np.asarray(truth, dtype=np.float64)
    forecast_array = np.asarray(forecast, dtype=np.float64)
    if truth_array.shape != forecast_array.shape or truth_array.shape != (
        len(O2_FIELDS),
        64,
        32,
        88,
    ):
        raise ValueError("per-target O2 field shapes differ")
    difference = forecast_array - truth_array
    axes = (1, 2, 3)
    rmse = np.sqrt(np.mean(difference * difference, axis=axes, dtype=np.float64))
    mae = np.mean(np.abs(difference), axis=axes, dtype=np.float64)
    bias = np.mean(difference, axis=axes, dtype=np.float64)
    return {
        "aggregate_equal_channel_rmse_standardized": float(
            np.sqrt(np.mean(difference * difference, dtype=np.float64))
        ),
        "aggregate_equal_channel_mae_standardized": float(
            np.mean(np.abs(difference), dtype=np.float64)
        ),
        "fields": {
            field: {
                "rmse": float(rmse[index]),
                "mae": float(mae[index]),
                "bias": float(bias[index]),
            }
            for index, field in enumerate(O2_FIELDS)
        },
    }


def _transport_states(
    *,
    physical_forecast_model88: np.ndarray,
    native_truth: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    forecast = np.asarray(physical_forecast_model88, dtype=np.float64)
    if forecast.shape != (len(O2_FIELDS), 64, 32, 88):
        raise ValueError("physical forecast shape differs before transport")
    native_forecast = periodic_resample_float32(
        forecast[[0, 1, 2, 3]][None],
        81,
        axis=-1,
    ).astype(np.float64)
    truth_state = direct_pressure_transport_state(
        native_truth["Ne"],
        native_truth["Pe"],
        native_truth["Pi"],
        native_truth["phi"],
    )
    forecast_state = direct_pressure_transport_state(
        native_forecast[:, 0],
        native_forecast[:, 1],
        native_forecast[:, 2],
        native_forecast[:, 3],
    )
    return truth_state, forecast_state


def score_o2_forecast(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: O2ForecastArtifact,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    target_frames: Sequence[int],
    scientific_authority: bool,
) -> dict[str, Any]:
    """Score one immutable forecast after prediction has already finished."""

    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=1,
    )
    if forecast_artifact.target_frames != targets:
        raise ValueError("forecast artifact and score target frames differ")
    if forecast_artifact.metadata.get("target_truth_read") is not False:
        raise ValueError("forecast artifact does not preserve the target-read lock")
    if scientific_authority:
        blocks = validation_blocks(targets)
    else:
        blocks = (targets,)
    block_for_position = {
        position: block_index
        for block_index, block in enumerate(blocks)
        for position in range(
            block[0] - targets[0], block[-1] - targets[0] + 1
        )
    }
    if set(block_for_position) != set(range(len(targets))):
        raise RuntimeError("O2 scoring blocks do not cover every target")

    truth_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=targets,
        context_frames=1,
        augment=False,
        seed=1701,
        return_physical=True,
    )
    overall_metrics = O2MetricAccumulator(n_z=88, zperiod=5)
    block_metrics = [O2MetricAccumulator(n_z=88, zperiod=5) for _ in blocks]
    overall_transport = TransportComparisonAccumulator(O2_TRANSPORT_COMPARISON)
    block_transport = [
        TransportComparisonAccumulator(O2_TRANSPORT_COMPARISON) for _ in blocks
    ]
    time_curve = []
    try:
        for position, target_frame in enumerate(targets):
            item = truth_dataset[position]
            if int(item["target_frame_index"]) != target_frame:
                raise RuntimeError("truth scoring order differs")
            standardized_forecast = forecast_artifact.read(position, position + 1)[0]
            standardized_truth = np.asarray(item["target"], dtype=np.float32)
            physical_truth = np.asarray(item["physical_target"], dtype=np.float32)
            physical_latest = np.asarray(
                item["physical_context"][-1], dtype=np.float32
            )
            physical_forecast = catalog.normalization.decode_volume(
                FAMILY_FIELDS["c5p"],
                standardized_forecast,
            )
            update = {
                "standardized_truth": standardized_truth[None],
                "standardized_forecast": standardized_forecast[None],
                "physical_truth": physical_truth[None],
                "physical_forecast": physical_forecast[None],
                "physical_latest_context": physical_latest[None],
            }
            overall_metrics.update(**update)
            block_index = block_for_position[position]
            block_metrics[block_index].update(**update)

            truth_native = native_truth.read(
                target_frame,
                target_frame + 1,
                fields=("Ne", "Pe", "Pi", "phi"),
            )
            truth_state, forecast_state = _transport_states(
                physical_forecast_model88=physical_forecast,
                native_truth=truth_native,
            )
            transport_paths = {
                "truth": evaluate_transport_state(truth_state, geometry),
                "forecast": evaluate_transport_state(forecast_state, geometry),
            }
            overall_transport.update(transport_paths)
            block_transport[block_index].update(transport_paths)
            time_curve.append(
                {
                    "target_frame": target_frame,
                    "horizon_frames": 1,
                    "horizon_microseconds": 3.131905426352636,
                    **_per_target_error(
                        standardized_truth,
                        standardized_forecast,
                    ),
                }
            )
    finally:
        truth_dataset.close()

    overall_record = overall_metrics.finalize()
    block_records = [accumulator.finalize() for accumulator in block_metrics]
    transport_record = overall_transport.finalize()
    transport_blocks = [accumulator.finalize() for accumulator in block_transport]
    expected_block_count = O2_VALIDATION_BLOCKS if scientific_authority else 1
    expected_block_frames = O2_BLOCK_FRAMES if scientific_authority else len(targets)
    if len(block_records) != expected_block_count or any(
        int(record["frames"]) != expected_block_frames for record in block_records
    ):
        raise RuntimeError("O2 scoring block record differs")
    return _jsonable(
        {
            "schema_version": 1,
            "scope": "O2_truth_separated_forecast_scoring",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "training_performed": False,
            "physics_derived_training_loss_used": False,
            "target_truth_used_during_forecast_generation": False,
            "scientific_authority": bool(scientific_authority),
            "target_frames": [targets[0], targets[-1] + 1],
            "target_count": len(targets),
            "validation_blocks": [
                [block[0], block[-1] + 1] for block in blocks
            ],
            "forecast_artifact": {
                "path": str(forecast_artifact.path.resolve(strict=True)),
                "sha256": forecast_artifact.sha256,
                "metadata": forecast_artifact.metadata,
                "timing": forecast_artifact.timing_record(),
            },
            "field_spectral_cross": {
                "overall": overall_record,
                "blocks": block_records,
            },
            "transport": {
                "overall": transport_record,
                "blocks": transport_blocks,
                "nonlinear_operator_applied_before_any_ensemble_or_time_reduction": True,
                "model88_to_native81_resampling": (
                    "frozen_unwindowed_periodic_scipy_signal_resample_float32"
                ),
            },
            "error_by_target": time_curve,
        }
    )
