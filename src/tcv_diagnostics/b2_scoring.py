"""Truth-separated scientific scoring of immutable B2 ensemble artifacts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, b2_region_masks
from .b2_field_scoring import (
    B2_VALIDATION_BLOCKS,
    B2_VALIDATION_TARGETS,
    B2FieldScoreAccumulator,
)
from .b2_forecast import B2ForecastArtifact
from .b2_spectral_metrics import B2SpectralAccumulator
from .b2_transport_metrics import (
    B2TransportAccumulator,
    memberwise_transport_outputs,
)
from .codec_transport import (
    TRANSPORT_QUANTITIES,
    CodecTransportGeometry,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .matched_o1_transport import NativeTruthCatalog
from .model_training_data import FAMILY_FIELDS, ModelDatasetCatalog
from .o2_training_data import OneStepWindowDataset, strict_o2_targets


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def compute_b2_transport_event_thresholds(
    *,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    chunk_frames: int = 8,
) -> dict[str, Any]:
    """Fit absolute separatrix-transport p90 thresholds on frames 0..431."""

    chunk = int(chunk_frames)
    if not 1 <= chunk <= 32:
        raise ValueError("B2 event-threshold chunk size must lie in 1..32")
    values = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
    for start in range(0, 432, chunk):
        stop = min(start + chunk, 432)
        fields = native_truth.read(
            start,
            stop,
            fields=("Ne", "Pe", "Pi", "phi"),
        )
        state = direct_pressure_transport_state(
            fields["Ne"], fields["Pe"], fields["Pi"], fields["phi"]
        )
        evaluated = evaluate_transport_state(state, geometry)
        for quantity in TRANSPORT_QUANTITIES:
            surface = np.asarray(
                evaluated[quantity]["separatrix_wedge"], dtype=np.float64
            )
            if surface.shape != (stop - start,) or not np.all(np.isfinite(surface)):
                raise ValueError("B2 training transport surface shape/values differ")
            values[quantity].append(surface)
    thresholds = {}
    summaries = {}
    for quantity in TRANSPORT_QUANTITIES:
        series = np.concatenate(values[quantity])
        if series.shape != (432,):
            raise RuntimeError("B2 training transport threshold series differs")
        absolute = np.abs(series)
        threshold = float(np.quantile(absolute, 0.90, method="linear"))
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("B2 transport event threshold is invalid")
        thresholds[quantity] = threshold
        summaries[quantity] = {
            "absolute_value_p90": threshold,
            "minimum": float(np.min(series)),
            "median": float(np.median(series)),
            "maximum": float(np.max(series)),
            "nonzero_count": int(np.count_nonzero(series)),
        }
    return {
        "schema_version": 1,
        "scope": "B2_training_only_transport_event_thresholds",
        "development_run": "85604",
        "training_frames": [0, 432],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "quantile_probability": 0.90,
        "quantile_method": "numpy_linear",
        "absolute_value_before_quantile": True,
        "thresholds": thresholds,
        "training_truth_summaries": summaries,
        "physics_derived_training_loss_used": False,
    }


def validate_b2_transport_event_thresholds(
    record: Mapping[str, Any],
) -> dict[str, float]:
    """Reject threshold artifacts not fitted solely on frozen 85604 training."""

    if (
        record.get("scope") != "B2_training_only_transport_event_thresholds"
        or record.get("development_run") != "85604"
        or record.get("training_frames") != [0, 432]
        or record.get("validation_frames_read") is not False
        or record.get("held_out_85606_read") is not False
        or record.get("quantile_probability") != 0.90
        or record.get("quantile_method") != "numpy_linear"
        or record.get("absolute_value_before_quantile") is not True
        or record.get("physics_derived_training_loss_used") is not False
    ):
        raise ValueError("B2 transport event-threshold artifact contract differs")
    thresholds = record.get("thresholds", {})
    if tuple(thresholds) != TRANSPORT_QUANTITIES:
        raise ValueError("B2 event-threshold quantity order differs")
    result = {name: float(value) for name, value in thresholds.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("B2 event-threshold value differs")
    return result


def decode_b2_member_forecasts(
    catalog: ModelDatasetCatalog,
    standardized_forecast: np.ndarray,
) -> np.ndarray:
    """Inverse-transform M32 C5P fields without clipping or member reduction."""

    values = np.asarray(standardized_forecast)
    if values.shape != (32, 1, len(B2_FIELDS), 64, 32, 88):
        raise ValueError("B2 standardized member forecast shape differs")
    if np.iscomplexobj(values) or not np.issubdtype(values.dtype, np.number):
        raise TypeError("B2 standardized forecast must be real numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError("B2 standardized forecast contains non-finite values")
    physical = np.stack(
        [
            catalog.normalization.records[field].decode(values[:, 0, channel])
            for channel, field in enumerate(FAMILY_FIELDS["c5p"])
        ],
        axis=1,
    )
    if physical.shape != (32, len(B2_FIELDS), 64, 32, 88):
        raise RuntimeError("B2 decoded physical ensemble shape differs")
    if not np.all(np.isfinite(physical)):
        raise ValueError("B2 decoded physical ensemble is non-finite")
    return np.asarray(physical, dtype=np.float64)


def score_b2_forecast(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: B2ForecastArtifact,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    model_seed: int,
) -> dict[str, Any]:
    """Score one closed and hash-verified M32 artifact against 85604 truth."""

    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=2,
    )
    if targets != B2_VALIDATION_TARGETS:
        raise ValueError("scientific B2 scoring requires targets 498..623")
    if forecast_artifact.target_frames != targets:
        raise ValueError("B2 forecast artifact/scorer target frames differ")
    if forecast_artifact.model_seed != int(model_seed):
        raise ValueError("B2 forecast artifact/scorer model seed differs")
    if forecast_artifact.metadata.get("target_truth_read") is not False:
        raise ValueError("B2 forecast metadata does not preserve truth separation")
    thresholds = validate_b2_transport_event_thresholds(event_threshold_record)
    masks = geometry.region_masks
    region_masks = b2_region_masks(masks, n_z=88)
    eligible_xy = np.asarray(
        masks.strict_wall_interior & masks.operator_interior,
        dtype=bool,
    )
    field = B2FieldScoreAccumulator(
        model_seed=model_seed,
        target_frames=targets,
        region_masks=region_masks,
        validation_blocks=B2_VALIDATION_BLOCKS,
    )
    spectral = B2SpectralAccumulator(
        model_seed=model_seed,
        target_frames=targets,
        eligible_xy_mask=eligible_xy,
    )
    spectral_blocks = [
        B2SpectralAccumulator(
            model_seed=model_seed,
            target_frames=block,
            eligible_xy_mask=eligible_xy,
        )
        for block in B2_VALIDATION_BLOCKS
    ]
    transport = B2TransportAccumulator(
        model_seed=model_seed,
        target_frames=targets,
        event_thresholds=thresholds,
        detailed=True,
    )
    transport_blocks = [
        B2TransportAccumulator(
            model_seed=model_seed,
            target_frames=block,
            event_thresholds=thresholds,
            detailed=False,
        )
        for block in B2_VALIDATION_BLOCKS
    ]
    block_for_target = {
        target: index
        for index, block in enumerate(B2_VALIDATION_BLOCKS)
        for target in block
    }
    truth_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=targets,
        context_frames=2,
        augment=False,
        seed=1701,
        return_physical=True,
    )
    try:
        for position, target in enumerate(targets):
            item = truth_dataset[position]
            if int(item["target_frame_index"]) != target:
                raise RuntimeError("B2 truth scoring order differs")
            standardized_forecast = forecast_artifact.read(position, position + 1)[0]
            standardized_truth = np.asarray(item["target"], dtype=np.float32)
            physical_truth = np.asarray(item["physical_target"], dtype=np.float64)
            physical_forecast = decode_b2_member_forecasts(
                catalog,
                standardized_forecast,
            )
            field.update(
                target_frame=target,
                standardized_forecast=standardized_forecast,
                standardized_truth=standardized_truth,
            )
            block_index = block_for_target[target]
            spectral.update(
                target_frame=target,
                physical_forecast=physical_forecast,
                physical_truth=physical_truth,
                mirrors=(spectral_blocks[block_index],),
            )
            truth_native = native_truth.read(
                target,
                target + 1,
                fields=("Ne", "Pe", "Pi", "phi"),
            )
            forecast_transport, truth_transport = memberwise_transport_outputs(
                physical_forecast_model88=physical_forecast,
                native_truth=truth_native,
                geometry=geometry,
            )
            transport.update(
                target_frame=target,
                forecast_outputs=forecast_transport,
                truth_outputs=truth_transport,
                mirrors=(transport_blocks[block_index],),
            )
    finally:
        truth_dataset.close()
    return _json_safe(
        {
            "schema_version": 1,
            "scope": "B2_truth_separated_probabilistic_scoring_85604",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "target_truth_used_during_forecast_generation": False,
            "truth_opened_only_after_forecast_was_closed_and_hash_verified": True,
            "training_performed": False,
            "physics_derived_training_loss_used": False,
            "model_seed": int(model_seed),
            "target_frames": [targets[0], targets[-1] + 1],
            "target_count": len(targets),
            "forecast_artifact": {
                "path": str(forecast_artifact.path.resolve(strict=True)),
                "sha256": forecast_artifact.sha256,
                "metadata": forecast_artifact.metadata,
                "timing": forecast_artifact.timing_record(),
            },
            "transport_event_thresholds": dict(event_threshold_record),
            "field_and_marginal_calibration": field.finalize(),
            "spectral_and_cross_field": {
                "overall": spectral.finalize(),
                "chronological_blocks": [
                    block.finalize() for block in spectral_blocks
                ],
            },
            "memberwise_transport": {
                "overall": transport.finalize(),
                "chronological_blocks": [
                    block.finalize() for block in transport_blocks
                ],
            },
        }
    )
