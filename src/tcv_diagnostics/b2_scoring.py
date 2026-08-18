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
from .b2_spectral_metrics import (
    B2_CROSS_PAIRS,
    B2_MODE_BANDS,
    B2SpectralAccumulator,
)
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
from .model_training_data import (
    FAMILY_FIELDS,
    CodecFrameDataset,
    ModelDatasetCatalog,
)
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


def _one_sided_spectral_weights(size: int) -> np.ndarray:
    weights = np.ones(int(size) // 2 + 1, dtype=np.float64)
    if int(size) % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


def compute_b2_spectral_materiality(
    *,
    dataset: CodecFrameDataset,
    eligible_xy_mask: np.ndarray,
    training_frame_count: int = 432,
) -> dict[str, Any]:
    """Freeze field-power/cross-amplitude material bands on training truth."""

    frame_count = int(training_frame_count)
    if frame_count <= 0 or tuple(dataset.frames) != tuple(range(frame_count)):
        raise ValueError("B2 spectral materiality requires exact training frames")
    if tuple(dataset.fields) != B2_FIELDS or dataset.return_physical is not True:
        raise ValueError("B2 spectral materiality dataset state differs")
    eligible = np.asarray(eligible_xy_mask, dtype=bool)
    if eligible.ndim != 2 or not np.any(eligible):
        raise ValueError("B2 spectral materiality eligible mask differs")
    field_auto: np.ndarray | None = None
    pair_cross: np.ndarray | None = None
    n_z: int | None = None
    pairs = tuple(
        (B2_FIELDS.index(first), B2_FIELDS.index(second))
        for first, second in B2_CROSS_PAIRS
    )
    for index in range(frame_count):
        item = dataset[index]
        if int(item["frame_index"]) != index:
            raise RuntimeError("B2 materiality frame order differs")
        physical = np.asarray(item["physical_volume"], dtype=np.float64)
        if physical.ndim != 4 or physical.shape[0] != len(B2_FIELDS):
            raise ValueError("B2 materiality physical volume shape differs")
        if physical.shape[1:3] != eligible.shape:
            raise ValueError("B2 materiality field/mask geometry differs")
        if n_z is None:
            n_z = int(physical.shape[-1])
            if n_z // 2 < 7:
                raise ValueError("B2 materiality grid cannot resolve k=7")
            modes = n_z // 2 + 1
            field_auto = np.zeros((len(B2_FIELDS), modes), dtype=np.float64)
            pair_cross = np.zeros((len(B2_CROSS_PAIRS), modes), dtype=np.complex128)
        elif physical.shape[-1] != n_z:
            raise ValueError("B2 materiality toroidal size changed")
        coefficients = np.fft.rfft(physical, axis=-1)[:, eligible, :]
        if field_auto is None or pair_cross is None:
            raise AssertionError("B2 materiality accumulators are absent")
        field_auto += np.sum(
            np.abs(coefficients) ** 2,
            axis=1,
            dtype=np.float64,
        )
        for pair_index, (first, second) in enumerate(pairs):
            pair_cross[pair_index] += np.sum(
                coefficients[first] * np.conjugate(coefficients[second]),
                axis=0,
                dtype=np.complex128,
            )
    if n_z is None or field_auto is None or pair_cross is None:
        raise RuntimeError("B2 spectral materiality consumed no frames")
    weights = _one_sided_spectral_weights(n_z)
    threshold = 0.01
    fields: dict[str, Any] = {}
    for channel, field in enumerate(B2_FIELDS):
        denominator = float(np.sum(field_auto[channel, 1:] * weights[1:]))
        if denominator <= 0.0:
            raise ValueError(f"B2 training non-axisymmetric power is zero for {field}")
        bands = {}
        for label, low, high in B2_MODE_BANDS:
            numerator = float(
                np.sum(field_auto[channel, low : high + 1] * weights[low : high + 1])
            )
            fraction = numerator / denominator
            bands[label] = {
                "stored_k": [low, high],
                "full_torus_n": [5 * low, 5 * high],
                "fraction_of_training_nonaxisymmetric_power": fraction,
                "material": fraction >= threshold,
            }
        fields[field] = {
            "training_nonaxisymmetric_power": denominator,
            "bands": bands,
        }
    cross_fields: dict[str, Any] = {}
    for pair_index, pair in enumerate(B2_CROSS_PAIRS):
        amplitude = np.abs(pair_cross[pair_index]) * weights
        denominator = float(np.sum(amplitude[1:]))
        if denominator <= 0.0:
            raise ValueError(f"B2 training cross amplitude is zero for {pair}")
        bands = {}
        for label, low, high in B2_MODE_BANDS:
            fraction = float(np.sum(amplitude[low : high + 1])) / denominator
            bands[label] = {
                "stored_k": [low, high],
                "full_torus_n": [5 * low, 5 * high],
                "fraction_of_training_nonaxisymmetric_cross_amplitude": fraction,
                "material": fraction >= threshold,
            }
        cross_fields[f"{pair[0]}-{pair[1]}"] = {
            "training_nonaxisymmetric_cross_amplitude": denominator,
            "bands": bands,
        }
    return {
        "schema_version": 1,
        "scope": "B2_training_only_spectral_materiality",
        "development_run": "85604",
        "training_frames": [0, frame_count],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "materiality_fraction_minimum": threshold,
        "field_denominator": "all_training_nonaxisymmetric_toroidal_power_k_ge_1",
        "cross_denominator": (
            "sum_mode_absolute_aggregate_training_cross_spectrum_k_ge_1"
        ),
        "fields": fields,
        "cross_fields": cross_fields,
        "physics_derived_training_loss_used": False,
    }


def inherited_b2_spectral_materiality(
    o2_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert the already frozen O2 training materiality into B2 schema."""

    if (
        o2_record.get("scope") != "O2_training_truth_materiality"
        or o2_record.get("development_run") != "85604"
        or o2_record.get("training_frames") != [0, 432]
        or o2_record.get("held_out_85606_read") is not False
        or o2_record.get("validation_truth_used_to_select_bands") is not False
    ):
        raise ValueError("inherited O2 materiality contract differs")
    source = o2_record.get("materiality", {})
    if (
        source.get("source_split") != "85604_training_[0,432)"
        or source.get("minimum_fraction") != 0.01
        or source.get("view", {}).get("fields") != list(B2_FIELDS)
        or source.get("view", {}).get("cross_pairs")
        != [list(pair) for pair in B2_CROSS_PAIRS]
    ):
        raise ValueError("inherited O2 materiality definition differs")
    expected_bands = tuple(label for label, _, _ in B2_MODE_BANDS)

    def converted_bands(values: Mapping[str, Any], *, cross: bool) -> dict[str, Any]:
        if tuple(values) != expected_bands:
            raise ValueError("inherited O2 materiality bands differ")
        converted = {}
        for label, low, high in B2_MODE_BANDS:
            item = values[label]
            fraction = float(item.get("truth_fraction"))
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("inherited O2 materiality fraction differs")
            if bool(item.get("material")) != (fraction >= 0.01):
                raise ValueError("inherited O2 materiality decision differs")
            fraction_name = (
                "fraction_of_training_nonaxisymmetric_cross_amplitude"
                if cross
                else "fraction_of_training_nonaxisymmetric_power"
            )
            converted[label] = {
                "stored_k": [low, high],
                "full_torus_n": [5 * low, 5 * high],
                fraction_name: fraction,
                "material": bool(item["material"]),
            }
        return converted

    fields_source = source.get("fields", {})
    crosses_source = source.get("cross_pairs", {})
    if set(fields_source) != set(B2_FIELDS):
        raise ValueError("inherited O2 materiality fields differ")
    expected_crosses = tuple(f"{a}-{b}" for a, b in B2_CROSS_PAIRS)
    if set(crosses_source) != set(expected_crosses):
        raise ValueError("inherited O2 materiality cross fields differ")
    return {
        "schema_version": 1,
        "scope": "B2_training_only_spectral_materiality",
        "development_run": "85604",
        "training_frames": [0, 432],
        "validation_frames_read": False,
        "held_out_85606_read": False,
        "zperiod": 5,
        "mode_mapping": "n=5k",
        "materiality_fraction_minimum": 0.01,
        "field_denominator": (
            "inherited_O2_full_model_crop_nonaxisymmetric_toroidal_power_k_ge_1"
        ),
        "cross_denominator": (
            "inherited_O2_sum_mode_absolute_aggregate_training_cross_spectrum_k_ge_1"
        ),
        "inherited_without_refitting": True,
        "fields": {
            field: {"bands": converted_bands(fields_source[field], cross=False)}
            for field in B2_FIELDS
        },
        "cross_fields": {
            pair: {"bands": converted_bands(crosses_source[pair], cross=True)}
            for pair in expected_crosses
        },
        "physics_derived_training_loss_used": False,
    }


def validate_b2_spectral_materiality(record: Mapping[str, Any]) -> None:
    if (
        record.get("scope") != "B2_training_only_spectral_materiality"
        or record.get("development_run") != "85604"
        or record.get("training_frames") != [0, 432]
        or record.get("validation_frames_read") is not False
        or record.get("held_out_85606_read") is not False
        or record.get("zperiod") != 5
        or record.get("mode_mapping") != "n=5k"
        or record.get("materiality_fraction_minimum") != 0.01
        or record.get("physics_derived_training_loss_used") is not False
    ):
        raise ValueError("B2 spectral materiality artifact contract differs")
    expected_bands = {label for label, _, _ in B2_MODE_BANDS}
    fields = record.get("fields", {})
    crosses = record.get("cross_fields", {})
    if set(fields) != set(B2_FIELDS):
        raise ValueError("B2 spectral materiality field keys differ")
    if set(crosses) != {f"{a}-{b}" for a, b in B2_CROSS_PAIRS}:
        raise ValueError("B2 spectral materiality cross-field keys differ")
    for group in (fields, crosses):
        for item in group.values():
            if set(item.get("bands", {})) != expected_bands:
                raise ValueError("B2 spectral materiality band keys differ")
            for band in item["bands"].values():
                keys = [
                    name for name in band if name.startswith("fraction_of_training_")
                ]
                if len(keys) != 1:
                    raise ValueError("B2 spectral materiality fraction differs")
                fraction = float(band[keys[0]])
                if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                    raise ValueError("B2 spectral materiality value differs")
                if bool(band.get("material")) != (fraction >= 0.01):
                    raise ValueError("B2 spectral materiality decision differs")


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
    if set(thresholds) != set(TRANSPORT_QUANTITIES):
        raise ValueError("B2 event-threshold quantity keys differ")
    result = {name: float(thresholds[name]) for name in TRANSPORT_QUANTITIES}
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

    return _score_b2_forecast(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        model_seed=model_seed,
        bounded_smoke=False,
    )


def score_b2_forecast_smoke(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: B2ForecastArtifact,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    model_seed: int,
) -> dict[str, Any]:
    """Run the same scorer on four targets as a non-scientific preflight."""

    return _score_b2_forecast(
        catalog=catalog,
        forecast_artifact=forecast_artifact,
        native_truth=native_truth,
        geometry=geometry,
        event_threshold_record=event_threshold_record,
        target_frames=target_frames,
        model_seed=model_seed,
        bounded_smoke=True,
    )


def _score_b2_forecast(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: B2ForecastArtifact,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    model_seed: int,
    bounded_smoke: bool,
) -> dict[str, Any]:
    """Shared full/smoke scorer with an explicit immutable scope switch."""

    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=2,
    )
    required_targets = (
        tuple(range(498, 502)) if bounded_smoke else B2_VALIDATION_TARGETS
    )
    if targets != required_targets:
        purpose = "bounded smoke" if bounded_smoke else "scientific"
        raise ValueError(f"{purpose} B2 scoring target interval differs")
    if forecast_artifact.target_frames != targets:
        raise ValueError("B2 forecast artifact/scorer target frames differ")
    if forecast_artifact.model_seed != int(model_seed):
        raise ValueError("B2 forecast artifact/scorer model seed differs")
    if forecast_artifact.metadata.get("target_truth_read") is not False:
        raise ValueError("B2 forecast metadata does not preserve truth separation")
    thresholds = validate_b2_transport_event_thresholds(event_threshold_record)
    validate_b2_spectral_materiality(
        event_threshold_record.get("spectral_materiality", {})
    )
    masks = geometry.region_masks
    region_masks = b2_region_masks(masks, n_z=88)
    eligible_xy = np.asarray(
        masks.strict_wall_interior & masks.operator_interior,
        dtype=bool,
    )
    validation_blocks = (targets,) if bounded_smoke else B2_VALIDATION_BLOCKS
    field = B2FieldScoreAccumulator(
        model_seed=model_seed,
        target_frames=targets,
        region_masks=region_masks,
        validation_blocks=validation_blocks,
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
        for block in validation_blocks
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
        for block in validation_blocks
    ]
    block_for_target = {
        target: index
        for index, block in enumerate(validation_blocks)
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
            "scope": (
                "bounded_non_scientific_B2_evaluator_smoke_scoring_85604"
                if bounded_smoke
                else "B2_truth_separated_probabilistic_scoring_85604"
            ),
            "bounded_non_scientific_smoke": bool(bounded_smoke),
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
                "chronological_blocks": [block.finalize() for block in spectral_blocks],
            },
            "memberwise_transport": {
                "overall": transport.finalize(),
                "chronological_blocks": [
                    block.finalize() for block in transport_blocks
                ],
            },
        }
    )
