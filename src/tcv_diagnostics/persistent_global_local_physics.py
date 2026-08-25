"""Truth-separated 85604 physics scoring for the persistent global--local pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, b2_region_masks
from .b2_field_scoring import B2FieldScoreAccumulator
from .b2_scoring import (
    decode_b2_member_forecasts,
    validate_b2_spectral_materiality,
    validate_b2_transport_event_thresholds,
)
from .b2_spectral_metrics import B2SpectralAccumulator
from .b2_transport_metrics import B2TransportAccumulator, memberwise_transport_outputs
from .b5_covariance_localization import TransportCovarianceAccumulator
from .codec_transport import TRANSPORT_QUANTITIES, CodecTransportGeometry
from .ecrd_scoring import (
    SpatialTransportCovarianceSketch,
    exact_local_transport_from_b2_outputs,
)
from .matched_o1_transport import NativeTruthCatalog
from .model_training_data import ModelDatasetCatalog
from .o2_training_data import OneStepWindowDataset
from .persistent_global_local_forecast import (
    PGL_EVALUATION_BLOCKS,
    PGL_EVALUATION_STARTS,
    PGLForecastArtifact,
)
from .persistent_global_local_gates import evaluate_pgl_physics_gates


PGL_EVALUATION_HORIZONS = (1, 2, 3, 4)
PGL_PRIMARY_HORIZONS = (1, 4)
PGL_BOOTSTRAP_REPLICATES = 2000
PGL_BOOTSTRAP_SEED = 85_604_405
PGL_BOOTSTRAP_BLOCK_LENGTH = 3


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


def _targets(horizon: int) -> tuple[int, ...]:
    step = int(horizon)
    if step not in PGL_EVALUATION_HORIZONS:
        raise ValueError("persistent evaluation horizon differs")
    return tuple(start + step for start in PGL_EVALUATION_STARTS)


def _target_blocks(horizon: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(start + int(horizon) for start in starts)
        for starts in PGL_EVALUATION_BLOCKS.values()
    )


def _replicate_mean(mean: np.ndarray) -> np.ndarray:
    values = np.asarray(mean, dtype=np.float32)
    if values.shape != (len(B2_FIELDS), 64, 32, 88):
        raise ValueError("persistent deterministic mean shape differs")
    return np.ascontiguousarray(np.broadcast_to(values, (32, *values.shape)))


def _field_accumulator(
    *, horizon: int, region_masks: Mapping[str, np.ndarray]
) -> B2FieldScoreAccumulator:
    return B2FieldScoreAccumulator(
        model_seed=1702,
        target_frames=_targets(horizon),
        region_masks=region_masks,
        validation_blocks=_target_blocks(horizon),
        allow_sparse_targets=True,
    )


def _spectral_accumulator(
    *, horizon: int, eligible_xy: np.ndarray
) -> B2SpectralAccumulator:
    return B2SpectralAccumulator(
        model_seed=1702,
        target_frames=_targets(horizon),
        eligible_xy_mask=eligible_xy,
        allow_sparse_targets=True,
    )


def _transport_accumulator(
    *, horizon: int, thresholds: Mapping[str, float], detailed: bool
) -> B2TransportAccumulator:
    return B2TransportAccumulator(
        model_seed=1702,
        target_frames=_targets(horizon),
        event_thresholds=thresholds,
        detailed=detailed,
        allow_sparse_targets=True,
    )


def selected_start_block_bootstrap_indices(
    *, replicates: int = PGL_BOOTSTRAP_REPLICATES, seed: int = PGL_BOOTSTRAP_SEED
) -> np.ndarray:
    """Frozen paired noncircular three-start bootstrap within V00/V01/V02."""

    if int(replicates) != PGL_BOOTSTRAP_REPLICATES or int(seed) != PGL_BOOTSTRAP_SEED:
        raise ValueError("persistent bootstrap configuration differs")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    result = np.empty((int(replicates), len(PGL_EVALUATION_STARTS)), dtype=np.int64)
    for replicate in range(int(replicates)):
        cursor = 0
        for block_index in range(3):
            offset = 12 * block_index
            candidates = generator.integers(0, 10, size=4)
            selected = np.concatenate(
                [np.arange(value, value + 3, dtype=np.int64) for value in candidates]
            )[:12]
            result[replicate, cursor : cursor + 12] = selected + offset
            cursor += 12
    return result


def _interval(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("persistent bootstrap statistic is invalid")
    return {
        "median": float(np.quantile(array, 0.5, method="linear")),
        "lower_2p5": float(np.quantile(array, 0.025, method="linear")),
        "upper_97p5": float(np.quantile(array, 0.975, method="linear")),
    }


def _field_bootstrap(
    candidate: Mapping[str, Any], selected: Mapping[str, Any], indices: np.ndarray
) -> dict[str, Any]:
    candidate_targets = candidate["per_target_eligible_union_sufficient_statistics"]
    selected_targets = selected["per_target_eligible_union_sufficient_statistics"]
    fair_equal = np.zeros(indices.shape[0], dtype=np.float64)
    mae_equal = np.zeros(indices.shape[0], dtype=np.float64)
    maximum_spread = np.zeros(indices.shape[0], dtype=np.float64)
    for field in B2_FIELDS:
        candidate_field = [item["fields"][field] for item in candidate_targets]
        selected_field = [item["fields"][field] for item in selected_targets]
        count = np.asarray([item["count"] for item in candidate_field], dtype=np.float64)
        fair = np.asarray([item["fair_crps_sum"] for item in candidate_field])
        squared = np.asarray([item["squared_error_sum"] for item in candidate_field])
        variance = np.asarray([item["member_variance_sum"] for item in candidate_field])
        selected_abs = np.asarray(
            [item["absolute_error_sum"] for item in selected_field], dtype=np.float64
        )
        sampled_count = np.sum(count[indices], axis=1)
        fair_equal += np.sum(fair[indices], axis=1) / sampled_count / len(B2_FIELDS)
        mae_equal += (
            np.sum(selected_abs[indices], axis=1) / sampled_count / len(B2_FIELDS)
        )
        rmse = np.sqrt(np.sum(squared[indices], axis=1) / sampled_count)
        spread = np.sqrt((33.0 / 32.0) * np.sum(variance[indices], axis=1) / sampled_count)
        maximum_spread = np.maximum(maximum_spread, spread / rmse)
    return {
        "candidate_equal_field_fair_CRPS": _interval(fair_equal),
        "selected_mean_equal_field_MAE": _interval(mae_equal),
        "candidate_minus_selected": _interval(fair_equal - mae_equal),
        "maximum_per_field_corrected_spread_skill": _interval(maximum_spread),
    }


def _transport_bootstrap(
    covariance: Mapping[str, Any],
    candidate_transport: Mapping[str, Any],
    parent_transport: Mapping[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    local_ratios = []
    integrated_ratios = []
    candidate_l2 = []
    parent_l2 = []
    covariance_targets = covariance["per_target"]
    candidate_targets = candidate_transport["per_target"]
    parent_targets = parent_transport["per_target"]
    for quantity in TRANSPORT_QUANTITIES:
        scalars = [item["quantities"][quantity] for item in covariance_targets]
        ensemble_local = np.asarray(
            [item["ensemble_diagonal_variance_sum"] for item in scalars]
        )
        ensemble_integrated = np.asarray(
            [item["ensemble_integrated_variance"] for item in scalars]
        )
        error_local = np.asarray(
            [item["error_diagonal_squared_sum"] for item in scalars]
        )
        error_integrated = np.asarray(
            [item["error_integrated_squared"] for item in scalars]
        )
        local_ratios.append(
            np.sqrt(
                (33.0 / 32.0)
                * np.sum(ensemble_local[indices], axis=1)
                / np.sum(error_local[indices], axis=1)
            )
        )
        integrated_ratios.append(
            np.sqrt(
                (33.0 / 32.0)
                * np.sum(ensemble_integrated[indices], axis=1)
                / np.sum(error_integrated[indices], axis=1)
            )
        )
        candidate_records = [
            item["quantities"][quantity]["separatrix_wedge"]
            for item in candidate_targets
        ]
        parent_records = [
            item["quantities"][quantity]["separatrix_wedge"]
            for item in parent_targets
        ]
        truth_energy = np.asarray(
            [item["truth_rms"] ** 2 * item["point_count"] for item in candidate_records]
        )
        candidate_error = np.asarray(
            [item["ensemble_mean_relative_l2"] ** 2 for item in candidate_records]
        ) * truth_energy
        parent_error = np.asarray(
            [item["ensemble_mean_relative_l2"] ** 2 for item in parent_records]
        ) * truth_energy
        denominator = np.sum(truth_energy[indices], axis=1)
        candidate_l2.append(np.sqrt(np.sum(candidate_error[indices], axis=1) / denominator))
        parent_l2.append(np.sqrt(np.sum(parent_error[indices], axis=1) / denominator))
    local_array = np.stack(local_ratios, axis=1)
    integrated_array = np.stack(integrated_ratios, axis=1)
    candidate_array = np.stack(candidate_l2, axis=1)
    parent_array = np.stack(parent_l2, axis=1)
    return {
        "median_local_corrected_spread_skill": _interval(np.median(local_array, axis=1)),
        "median_integrated_corrected_spread_skill": _interval(
            np.median(integrated_array, axis=1)
        ),
        "median_integrated_relative_L2_candidate_over_parent": _interval(
            np.median(candidate_array, axis=1) / np.median(parent_array, axis=1)
        ),
    }


def score_persistent_global_local_forecast(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: PGLForecastArtifact,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Open truth only after the forecast is closed and hash verified."""

    thresholds = validate_b2_transport_event_thresholds(event_threshold_record)
    validate_b2_spectral_materiality(event_threshold_record.get("spectral_materiality", {}))
    masks = geometry.region_masks
    regions = b2_region_masks(masks, n_z=88)
    eligible_xy = np.asarray(masks.strict_wall_interior & masks.operator_interior, dtype=bool)
    candidate: dict[str, Any] = {}
    selected_fields: dict[int, Mapping[str, Any]] = {}
    parent_fields: dict[int, Mapping[str, Any]] = {}
    parent_h4_spectral: Mapping[str, Any] | None = None
    parent_h4_transport: Mapping[str, Any] | None = None
    h4_covariance_record: Mapping[str, Any] | None = None
    h4_sketch_record: Mapping[str, Any] | None = None
    maximum_closure = 0.0

    for horizon in PGL_EVALUATION_HORIZONS:
        targets = _targets(horizon)
        candidate_field = _field_accumulator(horizon=horizon, region_masks=regions)
        candidate_spectral = _spectral_accumulator(horizon=horizon, eligible_xy=eligible_xy)
        candidate_transport = _transport_accumulator(
            horizon=horizon, thresholds=thresholds, detailed=horizon == 4
        )
        selected_field = (
            _field_accumulator(horizon=horizon, region_masks=regions)
            if horizon in PGL_PRIMARY_HORIZONS
            else None
        )
        parent_field = (
            _field_accumulator(horizon=horizon, region_masks=regions)
            if horizon in PGL_PRIMARY_HORIZONS
            else None
        )
        parent_spectral = (
            _spectral_accumulator(horizon=horizon, eligible_xy=eligible_xy)
            if horizon == 4
            else None
        )
        parent_transport = (
            _transport_accumulator(horizon=horizon, thresholds=thresholds, detailed=True)
            if horizon == 4
            else None
        )
        covariance = (
            TransportCovarianceAccumulator(quantities=TRANSPORT_QUANTITIES, rows=16, n_z=81)
            if horizon == 4
            else None
        )
        sketch = (
            SpatialTransportCovarianceSketch(
                quantities=TRANSPORT_QUANTITIES, rows=16, n_z=81
            )
            if horizon == 4
            else None
        )
        truth_dataset = OneStepWindowDataset(
            catalog,
            split="validation",
            target_frames=targets,
            context_frames=1,
            augment=False,
            seed=1702,
            return_physical=True,
            allow_sparse_targets=True,
        )
        try:
            for position, target in enumerate(targets):
                item = truth_dataset[position]
                if int(item["target_frame_index"]) != target:
                    raise RuntimeError("persistent truth target order differs")
                truth_standardized = np.asarray(item["target"], dtype=np.float32)
                truth_physical = np.asarray(item["physical_target"], dtype=np.float64)
                forecast_standardized = forecast_artifact.read_forecast_horizon(
                    position, horizon
                )[:, None]
                selected_standardized = _replicate_mean(
                    forecast_artifact.read_mean_horizon(
                        position, horizon, parent=False
                    )
                )[:, None]
                parent_standardized = _replicate_mean(
                    forecast_artifact.read_mean_horizon(position, horizon, parent=True)
                )[:, None]
                candidate_field.update(
                    target_frame=target,
                    standardized_forecast=forecast_standardized,
                    standardized_truth=truth_standardized,
                )
                if selected_field is not None:
                    selected_field.update(
                        target_frame=target,
                        standardized_forecast=selected_standardized,
                        standardized_truth=truth_standardized,
                    )
                if parent_field is not None:
                    parent_field.update(
                        target_frame=target,
                        standardized_forecast=parent_standardized,
                        standardized_truth=truth_standardized,
                    )
                forecast_physical = decode_b2_member_forecasts(
                    catalog, forecast_standardized
                )
                candidate_spectral.update(
                    target_frame=target,
                    physical_forecast=forecast_physical,
                    physical_truth=truth_physical,
                )
                parent_physical = None
                if parent_spectral is not None:
                    parent_physical = decode_b2_member_forecasts(
                        catalog, parent_standardized
                    )
                    parent_spectral.update(
                        target_frame=target,
                        physical_forecast=parent_physical,
                        physical_truth=truth_physical,
                    )
                truth_native = native_truth.read(
                    target, target + 1, fields=("Ne", "Pe", "Pi", "phi")
                )
                forecast_transport, truth_transport = memberwise_transport_outputs(
                    physical_forecast_model88=forecast_physical,
                    native_truth=truth_native,
                    geometry=geometry,
                )
                candidate_transport.update(
                    target_frame=target,
                    forecast_outputs=forecast_transport,
                    truth_outputs=truth_transport,
                )
                if covariance is not None and sketch is not None:
                    local_forecast, local_truth, closure = exact_local_transport_from_b2_outputs(
                        forecast_outputs=forecast_transport,
                        truth_outputs=truth_transport,
                        geometry=geometry,
                    )
                    maximum_closure = max(maximum_closure, closure)
                    covariance.update(
                        target_frame=target,
                        forecast=local_forecast,
                        truth=local_truth,
                    )
                    sketch.update(forecast=local_forecast, truth=local_truth)
                if parent_transport is not None:
                    if parent_physical is None:
                        raise AssertionError("persistent parent physical fields are absent")
                    parent_outputs, parent_truth = memberwise_transport_outputs(
                        physical_forecast_model88=parent_physical,
                        native_truth=truth_native,
                        geometry=geometry,
                    )
                    parent_transport.update(
                        target_frame=target,
                        forecast_outputs=parent_outputs,
                        truth_outputs=parent_truth,
                    )
        finally:
            truth_dataset.close()

        candidate[str(horizon)] = {
            "target_frames": list(targets),
            "field_and_marginal_calibration": candidate_field.finalize(),
            "spectral_and_cross_field": candidate_spectral.finalize(),
            "memberwise_transport": candidate_transport.finalize(),
        }
        if selected_field is not None:
            selected_fields[horizon] = selected_field.finalize()
        if parent_field is not None:
            parent_fields[horizon] = parent_field.finalize()
        if parent_spectral is not None:
            parent_h4_spectral = parent_spectral.finalize()
        if parent_transport is not None:
            parent_h4_transport = parent_transport.finalize()
        if covariance is not None and sketch is not None:
            h4_covariance_record, _ = covariance.finalize()
            h4_sketch_record = sketch.finalize()

    if any(
        value is None
        for value in (
            parent_h4_spectral,
            parent_h4_transport,
            h4_covariance_record,
            h4_sketch_record,
        )
    ):
        raise RuntimeError("persistent primary physics records are incomplete")
    gate = evaluate_pgl_physics_gates(
        candidate_h1_field=candidate["1"]["field_and_marginal_calibration"],
        candidate_h4_field=candidate["4"]["field_and_marginal_calibration"],
        selected_h1_field=selected_fields[1],
        selected_h4_field=selected_fields[4],
        candidate_h4_spectral=candidate["4"]["spectral_and_cross_field"],
        parent_h4_spectral=parent_h4_spectral,
        candidate_h4_transport=candidate["4"]["memberwise_transport"],
        parent_h4_transport=parent_h4_transport,
        candidate_h4_covariance=h4_covariance_record,
        candidate_h4_spatial_sketch=h4_sketch_record,
    )
    bootstrap_indices = selected_start_block_bootstrap_indices()
    bootstrap = {
        "method": "paired_non_circular_selected_start_block_bootstrap",
        "block_length_selected_starts": 3,
        "replicates": 2000,
        "seed": PGL_BOOTSTRAP_SEED,
        "resample_each_chronological_block_separately": True,
        "conditional_on_single_85604_run": True,
        "used_as_pass_fail_gate": False,
        "field": {
            f"h{horizon}": _field_bootstrap(
                candidate[str(horizon)]["field_and_marginal_calibration"],
                selected_fields[horizon],
                bootstrap_indices,
            )
            for horizon in PGL_PRIMARY_HORIZONS
        },
        "transport_h4": _transport_bootstrap(
            h4_covariance_record,
            candidate["4"]["memberwise_transport"],
            parent_h4_transport,
            bootstrap_indices,
        ),
        "spectral_and_spatial_covariance_intervals": {
            "reported": False,
            "reason": (
                "the frozen nonlinear pooled spectral and covariance-action point gates "
                "are retained; no proxy interval is substituted"
            ),
        },
    }
    return _json_safe(
        {
            "schema_version": 1,
            "scope": "old_85604_persistent_global_local_truth_separated_physics_scoring",
            "status": "completed",
            "development_run": "85604",
            "current_frames": list(PGL_EVALUATION_STARTS),
            "horizons_frames": list(PGL_EVALUATION_HORIZONS),
            "ensemble_members": 32,
            "candidate": candidate,
            "deterministic_baselines": {
                "selected_mean_field": {
                    str(horizon): record for horizon, record in selected_fields.items()
                },
                "frozen_parent_field": {
                    str(horizon): record for horizon, record in parent_fields.items()
                },
                "frozen_parent_h4_spectral": parent_h4_spectral,
                "frozen_parent_h4_transport": parent_h4_transport,
            },
            "h4_spatial_transport_covariance": {
                "local_covariance": h4_covariance_record,
                "full_spatial_covariance_sketch": h4_sketch_record,
                "maximum_relative_exact_separatrix_closure_error": maximum_closure,
            },
            "gate": gate,
            "conditional_uncertainty": bootstrap,
            "truth_opened_only_after_forecast_closed_and_hash_verified": True,
            "target_truth_used_during_generation": False,
            "nonlinear_transport_applied_memberwise": True,
            "unmodified_physical_phi_used_for_transport": True,
            "physics_derived_training_loss_used": False,
            "guard_frames_read": False,
            "held_out_85606_read": False,
            "new_nersc_data_read": False,
            "assimilation_performed": False,
            "diagnostic_ranking_performed": False,
            "steering_performed": False,
        }
    )
