"""Frozen 85604-only scientific scoring for the ECRD model ladder.

This module deliberately reuses the already validated B2 field, spectral,
cross-field, and transport operators.  Its additions are limited to the
three 42-frame ECRD validation blocks and a prospectively frozen sketch of
the full exact-separatrix transport covariance.  Forecast artifacts must be
closed and hash verified before this module is allowed to open target truth.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, b2_region_masks
from .b2_field_scoring import B2FieldScoreAccumulator
from .b2_scoring import (
    decode_b2_member_forecasts,
    validate_b2_spectral_materiality,
    validate_b2_transport_event_thresholds,
)
from .b2_spectral_metrics import B2SpectralAccumulator
from .b2_transport_metrics import (
    B2TransportAccumulator,
    memberwise_transport_outputs,
)
from .b5_covariance_localization import (
    B5_FINITE_MEMBER_FACTOR,
    TransportCovarianceAccumulator,
)
from .codec_transport import TRANSPORT_QUANTITIES, CodecTransportGeometry
from .ecrd_training import ECRD_ARMS, ECRD_VALIDATION_BLOCKS
from .matched_o1_transport import NativeTruthCatalog
from .model_training_data import ModelDatasetCatalog
from .o2_training_data import OneStepWindowDataset, strict_o2_targets


ECRD_EVALUATION_TARGETS = tuple(range(498, 624))
ECRD_EVALUATION_BLOCKS = {
    name: tuple(range(start, stop))
    for name, (start, stop) in ECRD_VALIDATION_BLOCKS.items()
}
ECRD_COVARIANCE_SKETCH_PROBES = 64
ECRD_COVARIANCE_SKETCH_SEED = 85_604_350


class ECRDForecastForScoring(Protocol):
    """Minimum closed-artifact interface consumed by the scorer."""

    path: Any
    sha256: str
    target_frames: tuple[int, ...]
    arm: str
    model_seed: int
    metadata: Mapping[str, Any]

    def read(self, start: int, stop: int) -> np.ndarray:
        ...

    def timing_record(self) -> Mapping[str, Any]:
        ...


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


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


class SpatialTransportCovarianceSketch:
    """Compare predictive and realized local-transport second moments.

    The exact local transport vector has 16 x 81 = 1,296 entries.  Rather
    than materializing a dense matrix for every model, this accumulator uses
    one frozen Rademacher probe bank to estimate the relative Frobenius error
    of

        (33/32) E_t Cov_m(q_t^m)

    against

        E_t [(q_t - mean_m q_t^m)(q_t - mean_m q_t^m)^T].

    The latter is intentionally an uncentered forecast-error second moment:
    deterministic bias is part of probabilistic forecast skill.  Identical
    probes are used for every arm, seed, block, and transport quantity.
    """

    def __init__(
        self,
        *,
        quantities: Sequence[str],
        rows: int,
        n_z: int,
        probes: int = ECRD_COVARIANCE_SKETCH_PROBES,
        seed: int = ECRD_COVARIANCE_SKETCH_SEED,
    ) -> None:
        names = tuple(str(value) for value in quantities)
        if names != TRANSPORT_QUANTITIES:
            raise ValueError("ECRD transport covariance quantity order differs")
        self.quantities = names
        self.rows = int(rows)
        self.n_z = int(n_z)
        self.point_count = self.rows * self.n_z
        self.probe_count = int(probes)
        self.seed = int(seed)
        if self.rows != 16 or self.n_z != 81 or self.probe_count != 64:
            raise ValueError("ECRD transport covariance sketch contract differs")
        generator = np.random.Generator(np.random.PCG64(self.seed))
        signs = generator.integers(
            0,
            2,
            size=(self.point_count, self.probe_count),
            dtype=np.int8,
        )
        self.probe_bank = np.asarray(
            (2.0 * signs - 1.0) / math.sqrt(self.probe_count),
            dtype=np.float64,
        )
        shape = self.probe_bank.shape
        self.predictive_action_sum = {
            name: np.zeros(shape, dtype=np.float64) for name in names
        }
        self.error_action_sum = {
            name: np.zeros(shape, dtype=np.float64) for name in names
        }
        self.target_count = 0

    def update(
        self,
        *,
        forecast: Mapping[str, np.ndarray],
        truth: Mapping[str, np.ndarray],
    ) -> None:
        if tuple(forecast) != self.quantities or tuple(truth) != self.quantities:
            raise ValueError("ECRD covariance-sketch quantity order differs")
        for name in self.quantities:
            members = _finite_real(
                f"{name} local transport members", forecast[name]
            )
            observed = _finite_real(f"{name} local transport truth", truth[name])
            if members.shape != (32, self.rows, self.n_z):
                raise ValueError("ECRD local transport member shape differs")
            if observed.shape != (self.rows, self.n_z):
                raise ValueError("ECRD local transport truth shape differs")
            flattened = members.reshape(32, self.point_count)
            mean = np.mean(flattened, axis=0, dtype=np.float64)
            anomaly = flattened - mean[None]
            covariance_action = (
                anomaly.T @ (anomaly @ self.probe_bank)
            ) / float(flattened.shape[0] - 1)
            error = observed.reshape(self.point_count) - mean
            error_action = error[:, None] * (error @ self.probe_bank)[None]
            self.predictive_action_sum[name] += (
                B5_FINITE_MEMBER_FACTOR * covariance_action
            )
            self.error_action_sum[name] += error_action
        self.target_count += 1

    def finalize(self) -> dict[str, Any]:
        if self.target_count < 1:
            raise RuntimeError("ECRD transport covariance sketch is empty")
        records: dict[str, Any] = {}
        for name in self.quantities:
            predictive = self.predictive_action_sum[name] / self.target_count
            realized = self.error_action_sum[name] / self.target_count
            difference = predictive - realized
            denominator = float(np.linalg.norm(realized))
            predictive_norm = float(np.linalg.norm(predictive))
            difference_norm = float(np.linalg.norm(difference))
            inner = float(np.vdot(predictive, realized).real)
            records[name] = {
                "relative_frobenius_error_sketch": (
                    difference_norm / denominator if denominator > 0.0 else None
                ),
                "predictive_to_realized_frobenius_norm_ratio_sketch": (
                    predictive_norm / denominator if denominator > 0.0 else None
                ),
                "action_cosine_similarity": (
                    inner / (predictive_norm * denominator)
                    if predictive_norm > 0.0 and denominator > 0.0
                    else None
                ),
            }
        return {
            "schema_version": 1,
            "method": "fixed_Rademacher_covariance_action_sketch",
            "target_count": self.target_count,
            "local_shape": [self.rows, self.n_z],
            "point_count": self.point_count,
            "probe_count": self.probe_count,
            "probe_seed": self.seed,
            "finite_member_variance_factor": B5_FINITE_MEMBER_FACTOR,
            "predictive_definition": (
                "33_over_32_times_mean_target_unbiased_member_covariance"
            ),
            "realized_definition": (
                "mean_target_uncentered_outer_product_of_truth_minus_ensemble_mean"
            ),
            "quantities": records,
        }


def exact_local_transport_from_b2_outputs(
    *,
    forecast_outputs: Mapping[str, Mapping[str, np.ndarray]],
    truth_outputs: Mapping[str, Mapping[str, np.ndarray]],
    geometry: CodecTransportGeometry,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Recover exact-separatrix local arrays from validated B2 reductions."""

    if tuple(forecast_outputs) != TRANSPORT_QUANTITIES:
        raise ValueError("ECRD forecast transport quantity order differs")
    if tuple(truth_outputs) != TRANSPORT_QUANTITIES:
        raise ValueError("ECRD truth transport quantity order differs")
    strict = np.asarray(geometry.strict_face_mask, dtype=bool)
    separatrix = np.asarray(geometry.separatrix_face_mask, dtype=bool)
    if strict.ndim != 2 or separatrix.shape != strict.shape:
        raise ValueError("ECRD transport face-mask geometry differs")
    if np.any(separatrix & ~strict):
        raise ValueError("ECRD separatrix mask leaves strict-face support")
    selector = separatrix[strict]
    strict_rows = int(np.sum(strict))
    if int(np.sum(selector)) != 16:
        raise ValueError("ECRD exact separatrix must contain 16 rows")
    forecast_local: dict[str, np.ndarray] = {}
    truth_local: dict[str, np.ndarray] = {}
    maximum_relative_closure = 0.0
    for name in TRANSPORT_QUANTITIES:
        forecast_flat = _finite_real(
            f"{name} strict-face forecast",
            forecast_outputs[name]["strict_face_contributions"],
        )
        truth_flat = _finite_real(
            f"{name} strict-face truth",
            truth_outputs[name]["strict_face_contributions"],
        )
        if forecast_flat.shape != (32, strict_rows * 81):
            raise ValueError("ECRD strict-face forecast shape differs")
        if truth_flat.shape != (strict_rows * 81,):
            raise ValueError("ECRD strict-face truth shape differs")
        member_values = forecast_flat.reshape(32, strict_rows, 81)[:, selector]
        truth_values = truth_flat.reshape(strict_rows, 81)[selector]
        forecast_wedge = _finite_real(
            f"{name} forecast wedge", forecast_outputs[name]["separatrix_wedge"]
        ).reshape(32)
        truth_wedge = _finite_real(
            f"{name} truth wedge", truth_outputs[name]["separatrix_wedge"]
        ).reshape(1)
        reconstructed_forecast = np.sum(member_values, axis=(1, 2), dtype=np.float64)
        reconstructed_truth = float(np.sum(truth_values, dtype=np.float64))
        differences = np.concatenate(
            (
                np.abs(reconstructed_forecast - forecast_wedge),
                np.asarray([abs(reconstructed_truth - float(truth_wedge[0]))]),
            )
        )
        scales = np.concatenate(
            (
                np.maximum.reduce(
                    (
                        np.abs(reconstructed_forecast),
                        np.abs(forecast_wedge),
                        np.ones(32),
                    )
                ),
                np.asarray(
                    [max(abs(reconstructed_truth), abs(float(truth_wedge[0])), 1.0)]
                ),
            )
        )
        maximum_relative_closure = max(
            maximum_relative_closure, float(np.max(differences / scales))
        )
        if not np.allclose(
            reconstructed_forecast, forecast_wedge, rtol=2e-12, atol=1e-12
        ) or not math.isclose(
            reconstructed_truth, float(truth_wedge[0]), rel_tol=2e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"{name} exact-separatrix transport does not close")
        forecast_local[name] = np.ascontiguousarray(member_values)
        truth_local[name] = np.ascontiguousarray(truth_values)
    return forecast_local, truth_local, maximum_relative_closure


def score_ecrd_forecast(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: ECRDForecastForScoring,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
    event_threshold_record: Mapping[str, Any],
    target_frames: Sequence[int],
    arm: str,
    model_seed: int,
) -> dict[str, Any]:
    """Score one closed M32 ECRD artifact on all three 85604 blocks."""

    targets = strict_o2_targets(
        target_frames,
        split="validation",
        context_frames=2 if arm == "ECRD-History" else 1,
    )
    if targets != ECRD_EVALUATION_TARGETS:
        raise ValueError("ECRD scientific evaluation target interval differs")
    if arm not in ECRD_ARMS or forecast_artifact.arm != arm:
        raise ValueError("ECRD scorer arm differs")
    if int(model_seed) not in (1701, 1702, 1703):
        raise ValueError("ECRD scorer model seed differs")
    if (
        forecast_artifact.target_frames != targets
        or forecast_artifact.model_seed != int(model_seed)
    ):
        raise ValueError("ECRD forecast/scorer identity differs")
    if forecast_artifact.metadata.get("target_truth_read") is not False:
        raise ValueError("ECRD forecast metadata does not prove truth separation")
    if forecast_artifact.metadata.get("held_out_85606_read") is not False:
        raise ValueError("ECRD forecast metadata does not preserve the blind holdout")

    thresholds = validate_b2_transport_event_thresholds(event_threshold_record)
    validate_b2_spectral_materiality(
        event_threshold_record.get("spectral_materiality", {})
    )
    validation_blocks = tuple(ECRD_EVALUATION_BLOCKS.values())
    block_for_target = {
        target: index
        for index, block in enumerate(validation_blocks)
        for target in block
    }
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
    covariance = TransportCovarianceAccumulator(
        quantities=TRANSPORT_QUANTITIES,
        rows=16,
        n_z=81,
    )
    covariance_blocks = [
        TransportCovarianceAccumulator(
            quantities=TRANSPORT_QUANTITIES,
            rows=16,
            n_z=81,
        )
        for _ in validation_blocks
    ]
    sketch = SpatialTransportCovarianceSketch(
        quantities=TRANSPORT_QUANTITIES,
        rows=16,
        n_z=81,
    )
    sketch_blocks = [
        SpatialTransportCovarianceSketch(
            quantities=TRANSPORT_QUANTITIES,
            rows=16,
            n_z=81,
        )
        for _ in validation_blocks
    ]
    truth_dataset = OneStepWindowDataset(
        catalog,
        split="validation",
        target_frames=targets,
        context_frames=2,
        augment=False,
        seed=1701,
        return_physical=True,
    )
    maximum_transport_closure = 0.0
    try:
        for position, target in enumerate(targets):
            item = truth_dataset[position]
            if int(item["target_frame_index"]) != target:
                raise RuntimeError("ECRD truth scoring order differs")
            standardized_forecast = forecast_artifact.read(position, position + 1)[0]
            standardized_truth = np.asarray(item["target"], dtype=np.float32)
            physical_truth = np.asarray(item["physical_target"], dtype=np.float64)
            physical_forecast = decode_b2_member_forecasts(
                catalog, standardized_forecast
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
            local_forecast, local_truth, closure = exact_local_transport_from_b2_outputs(
                forecast_outputs=forecast_transport,
                truth_outputs=truth_transport,
                geometry=geometry,
            )
            maximum_transport_closure = max(maximum_transport_closure, closure)
            covariance.update(
                target_frame=target,
                forecast=local_forecast,
                truth=local_truth,
            )
            covariance_blocks[block_index].update(
                target_frame=target,
                forecast=local_forecast,
                truth=local_truth,
            )
            sketch.update(forecast=local_forecast, truth=local_truth)
            sketch_blocks[block_index].update(
                forecast=local_forecast,
                truth=local_truth,
            )
    finally:
        truth_dataset.close()

    covariance_record, _ = covariance.finalize()
    block_covariance_records = []
    for (name, block), accumulator, block_sketch in zip(
        ECRD_EVALUATION_BLOCKS.items(), covariance_blocks, sketch_blocks
    ):
        record, _ = accumulator.finalize()
        block_covariance_records.append(
            {
                "name": name,
                "target_frames": [block[0], block[-1] + 1],
                "local_covariance": record,
                "full_spatial_covariance_sketch": block_sketch.finalize(),
            }
        )
    return _json_safe(
        {
            "schema_version": 1,
            "scope": "ECRD_truth_separated_probabilistic_scoring_85604",
            "development_run": "85604",
            "held_out_85606_read": False,
            "guard_frames_read": False,
            "target_truth_used_during_forecast_generation": False,
            "truth_opened_only_after_forecast_was_closed_and_hash_verified": True,
            "training_performed": False,
            "physics_derived_training_loss_used": False,
            "arm": arm,
            "model_seed": int(model_seed),
            "target_frames": [targets[0], targets[-1] + 1],
            "target_count": len(targets),
            "validation_blocks": {
                name: [block[0], block[-1] + 1]
                for name, block in ECRD_EVALUATION_BLOCKS.items()
            },
            "forecast_artifact": {
                "path": str(forecast_artifact.path),
                "sha256": forecast_artifact.sha256,
                "metadata": dict(forecast_artifact.metadata),
                "timing": dict(forecast_artifact.timing_record()),
            },
            "transport_event_thresholds": dict(event_threshold_record),
            "field_and_marginal_calibration": field.finalize(),
            "spectral_and_cross_field": {
                "overall": spectral.finalize(),
                "chronological_blocks": [
                    {"name": name, "score": accumulator.finalize()}
                    for name, accumulator in zip(
                        ECRD_EVALUATION_BLOCKS, spectral_blocks
                    )
                ],
            },
            "memberwise_transport": {
                "overall": transport.finalize(),
                "chronological_blocks": [
                    {"name": name, "score": accumulator.finalize()}
                    for name, accumulator in zip(
                        ECRD_EVALUATION_BLOCKS, transport_blocks
                    )
                ],
            },
            "spatial_transport_covariance": {
                "overall": {
                    "local_covariance": covariance_record,
                    "full_spatial_covariance_sketch": sketch.finalize(),
                },
                "chronological_blocks": block_covariance_records,
                "maximum_relative_exact_separatrix_closure_error": (
                    maximum_transport_closure
                ),
            },
        }
    )
