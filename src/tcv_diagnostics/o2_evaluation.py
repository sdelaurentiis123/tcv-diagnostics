"""Frozen deterministic one-step evaluation primitives for Paper 0 O2.

The functions in this module are deliberately independent of checkpoint and
file I/O.  They turn already-generated C5P forecasts into field, spectral,
cross-field, and transport records and apply the thresholds frozen before O2
training.  Physics-derived quantities are evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .codec_transport import TRANSPORT_QUANTITIES
from .matched_codec_metrics import (
    CodecViewSpec,
    MatchedCodecAccumulator,
    training_materiality,
)
from .resampling import finalize_paired_statistics


O2_FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
O2_CROSS_PAIRS = (("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi"))
O2_VIEW = CodecViewSpec(
    name="C5P_O2_common",
    fields=O2_FIELDS,
    spectral_fields=O2_FIELDS,
    cross_pairs=O2_CROSS_PAIRS,
)
O2_VALIDATION_TARGETS = (498, 624)
O2_VALIDATION_BLOCKS = 6
O2_BLOCK_FRAMES = 21
O2_CADENCE_MICROSECONDS = 3.131905426352636

O2_THRESHOLDS = {
    "minimum_fields_beating_persistence": 4,
    "maximum_field_persistence_rmse_ratio": 1.05,
    "spectral_power_ratio": (0.75, 1.30),
    "forecast_truth_coherence_min": 0.80,
    "cross_phase_error_degrees_max": 20.0,
    "cross_coherence_change_max": 0.15,
    "strict_faces": {
        "relative_l2_max": 0.40,
        "pearson_correlation_min": 0.70,
        "weighted_sign_disagreement_max": 0.20,
    },
    "separatrix": {
        "relative_l2_max": 0.30,
        "absolute_normalized_bias_max": 0.15,
        "pearson_correlation_min": 0.80,
        "weighted_sign_disagreement_max": 0.15,
    },
    "required_passing_blocks": 5,
}


def _finite_real(name: str, values: np.ndarray, *, ndim: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} axes")
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def validation_blocks(target_frames: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return the six frozen chronological 21-target O2 blocks."""

    targets = tuple(int(frame) for frame in target_frames)
    if targets != tuple(range(*O2_VALIDATION_TARGETS)):
        raise ValueError("O2 evaluation requires targets 498..623 exactly once")
    return tuple(
        targets[start : start + O2_BLOCK_FRAMES]
        for start in range(0, len(targets), O2_BLOCK_FRAMES)
    )


@dataclass
class O2FieldAccumulator:
    """Mergeable standardized field-error and training-anomaly statistics.

    Anomalies are defined relative to the training-only normalization mean.
    Consequently the standardized input arrays are already anomaly fields;
    validation means are not subtracted when computing anomaly correlation.
    """

    count_per_field: int
    error_sum: np.ndarray
    absolute_error_sum: np.ndarray
    squared_error_sum: np.ndarray
    truth_sum: np.ndarray
    forecast_sum: np.ndarray
    truth_squared_sum: np.ndarray
    forecast_squared_sum: np.ndarray
    truth_forecast_sum: np.ndarray

    @classmethod
    def empty(cls) -> "O2FieldAccumulator":
        zeros = lambda: np.zeros(len(O2_FIELDS), dtype=np.float64)
        return cls(0, zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros())

    def update(self, truth: np.ndarray, forecast: np.ndarray) -> None:
        truth_array = _finite_real("standardized truth", truth, ndim=5)
        forecast_array = _finite_real("standardized forecast", forecast, ndim=5)
        if truth_array.shape != forecast_array.shape:
            raise ValueError("field truth and forecast shapes differ")
        if truth_array.shape[1] != len(O2_FIELDS):
            raise ValueError("O2 field arrays must contain five C5P channels")
        axes = (0, 2, 3, 4)
        count = int(np.prod([truth_array.shape[axis] for axis in axes]))
        error = forecast_array - truth_array
        self.count_per_field += count
        self.error_sum += np.sum(error, axis=axes, dtype=np.float64)
        self.absolute_error_sum += np.sum(np.abs(error), axis=axes, dtype=np.float64)
        self.squared_error_sum += np.sum(error * error, axis=axes, dtype=np.float64)
        self.truth_sum += np.sum(truth_array, axis=axes, dtype=np.float64)
        self.forecast_sum += np.sum(forecast_array, axis=axes, dtype=np.float64)
        self.truth_squared_sum += np.sum(
            truth_array * truth_array, axis=axes, dtype=np.float64
        )
        self.forecast_squared_sum += np.sum(
            forecast_array * forecast_array, axis=axes, dtype=np.float64
        )
        self.truth_forecast_sum += np.sum(
            truth_array * forecast_array, axis=axes, dtype=np.float64
        )

    def merge(self, other: "O2FieldAccumulator") -> None:
        self.count_per_field += int(other.count_per_field)
        for name in (
            "error_sum",
            "absolute_error_sum",
            "squared_error_sum",
            "truth_sum",
            "forecast_sum",
            "truth_squared_sum",
            "forecast_squared_sum",
            "truth_forecast_sum",
        ):
            getattr(self, name)[:] += getattr(other, name)

    def finalize(self) -> dict[str, Any]:
        if self.count_per_field <= 0:
            raise ValueError("cannot finalize empty O2 field metrics")
        count = float(self.count_per_field)
        truth_mean = self.truth_sum / count
        forecast_mean = self.forecast_sum / count
        truth_variance = np.maximum(
            self.truth_squared_sum / count - truth_mean * truth_mean, 0.0
        )
        forecast_variance = np.maximum(
            self.forecast_squared_sum / count - forecast_mean * forecast_mean,
            0.0,
        )
        rmse = np.sqrt(self.squared_error_sum / count)
        mae = self.absolute_error_sum / count
        bias = self.error_sum / count
        variance_ratio = np.full(len(O2_FIELDS), np.nan, dtype=np.float64)
        np.divide(
            forecast_variance,
            truth_variance,
            out=variance_ratio,
            where=truth_variance > 0.0,
        )
        anomaly_denominator = np.sqrt(
            self.truth_squared_sum * self.forecast_squared_sum
        )
        anomaly_correlation = np.full(len(O2_FIELDS), np.nan, dtype=np.float64)
        np.divide(
            self.truth_forecast_sum,
            anomaly_denominator,
            out=anomaly_correlation,
            where=anomaly_denominator > 0.0,
        )
        metrics = {
            field: {
                "rmse": float(rmse[index]),
                "mae": float(mae[index]),
                "bias": float(bias[index]),
                "truth_mean": float(truth_mean[index]),
                "forecast_mean": float(forecast_mean[index]),
                "truth_variance": float(truth_variance[index]),
                "forecast_variance": float(forecast_variance[index]),
                "variance_ratio": (
                    float(variance_ratio[index])
                    if np.isfinite(variance_ratio[index])
                    else None
                ),
                "anomaly_correlation": (
                    float(anomaly_correlation[index])
                    if np.isfinite(anomaly_correlation[index])
                    else None
                ),
            }
            for index, field in enumerate(O2_FIELDS)
        }
        total_count = count * len(O2_FIELDS)
        return {
            "anomaly_definition": (
                "standardized_field_relative_to_training_only_normalization_mean"
            ),
            "spatial_samples_per_field": int(self.count_per_field),
            "field_metrics_standardized": metrics,
            "aggregate_equal_channel_rmse_standardized": math.sqrt(
                float(np.sum(self.squared_error_sum)) / total_count
            ),
            "aggregate_equal_channel_mae_standardized": (
                float(np.sum(self.absolute_error_sum)) / total_count
            ),
        }


def _complex_curve(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values, dtype=np.complex128)
    return {
        "real": np.real(array).astype(np.float64).tolist(),
        "imaginary": np.imag(array).astype(np.float64).tolist(),
    }


def _optional_curve(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _normalized_complex_correlation(
    cross: np.ndarray,
    first_auto: np.ndarray,
    second_auto: np.ndarray,
) -> np.ndarray:
    denominator = np.sqrt(first_auto * second_auto)
    output = np.full(cross.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    np.divide(cross, denominator, out=output, where=denominator > 0.0)
    return output


def _efolding_frames(magnitude: np.ndarray) -> np.ndarray:
    """Convert one-lag correlation magnitude to an exponential e-folding time.

    Values outside the open interval (0, 1) have no finite exponential
    e-folding estimate and are recorded as undefined rather than clipped.
    """

    values = np.asarray(magnitude, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = (values > 0.0) & (values < 1.0)
    result[valid] = -1.0 / np.log(values[valid])
    return result


class O2MetricAccumulator:
    """Stream O2 field, spectrum, cross-field, and one-lag metrics."""

    def __init__(self, *, n_z: int = 88, zperiod: int = 5) -> None:
        self.fields = O2FieldAccumulator.empty()
        self.spectra = MatchedCodecAccumulator(
            spec=O2_VIEW,
            n_z=n_z,
            zperiod=zperiod,
        )
        self.n_z = int(n_z)
        self.n_modes = self.n_z // 2 + 1
        shape = (len(O2_FIELDS), self.n_modes)
        self.latest_auto = np.zeros(shape, dtype=np.float64)
        self.truth_auto = np.zeros(shape, dtype=np.float64)
        self.forecast_auto = np.zeros(shape, dtype=np.float64)
        self.truth_lag_cross = np.zeros(shape, dtype=np.complex128)
        self.forecast_lag_cross = np.zeros(shape, dtype=np.complex128)
        self.frames = 0

    def update(
        self,
        *,
        standardized_truth: np.ndarray,
        standardized_forecast: np.ndarray,
        physical_truth: np.ndarray,
        physical_forecast: np.ndarray,
        physical_latest_context: np.ndarray,
    ) -> None:
        standardized_truth = _finite_real(
            "standardized truth", standardized_truth, ndim=5
        )
        standardized_forecast = _finite_real(
            "standardized forecast", standardized_forecast, ndim=5
        )
        physical_truth = _finite_real("physical truth", physical_truth, ndim=5)
        physical_forecast = _finite_real(
            "physical forecast", physical_forecast, ndim=5
        )
        physical_latest_context = _finite_real(
            "physical latest context", physical_latest_context, ndim=5
        )
        shapes = {
            tuple(array.shape)
            for array in (
                standardized_truth,
                standardized_forecast,
                physical_truth,
                physical_forecast,
                physical_latest_context,
            )
        }
        if len(shapes) != 1:
            raise ValueError("O2 metric arrays have different shapes")
        shape = next(iter(shapes))
        if shape[1:] != (len(O2_FIELDS), 64, 32, self.n_z):
            raise ValueError("O2 metric array shape differs from [T,5,64,32,Z]")

        self.fields.update(standardized_truth, standardized_forecast)
        self.spectra.update(
            standardized_truth,
            standardized_forecast,
            physical_truth,
            physical_forecast,
        )
        latest = np.fft.rfft(physical_latest_context, axis=-1)
        truth = np.fft.rfft(physical_truth, axis=-1)
        forecast = np.fft.rfft(physical_forecast, axis=-1)
        axes = (0, 2, 3)
        self.latest_auto += np.sum(np.abs(latest) ** 2, axis=axes, dtype=np.float64)
        self.truth_auto += np.sum(np.abs(truth) ** 2, axis=axes, dtype=np.float64)
        self.forecast_auto += np.sum(
            np.abs(forecast) ** 2, axis=axes, dtype=np.float64
        )
        self.truth_lag_cross += np.sum(
            truth * np.conjugate(latest), axis=axes, dtype=np.complex128
        )
        self.forecast_lag_cross += np.sum(
            forecast * np.conjugate(latest), axis=axes, dtype=np.complex128
        )
        self.frames += int(shape[0])

    def finalize(self) -> dict[str, Any]:
        if self.frames <= 0:
            raise ValueError("cannot finalize empty O2 metrics")
        field = self.fields.finalize()
        spectral = self.spectra.finalize()
        truth_rho = _normalized_complex_correlation(
            self.truth_lag_cross, self.truth_auto, self.latest_auto
        )
        forecast_rho = _normalized_complex_correlation(
            self.forecast_lag_cross, self.forecast_auto, self.latest_auto
        )
        truth_magnitude = np.abs(truth_rho)
        forecast_magnitude = np.abs(forecast_rho)
        truth_lifetime = _efolding_frames(truth_magnitude)
        forecast_lifetime = _efolding_frames(forecast_magnitude)
        lifetime = {}
        for index, name in enumerate(O2_FIELDS):
            lifetime[name] = {
                "truth_normalized_complex_lag_correlation": _complex_curve(
                    truth_rho[index]
                ),
                "forecast_normalized_complex_lag_correlation": _complex_curve(
                    forecast_rho[index]
                ),
                "truth_magnitude": _optional_curve(truth_magnitude[index]),
                "forecast_magnitude": _optional_curve(forecast_magnitude[index]),
                "truth_efolding_frames": _optional_curve(truth_lifetime[index]),
                "forecast_efolding_frames": _optional_curve(
                    forecast_lifetime[index]
                ),
                "truth_efolding_microseconds": _optional_curve(
                    truth_lifetime[index] * O2_CADENCE_MICROSECONDS
                ),
                "forecast_efolding_microseconds": _optional_curve(
                    forecast_lifetime[index] * O2_CADENCE_MICROSECONDS
                ),
            }
        return {
            **field,
            "frames": self.frames,
            "spectral_and_cross_field": spectral,
            "one_step_mode_lifetime": {
                "definition": (
                    "negative_inverse_log_of_one_lag_normalized_complex_"
                    "correlation_magnitude_when_strictly_between_zero_and_one"
                ),
                "descriptive_only_not_used_for_O2_gate": True,
                "cadence_microseconds": O2_CADENCE_MICROSECONDS,
                "fields": lifetime,
            },
        }


def o2_training_materiality(training_truth_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the inherited O1 1% material-band rule under an O2 label."""

    base = training_materiality(training_truth_metrics)
    return {
        **base,
        "used_by": "O2_spectral_and_cross_field_gate",
        "validation_truth_used_to_select_bands": False,
    }


def _transport_metrics(
    summary: Mapping[str, Any],
    quantity: str,
    reduction: str,
) -> Mapping[str, Any]:
    comparison = summary["comparisons"]["truth_vs_forecast"]
    return comparison["quantities"][quantity][reduction]["metrics"]


def _transport_reduction_pass(
    metrics: Mapping[str, Any],
    *,
    reduction: str,
) -> tuple[bool, dict[str, bool]]:
    thresholds = O2_THRESHOLDS[reduction]
    correlation = metrics["pearson_correlation"]
    sign = metrics["weighted_sign_disagreement"]
    criteria = {
        "relative_l2": bool(
            math.isfinite(float(metrics["relative_l2"]))
            and float(metrics["relative_l2"]) <= thresholds["relative_l2_max"]
        ),
        "pearson_correlation": bool(
            correlation is not None
            and math.isfinite(float(correlation))
            and float(correlation) >= thresholds["pearson_correlation_min"]
        ),
        "weighted_sign_disagreement": bool(
            sign is not None
            and math.isfinite(float(sign))
            and float(sign) <= thresholds["weighted_sign_disagreement_max"]
        ),
    }
    if reduction == "separatrix":
        criteria["absolute_normalized_bias"] = bool(
            math.isfinite(float(metrics["normalized_bias"]))
            and abs(float(metrics["normalized_bias"]))
            <= thresholds["absolute_normalized_bias_max"]
        )
    return bool(all(criteria.values())), criteria


def build_o2_transport_gate(
    *,
    overall: Mapping[str, Any],
    temporal_blocks: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply O2 transport thresholds overall and in at least five blocks."""

    if len(temporal_blocks) != O2_VALIDATION_BLOCKS:
        raise ValueError("O2 transport gate requires six temporal blocks")
    quantities: dict[str, Any] = {}
    required_blocks = int(O2_THRESHOLDS["required_passing_blocks"])
    for quantity in TRANSPORT_QUANTITIES:
        reductions = {}
        for reduction in ("strict_faces", "separatrix"):
            overall_metrics = _transport_metrics(overall, quantity, reduction)
            overall_pass, overall_criteria = _transport_reduction_pass(
                overall_metrics, reduction=reduction
            )
            block_records = []
            for index, block in enumerate(temporal_blocks):
                metrics = _transport_metrics(block, quantity, reduction)
                passes, criteria = _transport_reduction_pass(
                    metrics, reduction=reduction
                )
                block_records.append(
                    {"block_index": index, "passes": passes, "criteria": criteria}
                )
            passing_blocks = sum(item["passes"] for item in block_records)
            reductions[reduction] = {
                "overall_pass": overall_pass,
                "overall_criteria": overall_criteria,
                "passing_blocks": passing_blocks,
                "required_passing_blocks": required_blocks,
                "block_pass": [item["passes"] for item in block_records],
                "blocks_pass": passing_blocks >= required_blocks,
                "passes": overall_pass and passing_blocks >= required_blocks,
            }
        quantities[quantity] = {
            **reductions,
            "passes": all(item["passes"] for item in reductions.values()),
        }
    passes = all(item["passes"] for item in quantities.values())
    return {
        "thresholds": {
            key: (
                {nested: value for nested, value in threshold.items()}
                if isinstance(threshold, dict)
                else threshold
            )
            for key, threshold in O2_THRESHOLDS.items()
            if key in {"strict_faces", "separatrix", "required_passing_blocks"}
        },
        "quantities": quantities,
        "passes": passes,
        "status": "pass" if passes else "fail",
    }


def reference_skill_gate(
    *,
    candidate: Mapping[str, Any],
    candidate_blocks: list[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    reference_blocks: Mapping[str, list[Mapping[str, Any]]],
    applicable_references: tuple[str, ...],
) -> dict[str, Any]:
    """Apply aggregate-reference and per-field persistence requirements."""

    if "persistence" not in applicable_references:
        raise ValueError("persistence must be an applicable O2 reference")
    if set(applicable_references) - set(references):
        raise ValueError("an applicable O2 reference is missing")
    if len(candidate_blocks) != O2_VALIDATION_BLOCKS:
        raise ValueError("candidate requires six O2 field blocks")
    if any(len(reference_blocks[name]) != O2_VALIDATION_BLOCKS for name in applicable_references):
        raise ValueError("every reference requires six O2 field blocks")

    rmse_by_reference = {
        name: float(references[name]["aggregate_equal_channel_rmse_standardized"])
        for name in applicable_references
    }
    mae_by_reference = {
        name: float(references[name]["aggregate_equal_channel_mae_standardized"])
        for name in applicable_references
    }
    best_rmse_name = min(rmse_by_reference, key=rmse_by_reference.get)
    best_mae_name = min(mae_by_reference, key=mae_by_reference.get)
    candidate_rmse = float(candidate["aggregate_equal_channel_rmse_standardized"])
    candidate_mae = float(candidate["aggregate_equal_channel_mae_standardized"])
    aggregate = {
        "candidate_rmse": candidate_rmse,
        "candidate_mae": candidate_mae,
        "reference_rmse": rmse_by_reference,
        "reference_mae": mae_by_reference,
        "best_rmse_reference": best_rmse_name,
        "best_mae_reference": best_mae_name,
        "beats_best_rmse": candidate_rmse < rmse_by_reference[best_rmse_name],
        "beats_best_mae": candidate_mae < mae_by_reference[best_mae_name],
    }
    aggregate["passes"] = aggregate["beats_best_rmse"] and aggregate["beats_best_mae"]

    def compare_fields(
        candidate_record: Mapping[str, Any],
        persistence_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        fields = {}
        for field in O2_FIELDS:
            candidate_value = float(
                candidate_record["field_metrics_standardized"][field]["rmse"]
            )
            persistence_value = float(
                persistence_record["field_metrics_standardized"][field]["rmse"]
            )
            ratio = candidate_value / persistence_value
            fields[field] = {
                "candidate_rmse": candidate_value,
                "persistence_rmse": persistence_value,
                "ratio": ratio,
                "improves": candidate_value < persistence_value,
                "within_maximum_ratio": ratio
                <= O2_THRESHOLDS["maximum_field_persistence_rmse_ratio"],
            }
        improving = sum(item["improves"] for item in fields.values())
        passes = (
            improving >= O2_THRESHOLDS["minimum_fields_beating_persistence"]
            and all(item["within_maximum_ratio"] for item in fields.values())
        )
        return {
            "fields": fields,
            "improving_field_count": improving,
            "passes": passes,
        }

    overall_fields = compare_fields(candidate, references["persistence"])
    blocks = [
        compare_fields(candidate_blocks[index], reference_blocks["persistence"][index])
        for index in range(O2_VALIDATION_BLOCKS)
    ]
    passing_blocks = sum(block["passes"] for block in blocks)
    per_field = {
        "overall": overall_fields,
        "passing_blocks": passing_blocks,
        "required_passing_blocks": O2_THRESHOLDS["required_passing_blocks"],
        "blocks": blocks,
        "passes": overall_fields["passes"]
        and passing_blocks >= O2_THRESHOLDS["required_passing_blocks"],
    }
    return {
        "aggregate_vs_best_applicable_reference": aggregate,
        "per_field_vs_persistence": per_field,
        "passes": aggregate["passes"] and per_field["passes"],
    }


def spectral_cross_gate(
    *,
    candidate: Mapping[str, Any],
    candidate_blocks: list[Mapping[str, Any]],
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply inherited training-material O2 spectral and cross-field gates."""

    if len(candidate_blocks) != O2_VALIDATION_BLOCKS:
        raise ValueError("O2 spectral gate requires six temporal blocks")
    required = int(O2_THRESHOLDS["required_passing_blocks"])

    def spectral_pass(record: Mapping[str, Any], field: str, band: str) -> bool:
        values = record["field_band_summaries"][field][band]
        power = float(values["power_ratio"])
        coherence = float(values["truth_power_weighted_transfer_coherence"])
        low, high = O2_THRESHOLDS["spectral_power_ratio"]
        return bool(
            math.isfinite(power)
            and low <= power <= high
            and math.isfinite(coherence)
            and coherence >= O2_THRESHOLDS["forecast_truth_coherence_min"]
        )

    spectral_checks: dict[str, Any] = {}
    spectral_applicable = []
    for field, bands in materiality["fields"].items():
        spectral_checks[field] = {}
        for band, frozen in bands.items():
            if frozen["material"]:
                overall = spectral_pass(candidate, field, band)
                block_pass = [
                    spectral_pass(block, field, band) for block in candidate_blocks
                ]
                record = {
                    "material_from_training_truth": True,
                    "training_truth_fraction": float(frozen["truth_fraction"]),
                    "overall_pass": overall,
                    "block_pass": block_pass,
                    "passing_blocks": sum(block_pass),
                    "passes": overall and sum(block_pass) >= required,
                }
                spectral_applicable.append(record)
            else:
                record = {
                    "material_from_training_truth": False,
                    "training_truth_fraction": float(frozen["truth_fraction"]),
                    "overall_pass": None,
                    "block_pass": None,
                    "passing_blocks": None,
                    "passes": None,
                }
            spectral_checks[field][band] = record

    def cross_pass(record: Mapping[str, Any], pair: str, band: str) -> bool:
        values = record["cross_field_band_summaries"][pair][band]
        phase = float(
            values[
                "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
            ]
        )
        coherence = float(
            values[
                "truth_cross_amplitude_weighted_absolute_coherence_change"
            ]
        )
        return bool(
            math.isfinite(phase)
            and phase <= O2_THRESHOLDS["cross_phase_error_degrees_max"]
            and math.isfinite(coherence)
            and coherence <= O2_THRESHOLDS["cross_coherence_change_max"]
        )

    cross_checks: dict[str, Any] = {}
    cross_applicable = []
    for pair, bands in materiality["cross_pairs"].items():
        cross_checks[pair] = {}
        for band, frozen in bands.items():
            if frozen["material"]:
                overall = cross_pass(candidate, pair, band)
                block_pass = [cross_pass(block, pair, band) for block in candidate_blocks]
                record = {
                    "material_from_training_truth": True,
                    "training_truth_fraction": float(frozen["truth_fraction"]),
                    "overall_pass": overall,
                    "block_pass": block_pass,
                    "passing_blocks": sum(block_pass),
                    "passes": overall and sum(block_pass) >= required,
                }
                cross_applicable.append(record)
            else:
                record = {
                    "material_from_training_truth": False,
                    "training_truth_fraction": float(frozen["truth_fraction"]),
                    "overall_pass": None,
                    "block_pass": None,
                    "passing_blocks": None,
                    "passes": None,
                }
            cross_checks[pair][band] = record

    spectral_passes = bool(spectral_applicable) and all(
        item["passes"] for item in spectral_applicable
    )
    cross_passes = bool(cross_applicable) and all(
        item["passes"] for item in cross_applicable
    )
    return {
        "thresholds": {
            "spectral_power_ratio": list(O2_THRESHOLDS["spectral_power_ratio"]),
            "forecast_truth_coherence_min": O2_THRESHOLDS[
                "forecast_truth_coherence_min"
            ],
            "cross_phase_error_degrees_max": O2_THRESHOLDS[
                "cross_phase_error_degrees_max"
            ],
            "cross_coherence_change_max": O2_THRESHOLDS[
                "cross_coherence_change_max"
            ],
            "required_passing_blocks": required,
        },
        "materiality_source_split": materiality["source_split"],
        "spectral": {
            "applicable_check_count": len(spectral_applicable),
            "checks": spectral_checks,
            "passes": spectral_passes,
        },
        "cross_field": {
            "applicable_check_count": len(cross_applicable),
            "checks": cross_checks,
            "passes": cross_passes,
        },
        "passes": spectral_passes and cross_passes,
    }


def assert_transport_summary_finite(summary: Mapping[str, Any]) -> None:
    """Reject non-finite required transport values before gate construction."""

    for quantity in TRANSPORT_QUANTITIES:
        for reduction in ("strict_faces", "separatrix"):
            metrics = _transport_metrics(summary, quantity, reduction)
            required = (
                "relative_l2",
                "normalized_bias",
                "pearson_correlation",
                "weighted_sign_disagreement",
            )
            for name in required:
                value = metrics[name]
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(
                        f"required transport metric {quantity}.{reduction}.{name} "
                        "is undefined or non-finite"
                    )


def paired_metric_record(reference: np.ndarray, forecast: np.ndarray) -> dict[str, Any]:
    """Small public helper used by known-answer tests and time-curve writers."""

    from .resampling import paired_sufficient_statistics

    return finalize_paired_statistics(
        paired_sufficient_statistics(reference, forecast)
    )


def _assert_required_field_values_finite(record: Mapping[str, Any]) -> None:
    for name in (
        "aggregate_equal_channel_rmse_standardized",
        "aggregate_equal_channel_mae_standardized",
    ):
        if not math.isfinite(float(record[name])):
            raise ValueError(f"required O2 field metric {name} is non-finite")
    for field in O2_FIELDS:
        values = record["field_metrics_standardized"][field]
        for name in ("rmse", "mae", "bias"):
            if not math.isfinite(float(values[name])):
                raise ValueError(f"required O2 field metric {field}.{name} is non-finite")


def build_o2_seed_gate(
    *,
    arm: str,
    candidate_score: Mapping[str, Any],
    reference_scores: Mapping[str, Mapping[str, Any]],
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete frozen O2 decision for one arm/seed forecast."""

    if arm == "C5P-H1":
        applicable = ("persistence", "spectral_ar1")
    elif arm == "C5P-H2":
        applicable = (
            "persistence",
            "spectral_ar1",
            "linear_extrapolation",
        )
    else:
        raise ValueError(f"unsupported O2 gate arm {arm!r}")
    if set(reference_scores) != {
        "persistence",
        "spectral_ar1",
        "linear_extrapolation",
    }:
        raise ValueError("O2 reference-score inventory differs")
    if (
        candidate_score.get("scientific_authority") is not True
        or candidate_score.get("development_run") != "85604"
        or candidate_score.get("held_out_85606_read") is not False
        or candidate_score.get("guard_frames_read") is not False
        or candidate_score.get("target_truth_used_during_forecast_generation")
        is not False
        or candidate_score.get("physics_derived_training_loss_used") is not False
        or candidate_score.get("target_frames") != [498, 624]
        or int(candidate_score.get("target_count", -1)) != 126
    ):
        raise ValueError("candidate O2 score provenance contract differs")
    for name, score in reference_scores.items():
        if (
            score.get("scientific_authority") is not True
            or score.get("development_run") != "85604"
            or score.get("held_out_85606_read") is not False
            or score.get("guard_frames_read") is not False
            or score.get("target_truth_used_during_forecast_generation") is not False
            or score.get("target_frames") != [498, 624]
            or int(score.get("target_count", -1)) != 126
        ):
            raise ValueError(f"reference O2 score {name} provenance differs")

    candidate_section = candidate_score["field_spectral_cross"]
    candidate_overall = candidate_section["overall"]
    candidate_blocks = candidate_section["blocks"]
    if len(candidate_blocks) != O2_VALIDATION_BLOCKS:
        raise ValueError("candidate O2 field block count differs")
    reference_overall = {
        name: score["field_spectral_cross"]["overall"]
        for name, score in reference_scores.items()
    }
    reference_block_records = {
        name: score["field_spectral_cross"]["blocks"]
        for name, score in reference_scores.items()
    }
    _assert_required_field_values_finite(candidate_overall)
    for record in reference_overall.values():
        _assert_required_field_values_finite(record)
    skill = reference_skill_gate(
        candidate=candidate_overall,
        candidate_blocks=candidate_blocks,
        references=reference_overall,
        reference_blocks=reference_block_records,
        applicable_references=applicable,
    )
    spectral = spectral_cross_gate(
        candidate=candidate_overall["spectral_and_cross_field"],
        candidate_blocks=[
            block["spectral_and_cross_field"] for block in candidate_blocks
        ],
        materiality=materiality,
    )
    transport_section = candidate_score["transport"]
    assert_transport_summary_finite(transport_section["overall"])
    for block in transport_section["blocks"]:
        assert_transport_summary_finite(block)
    transport = build_o2_transport_gate(
        overall=transport_section["overall"],
        temporal_blocks=transport_section["blocks"],
    )
    provenance = {
        "development_run_85604_only": True,
        "held_out_85606_read": False,
        "guard_frames_read": False,
        "future_truth_used_during_forecast": False,
        "physics_derived_training_loss_used": False,
        "all_required_values_finite": True,
        "passes": True,
    }
    passes = skill["passes"] and spectral["passes"] and transport["passes"]
    return {
        "arm": arm,
        "applicable_references": list(applicable),
        "reference_skill": skill,
        "spectral_and_cross_field": spectral,
        "transport": transport,
        "provenance_and_finiteness": provenance,
        "passes": passes,
        "status": "pass" if passes else "fail",
    }
