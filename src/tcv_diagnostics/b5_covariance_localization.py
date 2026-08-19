"""Dependence localization for the frozen Paper 0 B5 one-step ensemble.

This module is deliberately data-independent.  It does not route datasets,
open forecast artifacts, load checkpoints, run inference, or train models.  It
consumes already constructed arrays and returns compact sufficient statistics
for the prospectively frozen post-B5 localization protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .b5_residual_audit import B5_FIELDS, curve_length_summary


B5_COVARIANCE_FIELDS = B5_FIELDS
B5_COVARIANCE_PHI_INDEX = B5_FIELDS.index("phi")
B5_COVARIANCE_TOROIDAL_BANDS = {
    "k0": (0, 0),
    "k1_3": (1, 3),
    "k4_5": (4, 5),
    "k6_7": (6, 7),
    "k_ge_8": (8, None),
}
B5_VARIOGRAM_LAGS = (1, 2, 4, 8, 16, 32, 40)
B5_FINITE_MEMBER_FACTOR = 33.0 / 32.0


def _finite_real(name: str, values: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


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


def gauge_fix_fields(values: np.ndarray) -> np.ndarray:
    """Gauge-fix standardized phi independently for every leading sample.

    The final four axes must be ``[field,x,y,z]``.  Other fields are copied
    unchanged.  The function is for field covariance and field error only;
    transport must use the unmodified physical potential.
    """

    array = _finite_real("standardized fields", values, dtype=np.float32)
    if array.ndim < 4 or array.shape[-4] != len(B5_FIELDS):
        raise ValueError("field tensor must end in [field,x,y,z]")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    phi = result[..., B5_COVARIANCE_PHI_INDEX, :, :, :]
    mean = np.mean(phi, axis=(-3, -2, -1), keepdims=True, dtype=np.float64)
    result[..., B5_COVARIANCE_PHI_INDEX, :, :, :] = phi - mean
    return result


def axisymmetric_bias(values: np.ndarray) -> np.ndarray:
    """Return ``mean(sample,z)`` for a canonical ``[sample,field,x,y,z]`` tensor."""

    array = _canonical_samples("axisymmetric-bias input", values)
    return np.mean(array, axis=(0, 4), dtype=np.float64)


def subtract_axisymmetric_bias(values: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Subtract one ``[field,x,y]`` bias from canonical samples."""

    array = _canonical_samples("axisymmetric fluctuation input", values)
    baseline = _finite_real("axisymmetric bias", bias)
    if baseline.shape != array.shape[1:4]:
        raise ValueError("axisymmetric bias shape differs")
    return np.asarray(array - baseline[None, ..., None], dtype=np.float32)


class MarginalAnchorAccumulator:
    """Recompute the frozen eligible-region RMSE and spread--skill anchors."""

    def __init__(self, *, region_mask_xy: np.ndarray) -> None:
        mask = np.asarray(region_mask_xy, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("marginal-anchor mask must be a nonempty x-y mask")
        self.mask = mask
        self.scalar_count = np.zeros(len(B5_FIELDS), dtype=np.int64)
        self.squared_error_sum = np.zeros(len(B5_FIELDS), dtype=np.float64)
        self.member_variance_sum = np.zeros(len(B5_FIELDS), dtype=np.float64)

    def update(self, members: np.ndarray, truth: np.ndarray) -> None:
        ensemble = _finite_real("marginal-anchor members", members, dtype=np.float32)
        observed = _finite_real("marginal-anchor truth", truth, dtype=np.float32)
        if ensemble.ndim != 5 or ensemble.shape[1] != len(B5_FIELDS):
            raise ValueError("marginal-anchor members must be [member,field,x,y,z]")
        if (
            observed.shape != ensemble.shape[1:]
            or self.mask.shape != observed.shape[1:3]
        ):
            raise ValueError("marginal-anchor truth or mask shape differs")
        mean = np.mean(ensemble, axis=0, dtype=np.float64)
        variance = np.var(ensemble, axis=0, ddof=1, dtype=np.float64)
        for channel in range(len(B5_FIELDS)):
            error = mean[channel, self.mask, :] - observed[channel, self.mask, :]
            selected_variance = variance[channel, self.mask, :]
            self.scalar_count[channel] += int(error.size)
            self.squared_error_sum[channel] += float(np.sum(error * error))
            self.member_variance_sum[channel] += float(np.sum(selected_variance))

    def finalize(self) -> dict[str, Any]:
        if np.any(self.scalar_count <= 0) or len(set(self.scalar_count.tolist())) != 1:
            raise RuntimeError("marginal-anchor accumulator is empty or unbalanced")
        rmse = np.sqrt(self.squared_error_sum / self.scalar_count)
        member_variance = self.member_variance_sum / self.scalar_count
        aggregate_rmse = float(np.sqrt(np.mean(rmse * rmse)))
        aggregate_spread = float(
            np.sqrt(B5_FINITE_MEMBER_FACTOR * np.mean(member_variance))
        )
        return {
            "region": "eligible_union",
            "member_variance_ddof": 1,
            "finite_member_variance_factor": B5_FINITE_MEMBER_FACTOR,
            "fields": {
                field: {
                    "scalar_count": int(self.scalar_count[channel]),
                    "ensemble_mean_RMSE": float(rmse[channel]),
                    "mean_unbiased_member_variance": float(member_variance[channel]),
                }
                for channel, field in enumerate(B5_FIELDS)
            },
            "equal_channel_ensemble_mean_RMSE": aggregate_rmse,
            "equal_channel_corrected_RMS_spread": aggregate_spread,
            "equal_channel_corrected_spread_skill_ratio": (
                aggregate_spread / aggregate_rmse if aggregate_rmse > 0.0 else math.nan
            ),
        }


def _canonical_samples(name: str, values: np.ndarray) -> np.ndarray:
    array = _finite_real(name, values, dtype=np.float32)
    if array.ndim != 5 or array.shape[1] != len(B5_FIELDS):
        raise ValueError(f"{name} must have axes [sample,field,x,y,z]")
    if array.shape[0] < 1 or min(array.shape[2:]) < 2:
        raise ValueError(f"{name} dimensions are too short")
    return np.ascontiguousarray(array, dtype=np.float32)


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.einsum(
            "...,...->",
            np.asarray(left),
            np.asarray(right),
            dtype=np.float64,
            optimize=True,
        )
    )


def _normalized_correlation(
    numerator: np.ndarray,
    left_energy: np.ndarray,
    right_energy: np.ndarray,
) -> np.ndarray:
    denominator = np.sqrt(left_energy * right_energy)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("correlation denominator is nonpositive or non-finite")
    result = numerator / denominator
    result[:, 0] = 1.0
    return np.clip(result, -1.0, 1.0)


class SpatialCorrelationAccumulator:
    """Stream the frozen pooled spatial-correlation estimator."""

    _AXES = {"x": 2, "y": 3, "stored_toroidal_z": 4}

    def __init__(self, *, axis: str, volume_shape: Sequence[int]) -> None:
        if axis not in self._AXES:
            raise ValueError(f"unsupported spatial axis {axis!r}")
        shape = tuple(int(value) for value in volume_shape)
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("volume shape must contain three axes of length >=2")
        self.axis = axis
        self.volume_shape = shape
        self.extent = shape[self._AXES[axis] - 2]
        self.maximum_lag = self.extent // 2
        accumulator_shape = (len(B5_FIELDS), self.maximum_lag + 1)
        self.numerator = np.zeros(accumulator_shape, dtype=np.float64)
        self.left_energy = np.zeros(accumulator_shape, dtype=np.float64)
        self.right_energy = np.zeros(accumulator_shape, dtype=np.float64)
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        array = _canonical_samples("spatial-correlation samples", values)
        if array.shape[2:] != self.volume_shape:
            raise ValueError("spatial-correlation volume shape differs")
        field_axis = self._AXES[self.axis] - 1
        for channel in range(len(B5_FIELDS)):
            field = np.asarray(array[:, channel], dtype=np.float64)
            total = _dot(field, field)
            if self.axis == "stored_toroidal_z":
                coefficients = np.fft.rfft(field, axis=-1)
                autocorrelation = np.fft.irfft(
                    coefficients * np.conjugate(coefficients),
                    n=self.extent,
                    axis=-1,
                )
                numerator = np.sum(
                    autocorrelation[..., : self.maximum_lag + 1],
                    axis=(0, 1, 2),
                    dtype=np.float64,
                )
                self.numerator[channel] += numerator
                self.left_energy[channel] += total
                self.right_energy[channel] += total
            else:
                # Zero padding turns the FFT circular correlation into the
                # exact valid-overlap linear correlation for nonnegative lags.
                moved = np.moveaxis(field, field_axis, -1)
                transform_length = 1 << (2 * self.extent - 1).bit_length()
                coefficients = np.fft.rfft(moved, n=transform_length, axis=-1)
                autocorrelation = np.fft.irfft(
                    coefficients * np.conjugate(coefficients),
                    n=transform_length,
                    axis=-1,
                )
                reduction_axes = tuple(range(autocorrelation.ndim - 1))
                self.numerator[channel] += np.sum(
                    autocorrelation[..., : self.maximum_lag + 1],
                    axis=reduction_axes,
                    dtype=np.float64,
                )
                position_energy = np.sum(
                    moved * moved,
                    axis=tuple(range(moved.ndim - 1)),
                    dtype=np.float64,
                )
                cumulative = np.concatenate(
                    ([0.0], np.cumsum(position_energy, dtype=np.float64))
                )
                for lag in range(self.maximum_lag + 1):
                    self.left_energy[channel, lag] += cumulative[self.extent - lag]
                    self.right_energy[channel, lag] += (
                        cumulative[self.extent] - cumulative[lag]
                    )
        self.sample_count += int(array.shape[0])

    def merge(self, other: "SpatialCorrelationAccumulator") -> None:
        if (
            not isinstance(other, SpatialCorrelationAccumulator)
            or other.axis != self.axis
            or other.volume_shape != self.volume_shape
        ):
            raise ValueError("spatial-correlation accumulator merge differs")
        self.numerator += other.numerator
        self.left_energy += other.left_energy
        self.right_energy += other.right_energy
        self.sample_count += other.sample_count

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self.sample_count < 1:
            raise RuntimeError("spatial-correlation accumulator is empty")
        correlation = _normalized_correlation(
            self.numerator, self.left_energy, self.right_energy
        )
        lags = np.arange(self.maximum_lag + 1, dtype=np.int64)
        record = {
            "axis": self.axis,
            "sample_count": self.sample_count,
            "estimator": (
                "pooled_circular_normalized_correlation"
                if self.axis == "stored_toroidal_z"
                else "pooled_valid_overlap_normalized_correlation"
            ),
            "axis_extent_cells": self.extent,
            "maximum_lag_cells": self.maximum_lag,
            "lags_cells": lags.tolist(),
            "fields": {
                field: {
                    "correlation": correlation[channel].tolist(),
                    "length_summary": curve_length_summary(correlation[channel]),
                }
                for channel, field in enumerate(B5_FIELDS)
            },
        }
        raw = {
            "lags": lags,
            "numerator": self.numerator.copy(),
            "left_energy": self.left_energy.copy(),
            "right_energy": self.right_energy.copy(),
            "correlation": correlation,
        }
        return record, raw


def correlation_curve_distance(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """RMS distance between equal lag curves, excluding the shared zero lag."""

    left = _finite_real("first correlation curve", np.asarray(first))
    right = _finite_real("second correlation curve", np.asarray(second))
    if left.ndim != 1 or left.shape != right.shape or left.size < 2:
        raise ValueError("correlation curves must be equal one-dimensional arrays")
    return float(np.sqrt(np.mean((left[1:] - right[1:]) ** 2)))


def _matrix_record(*, count: int, sums: np.ndarray, gram: np.ndarray) -> dict[str, Any]:
    if int(count) < 2:
        raise ValueError("correlation matrix needs at least two samples")
    centered = gram - np.outer(sums, sums) / int(count)
    diagonal = np.diag(centered)
    if np.any(diagonal <= 0.0):
        raise ValueError("correlation matrix contains nonpositive variance")
    correlation = centered / np.sqrt(np.outer(diagonal, diagonal))
    correlation = np.clip(0.5 * (correlation + correlation.T), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    nonnegative = np.clip(eigenvalues, 0.0, None)
    probabilities = nonnegative / np.sum(nonnegative)
    entropy = -float(
        np.sum(
            probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])
        )
    )
    participation = float(np.sum(nonnegative) ** 2 / np.sum(nonnegative**2))
    return {
        "sample_count": int(count),
        "correlation_matrix": correlation.tolist(),
        "eigenvalues_descending": eigenvalues.tolist(),
        "entropy_effective_rank": math.exp(entropy),
        "participation_ratio_effective_rank": participation,
        "centered_gram": centered,
    }


class CrossFieldCorrelationAccumulator:
    """Stream five-field correlations over fixed geometry masks."""

    def __init__(
        self,
        *,
        region_masks_xy: Mapping[str, np.ndarray],
        volume_shape: Sequence[int],
    ) -> None:
        shape = tuple(int(value) for value in volume_shape)
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("cross-field volume shape differs")
        self.volume_shape = shape
        self.masks = {
            str(name): np.asarray(mask, dtype=bool)
            for name, mask in region_masks_xy.items()
        }
        if "global" in self.masks or not self.masks:
            raise ValueError("regions must be nonempty and must not include global")
        for mask in self.masks.values():
            if mask.shape != shape[:2] or not np.any(mask):
                raise ValueError("cross-field region mask differs or is empty")
        names = ("global", *self.masks)
        self.count = {name: 0 for name in names}
        self.sums = {name: np.zeros(len(B5_FIELDS), dtype=np.float64) for name in names}
        self.gram = {
            name: np.zeros((len(B5_FIELDS), len(B5_FIELDS)), dtype=np.float64)
            for name in names
        }

    def _update_one(self, name: str, selected: np.ndarray) -> None:
        # selected has [sample,field,point,z]
        flat = np.moveaxis(selected, 1, 0).reshape(len(B5_FIELDS), -1)
        self.count[name] += int(flat.shape[1])
        self.sums[name] += np.sum(flat, axis=1, dtype=np.float64)
        self.gram[name] += np.einsum(
            "ik,jk->ij", flat, flat, dtype=np.float64, optimize=True
        )

    def update(self, values: np.ndarray) -> None:
        array = _canonical_samples("cross-field samples", values)
        if array.shape[2:] != self.volume_shape:
            raise ValueError("cross-field sample volume differs")
        self._update_one("global", array.reshape(array.shape[0], len(B5_FIELDS), -1, 1))
        for name, mask in self.masks.items():
            self._update_one(name, array[:, :, mask, :])

    def merge(self, other: "CrossFieldCorrelationAccumulator") -> None:
        if (
            not isinstance(other, CrossFieldCorrelationAccumulator)
            or other.volume_shape != self.volume_shape
            or tuple(other.masks) != tuple(self.masks)
            or any(
                not np.array_equal(other.masks[name], self.masks[name])
                for name in self.masks
            )
        ):
            raise ValueError("cross-field accumulator merge differs")
        for name in self.count:
            self.count[name] += other.count[name]
            self.sums[name] += other.sums[name]
            self.gram[name] += other.gram[name]

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        records: dict[str, Any] = {}
        raw: dict[str, np.ndarray] = {}
        for name in self.count:
            record = _matrix_record(
                count=self.count[name], sums=self.sums[name], gram=self.gram[name]
            )
            centered = np.asarray(record.pop("centered_gram"), dtype=np.float64)
            record["field_order"] = list(B5_FIELDS)
            records[name] = record
            raw[f"{name}__sample_count"] = np.asarray(
                [self.count[name]], dtype=np.int64
            )
            raw[f"{name}__sums"] = self.sums[name].copy()
            raw[f"{name}__uncentered_gram"] = self.gram[name].copy()
            raw[f"{name}__centered_gram"] = centered
            raw[f"{name}__correlation"] = np.asarray(
                record["correlation_matrix"], dtype=np.float64
            )
        return records, raw


def off_diagonal_rms_distance(first: np.ndarray, second: np.ndarray) -> float:
    """RMS distance over unique off-diagonal correlation entries."""

    left = _finite_real("first correlation matrix", first)
    right = _finite_real("second correlation matrix", second)
    if left.ndim != 2 or left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("correlation matrices must be equal and square")
    indices = np.triu_indices(left.shape[0], k=1)
    return float(np.sqrt(np.mean((left[indices] - right[indices]) ** 2)))


def dependence_distance_summary(
    *,
    training: Mapping[str, Any],
    validation_h1: Mapping[str, Any],
    b5_anomaly: Mapping[str, Any],
    b5_innovation: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare B5 dependence errors with training-to-validation drift."""

    spatial: dict[str, Any] = {}
    for axis in ("x", "y", "stored_toroidal_z"):
        axis_record: dict[str, Any] = {}
        for field in B5_FIELDS:
            train_curve = training["spatial_autocorrelation"][axis]["fields"][field][
                "correlation"
            ]
            validation_curve = validation_h1["spatial_autocorrelation"][axis]["fields"][
                field
            ]["correlation"]
            anomaly_curve = b5_anomaly["spatial_autocorrelation"][axis]["fields"][
                field
            ]["correlation"]
            innovation_curve = b5_innovation["spatial_autocorrelation"][axis]["fields"][
                field
            ]["correlation"]
            drift = correlation_curve_distance(train_curve, validation_curve)
            anomaly = correlation_curve_distance(anomaly_curve, validation_curve)
            innovation = correlation_curve_distance(innovation_curve, validation_curve)
            axis_record[field] = {
                "training_to_validation_H1_residual_RMS": drift,
                "B5_anomaly_to_validation_H1_residual_RMS": anomaly,
                "B5_innovation_to_validation_H1_residual_RMS": innovation,
                "B5_anomaly_exceeds_within_run_drift": anomaly > drift,
            }
        spatial[axis] = axis_record
    cross_field: dict[str, Any] = {}
    regions = tuple(validation_h1["cross_field"])
    if any(
        tuple(record["cross_field"]) != regions
        for record in (training, b5_anomaly, b5_innovation)
    ):
        raise ValueError("dependence-distance cross-field region order differs")
    for region in regions:
        validation_matrix = np.asarray(
            validation_h1["cross_field"][region]["correlation_matrix"]
        )
        drift = off_diagonal_rms_distance(
            np.asarray(training["cross_field"][region]["correlation_matrix"]),
            validation_matrix,
        )
        anomaly = off_diagonal_rms_distance(
            np.asarray(b5_anomaly["cross_field"][region]["correlation_matrix"]),
            validation_matrix,
        )
        innovation = off_diagonal_rms_distance(
            np.asarray(b5_innovation["cross_field"][region]["correlation_matrix"]),
            validation_matrix,
        )
        cross_field[region] = {
            "training_to_validation_H1_residual_RMS": drift,
            "B5_anomaly_to_validation_H1_residual_RMS": anomaly,
            "B5_innovation_to_validation_H1_residual_RMS": innovation,
            "B5_anomaly_exceeds_within_run_drift": anomaly > drift,
        }
    identities = [
        f"spatial:{axis}:{field}"
        for axis, records in spatial.items()
        for field, record in records.items()
        if record["B5_anomaly_exceeds_within_run_drift"]
    ] + [
        f"cross_field:{region}"
        for region, record in cross_field.items()
        if record["B5_anomaly_exceeds_within_run_drift"]
    ]
    return {
        "spatial": spatial,
        "cross_field": cross_field,
        "B5_anomaly_exceeds_drift_count": len(identities),
        "B5_anomaly_exceeds_drift_identities": identities,
        "empirical_within_run_drift_is_not_a_sampling_distribution": True,
    }


def blockwise_l3_summary(
    *,
    training: Mapping[str, Any],
    validation_h1_blocks: Mapping[str, Mapping[str, Any]],
    b5_anomaly_blocks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen five-of-six direction rule to L3 identities."""

    block_names = tuple(validation_h1_blocks)
    if len(block_names) != 6 or tuple(b5_anomaly_blocks) != block_names:
        raise ValueError("L3 requires six identically ordered chronological blocks")
    counts: dict[str, int] = {}
    by_block: dict[str, Any] = {}
    for block in block_names:
        comparison = dependence_distance_summary(
            training=training,
            validation_h1=validation_h1_blocks[block],
            b5_anomaly=b5_anomaly_blocks[block],
            b5_innovation=b5_anomaly_blocks[block],
        )
        identities = comparison["B5_anomaly_exceeds_drift_identities"]
        by_block[block] = {
            "B5_anomaly_exceeds_drift_count": len(identities),
            "B5_anomaly_exceeds_drift_identities": identities,
        }
        for identity in identities:
            counts[identity] = counts.get(identity, 0) + 1
    systematic = sorted(identity for identity, count in counts.items() if count >= 5)
    return {
        "chronological_block_count": 6,
        "required_direction_count": 5,
        "by_block": by_block,
        "direction_counts": dict(sorted(counts.items())),
        "systematic_identity_count": len(systematic),
        "systematic_identities": systematic,
        "L3_field_dependence_mismatch_beyond_within_run_drift_supported": bool(
            systematic
        ),
    }


class ToroidalPowerAccumulator:
    """Stream Parseval-weighted toroidal power for canonical field samples."""

    def __init__(self, *, volume_shape: Sequence[int], sample_chunk: int = 4) -> None:
        shape = tuple(int(value) for value in volume_shape)
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError("toroidal-power volume shape differs")
        if int(sample_chunk) < 1:
            raise ValueError("toroidal-power chunk must be positive")
        self.volume_shape = shape
        self.sample_chunk = int(sample_chunk)
        self.n_k = shape[-1] // 2 + 1
        self.unweighted_power = np.zeros((len(B5_FIELDS), self.n_k), dtype=np.float64)
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        array = _canonical_samples("toroidal-power samples", values)
        if array.shape[2:] != self.volume_shape:
            raise ValueError("toroidal-power sample volume differs")
        for start in range(0, array.shape[0], self.sample_chunk):
            stop = min(start + self.sample_chunk, array.shape[0])
            coefficients = np.fft.rfft(array[start:stop], axis=-1)
            self.unweighted_power += np.sum(
                coefficients.real * coefficients.real
                + coefficients.imag * coefficients.imag,
                axis=(0, 2, 3),
                dtype=np.float64,
            )
        self.sample_count += int(array.shape[0])

    def merge(self, other: "ToroidalPowerAccumulator") -> None:
        if (
            not isinstance(other, ToroidalPowerAccumulator)
            or other.volume_shape != self.volume_shape
        ):
            raise ValueError("toroidal-power accumulator merge differs")
        self.unweighted_power += other.unweighted_power
        self.sample_count += other.sample_count

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self.sample_count < 1:
            raise RuntimeError("toroidal-power accumulator is empty")
        n_z = self.volume_shape[-1]
        weights = np.full(self.n_k, 2.0, dtype=np.float64)
        weights[0] = 1.0
        if n_z % 2 == 0:
            weights[-1] = 1.0
        weighted = self.unweighted_power * weights[None]
        density = weighted / float(
            self.sample_count * self.volume_shape[0] * self.volume_shape[1] * n_z * n_z
        )
        fields: dict[str, Any] = {}
        for channel, field in enumerate(B5_FIELDS):
            total = float(np.sum(density[channel]))
            if total <= 0.0:
                raise ValueError("toroidal power is nonpositive")
            bands: dict[str, Any] = {}
            for label, (start, frozen_stop) in B5_COVARIANCE_TOROIDAL_BANDS.items():
                stop = (
                    self.n_k - 1
                    if frozen_stop is None
                    else min(frozen_stop, self.n_k - 1)
                )
                power = float(np.sum(density[channel, start : stop + 1]))
                bands[label] = {
                    "stored_k_inclusive": [int(start), int(stop)],
                    "full_torus_n_inclusive": [int(5 * start), int(5 * stop)],
                    "mean_parseval_power_density": power,
                    "power_fraction": power / total,
                }
            fields[field] = {"total_mean_parseval_power_density": total, "bands": bands}
        record = {
            "sample_count": self.sample_count,
            "stored_toroidal_cells": n_z,
            "stored_k_maximum": self.n_k - 1,
            "zperiod": 5,
            "mode_mapping": "n=5k",
            "normalization": "mean_over_samples_x_y_of_sum_abs_rfft_squared_over_z_squared",
            "fields": fields,
        }
        raw = {
            "stored_k": np.arange(self.n_k, dtype=np.int64),
            "parseval_weights": weights,
            "unweighted_positive_frequency_power": self.unweighted_power.copy(),
            "mean_parseval_power_density": density,
        }
        return record, raw


class CovarianceSummaryAccumulator:
    """Bundle the three frozen spatial, regional, and toroidal estimators."""

    def __init__(
        self,
        *,
        region_masks_xy: Mapping[str, np.ndarray],
        volume_shape: Sequence[int],
    ) -> None:
        self.spatial = {
            axis: SpatialCorrelationAccumulator(axis=axis, volume_shape=volume_shape)
            for axis in ("x", "y", "stored_toroidal_z")
        }
        self.cross_field = CrossFieldCorrelationAccumulator(
            region_masks_xy=region_masks_xy,
            volume_shape=volume_shape,
        )
        self.toroidal = ToroidalPowerAccumulator(volume_shape=volume_shape)

    def update(self, values: np.ndarray) -> None:
        array = _canonical_samples("covariance-summary samples", values)
        for accumulator in self.spatial.values():
            accumulator.update(array)
        self.cross_field.update(array)
        self.toroidal.update(array)

    def merge(self, other: "CovarianceSummaryAccumulator") -> None:
        if not isinstance(other, CovarianceSummaryAccumulator):
            raise TypeError("covariance-summary merge requires a matching accumulator")
        for axis in self.spatial:
            self.spatial[axis].merge(other.spatial[axis])
        self.cross_field.merge(other.cross_field)
        self.toroidal.merge(other.toroidal)

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        spatial: dict[str, Any] = {}
        cross: dict[str, Any]
        toroidal: dict[str, Any]
        raw: dict[str, np.ndarray] = {}
        for axis, accumulator in self.spatial.items():
            spatial[axis], values = accumulator.finalize()
            for name, array in values.items():
                raw[f"spatial__{axis}__{name}"] = array
        cross, values = self.cross_field.finalize()
        for name, array in values.items():
            raw[f"cross_field__{name}"] = array
        toroidal, values = self.toroidal.finalize()
        for name, array in values.items():
            raw[f"toroidal__{name}"] = array
        return {
            "spatial_autocorrelation": spatial,
            "cross_field": cross,
            "toroidal_support": toroidal,
        }, raw


def field_variogram_score(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    region_masks_xy: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Order-one variogram score over all ten same-cell field pairs.

    Pair-cell weights are normalized to sum to one independently in each
    region, making region scores comparable in scale without asserting cell
    independence.
    """

    members = _finite_real("variogram forecast", forecast, dtype=np.float32)
    observed = _finite_real("variogram truth", truth, dtype=np.float32)
    if members.ndim != 5 or members.shape[1] != len(B5_FIELDS):
        raise ValueError("variogram forecast must have [member,field,x,y,z]")
    if observed.shape != members.shape[1:]:
        raise ValueError("variogram truth shape differs")
    masks = {"global": np.ones(observed.shape[1:3], dtype=bool)}
    masks.update(
        {
            str(name): np.asarray(mask, dtype=bool)
            for name, mask in region_masks_xy.items()
        }
    )
    for mask in masks.values():
        if mask.shape != observed.shape[1:3] or not np.any(mask):
            raise ValueError("variogram mask differs or is empty")
    sums = {name: 0.0 for name in masks}
    pair_count = 0
    # Form each member-wise difference only once.  Region reductions then use
    # the same score map, which is both the frozen estimator and substantially
    # less memory traffic than recomputing it independently for every mask.
    for first in range(len(B5_FIELDS)):
        for second in range(first + 1, len(B5_FIELDS)):
            truth_difference = np.abs(observed[first] - observed[second])
            forecast_difference = np.mean(
                np.abs(members[:, first] - members[:, second]),
                axis=0,
                dtype=np.float64,
            )
            score_map = np.square(truth_difference - forecast_difference)
            for name, mask in masks.items():
                sums[name] += float(np.mean(score_map[mask, :]))
            pair_count += 1
    if pair_count != 10:
        raise RuntimeError("five-field variogram pair count differs")
    return {name: value / pair_count for name, value in sums.items()}


def transport_variogram_score(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    lags: Sequence[int] = B5_VARIOGRAM_LAGS,
) -> dict[str, Any]:
    """Order-one periodic-z variogram score for local separatrix contributions."""

    members = _finite_real("transport variogram forecast", forecast)
    observed = _finite_real("transport variogram truth", truth)
    if members.ndim != 3 or observed.shape != members.shape[1:]:
        raise ValueError(
            "transport variogram arrays must be [member,row,z] and [row,z]"
        )
    n_z = observed.shape[-1]
    records: dict[str, float] = {}
    for lag_value in lags:
        lag = int(lag_value)
        if lag <= 0 or lag > n_z // 2:
            raise ValueError(
                "transport variogram lag is outside the periodic half-domain"
            )
        truth_difference = np.abs(observed - np.roll(observed, -lag, axis=-1))
        member_difference = np.mean(
            np.abs(members - np.roll(members, -lag, axis=-1)),
            axis=0,
            dtype=np.float64,
        )
        records[f"lag_{lag}"] = float(
            np.mean((truth_difference - member_difference) ** 2)
        )
    return {
        "order_p": 1.0,
        "lags": [int(value) for value in lags],
        "equal_pair_weight_within_lag": True,
        "by_lag": records,
        "equal_lag_mean": float(np.mean(list(records.values()))),
    }


def exact_separatrix_local_contributions(
    evaluated_transport: Mapping[str, Mapping[str, np.ndarray]],
    *,
    strict_face_mask: np.ndarray,
    separatrix_face_mask: np.ndarray,
    expected_rows: int = 16,
) -> tuple[dict[str, np.ndarray], float]:
    """Select exact-separatrix rows and verify their sum against each wedge.

    ``strict_face_contributions`` already contain geometry and quadrature
    weights.  The selector maps the separatrix mask into that strict-face
    ordering; no image-coordinate interpretation is introduced here.
    """

    strict = np.asarray(strict_face_mask, dtype=bool)
    separatrix = np.asarray(separatrix_face_mask, dtype=bool)
    if strict.ndim != 2 or separatrix.shape != strict.shape:
        raise ValueError("transport face masks must be matched two-dimensional arrays")
    if np.any(separatrix & ~strict):
        raise ValueError("separatrix face mask leaves the strict-face support")
    selector = np.asarray(separatrix[strict], dtype=bool)
    if int(np.sum(selector)) != int(expected_rows):
        raise ValueError("exact separatrix row count differs")
    if not evaluated_transport:
        raise ValueError("evaluated transport mapping is empty")
    selected: dict[str, np.ndarray] = {}
    maximum_relative_closure = 0.0
    for quantity, reductions in evaluated_transport.items():
        contributions = _finite_real(
            f"{quantity} strict-face contributions",
            reductions["strict_face_contributions"],
        )
        wedge = _finite_real(
            f"{quantity} separatrix wedge", reductions["separatrix_wedge"]
        )
        if contributions.ndim != 3 or contributions.shape[1] != selector.size:
            raise ValueError("strict-face contribution tensor shape differs")
        if wedge.shape != (contributions.shape[0],):
            raise ValueError("separatrix wedge tensor shape differs")
        local = np.ascontiguousarray(contributions[:, selector, :], dtype=np.float64)
        reconstructed = np.sum(local, axis=(1, 2), dtype=np.float64)
        difference = np.abs(reconstructed - wedge)
        scale = np.maximum.reduce(
            [np.abs(wedge), np.abs(reconstructed), np.ones_like(wedge)]
        )
        maximum_relative_closure = max(
            maximum_relative_closure, float(np.max(difference / scale))
        )
        if not np.allclose(reconstructed, wedge, rtol=2e-12, atol=1e-12):
            raise RuntimeError(f"{quantity} exact-separatrix sum does not close")
        selected[str(quantity)] = local
    return selected, maximum_relative_closure


class ScalarCircularCorrelationAccumulator:
    """Stream circular correlation for scalar arrays ending in toroidal z."""

    def __init__(self, *, n_z: int) -> None:
        if int(n_z) < 2:
            raise ValueError("circular correlation needs at least two z cells")
        self.n_z = int(n_z)
        self.maximum_lag = self.n_z // 2
        self.numerator = np.zeros(self.maximum_lag + 1, dtype=np.float64)
        self.energy = np.zeros(self.maximum_lag + 1, dtype=np.float64)
        self.sample_count = 0

    def update(self, values: np.ndarray) -> None:
        array = _finite_real("scalar circular samples", values, dtype=np.float32)
        if array.ndim < 2 or array.shape[-1] != self.n_z:
            raise ValueError("scalar circular samples must end in the frozen z axis")
        total = _dot(array, array)
        self.numerator[0] += total
        self.energy += total
        for lag in range(1, self.maximum_lag + 1):
            self.numerator[lag] += _dot(array, np.roll(array, -lag, axis=-1))
        self.sample_count += int(array.shape[0])

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self.sample_count < 1 or np.any(self.energy <= 0.0):
            raise RuntimeError("scalar circular correlation is empty or degenerate")
        correlation = np.clip(self.numerator / self.energy, -1.0, 1.0)
        correlation[0] = 1.0
        lags = np.arange(self.maximum_lag + 1, dtype=np.int64)
        return (
            {
                "sample_count": self.sample_count,
                "lags_cells": lags.tolist(),
                "correlation": correlation.tolist(),
                "length_summary": curve_length_summary(correlation),
            },
            {
                "lags": lags,
                "numerator": self.numerator.copy(),
                "energy": self.energy.copy(),
                "correlation": correlation,
            },
        )


def _correlation_matrix_from_samples(values: np.ndarray) -> dict[str, Any]:
    samples = _finite_real("matrix samples", values)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 2:
        raise ValueError("matrix samples must have [sample,component]")
    return _matrix_record(
        count=samples.shape[0],
        sums=np.sum(samples, axis=0, dtype=np.float64),
        gram=np.einsum("si,sj->ij", samples, samples, dtype=np.float64),
    )


@dataclass
class _TransportSums:
    target_count: int = 0
    ensemble_diagonal_variance_sum: float = 0.0
    ensemble_integrated_variance_sum: float = 0.0
    error_diagonal_squared_sum: float = 0.0
    error_integrated_squared_sum: float = 0.0
    scalar_point_count: int = 0

    def update(
        self, members: np.ndarray, truth: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        ensemble = _finite_real("local transport members", members)
        observed = _finite_real("local transport truth", truth)
        if ensemble.ndim != 2 or observed.shape != ensemble.shape[1:]:
            raise ValueError(
                "local transport arrays must be [member,point] and [point]"
            )
        if ensemble.shape[0] < 2:
            raise ValueError("local transport covariance needs at least two members")
        mean = np.mean(ensemble, axis=0)
        error = observed - mean
        local_variance = np.var(ensemble, axis=0, ddof=1)
        integrated = np.sum(ensemble, axis=1)
        integrated_variance = float(np.var(integrated, ddof=1))
        self.target_count += 1
        self.ensemble_diagonal_variance_sum += float(np.sum(local_variance))
        self.ensemble_integrated_variance_sum += integrated_variance
        self.error_diagonal_squared_sum += float(np.sum(error * error))
        self.error_integrated_squared_sum += float(np.sum(error) ** 2)
        self.scalar_point_count += int(observed.size)
        return error, {
            "ensemble_diagonal_variance_sum": float(np.sum(local_variance)),
            "ensemble_integrated_variance": integrated_variance,
            "error_diagonal_squared_sum": float(np.sum(error * error)),
            "error_integrated_squared": float(np.sum(error) ** 2),
        }

    def finalize(self) -> dict[str, float | int | None]:
        if self.target_count < 1 or self.scalar_point_count < 1:
            raise RuntimeError("transport covariance sums are empty")
        diagonal_ssr = (
            math.sqrt(
                B5_FINITE_MEMBER_FACTOR
                * self.ensemble_diagonal_variance_sum
                / self.error_diagonal_squared_sum
            )
            if self.error_diagonal_squared_sum > 0.0
            else math.nan
        )
        integrated_ssr = (
            math.sqrt(
                B5_FINITE_MEMBER_FACTOR
                * self.ensemble_integrated_variance_sum
                / self.error_integrated_squared_sum
            )
            if self.error_integrated_squared_sum > 0.0
            else math.nan
        )
        ensemble_multiplier = (
            self.ensemble_integrated_variance_sum / self.ensemble_diagonal_variance_sum
            if self.ensemble_diagonal_variance_sum > 0.0
            else math.nan
        )
        error_multiplier = (
            self.error_integrated_squared_sum / self.error_diagonal_squared_sum
            if self.error_diagonal_squared_sum > 0.0
            else math.nan
        )
        alpha = 1.0 / integrated_ssr if integrated_ssr > 0.0 else math.nan
        return {
            "target_count": self.target_count,
            "local_point_count_per_target": self.scalar_point_count
            // self.target_count,
            "member_variance_ddof": 1,
            "finite_member_variance_factor": B5_FINITE_MEMBER_FACTOR,
            "ensemble_diagonal_variance_mean_over_targets": (
                self.ensemble_diagonal_variance_sum / self.target_count
            ),
            "ensemble_integrated_variance_mean_over_targets": (
                self.ensemble_integrated_variance_sum / self.target_count
            ),
            "ensemble_off_diagonal_variance_mean_over_targets": (
                self.ensemble_integrated_variance_sum
                - self.ensemble_diagonal_variance_sum
            )
            / self.target_count,
            "error_diagonal_MSE_sum_mean_over_targets": (
                self.error_diagonal_squared_sum / self.target_count
            ),
            "error_integrated_MSE_mean_over_targets": (
                self.error_integrated_squared_sum / self.target_count
            ),
            "local_corrected_spread_skill_ratio": diagonal_ssr,
            "integrated_corrected_spread_skill_ratio": integrated_ssr,
            "ensemble_coherence_multiplier": ensemble_multiplier,
            "error_coherence_multiplier": error_multiplier,
            "ensemble_to_error_coherence_multiplier_ratio": (
                ensemble_multiplier / error_multiplier
                if error_multiplier > 0.0
                else math.nan
            ),
            "scalar_factor_to_match_integrated_spread": alpha,
            "counterfactual_local_spread_skill_after_same_factor": alpha * diagonal_ssr,
        }


class TransportCovarianceAccumulator:
    """Localize exact-separatrix covariance for multiple transport quantities."""

    def __init__(self, *, quantities: Sequence[str], rows: int, n_z: int) -> None:
        names = tuple(str(value) for value in quantities)
        if not names or len(set(names)) != len(names):
            raise ValueError("transport quantities must be unique and nonempty")
        if int(rows) < 2 or int(n_z) < 2:
            raise ValueError("transport separatrix shape is too small")
        self.quantities = names
        self.rows = int(rows)
        self.n_z = int(n_z)
        self.sums = {name: _TransportSums() for name in names}
        self.ensemble_z = {
            name: ScalarCircularCorrelationAccumulator(n_z=self.n_z) for name in names
        }
        self.row_count = {name: 0 for name in names}
        self.row_sums = {name: np.zeros(self.rows) for name in names}
        self.row_gram = {name: np.zeros((self.rows, self.rows)) for name in names}
        self.errors: dict[str, list[np.ndarray]] = {name: [] for name in names}
        self.per_target: list[dict[str, Any]] = []

    def update(
        self,
        *,
        target_frame: int,
        forecast: Mapping[str, np.ndarray],
        truth: Mapping[str, np.ndarray],
    ) -> None:
        if tuple(forecast) != self.quantities or tuple(truth) != self.quantities:
            raise ValueError("transport quantity order differs")
        target_record: dict[str, Any] = {
            "target_frame": int(target_frame),
            "quantities": {},
        }
        for name in self.quantities:
            members = _finite_real(f"{name} forecast local transport", forecast[name])
            observed = _finite_real(f"{name} truth local transport", truth[name])
            if members.ndim != 3 or members.shape[1:] != (self.rows, self.n_z):
                raise ValueError("forecast local transport shape differs")
            if observed.shape != (self.rows, self.n_z):
                raise ValueError("truth local transport shape differs")
            error, scalar = self.sums[name].update(
                members.reshape(members.shape[0], -1), observed.reshape(-1)
            )
            anomalies = members - np.mean(members, axis=0, keepdims=True)
            self.ensemble_z[name].update(anomalies)
            row = np.sum(anomalies, axis=-1)
            self.row_count[name] += int(row.shape[0])
            self.row_sums[name] += np.sum(row, axis=0)
            self.row_gram[name] += np.einsum("mi,mj->ij", row, row, dtype=np.float64)
            self.errors[name].append(error.reshape(self.rows, self.n_z))
            scalar["transport_variogram_score"] = transport_variogram_score(
                members, observed
            )
            target_record["quantities"][name] = scalar
        self.per_target.append(target_record)

    def finalize(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if not self.per_target:
            raise RuntimeError("transport covariance accumulator is empty")
        quantities: dict[str, Any] = {}
        raw: dict[str, np.ndarray] = {}
        for name in self.quantities:
            aggregate = self.sums[name].finalize()
            error = np.stack(self.errors[name], axis=0)
            centered = error - np.mean(error, axis=0, keepdims=True)
            diagonal = float(np.sum(np.var(error, axis=0, ddof=0)))
            integrated = float(np.var(np.sum(error, axis=(1, 2)), ddof=0))
            innovation_multiplier = (
                integrated / diagonal if diagonal > 0.0 else math.nan
            )
            aggregate["time_centered_innovation_covariance_ddof"] = 0
            aggregate["innovation_diagonal_variance_sum"] = diagonal
            aggregate["innovation_integrated_variance"] = integrated
            aggregate["innovation_coherence_multiplier"] = innovation_multiplier
            ensemble_z_record, ensemble_z_raw = self.ensemble_z[name].finalize()
            innovation_z_accumulator = ScalarCircularCorrelationAccumulator(
                n_z=self.n_z
            )
            innovation_z_accumulator.update(centered)
            innovation_z_record, innovation_z_raw = innovation_z_accumulator.finalize()
            row_record = _matrix_record(
                count=self.row_count[name],
                sums=self.row_sums[name],
                gram=self.row_gram[name],
            )
            row_record.pop("centered_gram")
            innovation_rows = np.sum(centered, axis=-1)
            innovation_row_record = _correlation_matrix_from_samples(innovation_rows)
            innovation_row_record.pop("centered_gram")
            per_target_vs = [
                item["quantities"][name]["transport_variogram_score"]
                for item in self.per_target
            ]
            lag_keys = [f"lag_{lag}" for lag in B5_VARIOGRAM_LAGS]
            aggregate_variogram = {
                "order_p": 1.0,
                "lags": list(B5_VARIOGRAM_LAGS),
                "equal_lag_mean": float(
                    np.mean([item["equal_lag_mean"] for item in per_target_vs])
                ),
                "by_lag": {
                    key: float(np.mean([item["by_lag"][key] for item in per_target_vs]))
                    for key in lag_keys
                },
            }
            quantities[name] = {
                "covariance_decomposition": aggregate,
                "ensemble_local_toroidal_correlation": ensemble_z_record,
                "innovation_local_toroidal_correlation": innovation_z_record,
                "ensemble_toroidally_integrated_row_correlation": row_record,
                "innovation_toroidally_integrated_row_correlation": innovation_row_record,
                "transport_variogram_score": aggregate_variogram,
            }
            raw[f"{name}__innovation_local_error"] = error
            for prefix, source in (
                ("ensemble_z", ensemble_z_raw),
                ("innovation_z", innovation_z_raw),
            ):
                for key, values in source.items():
                    raw[f"{name}__{prefix}__{key}"] = values
            raw[f"{name}__ensemble_row_gram"] = self.row_gram[name].copy()
            raw[f"{name}__ensemble_row_sums"] = self.row_sums[name].copy()
        record = {
            "quantity_order": list(self.quantities),
            "target_count": len(self.per_target),
            "separatrix_rows": self.rows,
            "native_toroidal_cells": self.n_z,
            "quantities": quantities,
            "per_target": self.per_target,
            "nonlinear_operator_applied_memberwise_before_reduction": True,
        }
        return _json_safe(record), raw


def training_frozen_ar1_coefficients(
    raw_accumulators: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Return fieldwise lag-one least-squares coefficients from frozen training sums."""

    numerator = _finite_real(
        "training temporal numerator",
        raw_accumulators["temporal_pattern__numerator"],
    )
    left_energy = _finite_real(
        "training temporal left energy",
        raw_accumulators["temporal_pattern__left_energy"],
    )
    if numerator.shape != left_energy.shape or numerator.shape[0] != len(B5_FIELDS):
        raise ValueError("training temporal sufficient-statistic shape differs")
    if numerator.shape[1] < 2 or np.any(left_energy[:, 1] <= 0.0):
        raise ValueError("training lag-one predictor denominator is invalid")
    return np.asarray(numerator[:, 1] / left_energy[:, 1], dtype=np.float64)


def training_frozen_ar1_prediction(
    *,
    h1_mean: np.ndarray,
    previous_h1_residual: np.ndarray,
    coefficients: np.ndarray,
    axisymmetric_training_bias: np.ndarray,
) -> np.ndarray:
    """Construct the frozen teacher-forced residual-history prediction."""

    mean = _canonical_samples("H1 mean", h1_mean)
    previous = _canonical_samples("previous H1 residual", previous_h1_residual)
    if mean.shape != previous.shape:
        raise ValueError("H1 mean and previous residual shapes differ")
    alpha = _finite_real("AR1 coefficients", coefficients)
    bias = _finite_real("training axisymmetric bias", axisymmetric_training_bias)
    if alpha.shape != (len(B5_FIELDS),) or bias.shape != mean.shape[1:4]:
        raise ValueError("AR1 coefficient or bias shape differs")
    residual = bias[None, ..., None] + alpha[None, :, None, None, None] * (
        previous - bias[None, ..., None]
    )
    return np.asarray(mean + residual, dtype=np.float32)


def deterministic_field_error_summary(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    """Gauge-aware standardized RMSE, MAE, and bias for deterministic fields."""

    candidate = gauge_fix_fields(
        _canonical_samples("deterministic prediction", prediction)
    )
    observed = gauge_fix_fields(_canonical_samples("deterministic truth", truth))
    if candidate.shape != observed.shape:
        raise ValueError("deterministic prediction/truth shapes differ")
    error = np.asarray(candidate - observed, dtype=np.float64)
    fields = {}
    for channel, field in enumerate(B5_FIELDS):
        values = error[:, channel]
        fields[field] = {
            "RMSE": float(np.sqrt(np.mean(values * values))),
            "MAE": float(np.mean(np.abs(values))),
            "bias": float(np.mean(values)),
        }
    return {
        "target_count": int(error.shape[0]),
        "phi_gauge_fixed": True,
        "fields": fields,
        "equal_field_mean_RMSE": float(
            np.mean([item["RMSE"] for item in fields.values()])
        ),
        "equal_field_mean_MAE": float(
            np.mean([item["MAE"] for item in fields.values()])
        ),
    }


def deterministic_toroidal_summary(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    """Power ratio and realization coherence for deterministic toroidal bands."""

    candidate = gauge_fix_fields(_canonical_samples("spectral prediction", prediction))
    observed = gauge_fix_fields(_canonical_samples("spectral truth", truth))
    if candidate.shape != observed.shape:
        raise ValueError("spectral prediction/truth shapes differ")
    pred_fft = np.fft.rfft(candidate, axis=-1)
    truth_fft = np.fft.rfft(observed, axis=-1)
    fields: dict[str, Any] = {}
    for channel, field in enumerate(B5_FIELDS):
        bands: dict[str, Any] = {}
        for label, (low, frozen_high) in B5_COVARIANCE_TOROIDAL_BANDS.items():
            high = pred_fft.shape[-1] - 1 if frozen_high is None else frozen_high
            selected = slice(low, high + 1)
            pred = pred_fft[:, channel, ..., selected]
            target = truth_fft[:, channel, ..., selected]
            pred_power = float(np.sum(np.abs(pred) ** 2, dtype=np.float64))
            truth_power = float(np.sum(np.abs(target) ** 2, dtype=np.float64))
            cross = np.sum(target * np.conjugate(pred), dtype=np.complex128)
            coherence = (
                float(np.abs(cross) ** 2 / (truth_power * pred_power))
                if truth_power > 0.0 and pred_power > 0.0
                else math.nan
            )
            bands[label] = {
                "stored_k_inclusive": [int(low), int(high)],
                "full_torus_n_inclusive": [int(5 * low), int(5 * high)],
                "power_ratio": pred_power / truth_power
                if truth_power > 0.0
                else math.nan,
                "realization_coherence": coherence,
            }
        fields[field] = {"bands": bands}
    return _json_safe({"target_count": int(candidate.shape[0]), "fields": fields})


def association_summary(
    predicted_variance: Sequence[float], squared_error: Sequence[float]
) -> dict[str, Any]:
    """Pearson and Spearman flow-dependence summaries across targets."""

    spread = _finite_real("predicted variance", np.asarray(predicted_variance))
    error = _finite_real("squared error", np.asarray(squared_error))
    if spread.ndim != 1 or spread.shape != error.shape or spread.size < 3:
        raise ValueError("association series must be equal one-dimensional arrays")

    def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
        centered_first = first - np.mean(first)
        centered_second = second - np.mean(second)
        denominator = math.sqrt(
            float(np.sum(centered_first**2) * np.sum(centered_second**2))
        )
        return (
            float(np.sum(centered_first * centered_second) / denominator)
            if denominator > 0.0
            else None
        )

    return {
        "target_count": int(spread.size),
        "pearson": correlation(spread, error),
        "spearman": correlation(rankdata(spread), rankdata(error)),
        "calibration_proof": False,
    }


def classify_localization(
    *,
    transport_quantities: Mapping[str, Mapping[str, Any]],
    history_aggregate_improvement_fraction: float,
    history_improved_block_count: int,
) -> dict[str, Any]:
    """Apply the prospectively frozen L1/L2/L4 evidence labels."""

    amplitude = []
    covariance = []
    for name, record in transport_quantities.items():
        metrics = record["covariance_decomposition"]
        local = float(metrics["local_corrected_spread_skill_ratio"])
        integrated = float(metrics["integrated_corrected_spread_skill_ratio"])
        multiplier = float(metrics["ensemble_to_error_coherence_multiplier_ratio"])
        counterfactual = float(
            metrics["counterfactual_local_spread_skill_after_same_factor"]
        )
        same_outer_side = (local < 0.67 and integrated < 0.67) or (
            local > 1.50 and integrated > 1.50
        )
        if same_outer_side and 0.67 <= multiplier <= 1.50:
            amplitude.append(name)
        if (
            0.80 <= local <= 1.25
            and integrated < 0.67
            and multiplier < 0.67
            and counterfactual > 1.50
        ):
            covariance.append(name)
    history = (
        float(history_aggregate_improvement_fraction) >= 0.02
        and int(history_improved_block_count) >= 5
    )
    return {
        "L1_predominantly_amplitude_limited": {
            "supported": len(amplitude) >= 3,
            "supporting_quantities": amplitude,
            "required_quantity_count": 3,
        },
        "L2_covariance_organization_limited": {
            "supported": len(covariance) >= 3,
            "supporting_quantities": covariance,
            "required_quantity_count": 3,
        },
        "L4_explicit_residual_history_signal": {
            "supported": history,
            "aggregate_H1_RMSE_improvement_fraction": float(
                history_aggregate_improvement_fraction
            ),
            "improved_chronological_comparison_count": int(
                history_improved_block_count
            ),
            "required_improvement_fraction": 0.02,
            "required_block_count": 5,
        },
        "scientific_acceptance_gate": False,
        "training_authorized": False,
        "O3_authorized": False,
        "assimilation_authorized": False,
        "held_out_85606_access_authorized": False,
    }
