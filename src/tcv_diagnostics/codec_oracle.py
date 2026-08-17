"""Streaming statistics for the Phase 2 O1 codec oracle.

This module is NumPy-only.  Model loading and GPU inference live in the locked
cluster tool; the reductions here are independently testable on synthetic
fields with known answers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .data_protocol import C5_FIELDS
from .metrics import toroidal_mode_numbers


PRIMARY_FIELDS = ("Ne", "Te", "Ti", "phi")
PRIMARY_CROSS_PAIRS = (("Ne", "phi"), ("Te", "phi"), ("Ti", "phi"))
MODE_BANDS = (
    ("low_nonaxisymmetric", 1, 3),
    ("coherent_study", 4, 5),
    ("upper_study", 6, 7),
    ("measured_high", 8, 16),
    ("remaining_resolved", 17, 44),
)

MATERIAL_FRACTION = 0.01
FIELD_RMSE_MAX = 0.10
FIELD_VARIANCE_RATIO_MIN = 0.80
FIELD_VARIANCE_RATIO_MAX = 1.20
SPECTRAL_POWER_RATIO_MIN = 0.80
SPECTRAL_POWER_RATIO_MAX = 1.25
SPECTRAL_COHERENCE_MIN = 0.90
CROSS_PHASE_ERROR_MAX_DEGREES = 15.0
CROSS_COHERENCE_CHANGE_MAX = 0.10
REQUIRED_PASSING_BLOCKS = 7


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float):
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    output = np.full(
        np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan
    )
    np.divide(
        numerator_array,
        denominator_array,
        out=output,
        where=denominator_array > 0,
    )
    return float(output) if output.ndim == 0 else output


def _coherence(
    cross: np.ndarray,
    auto_a: np.ndarray,
    auto_b: np.ndarray,
) -> np.ndarray:
    return _safe_ratio(np.abs(cross) ** 2, auto_a * auto_b)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    denominator = float(np.sum(weights[valid]))
    if denominator <= 0:
        return math.nan
    return float(np.sum(values[valid] * weights[valid]) / denominator)


def _complex_curve(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values)
    return {
        "real": np.real(array).astype(np.float64).tolist(),
        "imag": np.imag(array).astype(np.float64).tolist(),
    }


def _float_curve(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


@dataclass
class _FieldMoments:
    count: int
    error_sum: np.ndarray
    absolute_error_sum: np.ndarray
    squared_error_sum: np.ndarray
    truth_sum: np.ndarray
    truth_squared_sum: np.ndarray
    reconstruction_sum: np.ndarray
    reconstruction_squared_sum: np.ndarray

    @classmethod
    def empty(cls, channels: int) -> "_FieldMoments":
        zeros = lambda: np.zeros(channels, dtype=np.float64)
        return cls(0, zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros())

    def update(self, truth: np.ndarray, reconstruction: np.ndarray) -> None:
        reduction_axes = (0, 2, 3, 4)
        error = reconstruction - truth
        cells_per_channel = int(np.prod([truth.shape[axis] for axis in reduction_axes]))
        self.count += cells_per_channel
        self.error_sum += np.sum(error, axis=reduction_axes, dtype=np.float64)
        self.absolute_error_sum += np.sum(
            np.abs(error), axis=reduction_axes, dtype=np.float64
        )
        self.squared_error_sum += np.sum(
            error * error, axis=reduction_axes, dtype=np.float64
        )
        self.truth_sum += np.sum(truth, axis=reduction_axes, dtype=np.float64)
        self.truth_squared_sum += np.sum(
            truth * truth, axis=reduction_axes, dtype=np.float64
        )
        self.reconstruction_sum += np.sum(
            reconstruction, axis=reduction_axes, dtype=np.float64
        )
        self.reconstruction_squared_sum += np.sum(
            reconstruction * reconstruction,
            axis=reduction_axes,
            dtype=np.float64,
        )

    def merge(self, other: "_FieldMoments") -> None:
        self.count += other.count
        for name in (
            "error_sum",
            "absolute_error_sum",
            "squared_error_sum",
            "truth_sum",
            "truth_squared_sum",
            "reconstruction_sum",
            "reconstruction_squared_sum",
        ):
            getattr(self, name)[:] += getattr(other, name)

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count <= 0:
            raise ValueError("cannot finalize empty field moments")
        count = float(self.count)
        truth_mean = self.truth_sum / count
        reconstruction_mean = self.reconstruction_sum / count
        truth_variance = np.maximum(
            self.truth_squared_sum / count - truth_mean**2, 0.0
        )
        reconstruction_variance = np.maximum(
            self.reconstruction_squared_sum / count - reconstruction_mean**2,
            0.0,
        )
        return {
            "rmse": np.sqrt(self.squared_error_sum / count),
            "mae": self.absolute_error_sum / count,
            "bias": self.error_sum / count,
            "truth_mean": truth_mean,
            "reconstruction_mean": reconstruction_mean,
            "truth_variance": truth_variance,
            "reconstruction_variance": reconstruction_variance,
            "variance_ratio": _safe_ratio(
                reconstruction_variance, truth_variance
            ),
        }


class CodecMetricAccumulator:
    """Accumulate deterministic codec metrics over chronological frame chunks."""

    def __init__(
        self,
        *,
        n_z: int,
        fields: tuple[str, ...] = C5_FIELDS,
        zperiod: int = 5,
    ) -> None:
        if tuple(fields) != tuple(C5_FIELDS):
            raise ValueError("O1 currently requires the canonical C5 field order")
        if n_z < 2:
            raise ValueError("n_z must be at least two")
        self.fields = tuple(fields)
        self.field_index = {field: index for index, field in enumerate(self.fields)}
        self.n_z = int(n_z)
        self.zperiod = int(zperiod)
        self.n_modes = n_z // 2 + 1
        self.frames = 0
        self.samples = 0
        self.model_moments = _FieldMoments.empty(len(self.fields))
        self.phi_gauge_moments = _FieldMoments.empty(1)
        self.truth_power_sum = np.zeros((len(self.fields), self.n_modes))
        self.reconstruction_power_sum = np.zeros_like(self.truth_power_sum)
        self.truth_auto_sum = np.zeros_like(self.truth_power_sum)
        self.reconstruction_auto_sum = np.zeros_like(self.truth_power_sum)
        self.transfer_cross_sum = np.zeros_like(
            self.truth_power_sum, dtype=np.complex128
        )
        self.truth_pair_cross_sum = {
            pair: np.zeros(self.n_modes, dtype=np.complex128)
            for pair in PRIMARY_CROSS_PAIRS
        }
        self.reconstruction_pair_cross_sum = {
            pair: np.zeros(self.n_modes, dtype=np.complex128)
            for pair in PRIMARY_CROSS_PAIRS
        }
        self.nonpositive_density_count = 0
        self.minimum_reconstructed_density = math.inf

    def update(
        self,
        model_truth: np.ndarray,
        model_reconstruction: np.ndarray,
        linear_truth: np.ndarray,
        linear_reconstruction: np.ndarray,
    ) -> None:
        model_truth = _finite_real("model_truth", model_truth)
        model_reconstruction = _finite_real(
            "model_reconstruction", model_reconstruction
        )
        linear_truth = _finite_real("linear_truth", linear_truth)
        linear_reconstruction = _finite_real(
            "linear_reconstruction", linear_reconstruction
        )
        expected_rank = 5
        arrays = (
            model_truth,
            model_reconstruction,
            linear_truth,
            linear_reconstruction,
        )
        if any(array.ndim != expected_rank for array in arrays):
            raise ValueError("codec arrays must have axes [T,C,X,Y,Z]")
        if any(array.shape != model_truth.shape for array in arrays[1:]):
            raise ValueError("truth and reconstruction arrays must have equal shape")
        if model_truth.shape[1] != len(self.fields) or model_truth.shape[-1] != self.n_z:
            raise ValueError(
                f"expected [T,{len(self.fields)},X,Y,{self.n_z}], got "
                f"{model_truth.shape}"
            )

        self.frames += model_truth.shape[0]
        self.samples += model_truth.shape[0] * model_truth.shape[2] * model_truth.shape[3]
        self.model_moments.update(model_truth, model_reconstruction)

        phi_index = self.field_index["phi"]
        phi_truth = model_truth[:, phi_index : phi_index + 1]
        phi_reconstruction = model_reconstruction[:, phi_index : phi_index + 1]
        spatial_axes = (2, 3, 4)
        phi_truth = phi_truth - np.mean(phi_truth, axis=spatial_axes, keepdims=True)
        phi_reconstruction = phi_reconstruction - np.mean(
            phi_reconstruction, axis=spatial_axes, keepdims=True
        )
        self.phi_gauge_moments.update(phi_truth, phi_reconstruction)

        density = linear_reconstruction[:, self.field_index["Ne"]]
        self.nonpositive_density_count += int(np.count_nonzero(density <= 0.0))
        self.minimum_reconstructed_density = min(
            self.minimum_reconstructed_density, float(np.min(density))
        )

        truth_coefficients = np.fft.rfft(linear_truth, axis=-1)
        reconstruction_coefficients = np.fft.rfft(linear_reconstruction, axis=-1)
        reduction_axes = (0, 2, 3)
        truth_auto = np.sum(
            np.abs(truth_coefficients) ** 2,
            axis=reduction_axes,
            dtype=np.float64,
        )
        reconstruction_auto = np.sum(
            np.abs(reconstruction_coefficients) ** 2,
            axis=reduction_axes,
            dtype=np.float64,
        )
        self.truth_auto_sum += truth_auto
        self.reconstruction_auto_sum += reconstruction_auto
        self.transfer_cross_sum += np.sum(
            truth_coefficients * np.conjugate(reconstruction_coefficients),
            axis=reduction_axes,
            dtype=np.complex128,
        )

        weights = np.ones(self.n_modes, dtype=np.float64)
        if self.n_z % 2 == 0:
            weights[1:-1] = 2.0
        else:
            weights[1:] = 2.0
        scale = weights[None, :] / float(self.n_z * self.n_z)
        self.truth_power_sum += truth_auto * scale
        self.reconstruction_power_sum += reconstruction_auto * scale

        for pair in PRIMARY_CROSS_PAIRS:
            first = self.field_index[pair[0]]
            second = self.field_index[pair[1]]
            self.truth_pair_cross_sum[pair] += np.sum(
                truth_coefficients[:, first]
                * np.conjugate(truth_coefficients[:, second]),
                axis=(0, 1, 2),
                dtype=np.complex128,
            )
            self.reconstruction_pair_cross_sum[pair] += np.sum(
                reconstruction_coefficients[:, first]
                * np.conjugate(reconstruction_coefficients[:, second]),
                axis=(0, 1, 2),
                dtype=np.complex128,
            )

    def merge(self, other: "CodecMetricAccumulator") -> None:
        if (
            self.fields != other.fields
            or self.n_z != other.n_z
            or self.zperiod != other.zperiod
        ):
            raise ValueError("cannot merge accumulators with different conventions")
        self.frames += other.frames
        self.samples += other.samples
        self.model_moments.merge(other.model_moments)
        self.phi_gauge_moments.merge(other.phi_gauge_moments)
        for name in (
            "truth_power_sum",
            "reconstruction_power_sum",
            "truth_auto_sum",
            "reconstruction_auto_sum",
            "transfer_cross_sum",
        ):
            getattr(self, name)[:] += getattr(other, name)
        for pair in PRIMARY_CROSS_PAIRS:
            self.truth_pair_cross_sum[pair] += other.truth_pair_cross_sum[pair]
            self.reconstruction_pair_cross_sum[pair] += (
                other.reconstruction_pair_cross_sum[pair]
            )
        self.nonpositive_density_count += other.nonpositive_density_count
        self.minimum_reconstructed_density = min(
            self.minimum_reconstructed_density,
            other.minimum_reconstructed_density,
        )

    def _field_band_summaries(
        self,
        truth_power: np.ndarray,
        reconstruction_power: np.ndarray,
        transfer_coherence: np.ndarray,
    ) -> dict[str, dict[str, dict[str, float | bool | int]]]:
        summaries: dict[str, dict[str, dict[str, float | bool | int]]] = {}
        for field_index, field in enumerate(self.fields):
            nonaxisymmetric_total = float(np.sum(truth_power[field_index, 1:]))
            field_bands: dict[str, dict[str, float | bool | int]] = {}
            for label, low, high in MODE_BANDS:
                high = min(high, self.n_modes - 1)
                indices = np.arange(low, high + 1) if low <= high else np.asarray([], dtype=int)
                band_truth = float(np.sum(truth_power[field_index, indices]))
                band_reconstruction = float(
                    np.sum(reconstruction_power[field_index, indices])
                )
                fraction = (
                    band_truth / nonaxisymmetric_total
                    if nonaxisymmetric_total > 0
                    else math.nan
                )
                field_bands[label] = {
                    "k_low": low,
                    "k_high": high,
                    "n_low": low * self.zperiod,
                    "n_high": high * self.zperiod,
                    "truth_power": band_truth,
                    "truth_power_fraction": fraction,
                    "material": bool(
                        np.isfinite(fraction) and fraction >= MATERIAL_FRACTION
                    ),
                    "power_ratio": _safe_ratio(
                        band_reconstruction, band_truth
                    ),
                    "truth_power_weighted_transfer_coherence": _weighted_mean(
                        transfer_coherence[field_index, indices],
                        truth_power[field_index, indices],
                    ),
                }
            summaries[field] = field_bands
        return summaries

    def _cross_band_summaries(
        self,
        truth_pair_cross: dict[tuple[str, str], np.ndarray],
        reconstruction_pair_cross: dict[tuple[str, str], np.ndarray],
        truth_coherence: dict[tuple[str, str], np.ndarray],
        reconstruction_coherence: dict[tuple[str, str], np.ndarray],
    ) -> dict[str, dict[str, dict[str, float | bool | int]]]:
        summaries: dict[str, dict[str, dict[str, float | bool | int]]] = {}
        for pair in PRIMARY_CROSS_PAIRS:
            key = f"{pair[0]}-{pair[1]}"
            truth_amplitude = np.abs(truth_pair_cross[pair])
            total = float(np.sum(truth_amplitude[1:]))
            phase_error = np.angle(
                np.exp(
                    1j
                    * (
                        np.angle(reconstruction_pair_cross[pair])
                        - np.angle(truth_pair_cross[pair])
                    )
                )
            )
            coherence_change = np.abs(
                reconstruction_coherence[pair] - truth_coherence[pair]
            )
            pair_bands: dict[str, dict[str, float | bool | int]] = {}
            for label, low, high in MODE_BANDS:
                high = min(high, self.n_modes - 1)
                indices = np.arange(low, high + 1) if low <= high else np.asarray([], dtype=int)
                amplitude = float(np.sum(truth_amplitude[indices]))
                fraction = amplitude / total if total > 0 else math.nan
                pair_bands[label] = {
                    "k_low": low,
                    "k_high": high,
                    "n_low": low * self.zperiod,
                    "n_high": high * self.zperiod,
                    "truth_cross_amplitude": amplitude,
                    "truth_cross_amplitude_fraction": fraction,
                    "material": bool(
                        np.isfinite(fraction) and fraction >= MATERIAL_FRACTION
                    ),
                    "truth_cross_amplitude_weighted_absolute_phase_error_degrees": (
                        math.degrees(
                            _weighted_mean(
                                np.abs(phase_error[indices]),
                                truth_amplitude[indices],
                            )
                        )
                    ),
                    "truth_cross_amplitude_weighted_absolute_coherence_change": (
                        _weighted_mean(
                            coherence_change[indices], truth_amplitude[indices]
                        )
                    ),
                }
            summaries[key] = pair_bands
        return summaries

    def finalize(self) -> dict[str, Any]:
        if self.frames <= 0 or self.samples <= 0:
            raise ValueError("cannot finalize empty codec metrics")
        moments = self.model_moments.finalize()
        phi_gauge = self.phi_gauge_moments.finalize()
        truth_power = self.truth_power_sum / float(self.samples)
        reconstruction_power = self.reconstruction_power_sum / float(self.samples)
        transfer_coherence = _coherence(
            self.transfer_cross_sum,
            self.truth_auto_sum,
            self.reconstruction_auto_sum,
        )
        transfer_phase = np.angle(self.transfer_cross_sum)
        stored_k, full_torus_n = toroidal_mode_numbers(
            self.n_z, zperiod=self.zperiod
        )

        truth_pair_coherence: dict[tuple[str, str], np.ndarray] = {}
        reconstruction_pair_coherence: dict[tuple[str, str], np.ndarray] = {}
        cross_field_curves: dict[str, Any] = {}
        for pair in PRIMARY_CROSS_PAIRS:
            first = self.field_index[pair[0]]
            second = self.field_index[pair[1]]
            truth_pair_coherence[pair] = _coherence(
                self.truth_pair_cross_sum[pair],
                self.truth_auto_sum[first],
                self.truth_auto_sum[second],
            )
            reconstruction_pair_coherence[pair] = _coherence(
                self.reconstruction_pair_cross_sum[pair],
                self.reconstruction_auto_sum[first],
                self.reconstruction_auto_sum[second],
            )
            phase_error = np.angle(
                np.exp(
                    1j
                    * (
                        np.angle(self.reconstruction_pair_cross_sum[pair])
                        - np.angle(self.truth_pair_cross_sum[pair])
                    )
                )
            )
            key = f"{pair[0]}-{pair[1]}"
            cross_field_curves[key] = {
                "truth_cross_spectrum_sum": _complex_curve(
                    self.truth_pair_cross_sum[pair]
                ),
                "reconstruction_cross_spectrum_sum": _complex_curve(
                    self.reconstruction_pair_cross_sum[pair]
                ),
                "truth_coherence": _float_curve(truth_pair_coherence[pair]),
                "reconstruction_coherence": _float_curve(
                    reconstruction_pair_coherence[pair]
                ),
                "truth_phase_radians": _float_curve(
                    np.angle(self.truth_pair_cross_sum[pair])
                ),
                "reconstruction_phase_radians": _float_curve(
                    np.angle(self.reconstruction_pair_cross_sum[pair])
                ),
                "signed_phase_error_radians": _float_curve(phase_error),
            }

        field_metrics: dict[str, Any] = {}
        spectral_curves: dict[str, Any] = {}
        for index, field in enumerate(self.fields):
            field_metrics[field] = {
                key: float(value[index]) for key, value in moments.items()
            }
            spectral_curves[field] = {
                "truth_power": _float_curve(truth_power[index]),
                "reconstruction_power": _float_curve(
                    reconstruction_power[index]
                ),
                "power_ratio": _float_curve(
                    _safe_ratio(reconstruction_power[index], truth_power[index])
                ),
                "truth_to_reconstruction_cross_spectrum_sum": _complex_curve(
                    self.transfer_cross_sum[index]
                ),
                "truth_to_reconstruction_coherence": _float_curve(
                    transfer_coherence[index]
                ),
                "truth_to_reconstruction_phase_radians": _float_curve(
                    transfer_phase[index]
                ),
            }

        aggregate_squared_error = float(np.sum(self.model_moments.squared_error_sum))
        aggregate_count = self.model_moments.count * len(self.fields)
        return {
            "frames": self.frames,
            "spatial_samples_per_field": self.samples * self.n_z,
            "stored_k": stored_k.tolist(),
            "full_torus_n": full_torus_n.tolist(),
            "field_metrics_legacy_standardized": field_metrics,
            "aggregate_five_field_rmse_legacy_standardized": math.sqrt(
                aggregate_squared_error / aggregate_count
            ),
            "phi_gauge_fixed_metrics_legacy_standardized": {
                key: float(value[0]) for key, value in phi_gauge.items()
            },
            "density_linear_reconstruction": {
                "nonpositive_cell_count": self.nonpositive_density_count,
                "minimum": self.minimum_reconstructed_density,
            },
            "toroidal_spectral_curves_linear_coordinates": spectral_curves,
            "field_band_summaries": self._field_band_summaries(
                truth_power, reconstruction_power, transfer_coherence
            ),
            "cross_field_curves_linear_coordinates": cross_field_curves,
            "cross_field_band_summaries": self._cross_band_summaries(
                self.truth_pair_cross_sum,
                self.reconstruction_pair_cross_sum,
                truth_pair_coherence,
                reconstruction_pair_coherence,
            ),
        }


def _field_reconstruction_condition(metrics: dict[str, Any], field: str) -> bool:
    values = metrics["field_metrics_legacy_standardized"][field]
    return bool(
        values["rmse"] <= FIELD_RMSE_MAX
        and FIELD_VARIANCE_RATIO_MIN
        <= values["variance_ratio"]
        <= FIELD_VARIANCE_RATIO_MAX
    )


def _spectral_condition(metrics: dict[str, Any], field: str, band: str) -> bool:
    values = metrics["field_band_summaries"][field][band]
    return bool(
        np.isfinite(values["power_ratio"])
        and SPECTRAL_POWER_RATIO_MIN
        <= values["power_ratio"]
        <= SPECTRAL_POWER_RATIO_MAX
        and np.isfinite(values["truth_power_weighted_transfer_coherence"])
        and values["truth_power_weighted_transfer_coherence"]
        >= SPECTRAL_COHERENCE_MIN
    )


def _cross_condition(metrics: dict[str, Any], pair: str, band: str) -> bool:
    values = metrics["cross_field_band_summaries"][pair][band]
    phase_error = values[
        "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
    ]
    coherence_change = values[
        "truth_cross_amplitude_weighted_absolute_coherence_change"
    ]
    return bool(
        np.isfinite(phase_error)
        and phase_error <= CROSS_PHASE_ERROR_MAX_DEGREES
        and np.isfinite(coherence_change)
        and coherence_change <= CROSS_COHERENCE_CHANGE_MAX
    )


def build_preliminary_gate(
    overall: dict[str, Any],
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the prospectively frozen O1 gates to finalized metric records."""

    if len(blocks) != 8:
        raise ValueError("the O1 gate requires exactly eight temporal blocks")

    field_checks: dict[str, Any] = {}
    for field in C5_FIELDS:
        block_values = [
            _field_reconstruction_condition(block, field) for block in blocks
        ]
        field_checks[field] = {
            "overall_pass": _field_reconstruction_condition(overall, field),
            "passing_blocks": int(sum(block_values)),
            "block_pass": block_values,
        }
    field_pass = all(
        item["overall_pass"]
        and item["passing_blocks"] >= REQUIRED_PASSING_BLOCKS
        for item in field_checks.values()
    )

    spectral_checks: dict[str, Any] = {}
    for field in PRIMARY_FIELDS:
        spectral_checks[field] = {}
        for band in ("low_nonaxisymmetric", "coherent_study", "upper_study"):
            material = bool(
                overall["field_band_summaries"][field][band]["material"]
            )
            block_values = [
                _spectral_condition(block, field, band) for block in blocks
            ]
            spectral_checks[field][band] = {
                "material_overall": material,
                "overall_pass_if_material": (
                    _spectral_condition(overall, field, band) if material else None
                ),
                "passing_blocks_if_material": (
                    int(sum(block_values)) if material else None
                ),
                "block_pass_if_material": block_values if material else None,
            }
    spectral_applicable = [
        item
        for field in spectral_checks.values()
        for item in field.values()
        if item["material_overall"]
    ]
    spectral_pass = bool(spectral_applicable) and all(
        item["overall_pass_if_material"]
        and item["passing_blocks_if_material"] >= REQUIRED_PASSING_BLOCKS
        for item in spectral_applicable
    )

    cross_checks: dict[str, Any] = {}
    for pair in ("Ne-phi", "Te-phi", "Ti-phi"):
        cross_checks[pair] = {}
        for band in ("low_nonaxisymmetric", "coherent_study", "upper_study"):
            material = bool(
                overall["cross_field_band_summaries"][pair][band]["material"]
            )
            block_values = [
                _cross_condition(block, pair, band) for block in blocks
            ]
            cross_checks[pair][band] = {
                "material_overall": material,
                "overall_pass_if_material": (
                    _cross_condition(overall, pair, band) if material else None
                ),
                "passing_blocks_if_material": (
                    int(sum(block_values)) if material else None
                ),
                "block_pass_if_material": block_values if material else None,
            }
    cross_applicable = [
        item
        for pair in cross_checks.values()
        for item in pair.values()
        if item["material_overall"]
    ]
    cross_pass = bool(cross_applicable) and all(
        item["overall_pass_if_material"]
        and item["passing_blocks_if_material"] >= REQUIRED_PASSING_BLOCKS
        for item in cross_applicable
    )

    preliminary_pass = field_pass and spectral_pass and cross_pass
    return {
        "thresholds": {
            "field_rmse_max": FIELD_RMSE_MAX,
            "field_variance_ratio": [
                FIELD_VARIANCE_RATIO_MIN,
                FIELD_VARIANCE_RATIO_MAX,
            ],
            "material_fraction_min": MATERIAL_FRACTION,
            "spectral_power_ratio": [
                SPECTRAL_POWER_RATIO_MIN,
                SPECTRAL_POWER_RATIO_MAX,
            ],
            "spectral_transfer_coherence_min": SPECTRAL_COHERENCE_MIN,
            "cross_phase_error_max_degrees": CROSS_PHASE_ERROR_MAX_DEGREES,
            "cross_coherence_change_max": CROSS_COHERENCE_CHANGE_MAX,
            "required_passing_blocks": REQUIRED_PASSING_BLOCKS,
        },
        "field_reconstruction": {
            "pass": field_pass,
            "checks": field_checks,
        },
        "spectral_transfer": {
            "pass": spectral_pass,
            "applicable_check_count": len(spectral_applicable),
            "checks": spectral_checks,
        },
        "cross_field": {
            "pass": cross_pass,
            "applicable_check_count": len(cross_applicable),
            "checks": cross_checks,
        },
        "preliminary_status": "pass" if preliminary_pass else "fail",
        "full_codec_acceptance": "blocked_pending_authoritative_transport",
    }
