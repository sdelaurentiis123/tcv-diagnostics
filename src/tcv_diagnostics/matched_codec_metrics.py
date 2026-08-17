"""Matched O1 codec metrics for the frozen 85604 development protocol.

The accumulator deliberately separates standardized field reconstruction from
physical-coordinate spectra and cross spectra.  Training-truth materiality is
extracted once and then supplied unchanged to the validation gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np

from .metrics import toroidal_mode_numbers


MODE_BANDS = (
    ("k1_3", 1, 3),
    ("k4_5", 4, 5),
    ("k6_7", 6, 7),
)
MATERIAL_FRACTION_MIN = 0.01
FIELD_RMSE_MAX = 0.10
FIELD_VARIANCE_RATIO = (0.80, 1.20)
SPECTRAL_POWER_RATIO = (0.80, 1.25)
SPECTRAL_TRANSFER_COHERENCE_MIN = 0.90
CROSS_PHASE_ERROR_MAX_DEGREES = 15.0
CROSS_COHERENCE_CHANGE_MAX = 0.10
REQUIRED_PASSING_BLOCKS = 7


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float):
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    output = np.full(
        np.broadcast_shapes(numerator_array.shape, denominator_array.shape),
        np.nan,
    )
    np.divide(
        numerator_array,
        denominator_array,
        out=output,
        where=denominator_array > 0.0,
    )
    return float(output) if output.ndim == 0 else output


def _coherence(
    cross: np.ndarray,
    auto_a: np.ndarray,
    auto_b: np.ndarray,
) -> np.ndarray:
    return _safe_ratio(np.abs(cross) ** 2, auto_a * auto_b)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (weights_array >= 0.0)
    )
    denominator = float(np.sum(weights_array[valid]))
    if denominator <= 0.0:
        return math.nan
    return float(
        np.sum(values_array[valid] * weights_array[valid]) / denominator
    )


def _float_curve(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _complex_curve(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values)
    return {
        "real": np.real(array).astype(np.float64).tolist(),
        "imag": np.imag(array).astype(np.float64).tolist(),
    }


@dataclass(frozen=True)
class CodecViewSpec:
    name: str
    fields: tuple[str, ...]
    spectral_fields: tuple[str, ...]
    cross_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.fields:
            raise ValueError("a codec view needs a name and at least one field")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("codec view fields must be unique")
        if not self.spectral_fields:
            raise ValueError("at least one spectral field is required")
        if not set(self.spectral_fields).issubset(self.fields):
            raise ValueError("spectral fields must belong to the view")
        if len(set(self.spectral_fields)) != len(self.spectral_fields):
            raise ValueError("spectral fields must be unique")
        if len(set(self.cross_pairs)) != len(self.cross_pairs):
            raise ValueError("cross pairs must be unique")
        for pair in self.cross_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("cross pairs require two distinct fields")
            if not set(pair).issubset(self.spectral_fields):
                raise ValueError("cross-pair fields must be spectral fields")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["fields"] = list(self.fields)
        record["spectral_fields"] = list(self.spectral_fields)
        record["cross_pairs"] = [list(pair) for pair in self.cross_pairs]
        return record


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
        def zeros() -> np.ndarray:
            return np.zeros(channels, dtype=np.float64)

        return cls(0, zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros())

    def update(self, truth: np.ndarray, reconstruction: np.ndarray) -> None:
        reduction_axes = (0, 2, 3, 4)
        error = reconstruction - truth
        count = int(np.prod([truth.shape[axis] for axis in reduction_axes]))
        self.count += count
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
            self.truth_squared_sum / count - truth_mean**2,
            0.0,
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
                reconstruction_variance,
                truth_variance,
            ),
        }


class MatchedCodecAccumulator:
    """Stream one native or common O1 codec view in chronological chunks."""

    def __init__(
        self,
        *,
        spec: CodecViewSpec,
        n_z: int = 88,
        zperiod: int = 5,
    ) -> None:
        if n_z < 2:
            raise ValueError("n_z must be at least two")
        if zperiod <= 0:
            raise ValueError("zperiod must be positive")
        self.spec = spec
        self.n_z = int(n_z)
        self.zperiod = int(zperiod)
        self.n_modes = self.n_z // 2 + 1
        self.field_index = {field: index for index, field in enumerate(spec.fields)}
        self.spectral_index = {
            field: index for index, field in enumerate(spec.spectral_fields)
        }
        spectral_count = len(spec.spectral_fields)
        self.frames = 0
        self.xy_samples = 0
        self.field_moments = _FieldMoments.empty(len(spec.fields))
        self.truth_power_sum = np.zeros((spectral_count, self.n_modes))
        self.reconstruction_power_sum = np.zeros_like(self.truth_power_sum)
        self.truth_auto_sum = np.zeros_like(self.truth_power_sum)
        self.reconstruction_auto_sum = np.zeros_like(self.truth_power_sum)
        self.transfer_cross_sum = np.zeros(
            (spectral_count, self.n_modes), dtype=np.complex128
        )
        self.truth_pair_cross_sum = {
            pair: np.zeros(self.n_modes, dtype=np.complex128)
            for pair in spec.cross_pairs
        }
        self.reconstruction_pair_cross_sum = {
            pair: np.zeros(self.n_modes, dtype=np.complex128)
            for pair in spec.cross_pairs
        }
        self.nonpositive_density_count = 0
        self.minimum_reconstructed_density = math.inf

    def update(
        self,
        standardized_truth: np.ndarray,
        standardized_reconstruction: np.ndarray,
        physical_truth: np.ndarray,
        physical_reconstruction: np.ndarray,
    ) -> None:
        arrays = tuple(
            _finite_real(name, value)
            for name, value in (
                ("standardized_truth", standardized_truth),
                ("standardized_reconstruction", standardized_reconstruction),
                ("physical_truth", physical_truth),
                ("physical_reconstruction", physical_reconstruction),
            )
        )
        reference_shape = arrays[0].shape
        if any(array.ndim != 5 for array in arrays):
            raise ValueError("codec arrays must have axes [T,C,X,Y,Z]")
        if any(array.shape != reference_shape for array in arrays[1:]):
            raise ValueError("all truth and reconstruction arrays must have equal shape")
        if reference_shape[1] != len(self.spec.fields):
            raise ValueError("codec channel count differs from the view")
        if reference_shape[-1] != self.n_z:
            raise ValueError("codec toroidal size differs from the view")
        (
            standardized_truth,
            standardized_reconstruction,
            physical_truth,
            physical_reconstruction,
        ) = arrays

        self.frames += reference_shape[0]
        self.xy_samples += reference_shape[0] * reference_shape[2] * reference_shape[3]
        self.field_moments.update(
            standardized_truth,
            standardized_reconstruction,
        )
        if "Ne" in self.field_index:
            density = physical_reconstruction[:, self.field_index["Ne"]]
            self.nonpositive_density_count += int(np.count_nonzero(density <= 0.0))
            self.minimum_reconstructed_density = min(
                self.minimum_reconstructed_density,
                float(np.min(density)),
            )

        indices = [self.field_index[field] for field in self.spec.spectral_fields]
        truth_coefficients = np.fft.rfft(physical_truth[:, indices], axis=-1)
        reconstruction_coefficients = np.fft.rfft(
            physical_reconstruction[:, indices], axis=-1
        )
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

        for pair in self.spec.cross_pairs:
            first = self.spectral_index[pair[0]]
            second = self.spectral_index[pair[1]]
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

    def merge(self, other: "MatchedCodecAccumulator") -> None:
        if (
            self.spec != other.spec
            or self.n_z != other.n_z
            or self.zperiod != other.zperiod
        ):
            raise ValueError("cannot merge accumulators with different conventions")
        self.frames += other.frames
        self.xy_samples += other.xy_samples
        self.field_moments.merge(other.field_moments)
        for name in (
            "truth_power_sum",
            "reconstruction_power_sum",
            "truth_auto_sum",
            "reconstruction_auto_sum",
            "transfer_cross_sum",
        ):
            getattr(self, name)[:] += getattr(other, name)
        for pair in self.spec.cross_pairs:
            self.truth_pair_cross_sum[pair] += other.truth_pair_cross_sum[pair]
            self.reconstruction_pair_cross_sum[pair] += (
                other.reconstruction_pair_cross_sum[pair]
            )
        self.nonpositive_density_count += other.nonpositive_density_count
        self.minimum_reconstructed_density = min(
            self.minimum_reconstructed_density,
            other.minimum_reconstructed_density,
        )

    @staticmethod
    def _band_indices(n_modes: int, low: int, high: int) -> np.ndarray:
        bounded_high = min(high, n_modes - 1)
        if low > bounded_high:
            return np.asarray([], dtype=np.int64)
        return np.arange(low, bounded_high + 1, dtype=np.int64)

    def finalize(self) -> dict[str, Any]:
        if self.frames <= 0 or self.xy_samples <= 0:
            raise ValueError("cannot finalize empty codec metrics")
        moments = self.field_moments.finalize()
        truth_power = self.truth_power_sum / float(self.xy_samples)
        reconstruction_power = self.reconstruction_power_sum / float(
            self.xy_samples
        )
        transfer_coherence = _coherence(
            self.transfer_cross_sum,
            self.truth_auto_sum,
            self.reconstruction_auto_sum,
        )
        stored_k, full_torus_n = toroidal_mode_numbers(
            self.n_z,
            zperiod=self.zperiod,
        )

        field_metrics = {
            field: {key: float(value[index]) for key, value in moments.items()}
            for index, field in enumerate(self.spec.fields)
        }
        spectral_curves: dict[str, Any] = {}
        field_bands: dict[str, Any] = {}
        for index, field in enumerate(self.spec.spectral_fields):
            spectral_curves[field] = {
                "truth_power": _float_curve(truth_power[index]),
                "reconstruction_power": _float_curve(reconstruction_power[index]),
                "power_ratio": _float_curve(
                    _safe_ratio(reconstruction_power[index], truth_power[index])
                ),
                "truth_to_reconstruction_cross_spectrum_sum": _complex_curve(
                    self.transfer_cross_sum[index]
                ),
                "truth_to_reconstruction_coherence": _float_curve(
                    transfer_coherence[index]
                ),
            }
            nonaxisymmetric_total = float(np.sum(truth_power[index, 1:]))
            field_bands[field] = {}
            for label, low, high in MODE_BANDS:
                band = self._band_indices(self.n_modes, low, high)
                band_truth = float(np.sum(truth_power[index, band]))
                fraction = (
                    band_truth / nonaxisymmetric_total
                    if nonaxisymmetric_total > 0.0
                    else math.nan
                )
                field_bands[field][label] = {
                    "k_low": low,
                    "k_high": min(high, self.n_modes - 1),
                    "n_low": low * self.zperiod,
                    "n_high": min(high, self.n_modes - 1) * self.zperiod,
                    "truth_power": band_truth,
                    "truth_nonaxisymmetric_power_fraction": fraction,
                    "power_ratio": _safe_ratio(
                        float(np.sum(reconstruction_power[index, band])),
                        band_truth,
                    ),
                    "truth_power_weighted_transfer_coherence": _weighted_mean(
                        transfer_coherence[index, band],
                        truth_power[index, band],
                    ),
                }

        cross_curves: dict[str, Any] = {}
        cross_bands: dict[str, Any] = {}
        for pair in self.spec.cross_pairs:
            first = self.spectral_index[pair[0]]
            second = self.spectral_index[pair[1]]
            truth_coherence = _coherence(
                self.truth_pair_cross_sum[pair],
                self.truth_auto_sum[first],
                self.truth_auto_sum[second],
            )
            reconstruction_coherence = _coherence(
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
            cross_curves[key] = {
                "truth_cross_spectrum_sum": _complex_curve(
                    self.truth_pair_cross_sum[pair]
                ),
                "reconstruction_cross_spectrum_sum": _complex_curve(
                    self.reconstruction_pair_cross_sum[pair]
                ),
                "truth_coherence": _float_curve(truth_coherence),
                "reconstruction_coherence": _float_curve(
                    reconstruction_coherence
                ),
                "signed_phase_error_radians": _float_curve(phase_error),
            }
            truth_amplitude = np.abs(self.truth_pair_cross_sum[pair])
            nonaxisymmetric_total = float(np.sum(truth_amplitude[1:]))
            coherence_change = np.abs(
                reconstruction_coherence - truth_coherence
            )
            cross_bands[key] = {}
            for label, low, high in MODE_BANDS:
                band = self._band_indices(self.n_modes, low, high)
                amplitude = float(np.sum(truth_amplitude[band]))
                fraction = (
                    amplitude / nonaxisymmetric_total
                    if nonaxisymmetric_total > 0.0
                    else math.nan
                )
                cross_bands[key][label] = {
                    "k_low": low,
                    "k_high": min(high, self.n_modes - 1),
                    "n_low": low * self.zperiod,
                    "n_high": min(high, self.n_modes - 1) * self.zperiod,
                    "truth_cross_amplitude": amplitude,
                    "truth_nonaxisymmetric_cross_amplitude_fraction": fraction,
                    "truth_cross_amplitude_weighted_absolute_phase_error_degrees": (
                        math.degrees(
                            _weighted_mean(
                                np.abs(phase_error[band]),
                                truth_amplitude[band],
                            )
                        )
                    ),
                    "truth_cross_amplitude_weighted_absolute_coherence_change": (
                        _weighted_mean(
                            coherence_change[band],
                            truth_amplitude[band],
                        )
                    ),
                }

        aggregate_squared_error = float(np.sum(self.field_moments.squared_error_sum))
        aggregate_count = self.field_moments.count * len(self.spec.fields)
        density_record = None
        if "Ne" in self.field_index:
            density_record = {
                "nonpositive_cell_count": self.nonpositive_density_count,
                "minimum": self.minimum_reconstructed_density,
            }
        return {
            "view": self.spec.to_record(),
            "frames": self.frames,
            "spatial_samples_per_field": self.xy_samples * self.n_z,
            "stored_k": stored_k.tolist(),
            "full_torus_n": full_torus_n.tolist(),
            "field_metrics_standardized": field_metrics,
            "aggregate_equal_channel_rmse_standardized": math.sqrt(
                aggregate_squared_error / aggregate_count
            ),
            "density_physical_reconstruction": density_record,
            "toroidal_spectral_curves_physical": spectral_curves,
            "field_band_summaries": field_bands,
            "cross_field_curves_physical": cross_curves,
            "cross_field_band_summaries": cross_bands,
        }


def training_materiality(training_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze applicable bands from training truth, never validation truth."""

    field_materiality = {
        field: {
            band: {
                "truth_fraction": float(
                    values["truth_nonaxisymmetric_power_fraction"]
                ),
                "material": bool(
                    np.isfinite(values["truth_nonaxisymmetric_power_fraction"])
                    and values["truth_nonaxisymmetric_power_fraction"]
                    >= MATERIAL_FRACTION_MIN
                ),
            }
            for band, values in bands.items()
        }
        for field, bands in training_metrics["field_band_summaries"].items()
    }
    cross_materiality = {
        pair: {
            band: {
                "truth_fraction": float(
                    values["truth_nonaxisymmetric_cross_amplitude_fraction"]
                ),
                "material": bool(
                    np.isfinite(
                        values[
                            "truth_nonaxisymmetric_cross_amplitude_fraction"
                        ]
                    )
                    and values[
                        "truth_nonaxisymmetric_cross_amplitude_fraction"
                    ]
                    >= MATERIAL_FRACTION_MIN
                ),
            }
            for band, values in bands.items()
        }
        for pair, bands in training_metrics["cross_field_band_summaries"].items()
    }
    return {
        "source_split": "85604_training_[0,432)",
        "minimum_fraction": MATERIAL_FRACTION_MIN,
        "view": dict(training_metrics["view"]),
        "fields": field_materiality,
        "cross_pairs": cross_materiality,
    }


def _field_pass(metrics: Mapping[str, Any], field: str) -> bool:
    values = metrics["field_metrics_standardized"][field]
    return bool(
        np.isfinite(values["rmse"])
        and values["rmse"] <= FIELD_RMSE_MAX
        and FIELD_VARIANCE_RATIO[0]
        <= values["variance_ratio"]
        <= FIELD_VARIANCE_RATIO[1]
    )


def _spectral_pass(metrics: Mapping[str, Any], field: str, band: str) -> bool:
    values = metrics["field_band_summaries"][field][band]
    return bool(
        np.isfinite(values["power_ratio"])
        and SPECTRAL_POWER_RATIO[0]
        <= values["power_ratio"]
        <= SPECTRAL_POWER_RATIO[1]
        and np.isfinite(values["truth_power_weighted_transfer_coherence"])
        and values["truth_power_weighted_transfer_coherence"]
        >= SPECTRAL_TRANSFER_COHERENCE_MIN
    )


def _cross_pass(metrics: Mapping[str, Any], pair: str, band: str) -> bool:
    values = metrics["cross_field_band_summaries"][pair][band]
    phase = values[
        "truth_cross_amplitude_weighted_absolute_phase_error_degrees"
    ]
    coherence = values[
        "truth_cross_amplitude_weighted_absolute_coherence_change"
    ]
    return bool(
        np.isfinite(phase)
        and phase <= CROSS_PHASE_ERROR_MAX_DEGREES
        and np.isfinite(coherence)
        and coherence <= CROSS_COHERENCE_CHANGE_MAX
    )


def build_matched_o1_view_gate(
    *,
    validation_overall: Mapping[str, Any],
    validation_blocks: list[Mapping[str, Any]],
    materiality: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen full-interval plus 7-of-8 validation rules."""

    if len(validation_blocks) != 8:
        raise ValueError("the O1 gate requires eight 16-frame validation blocks")
    if dict(validation_overall["view"]) != dict(materiality["view"]):
        raise ValueError("training materiality and validation view differ")
    if any(int(block["frames"]) != 16 for block in validation_blocks):
        raise ValueError("every O1 validation block must contain 16 frames")

    fields: dict[str, Any] = {}
    for field in validation_overall["view"]["fields"]:
        block_pass = [_field_pass(block, field) for block in validation_blocks]
        fields[field] = {
            "overall_pass": _field_pass(validation_overall, field),
            "passing_blocks": int(sum(block_pass)),
            "block_pass": block_pass,
        }
    field_pass = all(
        item["overall_pass"]
        and item["passing_blocks"] >= REQUIRED_PASSING_BLOCKS
        for item in fields.values()
    )

    spectral: dict[str, Any] = {}
    spectral_applicable: list[dict[str, Any]] = []
    for field, bands in materiality["fields"].items():
        spectral[field] = {}
        for band, frozen in bands.items():
            block_pass = [
                _spectral_pass(block, field, band)
                for block in validation_blocks
            ]
            record = {
                "material_from_training_truth": bool(frozen["material"]),
                "training_truth_fraction": float(frozen["truth_fraction"]),
                "overall_pass_if_material": (
                    _spectral_pass(validation_overall, field, band)
                    if frozen["material"]
                    else None
                ),
                "passing_blocks_if_material": (
                    int(sum(block_pass)) if frozen["material"] else None
                ),
                "block_pass_if_material": block_pass if frozen["material"] else None,
            }
            spectral[field][band] = record
            if frozen["material"]:
                spectral_applicable.append(record)
    spectral_pass = bool(spectral_applicable) and all(
        item["overall_pass_if_material"]
        and item["passing_blocks_if_material"] >= REQUIRED_PASSING_BLOCKS
        for item in spectral_applicable
    )

    cross: dict[str, Any] = {}
    cross_applicable: list[dict[str, Any]] = []
    for pair, bands in materiality["cross_pairs"].items():
        cross[pair] = {}
        for band, frozen in bands.items():
            block_pass = [
                _cross_pass(block, pair, band) for block in validation_blocks
            ]
            record = {
                "material_from_training_truth": bool(frozen["material"]),
                "training_truth_fraction": float(frozen["truth_fraction"]),
                "overall_pass_if_material": (
                    _cross_pass(validation_overall, pair, band)
                    if frozen["material"]
                    else None
                ),
                "passing_blocks_if_material": (
                    int(sum(block_pass)) if frozen["material"] else None
                ),
                "block_pass_if_material": block_pass if frozen["material"] else None,
            }
            cross[pair][band] = record
            if frozen["material"]:
                cross_applicable.append(record)
    cross_required = bool(materiality["cross_pairs"])
    cross_pass = (
        bool(cross_applicable)
        and all(
            item["overall_pass_if_material"]
            and item["passing_blocks_if_material"] >= REQUIRED_PASSING_BLOCKS
            for item in cross_applicable
        )
        if cross_required
        else True
    )

    density = validation_overall["density_physical_reconstruction"]
    positivity_pass = density is None or density["nonpositive_cell_count"] == 0
    passes = field_pass and spectral_pass and cross_pass and positivity_pass
    return {
        "thresholds": {
            "field_rmse_max": FIELD_RMSE_MAX,
            "field_variance_ratio": list(FIELD_VARIANCE_RATIO),
            "training_material_fraction_min": MATERIAL_FRACTION_MIN,
            "spectral_power_ratio": list(SPECTRAL_POWER_RATIO),
            "spectral_transfer_coherence_min": (
                SPECTRAL_TRANSFER_COHERENCE_MIN
            ),
            "cross_phase_error_max_degrees": CROSS_PHASE_ERROR_MAX_DEGREES,
            "cross_coherence_change_max": CROSS_COHERENCE_CHANGE_MAX,
            "required_passing_blocks": REQUIRED_PASSING_BLOCKS,
        },
        "materiality_source_split": materiality["source_split"],
        "field_reconstruction": {"passes": field_pass, "checks": fields},
        "spectral_transfer": {
            "passes": spectral_pass,
            "applicable_check_count": len(spectral_applicable),
            "checks": spectral,
        },
        "cross_field": {
            "required": cross_required,
            "passes": cross_pass,
            "applicable_check_count": len(cross_applicable),
            "checks": cross,
        },
        "density_positivity": {
            "passes": positivity_pass,
            "record": density,
        },
        "passes": passes,
        "status": "pass" if passes else "fail",
        "transport_gate": "pending_separate_authoritative_native81_evaluation",
    }
