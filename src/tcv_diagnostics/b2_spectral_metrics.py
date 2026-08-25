"""Member-wise spectral and cross-field evaluation for Paper 0 B2."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .b2_field_metrics import B2_FIELDS, B2_INTERVALS, gauge_fix_phi_channel
from .b2_forecast import sampler_seed
from .b2_probabilistic_metrics import (
    corrected_spread_skill_summary,
    deterministic_tie_uniform,
    ensemble_rank_histogram,
    monte_carlo_stability,
    order_statistic_interval_coverage,
)
from .metrics import fair_crps, ordinary_crps, toroidal_mode_numbers


B2_MODE_BANDS = (
    ("k1_3", 1, 3),
    ("k4_5", 4, 5),
    ("k6_7", 6, 7),
)
B2_CROSS_PAIRS = (("Ne", "phi"), ("Pe", "phi"), ("Pi", "phi"))
B2_SPECTRAL_PREFIXES = (4, 8, 16, 32)


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    first, second = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.full(first.shape, np.nan, dtype=np.float64)
    np.divide(first, second, out=result, where=second > 0.0)
    return result


def _safe_scalar_ratio(numerator: float, denominator: float) -> float:
    first = float(numerator)
    second = float(denominator)
    return first / second if second > 0.0 else math.nan


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _stability_or_undefined(first: float, second: float) -> dict[str, Any]:
    if math.isfinite(float(first)) and math.isfinite(float(second)):
        return monte_carlo_stability(first, second)
    return {
        "M16": _json_safe(float(first)),
        "M32": _json_safe(float(second)),
        "passes": False,
        "undefined_due_to_zero_skill_denominator": True,
    }


def _coherence(
    cross: np.ndarray,
    auto_a: np.ndarray,
    auto_b: np.ndarray,
) -> np.ndarray:
    return _safe_ratio(np.abs(cross) ** 2, auto_a * auto_b)


def _float_curve(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _complex_curve(values: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(values)
    return {
        "real": np.real(array).astype(np.float64).tolist(),
        "imag": np.imag(array).astype(np.float64).tolist(),
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(array) & np.isfinite(weight) & (weight >= 0.0)
    denominator = float(np.sum(weight[valid], dtype=np.float64))
    if denominator <= 0.0:
        return math.nan
    return float(np.sum(array[valid] * weight[valid], dtype=np.float64) / denominator)


def _one_sided_weights(size: int) -> np.ndarray:
    if int(size) < 2:
        raise ValueError("spectral dimension must contain at least two cells")
    weights = np.ones(int(size) // 2 + 1, dtype=np.float64)
    if int(size) % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


def _derived_tie_uniforms(
    *,
    model_seed: int,
    target_frames: Sequence[int],
    variable_code: int,
) -> np.ndarray:
    values = []
    for target in target_frames:
        values.append(
            float(
                deterministic_tie_uniform(
                    np.asarray(int(target), dtype=np.int64),
                    np.asarray(int(variable_code), dtype=np.int64),
                    np.asarray(0, dtype=np.int64),
                    seed=sampler_seed(model_seed, int(target)),
                )
            )
        )
    return np.asarray(values, dtype=np.float64)


def derived_ensemble_calibration(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    model_seed: int,
    target_frames: Sequence[int],
    variable_code: int,
) -> dict[str, Any]:
    """Calibrate one per-target scalar derived quantity without pooling time."""

    forecast_array = _finite_real("derived ensemble", forecast)
    truth_array = _finite_real("derived truth", truth)
    targets = tuple(int(item) for item in target_frames)
    if forecast_array.shape != (len(targets), 32) or truth_array.shape != (
        len(targets),
    ):
        raise ValueError("derived calibration arrays must have shapes [T,32] and [T]")
    if not targets:
        raise ValueError("derived calibration needs at least one target")
    tie_uniform = _derived_tie_uniforms(
        model_seed=model_seed,
        target_frames=targets,
        variable_code=variable_code,
    )
    rank = ensemble_rank_histogram(
        forecast_array,
        truth_array,
        member_axis=1,
        tie_uniform=tie_uniform,
    )
    intervals = {}
    for name, (lower, upper) in B2_INTERVALS.items():
        record = order_statistic_interval_coverage(
            forecast_array,
            truth_array,
            lower_order_one_indexed=lower,
            upper_order_one_indexed=upper,
            member_axis=1,
        )
        intervals[name] = {
            key: value
            for key, value in record.items()
            if key not in {"lower", "upper", "covered"}
        }
    prefixes = {}
    for members in B2_SPECTRAL_PREFIXES:
        values = forecast_array[:, :members]
        spread = corrected_spread_skill_summary(
            values,
            truth_array,
            member_axis=1,
        )
        prefixes[f"M{members}"] = {
            "ensemble_size": members,
            "fair_crps": float(
                np.mean(fair_crps(values, truth_array, member_axis=1))
            ),
            "ordinary_empirical_crps": float(
                np.mean(ordinary_crps(values, truth_array, member_axis=1))
            ),
            "corrected_spread_skill": spread,
        }
    stability = {
        metric: _stability_or_undefined(
            prefixes["M16"][metric],
            prefixes["M32"][metric],
        )
        for metric in ("fair_crps", "ordinary_empirical_crps")
    }
    stability["spread_skill_ratio"] = _stability_or_undefined(
        prefixes["M16"]["corrected_spread_skill"]["spread_skill_ratio"],
        prefixes["M32"]["corrected_spread_skill"]["spread_skill_ratio"],
    )
    return _json_safe(
        {
        "target_count": len(targets),
        "primary_M32": prefixes["M32"],
        "order_statistic_intervals": intervals,
        "rank_histogram": {
            key: (
                value.tolist() if isinstance(value, np.ndarray) else value
            )
            for key, value in rank.items()
        },
        "member_prefix_sensitivity": prefixes,
        "M16_vs_M32_stability": stability,
        "per_target": {
            "target_frame": list(targets),
            "truth": truth_array.tolist(),
            "ensemble_mean": np.mean(forecast_array, axis=1).tolist(),
            "fair_crps": fair_crps(
                forecast_array, truth_array, member_axis=1
            ).tolist(),
        },
        }
    )


class B2SpectralAccumulator:
    """Stream member-first field power, cross spectra, and directions."""

    def __init__(
        self,
        *,
        model_seed: int,
        target_frames: Sequence[int],
        eligible_xy_mask: np.ndarray,
        volume_shape: tuple[int, int, int] = (64, 32, 88),
        zperiod: int = 5,
        allow_sparse_targets: bool = False,
    ) -> None:
        if int(model_seed) not in (1701, 1702, 1703):
            raise ValueError("B2 spectral model seed differs")
        targets = tuple(int(item) for item in target_frames)
        if not targets or targets != tuple(sorted(set(targets))):
            raise ValueError("B2 spectral targets must be strictly increasing")
        contiguous = targets == tuple(range(targets[0], targets[-1] + 1))
        if not contiguous and not bool(allow_sparse_targets):
            raise ValueError("B2 spectral targets must be contiguous")
        self.model_seed = int(model_seed)
        self.target_frames = targets
        self.sparse_targets = not contiguous
        self.volume_shape = tuple(int(item) for item in volume_shape)
        if len(self.volume_shape) != 3 or min(self.volume_shape) < 2:
            raise ValueError("B2 spectral volume shape differs")
        self.n_x, self.n_y, self.n_z = self.volume_shape
        if self.n_z // 2 < 7:
            raise ValueError("B2 spectral toroidal grid cannot resolve k=7")
        if int(zperiod) != 5:
            raise ValueError("Paper 0 B2 requires zperiod=5")
        self.zperiod = int(zperiod)
        self.eligible = np.asarray(eligible_xy_mask, dtype=bool)
        if self.eligible.shape != (self.n_x, self.n_y) or not np.any(self.eligible):
            raise ValueError("B2 spectral eligible x-y mask differs")
        self.members = 32
        self.channels = len(B2_FIELDS)
        self.pairs = tuple(
            (B2_FIELDS.index(first), B2_FIELDS.index(second))
            for first, second in B2_CROSS_PAIRS
        )
        modes = self.n_z // 2 + 1
        self.truth_auto = np.zeros((self.channels, modes), dtype=np.float64)
        self.member_auto = np.zeros(
            (self.members, self.channels, modes), dtype=np.float64
        )
        self.mean_auto = np.zeros((self.channels, modes), dtype=np.float64)
        self.truth_mean_cross = np.zeros(
            (self.channels, modes), dtype=np.complex128
        )
        self.truth_member_cross = np.zeros(
            (self.members, self.channels, modes), dtype=np.complex128
        )
        self.truth_pair_cross = np.zeros(
            (len(self.pairs), modes), dtype=np.complex128
        )
        self.member_pair_cross = np.zeros(
            (self.members, len(self.pairs), modes), dtype=np.complex128
        )
        self.direction_truth = {
            "x": np.zeros((self.channels, self.n_x // 2 + 1)),
            "y": np.zeros((self.channels, self.n_y // 2 + 1)),
        }
        self.direction_member = {
            "x": np.zeros((self.members, self.channels, self.n_x // 2 + 1)),
            "y": np.zeros((self.members, self.channels, self.n_y // 2 + 1)),
        }
        self.direction_mean = {
            "x": np.zeros((self.channels, self.n_x // 2 + 1)),
            "y": np.zeros((self.channels, self.n_y // 2 + 1)),
        }
        self.band_truth = {
            (field, label): [] for field in B2_FIELDS for label, _, _ in B2_MODE_BANDS
        }
        self.band_members = {
            (field, label): [] for field in B2_FIELDS for label, _, _ in B2_MODE_BANDS
        }
        self.cross_truth = {
            (pair, label): []
            for pair in B2_CROSS_PAIRS
            for label, _, _ in B2_MODE_BANDS
        }
        self.cross_members = {
            (pair, label): []
            for pair in B2_CROSS_PAIRS
            for label, _, _ in B2_MODE_BANDS
        }
        self.cursor = 0
        self.xy_samples = 0
        self.direction_samples = {"x": 0, "y": 0}

    def _begin_target(self, target_frame: int) -> int:
        if self.cursor >= len(self.target_frames):
            raise ValueError("B2 spectral scorer received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(
                f"B2 spectral target {target_frame} differs from {expected}"
            )
        return expected

    def _consume(
        self,
        *,
        forecast: np.ndarray,
        truth: np.ndarray,
        forecast_z: np.ndarray,
        truth_z: np.ndarray,
        forecast_x: np.ndarray,
        truth_x: np.ndarray,
        forecast_y: np.ndarray,
        truth_y: np.ndarray,
    ) -> None:
        eligible_count = int(np.sum(self.eligible))
        forecast_selected = forecast_z[:, :, self.eligible, :]
        truth_selected = truth_z[:, self.eligible, :]
        self.xy_samples += eligible_count
        self.truth_auto += np.sum(
            np.abs(truth_selected) ** 2, axis=1, dtype=np.float64
        )
        self.member_auto += np.sum(
            np.abs(forecast_selected) ** 2, axis=2, dtype=np.float64
        )
        mean_selected = np.mean(forecast_selected, axis=0)
        self.mean_auto += np.sum(
            np.abs(mean_selected) ** 2, axis=1, dtype=np.float64
        )
        self.truth_mean_cross += np.sum(
            truth_selected * np.conjugate(mean_selected),
            axis=1,
            dtype=np.complex128,
        )
        self.truth_member_cross += np.sum(
            truth_selected[None] * np.conjugate(forecast_selected),
            axis=2,
            dtype=np.complex128,
        )
        for pair_index, (first, second) in enumerate(self.pairs):
            self.truth_pair_cross[pair_index] += np.sum(
                truth_selected[first] * np.conjugate(truth_selected[second]),
                axis=0,
                dtype=np.complex128,
            )
            self.member_pair_cross[:, pair_index] += np.sum(
                forecast_selected[:, first]
                * np.conjugate(forecast_selected[:, second]),
                axis=1,
                dtype=np.complex128,
            )

        z_scale = _one_sided_weights(self.n_z) / float(self.n_z * self.n_z)
        truth_power_target = (
            np.sum(np.abs(truth_selected) ** 2, axis=1) / eligible_count
        ) * z_scale[None]
        member_power_target = (
            np.sum(np.abs(forecast_selected) ** 2, axis=2) / eligible_count
        ) * z_scale[None, None]
        for channel, field in enumerate(B2_FIELDS):
            for label, low, high in B2_MODE_BANDS:
                band = slice(low, high + 1)
                self.band_truth[(field, label)].append(
                    float(np.sum(truth_power_target[channel, band]))
                )
                self.band_members[(field, label)].append(
                    np.sum(member_power_target[:, channel, band], axis=1)
                )
        for pair_index, pair in enumerate(B2_CROSS_PAIRS):
            first, second = self.pairs[pair_index]
            truth_cross_target = (
                np.sum(
                    truth_selected[first] * np.conjugate(truth_selected[second]),
                    axis=0,
                )
                / eligible_count
            ) * z_scale
            member_cross_target = (
                np.sum(
                    forecast_selected[:, first]
                    * np.conjugate(forecast_selected[:, second]),
                    axis=1,
                )
                / eligible_count
            ) * z_scale[None]
            for label, low, high in B2_MODE_BANDS:
                band = slice(low, high + 1)
                self.cross_truth[(pair, label)].append(
                    np.sum(truth_cross_target[band])
                )
                self.cross_members[(pair, label)].append(
                    np.sum(member_cross_target[:, band], axis=1)
                )

        directions = (
            ("x", forecast_x, truth_x, self.n_x, (3, 4), (2, 3)),
            ("y", forecast_y, truth_y, self.n_y, (2, 4), (1, 3)),
        )
        for (
            name,
            coefficients_forecast,
            coefficients_truth,
            size,
            reduce_forecast,
            reduce_truth,
        ) in directions:
            self.direction_truth[name] += np.sum(
                np.abs(coefficients_truth) ** 2,
                axis=reduce_truth,
                dtype=np.float64,
            )
            self.direction_member[name] += np.sum(
                np.abs(coefficients_forecast) ** 2,
                axis=reduce_forecast,
                dtype=np.float64,
            )
            mean_coefficients = np.mean(coefficients_forecast, axis=0)
            self.direction_mean[name] += np.sum(
                np.abs(mean_coefficients) ** 2,
                axis=tuple(axis - 1 for axis in reduce_forecast),
                dtype=np.float64,
            )
            other_samples = (
                self.n_y * self.n_z if name == "x" else self.n_x * self.n_z
            )
            self.direction_samples[name] += other_samples

    def update(
        self,
        *,
        target_frame: int,
        physical_forecast: np.ndarray,
        physical_truth: np.ndarray,
        mirrors: Sequence["B2SpectralAccumulator"] = (),
    ) -> None:
        expected = self._begin_target(target_frame)
        destinations = (self, *tuple(mirrors))
        for destination in destinations[1:]:
            if (
                destination.model_seed != self.model_seed
                or destination.volume_shape != self.volume_shape
                or not np.array_equal(destination.eligible, self.eligible)
            ):
                raise ValueError("B2 spectral mirror conventions differ")
            destination._begin_target(expected)
        forecast = _finite_real("B2 physical spectral ensemble", physical_forecast)
        truth = _finite_real("B2 physical spectral truth", physical_truth)
        if forecast.shape != (32, len(B2_FIELDS), *self.volume_shape):
            raise ValueError("B2 physical spectral ensemble shape differs")
        if truth.shape != (len(B2_FIELDS), *self.volume_shape):
            raise ValueError("B2 physical spectral truth shape differs")
        forecast = forecast.copy()
        truth = truth.copy()
        forecast[:, 3], truth[3] = gauge_fix_phi_channel(
            forecast[:, 3], truth[3]
        )
        forecast_z = np.fft.rfft(forecast, axis=4)
        truth_z = np.fft.rfft(truth, axis=3)
        forecast_x = np.fft.rfft(forecast, axis=2)
        truth_x = np.fft.rfft(truth, axis=1)
        forecast_y = np.fft.rfft(forecast, axis=3)
        truth_y = np.fft.rfft(truth, axis=2)
        for destination in destinations:
            destination._consume(
                forecast=forecast,
                truth=truth,
                forecast_z=forecast_z,
                truth_z=truth_z,
                forecast_x=forecast_x,
                truth_x=truth_x,
                forecast_y=forecast_y,
                truth_y=truth_y,
            )
            destination.cursor += 1

    def _field_records(
        self,
        truth_power: np.ndarray,
        expected_power: np.ndarray,
        mean_power: np.ndarray,
    ) -> dict[str, Any]:
        truth_mean_coherence = _coherence(
            self.truth_mean_cross,
            self.truth_auto,
            self.mean_auto,
        )
        truth_member_coherence = _coherence(
            self.truth_member_cross,
            self.truth_auto[None],
            self.member_auto,
        )
        records = {}
        for channel, field in enumerate(B2_FIELDS):
            curves = {
                "truth_power": _float_curve(truth_power[channel]),
                "member_expected_power": _float_curve(expected_power[channel]),
                "ensemble_mean_field_power": _float_curve(mean_power[channel]),
                "member_expected_power_ratio": _float_curve(
                    _safe_ratio(expected_power[channel], truth_power[channel])
                ),
                "ensemble_mean_realization_coherence_with_truth": _float_curve(
                    truth_mean_coherence[channel]
                ),
                "member_truth_realization_coherence": [
                    _float_curve(values)
                    for values in truth_member_coherence[:, channel]
                ],
            }
            bands = {}
            for band_index, (label, low, high) in enumerate(B2_MODE_BANDS):
                indices = np.arange(low, high + 1, dtype=np.int64)
                truth_weight = truth_power[channel, indices]
                member_band_coherence = [
                    _weighted_mean(
                        truth_member_coherence[member, channel, indices],
                        truth_weight,
                    )
                    for member in range(self.members)
                ]
                calibration = derived_ensemble_calibration(
                    np.asarray(self.band_members[(field, label)]),
                    np.asarray(self.band_truth[(field, label)]),
                    model_seed=self.model_seed,
                    target_frames=self.target_frames,
                    variable_code=100 + channel * 10 + band_index,
                )
                bands[label] = {
                    "stored_k": [low, high],
                    "full_torus_n": [low * self.zperiod, high * self.zperiod],
                    "truth_power": float(np.sum(truth_weight)),
                    "member_expected_power": float(
                        np.sum(expected_power[channel, indices])
                    ),
                    "member_expected_power_ratio": _safe_scalar_ratio(
                        float(np.sum(expected_power[channel, indices])),
                        float(np.sum(truth_weight)),
                    ),
                    "ensemble_mean_realization_coherence_with_truth": (
                        _weighted_mean(
                            truth_mean_coherence[channel, indices],
                            truth_weight,
                        )
                    ),
                    "member_truth_realization_coherence_distribution": {
                        "values": member_band_coherence,
                        "minimum": float(np.min(member_band_coherence)),
                        "median": float(np.median(member_band_coherence)),
                        "maximum": float(np.max(member_band_coherence)),
                    },
                    "per_target_band_power_calibration": calibration,
                }
            records[field] = {"curves": curves, "bands": bands}
        return records

    def _cross_records(self) -> dict[str, Any]:
        expected_auto = np.mean(self.member_auto, axis=0)
        expected_cross = np.mean(self.member_pair_cross, axis=0)
        records = {}
        for pair_index, pair in enumerate(B2_CROSS_PAIRS):
            first, second = self.pairs[pair_index]
            truth_coherence = _coherence(
                self.truth_pair_cross[pair_index],
                self.truth_auto[first],
                self.truth_auto[second],
            )
            forecast_coherence = _coherence(
                expected_cross[pair_index],
                expected_auto[first],
                expected_auto[second],
            )
            phase_error = np.angle(
                expected_cross[pair_index]
                * np.conjugate(self.truth_pair_cross[pair_index])
            )
            bands = {}
            for band_index, (label, low, high) in enumerate(B2_MODE_BANDS):
                indices = np.arange(low, high + 1, dtype=np.int64)
                truth_amplitude = np.abs(self.truth_pair_cross[pair_index, indices])
                expected_amplitude = np.abs(expected_cross[pair_index, indices])
                truth_projection = np.asarray(self.cross_truth[(pair, label)])
                member_projection = np.asarray(self.cross_members[(pair, label)])
                real_calibration = derived_ensemble_calibration(
                    np.real(member_projection),
                    np.real(truth_projection),
                    model_seed=self.model_seed,
                    target_frames=self.target_frames,
                    variable_code=200 + pair_index * 20 + band_index * 2,
                )
                imaginary_calibration = derived_ensemble_calibration(
                    np.imag(member_projection),
                    np.imag(truth_projection),
                    model_seed=self.model_seed,
                    target_frames=self.target_frames,
                    variable_code=201 + pair_index * 20 + band_index * 2,
                )
                truth_band_complex = np.sum(
                    self.truth_pair_cross[pair_index, indices]
                )
                expected_band_complex = np.sum(expected_cross[pair_index, indices])
                bands[label] = {
                    "stored_k": [low, high],
                    "full_torus_n": [low * self.zperiod, high * self.zperiod],
                    "truth_cross_amplitude": float(np.sum(truth_amplitude)),
                    "member_expected_cross_amplitude": float(
                        np.sum(expected_amplitude)
                    ),
                    "member_expected_cross_amplitude_ratio": _safe_scalar_ratio(
                        float(np.sum(expected_amplitude)),
                        float(np.sum(truth_amplitude)),
                    ),
                    "summed_complex_cross_phase_error_degrees": math.degrees(
                        float(
                            np.angle(
                                expected_band_complex
                                * np.conjugate(truth_band_complex)
                            )
                        )
                    ),
                    "truth_amplitude_weighted_absolute_phase_error_degrees": (
                        math.degrees(
                            _weighted_mean(
                                np.abs(phase_error[indices]), truth_amplitude
                            )
                        )
                    ),
                    "truth_amplitude_weighted_truth_coherence": _weighted_mean(
                        truth_coherence[indices], truth_amplitude
                    ),
                    "truth_amplitude_weighted_member_expected_coherence": (
                        _weighted_mean(forecast_coherence[indices], truth_amplitude)
                    ),
                    "truth_amplitude_weighted_absolute_coherence_change": (
                        _weighted_mean(
                            np.abs(
                                forecast_coherence[indices]
                                - truth_coherence[indices]
                            ),
                            truth_amplitude,
                        )
                    ),
                    "per_target_cross_projection_calibration": {
                        "real": real_calibration,
                        "imaginary": imaginary_calibration,
                    },
                }
            key = f"{pair[0]}-{pair[1]}"
            records[key] = {
                "curves": {
                    "truth_cross_spectrum": _complex_curve(
                        self.truth_pair_cross[pair_index]
                    ),
                    "member_expected_cross_spectrum": _complex_curve(
                        expected_cross[pair_index]
                    ),
                    "truth_coherence": _float_curve(truth_coherence),
                    "member_expected_coherence": _float_curve(forecast_coherence),
                    "signed_phase_error_degrees": _float_curve(
                        np.degrees(phase_error)
                    ),
                },
                "bands": bands,
            }
        return records

    def finalize(self) -> dict[str, Any]:
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B2 spectral scorer did not receive every target")
        weights = _one_sided_weights(self.n_z)
        scale = weights[None] / float(self.n_z * self.n_z * self.xy_samples)
        truth_power = self.truth_auto * scale
        expected_power = np.mean(self.member_auto, axis=0) * scale
        mean_power = self.mean_auto * scale
        directional = {}
        for name, size in (("x", self.n_x), ("y", self.n_y)):
            direction_scale = _one_sided_weights(size)[None] / float(
                size * size * self.direction_samples[name]
            )
            directional[name] = {
                "coordinate": (
                    "stored_grid_index_full_crop_descriptive_not_physical_wavenumber"
                ),
                "index": list(range(size // 2 + 1)),
                "fields": {
                    field: {
                        "truth_power": _float_curve(
                            self.direction_truth[name][channel] * direction_scale[0]
                        ),
                        "member_expected_power": _float_curve(
                            np.mean(
                                self.direction_member[name][:, channel], axis=0
                            )
                            * direction_scale[0]
                        ),
                        "ensemble_mean_field_power": _float_curve(
                            self.direction_mean[name][channel] * direction_scale[0]
                        ),
                    }
                    for channel, field in enumerate(B2_FIELDS)
                },
            }
        stored_k, full_torus_n = toroidal_mode_numbers(
            self.n_z, zperiod=self.zperiod
        )
        result = {
            "schema_version": 1,
            "scope": "B2_memberwise_spectral_and_cross_field_metrics_85604",
            "model_seed": self.model_seed,
            "target_frames": (
                list(self.target_frames)
                if self.sparse_targets
                else [self.target_frames[0], self.target_frames[-1] + 1]
            ),
            "target_count": len(self.target_frames),
            "zperiod": self.zperiod,
            "mode_mapping": "n=5k",
            "stored_k": stored_k.tolist(),
            "full_torus_n": full_torus_n.tolist(),
            "eligible_xy_cells": int(np.sum(self.eligible)),
            "memberwise_nonlinear_diagnostic_before_ensemble_reduction": True,
            "ensemble_mean_fields_used_as_probabilistic_spectrum": False,
            "toroidal_field_power": self._field_records(
                truth_power, expected_power, mean_power
            ),
            "toroidal_cross_field": self._cross_records(),
            "directional_index_spectra": directional,
            "potential_policy": "full_spatial_mean_removed_per_member_and_truth",
            "held_out_85606_read": False,
            "physics_derived_training_loss_used": False,
        }
        if self.sparse_targets:
            result["target_frames_are_explicit_indices"] = True
        return _json_safe(result)
