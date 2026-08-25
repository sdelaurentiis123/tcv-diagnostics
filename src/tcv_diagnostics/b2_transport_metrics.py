"""Member-wise transport and transport-calibration metrics for Paper 0 B2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import (
    B2_INTERVALS,
    FieldRegionAccumulator,
    PointwiseEnsembleDiagnostics,
    pointwise_ensemble_diagnostics,
)
from .b2_field_scoring import B2_MEMBER_PREFIXES, PrefixFieldAccumulator
from .b2_forecast import sampler_seed
from .b2_probabilistic_metrics import monte_carlo_stability
from .codec_transport import (
    TRANSPORT_QUANTITIES,
    CodecTransportGeometry,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .resampling import (
    finalize_paired_statistics,
    merge_paired_sufficient_statistics,
    paired_sufficient_statistics,
    periodic_resample_float32,
)


B2_TRANSPORT_REDUCTIONS = ("strict_face_contributions", "separatrix_wedge")


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
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


def memberwise_transport_outputs(
    *,
    physical_forecast_model88: np.ndarray,
    native_truth: Mapping[str, np.ndarray],
    geometry: CodecTransportGeometry,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Apply the authoritative nonlinear transport operator member by member."""

    forecast = _finite_real("physical B2 forecast", physical_forecast_model88)
    if forecast.shape != (32, 5, 64, 32, 88):
        raise ValueError("physical B2 forecast must have shape [32,5,64,32,88]")
    if set(native_truth) != {"Ne", "Pe", "Pi", "phi"}:
        raise ValueError("native transport truth fields differ")
    truth = {
        field: _finite_real(f"native truth {field}", values)
        for field, values in native_truth.items()
    }
    if any(values.shape != (1, 64, 32, 81) for values in truth.values()):
        raise ValueError("native transport truth must have shape [1,64,32,81]")
    native_forecast = periodic_resample_float32(
        forecast[:, :4],
        81,
        axis=-1,
    ).astype(np.float64)
    truth_state = direct_pressure_transport_state(
        truth["Ne"], truth["Pe"], truth["Pi"], truth["phi"]
    )
    truth_evaluated = evaluate_transport_state(truth_state, geometry)
    truth_outputs = {
        quantity: {
            "strict_face_contributions": np.asarray(
                truth_evaluated[quantity]["strict_face_contributions"][0],
                dtype=np.float64,
            ).reshape(-1),
            "separatrix_wedge": np.asarray(
                truth_evaluated[quantity]["separatrix_wedge"],
                dtype=np.float64,
            ),
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    member_lists = {
        quantity: {reduction: [] for reduction in B2_TRANSPORT_REDUCTIONS}
        for quantity in TRANSPORT_QUANTITIES
    }
    for member in range(32):
        state = direct_pressure_transport_state(
            native_forecast[member, 0][None],
            native_forecast[member, 1][None],
            native_forecast[member, 2][None],
            native_forecast[member, 3][None],
        )
        evaluated = evaluate_transport_state(state, geometry)
        for quantity in TRANSPORT_QUANTITIES:
            member_lists[quantity]["strict_face_contributions"].append(
                np.asarray(
                    evaluated[quantity]["strict_face_contributions"][0],
                    dtype=np.float64,
                ).reshape(-1)
            )
            member_lists[quantity]["separatrix_wedge"].append(
                float(evaluated[quantity]["separatrix_wedge"][0])
            )
    forecast_outputs = {
        quantity: {
            reduction: np.asarray(member_lists[quantity][reduction], dtype=np.float64)
            for reduction in B2_TRANSPORT_REDUCTIONS
        }
        for quantity in TRANSPORT_QUANTITIES
    }
    for quantity in TRANSPORT_QUANTITIES:
        if forecast_outputs[quantity]["separatrix_wedge"].shape != (32,):
            raise RuntimeError("B2 member separatrix transport shape differs")
        strict_shape = forecast_outputs[quantity]["strict_face_contributions"].shape
        if strict_shape[0] != 32 or strict_shape[1] <= 0:
            raise RuntimeError("B2 member strict-face transport shape differs")
        if (
            truth_outputs[quantity]["strict_face_contributions"].shape
            != strict_shape[1:]
        ):
            raise RuntimeError("B2 truth/member strict-face shapes differ")
    return forecast_outputs, truth_outputs


def _target_record(
    diagnostics: PointwiseEnsembleDiagnostics,
    truth: np.ndarray,
) -> dict[str, Any]:
    truth_array = _finite_real("transport target truth", truth)
    error = diagnostics.error
    denominator = float(np.sum(truth_array * truth_array, dtype=np.float64))
    relative_l2 = (
        math.sqrt(float(np.sum(error * error, dtype=np.float64)) / denominator)
        if denominator > 0.0
        else math.nan
    )
    return {
        "point_count": int(truth_array.size),
        "truth_rms": float(np.sqrt(np.mean(truth_array * truth_array))),
        "ensemble_mean_rms": float(
            np.sqrt(np.mean(diagnostics.ensemble_mean**2))
        ),
        "ensemble_mean_relative_l2": relative_l2,
        "fair_crps": float(np.mean(diagnostics.fair_crps)),
        "ordinary_empirical_crps": float(np.mean(diagnostics.ordinary_crps)),
        "mean_unbiased_member_variance": float(
            np.mean(diagnostics.unbiased_member_variance)
        ),
        "coverage": {
            name: float(np.mean(diagnostics.interval_covered[name]))
            for name in B2_INTERVALS
        },
    }


class B2TransportAccumulator:
    """Stream all four member-wise transport quantities and two reductions."""

    def __init__(
        self,
        *,
        model_seed: int,
        target_frames: Sequence[int],
        event_thresholds: Mapping[str, float],
        detailed: bool,
        allow_sparse_targets: bool = False,
    ) -> None:
        if int(model_seed) not in (1701, 1702, 1703):
            raise ValueError("B2 transport model seed differs")
        targets = tuple(int(item) for item in target_frames)
        contiguous = bool(targets) and targets == tuple(
            range(targets[0], targets[-1] + 1)
        )
        if (
            not targets
            or targets != tuple(sorted(set(targets)))
            or (not contiguous and not bool(allow_sparse_targets))
        ):
            raise ValueError("B2 transport targets must be contiguous")
        if tuple(event_thresholds) != TRANSPORT_QUANTITIES:
            raise ValueError("B2 transport event-threshold quantities differ")
        thresholds = {name: float(value) for name, value in event_thresholds.items()}
        if any(
            not math.isfinite(value) or value < 0.0
            for value in thresholds.values()
        ):
            raise ValueError("B2 transport event thresholds must be finite/nonnegative")
        self.model_seed = int(model_seed)
        self.target_frames = targets
        self.sparse_targets = not contiguous
        self.event_thresholds = thresholds
        self.detailed = bool(detailed)
        self.probabilistic = {
            quantity: {
                reduction: FieldRegionAccumulator()
                for reduction in B2_TRANSPORT_REDUCTIONS
            }
            for quantity in TRANSPORT_QUANTITIES
        }
        self.paired = {
            quantity: {reduction: [] for reduction in B2_TRANSPORT_REDUCTIONS}
            for quantity in TRANSPORT_QUANTITIES
        }
        self.prefixes = (
            {
                quantity: {
                    reduction: {
                        members: PrefixFieldAccumulator(members)
                        for members in B2_MEMBER_PREFIXES
                    }
                    for reduction in B2_TRANSPORT_REDUCTIONS
                }
                for quantity in TRANSPORT_QUANTITIES
            }
            if self.detailed
            else None
        )
        self.truth_distribution = (
            {
                quantity: {reduction: [] for reduction in B2_TRANSPORT_REDUCTIONS}
                for quantity in TRANSPORT_QUANTITIES
            }
            if self.detailed
            else None
        )
        self.member_distribution = (
            {
                quantity: {reduction: [] for reduction in B2_TRANSPORT_REDUCTIONS}
                for quantity in TRANSPORT_QUANTITIES
            }
            if self.detailed
            else None
        )
        self.per_target: list[dict[str, Any]] = []
        self.separatrix_truth = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
        self.separatrix_members = {quantity: [] for quantity in TRANSPORT_QUANTITIES}
        self.cursor = 0

    def _begin_target(self, target_frame: int) -> int:
        if self.cursor >= len(self.target_frames):
            raise ValueError("B2 transport scorer received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(
                f"B2 transport target {target_frame} differs from {expected}"
            )
        return expected

    def update(
        self,
        *,
        target_frame: int,
        forecast_outputs: Mapping[str, Mapping[str, np.ndarray]],
        truth_outputs: Mapping[str, Mapping[str, np.ndarray]],
        mirrors: Sequence["B2TransportAccumulator"] = (),
    ) -> None:
        expected = self._begin_target(target_frame)
        destinations = (self, *tuple(mirrors))
        for destination in destinations[1:]:
            if (
                destination.model_seed != self.model_seed
                or destination.event_thresholds != self.event_thresholds
            ):
                raise ValueError("B2 transport mirror conventions differ")
            destination._begin_target(expected)
        if tuple(forecast_outputs) != TRANSPORT_QUANTITIES:
            raise ValueError("B2 forecast transport quantities differ")
        if tuple(truth_outputs) != TRANSPORT_QUANTITIES:
            raise ValueError("B2 truth transport quantities differ")
        target_records = {
            id(destination): {"target_frame": expected, "quantities": {}}
            for destination in destinations
            if destination.detailed
        }
        tie_seed = sampler_seed(self.model_seed, expected)
        for quantity_index, quantity in enumerate(TRANSPORT_QUANTITIES):
            if tuple(forecast_outputs[quantity]) != B2_TRANSPORT_REDUCTIONS:
                raise ValueError(f"B2 forecast {quantity} reductions differ")
            if tuple(truth_outputs[quantity]) != B2_TRANSPORT_REDUCTIONS:
                raise ValueError(f"B2 truth {quantity} reductions differ")
            for reduction_index, reduction in enumerate(B2_TRANSPORT_REDUCTIONS):
                forecast = _finite_real(
                    f"B2 forecast {quantity}.{reduction}",
                    forecast_outputs[quantity][reduction],
                )
                truth = _finite_real(
                    f"B2 truth {quantity}.{reduction}",
                    truth_outputs[quantity][reduction],
                )
                if reduction == "separatrix_wedge":
                    if forecast.shape != (32,) or truth.shape not in {(1,), ()}:
                        raise ValueError("B2 separatrix transport shape differs")
                    forecast = forecast.reshape(32, 1)
                    truth = truth.reshape(1)
                elif (
                    forecast.ndim != 2
                    or forecast.shape[0] != 32
                    or truth.shape != forecast.shape[1:]
                ):
                    raise ValueError("B2 strict-face transport shape differs")
                diagnostics = pointwise_ensemble_diagnostics(
                    forecast,
                    truth,
                    target_frame=expected,
                    channel_index=quantity_index * 2 + reduction_index,
                    spatial_cell_index=np.arange(truth.size, dtype=np.int64),
                    tie_seed=tie_seed,
                )
                mask = np.ones(truth.size, dtype=bool)
                paired = paired_sufficient_statistics(
                    truth,
                    diagnostics.ensemble_mean,
                )
                for destination in destinations:
                    destination.probabilistic[quantity][reduction].update(
                        diagnostics,
                        truth,
                        mask,
                    )
                    destination.paired[quantity][reduction].append(paired)
                    if destination.detailed:
                        if destination.prefixes is None:
                            raise AssertionError("detailed B2 prefixes are missing")
                        for members in B2_MEMBER_PREFIXES:
                            accumulator = destination.prefixes[quantity][reduction][
                                members
                            ]
                            if members == 32:
                                accumulator.update_primary(diagnostics, mask)
                            else:
                                accumulator.update_raw(
                                    forecast[:members], truth, mask
                                )
                        if (
                            destination.truth_distribution is None
                            or destination.member_distribution is None
                        ):
                            raise AssertionError(
                                "detailed B2 distributions are missing"
                            )
                        destination.truth_distribution[quantity][reduction].append(
                            np.asarray(truth, dtype=np.float32)
                        )
                        destination.member_distribution[quantity][reduction].append(
                            np.asarray(forecast, dtype=np.float32).reshape(-1)
                        )
                        target_records[id(destination)]["quantities"].setdefault(
                            quantity, {}
                        )[reduction] = _target_record(diagnostics, truth)
                if reduction == "separatrix_wedge":
                    for destination in destinations:
                        destination.separatrix_truth[quantity].append(float(truth[0]))
                        destination.separatrix_members[quantity].append(
                            np.asarray(forecast[:, 0], dtype=np.float64)
                        )
        for destination in destinations:
            if destination.detailed:
                destination.per_target.append(target_records[id(destination)])
            destination.cursor += 1

    def _distribution_record(self, quantity: str, reduction: str) -> dict[str, Any]:
        if self.truth_distribution is None or self.member_distribution is None:
            raise RuntimeError("B2 detailed distributions were not retained")
        truth = np.concatenate(self.truth_distribution[quantity][reduction]).astype(
            np.float64
        )
        members = np.concatenate(
            self.member_distribution[quantity][reduction]
        ).astype(np.float64)
        probabilities = (0.05, 0.50, 0.95)
        return {
            "pooling": (
                "truth_over_targets_and_reduction_points;members_over_targets_"
                "members_and_reduction_points"
            ),
            "truth_count": int(truth.size),
            "member_count": int(members.size),
            "truth": {
                "mean": float(np.mean(truth)),
                "standard_deviation": float(np.std(truth, ddof=0)),
                "quantiles": {
                    f"p{int(probability * 100):02d}": float(
                        np.quantile(truth, probability, method="linear")
                    )
                    for probability in probabilities
                },
            },
            "members": {
                "mean": float(np.mean(members)),
                "standard_deviation": float(np.std(members, ddof=0)),
                "quantiles": {
                    f"p{int(probability * 100):02d}": float(
                        np.quantile(members, probability, method="linear")
                    )
                    for probability in probabilities
                },
            },
        }

    def _event_record(self, quantity: str) -> dict[str, Any]:
        truth = np.asarray(self.separatrix_truth[quantity], dtype=np.float64)
        members = np.asarray(self.separatrix_members[quantity], dtype=np.float64)
        expected = np.mean(members, axis=1)
        threshold = self.event_thresholds[quantity]
        selected = np.abs(truth) >= threshold
        if not np.any(selected):
            return {
                "training_truth_absolute_value_p90_threshold": threshold,
                "validation_event_count": 0,
                "defined": False,
                "magnitude_relative_error": None,
                "truth_magnitude_weighted_sign_disagreement": None,
            }
        truth_values = truth[selected]
        expected_values = expected[selected]
        denominator = float(np.sum(np.abs(truth_values), dtype=np.float64))
        if denominator <= 0.0:
            return {
                "training_truth_absolute_value_p90_threshold": threshold,
                "validation_event_count": int(np.sum(selected)),
                "defined": False,
                "undefined_reason": "selected_truth_has_zero_total_magnitude",
                "magnitude_relative_error": None,
                "truth_magnitude_weighted_sign_disagreement": None,
            }
        sign_disagreement = np.signbit(truth_values) != np.signbit(expected_values)
        return {
            "training_truth_absolute_value_p90_threshold": threshold,
            "validation_event_count": int(np.sum(selected)),
            "defined": True,
            "magnitude_relative_error": float(
                np.sum(
                    np.abs(np.abs(expected_values) - np.abs(truth_values)),
                    dtype=np.float64,
                )
                / denominator
            ),
            "truth_magnitude_weighted_sign_disagreement": float(
                np.sum(np.abs(truth_values)[sign_disagreement], dtype=np.float64)
                / denominator
            ),
        }

    def finalize(self) -> dict[str, Any]:
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B2 transport scorer did not receive every target")
        quantities = {}
        for quantity in TRANSPORT_QUANTITIES:
            reductions = {}
            for reduction in B2_TRANSPORT_REDUCTIONS:
                sufficient = merge_paired_sufficient_statistics(
                    self.paired[quantity][reduction]
                )
                reductions[reduction] = {
                    "ensemble_expected_paired_metrics": finalize_paired_statistics(
                        sufficient
                    ),
                    "ensemble_probabilistic_metrics": self.probabilistic[quantity][
                        reduction
                    ].finalize(),
                }
                if self.detailed:
                    if self.prefixes is None:
                        raise AssertionError("detailed B2 prefixes are missing")
                    prefix_records = {
                        f"M{members}": self.prefixes[quantity][reduction][
                            members
                        ].finalize()
                        for members in B2_MEMBER_PREFIXES
                    }
                    reductions[reduction].update(
                        {
                            "member_prefix_sensitivity": prefix_records,
                            "M16_vs_M32_stability": {
                                metric: monte_carlo_stability(
                                    prefix_records["M16"][metric],
                                    prefix_records["M32"][metric],
                                )
                                for metric in (
                                    "fair_crps",
                                    "ordinary_empirical_crps",
                                    "corrected_spread_skill_ratio",
                                )
                            },
                            "pooled_distribution": self._distribution_record(
                                quantity, reduction
                            ),
                        }
                    )
            sep_truth = np.asarray(
                self.separatrix_truth[quantity], dtype=np.float64
            )
            sep_members = np.asarray(
                self.separatrix_members[quantity], dtype=np.float64
            )
            quantities[quantity] = {
                "reductions": reductions,
                "separatrix_time_series": {
                    "target_frame": list(self.target_frames),
                    "truth": sep_truth.tolist(),
                    "ensemble_expected": np.mean(sep_members, axis=1).tolist(),
                    "member_standard_deviation_ddof1": np.std(
                        sep_members, axis=1, ddof=1
                    ).tolist(),
                },
                "upper_decile_event_conditioned": self._event_record(quantity),
            }
        return _json_safe(
            {
                "schema_version": 1,
                "scope": "B2_memberwise_authoritative_transport_85604",
                "model_seed": self.model_seed,
                "target_frames": (
                    list(self.target_frames)
                    if self.sparse_targets
                    else [self.target_frames[0], self.target_frames[-1] + 1]
                ),
                "target_count": len(self.target_frames),
                "detailed": self.detailed,
                "quantities": quantities,
                "per_target": self.per_target if self.detailed else None,
                "nonlinear_operator_applied_per_member_before_reduction": True,
                "transport_of_ensemble_mean_fields_used": False,
                "model88_to_native81_resampling": (
                    "frozen_unwindowed_periodic_scipy_signal_resample_float32"
                ),
                "complete_experimental_heat_flux_claimed": False,
                "held_out_85606_read": False,
                "physics_derived_training_loss_used": False,
                **(
                    {"target_frames_are_explicit_indices": True}
                    if self.sparse_targets
                    else {}
                ),
            }
        )
