"""Gauge-aware field and marginal-calibration primitives for Paper 0 B2.

The functions here are data-independent and contain no checkpoint or truth
file access.  They consume one already-generated ensemble and implement the
finite-member conventions frozen before B2 full training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .b2_probabilistic_metrics import (
    deterministic_tie_uniform,
    ensemble_rank_histogram,
)
from .geometry import SingleNullRegionMasks


B2_FIELDS = ("Ne", "Pe", "Pi", "phi", "Vi")
B2_PHI_INDEX = B2_FIELDS.index("phi")
B2_PRIMARY_REGIONS = (
    "confined_edge",
    "private_flux",
    "scrape_off_layer",
)
B2_OVERLAPPING_REGIONS = (
    "separatrix_cell_band",
    "outboard_midplane",
    "x_point_topology_stencil",
    "inner_divertor_leg",
    "outer_divertor_leg",
)
B2_ALL_REGIONS = (
    "eligible_union",
    *B2_PRIMARY_REGIONS,
    *B2_OVERLAPPING_REGIONS,
)
B2_INTERVALS = {
    "I17": (8, 25),
    "I27": (3, 30),
    "I31": (1, 32),
}


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def b2_region_masks(
    masks: SingleNullRegionMasks,
    *,
    n_z: int = 88,
) -> dict[str, np.ndarray]:
    """Return frozen flattened three-dimensional B2 cell masks."""

    if int(n_z) <= 0:
        raise ValueError("B2 toroidal cell count must be positive")
    shape = np.asarray(masks.strict_wall_interior).shape
    if shape != (64, 32):
        raise ValueError("B2 geometry masks must use the 64-by-32 model crop")
    eligible = np.asarray(
        masks.strict_wall_interior & masks.operator_interior,
        dtype=bool,
    )
    primary = {
        name: np.asarray(getattr(masks, name), dtype=bool)
        for name in B2_PRIMARY_REGIONS
    }
    multiplicity = sum(mask.astype(np.int8) for mask in primary.values())
    if not np.array_equal(multiplicity == 1, eligible) or np.any(multiplicity > 1):
        raise ValueError("B2 primary regions do not exactly partition eligible cells")
    two_dimensional: dict[str, np.ndarray] = {
        "eligible_union": eligible,
        **primary,
        **{
            name: np.asarray(getattr(masks, name), dtype=bool)
            for name in B2_OVERLAPPING_REGIONS
        },
    }
    if tuple(two_dimensional) != B2_ALL_REGIONS:
        raise RuntimeError("B2 geometry report order differs")
    result = {}
    for name, mask in two_dimensional.items():
        if mask.shape != shape or np.any(mask & ~eligible):
            raise ValueError(f"B2 region {name} leaves the eligible union")
        expanded = np.broadcast_to(mask[..., None], (*shape, int(n_z)))
        flattened = np.asarray(expanded.reshape(-1), dtype=bool)
        if not np.any(flattened):
            raise ValueError(f"B2 region {name} is empty")
        result[name] = flattened
    return result


def gauge_fix_phi_channel(
    forecast: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract one full-volume mean from every member and the truth."""

    forecast_array = _finite_real("phi ensemble", forecast)
    truth_array = _finite_real("phi truth", truth)
    if forecast_array.ndim != 4 or truth_array.shape != forecast_array.shape[1:]:
        raise ValueError("phi arrays must have shapes [M,X,Y,Z] and [X,Y,Z]")
    forecast_means = np.mean(forecast_array, axis=(1, 2, 3), keepdims=True)
    truth_mean = np.mean(truth_array, dtype=np.float64)
    return forecast_array - forecast_means, truth_array - truth_mean


@dataclass(frozen=True)
class PointwiseEnsembleDiagnostics:
    """Pointwise arrays produced once and reducible through any region mask."""

    members: int
    ensemble_mean: np.ndarray
    error: np.ndarray
    unbiased_member_variance: np.ndarray
    fair_crps: np.ndarray
    ordinary_crps: np.ndarray
    interval_covered: Mapping[str, np.ndarray]
    interval_width: Mapping[str, np.ndarray]
    ranks: np.ndarray
    tied: np.ndarray


def pointwise_ensemble_diagnostics(
    forecast: np.ndarray,
    truth: np.ndarray,
    *,
    target_frame: int,
    channel_index: int,
    spatial_cell_index: np.ndarray,
    tie_seed: int,
) -> PointwiseEnsembleDiagnostics:
    """Compute all frozen M32 scalar diagnostics with one member sort."""

    forecast_array = _finite_real("point ensemble", forecast)
    truth_array = _finite_real("point truth", truth)
    if forecast_array.ndim != 2 or truth_array.shape != forecast_array.shape[1:]:
        raise ValueError("point arrays must have shapes [M,N] and [N]")
    members, cells = forecast_array.shape
    if members != 32:
        raise ValueError("primary B2 diagnostics require exactly 32 members")
    indices = np.asarray(spatial_cell_index)
    if not np.issubdtype(indices.dtype, np.integer) or indices.shape != (cells,):
        raise ValueError("spatial cell indices must be one integer per point")
    if np.any(indices < 0) or np.unique(indices).size != cells:
        raise ValueError("spatial cell indices must be unique and nonnegative")
    if int(channel_index) < 0:
        raise ValueError("B2 diagnostic variable index must be nonnegative")

    ensemble_mean = np.mean(forecast_array, axis=0)
    error = ensemble_mean - truth_array
    unbiased_variance = np.var(forecast_array, axis=0, ddof=1)
    observation_term = np.mean(
        np.abs(forecast_array - truth_array[None]), axis=0
    )
    sorted_forecast = np.sort(forecast_array, axis=0)
    coefficients = 2.0 * np.arange(members, dtype=np.float64) - members + 1.0
    weighted_order_sum = np.sum(sorted_forecast * coefficients[:, None], axis=0)
    ordinary = observation_term - weighted_order_sum / float(members * members)
    fair = observation_term - weighted_order_sum / float(
        members * (members - 1)
    )

    covered = {}
    width = {}
    for name, (lower, upper) in B2_INTERVALS.items():
        lower_values = sorted_forecast[lower - 1]
        upper_values = sorted_forecast[upper - 1]
        covered[name] = (truth_array >= lower_values) & (truth_array <= upper_values)
        width[name] = upper_values - lower_values

    expanded_truth = truth_array[None]
    tied = np.any(forecast_array == expanded_truth, axis=0)
    uniforms = deterministic_tie_uniform(
        np.full(cells, int(target_frame), dtype=np.int64),
        np.full(cells, int(channel_index), dtype=np.int64),
        indices,
        seed=int(tie_seed),
    )
    rank_record = ensemble_rank_histogram(
        forecast_array,
        truth_array,
        member_axis=0,
        tie_uniform=uniforms,
        return_ranks=True,
    )
    ranks = np.asarray(rank_record["ranks"], dtype=np.int64)
    if ranks.shape != (cells,):
        raise RuntimeError("B2 pointwise rank shape differs")
    return PointwiseEnsembleDiagnostics(
        members=members,
        ensemble_mean=ensemble_mean,
        error=error,
        unbiased_member_variance=unbiased_variance,
        fair_crps=fair,
        ordinary_crps=ordinary,
        interval_covered=covered,
        interval_width=width,
        ranks=ranks,
        tied=tied,
    )


class FieldRegionAccumulator:
    """Stream one field and geometry region using target-level diagnostics."""

    def __init__(self, *, members: int = 32) -> None:
        if int(members) != 32:
            raise ValueError("primary B2 field-region metrics require M=32")
        self.members = int(members)
        self.count = 0
        self.error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.truth_sum = 0.0
        self.mean_sum = 0.0
        self.truth_squared_sum = 0.0
        self.mean_squared_sum = 0.0
        self.truth_mean_product_sum = 0.0
        self.member_variance_sum = 0.0
        self.positive_spread_count = 0
        self.maximum_member_variance = 0.0
        self.fair_crps_sum = 0.0
        self.ordinary_crps_sum = 0.0
        self.covered_count = {name: 0 for name in B2_INTERVALS}
        self.interval_width_sum = {name: 0.0 for name in B2_INTERVALS}
        self.rank_counts = np.zeros(self.members + 1, dtype=np.int64)
        self.tied_count = 0

    def update(
        self,
        diagnostics: PointwiseEnsembleDiagnostics,
        truth: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if diagnostics.members != self.members:
            raise ValueError("B2 diagnostics member count differs")
        truth_array = _finite_real("region truth", truth)
        selected = np.asarray(mask, dtype=bool)
        if truth_array.ndim != 1 or selected.shape != truth_array.shape:
            raise ValueError("B2 region truth and mask must be matched vectors")
        if not np.any(selected):
            raise ValueError("B2 field region mask is empty")
        arrays = (
            diagnostics.ensemble_mean,
            diagnostics.error,
            diagnostics.unbiased_member_variance,
            diagnostics.fair_crps,
            diagnostics.ordinary_crps,
            diagnostics.ranks,
            diagnostics.tied,
        )
        if any(np.asarray(value).shape != truth_array.shape for value in arrays):
            raise ValueError("B2 pointwise diagnostic shapes differ")
        truth_values = truth_array[selected]
        mean_values = diagnostics.ensemble_mean[selected]
        error_values = diagnostics.error[selected]
        variance_values = diagnostics.unbiased_member_variance[selected]
        count = int(np.sum(selected))
        self.count += count
        self.error_sum += float(np.sum(error_values, dtype=np.float64))
        self.absolute_error_sum += float(
            np.sum(np.abs(error_values), dtype=np.float64)
        )
        self.squared_error_sum += float(
            np.sum(error_values * error_values, dtype=np.float64)
        )
        self.truth_sum += float(np.sum(truth_values, dtype=np.float64))
        self.mean_sum += float(np.sum(mean_values, dtype=np.float64))
        self.truth_squared_sum += float(
            np.sum(truth_values * truth_values, dtype=np.float64)
        )
        self.mean_squared_sum += float(
            np.sum(mean_values * mean_values, dtype=np.float64)
        )
        self.truth_mean_product_sum += float(
            np.sum(truth_values * mean_values, dtype=np.float64)
        )
        self.member_variance_sum += float(np.sum(variance_values, dtype=np.float64))
        self.positive_spread_count += int(np.count_nonzero(variance_values > 0.0))
        self.maximum_member_variance = max(
            self.maximum_member_variance,
            float(np.max(variance_values)),
        )
        self.fair_crps_sum += float(
            np.sum(diagnostics.fair_crps[selected], dtype=np.float64)
        )
        self.ordinary_crps_sum += float(
            np.sum(diagnostics.ordinary_crps[selected], dtype=np.float64)
        )
        for name in B2_INTERVALS:
            self.covered_count[name] += int(
                np.count_nonzero(diagnostics.interval_covered[name][selected])
            )
            self.interval_width_sum[name] += float(
                np.sum(diagnostics.interval_width[name][selected], dtype=np.float64)
            )
        ranks = np.asarray(diagnostics.ranks[selected], dtype=np.int64)
        self.rank_counts += np.bincount(
            ranks, minlength=self.members + 1
        ).astype(np.int64)
        self.tied_count += int(np.count_nonzero(diagnostics.tied[selected]))

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("cannot finalize an empty B2 field region")
        count = float(self.count)
        truth_mean = self.truth_sum / count
        forecast_mean = self.mean_sum / count
        truth_variance = max(
            self.truth_squared_sum / count - truth_mean * truth_mean,
            0.0,
        )
        forecast_variance = max(
            self.mean_squared_sum / count - forecast_mean * forecast_mean,
            0.0,
        )
        rmse = math.sqrt(self.squared_error_sum / count)
        corrected_spread = math.sqrt(
            ((self.members + 1) / self.members)
            * self.member_variance_sum
            / count
        )
        anomaly_denominator = math.sqrt(
            self.truth_squared_sum * self.mean_squared_sum
        )
        rank_values = np.arange(self.members + 1, dtype=np.float64)
        frequencies = self.rank_counts.astype(np.float64) / count
        rank_mean = float(np.sum(rank_values * frequencies))
        rank_variance = float(
            np.sum((rank_values - rank_mean) ** 2 * frequencies)
        )
        uniform_variance = float(self.members * (self.members + 2) / 12.0)
        intervals = {}
        for name, (lower, upper) in B2_INTERVALS.items():
            intervals[name] = {
                "lower_order_one_indexed": lower,
                "upper_order_one_indexed": upper,
                "nominal_coverage": float((upper - lower) / (self.members + 1)),
                "empirical_coverage": self.covered_count[name] / count,
                "mean_interval_width": self.interval_width_sum[name] / count,
            }
        return {
            "ensemble_size": self.members,
            "scalar_count": self.count,
            "voxel_count_used_as_independent_sample_size": False,
            "ensemble_mean": {
                "rmse": rmse,
                "mae": self.absolute_error_sum / count,
                "bias": self.error_sum / count,
                "truth_mean": truth_mean,
                "forecast_mean": forecast_mean,
                "truth_population_variance": truth_variance,
                "forecast_population_variance": forecast_variance,
                "population_variance_ratio": (
                    forecast_variance / truth_variance
                    if truth_variance > 0.0
                    else None
                ),
                "training_mean_anomaly_correlation": (
                    self.truth_mean_product_sum / anomaly_denominator
                    if anomaly_denominator > 0.0
                    else None
                ),
            },
            "fair_crps": self.fair_crps_sum / count,
            "ordinary_empirical_crps": self.ordinary_crps_sum / count,
            "corrected_spread_skill": {
                "member_variance_ddof": 1,
                "finite_member_variance_factor": (self.members + 1) / self.members,
                "mean_unbiased_member_variance": self.member_variance_sum / count,
                "corrected_rms_spread": corrected_spread,
                "rmse_of_ensemble_mean": rmse,
                "ratio": corrected_spread / rmse if rmse > 0.0 else None,
            },
            "spread_integrity": {
                "positive_variance_count": self.positive_spread_count,
                "positive_variance_fraction": self.positive_spread_count / count,
                "maximum_unbiased_member_variance": self.maximum_member_variance,
                "nonzero_spread": self.maximum_member_variance > 0.0,
            },
            "order_statistic_intervals": intervals,
            "rank_histogram": {
                "bins": self.members + 1,
                "counts": self.rank_counts.tolist(),
                "frequencies": frequencies.tolist(),
                "tied_truth_values": self.tied_count,
                "normalized_rank_mean": rank_mean / self.members,
                "rank_variance": rank_variance,
                "uniform_rank_variance": uniform_variance,
                "rank_variance_ratio": rank_variance / uniform_variance,
                "total_variation_from_uniform": float(
                    0.5
                    * np.sum(
                        np.abs(frequencies - 1.0 / (self.members + 1))
                    )
                ),
                "pixel_iid_p_value_reported": False,
            },
        }
