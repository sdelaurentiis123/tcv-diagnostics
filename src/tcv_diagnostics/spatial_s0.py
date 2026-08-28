"""Leakage-safe primitives for the frozen S0 spatial reconstruction study.

S0 is a simultaneous reconstruction experiment on development simulation
85604.  It is deliberately independent of temporal forecasting and contains
no learned physics loss.  The observation footprints and chronological
intervals are frozen in
``paper0/protocol/PHYSICS_FIRST_SPATIAL_S0_PROTOCOL_2026-08-28.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg


S0_FIELDS = ("Ne", "Pe", "Pi", "phi")
TRAIN_INTERVAL = (0, 432)
INTERNAL_FIT_INTERVAL = (0, 320)
INTERNAL_GUARD_INTERVAL = (320, 336)
INTERNAL_TUNE_INTERVAL = (336, 432)
VALIDATION_INTERVAL = (496, 624)
VOLUME_SHAPE = (64, 32, 88)
FOOTPRINT_HALF_WIDTHS = (1, 1, 2)
FOOTPRINT_MINIMUM_FRACTION = 0.60
RIDGE_LAMBDAS = tuple(float(10.0**power) for power in range(-5, 2))
DISTANCE_BIN_EDGES_M = (0.0, 0.05, 0.10, 0.20, 0.35, 0.60, math.inf)

FOOTPRINT_CENTERS: Mapping[str, tuple[tuple[int, int, int], ...]] = {
    "A": tuple(
        (x, y, 0)
        for y in (17, 19)
        for x in (13, 16, 19)
    ),
    "B": tuple(
        (x, y, 22)
        for y in (23, 25)
        for x in (13, 16, 19)
    ),
    "C": tuple(
        (x, y, 44)
        for y in (8, 10)
        for x in (13, 16, 19)
    ),
}


def _finite(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a real numeric array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class DiagnosticFootprint:
    """One fixed boxcar density observation after geometry exclusion."""

    family: str
    channel: int
    center: tuple[int, int, int]
    flat_indices: np.ndarray
    retained_cells: int
    nominal_cells: int = 45

    @property
    def retained_fraction(self) -> float:
        return self.retained_cells / self.nominal_cells


@dataclass(frozen=True)
class OmittedFootprint:
    family: str
    channel: int
    center: tuple[int, int, int]
    retained_cells: int
    nominal_cells: int = 45

    @property
    def retained_fraction(self) -> float:
        return self.retained_cells / self.nominal_cells


def build_fixed_footprints(
    strict_operator_mask: np.ndarray,
    *,
    shape: tuple[int, int, int] = VOLUME_SHAPE,
) -> tuple[tuple[DiagnosticFootprint, ...], tuple[OmittedFootprint, ...]]:
    """Build preregistered footprints without moving failed channels.

    Only the toroidal coordinate wraps.  Nonperiodic x/y indices outside the
    supplied volume are discarded, as are x/y cells outside the strict
    operator/wall mask.  A channel with fewer than 27 of 45 cells is omitted.
    """

    n_x, n_y, n_z = map(int, shape)
    if (n_x, n_y, n_z) != VOLUME_SHAPE:
        raise ValueError(f"S0 requires frozen shape {VOLUME_SHAPE}, got {shape}")
    valid = np.asarray(strict_operator_mask, dtype=bool)
    if valid.shape != (n_x, n_y):
        raise ValueError("strict_operator_mask must have shape [64,32]")
    kept: list[DiagnosticFootprint] = []
    omitted: list[OmittedFootprint] = []
    minimum = math.ceil(45 * FOOTPRINT_MINIMUM_FRACTION)
    for family in ("A", "B", "C"):
        for channel, center in enumerate(FOOTPRINT_CENTERS[family]):
            x0, y0, z0 = center
            indices: list[int] = []
            for x in range(x0 - 1, x0 + 2):
                for y in range(y0 - 1, y0 + 2):
                    if not (0 <= x < n_x and 0 <= y < n_y and valid[x, y]):
                        continue
                    for z_offset in range(-2, 3):
                        z = (z0 + z_offset) % n_z
                        indices.append(np.ravel_multi_index((x, y, z), shape))
            retained = len(indices)
            if retained < minimum:
                omitted.append(
                    OmittedFootprint(family, channel, center, retained)
                )
            else:
                kept.append(
                    DiagnosticFootprint(
                        family=family,
                        channel=channel,
                        center=center,
                        flat_indices=np.asarray(indices, dtype=np.int64),
                        retained_cells=retained,
                    )
                )
    return tuple(kept), tuple(omitted)


def group_footprints(
    footprints: Iterable[DiagnosticFootprint],
) -> dict[str, tuple[DiagnosticFootprint, ...]]:
    result = {
        family: tuple(
            sorted(
                (item for item in footprints if item.family == family),
                key=lambda item: item.channel,
            )
        )
        for family in ("A", "B", "C")
    }
    if not result["A"] or not result["B"] or not result["C"]:
        raise ValueError("each S0 diagnostic family must retain at least one channel")
    return result


def observe_density(
    density_anomaly: np.ndarray,
    footprints: Sequence[DiagnosticFootprint],
) -> np.ndarray:
    """Apply fixed density boxcars to arrays ending in ``[x,y,z]``."""

    values = _finite("density_anomaly", density_anomaly)
    if values.shape[-3:] != VOLUME_SHAPE:
        raise ValueError("density_anomaly must end in [64,32,88]")
    flattened = values.reshape(*values.shape[:-3], -1)
    channels = [
        np.mean(flattened[..., footprint.flat_indices], axis=-1, dtype=np.float64)
        for footprint in footprints
    ]
    return np.stack(channels, axis=-1)


@dataclass(frozen=True)
class DualRidgeKernel:
    """Centered dual ridge system reusable across streamed target slabs."""

    fit_input_mean: np.ndarray
    fit_inputs_centered: np.ndarray
    cholesky_factor: np.ndarray
    lower: bool
    regularization: float

    @classmethod
    def fit(cls, inputs: np.ndarray, regularization: float) -> "DualRidgeKernel":
        x = np.asarray(_finite("ridge inputs", inputs), dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("ridge inputs must have shape [sample,feature]")
        value = float(regularization)
        if value not in RIDGE_LAMBDAS:
            raise ValueError("regularization is outside the frozen S0 grid")
        mean = np.mean(x, axis=0, dtype=np.float64)
        centered = x - mean
        gram = centered @ centered.T
        gram.flat[:: gram.shape[0] + 1] += value
        factor, lower = scipy.linalg.cho_factor(
            gram,
            lower=True,
            overwrite_a=False,
            check_finite=True,
        )
        return cls(mean, centered, factor, lower, value)

    def cross_kernel(self, query_inputs: np.ndarray) -> np.ndarray:
        query = np.asarray(_finite("ridge query inputs", query_inputs), dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.fit_input_mean.size:
            raise ValueError("ridge query feature dimension differs")
        return (query - self.fit_input_mean) @ self.fit_inputs_centered.T

    def fit_target_coefficients(
        self,
        fit_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(_finite("ridge fit targets", fit_targets), dtype=np.float64)
        if targets.ndim != 2 or targets.shape[0] != self.fit_inputs_centered.shape[0]:
            raise ValueError("ridge targets must have shape [fit_sample,output]")
        target_mean = np.mean(targets, axis=0, dtype=np.float64)
        coefficients = scipy.linalg.cho_solve(
            (self.cholesky_factor, self.lower),
            targets - target_mean,
            overwrite_b=False,
            check_finite=True,
        )
        return target_mean, coefficients

    def predict(
        self,
        query_inputs: np.ndarray,
        fit_targets: np.ndarray,
    ) -> np.ndarray:
        target_mean, coefficients = self.fit_target_coefficients(fit_targets)
        return target_mean + self.cross_kernel(query_inputs) @ coefficients

    def predict_equivalent_dual(
        self,
        query_inputs: np.ndarray,
        fit_targets: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the dual-ridge predictor through its low-rank identity.

        With far fewer observation channels than training frames, directly
        multiplying the sample-space dual coefficients wastes substantial
        work.  This evaluates exactly the same centered ridge predictor using
        ``(X.T X + lambda I)^-1 X.T Y``.  Output slabs remain streamed; no
        complete field target matrix is required.
        """

        targets = np.asarray(_finite("ridge fit targets", fit_targets), dtype=np.float64)
        if targets.ndim != 2 or targets.shape[0] != self.fit_inputs_centered.shape[0]:
            raise ValueError("ridge targets must have shape [fit_sample,output]")
        query = np.asarray(_finite("ridge query inputs", query_inputs), dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != self.fit_input_mean.size:
            raise ValueError("ridge query feature dimension differs")
        target_mean = np.mean(targets, axis=0, dtype=np.float64)
        design = self.fit_inputs_centered
        feature_gram = design.T @ design
        feature_gram.flat[:: feature_gram.shape[0] + 1] += self.regularization
        rhs = design.T @ (targets - target_mean)
        weights = scipy.linalg.solve(
            feature_gram,
            rhs,
            assume_a="pos",
            check_finite=True,
        )
        return target_mean + (query - self.fit_input_mean) @ weights


def basic_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Return finite standardized scalar metrics for matched arrays."""

    target = np.asarray(_finite("metric truth", truth), dtype=np.float64)
    estimate = np.asarray(_finite("metric prediction", prediction), dtype=np.float64)
    if target.shape != estimate.shape or target.size < 2:
        raise ValueError("metric arrays must have equal shape and at least two values")
    error = estimate - target
    rmse = float(np.sqrt(np.mean(error * error, dtype=np.float64)))
    target_scale = float(np.std(target, ddof=0))
    if target_scale <= 0.0:
        raise ValueError("metric truth has zero population standard deviation")
    truth_centered = target.ravel() - float(np.mean(target))
    prediction_centered = estimate.ravel() - float(np.mean(estimate))
    denominator = float(
        np.sqrt(
            np.sum(truth_centered * truth_centered)
            * np.sum(prediction_centered * prediction_centered)
        )
    )
    correlation = (
        float(np.sum(truth_centered * prediction_centered) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return {
        "count": int(target.size),
        "rmse": rmse,
        "nrmse": rmse / target_scale,
        "bias": float(np.mean(error, dtype=np.float64)),
        "pearson_correlation": correlation,
        "truth_population_standard_deviation": target_scale,
    }


def choose_regularization(
    records: Sequence[Mapping[str, float]],
) -> tuple[float, tuple[dict[str, float], ...]]:
    """Choose smallest lambda minimizing the preregistered mean score."""

    normalized: list[dict[str, float]] = []
    seen: set[float] = set()
    for record in records:
        value = float(record["regularization"])
        if value not in RIDGE_LAMBDAS or value in seen:
            raise ValueError("ridge selection grid is incomplete or duplicated")
        full = float(record["equal_field_full_state_rmse"])
        heldout = float(record["heldout_c_rmse"])
        if not all(math.isfinite(item) and item >= 0.0 for item in (full, heldout)):
            raise ValueError("ridge selection metrics must be finite/nonnegative")
        normalized.append(
            {
                "regularization": value,
                "equal_field_full_state_rmse": full,
                "heldout_c_rmse": heldout,
                "selection_score": 0.5 * (full + heldout),
            }
        )
        seen.add(value)
    if seen != set(RIDGE_LAMBDAS):
        raise ValueError("ridge selection must contain the complete frozen grid")
    ordered = tuple(sorted(normalized, key=lambda item: item["regularization"]))
    winner = min(ordered, key=lambda item: (item["selection_score"], item["regularization"]))
    return float(winner["regularization"]), ordered


def select_median_hero_frame(
    frames: Sequence[int],
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select the earliest frame at the lower median per-frame C NRMSE rank."""

    indices = np.asarray(tuple(int(frame) for frame in frames), dtype=np.int64)
    target = np.asarray(_finite("hero truth", truth), dtype=np.float64)
    estimate = np.asarray(_finite("hero prediction", prediction), dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 2 or target.shape[0] != indices.size:
        raise ValueError("hero arrays must have shape [frame,heldout_channel]")
    scale = float(np.std(target, ddof=0))
    if scale <= 0.0:
        raise ValueError("held-out C truth has zero scale")
    per_frame = np.sqrt(np.mean((estimate - target) ** 2, axis=1)) / scale
    order = np.lexsort((indices, per_frame))
    selected_position = int(order[(indices.size - 1) // 2])
    return int(indices[selected_position]), per_frame


def toroidal_mode_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    eligible_xy: np.ndarray,
    *,
    zperiod: int = 5,
) -> tuple[dict[str, float | int], ...]:
    """Score mode error and retained power after removing toroidal means."""

    target = np.asarray(_finite("mode truth", truth), dtype=np.float64)
    estimate = np.asarray(_finite("mode prediction", prediction), dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 4:
        raise ValueError("mode arrays must have shape [sample,x,y,z]")
    mask = np.asarray(eligible_xy, dtype=bool)
    if mask.shape != target.shape[1:3]:
        raise ValueError("eligible_xy shape differs")
    target = target[:, mask, :]
    estimate = estimate[:, mask, :]
    target = target - np.mean(target, axis=-1, keepdims=True)
    estimate = estimate - np.mean(estimate, axis=-1, keepdims=True)
    target_fft = np.fft.rfft(target, axis=-1, norm="ortho")
    estimate_fft = np.fft.rfft(estimate, axis=-1, norm="ortho")
    records: list[dict[str, float | int]] = []
    for k in range(1, target_fft.shape[-1]):
        truth_power = float(np.mean(np.abs(target_fft[..., k]) ** 2))
        prediction_power = float(np.mean(np.abs(estimate_fft[..., k]) ** 2))
        coefficient_mse = float(
            np.mean(np.abs(estimate_fft[..., k] - target_fft[..., k]) ** 2)
        )
        records.append(
            {
                "stored_k": k,
                "physical_n": int(zperiod * k),
                "truth_power": truth_power,
                "prediction_power": prediction_power,
                "retained_power_ratio": (
                    prediction_power / truth_power if truth_power > 0.0 else 0.0
                ),
                "coefficient_mse": coefficient_mse,
            }
        )
    return tuple(records)


def minimum_cylindrical_distance_to_observations(
    major_radius_m: np.ndarray,
    vertical_position_m: np.ndarray,
    observed: Sequence[DiagnosticFootprint],
    *,
    n_z: int = 88,
    zperiod: int = 5,
) -> np.ndarray:
    """Return minimum 3-D cylindrical distance for every model cell."""

    radius = np.asarray(_finite("major radius", major_radius_m), dtype=np.float64)
    vertical = np.asarray(_finite("vertical position", vertical_position_m), dtype=np.float64)
    if radius.shape != VOLUME_SHAPE[:2] or vertical.shape != radius.shape:
        raise ValueError("R and Z must have shape [64,32]")
    if n_z != VOLUME_SHAPE[2] or zperiod != 5:
        raise ValueError("S0 distance uses the frozen 88-cell one-fifth wedge")
    observed_flat = sorted(
        {
            int(index)
            for footprint in observed
            for index in footprint.flat_indices
        }
    )
    if not observed_flat:
        raise ValueError("at least one observed footprint cell is required")
    ox, oy, oz = np.unravel_index(observed_flat, VOLUME_SHAPE)
    observed_r = radius[ox, oy]
    observed_z = vertical[ox, oy]
    wedge = 2.0 * np.pi / zperiod
    observed_phi = wedge * oz / n_z
    result = np.full(VOLUME_SHAPE, np.inf, dtype=np.float64)
    for x in range(VOLUME_SHAPE[0]):
        for y in range(VOLUME_SHAPE[1]):
            phi = wedge * np.arange(n_z, dtype=np.float64) / n_z
            delta = np.abs(phi[:, None] - observed_phi[None, :])
            delta = np.minimum(delta, wedge - delta)
            squared = (
                radius[x, y] ** 2
                + observed_r[None, :] ** 2
                - 2.0 * radius[x, y] * observed_r[None, :] * np.cos(delta)
                + (vertical[x, y] - observed_z[None, :]) ** 2
            )
            result[x, y] = np.sqrt(np.maximum(np.min(squared, axis=1), 0.0))
    if not np.all(np.isfinite(result)):
        raise ValueError("distance construction produced non-finite values")
    return result


def flattened_cell_indices(
    *,
    field_index: int,
    x_slice: slice,
    shape: tuple[int, int, int] = VOLUME_SHAPE,
) -> np.ndarray:
    """Canonical joint-field flattened indices for an x slab."""

    if not 0 <= int(field_index) < len(S0_FIELDS):
        raise ValueError("field_index is outside S0_FIELDS")
    xs = np.arange(*x_slice.indices(shape[0]), dtype=np.int64)
    if xs.size == 0:
        raise ValueError("x_slice is empty")
    cells = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)[xs]
    return int(field_index) * int(np.prod(shape)) + cells.ravel()
