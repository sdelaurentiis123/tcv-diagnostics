"""Causal feature construction and lightweight chronological probes.

No function here sees a future field while constructing context features.
Targets are passed separately so the boundary is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .statistics import (
    fit_ridge,
    predict_ridge,
    regression_metrics,
    standardize_training_features,
)


FIELD_NAMES = ("Ne", "Pe", "Pi", "phi", "Vi")
TOROIDAL_BANDS = {
    "low_k1_3": (1, 3),
    "mid_k4_5": (4, 5),
    "high_k6_7": (6, 7),
    "fine_k_ge_8": (8, None),
}


def _state_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 5 or array.shape[1] != len(FIELD_NAMES):
        raise ValueError("state features require [sample,5,x,y,z]")
    if not np.all(np.isfinite(array)):
        raise ValueError("state features contain non-finite values")
    return array


def toroidal_band_energy(values: np.ndarray) -> dict[str, np.ndarray]:
    array = _state_array(values)
    spectrum = np.fft.rfft(array, axis=-1, norm="ortho")
    power = np.abs(spectrum) ** 2
    output: dict[str, np.ndarray] = {}
    for name, (lower, upper) in TOROIDAL_BANDS.items():
        stop = power.shape[-1] if upper is None else min(int(upper) + 1, power.shape[-1])
        if lower >= stop:
            output[name] = np.zeros(array.shape[:2], dtype=np.float64)
        else:
            output[name] = np.mean(power[..., int(lower) : stop], axis=(2, 3, 4))
    return output


def causal_context_features(
    states: np.ndarray,
    *,
    displacement_features: np.ndarray | None = None,
    extra_features: np.ndarray | None = None,
    radial_bins: int = 8,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize states available at context time, never target time."""

    array = _state_array(states)
    sample_count, field_count, nx, _, _ = array.shape
    if nx % int(radial_bins):
        raise ValueError("radial bins must divide the x extent")
    columns: list[np.ndarray] = []
    names: list[str] = []
    for field_index, field in enumerate(FIELD_NAMES):
        field_values = array[:, field_index]
        if field == "phi":
            field_values = field_values - np.mean(field_values, axis=(1, 2, 3), keepdims=True)
        columns.append(np.mean(field_values, axis=(1, 2, 3)))
        names.append(f"{field}.spatial_mean")
        fluctuation = field_values - np.mean(field_values, axis=(1, 2, 3), keepdims=True)
        columns.append(np.sqrt(np.mean(fluctuation * fluctuation, axis=(1, 2, 3))))
        names.append(f"{field}.fluctuation_RMS")
        profile = np.mean(field_values, axis=(2, 3))
        binned = profile.reshape(sample_count, int(radial_bins), nx // int(radial_bins)).mean(axis=-1)
        for radial in range(int(radial_bins)):
            columns.append(binned[:, radial])
            names.append(f"{field}.radial_bin_{radial}")
        gradients = np.diff(binned, axis=1)
        for radial in range(int(radial_bins) - 1):
            columns.append(gradients[:, radial])
            names.append(f"{field}.radial_gradient_{radial}_{radial + 1}")
    bands = toroidal_band_energy(array)
    for band, energy in bands.items():
        for field_index, field in enumerate(FIELD_NAMES):
            columns.append(np.log(np.maximum(energy[:, field_index], np.finfo(float).tiny)))
            names.append(f"{field}.{band}.log_energy")
    spectrum = np.fft.rfft(array, axis=-1, norm="ortho")
    for k in (4, 5):
        if k >= spectrum.shape[-1]:
            continue
        coefficient = np.sum(spectrum[..., k], axis=(2, 3))
        for field_index, field in enumerate(FIELD_NAMES):
            phase = np.angle(coefficient[:, field_index])
            columns.extend((np.sin(phase), np.cos(phase)))
            names.extend((f"{field}.k{k}.phase_sin", f"{field}.k{k}.phase_cos"))
    if displacement_features is not None:
        displacement = np.asarray(displacement_features, dtype=np.float64)
        if displacement.ndim != 2 or displacement.shape[0] != sample_count:
            raise ValueError("displacement feature matrix differs")
        for index in range(displacement.shape[1]):
            columns.append(displacement[:, index])
            names.append(f"recent_displacement.{index}")
    if extra_features is not None:
        extra = np.asarray(extra_features, dtype=np.float64)
        if extra.ndim != 2 or extra.shape[0] != sample_count:
            raise ValueError("extra feature matrix differs")
        for index in range(extra.shape[1]):
            columns.append(extra[:, index])
            names.append(f"existing_regime_or_exact_state.{index}")
    matrix = np.column_stack(columns)
    if matrix.shape[0] != sample_count or not np.all(np.isfinite(matrix)):
        raise RuntimeError("causal context feature construction failed")
    return matrix, tuple(names)


def residual_scalar_targets(residual: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    array = _state_array(residual)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for field_index, field in enumerate(FIELD_NAMES):
        columns.append(np.mean(array[:, field_index] ** 2, axis=(1, 2, 3)))
        names.append(f"residual_energy.{field}")
    bands = toroidal_band_energy(array)
    for band, energy in bands.items():
        for field_index, field in enumerate(FIELD_NAMES):
            columns.append(energy[:, field_index])
            names.append(f"residual_spectral_energy.{field}.{band}")
    centered = array - np.mean(array, axis=(2, 3, 4), keepdims=True)
    for first in range(len(FIELD_NAMES)):
        for second in range(first + 1, len(FIELD_NAMES)):
            columns.append(np.mean(centered[:, first] * centered[:, second], axis=(1, 2, 3)))
            names.append(f"residual_cross_covariance.{FIELD_NAMES[first]}_{FIELD_NAMES[second]}")
    return np.column_stack(columns), tuple(names)


def append_target_columns(
    base: tuple[np.ndarray, tuple[str, ...]],
    values: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix, names = base
    columns = [matrix]
    output_names = list(names)
    for name, value in values.items():
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != matrix.shape[0]:
            raise ValueError(f"target column {name} differs")
        columns.append(array)
        output_names.extend(
            [name] if array.shape[1] == 1 else [f"{name}.{index}" for index in range(array.shape[1])]
        )
    return np.column_stack(columns), tuple(output_names)


@dataclass(frozen=True)
class TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict_one(self, row: np.ndarray) -> float:
        if self.feature is None:
            return self.value
        child = self.left if row[self.feature] <= float(self.threshold) else self.right
        if child is None:
            raise RuntimeError("regression tree has a missing child")
        return child.predict_one(row)


def _fit_tree_node(
    features: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    *,
    depth: int,
    maximum_depth: int,
    minimum_leaf: int,
) -> TreeNode:
    values = target[indices]
    value = float(np.mean(values))
    if depth >= maximum_depth or indices.size < 2 * minimum_leaf:
        return TreeNode(value)
    base_error = float(np.sum((values - value) ** 2))
    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    quantiles = np.linspace(0.1, 0.9, 9)
    for feature in range(features.shape[1]):
        candidates = np.unique(np.quantile(features[indices, feature], quantiles))
        for threshold in candidates:
            mask = features[indices, feature] <= threshold
            left = indices[mask]
            right = indices[~mask]
            if left.size < minimum_leaf or right.size < minimum_leaf:
                continue
            error = float(
                np.sum((target[left] - np.mean(target[left])) ** 2)
                + np.sum((target[right] - np.mean(target[right])) ** 2)
            )
            gain = base_error - error
            proposal = (gain, feature, float(threshold), left, right)
            if best is None or proposal[:3] > best[:3]:
                best = proposal
    if best is None or best[0] <= 0.0:
        return TreeNode(value)
    _, feature, threshold, left, right = best
    return TreeNode(
        value=value,
        feature=feature,
        threshold=threshold,
        left=_fit_tree_node(features, target, left, depth=depth + 1,
                            maximum_depth=maximum_depth, minimum_leaf=minimum_leaf),
        right=_fit_tree_node(features, target, right, depth=depth + 1,
                             maximum_depth=maximum_depth, minimum_leaf=minimum_leaf),
    )


def fit_shallow_tree(
    features: np.ndarray,
    target: np.ndarray,
    *,
    maximum_depth: int = 2,
    minimum_leaf: int = 24,
) -> TreeNode:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("tree feature/target shapes differ")
    return _fit_tree_node(
        x, y, np.arange(x.shape[0]), depth=0,
        maximum_depth=int(maximum_depth), minimum_leaf=int(minimum_leaf)
    )


def predict_tree(tree: TreeNode, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("tree prediction features must be a matrix")
    return np.asarray([tree.predict_one(row) for row in matrix], dtype=np.float64)


def _select_ridge_alpha(
    training_features: np.ndarray,
    training_target: np.ndarray,
    *,
    alphas: Sequence[float],
    block_size: int,
    minimum_blocks: int,
) -> float:
    x = np.asarray(training_features, dtype=np.float64)
    y = np.asarray(training_target, dtype=np.float64)
    block_count = x.shape[0] // int(block_size)
    if block_count < minimum_blocks + 1:
        raise ValueError("insufficient blocks for nested chronological ridge")
    scores: dict[float, list[float]] = {float(alpha): [] for alpha in alphas}
    for test_block in range(int(minimum_blocks), block_count):
        train_stop = test_block * int(block_size)
        test = slice(train_stop, train_stop + int(block_size))
        standardized_train, standardized_test = standardize_training_features(x[:train_stop], x[test])
        for alpha in scores:
            coefficient = fit_ridge(standardized_train, y[:train_stop], alpha=alpha)
            prediction = predict_ridge(standardized_test, coefficient).reshape(-1)
            scores[alpha].append(regression_metrics(y[test], prediction)["RMSE"])
    return min(scores, key=lambda alpha: (float(np.mean(scores[alpha])), alpha))


def evaluate_chronological_probes(
    training_features: np.ndarray,
    training_time: np.ndarray,
    training_target: np.ndarray,
    validation_features: np.ndarray,
    validation_time: np.ndarray,
    validation_target: np.ndarray,
    *,
    validation_block_ids: Sequence[str],
    block_size: int = 42,
    ridge_alphas: Sequence[float] = (1e-4, 1e-2, 1.0, 100.0),
    minimum_training_blocks: int = 5,
    tree_maximum_depth: int = 2,
    tree_minimum_leaf: int = 24,
) -> list[dict[str, float | str | int]]:
    """Fit only on chronological training data and score fixed later blocks."""

    x_train = np.asarray(training_features, dtype=np.float64)
    x_validation = np.asarray(validation_features, dtype=np.float64)
    y_train = np.asarray(training_target, dtype=np.float64).reshape(-1)
    y_validation = np.asarray(validation_target, dtype=np.float64).reshape(-1)
    time_train = np.asarray(training_time, dtype=np.float64).reshape(-1, 1)
    time_validation = np.asarray(validation_time, dtype=np.float64).reshape(-1, 1)
    if x_train.shape[0] != y_train.size or x_validation.shape[0] != y_validation.size:
        raise ValueError("chronological probe shapes differ")
    alpha = _select_ridge_alpha(
        x_train, y_train, alphas=ridge_alphas, block_size=block_size,
        minimum_blocks=minimum_training_blocks
    )
    standardized_train, standardized_validation = standardize_training_features(x_train, x_validation)
    standardized_time_train, standardized_time_validation = standardize_training_features(
        time_train, time_validation
    )
    context_coefficient = fit_ridge(standardized_train, y_train, alpha=alpha)
    time_coefficient = fit_ridge(standardized_time_train, y_train, alpha=alpha)
    tree = fit_shallow_tree(
        standardized_train, y_train, maximum_depth=tree_maximum_depth,
        minimum_leaf=tree_minimum_leaf
    )
    predictions = {
        "constant": np.full(y_validation.size, np.mean(y_train)),
        "time_only_ridge": predict_ridge(standardized_time_validation, time_coefficient).reshape(-1),
        "context_ridge": predict_ridge(standardized_validation, context_coefficient).reshape(-1),
        "context_depth2_tree": predict_tree(tree, standardized_validation),
    }
    if len(validation_block_ids) * int(block_size) != y_validation.size:
        raise ValueError("validation probe block identifiers differ")
    rows: list[dict[str, float | str | int]] = []
    for block_index, block_id in enumerate(validation_block_ids):
        selected = slice(block_index * int(block_size), (block_index + 1) * int(block_size))
        scale = float(np.std(y_validation[selected]))
        for model, prediction in predictions.items():
            metrics = regression_metrics(y_validation[selected], prediction[selected])
            rows.append(
                {
                    "evaluation": "fixed_validation_block",
                    "block": str(block_id),
                    "probe": model,
                    "selected_ridge_alpha": float(alpha),
                    "sample_count": int(block_size),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "normalized_RMSE": metrics["RMSE"] / scale if scale > 0.0 else math.nan,
                    "R2": metrics["R2"],
                }
            )
    return rows


def evaluate_chronological_probes_multi(
    training_features: np.ndarray,
    training_time: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_time: np.ndarray,
    validation_targets: np.ndarray,
    *,
    target_names: Sequence[str],
    validation_block_ids: Sequence[str],
    validation_block_sizes: Sequence[int] | None = None,
    block_size: int = 42,
    ridge_alphas: Sequence[float] = (1e-4, 1e-2, 1.0, 100.0),
    minimum_training_blocks: int = 5,
    tree_maximum_depth: int = 2,
    tree_minimum_leaf: int = 24,
) -> list[dict[str, float | str | int]]:
    """Multi-target version sharing each chronological ridge factorization."""

    x_train = np.asarray(training_features, dtype=np.float64)
    x_validation = np.asarray(validation_features, dtype=np.float64)
    y_train = np.asarray(training_targets, dtype=np.float64)
    y_validation = np.asarray(validation_targets, dtype=np.float64)
    names = tuple(str(value) for value in target_names)
    if (
        x_train.ndim != 2
        or x_validation.ndim != 2
        or y_train.ndim != 2
        or y_validation.ndim != 2
        or y_train.shape[0] != x_train.shape[0]
        or y_validation.shape[0] != x_validation.shape[0]
        or y_train.shape[1] != y_validation.shape[1]
        or y_train.shape[1] != len(names)
    ):
        raise ValueError("multi-target chronological probe shapes differ")
    blocks = x_train.shape[0] // int(block_size)
    if blocks < int(minimum_training_blocks) + 1:
        raise ValueError("multi-target chronological probe has too few blocks")
    alpha_scores = {
        float(alpha): [[] for _ in names] for alpha in ridge_alphas
    }
    for test_block in range(int(minimum_training_blocks), blocks):
        train_stop = test_block * int(block_size)
        test = slice(train_stop, train_stop + int(block_size))
        standardized_train, standardized_test = standardize_training_features(
            x_train[:train_stop], x_train[test]
        )
        for alpha in alpha_scores:
            coefficient = fit_ridge(standardized_train, y_train[:train_stop], alpha=alpha)
            prediction = predict_ridge(standardized_test, coefficient)
            errors = prediction - y_train[test]
            rmse = np.sqrt(np.mean(errors * errors, axis=0))
            for target_index, value in enumerate(rmse):
                alpha_scores[alpha][target_index].append(float(value))
    selected_alphas = np.asarray(
        [
            min(
                alpha_scores,
                key=lambda alpha: (
                    float(np.mean(alpha_scores[alpha][target_index])), alpha
                ),
            )
            for target_index in range(len(names))
        ],
        dtype=np.float64,
    )
    standardized_train, standardized_validation = standardize_training_features(
        x_train, x_validation
    )
    time_train = np.asarray(training_time, dtype=np.float64).reshape(-1, 1)
    time_validation = np.asarray(validation_time, dtype=np.float64).reshape(-1, 1)
    standardized_time_train, standardized_time_validation = standardize_training_features(
        time_train, time_validation
    )
    context_prediction = np.empty_like(y_validation)
    time_prediction = np.empty_like(y_validation)
    for alpha in sorted(set(selected_alphas.tolist())):
        selected = np.flatnonzero(selected_alphas == alpha)
        context_coefficient = fit_ridge(
            standardized_train, y_train[:, selected], alpha=alpha
        )
        time_coefficient = fit_ridge(
            standardized_time_train, y_train[:, selected], alpha=alpha
        )
        context_prediction[:, selected] = predict_ridge(
            standardized_validation, context_coefficient
        )
        time_prediction[:, selected] = predict_ridge(
            standardized_time_validation, time_coefficient
        )
    tree_prediction = np.empty_like(y_validation)
    for target_index in range(len(names)):
        tree = fit_shallow_tree(
            standardized_train,
            y_train[:, target_index],
            maximum_depth=tree_maximum_depth,
            minimum_leaf=tree_minimum_leaf,
        )
        tree_prediction[:, target_index] = predict_tree(tree, standardized_validation)
    predictions = {
        "constant": np.broadcast_to(np.mean(y_train, axis=0), y_validation.shape),
        "time_only_ridge": time_prediction,
        "context_ridge": context_prediction,
        "context_depth2_tree": tree_prediction,
    }
    validation_sizes = (
        tuple(int(value) for value in validation_block_sizes)
        if validation_block_sizes is not None
        else (int(block_size),) * len(validation_block_ids)
    )
    if (
        len(validation_block_ids) != len(validation_sizes)
        or any(value < 3 for value in validation_sizes)
        or sum(validation_sizes) != y_validation.shape[0]
    ):
        raise ValueError("multi-target validation blocks differ")
    rows: list[dict[str, float | str | int]] = []
    offset = 0
    for block_id, validation_size in zip(validation_block_ids, validation_sizes):
        selected_rows = slice(offset, offset + validation_size)
        offset += validation_size
        for target_index, target_name in enumerate(names):
            observed = y_validation[selected_rows, target_index]
            scale = float(np.std(observed))
            for model, prediction in predictions.items():
                metrics = regression_metrics(
                    observed, prediction[selected_rows, target_index]
                )
                rows.append(
                    {
                        "evaluation": "fixed_validation_block",
                        "block": str(block_id),
                        "target": target_name,
                        "probe": model,
                        "selected_ridge_alpha": float(selected_alphas[target_index]),
                        "sample_count": int(validation_size),
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "normalized_RMSE": metrics["RMSE"] / scale if scale > 0 else math.nan,
                        "R2": metrics["R2"],
                    }
                )
    return rows


def nearest_preceding_neighbors(
    training_embedding: np.ndarray,
    training_targets: np.ndarray,
    query_embedding: np.ndarray,
    query_targets: np.ndarray,
    *,
    k: int,
    minimum_separation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices/distances for causal, temporally separated neighbors."""

    train = np.asarray(training_embedding, dtype=np.float64)
    query = np.asarray(query_embedding, dtype=np.float64)
    train_time = np.asarray(training_targets, dtype=np.int64)
    query_time = np.asarray(query_targets, dtype=np.int64)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("neighbor embedding dimensions differ")
    indices = np.full((query.shape[0], int(k)), -1, dtype=np.int64)
    distances = np.full((query.shape[0], int(k)), np.nan, dtype=np.float64)
    for row in range(query.shape[0]):
        eligible = np.flatnonzero(train_time <= query_time[row] - int(minimum_separation))
        if eligible.size < int(k):
            continue
        squared = np.sum((train[eligible] - query[row]) ** 2, axis=1)
        order = np.argsort(squared, kind="stable")[: int(k)]
        indices[row] = eligible[order]
        distances[row] = np.sqrt(squared[order])
    return indices, distances


def neighbor_conditional_variance(
    neighbor_indices: np.ndarray,
    training_outcome: np.ndarray,
    query_outcome: np.ndarray,
) -> dict[str, float | int]:
    indices = np.asarray(neighbor_indices, dtype=np.int64)
    train = np.asarray(training_outcome, dtype=np.float64)
    query = np.asarray(query_outcome, dtype=np.float64)
    valid = np.all(indices >= 0, axis=1)
    if not np.any(valid):
        return {"query_count": 0, "conditional_variance": math.nan,
                "prediction_RMSE": math.nan}
    predictions = np.mean(train[indices[valid]], axis=1)
    conditional = np.mean(np.var(train[indices[valid]], axis=1, ddof=1))
    rmse = np.sqrt(np.mean((predictions - query[valid]) ** 2))
    return {
        "query_count": int(np.sum(valid)),
        "conditional_variance": float(conditional),
        "prediction_RMSE": float(rmse),
    }
