"""Chronological statistics, ESS, subspace transfer, and small probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np


def finite_vector(name: str, values: np.ndarray, *, minimum: int = 3) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(f"{name} must be a vector with at least {minimum} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def raw_sample_gram(
    values: np.ndarray,
    *,
    torch_device: object | None = None,
    row_batch: int = 64,
) -> np.ndarray:
    """Compute ``X X^T`` in fixed feature order, optionally on a GPU.

    The optional dependency is imported only when a device is supplied so
    data-independent unit tests remain NumPy-only.
    """

    matrix = np.asarray(values)
    if matrix.ndim < 2 or matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("raw Gram input must contain finite samples")
    flat = matrix.reshape(matrix.shape[0], -1)
    if torch_device is None:
        return np.asarray(flat, dtype=np.float64) @ np.asarray(flat, dtype=np.float64).T
    import torch

    device = torch.device(torch_device)
    source = torch.as_tensor(np.asarray(flat, dtype=np.float32), device=device).to(torch.float64)
    output = torch.empty((flat.shape[0], flat.shape[0]), device=device, dtype=torch.float64)
    for start in range(0, flat.shape[0], int(row_batch)):
        stop = min(start + int(row_batch), flat.shape[0])
        # Accumulate float64 for stable small-eigenvalue ordering.  The H100
        # path is diagnostic and does not alter the stored float32 samples.
        output[start:stop] = source[start:stop] @ source.T
    result = output.cpu().numpy()
    del source, output
    return np.asarray(0.5 * (result + result.T), dtype=np.float64)


def autocorrelation(values: np.ndarray, *, detrend: bool = False) -> np.ndarray:
    """Biased normalized ACF with optional least-squares linear detrending."""

    series = finite_vector("autocorrelation series", values)
    if detrend:
        coordinate = np.linspace(-1.0, 1.0, series.size)
        design = np.column_stack((np.ones(series.size), coordinate))
        coefficient, *_ = np.linalg.lstsq(design, series, rcond=None)
        centered = series - design @ coefficient
    else:
        centered = series - np.mean(series)
    variance = float(np.dot(centered, centered))
    if variance <= 0.0:
        return np.concatenate(([1.0], np.zeros(series.size - 1)))
    size = 1 << (2 * series.size - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[: series.size]
    overlap = np.arange(series.size, 0, -1, dtype=np.float64)
    covariance = covariance / overlap
    return np.asarray(covariance / covariance[0], dtype=np.float64)


def geyer_initial_positive_iat(rho: np.ndarray) -> tuple[float, int, bool]:
    curve = finite_vector("ACF", rho, minimum=2)
    if not math.isclose(float(curve[0]), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("ACF lag zero must be one")
    pair_sums = []
    lag = 1
    while lag + 1 < curve.size:
        pair = float(curve[lag] + curve[lag + 1])
        if pair <= 0.0:
            break
        pair_sums.append(pair)
        lag += 2
    tau = max(1.0, 1.0 + 2.0 * float(np.sum(pair_sums)))
    right_censored = lag + 1 >= curve.size and bool(pair_sums)
    return tau, min(lag, curve.size - 1), right_censored


def fixed_window_iat(rho: np.ndarray, window: int) -> tuple[float, int]:
    curve = finite_vector("ACF", rho, minimum=2)
    stop = min(int(window), curve.size - 1)
    if stop < 1:
        raise ValueError("IAT window must retain at least lag one")
    return max(1.0, 1.0 + 2.0 * float(np.sum(curve[1 : stop + 1]))), stop


def self_consistent_iat(
    rho: np.ndarray,
    *,
    multiplier: float = 5.0,
) -> tuple[float, int, bool]:
    curve = finite_vector("ACF", rho, minimum=2)
    if multiplier <= 0.0:
        raise ValueError("self-consistent IAT multiplier must be positive")
    tau = 1.0
    for lag in range(1, curve.size):
        tau = max(1.0, 1.0 + 2.0 * float(np.sum(curve[1 : lag + 1])))
        if lag >= multiplier * tau:
            return tau, lag, False
    return tau, curve.size - 1, True


def effective_sample_record(
    values: np.ndarray,
    *,
    detrend: bool,
    fixed_windows: Sequence[int] = (8, 16, 32, 64),
    self_consistent_multiplier: float = 5.0,
) -> dict[str, object]:
    series = finite_vector("ESS series", values)
    rho = autocorrelation(series, detrend=detrend)
    primary, primary_lag, primary_censored = geyer_initial_positive_iat(rho)
    fixed = {}
    for window in fixed_windows:
        if int(window) < series.size:
            tau, used = fixed_window_iat(rho, int(window))
            fixed[str(window)] = {
                "tau_int": tau,
                "effective_sample_size": min(float(series.size), series.size / tau),
                "last_lag": used,
            }
    self_tau, self_lag, self_censored = self_consistent_iat(
        rho,
        multiplier=self_consistent_multiplier,
    )
    return {
        "sample_count": int(series.size),
        "detrended": bool(detrend),
        "primary_method": "Geyer_initial_positive_pair_sequence",
        "primary_tau_int": primary,
        "primary_effective_sample_size": min(float(series.size), series.size / primary),
        "primary_last_lag": int(primary_lag),
        "primary_right_censored": bool(primary_censored),
        "self_consistent_tau_int": self_tau,
        "self_consistent_effective_sample_size": min(
            float(series.size), series.size / self_tau
        ),
        "self_consistent_last_lag": int(self_lag),
        "self_consistent_right_censored": bool(self_censored),
        "fixed_windows": fixed,
        "rho": rho,
    }


def moving_block_indices(
    sample_count: int,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Non-circular moving-block bootstrap indices."""

    count = int(sample_count)
    length = int(block_length)
    draws = int(replicates)
    if count <= 1 or length <= 0 or length > count or draws <= 0:
        raise ValueError("invalid moving-block bootstrap dimensions")
    blocks_per_draw = math.ceil(count / length)
    generator = np.random.default_rng(int(seed))
    maximum_start = count - length
    result = np.empty((draws, count), dtype=np.int64)
    offsets = np.arange(length, dtype=np.int64)
    for replicate in range(draws):
        starts = generator.integers(0, maximum_start + 1, size=blocks_per_draw)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[:count]
        result[replicate] = indices
    return result


def block_bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    block_length: int,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    series = finite_vector("bootstrap series", values)
    indices = moving_block_indices(
        series.size,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )
    estimates = np.asarray([statistic(series[index]) for index in indices], dtype=np.float64)
    if not np.all(np.isfinite(estimates)):
        raise ValueError("bootstrap statistic returned non-finite values")
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "estimate": float(statistic(series)),
        "lower": float(np.quantile(estimates, tail)),
        "upper": float(np.quantile(estimates, 1.0 - tail)),
        "replicates": int(replicates),
        "block_length": int(block_length),
    }


@dataclass(frozen=True)
class PCABasis:
    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    sample_count: int

    @property
    def rank(self) -> int:
        return int(self.modes.shape[0])


def fit_pca(
    values: np.ndarray,
    *,
    maximum_rank: int | None = None,
    relative_threshold: float = 1e-10,
) -> PCABasis:
    """Method-of-snapshots PCA for `[sample,feature]` arrays."""

    samples = np.asarray(values)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("PCA values must have shape [sample>=2,feature>=1]")
    if not np.all(np.isfinite(samples)):
        raise ValueError("PCA values contain non-finite values")
    samples64 = np.asarray(samples, dtype=np.float64)
    mean = np.mean(samples64, axis=0)
    centered = samples64 - mean
    gram = centered @ centered.T / (samples64.shape[0] - 1)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[0] <= 0.0:
        return PCABasis(mean, np.empty((0, samples.shape[1])), np.empty(0), samples.shape[0])
    positive = eigenvalues > relative_threshold * eigenvalues[0]
    rank = int(np.sum(positive))
    if maximum_rank is not None:
        rank = min(rank, int(maximum_rank))
    eigenvalues = eigenvalues[:rank]
    eigenvectors = eigenvectors[:, :rank]
    modes = (eigenvectors.T @ centered) / np.sqrt(
        (samples64.shape[0] - 1) * eigenvalues[:, None]
    )
    return PCABasis(
        mean=np.asarray(mean, dtype=np.float64),
        modes=np.asarray(modes, dtype=np.float64),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        sample_count=int(samples64.shape[0]),
    )


def project_pca(values: np.ndarray, basis: PCABasis, *, rank: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != basis.mean.size:
        raise ValueError("PCA projection feature dimension differs")
    selected = int(rank)
    if selected < 0 or selected > basis.rank:
        raise ValueError("PCA projection rank is unavailable")
    if selected == 0:
        return np.broadcast_to(basis.mean, samples.shape).copy()
    modes = basis.modes[:selected]
    centered = samples - basis.mean
    return basis.mean + (centered @ modes.T) @ modes


def variance_capture(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    truth = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(reconstruction, dtype=np.float64)
    if truth.shape != candidate.shape:
        raise ValueError("variance-capture shapes differ")
    denominator = float(np.sum(truth * truth))
    if denominator == 0.0:
        return 1.0 if np.array_equal(truth, candidate) else -math.inf
    return 1.0 - float(np.sum((truth - candidate) ** 2)) / denominator


def principal_angle_summary(
    first: PCABasis,
    second: PCABasis,
    *,
    rank: int,
) -> dict[str, float]:
    selected = int(rank)
    if selected <= 0 or selected > min(first.rank, second.rank):
        raise ValueError("principal-angle rank is unavailable")
    if first.modes.shape[1] != second.modes.shape[1]:
        raise ValueError("principal-angle feature dimensions differ")
    singular = np.linalg.svd(
        first.modes[:selected] @ second.modes[:selected].T,
        compute_uv=False,
    )
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.arccos(singular)
    return {
        "rank": selected,
        "minimum_cosine": float(np.min(singular)),
        "mean_squared_cosine": float(np.mean(singular**2)),
        "maximum_angle_degrees": float(np.degrees(np.max(angles))),
    }


@dataclass(frozen=True)
class SnapshotSubspace:
    """Source-centered eigensystem that does not materialize spatial modes."""

    indices: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    sample_count: int

    @property
    def rank(self) -> int:
        return int(self.eigenvalues.size)


def fit_snapshot_subspace_from_raw_gram(
    raw_gram: np.ndarray,
    indices: Sequence[int],
    *,
    relative_threshold: float = 1e-10,
) -> SnapshotSubspace:
    gram = np.asarray(raw_gram, dtype=np.float64)
    selected = np.asarray(tuple(int(value) for value in indices), dtype=np.int64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or selected.size < 2:
        raise ValueError("raw Gram/subspace dimensions differ")
    block = gram[np.ix_(selected, selected)]
    centering = np.eye(selected.size) - np.ones((selected.size, selected.size)) / selected.size
    covariance = centering @ block @ centering / (selected.size - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if eigenvalues[0] <= 0.0:
        raise ValueError("snapshot source block has no positive variance")
    keep = eigenvalues > float(relative_threshold) * eigenvalues[0]
    return SnapshotSubspace(
        indices=selected,
        eigenvalues=np.asarray(eigenvalues[keep], dtype=np.float64),
        eigenvectors=np.asarray(eigenvectors[:, keep], dtype=np.float64),
        sample_count=int(selected.size),
    )


def snapshot_transfer_components(
    raw_gram: np.ndarray,
    source: SnapshotSubspace,
    target_indices: Sequence[int],
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-target captured and total energy about the source mean."""

    gram = np.asarray(raw_gram, dtype=np.float64)
    target = np.asarray(tuple(int(value) for value in target_indices), dtype=np.int64)
    selected_rank = min(int(rank), source.rank)
    if target.size < 1 or selected_rank < 0:
        raise ValueError("snapshot transfer target/rank differs")
    source_indices = source.indices
    n_source = source.sample_count
    cross = gram[np.ix_(target, source_indices)]
    source_centering = np.eye(n_source) - np.ones((n_source, n_source)) / n_source
    source_block = gram[np.ix_(source_indices, source_indices)]
    # D=(Y-mean_X) and Xc=H X.  The source empirical mean is not generally
    # orthogonal to each spatial PCA mode, so subtract its Xc inner products
    # explicitly; using only Y X^T H would project uncentered target samples.
    mean_source_against_centered_source = (
        np.mean(source_block, axis=0) @ source_centering
    )
    cross_centered = (
        cross @ source_centering
        - mean_source_against_centered_source[None, :]
    )
    if selected_rank:
        denominator = np.sqrt(
            (n_source - 1) * source.eigenvalues[:selected_rank]
        )
        coefficients = (
            cross_centered @ source.eigenvectors[:, :selected_rank]
        ) / denominator[None, :]
        captured = np.sum(coefficients * coefficients, axis=1)
    else:
        captured = np.zeros(target.size, dtype=np.float64)
    mean_norm = float(np.sum(source_block)) / (n_source * n_source)
    target_norm = np.diag(gram[np.ix_(target, target)])
    target_mean_dot = np.mean(gram[np.ix_(target, source_indices)], axis=1)
    total = target_norm - 2.0 * target_mean_dot + mean_norm
    if np.any(total <= 0.0):
        raise ValueError("snapshot transfer target energy is nonpositive")
    return np.asarray(captured), np.asarray(total)


def snapshot_transfer_capture(
    raw_gram: np.ndarray,
    source: SnapshotSubspace,
    target_indices: Sequence[int],
    *,
    rank: int,
) -> float:
    captured, total = snapshot_transfer_components(
        raw_gram, source, target_indices, rank=rank
    )
    return float(np.sum(captured) / np.sum(total))


def snapshot_principal_angles(
    raw_gram: np.ndarray,
    first: SnapshotSubspace,
    second: SnapshotSubspace,
    *,
    rank: int,
) -> dict[str, float]:
    selected = int(rank)
    if selected <= 0 or selected > min(first.rank, second.rank):
        raise ValueError("snapshot principal-angle rank is unavailable")
    left_centering = np.eye(first.sample_count) - np.ones(
        (first.sample_count, first.sample_count)
    ) / first.sample_count
    right_centering = np.eye(second.sample_count) - np.ones(
        (second.sample_count, second.sample_count)
    ) / second.sample_count
    cross = np.asarray(raw_gram, dtype=np.float64)[np.ix_(first.indices, second.indices)]
    numerator = (
        first.eigenvectors[:, :selected].T
        @ left_centering
        @ cross
        @ right_centering
        @ second.eigenvectors[:, :selected]
    )
    left_scale = np.sqrt((first.sample_count - 1) * first.eigenvalues[:selected])
    right_scale = np.sqrt((second.sample_count - 1) * second.eigenvalues[:selected])
    overlap = numerator / left_scale[:, None] / right_scale[None, :]
    singular = np.clip(np.linalg.svd(overlap, compute_uv=False), 0.0, 1.0)
    angles = np.arccos(singular)
    return {
        "rank": selected,
        "minimum_cosine": float(np.min(singular)),
        "mean_squared_cosine": float(np.mean(singular**2)),
        "maximum_angle_degrees": float(np.degrees(np.max(angles))),
    }


def standardize_training_features(
    training: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    train = np.asarray(training, dtype=np.float64)
    if train.ndim != 2 or train.shape[0] < 2:
        raise ValueError("training features must be a sample matrix")
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    output = [(train - mean) / scale]
    for values in others:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != train.shape[1]:
            raise ValueError("feature matrix dimensions differ")
        output.append((matrix - mean) / scale)
    return tuple(output)


def fit_ridge(features: np.ndarray, targets: np.ndarray, *, alpha: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim not in (1, 2) or y.shape[0] != x.shape[0]:
        raise ValueError("ridge feature/target shapes differ")
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("ridge alpha must be finite and nonnegative")
    y2 = y[:, None] if y.ndim == 1 else y
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y2, axis=0)
    centered_x = x - x_mean
    centered_y = y2 - y_mean
    ridge = float(alpha)
    if ridge == 0.0:
        slopes, *_ = np.linalg.lstsq(centered_x, centered_y, rcond=None)
    elif x.shape[1] <= x.shape[0]:
        system = centered_x.T @ centered_x + ridge * np.eye(x.shape[1])
        slopes = np.linalg.solve(system, centered_x.T @ centered_y)
    else:
        # The delay probes are deliberately wide.  The dual solve is exactly
        # equivalent to primal ridge while keeping the system chronological-
        # sample sized instead of feature sized.
        system = centered_x @ centered_x.T + ridge * np.eye(x.shape[0])
        dual = np.linalg.solve(system, centered_y)
        slopes = centered_x.T @ dual
    intercept = y_mean - x_mean @ slopes
    return np.vstack((intercept[None, :], slopes))


def predict_ridge(features: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    beta = np.asarray(coefficient, dtype=np.float64)
    if x.ndim != 2 or beta.ndim != 2 or beta.shape[0] != x.shape[1] + 1:
        raise ValueError("ridge prediction shapes differ")
    return np.column_stack((np.ones(x.shape[0]), x)) @ beta


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    observed = finite_vector("regression truth", np.asarray(truth).reshape(-1))
    predicted = finite_vector("regression prediction", np.asarray(prediction).reshape(-1))
    if observed.shape != predicted.shape:
        raise ValueError("regression truth/prediction sizes differ")
    error = predicted - observed
    centered = observed - np.mean(observed)
    denominator = float(np.sum(centered * centered))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error * error))),
        "R2": 1.0 - float(np.sum(error * error)) / denominator if denominator > 0 else math.nan,
    }


def rolling_origin_splits(
    block_sizes: Sequence[int],
    *,
    minimum_training_blocks: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    sizes = tuple(int(value) for value in block_sizes)
    if any(value <= 0 for value in sizes):
        raise ValueError("rolling-origin block sizes must be positive")
    minimum = int(minimum_training_blocks)
    if minimum < 1 or minimum >= len(sizes):
        raise ValueError("invalid rolling-origin minimum training blocks")
    boundaries = np.concatenate(([0], np.cumsum(sizes)))
    return tuple(
        (
            np.arange(0, boundaries[test_block], dtype=np.int64),
            np.arange(boundaries[test_block], boundaries[test_block + 1], dtype=np.int64),
        )
        for test_block in range(minimum, len(sizes))
    )


def permute_complete_blocks(
    block_sizes: Sequence[int],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    sizes = tuple(int(value) for value in block_sizes)
    if len(set(sizes)) != 1:
        raise ValueError("Phase 3.5 block permutations require equal block sizes")
    block_size = sizes[0]
    generator = np.random.default_rng(int(seed))
    result = np.empty((int(replicates), sum(sizes)), dtype=np.int64)
    base = np.arange(len(sizes), dtype=np.int64)
    offsets = np.arange(block_size, dtype=np.int64)
    for replicate in range(int(replicates)):
        order = generator.permutation(base)
        result[replicate] = (order[:, None] * block_size + offsets[None, :]).reshape(-1)
    return result
