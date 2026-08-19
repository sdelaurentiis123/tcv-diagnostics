"""Data-independent residual-KL primitives for the Paper 0 oracle.

This module cannot route datasets, open artifacts, load checkpoints, perform
model inference, or train parameters.  It receives already constructed arrays
and implements the prospectively frozen residual centering, method-of-
snapshots decomposition, projection, and static Gaussian reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .b5_residual_audit import B5_FIELDS


KL_FIELDS = B5_FIELDS
KL_PHI_INDEX = KL_FIELDS.index("phi")
KL_POSITIVE_EIGENVALUE_RELATIVE_THRESHOLD = 1e-10
KL_RANK_LADDER: tuple[int | str, ...] = (
    0,
    8,
    16,
    32,
    44,
    64,
    128,
    256,
    "full_positive_training_rank",
)
KL_STATIC_RANK_CANDIDATES = (8, 16, 32, 44, 64, 128)
KL_STATIC_VARIANCE_TARGET = 0.90
KL_MASTER_SEED = 2026081901


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _canonical_samples(name: str, values: np.ndarray) -> np.ndarray:
    array = _finite_real(name, values)
    if array.ndim != 5 or array.shape[1] != len(KL_FIELDS):
        raise ValueError(f"{name} must have axes [sample,field,x,y,z]")
    if array.shape[0] < 2 or min(array.shape[2:]) < 2:
        raise ValueError(f"{name} dimensions are too short")
    return np.ascontiguousarray(array, dtype=np.float64)


def _maximum_relative(values: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.max(np.abs(np.asarray(values, dtype=np.float64))))
    denominator = max(
        float(np.max(np.abs(np.asarray(reference, dtype=np.float64)))),
        np.finfo(np.float64).tiny,
    )
    return numerator / denominator


def gauge_fixed_residual(truth: np.ndarray, h1_mean: np.ndarray) -> np.ndarray:
    """Return truth-minus-H1 after independent per-sample phi gauge fixing."""

    observed = _canonical_samples("KL truth", truth)
    predicted = _canonical_samples("KL H1 mean", h1_mean)
    if observed.shape != predicted.shape:
        raise ValueError("KL truth and H1 mean shapes differ")
    observed = np.array(observed, copy=True, order="C")
    predicted = np.array(predicted, copy=True, order="C")
    for values in (observed, predicted):
        phi = values[:, KL_PHI_INDEX]
        phi_mean = np.mean(phi, axis=(1, 2, 3), keepdims=True, dtype=np.float64)
        values[:, KL_PHI_INDEX] = phi - phi_mean
    return np.asarray(observed - predicted, dtype=np.float64)


@dataclass(frozen=True)
class ResidualCentering:
    """Frozen separation of forecast bias and covariance centering."""

    axisymmetric_bias: np.ndarray
    covariance_empirical_mean: np.ndarray
    covariance_centered: np.ndarray
    maximum_relative_row_sum: float
    maximum_relative_empirical_mean_toroidal_average: float

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.covariance_centered.shape[1:])


def center_training_residual(residual: np.ndarray) -> ResidualCentering:
    """Separate axisymmetric bias and covariance-only empirical centering.

    The input must already use the frozen truth-minus-H1 sign and phi gauge.
    The returned empirical mean is never a forecast-mean correction.
    """

    values = _canonical_samples("training residual", residual)
    bias = np.mean(values, axis=(0, 4), dtype=np.float64)
    fluctuation = values - bias[None, ..., None]
    empirical = np.mean(fluctuation, axis=0, dtype=np.float64)
    centered = fluctuation - empirical[None]
    row_sum = np.sum(centered, axis=0, dtype=np.float64)
    empirical_toroidal_average = np.mean(empirical, axis=-1, dtype=np.float64)
    maximum_relative_row_sum = _maximum_relative(row_sum, centered) / float(
        values.shape[0]
    )
    maximum_relative_empirical_toroidal_average = _maximum_relative(
        empirical_toroidal_average,
        fluctuation,
    )
    tolerance = 5e-13
    if maximum_relative_row_sum > tolerance:
        raise RuntimeError("covariance-centered training rows do not sum to zero")
    if maximum_relative_empirical_toroidal_average > tolerance:
        raise RuntimeError(
            "covariance empirical mean has a nonzero toroidal average"
        )
    return ResidualCentering(
        axisymmetric_bias=np.asarray(bias, dtype=np.float64),
        covariance_empirical_mean=np.asarray(empirical, dtype=np.float64),
        covariance_centered=np.asarray(centered, dtype=np.float64),
        maximum_relative_row_sum=maximum_relative_row_sum,
        maximum_relative_empirical_mean_toroidal_average=(
            maximum_relative_empirical_toroidal_average
        ),
    )


@dataclass(frozen=True)
class SnapshotKLBasis:
    """Positive method-of-snapshots eigensystem in canonical sample axes."""

    eigenvalues: np.ndarray
    modes: np.ndarray
    gram: np.ndarray
    sample_shape: tuple[int, ...]
    relative_threshold: float
    maximum_orthonormality_error: float
    full_rank_training_relative_rms: float
    minimum_gram_eigenvalue: float

    @property
    def positive_rank(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def cumulative_variance_fraction(self) -> np.ndarray:
        total = float(np.sum(self.eigenvalues, dtype=np.float64))
        if total <= 0.0:
            raise RuntimeError("KL eigenvalue sum is nonpositive")
        return np.cumsum(self.eigenvalues, dtype=np.float64) / total

    def resolved_rank(self, rank: int | str) -> int:
        if rank == "full_positive_training_rank":
            return self.positive_rank
        resolved = int(rank)
        if resolved < 0 or resolved > self.positive_rank:
            raise ValueError("requested KL rank is unavailable")
        return resolved


@dataclass(frozen=True)
class SnapshotEigenSystem:
    """Small snapshot-space eigensystem used by the streaming implementation."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    gram: np.ndarray
    sample_count: int
    feature_count: int
    relative_threshold: float
    minimum_gram_eigenvalue: float
    maximum_relative_feature_mean: float

    @property
    def positive_rank(self) -> int:
        return int(self.eigenvalues.size)


def streaming_snapshot_gram(
    covariance_centered_2d: np.ndarray,
    *,
    feature_chunk_size: int,
) -> tuple[np.ndarray, float]:
    """Accumulate ``R R^T/(T-1)`` in one fixed contiguous feature order."""

    samples = np.asarray(covariance_centered_2d)
    if (
        samples.ndim != 2
        or samples.shape[0] < 2
        or samples.shape[1] < 1
        or np.iscomplexobj(samples)
        or not np.issubdtype(samples.dtype, np.number)
    ):
        raise ValueError("streaming KL matrix must be real [sample,feature]")
    chunk = int(feature_chunk_size)
    if chunk < 1:
        raise ValueError("streaming KL feature chunk must be positive")
    gram = np.zeros((samples.shape[0], samples.shape[0]), dtype=np.float64)
    maximum_relative_mean = 0.0
    for start in range(0, samples.shape[1], chunk):
        stop = min(start + chunk, samples.shape[1])
        block = _finite_real(
            "streaming covariance-centered feature block",
            samples[:, start:stop],
        )
        feature_sum = np.sum(block, axis=0, dtype=np.float64)
        maximum_relative_mean = max(
            maximum_relative_mean,
            _maximum_relative(feature_sum, block) / float(samples.shape[0]),
        )
        gram += block @ block.T
    if maximum_relative_mean > 5e-7:
        # Production snapshots are stored as float32, so this gate is looser
        # than the float64 construction gate while still detecting missing
        # centering or corrupted blocks by many orders of magnitude.
        raise ValueError("streaming KL snapshots are not centered over samples")
    gram /= float(samples.shape[0] - 1)
    gram = 0.5 * (gram + gram.T)
    if not np.all(np.isfinite(gram)):
        raise ValueError("streaming KL Gram matrix is non-finite")
    return gram, maximum_relative_mean


def diagonalize_snapshot_gram(
    gram: np.ndarray,
    *,
    sample_count: int,
    feature_count: int,
    maximum_relative_feature_mean: float,
    relative_threshold: float = KL_POSITIVE_EIGENVALUE_RELATIVE_THRESHOLD,
) -> SnapshotEigenSystem:
    """Diagonalize a precomputed snapshot Gram matrix with frozen truncation."""

    matrix = _finite_real("snapshot Gram matrix", gram)
    samples = int(sample_count)
    features = int(feature_count)
    if matrix.shape != (samples, samples) or samples < 2 or features < 1:
        raise ValueError("snapshot Gram dimensions differ")
    symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    symmetry_scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    if symmetry_error / symmetry_scale > 2e-13:
        raise ValueError("snapshot Gram matrix is not symmetric")
    threshold = float(relative_threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("KL relative eigenvalue threshold must lie in (0,1)")
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    values = np.asarray(values[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    largest = float(values[0])
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError("KL Gram matrix has no positive variance")
    minimum = float(values[-1])
    if minimum < -threshold * largest:
        raise ValueError("KL Gram matrix is not positive semidefinite")
    keep = values > threshold * largest
    positive_values = values[keep]
    positive_vectors = vectors[:, keep]
    maximum_rank = min(samples - 1, features)
    if positive_values.size < 1 or positive_values.size > maximum_rank:
        raise RuntimeError("KL positive rank violates centered-snapshot bound")
    return SnapshotEigenSystem(
        eigenvalues=positive_values,
        eigenvectors=positive_vectors,
        gram=np.asarray(matrix, dtype=np.float64),
        sample_count=samples,
        feature_count=features,
        relative_threshold=threshold,
        minimum_gram_eigenvalue=minimum,
        maximum_relative_feature_mean=float(maximum_relative_feature_mean),
    )


def snapshot_mode_block(
    covariance_centered_feature_block: np.ndarray,
    eigensystem: SnapshotEigenSystem,
) -> np.ndarray:
    """Construct all retained spatial modes for one contiguous feature block."""

    block = _finite_real(
        "covariance-centered snapshot feature block",
        covariance_centered_feature_block,
    )
    if block.ndim != 2 or block.shape[0] != eigensystem.sample_count:
        raise ValueError("snapshot mode feature block dimensions differ")
    denominator = np.sqrt(
        float(eigensystem.sample_count - 1) * eigensystem.eigenvalues,
        dtype=np.float64,
    )
    modes = (eigensystem.eigenvectors.T @ block) / denominator[:, None]
    if not np.all(np.isfinite(modes)):
        raise ValueError("snapshot mode block is non-finite")
    return np.asarray(modes, dtype=np.float64)


def fit_snapshot_kl(
    covariance_centered: np.ndarray,
    *,
    relative_threshold: float = KL_POSITIVE_EIGENVALUE_RELATIVE_THRESHOLD,
) -> SnapshotKLBasis:
    """Fit a float64 method-of-snapshots KL basis to centered samples."""

    samples = _finite_real("covariance-centered snapshots", covariance_centered)
    if samples.ndim < 2 or samples.shape[0] < 2:
        raise ValueError("KL snapshots must have a sample axis and feature axes")
    threshold = float(relative_threshold)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("KL relative eigenvalue threshold must lie in (0,1)")
    flat = np.ascontiguousarray(samples.reshape(samples.shape[0], -1))
    gram, maximum_relative_mean = streaming_snapshot_gram(
        flat,
        feature_chunk_size=flat.shape[1],
    )
    eigensystem = diagonalize_snapshot_gram(
        gram,
        sample_count=flat.shape[0],
        feature_count=flat.shape[1],
        maximum_relative_feature_mean=maximum_relative_mean,
        relative_threshold=threshold,
    )
    positive_values = eigensystem.eigenvalues
    modes_flat = snapshot_mode_block(flat, eigensystem)
    orthogonality = modes_flat @ modes_flat.T
    maximum_orthonormality_error = float(
        np.max(np.abs(orthogonality - np.eye(positive_values.size)))
    )
    coefficients = flat @ modes_flat.T
    reconstructed = coefficients @ modes_flat
    denominator_rms = float(np.sqrt(np.mean(flat * flat, dtype=np.float64)))
    full_rank_relative_rms = (
        float(
            np.sqrt(
                np.mean((flat - reconstructed) ** 2, dtype=np.float64)
            )
            / denominator_rms
        )
        if denominator_rms > 0.0
        else math.nan
    )
    return SnapshotKLBasis(
        eigenvalues=positive_values,
        modes=np.asarray(
            modes_flat.reshape(positive_values.size, *samples.shape[1:]),
            dtype=np.float64,
        ),
        gram=gram,
        sample_shape=tuple(int(value) for value in samples.shape[1:]),
        relative_threshold=threshold,
        maximum_orthonormality_error=maximum_orthonormality_error,
        full_rank_training_relative_rms=full_rank_relative_rms,
        minimum_gram_eigenvalue=eigensystem.minimum_gram_eigenvalue,
    )


def project_onto_kl(
    values: np.ndarray,
    basis: SnapshotKLBasis,
    *,
    rank: int | str,
) -> np.ndarray:
    """Project one or more fluctuation samples onto a frozen KL subspace."""

    array = _finite_real("KL projection values", values)
    if array.ndim < len(basis.sample_shape):
        raise ValueError("KL projection tensor has too few axes")
    if tuple(array.shape[-len(basis.sample_shape) :]) != basis.sample_shape:
        raise ValueError("KL projection sample shape differs")
    leading = array.shape[: -len(basis.sample_shape)]
    flat = array.reshape(-1, int(np.prod(basis.sample_shape)))
    resolved = basis.resolved_rank(rank)
    if resolved == 0:
        reconstructed = np.zeros_like(flat)
    else:
        modes = basis.modes[:resolved].reshape(resolved, -1)
        reconstructed = (flat @ modes.T) @ modes
    return np.asarray(reconstructed.reshape(*leading, *basis.sample_shape))


def reconstruction_variance_capture(
    reference: np.ndarray,
    reconstruction: np.ndarray,
) -> dict[str, Any]:
    """Return total and field-wise captured squared fluctuation energy."""

    observed = _canonical_samples("variance-capture reference", reference)
    candidate = _canonical_samples("variance-capture reconstruction", reconstruction)
    if observed.shape != candidate.shape:
        raise ValueError("variance-capture shapes differ")

    def fraction(first: np.ndarray, second: np.ndarray) -> float:
        denominator = float(np.sum(first * first, dtype=np.float64))
        numerator = float(np.sum((first - second) ** 2, dtype=np.float64))
        return 1.0 - numerator / denominator if denominator > 0.0 else math.nan

    return {
        "total": fraction(observed, candidate),
        "fields": {
            field: fraction(observed[:, channel], candidate[:, channel])
            for channel, field in enumerate(KL_FIELDS)
        },
    }


def select_static_rank(
    eigenvalues: np.ndarray,
    *,
    positive_rank: int | None = None,
    candidates: Sequence[int] = KL_STATIC_RANK_CANDIDATES,
    variance_target: float = KL_STATIC_VARIANCE_TARGET,
) -> dict[str, Any]:
    """Apply the frozen training-only static-rank selector."""

    values = _finite_real("KL eigenvalues", eigenvalues)
    if values.ndim != 1 or values.size < 1 or np.any(values <= 0.0):
        raise ValueError("KL eigenvalues must be one-dimensional and positive")
    if np.any(np.diff(values) > 0.0):
        raise ValueError("KL eigenvalues must be sorted descending")
    available = values.size if positive_rank is None else int(positive_rank)
    if available != values.size:
        raise ValueError("positive KL rank differs from eigenvalue count")
    frozen_candidates = tuple(int(value) for value in candidates)
    if frozen_candidates != KL_STATIC_RANK_CANDIDATES:
        raise ValueError("static KL rank candidates differ from protocol")
    target = float(variance_target)
    if target != KL_STATIC_VARIANCE_TARGET:
        raise ValueError("static KL variance target differs from protocol")
    cumulative = np.cumsum(values, dtype=np.float64) / float(np.sum(values))
    for rank in frozen_candidates:
        if rank <= available and cumulative[rank - 1] >= target:
            return {
                "rank": rank,
                "label": "training_90_percent_rank",
                "cumulative_training_variance_fraction": float(
                    cumulative[rank - 1]
                ),
                "validation_used": False,
            }
    fallback = 128
    if available < fallback:
        raise RuntimeError("training-variance fallback rank 128 is unavailable")
    return {
        "rank": fallback,
        "label": "training_variance_cap_bound",
        "cumulative_training_variance_fraction": float(cumulative[fallback - 1]),
        "validation_used": False,
    }


def generate_seed_bank(
    *,
    target_count: int = 126,
    ensemble_size: int = 32,
    master_seed: int = KL_MASTER_SEED,
) -> np.ndarray:
    """Generate the immutable target/member seed bank with PCG64 raw words."""

    targets = int(target_count)
    members = int(ensemble_size)
    if targets <= 0 or members <= 1:
        raise ValueError("KL seed-bank dimensions are invalid")
    if int(master_seed) != KL_MASTER_SEED:
        raise ValueError("KL master seed differs from protocol")
    generator = np.random.PCG64(int(master_seed))
    values = generator.random_raw(targets * members)
    result = np.asarray(values, dtype=np.uint64).reshape(targets, members)
    if np.unique(result).size != result.size:
        raise RuntimeError("KL seed bank unexpectedly contains duplicate seeds")
    return result


def static_standard_normal_coefficients(
    member_seeds: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    """Regenerate independent standard-normal coefficients from stored seeds."""

    seeds = np.asarray(member_seeds)
    if seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.unsignedinteger):
        raise ValueError("KL member seeds must be one-dimensional unsigned integers")
    resolved = int(rank)
    if resolved <= 0:
        raise ValueError("KL coefficient rank must be positive")
    coefficients = np.empty((seeds.size, resolved), dtype=np.float64)
    for member, seed in enumerate(seeds):
        generator = np.random.Generator(np.random.PCG64(int(seed)))
        coefficients[member] = generator.standard_normal(resolved, dtype=np.float64)
    return coefficients


def reconstruct_static_kl_members(
    *,
    h1_mean: np.ndarray,
    axisymmetric_bias: np.ndarray,
    basis: SnapshotKLBasis,
    rank: int,
    member_seeds: np.ndarray,
) -> np.ndarray:
    """Reconstruct one target's static Gaussian ensemble without truth input."""

    mean = _finite_real("KL H1 target mean", h1_mean)
    if mean.shape != basis.sample_shape:
        raise ValueError("KL H1 target mean shape differs")
    bias = _finite_real("KL axisymmetric bias", axisymmetric_bias)
    expected_bias_shape = basis.sample_shape[:-1]
    if bias.shape != expected_bias_shape:
        raise ValueError("KL axisymmetric bias shape differs")
    resolved = basis.resolved_rank(int(rank))
    if resolved <= 0:
        raise ValueError("static KL ensemble rank must be positive")
    coefficients = static_standard_normal_coefficients(
        member_seeds,
        rank=resolved,
    )
    scaled = coefficients * np.sqrt(basis.eigenvalues[:resolved])[None]
    modes = basis.modes[:resolved].reshape(resolved, -1)
    anomalies = scaled @ modes
    center = mean + bias[..., None]
    return np.asarray(
        center[None] + anomalies.reshape(len(member_seeds), *basis.sample_shape),
        dtype=np.float64,
    )


def classify_kl_outcome(
    *,
    minimum_passing_rank: int | None,
    full_positive_rank: int,
    tier_b_useful: bool,
    numerical_failure: bool = False,
) -> str:
    """Apply the frozen K1--K4 classification without nearest-label guessing."""

    if numerical_failure:
        return "execution_failed_without_scientific_outcome"
    full_rank = int(full_positive_rank)
    if full_rank < 1:
        raise ValueError("full positive KL rank must be positive")
    if minimum_passing_rank is None:
        return "K4_training_residual_span_does_not_transfer"
    rank = int(minimum_passing_rank)
    if rank < 1 or rank > full_rank:
        raise ValueError("minimum passing KL rank is invalid")
    if rank <= 64:
        return (
            "K1_compact_representation_static_covariance_useful"
            if bool(tier_b_useful)
            else "K2_compact_representation_conditional_coefficients_required"
        )
    if rank >= 128:
        return "K3_only_moderate_or_high_rank_adequate"
    return "inconsistent_diagnostic_requires_review"
