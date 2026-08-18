"""Frozen uncompressed one-step reference forecasts for Phase 2 O2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _context_array(context: np.ndarray) -> np.ndarray:
    values = np.asarray(context)
    if values.ndim < 3:
        raise ValueError("context must be [history,channel,spatial...]")
    if values.shape[0] not in (1, 2):
        raise ValueError("reference context must contain one or two frames")
    if not np.issubdtype(values.dtype, np.floating) or not np.all(np.isfinite(values)):
        raise ValueError("reference context must be finite and real")
    return values


def persistence(context: np.ndarray) -> np.ndarray:
    """Copy the latest uncompressed standardized state."""

    values = _context_array(context)
    return np.array(values[-1], copy=True)


def two_frame_linear_extrapolation(context: np.ndarray) -> np.ndarray:
    """Return ``2*x_t-x_(t-1)`` for the frozen C5P-H2 reference."""

    values = _context_array(context)
    if values.shape[0] != 2:
        raise ValueError("linear extrapolation requires exactly two frames")
    return np.asarray(2.0 * values[-1] - values[-2], dtype=values.dtype)


def _field(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 4:
        raise ValueError(f"{label} must be [channel,x,y,z]")
    if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite and real")
    return array


@dataclass(frozen=True)
class SpectralAR1:
    """One complex multiplier per state channel and stored toroidal mode."""

    coefficient: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    pair_count: int
    spatial_sample_count_per_pair: int
    toroidal_size: int
    relative_ridge: float

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.coefficient)
        numerator = np.asarray(self.numerator)
        denominator = np.asarray(self.denominator)
        if coefficient.ndim != 2 or not np.iscomplexobj(coefficient):
            raise ValueError("AR coefficients must be complex [channel,k]")
        if numerator.shape != coefficient.shape or not np.iscomplexobj(numerator):
            raise ValueError("AR numerators must match complex coefficients")
        if denominator.shape != coefficient.shape or np.iscomplexobj(denominator):
            raise ValueError("AR denominators must be real and match coefficients")
        if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(numerator)):
            raise ValueError("AR complex statistics must be finite")
        if not np.all(np.isfinite(denominator)) or np.any(denominator < 0):
            raise ValueError("AR denominator must be finite and nonnegative")
        if self.pair_count <= 0 or self.spatial_sample_count_per_pair <= 0:
            raise ValueError("AR fit counts must be positive")
        if self.toroidal_size <= 0 or coefficient.shape[1] != self.toroidal_size // 2 + 1:
            raise ValueError("AR mode count differs from toroidal size")
        if self.relative_ridge < 0 or not np.isfinite(self.relative_ridge):
            raise ValueError("AR relative ridge must be finite and nonnegative")

    def predict(self, latest: np.ndarray) -> np.ndarray:
        source = _field(latest, label="AR source")
        if source.shape[0] != self.coefficient.shape[0]:
            raise ValueError("AR source channel count differs")
        if source.shape[-1] != self.toroidal_size:
            raise ValueError("AR source toroidal size differs")
        spectrum = np.fft.rfft(source.astype(np.float64, copy=False), axis=-1)
        forecast = np.fft.irfft(
            spectrum * self.coefficient[:, None, None, :],
            n=self.toroidal_size,
            axis=-1,
        )
        return np.asarray(forecast, dtype=source.dtype)

    @staticmethod
    def _complex_record(values: np.ndarray) -> dict[str, Any]:
        return {
            "real": np.asarray(values.real, dtype=np.float64).tolist(),
            "imaginary": np.asarray(values.imag, dtype=np.float64).tolist(),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "training_only_toroidal_spectral_AR1",
            "coefficient": self._complex_record(self.coefficient),
            "sufficient_statistics": {
                "numerator": self._complex_record(self.numerator),
                "denominator": np.asarray(self.denominator, dtype=np.float64).tolist(),
                "pair_count": int(self.pair_count),
                "spatial_sample_count_per_pair": int(
                    self.spatial_sample_count_per_pair
                ),
            },
            "toroidal_size": int(self.toroidal_size),
            "relative_ridge": float(self.relative_ridge),
            "pooled_axes": ["time", "x", "y"],
            "one_complex_coefficient_per_channel_and_k": True,
            "validation_tuning_used": False,
        }


def fit_spectral_ar1(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    relative_ridge: float = 1.0e-8,
) -> SpectralAR1:
    """Fit the frozen training-only toroidal AR(1) sufficient statistics.

    The relative ridge is interpreted literally per channel/mode as
    ``lambda = relative_ridge * sum(|x_k|^2)``.  Thus every nonzero-power
    coefficient divides by ``(1 + relative_ridge) * sum(|x_k|^2)``.
    Zero-power modes receive coefficient zero.
    """

    if relative_ridge < 0 or not np.isfinite(relative_ridge):
        raise ValueError("relative ridge must be finite and nonnegative")
    numerator: np.ndarray | None = None
    denominator: np.ndarray | None = None
    shape: tuple[int, ...] | None = None
    pair_count = 0
    for source_values, target_values in pairs:
        source = _field(source_values, label="AR fit source")
        target = _field(target_values, label="AR fit target")
        if target.shape != source.shape:
            raise ValueError("AR source and target shapes differ")
        if shape is None:
            shape = tuple(source.shape)
            modes = shape[-1] // 2 + 1
            numerator = np.zeros((shape[0], modes), dtype=np.complex128)
            denominator = np.zeros((shape[0], modes), dtype=np.float64)
        elif tuple(source.shape) != shape:
            raise ValueError("AR training pair shapes are inconsistent")
        source_spectrum = np.fft.rfft(source.astype(np.float64, copy=False), axis=-1)
        target_spectrum = np.fft.rfft(target.astype(np.float64, copy=False), axis=-1)
        if numerator is None or denominator is None:
            raise AssertionError("AR accumulators were not initialized")
        numerator += np.sum(
            target_spectrum * np.conjugate(source_spectrum),
            axis=(1, 2),
            dtype=np.complex128,
        )
        denominator += np.sum(
            np.square(np.abs(source_spectrum)),
            axis=(1, 2),
            dtype=np.float64,
        )
        pair_count += 1
    if shape is None or numerator is None or denominator is None or pair_count == 0:
        raise ValueError("cannot fit spectral AR(1) without training pairs")
    coefficient = np.zeros_like(numerator)
    positive = denominator > 0.0
    coefficient[positive] = numerator[positive] / (
        (1.0 + float(relative_ridge)) * denominator[positive]
    )
    return SpectralAR1(
        coefficient=coefficient,
        numerator=numerator,
        denominator=denominator,
        pair_count=pair_count,
        spatial_sample_count_per_pair=shape[1] * shape[2],
        toroidal_size=shape[-1],
        relative_ridge=float(relative_ridge),
    )
