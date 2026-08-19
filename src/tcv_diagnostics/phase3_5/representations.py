"""Matched-budget diagnostic residual representations for Phase 3.5.

These are analysis oracles, not trainable production models.  Every allocator
uses source-block eigenvalues only; target arrays are accepted only by the
``reconstruct`` methods after the representation has been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .statistics import PCABasis, fit_pca, project_pca


def _samples(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 5 or array.shape[0] < 2:
        raise ValueError("representations require [sample,field,x,y,z]")
    if not np.all(np.isfinite(array)):
        raise ValueError("representation input contains non-finite values")
    return array


def centered_variance_capture(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    *,
    source_mean: np.ndarray,
) -> float:
    observed = np.asarray(truth, dtype=np.float64)
    candidate = np.asarray(reconstruction, dtype=np.float64)
    mean = np.asarray(source_mean, dtype=np.float64)
    if observed.shape != candidate.shape or mean.shape != observed.shape[1:]:
        raise ValueError("centered variance-capture shapes differ")
    denominator = float(np.sum((observed - mean) ** 2))
    return 1.0 - float(np.sum((observed - candidate) ** 2)) / denominator


@dataclass(frozen=True)
class GlobalPCARepresentation:
    basis: PCABasis
    sample_shape: tuple[int, ...]

    @classmethod
    def fit(cls, source: np.ndarray, *, maximum_rank: int | None = None) -> "GlobalPCARepresentation":
        values = _samples(source)
        basis = fit_pca(values.reshape(values.shape[0], -1), maximum_rank=maximum_rank)
        return cls(basis=basis, sample_shape=tuple(values.shape[1:]))

    def reconstruct(self, values: np.ndarray, *, budget: int) -> np.ndarray:
        array = _samples(values)
        if tuple(array.shape[1:]) != self.sample_shape:
            raise ValueError("global PCA sample shape differs")
        projected = project_pca(array.reshape(array.shape[0], -1), self.basis, rank=min(int(budget), self.basis.rank))
        return projected.reshape(array.shape)

    def accounting(self, budget: int) -> dict[str, int]:
        rank = min(int(budget), self.basis.rank)
        return {
            "real_coefficients": rank,
            "learned_basis_float_equivalents": rank * int(np.prod(self.sample_shape)),
            "fixed_transform_float_equivalents": 0,
            "index_integers": 0,
        }

    @property
    def mean(self) -> np.ndarray:
        return self.basis.mean.reshape(self.sample_shape)


@dataclass(frozen=True)
class _ComplexBasis:
    mean: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray
    real_cost: int


def _fit_complex_pca(values: np.ndarray, *, real_cost: int) -> _ComplexBasis:
    samples = np.asarray(values)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("complex PCA requires [sample,feature]")
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    gram = centered @ np.conjugate(centered.T) / (samples.shape[0] - 1)
    gram = 0.5 * (gram + np.conjugate(gram.T))
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = np.asarray(eigenvalues[order].real, dtype=np.float64)
    eigenvectors = eigenvectors[:, order]
    if eigenvalues.size == 0 or eigenvalues[0] <= 0.0:
        return _ComplexBasis(mean, np.empty((0, samples.shape[1]), dtype=samples.dtype), np.empty(0), real_cost)
    keep = eigenvalues > 1e-10 * eigenvalues[0]
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    modes = (np.conjugate(eigenvectors.T) @ centered) / np.sqrt(
        (samples.shape[0] - 1) * eigenvalues[:, None]
    )
    return _ComplexBasis(mean, modes, eigenvalues, int(real_cost))


def _allocate_directions(
    components: Sequence[_ComplexBasis],
    budget: int,
) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[float, int, int, int]] = []
    for component_index, component in enumerate(components):
        for mode_index, eigenvalue in enumerate(component.eigenvalues):
            candidates.append(
                (float(eigenvalue) / component.real_cost, component_index, mode_index, component.real_cost)
            )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used = 0
    selected: list[tuple[int, int]] = []
    for _, component, mode, cost in candidates:
        if used + cost <= int(budget):
            selected.append((component, mode))
            used += cost
    return tuple(selected)


@dataclass(frozen=True)
class FourierSeparatedRepresentation:
    components: tuple[_ComplexBasis, ...]
    sample_shape: tuple[int, ...]

    @classmethod
    def fit(cls, source: np.ndarray) -> "FourierSeparatedRepresentation":
        values = _samples(source)
        transformed = np.fft.rfft(values, axis=-1, norm="ortho")
        components: list[_ComplexBasis] = []
        nyquist = values.shape[-1] // 2
        for k in range(transformed.shape[-1]):
            real_only = k == 0 or (values.shape[-1] % 2 == 0 and k == nyquist)
            matrix = transformed[..., k].reshape(values.shape[0], -1)
            if real_only:
                matrix = matrix.real
            components.append(_fit_complex_pca(matrix, real_cost=1 if real_only else 2))
        return cls(tuple(components), tuple(values.shape[1:]))

    def reconstruct(self, values: np.ndarray, *, budget: int) -> np.ndarray:
        array = _samples(values)
        if tuple(array.shape[1:]) != self.sample_shape:
            raise ValueError("Fourier representation sample shape differs")
        transformed = np.fft.rfft(array, axis=-1, norm="ortho")
        output = np.empty_like(transformed, dtype=np.complex128)
        selected = _allocate_directions(self.components, int(budget))
        by_component: dict[int, list[int]] = {}
        for component, mode in selected:
            by_component.setdefault(component, []).append(mode)
        for k, component in enumerate(self.components):
            matrix = transformed[..., k].reshape(array.shape[0], -1)
            if component.real_cost == 1:
                matrix = matrix.real
            centered = matrix - component.mean
            reconstruction = np.broadcast_to(component.mean, matrix.shape).astype(
                np.result_type(component.mean, np.complex128), copy=True
            )
            modes = by_component.get(k, ())
            if modes:
                basis = component.modes[np.asarray(modes, dtype=np.int64)]
                reconstruction += (centered @ np.conjugate(basis.T)) @ basis
            output[..., k] = reconstruction.reshape(array.shape[:-1])
        return np.fft.irfft(output, n=array.shape[-1], axis=-1, norm="ortho").real

    def accounting(self, budget: int) -> dict[str, int]:
        selected = _allocate_directions(self.components, int(budget))
        coefficients = sum(self.components[index].real_cost for index, _ in selected)
        storage = sum(
            self.components[index].real_cost * self.components[index].modes.shape[1]
            for index, _ in selected
        )
        return {
            "real_coefficients": int(coefficients),
            "learned_basis_float_equivalents": int(storage),
            "fixed_transform_float_equivalents": 0,
            "index_integers": 2 * len(selected),
        }

    @property
    def mean(self) -> np.ndarray:
        spectrum = np.stack(
            [component.mean.reshape(self.sample_shape[:-1]) for component in self.components],
            axis=-1,
        )
        return np.fft.irfft(spectrum, n=self.sample_shape[-1], axis=-1, norm="ortho").real


def _haar_axis(values: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    if array.shape[axis] % 2:
        raise ValueError("Haar axis length must be even")
    even = np.take(array, np.arange(0, array.shape[axis], 2), axis=axis)
    odd = np.take(array, np.arange(1, array.shape[axis], 2), axis=axis)
    return (even + odd) / math.sqrt(2.0), (even - odd) / math.sqrt(2.0)


def _inverse_haar_axis(low: np.ndarray, high: np.ndarray, axis: int) -> np.ndarray:
    if low.shape != high.shape:
        raise ValueError("inverse Haar halves differ")
    shape = list(low.shape)
    shape[axis] *= 2
    output = np.empty(shape, dtype=np.result_type(low, high))
    even_index = [slice(None)] * output.ndim
    odd_index = [slice(None)] * output.ndim
    even_index[axis] = slice(0, None, 2)
    odd_index[axis] = slice(1, None, 2)
    output[tuple(even_index)] = (low + high) / math.sqrt(2.0)
    output[tuple(odd_index)] = (low - high) / math.sqrt(2.0)
    return output


def _haar_split3(values: np.ndarray) -> dict[str, np.ndarray]:
    xlow, xhigh = _haar_axis(values, -3)
    result: dict[str, np.ndarray] = {}
    for xname, xpart in (("L", xlow), ("H", xhigh)):
        ylow, yhigh = _haar_axis(xpart, -2)
        for yname, ypart in (("L", ylow), ("H", yhigh)):
            zlow, zhigh = _haar_axis(ypart, -1)
            result[xname + yname + "L"] = zlow
            result[xname + yname + "H"] = zhigh
    return result


def _haar_merge3(parts: Mapping[str, np.ndarray]) -> np.ndarray:
    xy: dict[str, np.ndarray] = {}
    for xname in ("L", "H"):
        for yname in ("L", "H"):
            xy[xname + yname] = _inverse_haar_axis(
                parts[xname + yname + "L"], parts[xname + yname + "H"], -1
            )
    xparts: dict[str, np.ndarray] = {}
    for xname in ("L", "H"):
        xparts[xname] = _inverse_haar_axis(xy[xname + "L"], xy[xname + "H"], -2)
    return _inverse_haar_axis(xparts["L"], xparts["H"], -3)


def haar_decompose(values: np.ndarray, *, levels: int = 3) -> dict[str, np.ndarray]:
    current = np.asarray(values)
    output: dict[str, np.ndarray] = {}
    for level in range(1, int(levels) + 1):
        parts = _haar_split3(current)
        for band, part in parts.items():
            if band != "LLL":
                output[f"L{level}_{band}"] = part
        current = parts["LLL"]
    output[f"L{int(levels)}_LLL"] = current
    return output


def haar_reconstruct(parts: Mapping[str, np.ndarray], *, levels: int = 3) -> np.ndarray:
    current = np.asarray(parts[f"L{int(levels)}_LLL"])
    for level in range(int(levels), 0, -1):
        level_parts = {"LLL": current}
        for band in ("LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"):
            level_parts[band] = np.asarray(parts[f"L{level}_{band}"])
        current = _haar_merge3(level_parts)
    return current


@dataclass(frozen=True)
class HaarSubbandRepresentation:
    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    components: tuple[_ComplexBasis, ...]
    sample_shape: tuple[int, ...]
    levels: int

    @classmethod
    def fit(cls, source: np.ndarray, *, levels: int = 3) -> "HaarSubbandRepresentation":
        values = _samples(source)
        transformed = haar_decompose(values, levels=levels)
        keys = tuple(sorted(transformed))
        components = tuple(
            _fit_complex_pca(transformed[key].reshape(values.shape[0], -1), real_cost=1)
            for key in keys
        )
        shapes = tuple(tuple(transformed[key].shape[1:]) for key in keys)
        return cls(keys, shapes, components, tuple(values.shape[1:]), int(levels))

    def reconstruct(self, values: np.ndarray, *, budget: int) -> np.ndarray:
        array = _samples(values)
        transformed = haar_decompose(array, levels=self.levels)
        selected = _allocate_directions(self.components, int(budget))
        by_component: dict[int, list[int]] = {}
        for component, mode in selected:
            by_component.setdefault(component, []).append(mode)
        parts: dict[str, np.ndarray] = {}
        for index, key in enumerate(self.keys):
            component = self.components[index]
            matrix = transformed[key].reshape(array.shape[0], -1)
            centered = matrix - component.mean
            reconstruction = np.broadcast_to(component.mean, matrix.shape).copy()
            modes = by_component.get(index, ())
            if modes:
                basis = component.modes[np.asarray(modes, dtype=np.int64)]
                reconstruction += (centered @ basis.T) @ basis
            parts[key] = reconstruction.reshape((array.shape[0], *self.shapes[index]))
        return haar_reconstruct(parts, levels=self.levels)

    def accounting(self, budget: int) -> dict[str, int]:
        selected = _allocate_directions(self.components, int(budget))
        storage = sum(self.components[index].modes.shape[1] for index, _ in selected)
        return {
            "real_coefficients": len(selected),
            "learned_basis_float_equivalents": int(storage),
            "fixed_transform_float_equivalents": 0,
            "index_integers": 2 * len(selected),
        }

    @property
    def mean(self) -> np.ndarray:
        parts = {
            key: component.mean.reshape(shape)
            for key, shape, component in zip(self.keys, self.shapes, self.components)
        }
        return haar_reconstruct(parts, levels=self.levels)


def patch_starts(length: int, patch: int, stride: int) -> tuple[int, ...]:
    if patch <= 0 or stride <= 0 or patch > length:
        raise ValueError("invalid nonperiodic patch geometry")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return tuple(starts)


@dataclass(frozen=True)
class PatchwisePCARepresentation:
    patches: tuple[tuple[int, int], ...]
    components: tuple[_ComplexBasis, ...]
    sample_shape: tuple[int, ...]
    patch_shape: tuple[int, int, int]
    stride: tuple[int, int, int]

    @classmethod
    def fit(
        cls,
        source: np.ndarray,
        *,
        patch_shape: tuple[int, int, int] = (16, 8, 88),
        stride: tuple[int, int, int] = (8, 4, 88),
    ) -> "PatchwisePCARepresentation":
        values = _samples(source)
        px, py, pz = patch_shape
        sx, sy, sz = stride
        if pz != values.shape[-1] or sz != values.shape[-1]:
            raise ValueError("Phase 3.5 patches must retain the complete z axis")
        patches = tuple(
            (x, y)
            for x in patch_starts(values.shape[-3], px, sx)
            for y in patch_starts(values.shape[-2], py, sy)
        )
        components = tuple(
            _fit_complex_pca(
                values[:, :, x : x + px, y : y + py, :].reshape(values.shape[0], -1),
                real_cost=1,
            )
            for x, y in patches
        )
        return cls(patches, components, tuple(values.shape[1:]), patch_shape, stride)

    def reconstruct(self, values: np.ndarray, *, budget: int) -> np.ndarray:
        array = _samples(values)
        if tuple(array.shape[1:]) != self.sample_shape:
            raise ValueError("patchwise representation sample shape differs")
        selected = _allocate_directions(self.components, int(budget))
        by_component: dict[int, list[int]] = {}
        for component, mode in selected:
            by_component.setdefault(component, []).append(mode)
        output = np.zeros_like(array, dtype=np.float64)
        weights = np.zeros(self.sample_shape[-3:], dtype=np.float64)
        px, py, pz = self.patch_shape
        for index, ((x, y), component) in enumerate(zip(self.patches, self.components)):
            matrix = array[:, :, x : x + px, y : y + py, :].reshape(array.shape[0], -1)
            centered = matrix - component.mean
            reconstruction = np.broadcast_to(component.mean, matrix.shape).copy()
            modes = by_component.get(index, ())
            if modes:
                basis = component.modes[np.asarray(modes, dtype=np.int64)]
                reconstruction += (centered @ basis.T) @ basis
            patch = reconstruction.reshape(array.shape[0], self.sample_shape[0], px, py, pz)
            output[:, :, x : x + px, y : y + py, :] += patch
            weights[x : x + px, y : y + py, :] += 1.0
        if np.any(weights <= 0.0):
            raise RuntimeError("nonperiodic patch blending left uncovered cells")
        return output / weights[None, None, ...]

    def accounting(self, budget: int) -> dict[str, int]:
        selected = _allocate_directions(self.components, int(budget))
        storage = sum(self.components[index].modes.shape[1] for index, _ in selected)
        return {
            "real_coefficients": len(selected),
            "learned_basis_float_equivalents": int(storage),
            "fixed_transform_float_equivalents": int(np.prod(self.sample_shape[-3:])),
            "index_integers": 3 * len(selected),
        }

    @property
    def mean(self) -> np.ndarray:
        dummy = np.zeros((2, *self.sample_shape), dtype=np.float64)
        return self.reconstruct(dummy, budget=0)[0]


def assert_storage_not_above_global(
    accounting: Mapping[str, int],
    *,
    global_budget: int,
    sample_shape: Sequence[int],
) -> None:
    maximum = int(global_budget) * int(np.prod(tuple(sample_shape)))
    if int(accounting["learned_basis_float_equivalents"]) > maximum:
        raise ValueError("representation exceeds matched global-PCA basis storage")
