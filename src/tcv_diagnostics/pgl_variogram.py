"""Fair variogram scores and differentiable PGL forecast sampling.

This module implements the bounded objective-only screen frozen in
``POST_ECRD_OLD_85604_PGL_BOUNDED_VARIOGRAM_FINETUNE_AMENDMENT_2026-08-28.md``.
It is data agnostic: it contains no shot paths, split discovery, or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .models.persistent_global_local import PersistentGlobalLocalEDM


PGL_VARIOGRAM_MEMBER_COUNT = 4
PGL_VARIOGRAM_DISTANCE_BINS = 6
PGL_VARIOGRAM_PAIRS_PER_BIN = 1024
PGL_VARIOGRAM_PAIR_SEED = 856_040_828
PGL_VARIOGRAM_TEMPORAL_LAGS = (1, 2, 3, 4)


@dataclass(frozen=True)
class IndexedPairBank:
    """Weighted pairs indexing one flattened trajectory."""

    left: np.ndarray
    right: np.ndarray
    weight: np.ndarray
    group: np.ndarray
    group_name: str
    group_values: tuple[float, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        left = np.asarray(self.left)
        right = np.asarray(self.right)
        weight = np.asarray(self.weight)
        group = np.asarray(self.group)
        if any(value.ndim != 1 for value in (left, right, weight, group)):
            raise ValueError("variogram pair arrays must be one-dimensional")
        if not (left.size == right.size == weight.size == group.size) or left.size == 0:
            raise ValueError("variogram pair arrays must have one nonempty size")
        if not (
            np.issubdtype(left.dtype, np.integer)
            and np.issubdtype(right.dtype, np.integer)
            and np.issubdtype(group.dtype, np.integer)
        ):
            raise TypeError("variogram pair indices and groups must be integral")
        if np.any(left < 0) or np.any(right < 0) or np.any(left == right):
            raise ValueError("variogram pair endpoints must be distinct/nonnegative")
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError("variogram pair weights must be finite and positive")
        if not math.isclose(float(np.sum(weight)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("variogram pair weights must sum to one")
        group_count = len(self.group_values)
        if group_count == 0 or np.any(group < 0) or np.any(group >= group_count):
            raise ValueError("variogram pair group index differs")
        totals = np.bincount(group, weights=weight, minlength=group_count)
        if not np.allclose(totals, np.full(group_count, 1.0 / group_count), atol=1e-12):
            raise ValueError("variogram groups must receive equal total weight")
        if not self.group_name:
            raise ValueError("variogram group name must be nonempty")

    @property
    def count(self) -> int:
        return int(np.asarray(self.left).size)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        for values, dtype in (
            (self.left, "<i8"),
            (self.right, "<i8"),
            (self.weight, "<f8"),
            (self.group, "<i8"),
        ):
            digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
        payload = {
            "group_name": self.group_name,
            "group_values": list(self.group_values),
            "metadata": dict(self.metadata),
        }
        digest.update(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
        return digest.hexdigest()

    def to_record(self) -> dict[str, Any]:
        groups = np.asarray(self.group, dtype=np.int64)
        weights = np.asarray(self.weight, dtype=np.float64)
        return {
            "count": self.count,
            "sha256": self.sha256,
            "group_name": self.group_name,
            "group_values": list(self.group_values),
            "group_counts": np.bincount(
                groups, minlength=len(self.group_values)
            ).astype(int).tolist(),
            "group_total_weights": np.bincount(
                groups, weights=weights, minlength=len(self.group_values)
            ).tolist(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FairVariogramResult:
    fair: Tensor
    ordinary: Tensor
    finite_member_correction: Tensor
    fair_by_group: Tensor
    ordinary_by_group: Tensor


def _pair_tensors(bank: IndexedPairBank, reference: Tensor) -> tuple[Tensor, ...]:
    device = reference.device
    return (
        torch.as_tensor(bank.left, device=device, dtype=torch.long),
        torch.as_tensor(bank.right, device=device, dtype=torch.long),
        torch.as_tensor(bank.weight, device=device, dtype=torch.float32),
        torch.as_tensor(bank.group, device=device, dtype=torch.long),
    )


def fair_variogram_score(
    members: Tensor,
    truth: Tensor,
    bank: IndexedPairBank,
) -> FairVariogramResult:
    """Return the fair order-one empirical variogram score.

    ``members`` has axes ``[batch,member,...]`` and ``truth`` has
    ``[batch,...]``. Every axis after member is flattened and indexed by the
    immutable pair bank. The estimator subtracts the exact finite-member
    ``sample_variance/M`` term and is therefore permitted to be negative.
    """

    if members.ndim < 3 or truth.ndim != members.ndim - 1:
        raise ValueError("variogram inputs require batch/member/sample axes")
    if members.shape[0] != truth.shape[0] or members.shape[2:] != truth.shape[1:]:
        raise ValueError("variogram member and truth shapes differ")
    member_count = int(members.shape[1])
    if member_count < 2:
        raise ValueError("fair variogram needs at least two ensemble members")
    if not torch.isfinite(members).all() or not torch.isfinite(truth).all():
        raise ValueError("variogram inputs must be finite")
    flat_members = members.float().flatten(2)
    flat_truth = truth.float().flatten(1)
    if max(int(np.max(bank.left)), int(np.max(bank.right))) >= flat_truth.shape[1]:
        raise ValueError("variogram pair bank exceeds flattened state")
    left, right, weight, group = _pair_tensors(bank, flat_truth)
    member_increment = torch.abs(
        flat_members.index_select(2, left) - flat_members.index_select(2, right)
    )
    truth_increment = torch.abs(
        flat_truth.index_select(1, left) - flat_truth.index_select(1, right)
    )
    ensemble_increment = member_increment.mean(dim=1)
    ordinary_pair = (truth_increment - ensemble_increment).square()
    correction_pair = member_increment.var(dim=1, unbiased=True) / member_count
    fair_pair = ordinary_pair - correction_pair

    weighted_fair = (fair_pair * weight).sum(dim=1)
    weighted_ordinary = (ordinary_pair * weight).sum(dim=1)
    weighted_correction = (correction_pair * weight).sum(dim=1)
    group_count = len(bank.group_values)
    group_weight = torch.zeros(
        (group_count,), device=members.device, dtype=torch.float32
    ).scatter_add_(0, group, weight)
    fair_group_sum = torch.zeros(
        (members.shape[0], group_count), device=members.device, dtype=torch.float32
    ).scatter_add_(1, group[None].expand(members.shape[0], -1), fair_pair * weight)
    ordinary_group_sum = torch.zeros_like(fair_group_sum).scatter_add_(
        1, group[None].expand(members.shape[0], -1), ordinary_pair * weight
    )
    fair_by_group = fair_group_sum / group_weight[None]
    ordinary_by_group = ordinary_group_sum / group_weight[None]
    return FairVariogramResult(
        fair=weighted_fair.mean(),
        ordinary=weighted_ordinary.mean(),
        finite_member_correction=weighted_correction.mean(),
        fair_by_group=fair_by_group.mean(dim=0),
        ordinary_by_group=ordinary_by_group.mean(dim=0),
    )


def minimum_wedge_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    wedge_angle: float = 2.0 * math.pi / 5.0,
) -> np.ndarray:
    """Cylindrical 3-D distance with minimum-image toroidal separation.

    Position arrays have final coordinates ``[R,Z,phi]``.
    """

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 1 or a.shape[-1] != 3:
        raise ValueError("physical positions must share shape [...,3]")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("physical positions must be finite")
    delta = np.mod(np.abs(a[..., 2] - b[..., 2]), wedge_angle)
    delta = np.minimum(delta, wedge_angle - delta)
    squared = (
        a[..., 0] ** 2
        + b[..., 0] ** 2
        - 2.0 * a[..., 0] * b[..., 0] * np.cos(delta)
        + (a[..., 1] - b[..., 1]) ** 2
    )
    return np.sqrt(np.maximum(squared, 0.0))


def _candidate_pairs(
    eligible: np.ndarray,
    *,
    generator: np.random.Generator,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(eligible, dtype=np.int64)
    if values.ndim != 1 or values.size < 2 or np.unique(values).size != values.size:
        raise ValueError("eligible cell indices must be unique and nontrivial")
    left = generator.choice(values, size=int(candidate_count), replace=True)
    right = generator.choice(values, size=int(candidate_count), replace=True)
    same = left == right
    while np.any(same):
        right[same] = generator.choice(values, size=int(np.sum(same)), replace=True)
        same = left == right
    return left, right


def spatial_base_pairs(
    positions: np.ndarray,
    eligible: np.ndarray,
    *,
    bins: int = PGL_VARIOGRAM_DISTANCE_BINS,
    pairs_per_bin: int = PGL_VARIOGRAM_PAIRS_PER_BIN,
    seed: int = PGL_VARIOGRAM_PAIR_SEED,
    candidate_count: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic geometry-only pairs balanced by distance quantile."""

    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape [cell,3]")
    eligible_values = np.asarray(eligible, dtype=np.int64)
    if np.any(eligible_values < 0) or np.any(eligible_values >= coordinates.shape[0]):
        raise ValueError("eligible position index differs")
    if bins <= 1 or pairs_per_bin <= 0 or candidate_count < bins * pairs_per_bin:
        raise ValueError("spatial pair-bank budget is invalid")
    generator = np.random.default_rng(int(seed))
    candidate_left, candidate_right = _candidate_pairs(
        eligible_values, generator=generator, candidate_count=candidate_count
    )
    distances = minimum_wedge_distance(
        coordinates[candidate_left], coordinates[candidate_right]
    )
    edges = np.quantile(
        distances, np.linspace(0.0, 1.0, bins + 1), method="linear"
    )
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("physical-distance quantile edges are degenerate")
    labels = np.searchsorted(edges[1:-1], distances, side="right")
    selected_left: list[np.ndarray] = []
    selected_right: list[np.ndarray] = []
    selected_group: list[np.ndarray] = []
    for group in range(bins):
        available = np.flatnonzero(labels == group)
        if available.size < pairs_per_bin:
            raise ValueError("a physical-distance bin lacks frozen pairs")
        chosen = generator.choice(available, size=pairs_per_bin, replace=False)
        selected_left.append(candidate_left[chosen])
        selected_right.append(candidate_right[chosen])
        selected_group.append(np.full(pairs_per_bin, group, dtype=np.int64))
    return (
        np.concatenate(selected_left),
        np.concatenate(selected_right),
        np.concatenate(selected_group),
        np.asarray(edges, dtype=np.float64),
    )


def build_spatial_pair_bank(
    positions: np.ndarray,
    eligible: np.ndarray,
    *,
    future_times: int,
    variables: int,
    bins: int = PGL_VARIOGRAM_DISTANCE_BINS,
    pairs_per_bin: int = PGL_VARIOGRAM_PAIRS_PER_BIN,
    seed: int = PGL_VARIOGRAM_PAIR_SEED,
    label: str = "physical_distance_m",
) -> IndexedPairBank:
    """Replicate one balanced spatial pair table over times and variables."""

    if future_times <= 0 or variables <= 0:
        raise ValueError("spatial pair-bank trajectory dimensions must be positive")
    base_left, base_right, base_group, edges = spatial_base_pairs(
        positions,
        eligible,
        bins=bins,
        pairs_per_bin=pairs_per_bin,
        seed=seed,
    )
    cells = int(np.asarray(positions).shape[0])
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for time in range(future_times):
        for variable in range(variables):
            offset = (time * variables + variable) * cells
            left.append(base_left + offset)
            right.append(base_right + offset)
            groups.append(base_group)
    left_array = np.concatenate(left).astype(np.int64)
    right_array = np.concatenate(right).astype(np.int64)
    group_array = np.concatenate(groups).astype(np.int64)
    counts = np.bincount(group_array, minlength=bins)
    weight = np.concatenate(
        [
            np.full(int(counts[group]), 1.0 / bins / int(counts[group]))
            for group in range(bins)
        ]
    )
    # The concatenated arrays are ordered time/variable/base-bin, not by bin;
    # assign weights through the group lookup rather than relying on ordering.
    weight = np.asarray(
        [1.0 / bins / counts[group] for group in group_array], dtype=np.float64
    )
    return IndexedPairBank(
        left=left_array,
        right=right_array,
        weight=weight,
        group=group_array,
        group_name=label,
        group_values=tuple(float(value) for value in edges[1:]),
        metadata={
            "distance_bin_edges_m": edges.tolist(),
            "distance_bins": bins,
            "pairs_per_bin_per_time_variable": pairs_per_bin,
            "future_times": future_times,
            "variables": variables,
            "seed": seed,
            "minimum_image_wedge_angle_radians": 2.0 * math.pi / 5.0,
        },
    )


def build_temporal_pair_bank(
    eligible: np.ndarray,
    *,
    cells: int,
    trajectory_times: int,
    variables: int,
    lags: Sequence[int] = PGL_VARIOGRAM_TEMPORAL_LAGS,
    pairs_per_time_variable: int = PGL_VARIOGRAM_PAIRS_PER_BIN,
    seed: int = PGL_VARIOGRAM_PAIR_SEED + 1,
    cadence_microseconds: float = 3.131905426352636,
) -> IndexedPairBank:
    """Build same-cell temporal pairs including the observed current state."""

    eligible_values = np.asarray(eligible, dtype=np.int64)
    if (
        eligible_values.ndim != 1
        or eligible_values.size < pairs_per_time_variable
        or np.any(eligible_values < 0)
        or np.any(eligible_values >= cells)
    ):
        raise ValueError("temporal eligible-cell bank differs")
    lag_values = tuple(int(value) for value in lags)
    if (
        not lag_values
        or tuple(sorted(set(lag_values))) != lag_values
        or lag_values[0] <= 0
        or lag_values[-1] >= trajectory_times
        or variables <= 0
    ):
        raise ValueError("temporal variogram lags differ")
    generator = np.random.default_rng(int(seed))
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    group: list[np.ndarray] = []
    for lag_index, lag in enumerate(lag_values):
        for first_time in range(trajectory_times - lag):
            second_time = first_time + lag
            for variable in range(variables):
                selected = generator.choice(
                    eligible_values, size=pairs_per_time_variable, replace=False
                )
                left.append((first_time * variables + variable) * cells + selected)
                right.append((second_time * variables + variable) * cells + selected)
                group.append(
                    np.full(pairs_per_time_variable, lag_index, dtype=np.int64)
                )
    left_array = np.concatenate(left).astype(np.int64)
    right_array = np.concatenate(right).astype(np.int64)
    group_array = np.concatenate(group).astype(np.int64)
    counts = np.bincount(group_array, minlength=len(lag_values))
    weight = np.asarray(
        [1.0 / len(lag_values) / counts[value] for value in group_array],
        dtype=np.float64,
    )
    return IndexedPairBank(
        left=left_array,
        right=right_array,
        weight=weight,
        group=group_array,
        group_name="temporal_lag_microseconds",
        group_values=tuple(float(lag) * cadence_microseconds for lag in lag_values),
        metadata={
            "lags_frames": list(lag_values),
            "cadence_microseconds": float(cadence_microseconds),
            "pairs_per_time_variable": pairs_per_time_variable,
            "trajectory_times_including_current": trajectory_times,
            "variables": variables,
            "seed": seed,
        },
    )


def prepend_observed_current(members: Tensor, truth: Tensor, current: Tensor) -> tuple[Tensor, Tensor]:
    """Prepend an observed current state to member and truth trajectories."""

    if members.ndim != 7 or truth.ndim != 6 or current.ndim != 5:
        raise ValueError("current prepend expects field trajectories")
    if (
        members.shape[0] != truth.shape[0]
        or members.shape[2:] != truth.shape[1:]
        or current.shape[0] != truth.shape[0]
        or current.shape[1:] != truth.shape[2:]
    ):
        raise ValueError("current and future field trajectories differ")
    expanded = current[:, None].expand(current.shape[0], members.shape[1], *current.shape[1:])
    return (
        torch.cat((expanded[:, :, None], members), dim=2),
        torch.cat((current[:, None], truth), dim=1),
    )


def gauge_fix_phi(
    values: Tensor,
    *,
    phi_index: int = 3,
    spatial_mask: Tensor | None = None,
) -> Tensor:
    """Subtract one independent spatial mean from every phi frame/member."""

    if values.ndim not in (6, 7):
        raise ValueError("gauge fixing expects truth or member trajectories")
    field_axis = 2 if values.ndim == 6 else 3
    if not 0 <= int(phi_index) < values.shape[field_axis]:
        raise ValueError("phi channel leaves trajectory")
    result = values.clone()
    phi = result.select(field_axis, int(phi_index))
    if spatial_mask is None:
        mean = phi.mean(dim=(-3, -2, -1), keepdim=True)
    else:
        mask = torch.as_tensor(spatial_mask, device=phi.device, dtype=phi.dtype)
        if mask.shape != phi.shape[-3:]:
            raise ValueError("phi gauge mask differs")
        denominator = mask.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("phi gauge mask is empty")
        mean = (phi * mask).sum(dim=(-3, -2, -1), keepdim=True) / denominator
    phi.sub_(mean)
    return result


def differentiable_sample_normalized(
    model: PersistentGlobalLocalEDM,
    current: Tensor,
    mean: Tensor,
    initial_noise: Tensor,
    *,
    steps: int = 18,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    rho: float = 7.0,
    activation_checkpointing: bool = True,
) -> Tensor:
    """Differentiable clone of the frozen PGL Heun sampler.

    The forward algorithm intentionally matches ``sample_normalized``. Only
    autograd/activation storage differs.
    """

    if initial_noise.ndim != 7 or initial_noise.shape[0] != current.shape[0]:
        raise ValueError("initial noise must have [B,M,K,C,x,y,z]")
    if initial_noise.shape[2:] != mean.shape[1:]:
        raise ValueError("initial noise and mean trajectory differ")
    if not torch.isfinite(initial_noise).all():
        raise ValueError("initial noise must be finite")
    batch, members = initial_noise.shape[:2]
    expanded_current = current[:, None].expand(
        batch, members, *current.shape[1:]
    ).reshape(batch * members, *current.shape[1:]).contiguous()
    expanded_mean = mean[:, None].expand(
        batch, members, *mean.shape[1:]
    ).reshape(batch * members, *mean.shape[1:]).contiguous()
    sample = initial_noise.reshape(batch * members, *initial_noise.shape[2:])
    schedule = model.sampling_schedule(
        steps=steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
        device=sample.device,
        dtype=sample.dtype,
    )
    sample = sample * schedule[0]

    def denoise(value: Tensor, sigma_value: Tensor) -> Tensor:
        sigma_batch = sigma_value.expand(value.shape[0])
        if not activation_checkpointing:
            return model.denoise(value, expanded_current, expanded_mean, sigma_batch)

        def call(argument: Tensor) -> Tensor:
            return model.denoise(
                argument, expanded_current, expanded_mean, sigma_batch
            )

        return checkpoint(
            call,
            value,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    for index in range(len(schedule) - 1):
        current_sigma = schedule[index]
        next_sigma = schedule[index + 1]
        denoised = denoise(sample, current_sigma)
        derivative = (sample - denoised) / current_sigma
        proposed = sample + (next_sigma - current_sigma) * derivative
        if float(next_sigma) != 0.0:
            next_denoised = denoise(proposed, next_sigma)
            next_derivative = (proposed - next_denoised) / next_sigma
            sample = sample + (next_sigma - current_sigma) * 0.5 * (
                derivative + next_derivative
            )
        else:
            sample = proposed
    return sample.reshape(batch, members, *sample.shape[1:])
