"""Hierarchical proper scores for member-wise separatrix transport.

This module is intentionally data agnostic.  It operates on the already
quadrature-weighted output of :class:`TorchSeparatrixTransport` and implements
the prospectively frozen 2026-09-02 local/regional/global objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor

from .pgl_torch_transport import PGL_TRANSPORT_QUANTITIES
from .pgl_variogram import FairVariogramResult, IndexedPairBank, fair_variogram_score


PGL_HIERARCHICAL_MEMBERS = 4
PGL_HIERARCHICAL_POLOIDAL_GROUPS = 4
PGL_HIERARCHICAL_TOROIDAL_GROUPS = 3
PGL_HIERARCHICAL_REGION_COUNT = 12
PGL_HIERARCHICAL_LOW_K = (1, 2, 3)
PGL_HIERARCHICAL_TRANSPORT_K = (4, 5, 6, 7)
PGL_HIERARCHICAL_LOW_N = tuple(5 * value for value in PGL_HIERARCHICAL_LOW_K)
PGL_HIERARCHICAL_TRANSPORT_N = tuple(
    5 * value for value in PGL_HIERARCHICAL_TRANSPORT_K
)


@dataclass(frozen=True)
class FairScoreResult:
    """Fair finite-ensemble score and the ordinary plug-in estimator."""

    fair: Tensor
    ordinary: Tensor
    finite_member_correction: Tensor


@dataclass(frozen=True)
class HierarchicalTransportScores:
    """Unnormalized hierarchy scores, separately for all four quantities."""

    local_spatial: tuple[Tensor, ...]
    local_temporal: tuple[Tensor, ...]
    regional: tuple[Tensor, ...]
    fourier_low: tuple[Tensor, ...]
    fourier_transport_band: tuple[Tensor, ...]
    global_crps: tuple[Tensor, ...]
    ordinary: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        values = (
            self.local_spatial,
            self.local_temporal,
            self.regional,
            self.fourier_low,
            self.fourier_transport_band,
            self.global_crps,
        )
        if any(len(group) != len(PGL_TRANSPORT_QUANTITIES) for group in values):
            raise ValueError("hierarchical scores require all four transport quantities")


def _validated_members_truth(members: Tensor, truth: Tensor) -> tuple[Tensor, Tensor]:
    if members.ndim < 3 or truth.ndim != members.ndim - 1:
        raise ValueError("ensemble score requires batch/member/sample axes")
    if members.shape[0] != truth.shape[0] or members.shape[2:] != truth.shape[1:]:
        raise ValueError("ensemble members and truth shapes differ")
    if members.shape[1] < 2:
        raise ValueError("fair ensemble score requires at least two members")
    if not torch.isfinite(members).all() or not torch.isfinite(truth).all():
        raise ValueError("ensemble score inputs must be finite")
    return members.float(), truth.float()


def fair_energy_score(members: Tensor, truth: Tensor) -> FairScoreResult:
    """Fair multivariate energy score using RMS Euclidean distance.

    Axes after ``[batch, member]`` form one joint vector.  Division by the
    square root of vector dimension makes the distance an RMS norm; frozen
    initial controls remove the remaining physical scale.
    """

    ensemble, target = _validated_members_truth(members, truth)
    flat_members = ensemble.flatten(2)
    flat_truth = target.flatten(1)
    dimension = int(flat_truth.shape[1])
    if dimension <= 0:
        raise ValueError("energy-score vector is empty")
    scale = math.sqrt(dimension)
    truth_distance = torch.linalg.vector_norm(
        flat_members - flat_truth[:, None], dim=-1
    ) / scale
    first = truth_distance.mean(dim=1)
    member_count = int(flat_members.shape[1])
    unordered: list[Tensor] = []
    for left in range(member_count):
        for right in range(left + 1, member_count):
            unordered.append(
                torch.linalg.vector_norm(
                    flat_members[:, left] - flat_members[:, right], dim=-1
                )
                / scale
            )
    pair_sum = torch.stack(unordered, dim=1).sum(dim=1)
    fair_second = pair_sum / float(member_count * (member_count - 1))
    ordinary_second = pair_sum / float(member_count * member_count)
    fair = first - fair_second
    ordinary = first - ordinary_second
    correction = fair_second - ordinary_second
    return FairScoreResult(
        fair=fair.mean(),
        ordinary=ordinary.mean(),
        finite_member_correction=correction.mean(),
    )


def fair_crps_score(members: Tensor, truth: Tensor) -> FairScoreResult:
    """Fair scalar CRPS averaged equally over all nonensemble coordinates."""

    ensemble, target = _validated_members_truth(members, truth)
    first = torch.abs(ensemble - target[:, None]).mean(dim=1)
    member_count = int(ensemble.shape[1])
    unordered: list[Tensor] = []
    for left in range(member_count):
        for right in range(left + 1, member_count):
            unordered.append(torch.abs(ensemble[:, left] - ensemble[:, right]))
    pair_sum = torch.stack(unordered, dim=1).sum(dim=1)
    fair_second = pair_sum / float(member_count * (member_count - 1))
    ordinary_second = pair_sum / float(member_count * member_count)
    fair = first - fair_second
    ordinary = first - ordinary_second
    correction = fair_second - ordinary_second
    return FairScoreResult(
        fair=fair.mean(),
        ordinary=ordinary.mean(),
        finite_member_correction=correction.mean(),
    )


def regional_transport_sums(local: Tensor) -> Tensor:
    """Sum weighted local transport into a fixed 4-by-3 regional partition."""

    if local.ndim < 2 or local.shape[-2:] != (16, 81):
        raise ValueError("regional transport expects trailing [16,81]")
    reshaped = local.reshape(
        *local.shape[:-2],
        PGL_HIERARCHICAL_POLOIDAL_GROUPS,
        4,
        PGL_HIERARCHICAL_TOROIDAL_GROUPS,
        27,
    )
    regions = reshaped.sum(dim=(-3, -1))
    return regions.reshape(*local.shape[:-2], PGL_HIERARCHICAL_REGION_COUNT)


def global_transport_sum(local: Tensor) -> Tensor:
    """Return the exact separatrix-integrated transport for every leading row."""

    if local.ndim < 2 or local.shape[-2:] != (16, 81):
        raise ValueError("global transport expects trailing [16,81]")
    return local.sum(dim=(-2, -1))


def toroidal_transport_coefficients(local: Tensor) -> Tensor:
    """Return orthonormal rFFT coefficients after summing poloidally."""

    if local.ndim < 2 or local.shape[-2:] != (16, 81):
        raise ValueError("transport Fourier summary expects trailing [16,81]")
    profile = local.sum(dim=-2)
    return torch.fft.rfft(profile.float(), dim=-1, norm="ortho")


def transport_fourier_features(local: Tensor, modes: tuple[int, ...]) -> Tensor:
    """Stack real/imaginary coefficients for one preregistered mode band."""

    if not modes or tuple(sorted(set(modes))) != tuple(modes) or modes[0] <= 0:
        raise ValueError("transport Fourier modes must be positive and ordered")
    coefficients = toroidal_transport_coefficients(local)
    if modes[-1] >= coefficients.shape[-1]:
        raise ValueError("transport Fourier mode exceeds native resolution")
    selected = coefficients.index_select(
        -1, torch.as_tensor(modes, device=local.device, dtype=torch.long)
    )
    return torch.view_as_real(selected)


def global_sum_from_k0(local: Tensor) -> Tensor:
    """Recover the total integral from the orthonormal Fourier zero mode."""

    coefficients = toroidal_transport_coefficients(local)
    return coefficients[..., 0].real * math.sqrt(81.0)


def score_hierarchical_transport(
    *,
    local_members: Tensor,
    local_future_truth: Tensor,
    local_trajectory_members: Tensor,
    local_trajectory_truth: Tensor,
    spatial_bank: IndexedPairBank,
    temporal_bank: IndexedPairBank,
) -> HierarchicalTransportScores:
    """Compute local, regional, Fourier, and global member-wise scores.

    ``local_members`` is ``[B,M,4,Q,16,81]``.  The trajectory inputs prepend
    the observed current state and therefore have five time steps.
    """

    quantity_count = len(PGL_TRANSPORT_QUANTITIES)
    if local_members.ndim != 6 or local_members.shape[2:] != (
        4,
        quantity_count,
        16,
        81,
    ):
        raise ValueError("future transport-member shape differs")
    if local_future_truth.shape != (
        local_members.shape[0],
        4,
        quantity_count,
        16,
        81,
    ):
        raise ValueError("future transport-truth shape differs")
    if local_trajectory_members.shape != (
        local_members.shape[0],
        local_members.shape[1],
        5,
        quantity_count,
        16,
        81,
    ) or local_trajectory_truth.shape != (
        local_members.shape[0],
        5,
        quantity_count,
        16,
        81,
    ):
        raise ValueError("transport trajectory shape differs")

    local_spatial: list[Tensor] = []
    local_temporal: list[Tensor] = []
    regional: list[Tensor] = []
    fourier_low: list[Tensor] = []
    fourier_transport: list[Tensor] = []
    global_crps: list[Tensor] = []
    ordinary: dict[str, Tensor] = {}
    member_regions = regional_transport_sums(local_members)
    truth_regions = regional_transport_sums(local_future_truth)
    member_low = transport_fourier_features(local_members, PGL_HIERARCHICAL_LOW_K)
    truth_low = transport_fourier_features(local_future_truth, PGL_HIERARCHICAL_LOW_K)
    member_band = transport_fourier_features(
        local_members, PGL_HIERARCHICAL_TRANSPORT_K
    )
    truth_band = transport_fourier_features(
        local_future_truth, PGL_HIERARCHICAL_TRANSPORT_K
    )
    member_global = global_transport_sum(local_members)
    truth_global = global_transport_sum(local_future_truth)

    for index, name in enumerate(PGL_TRANSPORT_QUANTITIES):
        spatial = fair_variogram_score(
            local_members[:, :, :, index : index + 1],
            local_future_truth[:, :, index : index + 1],
            spatial_bank,
        )
        temporal = fair_variogram_score(
            local_trajectory_members[:, :, :, index : index + 1],
            local_trajectory_truth[:, :, index : index + 1],
            temporal_bank,
        )
        region = fair_energy_score(
            member_regions[:, :, :, index], truth_regions[:, :, index]
        )
        low = fair_energy_score(member_low[:, :, :, index], truth_low[:, :, index])
        band = fair_energy_score(
            member_band[:, :, :, index], truth_band[:, :, index]
        )
        total = fair_crps_score(
            member_global[:, :, :, index], truth_global[:, :, index]
        )
        local_spatial.append(spatial.fair)
        local_temporal.append(temporal.fair)
        regional.append(region.fair)
        fourier_low.append(low.fair)
        fourier_transport.append(band.fair)
        global_crps.append(total.fair)
        ordinary[f"local_spatial/{name}"] = spatial.ordinary
        ordinary[f"local_temporal/{name}"] = temporal.ordinary
        ordinary[f"regional/{name}"] = region.ordinary
        ordinary[f"fourier_low/{name}"] = low.ordinary
        ordinary[f"fourier_transport_band/{name}"] = band.ordinary
        ordinary[f"global_crps/{name}"] = total.ordinary
    return HierarchicalTransportScores(
        local_spatial=tuple(local_spatial),
        local_temporal=tuple(local_temporal),
        regional=tuple(regional),
        fourier_low=tuple(fourier_low),
        fourier_transport_band=tuple(fourier_transport),
        global_crps=tuple(global_crps),
        ordinary=ordinary,
    )
