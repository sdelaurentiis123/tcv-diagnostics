"""Held-in 85604 validation summaries for the trained transport hierarchy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .b2_scoring import decode_b2_member_forecasts
from .b2_probabilistic_metrics import corrected_spread_skill_summary
from .b2_transport_metrics import memberwise_transport_outputs
from .codec_transport import (
    TRANSPORT_QUANTITIES,
    CodecTransportGeometry,
    direct_pressure_transport_state,
    evaluate_transport_state,
)
from .ecrd_scoring import exact_local_transport_from_b2_outputs
from .matched_o1_transport import NativeTruthCatalog
from .model_training_data import ModelDatasetCatalog
from .persistent_global_local_forecast import PGL_EVALUATION_STARTS, PGLForecastArtifact
from .pgl_hierarchical_transport import (
    PGL_HIERARCHICAL_LOW_K,
    PGL_HIERARCHICAL_TRANSPORT_K,
    global_transport_sum,
    regional_transport_sums,
    score_hierarchical_transport,
    transport_fourier_features,
)
from .pgl_variogram import IndexedPairBank, fair_variogram_score


def _truth_local_transport(
    native: dict[str, np.ndarray], geometry: CodecTransportGeometry
) -> dict[str, np.ndarray]:
    state = direct_pressure_transport_state(
        native["Ne"], native["Pe"], native["Pi"], native["phi"]
    )
    evaluated = evaluate_transport_state(state, geometry)
    strict = np.asarray(geometry.strict_face_mask, dtype=bool)
    separatrix = np.asarray(geometry.separatrix_face_mask, dtype=bool)
    selector = separatrix[strict]
    strict_rows = int(np.sum(strict))
    if int(np.sum(selector)) != 16:
        raise ValueError("hierarchical validation separatrix geometry differs")
    result = {}
    for quantity in TRANSPORT_QUANTITIES:
        flat = np.asarray(
            evaluated[quantity]["strict_face_contributions"][0], dtype=np.float64
        ).reshape(-1)
        result[quantity] = np.ascontiguousarray(
            flat.reshape(strict_rows, 81)[selector], dtype=np.float32
        )
    return result


def collect_hierarchical_local_transport(
    *,
    catalog: ModelDatasetCatalog,
    forecast_artifact: PGLForecastArtifact,
    native_truth: NativeTruthCatalog,
    geometry: CodecTransportGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Collect memberwise current/future local transport on 36 frozen starts."""

    starts = tuple(PGL_EVALUATION_STARTS)
    quantity_count = len(TRANSPORT_QUANTITIES)
    members = np.empty(
        (len(starts), 32, 4, quantity_count, 16, 81), dtype=np.float32
    )
    truth = np.empty((len(starts), 4, quantity_count, 16, 81), dtype=np.float32)
    current = np.empty((len(starts), 1, quantity_count, 16, 81), dtype=np.float32)
    maximum_closure = 0.0
    for position, start in enumerate(starts):
        current_native = native_truth.read(
            start, start + 1, fields=("Ne", "Pe", "Pi", "phi")
        )
        current_local = _truth_local_transport(current_native, geometry)
        for quantity_index, quantity in enumerate(TRANSPORT_QUANTITIES):
            current[position, 0, quantity_index] = current_local[quantity]
        for horizon in range(1, 5):
            target = start + horizon
            standardized = forecast_artifact.read_forecast_horizon(
                position, horizon
            )[:, None]
            physical = decode_b2_member_forecasts(catalog, standardized)
            target_native = native_truth.read(
                target, target + 1, fields=("Ne", "Pe", "Pi", "phi")
            )
            forecast_outputs, truth_outputs = memberwise_transport_outputs(
                physical_forecast_model88=physical,
                native_truth=target_native,
                geometry=geometry,
            )
            local_members, local_truth, closure = exact_local_transport_from_b2_outputs(
                forecast_outputs=forecast_outputs,
                truth_outputs=truth_outputs,
                geometry=geometry,
            )
            maximum_closure = max(maximum_closure, float(closure))
            for quantity_index, quantity in enumerate(TRANSPORT_QUANTITIES):
                members[position, :, horizon - 1, quantity_index] = local_members[
                    quantity
                ]
                truth[position, horizon - 1, quantity_index] = local_truth[quantity]
    if not all(np.all(np.isfinite(value)) for value in (members, truth, current)):
        raise ValueError("hierarchical validation local transport is non-finite")
    return members, truth, current, maximum_closure


def _spread_skill(members: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    """Apply the frozen finite-M convention and serialize undefined ratios safely."""

    summary = corrected_spread_skill_summary(
        members.detach().cpu().numpy(),
        truth.detach().cpu().numpy(),
        member_axis=1,
    )
    ratio = float(summary["spread_skill_ratio"])
    summary["spread_skill_ratio"] = ratio if np.isfinite(ratio) else None
    return summary


def _pooled_covariance_match(
    members: torch.Tensor, truth: torch.Tensor
) -> dict[str, Any]:
    """Compare pooled ensemble covariance with realized ensemble-mean errors."""

    if members.ndim < 4 or truth.shape != members.shape[:1] + members.shape[2:]:
        raise ValueError("hierarchical covariance inputs differ")
    member_count = int(members.shape[1])
    feature_count = int(np.prod(members.shape[3:]))
    cases = int(members.shape[0] * members.shape[2])
    ensemble_mean = members.mean(dim=1)
    anomalies = (members - ensemble_mean[:, None]).permute(
        0, 2, 1, *range(3, members.ndim)
    ).reshape(cases, member_count, feature_count).double()
    error = (ensemble_mean - truth).reshape(cases, feature_count).double()
    ensemble_covariance = torch.einsum("cmd,cme->de", anomalies, anomalies)
    ensemble_covariance /= float(cases * (member_count - 1))
    error_covariance = error.T @ error / float(cases)
    error_norm = float(torch.linalg.matrix_norm(error_covariance))
    error_trace = float(torch.trace(error_covariance))
    ensemble_trace = float(torch.trace(ensemble_covariance))
    relative = (
        float(torch.linalg.matrix_norm(ensemble_covariance - error_covariance))
        / error_norm
        if error_norm > 0.0
        else None
    )
    return {
        "case_count": cases,
        "ensemble_size": member_count,
        "feature_count": feature_count,
        "ensemble_covariance_trace": ensemble_trace,
        "realized_error_outer_product_trace": error_trace,
        "trace_ratio": ensemble_trace / error_trace if error_trace > 0.0 else None,
        "relative_frobenius_error": relative,
        "pooled_over_starts_and_four_future_frames": True,
    }


def score_hierarchical_validation_arrays(
    *,
    local_members: np.ndarray,
    local_truth: np.ndarray,
    current_truth: np.ndarray,
    spatial_bank: IndexedPairBank,
    temporal_bank: IndexedPairBank,
) -> dict[str, Any]:
    """Score local, regional, Fourier, and global validation behavior."""

    members = torch.as_tensor(local_members, dtype=torch.float32)
    truth = torch.as_tensor(local_truth, dtype=torch.float32)
    current = torch.as_tensor(current_truth, dtype=torch.float32)
    if members.shape != (36, 32, 4, 4, 16, 81):
        raise ValueError("hierarchical validation member shape differs")
    if truth.shape != (36, 4, 4, 16, 81) or current.shape != (36, 1, 4, 16, 81):
        raise ValueError("hierarchical validation truth shape differs")
    trajectory_truth = torch.cat((current, truth), dim=1)
    trajectory_members = torch.cat(
        (current[:, None].expand(36, 32, 1, 4, 16, 81), members), dim=2
    )
    hierarchy = score_hierarchical_transport(
        local_members=members,
        local_future_truth=truth,
        local_trajectory_members=trajectory_members,
        local_trajectory_truth=trajectory_truth,
        spatial_bank=spatial_bank,
        temporal_bank=temporal_bank,
    )
    regions_m = regional_transport_sums(members)
    regions_y = regional_transport_sums(truth)
    low_m = transport_fourier_features(members, PGL_HIERARCHICAL_LOW_K)
    low_y = transport_fourier_features(truth, PGL_HIERARCHICAL_LOW_K)
    band_m = transport_fourier_features(members, PGL_HIERARCHICAL_TRANSPORT_K)
    band_y = transport_fourier_features(truth, PGL_HIERARCHICAL_TRANSPORT_K)
    global_m = global_transport_sum(members)
    global_y = global_transport_sum(truth)
    quantities: dict[str, Any] = {}
    for index, quantity in enumerate(TRANSPORT_QUANTITIES):
        spatial = fair_variogram_score(
            members[:, :, :, index : index + 1],
            truth[:, :, index : index + 1],
            spatial_bank,
        )
        temporal = fair_variogram_score(
            trajectory_members[:, :, :, index : index + 1],
            trajectory_truth[:, :, index : index + 1],
            temporal_bank,
        )
        quantities[quantity] = {
            "fair_scores": {
                "local_spatial_variogram": float(hierarchy.local_spatial[index]),
                "local_temporal_variogram": float(hierarchy.local_temporal[index]),
                "regional_energy": float(hierarchy.regional[index]),
                "fourier_low_energy": float(hierarchy.fourier_low[index]),
                "fourier_n20_35_energy": float(
                    hierarchy.fourier_transport_band[index]
                ),
                "global_crps": float(hierarchy.global_crps[index]),
            },
            "ordinary_scores": {
                component: float(hierarchy.ordinary[f"{component}/{quantity}"])
                for component in (
                    "local_spatial",
                    "local_temporal",
                    "regional",
                    "fourier_low",
                    "fourier_transport_band",
                    "global_crps",
                )
            },
            "spatial_variogram_by_distance_bin": [
                float(value) for value in spatial.fair_by_group
            ],
            "temporal_variogram_by_lag": [
                float(value) for value in temporal.fair_by_group
            ],
            "spread_skill": {
                "regional": _spread_skill(regions_m[:, :, :, index], regions_y[:, :, index]),
                "fourier_low_n5_15": _spread_skill(
                    low_m[:, :, :, index], low_y[:, :, index]
                ),
                "fourier_n20_35": _spread_skill(
                    band_m[:, :, :, index], band_y[:, :, index]
                ),
                "global_n0": _spread_skill(global_m[:, :, :, index], global_y[:, :, index]),
            },
            "covariance_match": {
                "regional_12_sector": _pooled_covariance_match(
                    regions_m[:, :, :, index], regions_y[:, :, index]
                ),
                "fourier_low_n5_15": _pooled_covariance_match(
                    low_m[:, :, :, index], low_y[:, :, index]
                ),
                "fourier_n20_35": _pooled_covariance_match(
                    band_m[:, :, :, index], band_y[:, :, index]
                ),
            },
        }
    return {
        "schema_version": 1,
        "scope": "old_85604_pgl_hierarchical_validation_scores",
        "development_run": "85604",
        "current_frames": list(PGL_EVALUATION_STARTS),
        "ensemble_members": 32,
        "future_frames": 4,
        "physical_mode_mapping": "n=5k",
        "low_modes_n": [5, 10, 15],
        "transport_band_n": [20, 25, 30, 35],
        "spatial_distance_bin_upper_edges_m": list(spatial_bank.group_values),
        "temporal_lags_microseconds": list(temporal_bank.group_values),
        "quantities": quantities,
        "transport_computed_memberwise": True,
        "held_out_85606_read": False,
        "new_nersc_data_read": False,
    }
