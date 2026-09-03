from __future__ import annotations

import numpy as np

from tcv_diagnostics.pgl_hierarchical_validation import (
    score_hierarchical_validation_arrays,
)
from tcv_diagnostics.pgl_variogram import IndexedPairBank


def _bank(name: str, values: tuple[float, ...]) -> IndexedPairBank:
    count = len(values)
    return IndexedPairBank(
        left=np.arange(count, dtype=np.int64),
        right=np.arange(1, count + 1, dtype=np.int64),
        weight=np.full(count, 1.0 / count, dtype=np.float64),
        group=np.arange(count, dtype=np.int64),
        group_name=name,
        group_values=values,
        metadata={"test": True},
    )


def _temporal_bank() -> IndexedPairBank:
    cells = 16 * 81
    return IndexedPairBank(
        left=np.zeros(4, dtype=np.int64),
        right=np.arange(1, 5, dtype=np.int64) * cells,
        weight=np.full(4, 0.25, dtype=np.float64),
        group=np.arange(4, dtype=np.int64),
        group_name="lag_frames",
        group_values=(1.0, 2.0, 3.0, 4.0),
        metadata={"test": True},
    )


def test_truth_like_validation_reports_all_physical_scales() -> None:
    truth = np.zeros((36, 4, 4, 16, 81), dtype=np.float32)
    current = np.zeros((36, 1, 4, 16, 81), dtype=np.float32)
    members = np.broadcast_to(
        truth[:, None], (36, 32, 4, 4, 16, 81)
    ).copy()
    result = score_hierarchical_validation_arrays(
        local_members=members,
        local_truth=truth,
        current_truth=current,
        spatial_bank=_bank("distance_m", (0.1, 0.2)),
        temporal_bank=_temporal_bank(),
    )
    assert result["physical_mode_mapping"] == "n=5k"
    assert result["low_modes_n"] == [5, 10, 15]
    assert result["transport_band_n"] == [20, 25, 30, 35]
    assert set(result["quantities"]) == {
        "particle",
        "electron_internal_energy",
        "ion_internal_energy",
        "total_internal_energy",
    }
    for quantity in result["quantities"].values():
        assert all(abs(value) <= 1.0e-8 for value in quantity["fair_scores"].values())
        assert len(quantity["spatial_variogram_by_distance_bin"]) == 2
        assert len(quantity["temporal_variogram_by_lag"]) == 4
        assert quantity["spread_skill"]["global_n0"]["spread_skill_ratio"] is None


def test_hierarchy_exposes_common_transport_bias_hidden_from_spatial_variogram() -> None:
    truth = np.zeros((36, 4, 4, 16, 81), dtype=np.float32)
    current = np.zeros((36, 1, 4, 16, 81), dtype=np.float32)
    members = np.ones((36, 32, 4, 4, 16, 81), dtype=np.float32)
    result = score_hierarchical_validation_arrays(
        local_members=members,
        local_truth=truth,
        current_truth=current,
        spatial_bank=_bank("distance_m", (0.1, 0.2)),
        temporal_bank=_temporal_bank(),
    )
    for quantity in result["quantities"].values():
        scores = quantity["fair_scores"]
        assert abs(scores["local_spatial_variogram"]) <= 1.0e-8
        assert scores["local_temporal_variogram"] > 0.0
        assert scores["regional_energy"] > 0.0
        assert scores["global_crps"] > 0.0
        assert quantity["ordinary_scores"]["global_crps"] == scores["global_crps"]
        assert (
            quantity["covariance_match"]["regional_12_sector"]
            ["relative_frobenius_error"]
            == 1.0
        )
