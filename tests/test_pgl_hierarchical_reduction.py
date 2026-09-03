from __future__ import annotations

import csv

from paper0.tools.reduce_pgl_hierarchical_screen import (
    write_decision_readout,
    write_hierarchy_tables,
)


QUANTITIES = (
    "particle",
    "electron_internal_energy",
    "ion_internal_energy",
    "total_internal_energy",
)


def _quantity() -> dict:
    spread = {
        name: {"spread_skill_ratio": 0.5}
        for name in (
            "regional",
            "fourier_low_n5_15",
            "fourier_n20_35",
            "global_n0",
        )
    }
    covariance = {
        name: {"relative_frobenius_error": 0.8}
        for name in (
            "regional_12_sector",
            "fourier_low_n5_15",
            "fourier_n20_35",
        )
    }
    return {
        "fair_scores": {
            "local_spatial_variogram": 1.0,
            "local_temporal_variogram": 2.0,
            "regional_energy": 3.0,
            "fourier_low_energy": 4.0,
            "fourier_n20_35_energy": 5.0,
            "global_crps": 6.0,
        },
        "spread_skill": spread,
        "covariance_match": covariance,
        "spatial_variogram_by_distance_bin": [1.0, 2.0],
        "temporal_variogram_by_lag": [1.0, 2.0, 3.0, 4.0],
    }


def _scores() -> dict:
    result = {}
    for arm in ("CONTROL", "TRANSPORT"):
        for update in (107, 214, 428):
            result[(arm, update)] = {
                "hierarchical_transport_evaluation": {
                    "spatial_distance_bin_upper_edges_m": [0.1, 0.2],
                    "temporal_lags_microseconds": [3.1, 6.2, 9.3, 12.4],
                    "quantities": {name: _quantity() for name in QUANTITIES},
                }
            }
    return result


def test_reduction_writes_tidy_hierarchy_tables(tmp_path) -> None:
    hierarchy, curves = write_hierarchy_tables(_scores(), tmp_path)
    with hierarchy.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with curves.open(newline="", encoding="utf-8") as handle:
        curve_rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 3 * 4
    assert len(curve_rows) == 2 * 3 * 4 * (2 + 4)
    assert {row["arm"] for row in rows} == {"CONTROL", "TRANSPORT"}


def test_generated_readout_discloses_transport_supervision(tmp_path) -> None:
    metric = {
        "integrated_spread_skill": 0.4,
        "spatial_covariance_error": 0.95,
        "local_spread_skill_median": 0.8,
        "mean_transport_relative_l2": 0.3,
    }
    decision = {
        "metrics": {
            "CONTROL_update_428": metric,
            "TRANSPORT_update_428": {**metric, "integrated_spread_skill": 0.5},
        },
        "epoch_two_matched_difference": {
            "integrated_spread_skill_gain": 0.1,
            "spatial_covariance_error_reduction": 0.0,
        },
        "next_action": "stop_hierarchical_transport_training",
    }
    text = write_decision_readout(decision, tmp_path).read_text(encoding="utf-8")
    assert "explicitly transport-supervised" in text
    assert "stop_hierarchical_transport_training" in text
    assert "No 85606" in text
