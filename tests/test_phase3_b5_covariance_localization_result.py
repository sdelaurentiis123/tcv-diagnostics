"""Regression locks for the frozen B5 covariance-localization result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "paper0/results/phase3_b5_covariance_localization_6901914.json"
EXECUTION = (
    ROOT
    / "paper0/results/phase3_b5_covariance_localization_execution_6901914.json"
)
FIGURES = ROOT / "paper0/figures/phase3_b5_covariance_localization"
PLOTTER = ROOT / "paper0/tools/plot_b5_covariance_localization.py"
EXPECTED_RESULT_SHA256 = (
    "331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2"
)
EXPECTED_EXECUTION_SHA256 = (
    "e11127522821e6837125bae195de424810b0e0e2aee1b2b575ed3fe9ee6e7a41"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_covariance_localization_is_byte_locked_and_85604_only() -> None:
    assert _digest(RESULT) == EXPECTED_RESULT_SHA256
    assert _digest(EXECUTION) == EXPECTED_EXECUTION_SHA256
    result = _load()
    assert result["scope"] == "B5_read_only_covariance_localization_85604"
    assert result["development_run"] == "85604"
    assert result["held_out_85606_read"] is False
    assert result["slurm_job_id"] == "6901914"
    assert result["paper0_commit"] == (
        "1bb48ac93b5c6cb7ecb1ed357bf11bee8fdaa198"
    )
    assert result["mode_mapping"] == "n=5k"
    assert result["zperiod"] == 5


def test_covariance_localization_preserves_closed_scientific_boundaries() -> None:
    result = _load()
    for key in (
        "model_training_performed",
        "model_inference_performed",
        "assimilation_performed",
        "diagnostic_ranking_performed",
        "O3_launched",
        "held_out_85606_read",
    ):
        assert result["scientific_boundaries"][key] is False
    labels = result["interpretation_labels"]
    assert labels["training_authorized"] is False
    assert labels["O3_authorized"] is False
    assert labels["assimilation_authorized"] is False
    assert labels["held_out_85606_access_authorized"] is False


def test_covariance_organization_not_scalar_amplitude_is_supported() -> None:
    result = _load()
    labels = result["interpretation_labels"]
    assert labels["L1_predominantly_amplitude_limited"]["supported"] is False
    assert labels["L2_covariance_organization_limited"]["supported"] is True
    assert set(labels["L2_covariance_organization_limited"][
        "supporting_quantities"
    ]) == {
        "particle",
        "electron_internal_energy",
        "ion_internal_energy",
        "total_internal_energy",
    }
    for record in result["transport_covariance"]["quantities"].values():
        covariance = record["covariance_decomposition"]
        assert 0.99 < covariance["local_corrected_spread_skill_ratio"] < 1.01
        assert covariance["integrated_corrected_spread_skill_ratio"] < 0.49
        assert covariance["ensemble_to_error_coherence_multiplier_ratio"] < 0.24
        assert covariance[
            "counterfactual_local_spread_skill_after_same_factor"
        ] > 2.0


def test_blockwise_dependence_and_history_labels_are_exact() -> None:
    labels = _load()["interpretation_labels"]
    L3 = labels["L3_field_dependence_mismatch_beyond_within_run_drift"]
    assert L3[
        "L3_field_dependence_mismatch_beyond_within_run_drift_supported"
    ] is True
    assert L3["systematic_identity_count"] == 11
    assert L3["direction_counts"]["cross_field:private_flux"] == 6
    for field in ("Ne", "Pe", "Pi", "phi", "Vi"):
        assert L3["direction_counts"][f"spatial:y:{field}"] == 6

    L4 = labels["L4_explicit_residual_history_signal"]
    assert L4["supported"] is False
    assert L4["improved_chronological_comparison_count"] == 6
    assert L4["aggregate_H1_RMSE_improvement_fraction"] == (
        0.017198823220332334
    )
    assert labels["L5_unresolved_by_one_realized_trajectory"]["supported"] is True


def test_marginal_and_transport_integrity_recomputations_pass() -> None:
    anchors = _load()["integrity_anchors"]
    assert anchors["B5_marginal_recomputation"]["passed"] is True
    recomputed = anchors["B5_marginal_recomputation"]["recomputed"]
    assert recomputed["equal_channel_ensemble_mean_RMSE"] == 0.07490702593935165
    assert recomputed["equal_channel_corrected_spread_skill_ratio"] == (
        0.801695328225638
    )
    assert anchors["maximum_absolute_B5_anomaly_member_mean"] < 2e-6
    assert anchors["maximum_exact_separatrix_relative_sum_closure"] < 2e-12
    assert anchors["nonlinear_transport_applied_memberwise_before_reduction"] is True


def test_all_six_covariance_localization_figure_pairs_are_labeled() -> None:
    stems = (
        "b5-covariance-spatial-acf",
        "b5-covariance-cross-field",
        "b5-covariance-toroidal-power",
        "b5-covariance-separatrix-transport",
        "b5-covariance-variogram-scores",
        "b5-covariance-history-probe",
    )
    assert {path.name for path in FIGURES.iterdir()} == {
        f"{stem}.{suffix}" for stem in stems for suffix in ("svg", "png")
    }
    labels = "\n".join(
        (FIGURES / f"{stem}.svg").read_text(encoding="utf-8") for stem in stems
    )
    for phrase in (
        "B5 ensemble anomaly",
        "stored k maps to full-torus n=5k",
        "local SSR after",
        "diagnostic only; no pass threshold",
        "not an autonomous rollout",
    ):
        assert phrase in labels


def test_plotter_is_bound_to_the_frozen_result_hash() -> None:
    source = PLOTTER.read_text(encoding="utf-8")
    assert "EXPECTED_SHA256" in source
    assert EXPECTED_RESULT_SHA256 in source
    assert "held_out_85606_read" in source
    assert "write_readout_figures" in source
