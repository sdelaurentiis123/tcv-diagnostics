"""Contract tests for the frozen post-B5 covariance-localization protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "paper0/manifests/phase3_b5_covariance_localization_85604.json"
)
PROTOCOL = (
    ROOT / "paper0/protocol/PHASE3_B5_COVARIANCE_LOCALIZATION_PROTOCOL.md"
)
LOCALIZATION = (
    ROOT
    / "paper0/results/phase3_b5_residual_edm_one_seed_localization_6901661.json"
)


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scope_is_one_read_only_85604_analysis() -> None:
    record = load()
    assert record["protocol_status"] == (
        "preexecution_amendment_adds_existing_training_H1_forecast_for_"
        "gauge_consistent_drift_reference"
    )
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert set(record["authorized_scope"]) == {
        "one_read_only_B5_M32_covariance_localization",
        "one_training_frozen_AR1_residual_history_diagnostic",
        "one_Rocky9_CPU_execution_on_Rusty",
        "online_WandB_monitoring",
    }
    assert {
        "checkpoint_loading",
        "model_inference",
        "forecast_mutation",
        "posthoc_inflation_or_calibration",
        "model_training",
        "additional_model_seeds",
        "O3_fixed_block_forecast",
        "O4_autonomous_rollout",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    } <= set(record["forbidden_scope"])


def test_completed_B5_artifacts_and_training_comparators_are_hash_locked() -> None:
    locks = load()["evidence_locks"]
    assert locks["B5_forecast"]["job_id"] == "6901587"
    assert locks["B5_forecast"]["sha256"] == (
        "1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b"
    )
    assert locks["B5_forecast"]["bytes"] == 14_535_535_504
    assert locks["B5_forecast"]["shape"] == [126, 32, 1, 5, 64, 32, 88]
    assert locks["B5_forecast"]["mutated_or_regenerated"] is False
    assert locks["B5_scientific_sampler_seed_bank"] == {
        "path": (
            "/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/"
            "phase3_b5_residual_edm_evaluation_full/job_6901587/"
            "b5_joint_field_residual_edm_seed_1701/"
            "scientific_sampler_seeds_M32.npy"
        ),
        "sha256": (
            "013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14"
        ),
        "bytes": 32384,
        "shape": [126, 32],
        "dtype": "uint64",
    }
    assert locks["H1_validation_forecast"]["sha256"] == (
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
    )
    assert locks["B5_score"]["sha256"] == (
        "c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6"
    )
    assert locks["B5_final_gate"]["sha256"] == (
        "a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b"
    )
    assert locks["B5_final_gate"]["disposition"] == (
        "B5_one_step_gate_failed_localize_without_retuning"
    )
    assert locks["training_residual_audit"]["sha256"] == (
        "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
    )
    assert locks["training_residual_sufficient_statistics"]["sha256"] == (
        "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b"
    )
    assert locks["H1_training_forecast"] == {
        "job_id": "6901393",
        "path": (
            "/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/"
            "phase3_b5_h1_residual_audit/job_6901393/audit/"
            "h1_training_forecast.h5"
        ),
        "sha256": (
            "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
        ),
        "bytes": 1_550_112_936,
        "shape": [430, 5, 64, 32, 88],
        "target_frames": [2, 432],
        "model_inference_performed_by_localization": False,
        "purpose": (
            "reconstruct_gauge_fixed_training_H1_residual_for_like_for_like_"
            "covariance_drift_reference"
        ),
    }
    assert sha256(LOCALIZATION) == (
        "ae10349b98394914f6a87dc99bebdc965056a941356f32b0392e261169cbf1f6"
    )


def test_data_axes_periodicity_and_chronology_are_exact() -> None:
    data = load()["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["validation_targets"] == [498, 624]
    assert data["validation_target_count"] == 126
    assert data["chronological_blocks"] == [
        [498, 519],
        [519, 540],
        [540, 561],
        [561, 582],
        [582, 603],
        [603, 624],
    ]
    assert data["history_probe_targets"] == [499, 624]
    assert data["volume_shape"] == [5, 64, 32, 88]
    assert data["native_toroidal_cells"] == 81
    assert data["ensemble_size"] == 32
    assert data["zperiod"] == 5 and data["mode_mapping"] == "n=5k"
    assert data["absolute_time_input_used"] is False
    assert data["guard_frames_read_allowed"] is False
    assert data["temporal_windows_are_independent_shots"] is False


def test_residual_anomaly_and_innovation_are_not_conflated() -> None:
    objects = load()["objects"]
    assert objects == {
        "H1_residual": "truth_minus_H1_mean",
        "B5_generated_residual": "B5_member_minus_H1_mean",
        "B5_anomaly": "B5_member_minus_B5_ensemble_mean",
        "B5_innovation": "truth_minus_B5_ensemble_mean",
        "phi_gauge_policy": (
            "subtract_full_spatial_mean_separately_per_sample_for_field_"
            "covariance_only"
        ),
        "training_covariance_reference": (
            "gauge_fixed_truth_minus_existing_hash_locked_H1_training_forecast"
        ),
        "legacy_training_sufficient_statistics_policy": (
            "retain_as_ungauged_cross_check_and_AR1_coefficient_source_not_as_"
            "phi_gauge_fixed_L3_reference"
        ),
        "axisymmetric_bias_policy": (
            "subtract_mean_over_target_and_z_for_realized_residual_and_"
            "innovation_covariance"
        ),
        "conditional_covariance_identified": False,
    }


def test_dependence_metrics_are_frozen_and_evaluation_only() -> None:
    record = load()
    spatial = record["spatial_covariance"]
    assert spatial["axes"] == ["x", "y", "stored_toroidal_z"]
    assert spatial["maximum_lag"] == "half_axis_extent"
    assert spatial["stable_near_zero_absolute_threshold"] == 0.1
    assert spatial["stable_near_zero_consecutive_lags"] == 3
    cross = record["cross_field_covariance"]
    assert cross["matrix_dimension"] == 5
    assert cross["distance"] == "off_diagonal_RMS"
    assert len(cross["regions"]) == 10
    variogram = record["variogram_score"]
    assert variogram["order_p"] == 1.0
    assert variogram["transport_toroidal_lags"] == [1, 2, 4, 8, 16, 32, 40]
    assert variogram["used_as_training_loss"] is False
    assert variogram["pass_threshold"] is None


def test_toroidal_mapping_and_transport_covariance_are_physical() -> None:
    record = load()
    assert record["toroidal_bands"] == {
        "k": [[0, 0], [1, 3], [4, 5], [6, 7], [8, 44]],
        "n": [[0, 0], [5, 15], [20, 25], [30, 35], [40, 220]],
        "parseval_positive_frequency_weighting": True,
    }
    transport = record["transport_covariance"]
    assert transport["surface"] == "exact_confined_separatrix"
    assert transport["operator"] == (
        "authoritative_native81_shifted_y_radial_ExB_memberwise"
    )
    assert transport["finite_member_variance_factor"] == 33 / 32
    assert transport["member_variance_ddof"] == 1
    assert transport["time_centered_innovation_covariance_ddof"] == 0
    assert transport["retain_local_contributions"] is True
    assert transport["report_diagonal_offdiagonal_and_integrated_variance"] is True
    assert transport["scalar_inflation_counterfactual_only"] is True
    assert transport["inflated_forecast_written"] is False


def test_history_probe_is_training_frozen_causal_and_not_a_rollout() -> None:
    probe = load()["history_probe"]
    assert probe["name"] == "training_frozen_scalar_fieldwise_residual_AR1"
    assert probe["coefficient_source"] == (
        "training_residual_temporal_pattern_lag1_sufficient_statistics"
    )
    assert probe["bias_source"] == "training_axisymmetric_residual_bias"
    assert probe["fit_on_validation"] is False
    assert probe["teacher_forced"] is True
    assert probe["autonomous_rollout"] is False
    assert probe["aggregate_improvement_minimum_fraction_for_history_signal"] == 0.02
    assert probe["chronological_comparisons_required"] == 5
    assert probe["chronological_comparisons_total"] == 6


def test_interpretation_rules_do_not_open_the_downstream_pipeline() -> None:
    record = load()
    interpretation = record["interpretation"]
    assert interpretation["amplitude_spread_skill_outer_range"] == [0.67, 1.5]
    assert interpretation["local_calibrated_spread_skill_range"] == [0.8, 1.25]
    assert interpretation["covariance_multiplier_ratio_maximum"] == 0.67
    assert interpretation["scalar_counterfactual_local_overdispersion_minimum"] == 1.5
    assert interpretation["scientific_acceptance_gate"] is False
    assert interpretation["universal_threshold_claimed"] is False
    post = record["post_analysis"]
    assert post["one_decision_memo_authorized"] is True
    assert post["one_next_85604_experiment_protocol_may_be_proposed"] is True
    assert post["next_experiment_implementation_authorized"] is False
    for key in (
        "additional_seed_training_authorized",
        "O3_launch_authorized",
        "assimilation_authorized",
        "diagnostic_ranking_authorized",
        "held_out_85606_access_authorized",
    ):
        assert post[key] is False


def test_protocol_contains_formal_equations_references_and_claim_boundaries() -> None:
    raw = PROTOCOL.read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for required in (
        "R^{\\mathrm{H1}}_t",
        "A_{t,m}",
        "D_t=x_t-\\bar X_t",
        "d_{\\rho}(a,b)",
        "d_C(C_1,C_2)",
        "\\mathrm{VS}(F,y)",
        "K_{\\mathrm{ens}}",
        "K_{\\mathrm{innov}}",
        "\\widehat R^{\\mathrm{AR1}}",
        "Variogram-Based Proper Scoring Rules",
        "cannot identify irreducible aleatoric noise",
        "Pre-execution gauge amendment",
        "not an autonomous rollout",
        "85606 remain closed",
    ):
        assert required in text
