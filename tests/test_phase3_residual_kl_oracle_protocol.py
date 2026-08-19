"""Contract tests for the frozen residual-KL representation oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "paper0/PHASE3_POST_LOCALIZATION_DECISION.md"
PROTOCOL = ROOT / "paper0/protocol/PHASE3_RESIDUAL_KL_ORACLE_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase3_residual_kl_oracle_85604.json"

DECISION_SHA256 = (
    "742ed3bbdafca1949baba67af19840e83f9e8c28fb9efd93cdee53beef7969bb"
)
PROTOCOL_SHA256 = (
    "3e1006e52793e612a0daaf21c67e9da2298fc83bfa542af1be4ba376a6acaff7"
)
MANIFEST_SHA256 = (
    "9255699ed902b314cdc27b9d252d1df2fcff794866299ca6dc8708d9671bf575"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_decision_protocol_and_manifest_are_byte_locked() -> None:
    record = load()
    assert sha256(DECISION) == DECISION_SHA256
    assert sha256(PROTOCOL) == PROTOCOL_SHA256
    assert sha256(MANIFEST) == MANIFEST_SHA256
    assert record["decision_memo"] == {
        "path": "paper0/PHASE3_POST_LOCALIZATION_DECISION.md",
        "sha256": DECISION_SHA256,
    }
    assert record["protocol"] == {
        "path": "paper0/protocol/PHASE3_RESIDUAL_KL_ORACLE_PROTOCOL.md",
        "sha256": PROTOCOL_SHA256,
    }


def test_scope_is_one_read_only_85604_representation_experiment() -> None:
    record = load()
    assert record["protocol_status"] == (
        "frozen_post_B5_preimplementation_residual_KL_representation_oracle"
    )
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert set(record["authorized_scope"]) == {
        "one_training_fitted_truth_projected_85604_validation_residual_KL_"
        "representation_oracle",
        "one_training_rank_selected_condition_independent_static_Gaussian_KL_"
        "ensemble_around_the_frozen_H1_mean",
        "one_successful_Rocky9_CPU_execution_after_fail_closed_nonscientific_"
        "retries",
        "online_WandB_compact_monitoring",
    }
    assert {
        "model_checkpoint_loading",
        "model_inference",
        "optimizer_or_trainable_parameter_creation",
        "neural_network_training_or_finetuning",
        "validation_selected_rank",
        "guard_frame_reads",
        "85606_access",
        "O3_fixed_block_forecast",
        "O4_autonomous_rollout",
        "O5_periodic_oracle_reset",
        "assimilation",
        "diagnostic_ranking",
        "steering_or_control",
    } <= set(record["forbidden_scope"])


def test_all_completed_evidence_is_hash_locked() -> None:
    locks = load()["evidence_locks"]
    assert locks["H1_training_forecast"]["sha256"] == (
        "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
    )
    assert locks["H1_training_forecast"]["shape"] == [430, 5, 64, 32, 88]
    assert locks["H1_validation_forecast"]["sha256"] == (
        "a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed"
    )
    assert locks["H1_validation_forecast"]["shape"] == [126, 5, 64, 32, 88]
    assert locks["training_residual_audit"]["sha256"] == (
        "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
    )
    assert locks["training_residual_sufficient_statistics"]["sha256"] == (
        "50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b"
    )
    assert locks["model_dataset"]["manifest_sha256"] == (
        "27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18"
    )
    assert locks["model_dataset"]["normalization_sha256"] == (
        "f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7"
    )
    assert locks["native_truth_result"]["sha256"] == (
        "cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae"
    )
    assert locks["geometry_manifest"]["sha256"] == (
        "76426ca83f711aaf9dec79c6df4c4503c2c8eece847b78027633e93e4b2cd460"
    )
    assert locks["geometry"]["sha256"] == (
        "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8"
    )
    assert locks["B5_covariance_localization_result"] == {
        "job_id": "6901914",
        "tracked_path": (
            "paper0/results/phase3_b5_covariance_localization_6901914.json"
        ),
        "sha256": (
            "331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2"
        ),
    }


def test_chronology_axes_and_toroidal_mapping_are_frozen() -> None:
    data = load()["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["volume_shape"] == [5, 64, 32, 88]
    assert data["training_targets"] == [2, 432]
    assert data["training_target_count"] == 430
    assert data["guard_targets"] == [432, 496]
    assert data["guard_target_count"] == 64
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
    assert data["cadence_microseconds"] == 3.131905426352636
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    assert data["absolute_time_input_used"] is False
    assert data["guard_frames_read_allowed"] is False
    assert data["temporal_windows_are_independent_shots"] is False


def test_residual_gauge_bias_and_basis_are_not_conflated() -> None:
    record = load()
    residual = record["residual_definition"]
    assert residual["sign"] == "truth_minus_H1_mean"
    assert residual["space"] == "decoded_training_standardized_field_space"
    assert residual["phi_gauge_policy"].startswith(
        "subtract_full_spatial_mean_separately"
    )
    assert residual["bias_shape"] == [5, 64, 32]
    assert residual["covariance_empirical_mean_toroidal_average"] == (
        "zero_by_construction"
    )
    assert residual["covariance_empirical_mean_added_to_forecast_mean"] is False
    assert residual["covariance_centered_training_matrix"] == (
        "training_fluctuation_minus_covariance_empirical_mean"
    )
    assert residual["transport_evaluation_space"] == (
        "original_physical_fields_with_authoritative_geometry"
    )

    basis = record["KL_basis"]
    assert basis["method"] == "method_of_snapshots"
    assert basis["fit_region"] == "training_targets_only"
    assert basis["matrix_shape"] == [430, 901120]
    assert basis["compute_dtype"] == "float64"
    assert basis["positive_eigenvalue_relative_threshold"] == 1e-10
    assert basis["maximum_centered_rank"] == 429
    assert basis["rank_ladder"] == [
        0,
        8,
        16,
        32,
        44,
        64,
        128,
        256,
        "full_positive_training_rank",
    ]
    assert basis["rank_44_relation_to_historical_z44"] == (
        "scale_comparison_only_no_shared_representation_or_meaning"
    )


def test_static_rank_is_selected_without_validation() -> None:
    selector = load()["training_only_static_rank_selection"]
    assert selector == {
        "candidates": [8, 16, 32, 44, 64, 128],
        "cumulative_training_variance_target": 0.9,
        "selection": "smallest_available_candidate_reaching_target",
        "fallback_rank": 128,
        "fallback_label": "training_variance_cap_bound",
        "validation_metrics_may_change_rank": False,
    }


def test_tier_A_is_explicitly_an_oracle_with_a_conjunctive_gate() -> None:
    tier = load()["tier_A_projection_oracle"]
    assert tier["uses_current_validation_truth_coefficients"] is True
    assert tier["is_forecast"] is False
    assert tier["is_ensemble"] is False
    assert tier["must_be_labeled_oracle"] is True
    rule = tier["pass_rule"]
    assert rule["minimum_total_validation_variance_captured"] == 0.8
    assert rule["minimum_each_field_validation_variance_captured"] == 0.6
    assert rule["systematic_L3_identities_required"] == 9
    assert rule["systematic_L3_identity_total"] == 11
    assert rule["required_blocks_per_identity"] == 5
    assert rule["material_band_power_ratio_range"] == [0.8, 1.2]
    assert rule["material_field_bands_required"] == 12
    assert rule["strict_face_relative_L2_maximum"] == 0.4
    assert rule["exact_separatrix_relative_L2_maximum"] == 0.3
    assert len(load()["systematic_L3_identities"]) == 11


def test_tier_B_is_one_training_selected_static_covariance_baseline() -> None:
    tier = load()["tier_B_static_Gaussian_KL"]
    assert tier["rank_source"] == "training_only_static_rank_selection"
    assert tier["ensemble_size"] == 32
    assert tier["target_count"] == 126
    assert tier["master_seed"] == 2026081901
    assert tier["generator"] == "numpy_random_PCG64"
    assert tier["seed_bank_shape"] == [126, 32]
    assert tier["condition_independent"] is True
    assert tier["compressed_forecast_must_close_before_validation_truth_read"] is True
    assert tier["nonlinear_metrics_memberwise_before_ensemble_reduction"] is True
    rule = tier["usefulness_rule"]
    assert rule["transport_quantities_required"] == 3
    assert rule["local_corrected_spread_skill_range"] == [0.8, 1.25]
    assert rule["integrated_corrected_spread_skill_range"] == [0.67, 1.5]
    assert rule["minimum_ensemble_to_error_coherence_multiplier_ratio"] == 0.67
    assert rule["maximum_scalar_counterfactual_local_spread_skill"] == 1.5
    assert rule["paper0_forecast_acceptance_gate"] is False


def test_outcomes_and_post_experiment_boundary_are_closed() -> None:
    record = load()
    assert set(record["outcome_classification"]) == {
        "K1_compact_representation_static_covariance_useful",
        "K2_compact_representation_conditional_coefficients_required",
        "K3_only_moderate_or_high_rank_adequate",
        "K4_training_residual_span_does_not_transfer",
        "execution_failed_without_scientific_outcome",
        "inconsistent_diagnostic_requires_review",
    }
    post = record["post_experiment"]
    assert post["one_interpretation_memo_authorized"] is True
    assert post["conditional_low_rank_model_proposal_authorized_only_for_K1_or_K2"]
    assert post["model_implementation_or_training_automatically_authorized"] is False
    for key in (
        "O3_authorized",
        "O4_authorized",
        "O5_authorized",
        "assimilation_authorized",
        "diagnostic_ranking_authorized",
        "steering_authorized",
        "held_out_85606_access_authorized",
    ):
        assert post[key] is False


def test_execution_is_Rusty_CPU_fail_closed_and_compact_WandB_only() -> None:
    execution = load()["execution"]
    assert execution["cluster"] == "Rusty"
    assert execution["os"] == "Rocky_Linux_9"
    assert execution["submission"] == "sbatch"
    assert execution["accelerator"] == "none_CPU_only"
    assert execution["maximum_cpus"] == 32
    assert execution["maximum_memory_GB"] == 256
    assert execution["maximum_walltime_hours"] == 6
    assert execution["wandb_online_required"] is True
    assert execution["wandb_entity"] == (
        "sdelaurentiis123-columbia-university"
    )
    assert execution["wandb_project"] == "tcv-diagnostics-paper0"
    assert {
        "raw_fields",
        "basis_arrays",
        "forecast_members",
        "raw_accumulators",
        "figures",
        "tables",
        "checkpoints",
    } == set(execution["wandb_upload_forbidden"])


def test_human_documents_state_the_scientific_boundaries() -> None:
    decision = " ".join(DECISION.read_text(encoding="utf-8").split())
    protocol = " ".join(PROTOCOL.read_text(encoding="utf-8").split())
    for required in (
        "Do not train another stochastic architecture yet.",
        "truth-projected validation oracle",
        "condition-independent Gaussian KL ensemble",
        "not a return to the old DCAE question",
        "our 430 adjacent training targets are not independent stochastic realizations",
        "K4: training residual span does not transfer",
    ):
        assert required in decision
    for required in (
        "They must never be conflated.",
        "No checkpoint is loaded and no model inference is performed.",
        "Only (R) is used to estimate covariance.",
        "never added to a forecast mean",
        "the rows of (R) sum to zero and the maximum rank is 429",
        "Ranks exceeding the numerical positive rank are reported as unavailable",
        "No validation, transport, spectrum, cross-field, or B5 metric may change this rank.",
        "This uses current target truth to obtain projection coefficients.",
        "Every nonlinear quantity is computed separately for each member",
        "Failure stops without an outcome label.",
        "All outcomes keep O3/O4/O5, additional model seeds, assimilation, diagnostic ranking, steering, and 85606 closed.",
    ):
        assert required in protocol
