"""Regression checks for the prospective B2 full/evaluation protocol."""

from pathlib import Path

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.model_data import load_strict_json


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B2_FULL_EVALUATION_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase3_b2_full_evaluation_85604.json"


def manifest() -> dict:
    return load_strict_json(MANIFEST)


def test_protocol_is_frozen_before_full_training_and_locks_its_hash() -> None:
    record = manifest()
    assert record["protocol_status"] == (
        "frozen_before_B2_full_training_or_scientific_metric_implementation"
    )
    assert record["decision_timing"] == (
        "after_passing_B2_smoke_before_full_training_or_scientific_metric_implementation"
    )
    assert sha256_path(PROTOCOL) == (
        "8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490"
    )
    assert record["protocol"]["sha256"] == sha256_path(PROTOCOL)
    assert record["full_training_authorized"] is True
    assert record["probabilistic_evaluation_authorized"] is True


def test_scope_remains_85604_only_one_step_and_noncausal() -> None:
    record = manifest()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["model"]["context_frames"] == 2
    assert record["model"]["future_frames"] == 1
    assert record["data"]["future_truth_input_allowed"] is False
    assert record["data"]["absolute_time_input_allowed"] is False
    for forbidden in (
        "85606_access",
        "O3_or_longer_rollout",
        "assimilation",
        "diagnostic_ranking",
        "steering_or_control",
        "physics_derived_training_loss",
        "validation_tuning",
    ):
        assert forbidden in record["forbidden_scope"]


def test_full_budget_and_checkpoint_selection_are_unchanged() -> None:
    training = manifest()["training"]
    assert training["seeds"] == [1701, 1702, 1703]
    assert training["epochs"] == 200
    assert training["targets_per_epoch"] == 430
    assert training["validation_targets"] == 126
    assert training["optimizer_steps_per_epoch"] == 27
    assert training["total_optimizer_steps"] == 5400
    assert training["learning_rate"] == 1.0e-4
    assert training["betas"] == [0.9, 0.99]
    assert training["weight_decay"] == 0.0
    assert training["warmup_steps"] == 0
    assert training["physics_derived_loss_allowed"] is False
    assert "complete_trajectory_denoising_loss" in training["checkpoint_selection"]


def test_fair_crps_member_count_and_prefix_rules_are_explicit() -> None:
    record = manifest()
    ensemble = record["ensemble_generation"]
    assert ensemble["members"] == 32
    assert ensemble["member_prefix_sensitivity"] == [4, 8, 16, 32]
    assert ensemble["regeneration_for_member_prefixes_allowed"] is False
    assert ensemble["member_interaction_allowed"] is False
    assert ensemble["canonical_shape"] == [126, 32, 1, 5, 64, 32, 88]
    assert record["calibration"]["fair_crps"]["primary"] is True
    assert record["calibration"]["fair_crps"]["member_assumption"] == (
        "conditionally_independent_exchangeable"
    )
    assert record["gates"]["monte_carlo"] == {
        "absolute_floor": 1e-08,
        "comparison": "M16_vs_M32",
        "relative_difference_max": 0.1,
    }


def test_phi_geometry_and_memberwise_nonlinear_rules_are_locked() -> None:
    record = manifest()
    assert record["normalization_and_gauge"]["phi_primary_marginal_policy"] == (
        "subtract_each_truth_and_member_spatial_mean_per_target"
    )
    assert record["geometry_regions"]["primary_partition"] == [
        "confined_edge",
        "private_flux",
        "scrape_off_layer",
    ]
    assert record["spectral"][
        "ensemble_mean_fields_for_nonlinear_cross_spectrum_allowed"
    ] is False
    assert record["transport"]["ensemble_mean_fields_for_transport_allowed"] is False
    assert record["transport"]["memberwise_first"] is True
    assert record["transport"]["complete_heat_flux_claim_allowed"] is False
    assert record["data"]["zperiod"] == 5
    assert record["data"]["mode_mapping"] == "n=5k"


def test_acceptance_uses_seed_and_temporal_robustness_without_cherry_pick() -> None:
    record = manifest()
    decision = record["architecture_decision_rule"]
    assert decision["seed_count"] == 3
    assert decision["at_least_passing_seeds"] == 2
    assert decision["median_seed_metrics_must_pass"] is True
    assert decision["nonpassing_seed_catastrophic_bounds"][
        "integrity_failure_allowed"
    ] is False
    assert record["gates"]["blocks_total"] == 6
    assert record["gates"]["blocks_required_passing"] == 5
    assert record["bootstrap"]["block_length_frames"] == 21
    assert record["bootstrap"]["replicates"] == 2000
    assert record["bootstrap"]["voxel_count_as_independent_sample_size_allowed"] is False


def test_smoke_and_deterministic_evidence_are_hash_locked() -> None:
    evidence = manifest()["evidence_locks"]
    assert evidence["B2_smoke"]["job_id"] == "6896402"
    assert evidence["B2_smoke"]["sha256"] == (
        "fa2b29665b4b39b60c9ce24c1e8b067ebc6165322d40bb8de169bf9492ae5360"
    )
    deterministic = manifest()["comparators"]["deterministic"]
    assert deterministic["paired_by_seed"] is True
    assert deterministic["reselection_allowed"] is False
    assert deterministic["result_sha256"] == (
        "251820a6f81d97ffdb046eba7a23cd12505c1179e600e7942231de2fd1feeacb"
    )

