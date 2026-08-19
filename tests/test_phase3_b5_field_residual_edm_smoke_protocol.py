"""Contract tests for the frozen B5 joint field-residual EDM smoke."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "paper0/manifests/phase3_b5_field_residual_edm_smoke_85604.json"
)
PROTOCOL = (
    ROOT / "paper0/protocol/PHASE3_B5_FIELD_RESIDUAL_EDM_SMOKE_PROTOCOL.md"
)


def load() -> dict:
    return json.loads(MANIFEST.read_text())


def test_B5_EDM_smoke_keeps_validation_downstream_and_85606_closed() -> None:
    record = load()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert set(record["authorized_scope"]) == {
        "B5_joint_field_residual_EDM_implementation",
        "B5_joint_field_residual_EDM_CPU_tests",
        "one_Rocky9_H100_bounded_full_field_mechanical_smoke",
    }
    forbidden = set(record["forbidden_scope"])
    assert {
        "B5_full_training",
        "B5_validation_read_or_scoring",
        "scientific_checkpoint_selection",
        "O3",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    } <= forbidden
    post = record["post_smoke"]
    assert post["B5_full_training_authorized"] is False
    assert post["validation_read_allowed"] is False
    assert post["held_out_85606_access_allowed"] is False


def test_B5_EDM_smoke_pins_exact_audit_forecast_and_parent() -> None:
    locks = load()["evidence_locks"]
    assert locks["residual_audit"]["job_id"] == "6901393"
    assert locks["residual_audit"]["sha256"] == (
        "d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67"
    )
    forecast = locks["H1_training_forecast"]
    assert forecast["sha256"] == (
        "d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18"
    )
    assert forecast["canonical_shape"] == [430, 5, 64, 32, 88]
    assert forecast["forecast_closed_and_hashed_before_truth_read"] is True
    parent = locks["H1_checkpoint"]
    assert parent["sha256"] == (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
    )
    assert parent["trainable"] is False
    assert parent["reselection_allowed"] is False


def test_B5_EDM_smoke_uses_joint_physical_state_condition_without_time() -> None:
    data = load()["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["smoke_target_frames"] == [2, 10]
    assert data["smoke_target_count"] == 8
    assert data["context_offset_frames"] == -1
    assert data["volume_shape"] == [5, 64, 32, 88]
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    assert data["dynamic_condition"] == [
        "x_t_minus_1_five_fields",
        "frozen_H1_mean_five_fields",
    ]
    for key in (
        "guard_frames_read_allowed",
        "validation_frames_read_allowed",
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "diagnostic_input_allowed",
        "region_mask_input_allowed",
        "future_truth_condition_allowed",
    ):
        assert data[key] is False
    position = data["internal_static_position_channels"]
    assert position == {
        "count": 2,
        "axes": ["x", "y"],
        "range": [-1.0, 1.0],
        "absolute_z_coordinate_allowed": False,
    }


def test_B5_EDM_smoke_residual_scaling_is_exact_and_not_centered() -> None:
    normalization = load()["residual_normalization"]
    assert normalization["operation"] == "divide_without_centering"
    assert normalization["field_order"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert normalization["scale"] == [
        0.05503048051260375,
        0.04825854004472835,
        0.06096460194410047,
        0.04632595196855943,
        0.10251610501339582,
    ]
    assert normalization["nonzero_mean_preserved"] is True
    assert normalization["pointwise_or_region_scaling_allowed"] is False


def test_B5_EDM_smoke_architecture_is_joint_full_field_and_periodic() -> None:
    model = load()["model"]
    assert model["name"] == "B5-H1-JOINT-FIELD-EDM-UNET3D-MINI"
    assert model["joint_output_fields"] == model["noisy_input_channels"] == 5
    assert model["dynamic_condition_channels"] == 10
    assert model["internal_position_channels"] == 2
    assert model["base_channels"] == 32
    assert model["channel_multipliers"] == [1, 2, 4, 4]
    assert model["resolution_path"] == [
        [64, 32, 88],
        [32, 16, 44],
        [16, 8, 22],
        [8, 4, 11],
    ]
    assert model["padding_by_axis"] == ["zeros", "zeros", "circular"]
    assert model["upsampling"] == (
        "bilinear_xy_plus_align_corners_false_periodic_linear_z_to_exact_"
        "skip_shape_then_mixed_boundary_conv3d"
    )
    assert model["noise_embedding_features"] == 256
    assert model["full_field_required"] is True
    assert model["patch_fallback_allowed_in_this_protocol"] is False
    assert model["DCAE_or_latent_representation_allowed"] is False
    assert model["deterministic_parent_trainable"] is False


def test_B5_EDM_smoke_objective_is_standardized_data_only_EDM() -> None:
    edm = load()["edm"]
    assert edm["sigma_data"] == 1.0
    assert edm["preconditioning"] == {
        "c_in": "1/sqrt(sigma^2+1)",
        "c_skip": "1/(sigma^2+1)",
        "c_out": "sigma/sqrt(sigma^2+1)",
        "c_noise": "log(sigma)/4",
    }
    assert edm["training_sigma"] == {
        "distribution": "log_normal",
        "P_mean": -1.2,
        "P_std": 1.2,
    }
    assert edm["loss"] == "((sigma^2+1)/sigma^2)*mean((D_theta-z)^2)"
    assert edm["equal_normalized_channel_weight"] is True
    assert edm["physics_derived_training_loss_allowed"] is False


def test_B5_EDM_smoke_budget_and_sampler_are_bounded_and_exact() -> None:
    record = load()
    optimization = record["optimization"]
    assert optimization["seed"] == 1701
    assert optimization["optimizer_steps"] == 64
    assert optimization["microbatch_targets"] == 1
    assert optimization["gradient_accumulation_targets"] == 1
    assert math.isclose(optimization["learning_rate"], 1e-4)
    assert optimization["betas"] == [0.9, 0.99]
    assert optimization["training_order_seed"] == 67001
    assert optimization["training_noise_seed"] == 67002
    assert optimization["sampler_seed"] == 67003
    assert optimization["fixed_probe_seed"] == 67004
    assert optimization["fixed_probe_targets"] == [2, 6]
    assert optimization["scientific_checkpoint_selection"] is False

    sampler = record["sampler_probe"]
    assert sampler["steps"] == 18
    assert sampler["sigma_max"] == 80.0
    assert sampler["sigma_min"] == 0.002
    assert sampler["rho"] == 7.0
    assert sampler["stochastic_churn"] == 0.0
    assert sampler["ensemble_members"] == 2
    assert sampler["fresh_noise_per_future_autoregressive_step_required"] is True
    assert sampler["trajectory_constant_noise_primary_allowed"] is False
    assert sampler["expected_shape"] == [1, 2, 1, 5, 64, 32, 88]
    assert sampler["scientific_calibration_result"] is False


def test_B5_EDM_smoke_gates_fail_closed_without_scientific_authority() -> None:
    gates = load()["mechanical_gates"]
    assert gates["optimizer_steps_exact"] == 64
    assert gates["peak_allocated_cuda_memory_GiB_strictly_below"] == 75.0
    assert gates["all_values_finite"] is True
    assert gates["final_fixed_probe_loss_strictly_below_initial"] is True
    assert gates["checkpoint_reload_denoiser_probe_bitwise_exact"] is True
    assert gates["toroidal_circular_shift_equivariance_FP32_test"] is True
    assert gates["toroidal_equivariance_shift_cells"] == 8
    assert gates["toroidal_equivariance_rtol"] == 2e-5
    assert gates["toroidal_equivariance_atol"] == 2e-5
    assert gates["sampler_member_diversity_nonzero"] is True
    assert gates["held_out_85606_untouched"] is True
    assert load()["sampler_probe"]["scientific_calibration_result"] is False


def test_B5_EDM_smoke_protocol_states_data_limit_and_boundaries() -> None:
    text = PROTOCOL.read_text()
    compact = " ".join(text.split())
    for required in (
        "at least 50,000 samples",
        "not independent simulations",
        "r_t=x_t-\\mu_t",
        "No mean or axisymmetric bias is subtracted",
        "complete `64x32x88` volume",
        "c_{\\mathrm{noise}}=\\tfrac14\\log\\sigma",
        "a new initial noise volume at every rollout step",
        "does not authorize full training",
    ):
        assert required in compact
