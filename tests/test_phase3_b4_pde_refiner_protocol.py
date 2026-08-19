"""Regression locks for the prospective B4 PDE-Refiner smoke protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper0/protocol/PHASE3_B4_PDE_REFINER_PROTOCOL.md"
MANIFEST = ROOT / "paper0/manifests/phase3_b4_pde_refiner_85604.json"


def load() -> dict:
    return json.loads(MANIFEST.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_and_manifest_are_byte_locked() -> None:
    assert digest(PROTOCOL) == (
        "29b71b2209df816c0847ea1b9170283f94a579fab6e9637ee0d33ccd55cd701f"
    )
    assert digest(MANIFEST) == (
        "29848b5ec080cbd1dbcb1932594842d493e18d3b0b87d97429476a2b1526f7c5"
    )
    assert load()["protocol"] == {
        "path": "paper0/protocol/PHASE3_B4_PDE_REFINER_PROTOCOL.md",
        "sha256": digest(PROTOCOL),
    }


def test_only_implementation_and_bounded_smoke_are_authorized() -> None:
    record = load()
    assert record["protocol_status"] == (
        "frozen_after_failed_B3_before_B4_implementation_smoke_or_training"
    )
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["full_training_authorized"] is False
    assert record["prospective_full_training"]["authorized"] is False
    assert record["later_full_evaluation_outline"]["authorized"] is False
    assert record["decision_rules"]["85606_opened_by_any_B4_85604_result"] is False
    assert set(record["authorized_scope"]) == {
        "B4_latent_PDE_Refiner_H1_implementation",
        "B4_latent_PDE_Refiner_H1_CPU_tests",
        "B4_latent_PDE_Refiner_H1_single_seed_bounded_GPU_smoke_85604",
    }
    forbidden = set(record["forbidden_scope"])
    assert {
        "B4_full_training_before_new_authorization",
        "B4_scientific_evaluation_before_frozen_full_evaluation_protocol",
        "O3_or_longer_rollout",
        "assimilation",
        "diagnostic_ranking",
        "85606_access",
    } <= forbidden


def test_data_contract_is_c5p_h1_without_time_or_future_information() -> None:
    data = load()["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["input_channels"] == "physically_valid_complete_C5P_state"
    assert data["context_frames"] == data["future_frames"] == 1
    assert data["training_frames"] == [0, 432]
    assert data["training_targets"] == [2, 432]
    assert data["guard_frames"] == [432, 496]
    assert data["validation_frames"] == [496, 624]
    assert data["validation_targets"] == [498, 624]
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    assert data["volume_shape"] == [5, 64, 32, 88]
    assert data["cadence_microseconds"] == 3.131905426352636
    for key in (
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "diagnostic_input_allowed",
        "future_truth_input_allowed_during_forecast",
        "guard_frames_read_allowed",
    ):
        assert data[key] is False


def test_exact_parent_and_frozen_codec_are_required() -> None:
    record = load()
    codec = record["codec"]
    assert codec["name"] == "C5P-dcae_l10"
    assert codec["checkpoint_sha256"] == (
        "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
    )
    assert codec["latent_channels"] == 32
    assert codec["latent_grid"] == [16, 8, 22]
    assert codec["evaluation_mode_required"] is True
    assert codec["trainable_during_B4"] is False
    assert codec["normalization_refit_allowed"] is False
    assert codec["noise_coordinate"] == "per_channel_standardized_codec_latent"
    assert codec["field_space_noise_claimed"] is False

    parent = record["deterministic_parent"]
    assert parent["arm"] == "C5P-H1"
    assert parent["seed"] == 1701
    assert parent["checkpoint_sha256"] == (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
    )
    assert parent["parameter_count"] == 51_612_800
    assert parent["exact_shape_matched_load_required"] is True
    assert parent["unexpected_parent_keys_allowed"] is False
    assert parent["new_adapter_final_layers_zero_initialized"] is True
    assert parent["level0_preoptimization_bitwise_identity_required"] is True


def test_refinement_schedule_and_precision_are_explicit() -> None:
    record = load()
    schedule = record["noise_schedule"]
    assert schedule["implementation"] == (
        "explicit_denoising_not_library_DDPM_scheduler"
    )
    assert schedule["refinement_steps"] == 3
    assert schedule["minimum_noise_variance"] == 4e-7
    assert schedule["minimum_noise_standard_deviation"] == math.sqrt(4e-7)
    expected = {
        "1": 0.08583742189325572,
        "2": 0.007368062997280775,
        "3": 0.0006324555320336759,
    }
    assert schedule["sigma_by_level"] == expected
    for key, value in expected.items():
        level = int(key)
        assert math.isclose(
            value,
            math.sqrt(4e-7) ** (level / 3),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    assert schedule["schedule_selected_from_B4_validation"] is False
    assert schedule["spectrum_used_to_tune_schedule"] is False

    precision = record["precision"]
    assert {
        precision["training"],
        precision["validation"],
        precision["inference"],
    } == {"float32_no_autocast"}
    assert precision["torch_float32_matmul_precision"] == "highest"
    assert precision["cuda_matmul_allow_tf32"] is False
    assert precision["cudnn_allow_tf32"] is False


def test_training_objective_is_data_only_and_level_conditioning_is_identity_safe() -> None:
    record = load()
    objective = record["objective"]
    assert objective["refinement_level_sampling"] == (
        "uniform_over_0_1_2_3_per_training_target"
    )
    assert objective["level0_prediction"] == "standardized_latent_increment"
    assert objective["level0_target"] == "z_true_minus_z_previous"
    assert objective["level1_to_3_prediction"] == "epsilon"
    assert objective["level1_to_3_target"] == "epsilon"
    assert objective["loss"] == "mean_squared_error"
    assert objective["codec_decode_in_gradient_path"] is False
    assert objective["codec_parameters_trainable"] is False
    assert objective["physics_derived_loss_allowed"] is False
    assert record["model"]["physics_derived_loss_allowed"] is False

    conditioning = record["refinement_conditioning"]
    assert conditioning["levels"] == [0, 1, 2, 3]
    assert conditioning["normalized_level_coordinates"] == [0.0, 1 / 3, 2 / 3, 1.0]
    assert conditioning["sinusoidal_features"] == 256
    assert conditioning["embedded_dimension"] == 256
    assert conditioning["layer_semantics"] == (
        "same_embedding_supplied_to_all_16_blocks"
    )
    assert conditioning["adapter_outputs"] == [
        "scale",
        "shift",
        "residual_scale",
        "skip_scale",
    ]
    assert conditioning["adapter_last_weight_zero"] is True
    assert conditioning["adapter_last_bias_zero"] is True


def test_inference_is_four_calls_with_no_posthoc_covariance_repairs() -> None:
    inference = load()["inference"]
    assert inference["network_calls_per_member"] == 4
    assert inference["decoded_level"] == 3
    assert inference["level0_shared_across_members"] is True
    assert inference["level1_to_3_noise_independent_across_members_and_levels"] is True
    assert inference["member_interaction"] is False
    for key in (
        "recentring_allowed",
        "inflation_allowed",
        "clipping_allowed",
        "rejection_sampling_allowed",
        "posthoc_calibration_allowed",
    ):
        assert inference[key] is False
    interface = load()["forecast_interface"]
    assert interface["call"] == "model.predict(context,horizon,ensemble_size)"
    assert interface["canonical_axes"] == [
        "batch",
        "ensemble_member",
        "future_time",
        "channel",
        "x",
        "y",
        "z",
    ]
    assert interface["horizon_allowed_by_this_protocol"] == 1
    assert interface["target_argument_allowed"] is False


def test_smoke_is_bounded_and_cannot_be_misreported_as_science() -> None:
    smoke = load()["implementation_gate"]["gpu_smoke"]
    assert smoke["scientific_result"] is False
    assert smoke["accelerator"] == "one Rocky9 H100 or H200"
    assert smoke["epochs"] == 2
    assert smoke["training_targets"] == [2, 18]
    assert smoke["training_target_count"] == 16
    assert smoke["validation_targets"] == [498, 502]
    assert smoke["validation_target_count"] == 4
    assert smoke["validation_members"] == 2
    assert smoke["refinement_levels_explicitly_probed"] == [0, 1, 2, 3]
    assert smoke["optimizer_steps"] == 2
    assert smoke["intermediate_stages_saved"] is True
    assert smoke["wandb_online_required"] is True


def test_hypotheses_and_decisions_do_not_conflate_sharpness_with_covariance() -> None:
    record = load()
    hypotheses = record["hypotheses"]
    assert hypotheses["decisions_reported_separately"] is True
    assert hypotheses["H_det_pass_implies_assimilation_allowed"] is False
    assert hypotheses["H_prob_pass_required_for_future_assimilation_covariance"] is True
    decisions = record["decision_rules"]
    assert decisions["smoke_pass"] == (
        "may_write_separate_full_training_and_evaluation_protocol_only"
    )
    assert decisions["H_det_fail"] == "stop_B4_before_replication_or_O3"
    assert decisions["H_det_pass_H_prob_fail"] == (
        "B4_may_be_considered_only_as_refined_deterministic_transition_no_assimilation_covariance"
    )


def test_primary_provenance_and_preimplementation_source_locks_are_frozen() -> None:
    provenance = load()["provenance"]
    assert provenance["primary_paper"]["arxiv"] == "2308.05732v2"
    official = provenance["official_repository"]
    assert official["url"] == "https://github.com/pdearena/pdearena"
    assert official["commit"] == "327424a46020c2afcfd777e8339e4b61b20d0e72"
    assert official["license"] == "MIT"
    assert official["source_files"]["pdearena/models/pderefiner.py"] == (
        "f7f79d53b6bedcb4dc903133fe3e1e7f22509513fdb1ebfb14da2384b0c31131"
    )
    assert provenance["later_primary_caveat"]["arxiv"] == "2506.10711"
    assert provenance["later_primary_caveat"]["used_as_training_method"] is False

    locks = provenance["existing_Paper0_sources_locked_against_initial_B4_modification"]
    assert locks == {
        "src/tcv_diagnostics/models/o2.py": (
            "5425b76d501c7385bb63e53f2a01e8882f29759b64f41d71d8372bb55e628ceb"
        ),
        "src/tcv_diagnostics/models/vit.py": (
            "0d1a6863f399fe43c57b7bc4b8b52f29d6baf48d59b33d8c9529b51f6843d853"
        ),
        "src/tcv_diagnostics/o2_training.py": (
            "0ff54f09bbb62fad11274e4f8561e6f15d95a332926520a427e65e28c24e5111"
        ),
    }
    for relative_path, expected in locks.items():
        assert digest(ROOT / relative_path) == expected


def test_wandb_and_full_budget_are_prospective_not_results() -> None:
    record = load()
    wandb = record["wandb"]
    assert wandb["entity"] == "sdelaurentiis123-columbia-university"
    assert wandb["project"] == "tcv-diagnostics-paper0"
    assert wandb["smoke_and_full_mode"] == "online"
    assert wandb["successful_initialization_required_before_training"] is True
    assert wandb["finished_remote_state_required"] is True

    budget = record["prospective_full_training"]
    assert budget["authorized"] is False
    assert budget["epochs"] == 100
    assert budget["targets_per_epoch"] == 430
    assert budget["gradient_accumulation_targets"] == 16
    assert budget["total_optimizer_steps"] == 2700
    assert budget["optimizer"] == "AdamW"
    assert budget["peak_learning_rate"] == 1e-4
    assert budget["minimum_learning_rate"] == 1e-6
    assert budget["ema_decay"] == 0.995
    assert budget["precision"] == "float32_no_autocast_TF32_disabled"
