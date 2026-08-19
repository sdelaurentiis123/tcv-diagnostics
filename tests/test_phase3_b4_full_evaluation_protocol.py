"""Regression locks for the prospective B4 full-training/evaluation protocol."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "paper0/protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md"
)
MANIFEST = ROOT / "paper0/manifests/phase3_b4_full_evaluation_85604.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def npy_digest(values: np.ndarray) -> str:
    handle = io.BytesIO()
    np.save(handle, values, allow_pickle=False)
    return hashlib.sha256(handle.getvalue()).hexdigest()


def test_protocol_and_manifest_are_byte_locked() -> None:
    assert digest(PROTOCOL) == (
        "ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1"
    )
    assert digest(MANIFEST) == (
        "e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229"
    )
    assert load()["protocol"] == {
        "path": "paper0/protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md",
        "sha256": digest(PROTOCOL),
    }


def test_passing_smoke_is_exact_evidence_not_scientific_evidence() -> None:
    record = load()
    smoke = record["evidence_locks"]["B4_smoke"]
    smoke_path = ROOT / smoke["path"]
    assert smoke["job_id"] == "6899469"
    assert smoke["status"] == "passed_bounded_non_scientific_implementation_gate"
    assert smoke["scientific_result"] is False
    assert smoke["sha256"] == digest(smoke_path) == (
        "fd2b5465f612eb8da4943f6284e317145eff64b25346895137981ce3e3993eef"
    )
    assert record["implementation_protocol"]["sha256"] == digest(
        ROOT / record["implementation_protocol"]["path"]
    )
    assert record["implementation_manifest"]["sha256"] == digest(
        ROOT / record["implementation_manifest"]["path"]
    )


def test_scope_authorizes_one_seed_85604_and_keeps_later_phases_closed() -> None:
    record = load()
    assert record["development_run"] == "85604"
    assert record["sequestered_run"] == "85606"
    assert record["held_out_85606_access_allowed"] is False
    assert record["full_training_authorized"] is True
    assert record["scientific_one_step_evaluation_authorized"] is True
    assert "B4_PDE_Refiner_H1_seed1701_full_training_85604" in record[
        "authorized_scope"
    ]
    forbidden = set(record["forbidden_scope"])
    assert {
        "85606_access",
        "B4_seed1702_or_seed1703_training",
        "B4_architecture_schedule_noise_or_loss_ablation",
        "O3_or_longer_rollout_execution",
        "assimilation",
        "diagnostic_ranking",
    } <= forbidden
    decision = record["decision_rule"]
    assert decision["additional_seed_training_authorized"] is False
    assert decision["O3_execution_authorized"] is False
    assert decision["assimilation_authorized"] is False
    assert decision["diagnostic_ranking_authorized"] is False
    assert decision["85606_authorized_by_any_B4_85604_result"] is False


def test_data_parent_codec_and_no_time_contract_are_unchanged() -> None:
    record = load()
    data = record["data"]
    assert data["fields"] == ["Ne", "Pe", "Pi", "phi", "Vi"]
    assert data["input_channels"] == "physically_valid_complete_C5P_state"
    assert data["context_frames"] == data["future_frames"] == 1
    assert data["training_targets"] == [2, 432]
    assert data["guard_frames"] == [432, 496]
    assert data["validation_targets"] == [498, 624]
    assert data["zperiod"] == 5
    assert data["mode_mapping"] == "n=5k"
    for key in (
        "absolute_time_input_allowed",
        "normalized_frame_index_input_allowed",
        "shot_label_input_allowed",
        "diagnostic_input_allowed",
        "future_truth_input_allowed_during_generation",
        "guard_frames_read_allowed",
    ):
        assert data[key] is False

    assert record["codec"]["checkpoint_sha256"] == (
        "9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"
    )
    assert record["codec"]["trainable"] is False
    assert record["codec"]["normalization_refit_allowed"] is False
    assert record["deterministic_parent"]["checkpoint_sha256"] == (
        "5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"
    )


def test_full_budget_and_inclusive_cosine_schedule_are_exact() -> None:
    training = load()["training"]
    assert training["epochs"] == 100
    assert training["targets_per_epoch"] == 430
    assert training["gradient_accumulation_targets"] == 16
    assert training["final_partial_accumulation_targets"] == 14
    assert training["optimizer_steps_per_epoch"] == 27
    assert training["total_optimizer_steps"] == 2700
    assert training["optimizer"] == "AdamW"
    assert training["betas"] == [0.9, 0.999]
    assert training["weight_decay"] == 1e-5
    assert training["ema_decay"] == 0.995
    assert training["precision"] == "float32_no_autocast_TF32_disabled"
    assert training["early_stopping"] is False
    assert training["physics_derived_loss_allowed"] is False

    def learning_rate(j: int) -> float:
        return 1e-6 + 0.5 * (1e-4 - 1e-6) * (1.0 + math.cos(math.pi * j / 2699))

    assert learning_rate(0) == 1e-4
    assert learning_rate(2699) == 1e-6
    assert learning_rate(1350) < learning_rate(1349)


def test_full_training_level_matrix_is_reproducible_and_balanced() -> None:
    frozen = load()["training"]["training_level_matrix"]
    values = np.random.Generator(np.random.PCG64(41001)).integers(
        0,
        4,
        size=(100, 430),
        dtype=np.int64,
    )
    assert values.shape == tuple(frozen["shape"])
    assert hashlib.sha256(values.tobytes()).hexdigest() == frozen[
        "raw_C_order_sha256"
    ]
    assert np.bincount(values.reshape(-1), minlength=4).tolist() == frozen[
        "counts_by_level_0_1_2_3"
    ] == [10831, 10680, 10722, 10767]


def test_selection_and_scientific_seed_banks_are_independent_and_exact() -> None:
    selection = load()["checkpoint_selection"]["selection_noise_bank"]
    selection_values = np.random.Generator(np.random.PCG64(41003)).integers(
        0,
        np.iinfo(np.uint64).max,
        size=(126, 2, 3),
        dtype=np.uint64,
    )
    assert hashlib.sha256(selection_values.tobytes()).hexdigest() == selection[
        "raw_C_order_sha256"
    ]
    assert npy_digest(selection_values) == selection["npy_sha256"]

    scientific = load()["scientific_ensemble"]
    scientific_values = np.random.Generator(np.random.PCG64(41032)).integers(
        0,
        np.iinfo(np.uint64).max,
        size=(126, 32, 3),
        dtype=np.uint64,
    )
    assert hashlib.sha256(scientific_values.tobytes()).hexdigest() == scientific[
        "seed_bank_raw_C_order_sha256"
    ]
    assert npy_digest(scientific_values) == scientific["seed_bank_npy_sha256"]
    assert scientific["independent_of_checkpoint_selection_noise"] is True
    assert scientific["seed_bank_npy_sha256"] != selection["npy_sha256"]


def test_checkpoint_selection_cannot_use_physics_or_pick_an_intermediate_stage() -> None:
    selection = load()["checkpoint_selection"]
    assert selection["completed_epoch_candidates"] == list(range(5, 101, 5))
    assert selection["weights"] == "EMA"
    assert selection["target_count"] == 126
    assert selection["ensemble_members"] == 2
    assert selection["metric"] == (
        "equal_channel_decoded_standardized_field_MAE_of_ensemble_mean_at_level3"
    )
    assert selection["selection"] == (
        "earliest_numerically_lowest_after_full_100_epoch_budget"
    )
    assert selection[
        "physics_spectrum_transport_calibration_or_WandB_selection_allowed"
    ] is False
    assert load()["stagewise_H_det_repair"][
        "intermediate_levels_select_checkpoint_or_final_stage"
    ] is False


def test_truth_separated_M32_and_M4_stage_artifacts_are_frozen() -> None:
    ensemble = load()["scientific_ensemble"]
    assert ensemble["final_forecast_shape"] == [126, 32, 1, 5, 64, 32, 88]
    assert ensemble["stage_forecast_shape"] == [126, 4, 4, 5, 64, 32, 88]
    assert ensemble["member_prefix_sensitivity"] == [4, 8, 16, 32]
    assert ensemble["M4_stage3_bitwise_prefix_of_M32_required"] is True
    assert ensemble["level0_bitwise_shared_across_members_required"] is True
    assert ensemble["context_only_generation_required"] is True
    assert ensemble["truth_opened_after_both_forecast_hashes_only"] is True
    assert ensemble["regeneration_or_posthoc_calibration_allowed"] is False
    model = load()["model"]
    assert model["network_calls_per_unamortized_member"] == 4
    assert model["network_calls_for_M_members_with_shared_level0"] == "1+3M"


def test_H_det_and_H_prob_are_distinct_strict_gates() -> None:
    record = load()
    hypotheses = record["hypotheses"]
    assert hypotheses["reported_separately"] is True
    assert hypotheses["H_det_alone_authorizes_assimilation_covariance"] is False
    assert hypotheses[
        "joint_H_det_H_prob_pass_required_for_future_assimilation_covariance"
    ] is True
    repair = record["stagewise_H_det_repair"]
    assert repair["final_field_relative_to_level0_max"] == 1.05
    assert repair["each_physics_error_relative_to_level0_max"] == 1.05
    assert repair["spectral_or_cross_errors_required_strictly_improved"] == 1
    assert repair["transport_strict_improvement_required"] is True
    assert record["gates"]["blocks_required_passing"] == 5
    assert record["gates"]["H_det"]["material_power_ratio_range"] == [0.75, 1.3]
    assert record["gates"]["H_prob"]["primary_field_spread_skill_range"] == [
        0.8,
        1.25,
    ]
    assert record["gates"]["H_prob"][
        "monte_carlo_M16_vs_M32_relative_difference_max"
    ] == 0.1


def test_metric_engine_sources_are_unchanged_and_hash_locked() -> None:
    for relative, expected in load()["metric_engine"]["source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative


def test_execution_is_rusty_only_with_evaluator_smoke_and_online_wandb() -> None:
    execution = load()["execution"]
    assert execution["cluster"] == "Rusty"
    assert execution["operating_system"] == "Rocky9"
    assert execution["accelerator"] == "one_H100_or_H200"
    assert execution["GPU_outside_Rusty_allowed"] is False
    assert execution["full_only_entrypoint_required"] is True
    assert execution["four_target_evaluator_smoke_required_before_full_evaluation"]
    smoke = load()["evaluator_smoke"]
    assert smoke["scientific_result"] is False
    assert smoke["targets"] == [498, 502]
    assert smoke["final_members"] == 32
    assert smoke["stage_prefix_members"] == 4
    assert smoke["must_pass_before_full_evaluation"] is True
    wandb = load()["wandb"]
    assert wandb["mode"] == "online"
    assert wandb["successful_initialization_required_before_training"] is True
    assert wandb["finished_remote_state_required"] is True
